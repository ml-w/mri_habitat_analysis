"""
State management for the habitat analysis pipeline.

HabitatState bundles all artefacts required to reproduce inference:
    - Per-sequence normaliser state directories  (``normaliser_state/{seq}/``)
    - PyRadiomics filter config (YAML)
    - Fitted clusterer (joblib file)
    - Training feature table (``features.parquet``)
    - Metadata JSON (best_k, metrics, hyperparameters, versions)

All artefacts are packed into a single ``.zip`` archive for portability.

Archive layout (v2)::

    normaliser_state/
        T1/                      ← mnts state for sequence "T1"
        T2/                      ← mnts state for sequence "T2"
    clusterer.joblib
    pyradiomics_config.yaml
    features.parquet             ← voxel table: case_id, z, y, x, <feature cols>
    metadata.json

Legacy (v1) archives contain a flat ``normaliser_state/`` directory and no
``features.parquet``.  :meth:`load` detects and handles both formats.

Feature table schema
--------------------
Each row is one foreground voxel.  Columns:

* ``case_id``   — string key extracted by id_globber
* ``z``, ``y``, ``x`` — voxel coordinates in the original 3-D volume
* ``{seq}__{filter}`` — one float32 column per (sequence, filter) combination,
  e.g. ``T1__Original``, ``T1__LBP2D``, ``T2__Original``, …
"""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class _JSONEncoder(json.JSONEncoder):
    """Encode numpy scalar types and other non-standard objects for JSON."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

from mnts.mnts_logger import MNTSLogger

logger = MNTSLogger[__name__]

_METADATA_FNAME = "metadata.json"
_CLUSTERER_FNAME = "clusterer.joblib"
_PYRAD_CONFIG_FNAME = "pyradiomics_config.yaml"
_NORM_STATE_DIRNAME = "normaliser_state"
_FEATURES_FNAME = "features.parquet"

# Coordinate columns that are NOT feature columns in features.parquet
_COORD_COLS = {"case_id", "z", "y", "x"}


class HabitatState:
    """Container for all trained artefacts of a habitat analysis run.

    Attributes:
        clusterer_path: Path to the saved clusterer joblib file.
        pyrad_config_path: Path to the pyradiomics YAML used for training.
        norm_state_paths: ``{sequence_name: normaliser_state_dir}`` mapping.
        metadata: Dict of hyperparameters and evaluation results.
        extract_dir: Root of the extracted archive (needed to locate
            ``features.parquet``).
    """

    def __init__(
        self,
        clusterer_path: Union[str, Path],
        pyrad_config_path: Union[str, Path],
        norm_state_paths: Union[Dict[str, Path], str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        extract_dir: Optional[Path] = None,
    ):
        self.clusterer_path = Path(clusterer_path)
        self.pyrad_config_path = Path(pyrad_config_path)
        # Accept a plain path for single-sequence backward compatibility
        if isinstance(norm_state_paths, (str, Path)):
            self.norm_state_paths: Dict[str, Path] = {"image": Path(norm_state_paths)}
        else:
            self.norm_state_paths = {k: Path(v) for k, v in norm_state_paths.items()}
        self.metadata = metadata or {}
        self.extract_dir = Path(extract_dir) if extract_dir else None

    # ------------------------------------------------------------------
    # Backward-compat shim
    # ------------------------------------------------------------------

    @property
    def norm_state_path(self) -> Path:
        """Path to the primary (first) normaliser state directory."""
        return next(iter(self.norm_state_paths.values()))

    # ------------------------------------------------------------------
    # Sequence validation
    # ------------------------------------------------------------------

    @property
    def required_sequences(self) -> List[str]:
        """Ordered sequence names recorded at training time."""
        return list(self.metadata.get("sequences", []))

    def validate_sequences(self, provided: List[str]) -> List[str]:
        """Check *provided* names against those recorded at training.

        Returns:
            List of non-fatal warning strings (empty if all match).

        Raises:
            ValueError: If any required sequence is absent.
        """
        required = self.required_sequences
        if not required:
            return []

        missing = set(required) - set(provided)
        extra = set(provided) - set(required)

        warnings: List[str] = []
        if extra:
            warnings.append(f"Sequences provided but not in training: {sorted(extra)}")
        if missing:
            raise ValueError(
                f"Missing required sequences (use --override to ignore): {sorted(missing)}"
            )
        return warnings

    # ------------------------------------------------------------------
    # Cached training features (parquet)
    # ------------------------------------------------------------------

    @property
    def _features_path(self) -> Optional[Path]:
        if self.extract_dir is None:
            return None
        p = self.extract_dir / _FEATURES_FNAME
        return p if p.exists() else None

    def load_features_df(self):
        """Load the full training feature table as a ``pandas.DataFrame``.

        Columns: ``case_id``, ``z``, ``y``, ``x``, then one column per
        ``{sequence}__{filter}`` combination.

        Returns:
            ``DataFrame`` or ``None`` if the archive has no cached features.
        """
        import pandas as pd

        p = self._features_path
        return pd.read_parquet(p) if p else None

    def get_case_features(self, case_id: str):
        """Return the feature sub-table for one case as a ``DataFrame``.

        Returns ``None`` if the archive has no cached features or the case
        is unknown.
        """
        df = self.load_features_df()
        if df is None:
            return None
        sub = df[df["case_id"] == case_id]
        return sub if len(sub) > 0 else None

    @property
    def feature_columns(self) -> List[str]:
        """Ordered list of feature column names (excludes ``case_id``, ``z``, ``y``, ``x``).

        Returns an empty list if no features are cached.
        """
        p = self._features_path
        if p is None:
            return []
        import pandas as pd
        schema = pd.read_parquet(p, columns=[]).columns.tolist()
        return [c for c in schema if c not in _COORD_COLS]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        archive_path: Union[str, Path],
        features_df=None,
    ) -> None:
        """Write all artefacts to *archive_path*.

        If *archive_path* ends with ``.zip`` the artefacts are bundled into a
        zip archive.  Otherwise they are written directly into *archive_path*
        as a plain directory.

        Args:
            archive_path: Destination path — either a ``.zip`` file or a
                directory path.
            features_df: Optional ``pandas.DataFrame`` (from
                :func:`build_features_df`) to embed as ``features.parquet``.
        """
        archive_path = Path(archive_path)
        use_zip = archive_path.suffix == ".zip"

        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp) / "state"
            staging.mkdir()

            # 1. Clusterer
            shutil.copy2(self.clusterer_path, staging / _CLUSTERER_FNAME)

            # 2. PyRadiomics config
            shutil.copy2(self.pyrad_config_path, staging / _PYRAD_CONFIG_FNAME)

            # 3. Per-sequence normaliser states
            norm_root = staging / _NORM_STATE_DIRNAME
            norm_root.mkdir()
            for seq_name, state_dir in self.norm_state_paths.items():
                dst = norm_root / seq_name
                state_dir = Path(state_dir)
                if state_dir.is_dir():
                    shutil.copytree(state_dir, dst)
                elif state_dir.is_file():
                    dst.mkdir()
                    shutil.copy2(state_dir, dst / state_dir.name)
                else:
                    logger.warning(f"Normaliser state not found for '{seq_name}': {state_dir}")
                    dst.mkdir()

            # 4. Feature table
            if features_df is not None:
                features_df.to_parquet(
                    staging / _FEATURES_FNAME,
                    index=False,
                    compression="snappy",
                )

            # 5. Metadata
            (staging / _METADATA_FNAME).write_text(
                json.dumps(self.metadata, indent=2, cls=_JSONEncoder)
            )

            if use_zip:
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.make_archive(str(archive_path.with_suffix("")), "zip", staging)
            else:
                if archive_path.exists():
                    shutil.rmtree(archive_path)
                shutil.copytree(staging, archive_path)

        logger.info(f"State saved to {archive_path}")

    @classmethod
    def load(
        cls,
        archive_path: Union[str, Path],
        extract_dir: Union[str, Path, None] = None,
    ) -> "HabitatState":
        """Load artefacts from a zip archive.

        Handles both v1 (flat ``normaliser_state/``) and v2 (per-sequence
        subdirs) layouts automatically.
        """
        archive_path = Path(archive_path)
        if not archive_path.exists():
            raise FileNotFoundError(f"State archive not found: {archive_path}")

        if archive_path.is_dir():
            # Plain directory — use directly, no extraction needed
            extract_dir = archive_path
            logger.info(f"Loading state from directory {extract_dir}")
        else:
            if extract_dir is None:
                extract_dir = Path(tempfile.mkdtemp(prefix="habitat_state_"))
            else:
                extract_dir = Path(extract_dir)
                extract_dir.mkdir(parents=True, exist_ok=True)
            shutil.unpack_archive(str(archive_path), str(extract_dir), "zip")
            logger.info(f"State extracted to {extract_dir}")

        metadata_path = extract_dir / _METADATA_FNAME
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

        # Detect v1 vs v2 normaliser layout
        norm_root = extract_dir / _NORM_STATE_DIRNAME
        seq_names = metadata.get("sequences", [])
        norm_state_paths: Dict[str, Path] = {}

        if seq_names:
            for seq in seq_names:
                seq_dir = norm_root / seq
                if seq_dir.exists():
                    norm_state_paths[seq] = seq_dir
                else:
                    logger.warning(f"Normaliser state dir missing: {seq_dir}")
                    norm_state_paths[seq] = norm_root
        else:
            # v1: treat norm_root as the single state
            norm_state_paths["image"] = norm_root

        return cls(
            clusterer_path=extract_dir / _CLUSTERER_FNAME,
            pyrad_config_path=extract_dir / _PYRAD_CONFIG_FNAME,
            norm_state_paths=norm_state_paths,
            metadata=metadata,
            extract_dir=extract_dir,
        )

    # ------------------------------------------------------------------
    # Convenience factory
    # ------------------------------------------------------------------

    @classmethod
    def from_parts(
        cls,
        clusterer_path: Union[str, Path],
        pyrad_config_path: Union[str, Path],
        norm_state_paths: Dict[str, Union[str, Path]],
        best_k: int,
        metrics: Dict,
        extra: Optional[Dict] = None,
    ) -> "HabitatState":
        """Create a :class:`HabitatState` from individual artefact paths.

        Args:
            norm_state_paths: ``{sequence_name: normaliser_state_dir}`` mapping.
        """
        import platform

        try:
            import radiomics
            pyrad_version = radiomics.__version__
        except Exception:
            pyrad_version = "unknown"
        try:
            import sklearn
            sklearn_version = sklearn.__version__
        except Exception:
            sklearn_version = "unknown"

        metadata = {
            "best_k": best_k,
            "metrics": {str(k): v for k, v in metrics.items()},
            "python": platform.python_version(),
            "pyradiomics_version": pyrad_version,
            "sklearn_version": sklearn_version,
            "numpy_version": np.__version__,
        }
        if extra:
            metadata.update(extra)

        return cls(
            clusterer_path=clusterer_path,
            pyrad_config_path=pyrad_config_path,
            norm_state_paths={k: Path(v) for k, v in norm_state_paths.items()},
            metadata=metadata,
        )
