"""
State management for the habitat analysis pipeline.

HabitatState bundles all artefacts required to reproduce inference:
    - Normaliser state directory (trained NyulNormalizer parameters)
    - PyRadiomics filter config (YAML)
    - Fitted clusterer (joblib file)
    - Metadata JSON (best_k, metrics, hyperparameters, versions)

All artefacts are saved together in a single .zip archive for portability.
"""

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from mnts.mnts_logger import MNTSLogger

logger = MNTSLogger[__name__]

_METADATA_FNAME = "metadata.json"
_CLUSTERER_FNAME = "clusterer.joblib"
_PYRAD_CONFIG_FNAME = "pyradiomics_config.yaml"
_NORM_STATE_DIRNAME = "normaliser_state"


class HabitatState:
    """Container for all trained artefacts of a habitat analysis run.

    Attributes:
        clusterer_path: Path to the saved :class:`~clusterer.HabitatClusterer` joblib file.
        pyrad_config_path: Path to the pyradiomics YAML config used for training.
        norm_state_path: Path to the mnts normaliser state directory.
        metadata: Dict of hyperparameters and evaluation results.
    """

    def __init__(
        self,
        clusterer_path: Union[str, Path],
        pyrad_config_path: Union[str, Path],
        norm_state_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.clusterer_path = Path(clusterer_path)
        self.pyrad_config_path = Path(pyrad_config_path)
        self.norm_state_path = Path(norm_state_path)
        self.metadata = metadata or {}

    # ------------------------------------------------------------------
    # Sequence validation
    # ------------------------------------------------------------------

    @property
    def required_sequences(self) -> List[str]:
        """Ordered list of sequence names recorded at training time.

        Returns an empty list for legacy state archives that predate
        multi-sequence support.
        """
        return list(self.metadata.get("sequences", []))

    def validate_sequences(self, provided: List[str]) -> List[str]:
        """Check *provided* sequence names against those recorded at training.

        Args:
            provided: Sequence names the caller intends to use for inference.

        Returns:
            List of non-fatal warning strings (empty if everything matches).

        Raises:
            ValueError: If any required sequence is absent from *provided*.
        """
        required = self.required_sequences
        if not required:
            return []  # legacy archive — no sequence info recorded

        provided_set = set(provided)
        required_set = set(required)
        missing = required_set - provided_set
        extra = provided_set - required_set

        warnings: List[str] = []
        if extra:
            warnings.append(
                f"Sequences provided but not used in training: {sorted(extra)}"
            )
        if missing:
            raise ValueError(
                f"Missing required sequences (use --override to ignore): {sorted(missing)}"
            )
        return warnings

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, archive_path: Union[str, Path]) -> None:
        """Write all artefacts into a zip archive at *archive_path*.

        Existing file is overwritten.

        Args:
            archive_path: Destination path (a ``.zip`` extension is appended
                if not already present).
        """
        archive_path = Path(archive_path)
        if archive_path.suffix != ".zip":
            archive_path = archive_path.with_suffix(".zip")

        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp) / "state"
            staging.mkdir()

            # 1. Clusterer
            shutil.copy2(self.clusterer_path, staging / _CLUSTERER_FNAME)

            # 2. PyRadiomics config
            shutil.copy2(self.pyrad_config_path, staging / _PYRAD_CONFIG_FNAME)

            # 3. Normaliser state directory
            norm_dst = staging / _NORM_STATE_DIRNAME
            if self.norm_state_path.is_dir():
                shutil.copytree(self.norm_state_path, norm_dst)
            elif self.norm_state_path.is_file():
                norm_dst.mkdir()
                shutil.copy2(self.norm_state_path, norm_dst / self.norm_state_path.name)
            else:
                logger.warning(f"Normaliser state path not found: {self.norm_state_path}")
                norm_dst.mkdir()

            # 4. Metadata
            (staging / _METADATA_FNAME).write_text(json.dumps(self.metadata, indent=2))

            # Create zip
            shutil.make_archive(str(archive_path.with_suffix("")), "zip", staging)

        logger.info(f"State saved to {archive_path}")

    @classmethod
    def load(cls, archive_path: Union[str, Path], extract_dir: Union[str, Path, None] = None) -> "HabitatState":
        """Load artefacts from a zip archive.

        The archive is extracted to *extract_dir* (a temporary directory is used
        if not specified).  The returned :class:`HabitatState` references files
        inside *extract_dir*, so the directory must remain valid for the lifetime
        of the returned object.

        Args:
            archive_path: Path to the ``.zip`` state archive.
            extract_dir: Directory to extract into.  Created if needed.  If
                ``None``, a caller-managed temporary directory is used (you are
                responsible for cleanup).

        Returns:
            Populated :class:`HabitatState`.
        """
        archive_path = Path(archive_path)
        if not archive_path.exists():
            raise FileNotFoundError(f"State archive not found: {archive_path}")

        if extract_dir is None:
            # Use a persistent temp dir; caller owns cleanup
            extract_dir = Path(tempfile.mkdtemp(prefix="habitat_state_"))
        else:
            extract_dir = Path(extract_dir)
            extract_dir.mkdir(parents=True, exist_ok=True)

        shutil.unpack_archive(str(archive_path), str(extract_dir), "zip")
        logger.info(f"State extracted to {extract_dir}")

        metadata_path = extract_dir / _METADATA_FNAME
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

        return cls(
            clusterer_path=extract_dir / _CLUSTERER_FNAME,
            pyrad_config_path=extract_dir / _PYRAD_CONFIG_FNAME,
            norm_state_path=extract_dir / _NORM_STATE_DIRNAME,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Convenience factory
    # ------------------------------------------------------------------

    @classmethod
    def from_parts(
        cls,
        clusterer_path: Union[str, Path],
        pyrad_config_path: Union[str, Path],
        norm_state_path: Union[str, Path],
        best_k: int,
        metrics: Dict,
        extra: Optional[Dict] = None,
    ) -> "HabitatState":
        """Create a HabitatState with a populated metadata dict.

        Args:
            clusterer_path: Path to saved clusterer joblib.
            pyrad_config_path: Path to pyradiomics YAML config.
            norm_state_path: Path to normaliser state directory.
            best_k: The cluster count selected by the training run.
            metrics: Per-k evaluation metrics from :class:`~clusterer.HabitatClusterer`.
            extra: Any additional key/value pairs to store in metadata.
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
        try:
            import numpy as np
            numpy_version = np.__version__
        except Exception:
            numpy_version = "unknown"

        metadata = {
            "best_k": best_k,
            "metrics": {str(k): v for k, v in metrics.items()},
            "python": platform.python_version(),
            "pyradiomics_version": pyrad_version,
            "sklearn_version": sklearn_version,
            "numpy_version": numpy_version,
        }
        if extra:
            metadata.update(extra)

        return cls(
            clusterer_path=clusterer_path,
            pyrad_config_path=pyrad_config_path,
            norm_state_path=norm_state_path,
            metadata=metadata,
        )
