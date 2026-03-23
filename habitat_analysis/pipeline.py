"""
Orchestration for habitat analysis training and inference.

HabitatPipeline.train()  — normalise → extract → cluster → save state + features
HabitatPipeline.infer()  — load state → normalise → extract (or use cache) → predict → write
"""

import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import SimpleITK as sitk
from rich.progress import track

from .clusterer import HabitatClusterer
from .feature_extractor import PixelwiseFeatureExtractor
from .io_manager import HabitatIOManager
from .normalizer import HabitatNormalizer
from .state import HabitatState
from .visualization import label_map_to_nifti, render_habitat_overlay

from mnts.mnts_logger import MNTSLogger

logger: MNTSLogger = MNTSLogger[__name__]

_DEFAULT_NORM_CONFIG = Path(__file__).parent.parent / "configs" / "normalization.yaml"
_DEFAULT_PYRAD_CONFIG = Path(__file__).parent.parent / "configs" / "pyradiomics_habitat.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_id(path: Path, globber: str) -> str:
    m = re.search(globber, path.name)
    return m.group() if m else path.stem


def _index_dir(directory: Path, globber: str) -> Dict[str, Path]:
    """Return ``{case_id: path}`` for all ``.nii.gz`` files in *directory*."""
    return {_extract_id(f, globber): f for f in sorted(directory.glob("*.nii.gz"))}


def _common_ids(
    seq_dirs: Dict[str, Path],
    mask_dir: Path,
    globber: str,
) -> Tuple[List[str], Dict[str, Dict[str, Path]], Dict[str, Path]]:
    """Find case IDs present in every sequence directory and the mask directory.

    Returns:
        common_ids: Sorted list of shared case IDs.
        seq_by_id:  ``{case_id: {seq_name: img_path}}`` for every common ID.
        mask_by_id: ``{case_id: mask_path}`` for every common ID.
    """
    mask_index = _index_dir(mask_dir, globber)
    seq_indices = {seq: _index_dir(d, globber) for seq, d in seq_dirs.items()}

    # Intersect IDs across all sequences and masks
    all_id_sets = [set(mask_index)] + [set(idx) for idx in seq_indices.values()]
    common = sorted(set.intersection(*all_id_sets))

    if not common:
        raise ValueError(
            "No common case IDs found across sequences and masks. "
            "Check id_globber and filenames."
        )

    seq_by_id: Dict[str, Dict[str, Path]] = {
        cid: {seq: seq_indices[seq][cid] for seq in seq_dirs}
        for cid in common
    }
    mask_by_id = {cid: mask_index[cid] for cid in common}
    return common, seq_by_id, mask_by_id


def build_features_df(
    features: np.ndarray,
    io_manager: HabitatIOManager,
    column_labels: List[str],
) -> pd.DataFrame:
    """Assemble the full voxel feature table from a combined feature matrix.

    Args:
        features: ``(N_total_voxels, N_features)`` float array.
        io_manager: Populated :class:`~io_manager.HabitatIOManager`.
        column_labels: Feature column names (``"{seq}__{filter}"`` format).

    Returns:
        DataFrame with columns ``case_id, z, y, x, <feature cols…>``.
    """
    case_ids: List[str] = []
    coords_list: List[np.ndarray] = []

    for case_id, rec in io_manager.records.items():
        case_ids.extend([case_id] * rec.n_voxels)
        coords_list.append(rec.voxel_indices)

    coords = np.concatenate(coords_list, axis=0)  # (N_total, 3)

    df = pd.DataFrame({
        "case_id": case_ids,
        "z": coords[:, 0].astype(np.int32),
        "y": coords[:, 1].astype(np.int32),
        "x": coords[:, 2].astype(np.int32),
    })
    for j, col in enumerate(column_labels):
        df[col] = features[:, j].astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class HabitatPipeline:
    """End-to-end habitat analysis pipeline.

    Args:
        norm_config: Path to mnts normalisation YAML (default: bundled).
        pyrad_config: Path to pyradiomics YAML (default: bundled).
        id_globber: Regex to extract case IDs from filenames.
        cluster_method: ``"kmeans"`` or ``"gmm"``.
        k_range: Candidate cluster counts for the grid search.
        subsample: Max voxels used for clustering fit (``None`` = use all).
        random_state: RNG seed.
    """

    def __init__(
        self,
        norm_config: Union[str, Path, None] = None,
        pyrad_config: Union[str, Path, None] = None,
        id_globber: str = r"^[0-9a-zA-Z]+",
        cluster_method: str = "kmeans",
        k_range=range(2, 7),
        subsample: Optional[int] = 200_000,
        random_state: int = 42,
    ):
        self.norm_config = Path(norm_config) if norm_config else _DEFAULT_NORM_CONFIG
        self.pyrad_config = Path(pyrad_config) if pyrad_config else _DEFAULT_PYRAD_CONFIG
        self.id_globber = id_globber
        self.cluster_method = cluster_method
        self.k_range = list(k_range)
        self.subsample = subsample
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_seq_dirs(
        self,
        seq_dirs: Union[Dict[str, Union[str, Path]], str, Path],
    ) -> Dict[str, Path]:
        """Accept a dict of sequences or a plain path (single-sequence compat)."""
        if isinstance(seq_dirs, (str, Path)):
            return {"image": Path(seq_dirs)}
        return {k: Path(v) for k, v in seq_dirs.items()}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        seq_dirs: Union[Dict[str, Union[str, Path]], str, Path],
        mask_dir: Union[str, Path],
        out_state: Union[str, Path],
        out_seg_dir: Optional[Union[str, Path]] = None,
        io_manager: Optional[HabitatIOManager] = None,
        extra_metadata: Optional[dict] = None,
        skip_norm: bool = False,
        max_cases: Optional[int] = None,
    ) -> HabitatState:
        """Train normalisation and clustering and save the state archive.

        Steps:
            1. Train one NyulNormalizer per sequence  *(skipped if skip_norm)*.
            2. Apply normalisation for all sequences  *(skipped if skip_norm)*.
            3. Extract per-voxel multi-sequence features for every case.
            4. Fit :class:`~clusterer.HabitatClusterer` on the combined matrix.
            5. Save :class:`~state.HabitatState` (includes ``features.parquet``).

        Args:
            seq_dirs: ``{sequence_name: image_dir}`` mapping, or a single
                image directory path (treated as ``{"image": path}``).
            mask_dir: Directory of binary mask NIfTI files.
            out_state: Destination path for the state archive (``.zip``).
            io_manager: Optional :class:`~io_manager.HabitatIOManager` to
                receive voxel provenance records.  A temporary one is created
                internally if not supplied.
            extra_metadata: Extra key/value pairs merged into ``metadata.json``.
            skip_norm: If ``True``, assume *seq_dirs* already contain
                normalised images and skip normaliser training and application.
                An empty normaliser state is stored in the archive.

        Returns:
            The saved and re-loaded :class:`~state.HabitatState`.
        """
        seq_dirs = self._resolve_seq_dirs(seq_dirs)
        mask_dir = Path(mask_dir)
        out_state = Path(out_state)
        seq_names = list(seq_dirs.keys())

        # Use caller's IO manager or create an internal one for provenance
        own_manager = io_manager is None
        if own_manager:
            io_manager = HabitatIOManager(
                pipeline=self,
                sequence_paths=seq_dirs,
                segmentation_path=mask_dir,
                id_globber=self.id_globber,
            )

        normalizer = HabitatNormalizer(self.norm_config, self.id_globber)
        extractor = PixelwiseFeatureExtractor(self.pyrad_config)

        with tempfile.TemporaryDirectory() as _tmp:
            tmp = Path(_tmp)

            norm_state_dirs: Dict[str, Path] = {}

            if skip_norm:
                logger.info("=== Steps 1–2/4: Skipping normalisation (--skip-norm) ===")
                for seq in seq_names:
                    empty_dir = tmp / f"norm_state_{seq}"
                    empty_dir.mkdir()
                    norm_state_dirs[seq] = empty_dir
                norm_img_dirs: Dict[str, Path] = dict(seq_dirs)
            else:
                # Step 1: Train one normaliser per sequence
                logger.info(f"=== Step 1/4: Training normalisers ({seq_names}) ===")
                for seq, img_dir in seq_dirs.items():
                    nsd = tmp / f"norm_state_{seq}"
                    normalizer.train(img_dir, mask_dir, nsd)
                    norm_state_dirs[seq] = nsd

                # Step 2: Apply normalisation for each sequence
                logger.info("=== Step 2/4: Applying normalisation ===")
                norm_img_dirs = {}
                for seq, img_dir in seq_dirs.items():
                    nid = tmp / f"norm_images_{seq}"
                    normalizer.infer(img_dir, nid, norm_state_dirs[seq], mask_dir=mask_dir)
                    norm_img_dirs[seq] = nid

            # Step 3: Multi-sequence feature extraction
            logger.info("=== Step 3/4: Extracting multi-sequence pixelwise features ===")
            common_ids, seq_by_id, mask_by_id = _common_ids(
                norm_img_dirs, mask_dir, self.id_globber
            )
            if max_cases is not None:
                common_ids = common_ids[:max_cases]
                logger.info(f"Debug: limiting to {len(common_ids)} cases.")

            all_features: List[np.ndarray] = []
            column_labels: List[str] = []

            for case_id in track(common_ids, description="[cyan]Feature extraction"):
                try:
                    images = {
                        seq: sitk.ReadImage(str(seq_by_id[case_id][seq]), sitk.sitkFloat32)
                        for seq in seq_names
                    }
                    mask_sitk = sitk.ReadImage(str(mask_by_id[case_id]), sitk.sitkUInt8)

                    feats, indices, col_lbls = extractor.extract_multi_sequence(images, mask_sitk)
                    all_features.append(feats)
                    if not column_labels:
                        column_labels = col_lbls

                    io_manager.register(
                        sequence_paths=seq_by_id[case_id],
                        segmentation_path=mask_by_id[case_id],
                        voxel_indices=indices,
                        column_labels=col_lbls,
                    )
                except Exception as exc:
                    logger.warning(f"Skipping '{case_id}': {exc}")

            if not all_features:
                raise RuntimeError("No feature matrices were produced. Check input data.")

            X = np.concatenate(all_features, axis=0)
            logger.info(f"Combined feature matrix: {X.shape[0]} voxels × {X.shape[1]} features")

            # Step 4: Cluster
            logger.info("=== Step 4/4: Clustering ===")
            clusterer = HabitatClusterer(
                method=self.cluster_method,
                k_range=self.k_range,
                random_state=self.random_state,
                subsample=self.subsample,
            )
            clusterer.fit(X)
            logger.info(f"\n{clusterer.metrics_summary()}")

            # Build feature DataFrame for archiving (includes cluster labels)
            features_df = build_features_df(X, io_manager, column_labels)
            all_labels = clusterer.predict(X).astype(np.int32)
            features_df["cluster"] = all_labels

            # Save state (must happen before segmentation write — state.save() may
            # rmtree the output directory if it already exists as a plain dir)
            clusterer_path = tmp / "clusterer.joblib"
            clusterer.save(clusterer_path)

            state = HabitatState.from_parts(
                clusterer_path=clusterer_path,
                pyrad_config_path=self.pyrad_config,
                norm_state_paths=norm_state_dirs,
                best_k=clusterer.best_k,
                metrics=clusterer.metrics,
                extra={
                    "cluster_method": self.cluster_method,
                    "k_range": self.k_range,
                    "sequences": seq_names,
                    "feature_columns": column_labels,
                    "skip_norm": skip_norm,
                    **(extra_metadata or {}),
                },
            )
            state.save(out_state, features_df=features_df)

            # Write per-case habitat segmentations (after state.save so the
            # output directory is not wiped underneath us)
            if out_seg_dir is not None:
                seg_dir = Path(out_seg_dir)
                seg_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Writing segmentations to {seg_dir}")
                for case_id, rec in io_manager.records.items():
                    case_labels = all_labels[rec.row_start:rec.row_end]
                    mask_sitk = sitk.ReadImage(str(rec.segmentation_path), sitk.sitkUInt8)
                    mask_array = sitk.GetArrayFromImage(mask_sitk)
                    label_array = np.zeros_like(mask_array, dtype=np.int32)
                    label_array[
                        rec.voxel_indices[:, 0],
                        rec.voxel_indices[:, 1],
                        rec.voxel_indices[:, 2],
                    ] = case_labels
                    label_sitk = sitk.GetImageFromArray(label_array)
                    label_sitk.CopyInformation(mask_sitk)
                    out_path = seg_dir / f"{case_id}_habitat.nii.gz"
                    sitk.WriteImage(label_sitk, str(out_path))
                    logger.info(f"Written: {out_path.absolute()}")
                logger.info(f"Segmentations written to {seg_dir}")

        logger.info(f"Training complete. State saved to {out_state}")
        return HabitatState.load(out_state)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(
        self,
        seq_dirs: Union[Dict[str, Union[str, Path]], str, Path],
        mask_dir: Union[str, Path],
        state_path: Union[str, Path],
        out_dir: Union[str, Path],
        visualize: bool = True,
        io_manager: Optional[HabitatIOManager] = None,
        skip_norm: bool = False,
        max_cases: Optional[int] = None,
    ) -> List[Path]:
        """Apply a trained pipeline to new data and write habitat segmentations.

        For each case the pipeline:
        1. Checks whether cached features exist in the state archive.
        2. If not, normalises *(unless skip_norm)* and extracts features.
        3. Predicts cluster labels and writes NIfTI + optional PNG overlays.

        Args:
            seq_dirs: ``{sequence_name: image_dir}`` or a single image dir.
            mask_dir: Directory of binary mask NIfTI files.
            state_path: Path to the ``.zip`` state archive.
            out_dir: Directory for output label maps and overlays.
            visualize: Write PNG overlay images alongside NIfTI outputs.
            io_manager: Optional :class:`~io_manager.HabitatIOManager` to
                receive voxel provenance records for inferred cases.
            skip_norm: If ``True``, assume *seq_dirs* already contain
                normalised images and skip the normalisation step.

        Returns:
            List of output NIfTI label-map paths.
        """
        seq_dirs = self._resolve_seq_dirs(seq_dirs)
        mask_dir = Path(mask_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        own_manager = io_manager is None
        if own_manager:
            io_manager = HabitatIOManager(
                pipeline=self,
                sequence_paths=seq_dirs,
                segmentation_path=mask_dir,
                id_globber=self.id_globber,
            )

        with tempfile.TemporaryDirectory() as _tmp:
            tmp = Path(_tmp)

            state = HabitatState.load(state_path, extract_dir=tmp / "state")
            clusterer = HabitatClusterer.load(state.clusterer_path)
            extractor = PixelwiseFeatureExtractor(state.pyrad_config_path)
            normalizer = HabitatNormalizer(self.norm_config, self.id_globber)

            if skip_norm:
                logger.info("Skipping normalisation (--skip-norm).")
                norm_img_dirs: Dict[str, Path] = dict(seq_dirs)
            else:
                logger.info("Normalising input sequences...")
                norm_img_dirs = {}
                for seq, img_dir in seq_dirs.items():
                    nid = tmp / f"norm_images_{seq}"
                    norm_state = state.norm_state_paths.get(seq, state.norm_state_path)
                    normalizer.infer(img_dir, nid, norm_state, mask_dir=mask_dir)
                    norm_img_dirs[seq] = nid

            common_ids, seq_by_id, mask_by_id = _common_ids(
                norm_img_dirs, mask_dir, self.id_globber
            )
            if max_cases is not None:
                common_ids = common_ids[:max_cases]
                logger.info(f"Debug: limiting to {len(common_ids)} cases.")

            output_paths: List[Path] = []

            for case_id in track(common_ids, description="[cyan]Inference"):
                try:
                    # --- Attempt to reuse cached training features ---
                    cached = state.get_case_features(case_id)
                    if cached is not None:
                        logger.info(f"Using cached features for '{case_id}'.")
                        feats = cached[
                            [c for c in cached.columns if c not in {"case_id", "z", "y", "x", "cluster"}]
                        ].values.astype(np.float16)
                        voxel_indices = cached[["z", "y", "x"]].values.astype(int)
                        # Load image geometry from first sequence for NIfTI writing
                        image = sitk.ReadImage(
                            str(seq_by_id[case_id][next(iter(seq_dirs))]),
                            sitk.sitkFloat32,
                        )
                    else:
                        images = {
                            seq: sitk.ReadImage(str(seq_by_id[case_id][seq]), sitk.sitkFloat32)
                            for seq in seq_dirs
                        }
                        mask_sitk = sitk.ReadImage(str(mask_by_id[case_id]), sitk.sitkUInt8)
                        feats, voxel_indices, col_lbls = extractor.extract_multi_sequence(
                            images, mask_sitk
                        )
                        image = next(iter(images.values()))

                    io_manager.register(
                        sequence_paths=seq_by_id[case_id],
                        segmentation_path=mask_by_id[case_id],
                        voxel_indices=voxel_indices,
                        column_labels=state.metadata.get("feature_columns"),
                    )

                    labels = clusterer.predict(feats)

                    # Reconstruct 3-D label volume
                    mask_sitk = sitk.ReadImage(str(mask_by_id[case_id]), sitk.sitkUInt8)
                    mask_array = sitk.GetArrayFromImage(mask_sitk)
                    label_array = np.zeros_like(mask_array, dtype=np.int32)
                    label_array[
                        voxel_indices[:, 0],
                        voxel_indices[:, 1],
                        voxel_indices[:, 2],
                    ] = labels

                    nifti_path = out_dir / f"{case_id}_habitat.nii.gz"
                    label_map_to_nifti(label_array, mask_sitk, nifti_path)
                    output_paths.append(nifti_path)

                    if visualize:
                        img_array = sitk.GetArrayFromImage(image)
                        render_habitat_overlay(
                            img_array, label_array, mask_array,
                            out_dir / f"{case_id}_overlay.png",
                        )

                except Exception as exc:
                    logger.error(f"Failed on case '{case_id}': {exc}")

        logger.info(f"Inference complete. {len(output_paths)} cases written to {out_dir}")
        return output_paths
