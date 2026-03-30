"""
Orchestration for habitat analysis training and inference.

HabitatPipeline.train()  — normalise → extract → cluster → save state + features
HabitatPipeline.infer()  — load state → normalise → extract (or use cache) → predict → write
"""

import re
import tempfile
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import SimpleITK as sitk
from rich.console import Console as RichConsole
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

from .clusterer import HabitatClusterer
from .feature_extractor import PixelwiseFeatureExtractor
from .io_manager import HabitatIOManager
from .normalizer import HabitatNormalizer
from .state import HabitatState
from .visualization import label_map_to_nifti, render_habitat_overlay

from mnts.mnts_logger import MNTSLogger

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

logger: MNTSLogger = MNTSLogger[__name__]

@contextmanager
def _make_progress():
    """Progress bar that redirects all logging output through its console.

    Creates a dedicated Console with ``force_terminal=True``, temporarily
    swaps the MNTSLogger stream handler to use it, and suppresses any
    third-party loggers (e.g. pyradiomics) that have their own handlers
    writing to stderr.  This ensures ``Live`` is the sole writer so the
    progress bar stays pinned.
    """
    import logging

    handler = MNTSLogger.shared_handlers.get('stream_handler')
    old_console = handler.console if handler and hasattr(handler, 'console') else None

    progress_console = RichConsole(stderr=True, force_terminal=True)
    progress = Progress(
        SpinnerColumn(), BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=progress_console,
    )

    # Suppress third-party loggers that have their own StreamHandlers
    # (e.g. pyradiomics adds one on the 'radiomics' logger)
    suppressed = []
    for name in ('radiomics', 'pykwalify'):
        lg = logging.getLogger(name)
        for h in lg.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                old_level = h.level
                h.setLevel(logging.CRITICAL)
                suppressed.append((h, old_level))

    try:
        if handler and old_console is not None:
            handler.console = progress_console
        with progress:
            yield progress
    finally:
        if handler and old_console is not None:
            handler.console = old_console
        for h, old_level in suppressed:
            h.setLevel(old_level)


_DEFAULT_NORM_CONFIG = Path(__file__).parent.parent / "configs" / "normalization.yaml"
_DEFAULT_PYRAD_CONFIG = Path(__file__).parent.parent / "configs" / "pyradiomics_habitat.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_case(
    case_id: str,
    seq_paths: Dict[str, Path],
    mask_path: Path,
    seq_names: List[str],
    config_path: Path,
) -> Tuple[str, np.ndarray, np.ndarray, List[str]]:
    """Worker: extract multi-sequence features for one case (picklable)."""
    extractor = PixelwiseFeatureExtractor(config_path)
    images = {
        seq: sitk.ReadImage(str(seq_paths[seq]), sitk.sitkFloat32)
        for seq in seq_names
    }
    mask = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
    feats, indices, col_lbls = extractor.extract_multi_sequence(images, mask)
    return case_id, feats, indices, col_lbls


def _infer_case(
    case_id: str,
    seq_paths: Dict[str, Path],
    mask_path: Path,
    seq_dirs_keys: List[str],
    config_path: Path,
    clusterer_path: Path,
    out_dir: Path,
    visualize: bool,
    cached_feats: Optional[np.ndarray],
    cached_indices: Optional[np.ndarray],
    feature_columns: Optional[List[str]],
) -> Tuple[str, np.ndarray, Optional[List[str]], Path]:
    """Worker: extract (or use cache) → predict → write NIfTI/PNG for one case."""
    from .visualization import label_map_to_nifti, render_habitat_overlay

    clusterer = HabitatClusterer.load(clusterer_path)

    if cached_feats is not None:
        feats = cached_feats
        voxel_indices = cached_indices
        col_lbls = feature_columns
        image = sitk.ReadImage(
            str(seq_paths[next(iter(seq_dirs_keys))]), sitk.sitkFloat32
        )
    else:
        extractor = PixelwiseFeatureExtractor(config_path)
        images = {
            seq: sitk.ReadImage(str(seq_paths[seq]), sitk.sitkFloat32)
            for seq in seq_dirs_keys
        }
        mask_sitk = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
        feats, voxel_indices, col_lbls = extractor.extract_multi_sequence(images, mask_sitk)
        image = next(iter(images.values()))

    labels = clusterer.predict(feats)

    mask_sitk = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
    mask_array = sitk.GetArrayFromImage(mask_sitk)
    label_array = np.zeros_like(mask_array, dtype=np.int32)
    label_array[
        voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]
    ] = labels

    nifti_path = out_dir / f"{case_id}_habitat.nii.gz"
    label_map_to_nifti(label_array, mask_sitk, nifti_path)

    if visualize:
        img_array = sitk.GetArrayFromImage(image)
        render_habitat_overlay(
            img_array, label_array, mask_array,
            out_dir / f"{case_id}_overlay.png",
        )

    return case_id, voxel_indices, col_lbls, nifti_path


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


def _load_cached_features(
    out_state: Path,
    common_ids: List[str],
    seq_by_id: Dict[str, Dict[str, Path]],
    mask_by_id: Dict[str, Path],
    io_manager: HabitatIOManager,
) -> Optional[Tuple[np.ndarray, List[str]]]:
    """Try to load features from an existing state archive or output directory.

    If ``out_state`` already contains a ``features.parquet`` (from a previous
    training run), load it and reconstruct the feature matrix and io_manager
    records — skipping the expensive normalisation + extraction steps.

    Returns:
        ``(X, column_labels)`` if all *common_ids* are present in the cache,
        or ``None`` if the cache is missing, empty, or incomplete.
    """
    # Locate features.parquet
    parquet_path: Optional[Path] = None
    _tmp_state: Optional[HabitatState] = None

    if out_state.is_dir():
        candidate = out_state / "features.parquet"
        if candidate.is_file():
            parquet_path = candidate
    elif out_state.is_file() and out_state.suffix == ".zip":
        try:
            _tmp_state = HabitatState.load(out_state)
            if _tmp_state._features_path:
                parquet_path = _tmp_state._features_path
        except Exception as exc:
            logger.warning(f"Could not load state archive for caching: {exc}")

    if parquet_path is None:
        logger.info("No cached features.parquet found — will extract from scratch.")
        return None

    logger.info(f"Found cached features at {parquet_path}")
    df = pd.read_parquet(parquet_path)
    if len(df) == 0:
        logger.warning("Cached features.parquet is empty — will extract from scratch.")
        return None

    # Check coverage
    _meta_cols = {"case_id", "z", "y", "x", "cluster"}
    cached_ids = set(df["case_id"].unique())
    missing = set(common_ids) - cached_ids
    if missing:
        logger.warning(
            f"Cached features missing {len(missing)} case(s): "
            f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''} "
            f"— will extract all features from scratch."
        )
        return None

    # Filter to common_ids only and sort in common_ids order
    order = pd.DataFrame({"case_id": common_ids, "_order": range(len(common_ids))})
    df = df.merge(order, on="case_id", how="inner")
    df = df.sort_values(["_order", "z"], kind="stable").drop("_order", axis=1)

    # Extract feature columns and column labels
    feature_cols = [c for c in df.columns if c not in _meta_cols]
    column_labels = feature_cols

    # Reconstruct io_manager records from parquet
    for case_id in common_ids:
        sub = df[df["case_id"] == case_id]
        voxel_indices = sub[["z", "y", "x"]].values.astype(int)
        io_manager.register(
            sequence_paths=seq_by_id[case_id],
            segmentation_path=mask_by_id[case_id],
            voxel_indices=voxel_indices,
            column_labels=column_labels,
        )

    X = df[feature_cols].values.astype(np.float16)
    logger.info(
        f"Loaded cached features: {X.shape[0]} voxels x {X.shape[1]} features "
        f"for {len(common_ids)} cases"
    )
    return X, column_labels


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
        k_selection: str = "elbow",
        subsample: Optional[int] = 200_000,
        random_state: int = 42,
        visualize: bool = True,
    ):
        self.norm_config = Path(norm_config) if norm_config else _DEFAULT_NORM_CONFIG
        self.pyrad_config = Path(pyrad_config) if pyrad_config else _DEFAULT_PYRAD_CONFIG
        self.id_globber = id_globber
        self.cluster_method = cluster_method
        self.k_range = list(k_range)
        self.k_selection = k_selection
        self.subsample = subsample
        self.random_state = random_state
        self.visualize = visualize

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
        n_workers: int = 1,
        y_true: Optional[Dict[str, int]] = None,
        force_extract: bool = False,
    ) -> HabitatState:
        """Train normalisation and clustering and save the state archive.

        Steps:
            0. Check for cached ``features.parquet`` in *out_state*
               *(skipped if force_extract)*.  If found and complete, skip
               steps 1–3 entirely.
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
            y_true: Optional ``{case_id: label}`` mapping for supervised
                feature selection via t-test during clustering.  Each case's
                label is broadcast to all its voxels.
            force_extract: If ``True``, skip the feature cache check and
                always re-extract features from images.

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

        # Step 0: Check for cached features before expensive work
        # Resolve common IDs from raw seq_dirs (cheap directory listing)
        raw_common_ids, raw_seq_by_id, raw_mask_by_id = _common_ids(
            seq_dirs, mask_dir, self.id_globber
        )
        if max_cases is not None:
            raw_common_ids = raw_common_ids[:max_cases]
            logger.info(f"Debug: limiting to {len(raw_common_ids)} cases.")

        cached_result = None
        if not force_extract:
            cached_result = _load_cached_features(
                out_state, raw_common_ids, raw_seq_by_id, raw_mask_by_id, io_manager
            )

        normalizer = HabitatNormalizer(self.norm_config, self.id_globber)
        extractor = PixelwiseFeatureExtractor(self.pyrad_config)

        with tempfile.TemporaryDirectory() as _tmp:
            tmp = Path(_tmp)

            if cached_result is not None:
                # Cache hit — skip normalization and extraction
                X, column_labels = cached_result
                common_ids = raw_common_ids
                seq_by_id = raw_seq_by_id
                mask_by_id = raw_mask_by_id
                logger.info("=== Steps 1–3/4: Using cached features ===")

                # Still need norm_state_dirs for the state archive
                norm_state_dirs: Dict[str, Path] = {}
                for seq in seq_names:
                    empty_dir = tmp / f"norm_state_{seq}"
                    empty_dir.mkdir()
                    norm_state_dirs[seq] = empty_dir
            else:
                # Cache miss — full normalization + extraction pipeline
                if force_extract:
                    logger.info("--force-extract: ignoring cached features.")

                norm_state_dirs = {}

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

                # Get features for each image pair
                if n_workers <= 1:
                    # Sequential path
                    with _make_progress() as progress:
                        task = progress.add_task("[cyan]Feature extraction", total=len(common_ids))
                        for case_id in common_ids:
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
                            progress.advance(task)
                else:
                    # Parallel path
                    logger.info(f"Using {n_workers} workers for feature extraction.")
                    results: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]] = {}
                    with ProcessPoolExecutor(max_workers=n_workers) as pool:
                        future_to_id = {
                            pool.submit(
                                _extract_case,
                                cid, seq_by_id[cid], mask_by_id[cid],
                                seq_names, self.pyrad_config,
                            ): cid
                            for cid in common_ids
                        }
                        with _make_progress() as progress:
                            task = progress.add_task("[cyan]Feature extraction", total=len(common_ids))
                            for future in as_completed(future_to_id):
                                cid = future_to_id[future]
                                try:
                                    _, feats, indices, col_lbls = future.result()
                                    results[cid] = (feats, indices, col_lbls)
                                except Exception as exc:
                                    logger.warning(f"Skipping '{cid}': {exc}")
                                progress.advance(task)

                    # Merge in case order to preserve io_manager row ranges
                    for case_id in common_ids:
                        if case_id not in results:
                            continue
                        feats, indices, col_lbls = results[case_id]
                        all_features.append(feats)
                        if not column_labels:
                            column_labels = col_lbls
                        io_manager.register(
                            sequence_paths=seq_by_id[case_id],
                            segmentation_path=mask_by_id[case_id],
                            voxel_indices=indices,
                            column_labels=col_lbls,
                        )

                if not all_features:
                    raise RuntimeError("No feature matrices were produced. Check input data.")

                X = np.concatenate(all_features, axis=0)
                logger.info(f"Combined feature matrix: {X.shape[0]} voxels × {X.shape[1]} features")

            # Build per-voxel y_true array from case-level labels
            y_voxel = None
            if y_true is not None:
                y_parts = []
                for case_id, rec in io_manager.records.items():
                    if case_id in y_true:
                        y_parts.append(np.full(rec.n_voxels, y_true[case_id], dtype=np.float32))
                    else:
                        logger.warning(f"y_true missing for case '{case_id}', using NaN")
                        y_parts.append(np.full(rec.n_voxels, np.nan, dtype=np.float32))
                y_voxel = np.concatenate(y_parts, axis=0)
                # Drop voxels with missing labels for the selector
                valid = ~np.isnan(y_voxel)
                if not valid.all():
                    logger.warning(f"y_true: {(~valid).sum()} voxels have no label and will be ignored by t-test")
                    y_voxel[~valid] = 0  # placeholder; selector uses only valid rows via subsample

            # Step 4: Cluster
            logger.info("=== Step 4/4: Clustering ===")
            clusterer = HabitatClusterer(
                method=self.cluster_method,
                k_range=self.k_range,
                k_selection=self.k_selection,
                random_state=self.random_state,
                subsample=self.subsample,
                visualize=self.visualize,
            )
            clusterer.fit(X, y_true=y_voxel)
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

            # Write cluster visualization and per-case habitat segmentations.
            # Must happen AFTER state.save() — when out_state is a plain
            # directory, save() rmtrees it before recreating from staging.
            if out_seg_dir is not None:
                seg_dir = Path(out_seg_dir)
                seg_dir.mkdir(parents=True, exist_ok=True)

                if clusterer.visualize:
                    X_vis = np.clip(X.astype(np.float32), clusterer.clip_lo_, clusterer.clip_hi_)
                    X_vis = clusterer._apply_selector(X_vis)
                    X_scaled = clusterer.scaler.transform(X_vis)
                    # Filter feature names to match selected columns
                    vis_names = column_labels
                    if clusterer.feature_mask_ is not None:
                        vis_names = [n for n, m in zip(column_labels, clusterer.feature_mask_) if m]
                    clusterer.visualize_cluster_results(
                        X_scaled, seg_dir / f"cluster_pca_k{clusterer.best_k}.png",
                        feature_names=vis_names,
                    )

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
        n_workers: int = 1,
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
            feature_columns = state.metadata.get("feature_columns")

            if n_workers <= 1:
                # Sequential path
                with _make_progress() as progress:
                    infer_task = progress.add_task("[cyan]Inference", total=len(common_ids))
                    for case_id in common_ids:
                        try:
                            cached = state.get_case_features(case_id)
                            if cached is not None:
                                logger.info(f"Using cached features for '{case_id}'.")
                                feats = cached[
                                    [c for c in cached.columns if c not in {"case_id", "z", "y", "x", "cluster"}]
                                ].values.astype(np.float16)
                                voxel_indices = cached[["z", "y", "x"]].values.astype(int)
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
                                column_labels=feature_columns,
                            )

                            labels = clusterer.predict(feats)

                            mask_sitk = sitk.ReadImage(str(mask_by_id[case_id]), sitk.sitkUInt8)
                            mask_array = sitk.GetArrayFromImage(mask_sitk)
                            label_array = np.zeros_like(mask_array, dtype=np.int32)
                            label_array[
                                voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2],
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
                        progress.advance(infer_task)
            else:
                # Parallel path
                logger.info(f"Using {n_workers} workers for inference.")
                # Pre-resolve cached features (state DataFrame access is not picklable)
                cache_by_id: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}
                for cid in common_ids:
                    cached = state.get_case_features(cid)
                    if cached is not None:
                        logger.info(f"Using cached features for '{cid}'.")
                        c_feats = cached[
                            [c for c in cached.columns if c not in {"case_id", "z", "y", "x", "cluster"}]
                        ].values.astype(np.float16)
                        c_indices = cached[["z", "y", "x"]].values.astype(int)
                        cache_by_id[cid] = (c_feats, c_indices)
                    else:
                        cache_by_id[cid] = (None, None)

                infer_results: Dict[str, Tuple[np.ndarray, Optional[List[str]], Path]] = {}
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    future_to_id = {
                        pool.submit(
                            _infer_case,
                            cid, seq_by_id[cid], mask_by_id[cid],
                            list(seq_dirs.keys()), state.pyrad_config_path,
                            state.clusterer_path, out_dir, visualize,
                            cache_by_id[cid][0], cache_by_id[cid][1],
                            feature_columns,
                        ): cid
                        for cid in common_ids
                    }
                    with _make_progress() as progress:
                        task = progress.add_task("[cyan]Inference", total=len(common_ids))
                        for future in as_completed(future_to_id):
                            cid = future_to_id[future]
                            try:
                                _, voxel_indices, col_lbls, nifti_path = future.result()
                                infer_results[cid] = (voxel_indices, col_lbls, nifti_path)
                            except Exception as exc:
                                logger.error(f"Failed on case '{cid}': {exc}")
                            progress.advance(task)

                # Merge in case order
                for case_id in common_ids:
                    if case_id not in infer_results:
                        continue
                    voxel_indices, col_lbls, nifti_path = infer_results[case_id]
                    io_manager.register(
                        sequence_paths=seq_by_id[case_id],
                        segmentation_path=mask_by_id[case_id],
                        voxel_indices=voxel_indices,
                        column_labels=feature_columns,
                    )
                    output_paths.append(nifti_path)

        logger.info(f"Inference complete. {len(output_paths)} cases written to {out_dir}")
        return output_paths
