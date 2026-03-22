"""
Orchestration for habitat analysis training and inference.

HabitatPipeline.train()  — normalise → extract features → cluster → save state
HabitatPipeline.infer()  — load state → normalise → extract → predict → write outputs
"""

import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from .clusterer import HabitatClusterer
from .feature_extractor import PixelwiseFeatureExtractor
from .normalizer import HabitatNormalizer
from .state import HabitatState
from .visualization import label_map_to_nifti, render_habitat_overlay

from mnts.mnts_logger import MNTSLogger

logger = MNTSLogger[__name__]

_DEFAULT_NORM_CONFIG = Path(__file__).parent.parent / "configs" / "normalization.yaml"
_DEFAULT_PYRAD_CONFIG = Path(__file__).parent.parent / "configs" / "pyradiomics_habitat.yaml"


def _match_pairs(img_dir: Path, mask_dir: Path, id_globber: str) -> List[Tuple[Path, Path]]:
    """Return sorted (image_path, mask_path) pairs matched by case ID.

    Falls back to alphabetical ordering if ID extraction fails.
    """
    import re

    img_files = sorted(img_dir.glob("*.nii.gz"))
    mask_files = sorted(mask_dir.glob("*.nii.gz"))

    if len(img_files) != len(mask_files):
        raise ValueError(
            f"Number of images ({len(img_files)}) does not match masks ({len(mask_files)})."
        )

    def extract_id(p: Path) -> str:
        m = re.search(id_globber, p.name)
        return m.group() if m else p.stem

    img_by_id = {extract_id(f): f for f in img_files}
    mask_by_id = {extract_id(f): f for f in mask_files}
    common = sorted(set(img_by_id) & set(mask_by_id))

    if not common:
        logger.warning("ID matching found no common IDs; falling back to alphabetical order.")
        return list(zip(img_files, mask_files))

    return [(img_by_id[i], mask_by_id[i]) for i in common]


class HabitatPipeline:
    """End-to-end habitat analysis pipeline.

    Args:
        norm_config: Path to mnts normalisation YAML (default: bundled config).
        pyrad_config: Path to pyradiomics YAML (default: bundled config).
        id_globber: Regex to extract case IDs from filenames.
        cluster_method: ``"kmeans"`` or ``"gmm"``.
        k_range: Candidate cluster counts for the grid search.
        subsample: Max voxels to use for clustering fit (None = use all).
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
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        img_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        out_state: Union[str, Path],
        io_manager=None,
        extra_metadata: Optional[dict] = None,
    ) -> HabitatState:
        """Train normalisation and clustering on a dataset and save the state.

        Steps:
            1. Train NyulNormalizer on *img_dir* / *mask_dir*.
            2. Run normalisation inference to a temp directory.
            3. Extract pixelwise feature matrices for all cases.
            4. Fit :class:`~clusterer.HabitatClusterer` on the combined matrix.
            5. Save :class:`~state.HabitatState` to *out_state*.

        Args:
            img_dir: Directory of input NIfTI images.
            mask_dir: Directory of corresponding binary masks.
            out_state: Destination path for the state archive (``.zip``).

        Returns:
            The saved :class:`~state.HabitatState`.
        """
        img_dir = Path(img_dir)
        mask_dir = Path(mask_dir)
        out_state = Path(out_state)

        normalizer = HabitatNormalizer(self.norm_config, self.id_globber)
        extractor = PixelwiseFeatureExtractor(self.pyrad_config)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            norm_state_dir = tmp / "norm_state"
            norm_img_dir = tmp / "norm_images"

            # Step 1: Train normaliser
            logger.info("=== Step 1/4: Training normaliser ===")
            normalizer.train(img_dir, mask_dir, norm_state_dir)

            # Step 2: Apply normalisation
            logger.info("=== Step 2/4: Applying normalisation ===")
            normalizer.infer(img_dir, norm_img_dir, norm_state_dir, mask_dir=mask_dir)

            # Step 3: Extract features
            logger.info("=== Step 3/4: Extracting pixelwise features ===")
            pairs = _match_pairs(norm_img_dir, mask_dir, self.id_globber)
            all_features: List[np.ndarray] = []

            for img_path, mask_path in tqdm(pairs, desc="Feature extraction"):
                try:
                    feats, indices, _ = extractor.extract_from_files(img_path, mask_path)
                    all_features.append(feats)
                    if io_manager is not None:
                        seq_name = next(iter(io_manager.sequence_names), "image")
                        io_manager.register(
                            sequence_paths={seq_name: img_path},
                            segmentation_path=mask_path,
                            voxel_indices=indices,
                            column_labels=[f"{seq_name}__{f}" for f in extractor.column_labels_],
                        )
                except Exception as exc:
                    logger.warning(f"Skipping {img_path.name}: {exc}")

            if not all_features:
                raise RuntimeError("No feature matrices were produced. Check input data.")

            X = np.concatenate(all_features, axis=0)
            logger.info(f"Combined feature matrix: {X.shape[0]} voxels × {X.shape[1]} filters")

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

            # Save clusterer and build state
            clusterer_path = tmp / "clusterer.joblib"
            clusterer.save(clusterer_path)

            state = HabitatState.from_parts(
                clusterer_path=clusterer_path,
                pyrad_config_path=self.pyrad_config,
                norm_state_path=norm_state_dir,
                best_k=clusterer.best_k,
                metrics=clusterer.metrics,
                extra={"cluster_method": self.cluster_method, "k_range": self.k_range,
                       **(extra_metadata or {})},
            )
            state.save(out_state)

        logger.info(f"Training complete. State saved to {out_state}")
        return HabitatState.load(out_state)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(
        self,
        img_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        state_path: Union[str, Path],
        out_dir: Union[str, Path],
        visualize: bool = True,
        io_manager=None,
    ) -> List[Path]:
        """Apply a trained pipeline to new images and write habitat segmentations.

        Args:
            img_dir: Directory of input NIfTI images.
            mask_dir: Directory of corresponding binary masks.
            state_path: Path to the ``.zip`` state archive.
            out_dir: Directory to write output files.
            visualize: If True, write PNG overlay images alongside NIfTI outputs.

        Returns:
            List of paths to the output NIfTI label maps.
        """
        img_dir = Path(img_dir)
        mask_dir = Path(mask_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            # Load state
            state = HabitatState.load(state_path, extract_dir=tmp / "state")
            clusterer = HabitatClusterer.load(state.clusterer_path)
            extractor = PixelwiseFeatureExtractor(state.pyrad_config_path)
            normalizer = HabitatNormalizer(self.norm_config, self.id_globber)

            # Normalise images
            logger.info("Normalising images...")
            norm_img_dir = tmp / "norm_images"
            normalizer.infer(img_dir, norm_img_dir, state.norm_state_path, mask_dir=mask_dir)

            pairs = _match_pairs(norm_img_dir, mask_dir, self.id_globber)
            output_paths: List[Path] = []

            for img_path, mask_path in tqdm(pairs, desc="Inference"):
                case_id = img_path.stem.replace(".nii", "")
                try:
                    feats, voxel_indices, image = extractor.extract_from_files(img_path, mask_path)
                    if io_manager is not None:
                        seq_name = next(iter(io_manager.sequence_names), "image")
                        io_manager.register(
                            sequence_paths={seq_name: img_path},
                            segmentation_path=mask_path,
                            voxel_indices=voxel_indices,
                            column_labels=[f"{seq_name}__{f}" for f in extractor.column_labels_],
                        )
                    labels = clusterer.predict(feats)

                    # Build 3-D label volume
                    mask_sitk = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
                    mask_array = sitk.GetArrayFromImage(mask_sitk)
                    label_array = np.zeros_like(mask_array, dtype=np.int32)
                    label_array[
                        voxel_indices[:, 0],
                        voxel_indices[:, 1],
                        voxel_indices[:, 2],
                    ] = labels

                    nifti_path = out_dir / f"{case_id}_habitat.nii.gz"
                    label_map_to_nifti(label_array, image, nifti_path)
                    output_paths.append(nifti_path)

                    if visualize:
                        img_array = sitk.GetArrayFromImage(image)
                        png_path = out_dir / f"{case_id}_overlay.png"
                        render_habitat_overlay(img_array, label_array, mask_array, png_path)

                except Exception as exc:
                    logger.error(f"Failed on case {case_id}: {exc}")

        logger.info(f"Inference complete. {len(output_paths)} cases written to {out_dir}")
        return output_paths
