"""
Pixelwise multi-filter feature extractor for habitat analysis.

For each pyradiomics image-type filter (Original, LBP2D, LBP3D, LoG, Gradient,
Exponential), a filtered volume is produced and then smoothed with a 3×3×1
in-plane average convolution (pseudo-3D kernel).  Only voxels within the provided
mask are retained, yielding a feature matrix of shape (N_voxels, N_filters).

The matrix is stored as FP16 to limit memory use.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import uniform_filter
from mnts.mnts_logger import MNTSLogger

logger = MNTSLogger[__name__]

_DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "pyradiomics_habitat.yaml"

# Convolution kernel size: [row, col, slice] — 3×3 in-plane, 1 slice
_CONV_SIZE = [3, 3, 1]


def _get_filtered_image(extractor, image: sitk.Image, mask: sitk.Image, image_type: str) -> Optional[sitk.Image]:
    """Return the filtered image for a given pyradiomics imageType.

    Pyradiomics applies image type filters internally when features are extracted.
    This helper uses the low-level filter API to obtain just the filtered image.
    """
    try:
        import radiomics.imageoperations as imgops

        func_name = f"get{image_type}Image"
        func = getattr(imgops, func_name, None)
        if func is None:
            logger.warning(f"No filter function found for image type '{image_type}'.")
            return None

        filtered_imgs = list(func(image, mask, **extractor.settings))
        if not filtered_imgs:
            return None
        # Each yielded item is (filtered_image, filtered_mask, kwargs)
        return filtered_imgs[0][0]

    except Exception as exc:
        logger.warning(f"Failed to apply filter '{image_type}': {exc}")
        return None


class PixelwiseFeatureExtractor:
    """Extract a per-voxel multi-filter feature matrix from an MRI image.

    Each enabled image-type filter from pyradiomics is applied to the input
    image.  The resulting filter volumes are smoothed with a 3×3×1 in-plane
    average convolution, then stacked.  Only voxels inside the mask are
    included in the output matrix.

    Args:
        config_path: Path to pyradiomics YAML config.  Defaults to the bundled
            ``configs/pyradiomics_habitat.yaml``.
    """

    def __init__(self, config_path: Union[str, Path, None] = None):
        self.config_path = Path(config_path) if config_path else _DEFAULT_CONFIG
        self._extractor = None
        self.column_labels_: List[str] = []  # populated after each extract() call

    def _get_extractor(self):
        if self._extractor is None:
            from radiomics import featureextractor

            self._extractor = featureextractor.RadiomicsFeatureExtractor(str(self.config_path))
            # Disable all feature classes — we only need the filtered images
            self._extractor.disableAllFeatures()
        return self._extractor

    @property
    def enabled_image_types(self) -> List[str]:
        """List of image type names that will be processed."""
        ext = self._get_extractor()
        return list(ext.enabledImagetypes.keys())

    def extract(
        self,
        image: sitk.Image,
        mask: sitk.Image,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the pixelwise feature matrix for one image-mask pair.

        Args:
            image: Normalised SimpleITK image.
            mask: Binary (or single-class) SimpleITK mask with the same geometry.

        Returns:
            features: Float16 array of shape ``(N_voxels, N_filters)``.
            voxel_indices: Int array of shape ``(N_voxels, 3)`` giving the
                (z, y, x) index of each row in *features* within the original
                volume.
        """
        ext = self._get_extractor()

        mask_array = sitk.GetArrayFromImage(mask)  # (Z, Y, X)
        voxel_indices = np.argwhere(mask_array > 0)  # (N, 3) in (Z, Y, X) order

        if len(voxel_indices) == 0:
            raise ValueError("Mask contains no foreground voxels.")

        filter_vectors: List[np.ndarray] = []
        filter_names: List[str] = []

        for img_type in ext.enabledImagetypes.keys():
            filtered = _get_filtered_image(ext, image, mask, img_type)
            if filtered is None:
                logger.warning(f"Skipping image type '{img_type}' — filter returned None.")
                continue

            vol = sitk.GetArrayFromImage(filtered).astype(np.float32)  # (Z, Y, X)

            # 3×3×1 in-plane average convolution (axes: Z=0, Y=1, X=2)
            smoothed = uniform_filter(vol, size=_CONV_SIZE)

            # Extract only masked voxels
            vals = smoothed[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]]
            filter_vectors.append(vals.astype(np.float16))
            filter_names.append(img_type)

        if not filter_vectors:
            raise RuntimeError("No filter volumes were produced. Check pyradiomics config.")

        # Stack: each column is one filter → shape (N_voxels, N_filters)
        features = np.stack(filter_vectors, axis=1)  # (N, F)
        self.column_labels_ = filter_names  # record successful filter names for callers
        logger.debug(f"Feature matrix: {features.shape[0]} voxels × {features.shape[1]} filters ({', '.join(filter_names)})")
        return features, voxel_indices

    def extract_multi_sequence(
        self,
        sequences: Dict[str, sitk.Image],
        mask: sitk.Image,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Extract features from multiple MRI sequences and concatenate horizontally.

        For each sequence, :meth:`extract` is applied, producing one block of
        columns per sequence.  Column labels follow the format
        ``"{sequence_name}__{filter_name}"`` (double underscore separator).

        Args:
            sequences: Ordered mapping of ``{sequence_name: sitk.Image}``.
                All images must share the same geometry as *mask*.
            mask: Binary SimpleITK mask common to all sequences.

        Returns:
            features: Float16 array ``(N_voxels, N_seq × N_filters)``.
            voxel_indices: ``(N_voxels, 3)`` array of ``(z, y, x)`` coordinates.
            column_labels: List of ``"{seq}__{filter}"`` strings, one per column.
        """
        if not sequences:
            raise ValueError("sequences dict must not be empty.")

        all_blocks: List[np.ndarray] = []
        all_labels: List[str] = []
        voxel_indices: Optional[np.ndarray] = None

        for seq_name, image in sequences.items():
            feats, indices = self.extract(image, mask)
            if voxel_indices is None:
                voxel_indices = indices
            all_blocks.append(feats)
            for filter_name in self.column_labels_:
                all_labels.append(f"{seq_name}__{filter_name}")

        features = np.concatenate(all_blocks, axis=1)
        self.column_labels_ = all_labels
        logger.debug(
            f"Multi-sequence matrix: {features.shape[0]} voxels × {features.shape[1]} features "
            f"({len(sequences)} sequences × {features.shape[1] // len(sequences)} filters)"
        )
        return features, voxel_indices, all_labels

    def extract_from_files(
        self,
        image_path: Union[str, Path],
        mask_path: Union[str, Path],
    ) -> Tuple[np.ndarray, np.ndarray, sitk.Image]:
        """Convenience wrapper that reads files and calls :meth:`extract`.

        Returns:
            features: Float16 feature matrix ``(N_voxels, N_filters)``.
            voxel_indices: ``(N_voxels, 3)`` index array.
            image: The loaded SimpleITK image (for geometry reference).
        """
        image = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
        mask = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
        features, indices = self.extract(image, mask)
        return features, indices, image
