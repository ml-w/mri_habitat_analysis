"""Tests for PixelwiseFeatureExtractor."""

import numpy as np
import pytest
import SimpleITK as sitk

from habitat_analysis.feature_extractor import PixelwiseFeatureExtractor


class TestPixelwiseFeatureExtractor:
    def test_output_shapes(self, synthetic_image, synthetic_mask):
        """Feature matrix rows == number of masked voxels; columns == filters."""
        ext = PixelwiseFeatureExtractor()
        features, indices = ext.extract(synthetic_image, synthetic_mask)

        mask_arr = sitk.GetArrayFromImage(synthetic_mask)
        n_voxels = int((mask_arr > 0).sum())

        assert features.ndim == 2
        assert features.shape[0] == n_voxels, "Row count should equal masked voxel count."
        assert features.shape[1] >= 1, "At least one filter column expected."
        assert indices.shape == (n_voxels, 3), "Indices should be (N, 3)."

    def test_output_dtype(self, synthetic_image, synthetic_mask):
        """Feature matrix should be FP16."""
        ext = PixelwiseFeatureExtractor()
        features, _ = ext.extract(synthetic_image, synthetic_mask)
        assert features.dtype == np.float16

    def test_empty_mask_raises(self, synthetic_image):
        """An all-zero mask must raise ValueError."""
        arr = sitk.GetArrayFromImage(synthetic_image)
        empty_mask = sitk.GetImageFromArray(np.zeros(arr.shape, dtype=np.uint8))
        empty_mask.CopyInformation(synthetic_image)

        ext = PixelwiseFeatureExtractor()
        with pytest.raises(ValueError, match="no foreground"):
            ext.extract(synthetic_image, empty_mask)

    def test_enabled_image_types_excludes_wavelet(self):
        """The default config must not include Wavelet image types."""
        ext = PixelwiseFeatureExtractor()
        for name in ext.enabled_image_types:
            assert "wavelet" not in name.lower(), f"Wavelet type found: {name}"

