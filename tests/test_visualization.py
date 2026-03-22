"""Tests for visualization utilities."""

import numpy as np
import pytest
import SimpleITK as sitk

from habitat_analysis.visualization import label_map_to_nifti, render_habitat_overlay


class TestLabelMapToNifti:
    def test_output_file_created(self, tmp_path, synthetic_image):
        arr = np.zeros((20, 32, 32), dtype=np.int32)
        arr[8:14, 12:22, 12:22] = 1
        out = tmp_path / "labels.nii.gz"
        label_map_to_nifti(arr, synthetic_image, out)
        assert out.exists()

    def test_geometry_preserved(self, tmp_path, synthetic_image):
        arr = np.ones((20, 32, 32), dtype=np.int32)
        out = tmp_path / "labels.nii.gz"
        label_map_to_nifti(arr, synthetic_image, out)
        loaded = sitk.ReadImage(str(out))
        assert loaded.GetSize() == synthetic_image.GetSize()
        assert loaded.GetSpacing() == pytest.approx(synthetic_image.GetSpacing(), abs=1e-4)


class TestRenderHabitatOverlay:
    def test_png_created(self, tmp_path, synthetic_image, synthetic_mask):
        import SimpleITK as sitk

        img_arr = sitk.GetArrayFromImage(synthetic_image)
        mask_arr = sitk.GetArrayFromImage(synthetic_mask)
        lbl_arr = mask_arr.copy().astype(np.int32)  # label 1 where mask is 1

        out = tmp_path / "overlay.png"
        render_habitat_overlay(img_arr, lbl_arr, mask_arr, out)
        assert out.exists()
        assert out.stat().st_size > 0
