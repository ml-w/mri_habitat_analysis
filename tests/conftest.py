"""Shared test fixtures for habitat analysis tests."""

import numpy as np
import pytest
import SimpleITK as sitk


def _make_synthetic_image(shape=(20, 32, 32), seed=0) -> sitk.Image:
    """Create a synthetic float32 MRI-like volume."""
    rng = np.random.default_rng(seed)
    arr = rng.random(shape).astype(np.float32) * 1000
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing([0.45, 0.45, 3.0])
    return img


def _make_synthetic_mask(image: sitk.Image, fraction: float = 0.3) -> sitk.Image:
    """Create a binary mask covering the centre *fraction* of the FOV."""
    arr = sitk.GetArrayFromImage(image)
    mask_arr = np.zeros(arr.shape, dtype=np.uint8)
    z, y, x = arr.shape
    # Central cuboid
    z0, z1 = int(z * 0.3), int(z * 0.7)
    y0, y1 = int(y * 0.3), int(y * 0.7)
    x0, x1 = int(x * 0.3), int(x * 0.7)
    mask_arr[z0:z1, y0:y1, x0:x1] = 1
    mask = sitk.GetImageFromArray(mask_arr)
    mask.CopyInformation(image)
    return mask


@pytest.fixture
def synthetic_image():
    return _make_synthetic_image()


@pytest.fixture
def synthetic_mask(synthetic_image):
    return _make_synthetic_mask(synthetic_image)


@pytest.fixture
def synthetic_nifti_pair(tmp_path):
    """Write a synthetic image + mask pair to disk and return their paths."""
    img = _make_synthetic_image()
    msk = _make_synthetic_mask(img)
    img_path = tmp_path / "case001.nii.gz"
    msk_path = tmp_path / "case001.nii.gz"
    img_dir = tmp_path / "images"
    msk_dir = tmp_path / "masks"
    img_dir.mkdir()
    msk_dir.mkdir()
    sitk.WriteImage(img, str(img_dir / "case001.nii.gz"))
    sitk.WriteImage(msk, str(msk_dir / "case001.nii.gz"))
    return img_dir, msk_dir

