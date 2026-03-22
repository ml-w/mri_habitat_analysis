"""
02 — Pixelwise feature extraction
==================================

Demonstrates PixelwiseFeatureExtractor using a synthetic SimpleITK image.
Requires pyradiomics to be installed.

Steps
-----
1. Create a synthetic 3-D MRI image and a spherical mask with SimpleITK.
2. Instantiate PixelwiseFeatureExtractor with the bundled config.
3. Inspect enabled image-type filters.
4. Extract the feature matrix and voxel indices.
5. Save the feature matrix to disk for later use.
"""

import numpy as np
import SimpleITK as sitk

from habitat_analysis.feature_extractor import PixelwiseFeatureExtractor

# ── Step 1: Synthetic image and mask ─────────────────────────────────────────
# Shape: (Z=20, Y=64, X=64) — small enough to run quickly.
shape = (20, 64, 64)
rng = np.random.default_rng(seed=1)

# Smooth random image mimicking MRI intensity texture
raw = rng.random(shape).astype(np.float32) * 1000
image_sitk = sitk.GetImageFromArray(raw)
image_sitk.SetSpacing((0.4492, 0.4492, 0.4492))   # mm, matching bundled config

# Spherical mask centred in the volume
z0, y0, x0 = np.array(shape) // 2
zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
sphere = ((zz - z0) ** 2 + (yy - y0) ** 2 + (xx - x0) ** 2) < 20 ** 2
mask_array = sphere.astype(np.uint8)
mask_sitk = sitk.GetImageFromArray(mask_array)
mask_sitk.CopyInformation(image_sitk)

n_foreground = int(mask_array.sum())
print(f"Image shape          : {shape}")
print(f"Foreground voxels    : {n_foreground}")

# ── Step 2: Instantiate extractor ────────────────────────────────────────────
# Uses configs/pyradiomics_habitat.yaml by default.
extractor = PixelwiseFeatureExtractor()

# ── Step 3: Inspect enabled filters ──────────────────────────────────────────
print(f"\nEnabled image types  : {extractor.enabled_image_types}")

# ── Step 4: Extract feature matrix ───────────────────────────────────────────
features, voxel_indices = extractor.extract(image_sitk, mask_sitk)

print(f"\nFeature matrix       : {features.shape}  dtype={features.dtype}")
print(f"  rows = voxels inside mask : {features.shape[0]} (expected ~{n_foreground})")
print(f"  cols = one per filter     : {features.shape[1]}")
print(f"\nVoxel index sample (z,y,x):")
for row in voxel_indices[:5]:
    print(f"  {row}")

# ── Step 5: Persist feature matrix ───────────────────────────────────────────
# Store as compressed NPZ for use in subsequent examples.
save_path = "/tmp/features_example.npz"
np.savez_compressed(save_path, features=features, voxel_indices=voxel_indices)
print(f"\nFeatures saved to    : {save_path}")
print("Load with            : np.load(save_path)['features']")
