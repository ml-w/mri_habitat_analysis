"""
03 — MRI normalisation
=======================

Demonstrates HabitatNormalizer: training the Nyul histogram normaliser on a
small dataset of synthetic NIfTI files, then applying the learned normalisation
to a new image.

Requires mnts (mri-normalization-tools) to be installed.

Steps
-----
1. Write synthetic NIfTI images and masks to a temporary directory.
2. Train the NyulNormalizer on those images.
3. Apply normalisation (inference) to the same images.
4. Inspect the output by reading one normalised image.
5. Normalise a single in-memory SimpleITK image without extra disk I/O.
"""

import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from habitat_analysis.normalizer import HabitatNormalizer

# ── Step 1: Write synthetic NIfTI dataset ────────────────────────────────────
# A real dataset would contain actual patient MRI scans.
# Here we create 5 random volumes to satisfy the NyulNormalizer's minimum.

N_CASES = 5
SHAPE = (20, 64, 64)

tmpdir = Path(tempfile.mkdtemp(prefix="hab_norm_example_"))
img_dir = tmpdir / "images"
mask_dir = tmpdir / "masks"
img_dir.mkdir()
mask_dir.mkdir()

rng = np.random.default_rng(seed=2)
for i in range(N_CASES):
    # Synthetic MRI: random intensity with a rough sphere of higher signal
    vol = rng.random(SHAPE).astype(np.float32) * 500
    z0, y0, x0 = [s // 2 for s in SHAPE]
    zz, yy, xx = np.ogrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]]
    sphere = (zz - z0) ** 2 + (yy - y0) ** 2 + (xx - x0) ** 2 < 18 ** 2
    vol[sphere] += 300

    img_sitk = sitk.GetImageFromArray(vol)
    img_sitk.SetSpacing((0.4492, 0.4492, 0.4492))

    mask_arr = sphere.astype(np.uint8)
    mask_sitk = sitk.GetImageFromArray(mask_arr)
    mask_sitk.CopyInformation(img_sitk)

    case_id = f"case{i+1:02d}"
    sitk.WriteImage(img_sitk, str(img_dir / f"{case_id}.nii.gz"))
    sitk.WriteImage(mask_sitk, str(mask_dir / f"{case_id}.nii.gz"))

print(f"Synthetic dataset    : {N_CASES} cases in {tmpdir}")

# ── Step 2: Train the normaliser ─────────────────────────────────────────────
norm_state_dir = tmpdir / "norm_state"
normalizer = HabitatNormalizer()        # uses bundled configs/normalization.yaml

normalizer.train(img_dir, mask_dir, norm_state_dir)
print(f"Normaliser state     : {norm_state_dir}")

# ── Step 3: Apply normalisation to all images ─────────────────────────────────
norm_out_dir = tmpdir / "normalised"
normalizer.infer(img_dir, norm_out_dir, norm_state_dir, mask_dir=mask_dir)
print(f"Normalised images    : {norm_out_dir}")

# ── Step 4: Inspect one normalised image ─────────────────────────────────────
norm_files = sorted(norm_out_dir.rglob("*.nii.gz"))
if norm_files:
    norm_img = sitk.ReadImage(str(norm_files[0]))
    norm_arr = sitk.GetArrayFromImage(norm_img)
    print(f"\nNormalised image     : {norm_files[0].name}")
    print(f"  shape              : {norm_arr.shape}")
    print(f"  intensity range    : [{norm_arr.min():.1f}, {norm_arr.max():.1f}]")
else:
    print("No normalised files found — check mnts installation.")

# ── Step 5: Normalise a single in-memory image ────────────────────────────────
raw_img = sitk.ReadImage(str(sorted(img_dir.glob("*.nii.gz"))[0]))
norm_single = normalizer.infer_single(raw_img, norm_state_dir)
norm_single_arr = sitk.GetArrayFromImage(norm_single)
print(f"\nIn-memory normalise  : output shape {norm_single_arr.shape}")
