"""
06 — Visualisation
===================

Demonstrates the two visualisation utilities:

    label_map_to_nifti      — write a label volume as a NIfTI file
    render_habitat_overlay  — colour-coded PNG overlay on the axial MRI slice
                              with the largest mask area

No external MRI data is needed; a synthetic image and label map are generated.

Steps
-----
1. Create a synthetic 3-D image, mask, and multi-cluster label volume.
2. Write the label volume as a NIfTI file.
3. Render a colour-coded overlay PNG and save it.
4. Preview the overlay dimensions (and open it with an image viewer if desired).
"""

from pathlib import Path
import tempfile

import numpy as np
import SimpleITK as sitk

from habitat_analysis.visualization import label_map_to_nifti, render_habitat_overlay

# ── Step 1: Synthetic volumes ─────────────────────────────────────────────────
SHAPE = (20, 128, 128)
rng = np.random.default_rng(seed=5)

# MRI image: smooth random field
image_arr = rng.random(SHAPE).astype(np.float32) * 1000

image_sitk = sitk.GetImageFromArray(image_arr)
image_sitk.SetSpacing((0.4492, 0.4492, 0.4492))

# Spherical mask
z0, y0, x0 = [s // 2 for s in SHAPE]
zz, yy, xx = np.ogrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]]
dist2 = (zz - z0) ** 2 + (yy - y0) ** 2 + (xx - x0) ** 2
mask_arr = (dist2 < 35 ** 2).astype(np.uint8)

# Label map with 4 concentric shells simulating four habitats
label_arr = np.zeros(SHAPE, dtype=np.int32)
label_arr[dist2 < 10 ** 2] = 1
label_arr[(dist2 >= 10 ** 2) & (dist2 < 20 ** 2)] = 2
label_arr[(dist2 >= 20 ** 2) & (dist2 < 28 ** 2)] = 3
label_arr[(dist2 >= 28 ** 2) & (dist2 < 35 ** 2)] = 4

print(f"Image shape          : {SHAPE}")
n_foreground = int(mask_arr.sum())
print(f"Foreground voxels    : {n_foreground:,}")
for hab in range(1, 5):
    print(f"  habitat {hab}: {int((label_arr == hab).sum()):,} voxels")

# ── Step 2: Write NIfTI label map ────────────────────────────────────────────
outdir = Path(tempfile.mkdtemp(prefix="hab_viz_"))
nifti_path = outdir / "label_map.nii.gz"

label_map_to_nifti(label_arr, image_sitk, nifti_path)
print(f"\nNIfTI label map      : {nifti_path}")

# Round-trip check: read back and confirm values
rt = sitk.GetArrayFromImage(sitk.ReadImage(str(nifti_path)))
assert np.array_equal(rt, label_arr), "NIfTI round-trip mismatch!"
print("NIfTI round-trip     : PASSED")

# ── Step 3: Render PNG overlay ────────────────────────────────────────────────
png_path = outdir / "habitat_overlay.png"

render_habitat_overlay(
    image_array=image_arr,
    label_array=label_arr,
    mask_array=mask_arr,
    out_path=png_path,
    alpha=0.45,         # habitat colour opacity
    cmap_name="tab10",  # any matplotlib colormap name
)
print(f"PNG overlay          : {png_path}")

# ── Step 4: Verify PNG dimensions ─────────────────────────────────────────────
try:
    from PIL import Image
    img = Image.open(png_path)
    print(f"  size: {img.size}  mode: {img.mode}")
except ImportError:
    print("  (install Pillow to inspect PNG dimensions)")

print(f"\nAll outputs in       : {outdir}")
