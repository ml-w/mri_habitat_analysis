"""
05 — Inference pipeline
========================

Demonstrates HabitatPipeline.infer() using the state archive produced by
example 04.  Sequence validation is performed before loading the model to
catch mismatches early.

For each input case the pipeline:
    normalise  →  extract features  →  predict labels  →  write NIfTI + PNG

Steps
-----
1. Generate a new synthetic test dataset with the same sequences as training.
2. Load the state archive and validate the provided sequences.
3. Run infer() and list the output files.
4. Demonstrate --override behaviour with a deliberately wrong sequence set.

CLI equivalent::

    # Normal inference (sequences validated)
    habitat-infer \\
        --seq T1:/data/test_T1 \\
        --seq T2:/data/test_T2 \\
        --mask_dir /data/test_masks \\
        --state state.zip \\
        --out results/

    # Skip sequence check (e.g. ablation study)
    habitat-infer \\
        --seq T1:/data/test_T1 \\
        --mask_dir /data/test_masks \\
        --state state.zip \\
        --out results/ \\
        --override
"""

import glob
import os
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from habitat_analysis.pipeline import HabitatPipeline
from habitat_analysis.state import HabitatState

# ── Step 1: Synthetic test set ────────────────────────────────────────────────
N_TEST = 3
SHAPE = (16, 48, 48)
SEQUENCES = ["T1", "T2"]   # must match what was used in example 04

workdir = Path(tempfile.mkdtemp(prefix="hab_infer_"))
test_seq_dirs = {seq: workdir / seq for seq in SEQUENCES}
test_mask_dir = workdir / "masks"
for d in [*test_seq_dirs.values(), test_mask_dir]:
    d.mkdir()

rng = np.random.default_rng(seed=7)
for i in range(N_TEST):
    vol_base = rng.random(SHAPE).astype(np.float32)
    z0, y0, x0 = [s // 2 for s in SHAPE]
    zz, yy, xx = np.ogrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]]
    sphere = (zz - z0) ** 2 + (yy - y0) ** 2 + (xx - x0) ** 2 < 15 ** 2

    img_ref = sitk.GetImageFromArray(vol_base)
    img_ref.SetSpacing((0.4492, 0.4492, 0.4492))

    cid = f"test{i+1:02d}"

    t1 = vol_base.copy() * 600
    t1[sphere] += 150 + rng.random() * 80
    t1_sitk = sitk.GetImageFromArray(t1)
    t1_sitk.CopyInformation(img_ref)
    sitk.WriteImage(t1_sitk, str(test_seq_dirs["T1"] / f"{cid}.nii.gz"))

    t2 = vol_base.copy() * 400
    t2[sphere] += 40 + rng.random() * 40
    t2_sitk = sitk.GetImageFromArray(t2)
    t2_sitk.CopyInformation(img_ref)
    sitk.WriteImage(t2_sitk, str(test_seq_dirs["T2"] / f"{cid}.nii.gz"))

    mask_sitk = sitk.GetImageFromArray(sphere.astype(np.uint8))
    mask_sitk.CopyInformation(img_ref)
    sitk.WriteImage(mask_sitk, str(test_mask_dir / f"{cid}.nii.gz"))

print(f"Test set             : {N_TEST} cases, sequences={SEQUENCES} → {workdir}")

# ── Step 2: Locate state archive from example 04 ─────────────────────────────
candidates = sorted(
    glob.glob("/tmp/hab_train_*/habitat_state.zip"),
    key=os.path.getmtime,
    reverse=True,
)
if not candidates:
    raise FileNotFoundError(
        "No state archive found. Run example 04 first, or set state_path manually."
    )
state_path = Path(candidates[0])
print(f"State archive        : {state_path}")

# ── Sequence validation ───────────────────────────────────────────────────────
# Load the state metadata (lightweight — no model objects deserialised yet).
loaded_state = HabitatState.load(state_path)
required = loaded_state.required_sequences
print(f"Required sequences   : {required}")

# Normal path — all sequences present.
try:
    warnings = loaded_state.validate_sequences(SEQUENCES)
    for w in warnings:
        print(f"WARNING: {w}")
    print("Sequence validation  : PASSED")
except ValueError as exc:
    raise SystemExit(f"Sequence mismatch: {exc}") from exc

# Demonstrate --override: pass only T1, which is missing T2.
print("\n--- Demonstrating --override (missing T2) ---")
try:
    loaded_state.validate_sequences(["T1"])   # raises because T2 is required
    print("Validation passed (unexpected)")
except ValueError as exc:
    print(f"Without --override   : {exc}")

# With override the caller simply skips validate_sequences entirely:
print("With --override      : validation skipped, proceeding anyway")

# ── Step 3: Run inference ─────────────────────────────────────────────────────
out_dir = workdir / "outputs"
pipeline = HabitatPipeline(random_state=42)

output_paths = pipeline.infer(
    img_dir=test_seq_dirs[SEQUENCES[0]],   # primary sequence
    mask_dir=test_mask_dir,
    state_path=state_path,
    out_dir=out_dir,
    visualize=True,
)

print(f"\nOutput files ({len(output_paths)}):")
for p in sorted(out_dir.iterdir()):
    print(f"  {p.name}")

# ── Step 4: Inspect one label map ─────────────────────────────────────────────
if output_paths:
    label_img = sitk.ReadImage(str(output_paths[0]))
    label_arr = sitk.GetArrayFromImage(label_img)
    unique, counts = np.unique(label_arr[label_arr > 0], return_counts=True)
    print(f"\nLabel map            : {output_paths[0].name}")
    print(f"  volume shape       : {label_arr.shape}")
    print("  habitat voxel counts:")
    for hab, cnt in zip(unique, counts):
        print(f"    habitat {hab:2d}  →  {cnt:,} voxels")
