"""
04 — Full training pipeline
============================

Demonstrates HabitatPipeline.train() end-to-end with multi-sequence MRI input:

    normalise  →  extract features  →  cluster  →  save state

Two synthetic sequences (T1 and T2) are created per case.  The state archive
records the required sequence names so that inference can validate its inputs.

Steps
-----
1. Generate a synthetic multi-sequence NIfTI dataset (T1, T2, masks).
2. Instantiate HabitatPipeline with a narrow k range for speed.
3. Run train() passing sequence metadata via extra_metadata.
4. Inspect the state: best k, per-k metrics, and required sequences.

CLI equivalent::

    habitat-train \\
        --seq T1:/data/T1 \\
        --seq T2:/data/T2 \\
        --mask_dir /data/masks \\
        --out state.zip \\
        --method kmeans --k_min 2 --k_max 3
"""

import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from habitat_analysis.pipeline import HabitatPipeline

# ── Step 1: Synthetic multi-sequence dataset ──────────────────────────────────
N_CASES = 5
SHAPE = (16, 48, 48)
SEQUENCES = ["T1", "T2"]   # names that will be embedded in the state archive

workdir = Path(tempfile.mkdtemp(prefix="hab_train_"))
seq_dirs = {seq: workdir / seq for seq in SEQUENCES}
mask_dir = workdir / "masks"
for d in [*seq_dirs.values(), mask_dir]:
    d.mkdir()

rng = np.random.default_rng(seed=3)
for i in range(N_CASES):
    # Shared geometry
    vol_base = rng.random(SHAPE).astype(np.float32)
    z0, y0, x0 = [s // 2 for s in SHAPE]
    zz, yy, xx = np.ogrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]]
    sphere = (zz - z0) ** 2 + (yy - y0) ** 2 + (xx - x0) ** 2 < 15 ** 2

    img_ref = sitk.GetImageFromArray(vol_base)
    img_ref.SetSpacing((0.4492, 0.4492, 0.4492))

    cid = f"case{i+1:02d}"

    # T1 — brighter inside the sphere
    t1 = vol_base.copy() * 600
    t1[sphere] += 200 + rng.random() * 100
    t1_sitk = sitk.GetImageFromArray(t1)
    t1_sitk.CopyInformation(img_ref)
    sitk.WriteImage(t1_sitk, str(seq_dirs["T1"] / f"{cid}.nii.gz"))

    # T2 — different contrast (inverted relative pattern)
    t2 = vol_base.copy() * 400
    t2[sphere] += 50 + rng.random() * 50
    t2_sitk = sitk.GetImageFromArray(t2)
    t2_sitk.CopyInformation(img_ref)
    sitk.WriteImage(t2_sitk, str(seq_dirs["T2"] / f"{cid}.nii.gz"))

    # Shared mask
    mask_sitk = sitk.GetImageFromArray(sphere.astype(np.uint8))
    mask_sitk.CopyInformation(img_ref)
    sitk.WriteImage(mask_sitk, str(mask_dir / f"{cid}.nii.gz"))

print(f"Synthetic dataset    : {N_CASES} cases, sequences={SEQUENCES} → {workdir}")

# ── Step 2: Pipeline configuration ───────────────────────────────────────────
# k_range=range(2, 4) keeps the sweep fast for a quick demonstration.
# In production use a wider range such as range(2, 7).
pipeline = HabitatPipeline(
    cluster_method="kmeans",
    k_range=range(2, 4),
    subsample=100_000,
    random_state=42,
)

# ── Step 3: Train ─────────────────────────────────────────────────────────────
# extra_metadata embeds the sequence names in the state archive so that
# habitat-infer (and HabitatState.validate_sequences) can verify inputs.
#
# NOTE: The current pipeline processes the primary (first) sequence for
# normalisation and feature extraction.  Full multi-sequence feature
# concatenation is handled via HabitatIOManager.extract_multi_sequence().
state_path = workdir / "habitat_state.zip"
state = pipeline.train(
    img_dir=seq_dirs[SEQUENCES[0]],   # primary sequence for this run
    mask_dir=mask_dir,
    out_state=state_path,
    extra_metadata={"sequences": SEQUENCES},
)

print(f"\nState archive        : {state_path}")
print(f"Required sequences   : {state.required_sequences}")
print(f"Best k               : {state.metadata['best_k']}")

# ── Step 4: Inspect metrics ───────────────────────────────────────────────────
print("\nPer-k metrics (from state metadata):")
for k, m in state.metadata.get("metrics", {}).items():
    print(f"  k={k}  sil={m['silhouette']:.4f}  dbi={m['davies_bouldin']:.4f}"
          f"  chi={m['calinski_harabasz']:.1f}  composite={m.get('composite', float('nan')):.4f}")

print(f"\nPyradiomics version  : {state.metadata.get('pyradiomics_version', 'unknown')}")
print(f"sklearn version      : {state.metadata.get('sklearn_version', 'unknown')}")
print(f"Python version       : {state.metadata.get('python', 'unknown')}")
print("\nState archive is self-contained — pass it to example 05.")
