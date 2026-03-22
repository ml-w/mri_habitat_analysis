"""
01 — Clustering basics
======================

Demonstrates HabitatClusterer with purely synthetic data.
No MRI files or external dependencies are required.

Steps
-----
1. Generate a synthetic pixelwise feature matrix.
2. Fit HabitatClusterer over a range of k values.
3. Inspect the composite evaluation metrics.
4. Predict cluster labels and remap them to a 3-D volume.
5. Save the fitted clusterer to disk and reload it.
"""

import numpy as np

from habitat_analysis.clusterer import HabitatClusterer

# ── Step 1: Synthetic feature matrix ─────────────────────────────────────────
# Simulate 8 000 voxels, each described by 6 filter signals (FP16).
# In practice this comes from PixelwiseFeatureExtractor — see example 02.

rng = np.random.default_rng(seed=0)

# Create three separable Gaussian clusters to make the sweep meaningful.
n_per_cluster = 2667
centres = np.array([[0, 0, 0, 0, 0, 0],
                    [5, 5, 5, 5, 5, 5],
                    [10, 0, 10, 0, 10, 0]], dtype=np.float32)
X = np.vstack([
    rng.normal(loc=c, scale=1.5, size=(n_per_cluster, 6)).astype(np.float16)
    for c in centres
])
print(f"Feature matrix shape : {X.shape}  dtype={X.dtype}")

# ── Step 2: Fit clusterer ────────────────────────────────────────────────────
clust = HabitatClusterer(
    method="kmeans",
    k_range=range(2, 6),
    random_state=42,
)
clust.fit(X)
print(f"\nBest k selected      : {clust.best_k}")

# ── Step 3: Evaluation metrics ───────────────────────────────────────────────
print("\n" + clust.metrics_summary())

# ── Step 4: Predict and remap to a 3-D volume ────────────────────────────────
# Suppose the 8 001 voxels came from a (30, 30, 10) volume.
# voxel_indices would be provided by PixelwiseFeatureExtractor.

vol_shape = (10, 30, 30)
n_voxels = X.shape[0]

# Fake voxel indices: spread voxels across the volume
all_coords = np.array(
    [(z, y, x)
     for z in range(vol_shape[0])
     for y in range(vol_shape[1])
     for x in range(vol_shape[2])],
    dtype=np.int32,
)[:n_voxels]

labels = clust.predict(X)  # shape (n_voxels,), values in [1, best_k]
print(f"\nLabel range          : {labels.min()} – {labels.max()}")

# Remap into a 3-D array (0 = background, 1..k = habitats)
label_vol = np.zeros(vol_shape, dtype=np.int32)
label_vol[all_coords[:, 0], all_coords[:, 1], all_coords[:, 2]] = labels
print(f"Label volume shape   : {label_vol.shape}")
print(f"Unique labels        : {np.unique(label_vol)}")

# ── Step 5: Save and reload ───────────────────────────────────────────────────
save_path = "/tmp/clusterer_example.joblib"
clust.save(save_path)

reloaded = HabitatClusterer.load(save_path)
assert (reloaded.predict(X) == labels).all(), "Reloaded model predictions differ!"
print(f"\nClusterer saved to   : {save_path}")
print("Round-trip check     : PASSED")
