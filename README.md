<div align="center">

# Habitat Analysis on MRI

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyRadiomics](https://img.shields.io/badge/PyRadiomics-enabled-orange.svg)](https://pyradiomics.readthedocs.io/)

</div>

---

## Introduction and motivations

This repository implements a **radiomics habitat analysis pipeline** for nasopharyngeal carcinoma (NPC) on MRI.

Habitat analysis is an emerging radiomics technique that subdivides a tumour ROI into spatially distinct sub-regions ("habitats") based on local imaging phenotype. The approach:

1. Computes a multi-filter feature vector per voxel (using pyradiomics image filters).
2. Clusters voxels by their feature vectors into *k* groups.
3. Treats each cluster as an independent sub-segmentation for downstream radiomics feature extraction.


## Quick start — CLI

**Install**

```bash
git clone https://github.com/ml-w/mri_habitat_analysis
cd mri_habitat_analysis
uv sync python=3.9
```

**Train** on a directory of NIfTI images and masks:

```
│   ├── train.py                    # habitat-train entry point
│   └── infer.py                    # habitat-infer entry point
├── examples/
│   ├── 01_clustering_basics.py
│   ├── 02_feature_extraction.py
│   ├── 03_normalization.py
│   ├── 04_full_training_pipeline.py
│   ├── 05_inference_pipeline.py
│   └── 06_visualization.py
└── tests/
    ├── test_clusterer.py
    ├── test_feature_extractor.py
    ├── test_state.py
    └── test_visualization.py
```

---

## Configuration

### `configs/normalization.yaml`

Defines the mnts `MNTSFilterGraph`:

```
SpatialNorm (resample to 0.4492 mm in-plane)
  └─► HuangThresholding (auto-mask)
        └─► N4ITKBiasFieldCorrection
              └─► NyulNormalizer  ← requires training & brain mask
```

See more information in the original repo [here](https://github.com/alabamagan/mri_normalization_tools).

### `configs/pyradiomics_habitat.yaml`

Enabled image types (Wavelets excluded to avoid feature inflation):

| Filter | Notes |
|--------|-------|
| `Original` | Raw normalised intensity |
| `LBP2D` | 2-D local binary pattern |
| `LBP3D` | 3-D local binary pattern |
| `LoG` | Laplacian of Gaussian (σ = 0.45, 1.0, 2.0 mm) |
| `Gradient` | Image gradient magnitude |
| `Exponential` | Exponential transform |

Each filter volume is smoothed with a **3×3×1** in-plane average convolution
before voxels are extracted, providing local spatial context while preserving
slice independence.

---

## Reproducibility

All hyperparameters and trained artefacts are bundled in the `.zip` state archive:

- Trained NyulNormalizer parameters
- PyRadiomics filter config (YAML)
- Fitted clusterer (joblib)
- Metadata JSON: best k, per-k evaluation scores, software versions

Loading the archive and running inference on the same data reproduces identical
cluster assignments.

---

## State serialisation and version compatibility

The trained clusterer (sklearn `KMeans` / `GaussianMixture` + `StandardScaler`) is
persisted with **joblib**, which uses Python's `pickle` protocol under the hood.
Pickled sklearn objects are **not guaranteed to load across package version
boundaries**.

### What is recorded

Every `.zip` state archive stores a `metadata.json` with the exact versions used
at training time:

```json
{
  "python": "3.9.18",
  "sklearn_version": "1.3.2",
  "numpy_version": "1.26.4",
  "pyradiomics_version": "3.0.1"
}
```

### Compatibility rules of thumb

| Scenario | Safe to load? |
|----------|---------------|
| Same environment, different data | ✅ Always |
| Patch version bump (e.g. sklearn 1.3.1 → 1.3.2) | ✅ Usually |
| Minor version bump (e.g. sklearn 1.2 → 1.3) | ⚠️ Check sklearn release notes |
| Major version bump (e.g. sklearn 0.x → 1.x) | ❌ Re-train required |
| Different Python minor version (e.g. 3.9 → 3.10) | ⚠️ Usually safe; verify |
| numpy < 2.0 → ≥ 2.0 | ❌ Re-train required |

### Recommended workflow

1. **Lock your environment.** `uv.lock` pins every transitive dependency.
   Reproduce the exact training environment with:
   ```bash
   uv sync --frozen --python 3.9
   ```

2. **Check metadata before inference.** When loading an archive in a new
   environment, compare `metadata.json` against your installed versions:
   ```python
   import sklearn, numpy as np
   state = HabitatState.load("model.zip")
   print(state.metadata)   # inspect sklearn_version, numpy_version
   print(sklearn.__version__, np.__version__)
   ```

3. **Upgrading dependencies.** If you must upgrade sklearn or numpy, re-run
   training from scratch and save a new archive. Do not attempt to load a
   state file trained on a different major version.

4. **Long-term archival.** For multi-year archival, consider committing the
   `uv.lock` file alongside the `.zip` state archive so the training
   environment can always be reconstructed exactly.
