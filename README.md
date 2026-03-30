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

This repository implements a **radiomics habitat analysis pipeline**, primarily for nasopharyngeal carcinoma (NPC) on MRI, but can also applies to other modalities.

Habitat analysis is an emerging radiomics technique that subdivides a tumour ROI into spatially distinct sub-regions ("habitats") based on local imaging phenotype. The approach we adopt in this repo:

1. Computes a multi-filter feature vector per voxel (using pyradiomics image filters).
2. Clusters voxels by their feature vectors into *k* groups.
3. Treats each cluster as an independent sub-segmentation for downstream task including classification.

We see in some manuscript that the subregions are only clused by simple imaging properties (e.g., T1w/T2w plain signals) then further used for radiomics features extraction with a regular pipeline. We question this methodology here as from our experiences the clustered subregion is extremely fragmented such that the validity of the second order features become questionable (e.g., GLCM, GLRLM). Therefore, we decided to apply the radiomics imaging filter first for the clustering of habtitats, then go on to perform classification with first order features of these subregions.


## Quick start — CLI

### **Install**

```bash
git clone https://github.com/ml-w/mri_habitat_analysis
cd mri_habitat_analysis
uv sync python=3.9
```

### **Train** on a directory of NIfTI images and masks:

```bash
│   ├── train.py                    # habitat-train entry point
│   └── infer.py                    # habitat-infer entry point
├── Inputs/
│   ├── T1W_TRA/UID123_*.nii.gz     # Images are paired with a unique ID
│   └── T2W_TRA/UID123_*.nii.gz
├── Segmentations/
│   └── Shared/UID123_*.nii.gz      # Only one set of segmentations 
└── Configs/
    ├── norm_graph.yaml             # TODO - Do not use
    └── pyradiomics.yaml            # Only imaging filter and preprocessing settings are respected
```

Once you've got the directory readied, you can run the training script directly through the entry point `habitat-train`:

```bash
habitat-train 
   --seq T1:inputs/T1W_TRA                   # There can be multiple sequences as inputs
   --seq T2:inputs/T2W_TRA                   # There can be multiple sequences as inputs
   --id-globber "UID\d+"                     # Regex for globbing the ID of the case from the file name. 
   --out output                              # Where the results will be stored
   --method kmeans                           # The method for clustering habitats
   --subsample 400000                        # This will perfrom subsampling as there can be many voxels in a dataset
   --skip-norm                               # normalization is not yet readied as part of the pipeline
   --seg-dir Segmentations/Shared            # Directory holding the segmetnation .nii.gz files
   --pyrad-config Configs/pyradiomics.yaml   # Setting for pyradiomics. Only those configurations for imaging filters are in effect
   --verbose                                 # Enables DEBUG logging
   -n                                        # Multithread for feature extraction step (doesn't affect the training step)
```

All outputs will be written to the designated output folder. If you add .zip to the suffix, it will be zipped as well. Otherwise, the output folder structure is typically this:

```bash
output/
├── clustered_labels/               # Per-case habitat segmentation NIfTI files
│   ├── {case_id}_habitat.nii.gz    # Label map where voxel values = cluster assignment (1..k)
│   └── ...
├── normaliser_state/               # Trained normalisation parameters (per sequence)
│   ├── T1/
│   └── T2/
├── clusterer.joblib                # Fitted clustering pipeline (scaler + KMeans/GMM)
├── features.parquet                # Extracted per-voxel features with case_id, coordinates, and cluster labels
├── metadata.json                   # Best k, per-k evaluation metrics, software versions, feature columns
└── pyradiomics_config.yaml         # Copy of the pyradiomics filter config used during training
```

Next step, you can either use the output sub-regioned segmentation (`clustered_labels`) or the already extracted and tabulated features from radiomics imaging filters (`features.parquet`) for downstream classification tasks. 

For sub-regioning, the assigment of cluster class is deterministics, with the cluster code as integer classes based on the position of their centriod. So unless the fit is really bad with too few data, the output multi-class segmentation should be more or less consistent. 

For `features.parquet`, each row represents one voxel, and they are associated with a case-level IDs (globbed by the provided regex). This enables you to do some voxel level analysis as the filtered images are not output for the sake of space management. 

<details>
<summary><b>Script usages</b></summary>

```bash
Usage: habitat-train [OPTIONS]

  Train the habitat analysis pipeline and save the state archive.

Options:
  --seq NAME:DIR                  Sequence name:directory pair (repeatable).
                                  E.g. --seq T1:/data/T1 --seq T2:/data/T2
  --img-dir DIRECTORY             Single-sequence image directory (alias for
                                  --seq image:DIR).
  --mask-dir DIRECTORY            Directory of binary mask NIfTI files.
                                  [required]
  --out PATH                      Output state path — a .zip archive if the
                                  suffix is .zip, otherwise a plain directory.
                                  [required]
  --norm-config FILE              mnts normalisation YAML (default: bundled).
  --pyrad-config FILE             PyRadiomics filter YAML (default: bundled).
  --method [kmeans|gmm]           Clustering algorithm.
  --k-min INTEGER                 Minimum number of clusters to evaluate.
                                  [default: 2]
  --k-max INTEGER                 Maximum number of clusters to evaluate
                                  (inclusive).  [default: 6]
  --k-selection [elbow|composite]
                                  Strategy for selecting best k: 'elbow'
                                  (recommended) or 'composite'.  [default:
                                  elbow]
  --subsample INTEGER             Max voxels for clustering fit (0 = use all).
                                  [default: 200000]
  --seed INTEGER                  Random seed.  [default: 42]
  --id-globber TEXT               Regex to extract case IDs from filenames.
                                  [default: ^[0-9a-zA-Z]+]
  --skip-norm                     Skip normalisation (use when images are
                                  already normalised).
  --seg-dir PATH                  Directory for output habitat segmentations
                                  (default: <out>/clustered_labels/).
  --no-vis                        Disable cluster PCA visualization PNG.
  -n, --workers INTEGER           Number of parallel workers for feature
                                  extraction.  [default: 1]
  --y-true FILE                   CSV or XLSX file for supervised feature
                                  selection (t-test). First column = case ID
                                  index, second column = binary label (0/1).
  --force-extract                 Force feature re-extraction even if cached
                                  features.parquet exists.
  --debug                         Debug mode: process only the first 3 cases.
  -v, --verbose                   Enable DEBUG logging.
  --help                          Show this message and exit.
```

</details>

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

Note that some of the filters might require and additional training step that is not supported by the current trianing CLI (e.g., Nyul normalization). See more information in the original repo [here](https://github.com/alabamagan/mri_normalization_tools).

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

# Inference

All hyperparameters and trained artefacts are bundled in the output folder created by the training script:

- TODO: Trained NyulNormalizer parameters 
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
