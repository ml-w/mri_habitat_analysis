"""
Clustering module for habitat analysis.

Sweeps over a range of cluster counts (k), evaluates each with three criteria,
and selects the best k by a composite score.  Supports K-Means (default) and
Gaussian Mixture Model.

Evaluation metrics:
    - Silhouette Score      (higher = better, range [-1, 1])
    - Davies-Bouldin Index  (lower = better, range [0, ∞))
    - Calinski-Harabasz Index (higher = better, range [0, ∞))

Composite score: mean of the three normalised scores (all oriented so higher
is better after inversion of Davies-Bouldin).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from mnts.mnts_logger import MNTSLogger

logger = MNTSLogger[__name__]


def _normalise_scores(values: List[float], invert: bool = False) -> List[float]:
    """Min-max normalise a list; optionally invert (1 - x)."""
    arr = np.array(values, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        out = np.zeros_like(arr)
    else:
        out = (arr - mn) / (mx - mn)
    if invert:
        out = 1.0 - out
    return out.tolist()


class HabitatClusterer:
    """Cluster pixelwise feature vectors into *k* habitat classes.

    Args:
        method: ``"kmeans"`` (default) or ``"gmm"``.
        k_range: Iterable of candidate cluster counts.  Default ``range(2, 7)``.
        kmeans_n_init: Number of K-Means initialisations (default 10).
        random_state: Random seed for reproducibility.
        subsample: If set, randomly subsample this many voxels before fitting
            to keep memory/time manageable for large datasets.

    Example — basic fit/predict::

        import numpy as np
        from habitat_analysis.clusterer import HabitatClusterer

        # Simulate a pixelwise feature matrix: 10 000 voxels × 6 filters
        rng = np.random.default_rng(0)
        X = rng.random((10_000, 6)).astype(np.float16)

        clust = HabitatClusterer(k_range=range(2, 6), random_state=0)
        clust.fit(X)

        print(clust.metrics_summary())
        # k   silhouette  davies_bouldin  calinski_harabasz  composite
        # 2       0.1234          1.2345             456.7     0.6789 *
        # ...

        labels = clust.predict(X)   # shape (10_000,), values in [1, best_k]

    Example — save and reload::

        clust.save("clusterer.joblib")
        reloaded = HabitatClusterer.load("clusterer.joblib")
        assert (reloaded.predict(X) == labels).all()

    Example — GMM with larger k sweep and subsampling::

        clust = HabitatClusterer(
            method="gmm",
            k_range=range(2, 9),
            subsample=50_000,
            random_state=42,
        )
        clust.fit(X)
        print(f"Best k = {clust.best_k}")
    """

    def __init__(
        self,
        method: str = "kmeans",
        k_range=range(2, 7),
        kmeans_n_init: int = 10,
        random_state: int = 42,
        subsample: Optional[int] = None,
    ):
        if method not in ("kmeans", "gmm"):
            raise ValueError(f"method must be 'kmeans' or 'gmm', got '{method}'")
        self.method = method
        self.k_range = list(k_range)
        self.kmeans_n_init = kmeans_n_init
        self.random_state = random_state
        self.subsample = subsample

        self.best_k: Optional[int] = None
        self.best_model = None
        self.scaler = StandardScaler()
        self.metrics: Dict[int, Dict[str, float]] = {}
        self._is_fitted = False

    def _make_model(self, k: int):
        if self.method == "kmeans":
            return KMeans(
                n_clusters=k,
                n_init=self.kmeans_n_init,
                random_state=self.random_state,
            )
        else:
            return GaussianMixture(
                n_components=k,
                random_state=self.random_state,
            )

    def _predict_labels(self, model, X: np.ndarray) -> np.ndarray:
        if self.method == "gmm":
            return model.predict(X)
        return model.labels_

    def fit(self, X: np.ndarray) -> "HabitatClusterer":
        """Fit clustering models over the k range and select the best k.

        Args:
            X: Feature matrix ``(N_voxels, N_filters)``.  FP16 is upcast
                internally.

        Returns:
            self

        Example::

            clust = HabitatClusterer(k_range=range(2, 5))
            clust.fit(X)   # X shape (N, F)
            print(clust.best_k)
        """
        X = X.astype(np.float32)

        rng = np.random.default_rng(self.random_state)
        if self.subsample and len(X) > self.subsample:
            idx = rng.choice(len(X), size=self.subsample, replace=False)
            X_fit = X[idx]
            logger.info(f"Subsampling {self.subsample} / {len(X)} voxels for clustering fit.")
        else:
            X_fit = X

        X_scaled = self.scaler.fit_transform(X_fit)

        silhouettes, dbi_scores, chi_scores = [], [], []

        for k in self.k_range:
            model = self._make_model(k)
            if self.method == "gmm":
                model.fit(X_scaled)
                labels = model.predict(X_scaled)
            else:
                model.fit(X_scaled)
                labels = model.labels_

            sil = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)), random_state=self.random_state)
            dbi = davies_bouldin_score(X_scaled, labels)
            chi = calinski_harabasz_score(X_scaled, labels)

            self.metrics[k] = {"silhouette": sil, "davies_bouldin": dbi, "calinski_harabasz": chi}
            silhouettes.append(sil)
            dbi_scores.append(dbi)
            chi_scores.append(chi)

            logger.info(f"k={k}  sil={sil:.4f}  dbi={dbi:.4f}  chi={chi:.1f}")

        # Composite score: normalise each metric (DBI inverted) and average
        norm_sil = _normalise_scores(silhouettes)
        norm_dbi = _normalise_scores(dbi_scores, invert=True)
        norm_chi = _normalise_scores(chi_scores)

        composite = [(s + d + c) / 3.0 for s, d, c in zip(norm_sil, norm_dbi, norm_chi)]
        best_idx = int(np.argmax(composite))
        self.best_k = self.k_range[best_idx]

        for i, k in enumerate(self.k_range):
            self.metrics[k]["composite"] = composite[i]

        logger.info(f"Best k={self.best_k} (composite={composite[best_idx]:.4f})")

        # Refit best model on full (unsubsampled) dataset
        X_full_scaled = self.scaler.transform(X.astype(np.float32))
        self.best_model = self._make_model(self.best_k)
        if self.method == "gmm":
            self.best_model.fit(X_full_scaled)
        else:
            self.best_model.fit(X_full_scaled)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels (1-indexed) for feature matrix *X*.

        Args:
            X: Feature matrix ``(N_voxels, N_filters)``, same order of filters
               as used during :meth:`fit`.

        Returns:
            Integer array of shape ``(N_voxels,)`` with values in
            ``[1, best_k]``.

        Example::

            labels = clust.predict(X)
            # Remap back to a 3-D volume
            label_vol = np.zeros(mask_array.shape, dtype=np.int32)
            label_vol[voxel_indices[:, 0],
                      voxel_indices[:, 1],
                      voxel_indices[:, 2]] = labels
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")
        X_scaled = self.scaler.transform(X.astype(np.float32))
        if self.method == "gmm":
            labels = self.best_model.predict(X_scaled)
        else:
            labels = self.best_model.predict(X_scaled)
        return (labels + 1).astype(np.int32)  # shift to 1-based

    def save(self, path: Union[str, Path]) -> None:
        """Save the fitted clusterer (model + scaler + metadata) to *path*.

        Args:
            path: File path with a ``.joblib`` extension.

        Example::

            clust.save("my_clusterer.joblib")
            reloaded = HabitatClusterer.load("my_clusterer.joblib")
        """
        path = Path(path)
        payload = {
            "method": self.method,
            "k_range": self.k_range,
            "best_k": self.best_k,
            "metrics": self.metrics,
            "scaler": self.scaler,
            "best_model": self.best_model,
            "kmeans_n_init": self.kmeans_n_init,
            "random_state": self.random_state,
        }
        joblib.dump(payload, path)
        logger.info(f"Clusterer saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "HabitatClusterer":
        """Load a previously saved clusterer.

        Args:
            path: Path written by :meth:`save`.

        Returns:
            Restored :class:`HabitatClusterer` instance.
        """
        payload = joblib.load(path)
        obj = cls(
            method=payload["method"],
            k_range=payload["k_range"],
            kmeans_n_init=payload["kmeans_n_init"],
            random_state=payload["random_state"],
        )
        obj.best_k = payload["best_k"]
        obj.metrics = payload["metrics"]
        obj.scaler = payload["scaler"]
        obj.best_model = payload["best_model"]
        obj._is_fitted = True
        return obj

    def metrics_summary(self) -> str:
        """Return a human-readable summary of per-k evaluation metrics.

        Example output::

            k   silhouette  davies_bouldin  calinski_harabasz  composite
            2       0.3412          1.1023             892.4     0.5120
            3       0.4891          0.8754            1243.7     0.7865 *
            4       0.4203          0.9901            1102.1     0.6534
        """
        lines = ["k   silhouette  davies_bouldin  calinski_harabasz  composite"]
        for k in sorted(self.metrics):
            m = self.metrics[k]
            marker = " *" if k == self.best_k else ""
            lines.append(
                f"{k:<4}{m['silhouette']:>10.4f}  {m['davies_bouldin']:>14.4f}  "
                f"{m['calinski_harabasz']:>17.1f}  {m.get('composite', float('nan')):>9.4f}{marker}"
            )
        return "\n".join(lines)

    def visualize_cluster_results(self) -> None:
        """Visualize the cluster results with a dimensionality reduction plot.
        
        This function uses PCA to reduce the feature space to 2D and colors the points by the 
        cluster labels.
        """
        
        if not self._is_fitted:
            raise RuntimeError("Call fit() before visualize().")

        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA

            X_scaled = self.scaler.transform(self.best_model._X)  # type: ignore[attr-defined]
            labels = self._predict_labels(self.best_model, X_scaled)

            pca = PCA(n_components=2, random_state=self.random_state)
            X_2d = pca.fit_transform(X_scaled)

            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=5, alpha=0.7)
            plt.title(f"Habitat Clusters (k={self.best_k})")
            plt.xlabel("PCA Component 1")
            plt.ylabel("PCA Component 2")
            plt.legend(*scatter.legend_elements(), title="Cluster")
            plt.grid(True)
            plt.show()

        except ImportError:
            logger.warning("matplotlib or sklearn not available; cannot visualize clusters.")