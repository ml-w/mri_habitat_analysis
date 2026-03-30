"""
Clustering module for habitat analysis.

Sweeps over a range of cluster counts (k), evaluates each with three criteria,
and selects the best k via one of two strategies:

- ``"composite"`` (default) — normalise Silhouette, Davies-Bouldin (inverted),
  and Calinski-Harabasz to [0, 1] and average.  May favour k = 2.
- ``"elbow"`` — use KMeans inertia (or GMM negative log-likelihood) and detect
  the knee/elbow point where adding more clusters gives diminishing returns.

Evaluation metrics (always computed for diagnostics):
    - Silhouette Score        (higher = better, range [-1, 1])
    - Davies-Bouldin Index    (lower = better, range [0, ∞))
    - Calinski-Harabasz Index (higher = better, range [0, ∞))
    - Inertia / NLL           (lower = better — used by elbow strategy)
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
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler
from scipy.stats import skew

from mnts.mnts_logger import MNTSLogger

logger: MNTSLogger = MNTSLogger[__name__]

_VALID_K_STRATEGIES = ("composite", "elbow")


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


def _find_elbow(k_values: List[int], costs: List[float]) -> int:
    """Detect the elbow/knee in a cost curve using the maximum-distance method.

    Draws a line from the first point to the last point on the (k, cost) curve
    and returns the k whose perpendicular distance to that line is greatest.
    This is the point of maximum curvature — the elbow.

    If the curve is perfectly linear (or has ≤ 2 points), returns the middle k.
    """
    n = len(k_values)
    if n <= 2:
        return k_values[n // 2]

    # Normalise both axes to [0, 1] so the distance isn't dominated by scale
    ks = np.array(k_values, dtype=float)
    cs = np.array(costs, dtype=float)

    k_norm = (ks - ks.min()) / max(ks.max() - ks.min(), 1e-12)
    c_norm = (cs - cs.min()) / max(cs.max() - cs.min(), 1e-12)

    # Line from first to last point
    p1 = np.array([k_norm[0], c_norm[0]])
    p2 = np.array([k_norm[-1], c_norm[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-12:
        return k_values[n // 2]

    # Perpendicular distance from each point to the line
    distances = np.abs(
        np.cross(line_vec, p1 - np.column_stack([k_norm, c_norm]))
    ) / line_len

    return k_values[int(np.argmax(distances))]


class HabitatClusterer:
    """Cluster pixelwise feature vectors into *k* habitat classes.

    Args:
        method: ``"kmeans"`` (default) or ``"gmm"``.
        k_range: Iterable of candidate cluster counts.  Default ``range(2, 7)``.
        k_selection: Strategy for choosing the best k.

            - ``"composite"`` — normalise Silhouette / DBI / CHI and average.
            - ``"elbow"`` — detect the elbow in the inertia (KMeans) or
              negative log-likelihood (GMM) curve.

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
        # k   silhouette  davies_bouldin  calinski_harabasz  composite  inertia
        # 2       0.1234          1.2345             456.7     0.6789    12345.6
        # ...

        labels = clust.predict(X)   # shape (10_000,), values in [1, best_k]

    Example — elbow method::

        clust = HabitatClusterer(k_range=range(2, 8), k_selection="elbow")
        clust.fit(X)
        print(f"Best k = {clust.best_k}")

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
        k_selection: str = "composite",
        kmeans_n_init: int = 10,
        random_state: int = 42,
        subsample: Optional[int] = None,
        visualize: bool = True,
    ):
        if method not in ("kmeans", "gmm"):
            raise ValueError(f"method must be 'kmeans' or 'gmm', got '{method}'")
        if k_selection not in _VALID_K_STRATEGIES:
            raise ValueError(
                f"k_selection must be one of {_VALID_K_STRATEGIES}, got '{k_selection}'"
            )
            
        # -- Attributes
        self.method = method
        self.k_range = list(k_range)
        self.k_selection = k_selection
        self.kmeans_n_init = kmeans_n_init
        self.random_state = random_state
        self.subsample = subsample
        self.visualize = visualize
        self._is_fitted = False

        # -- Trained parameters
        self.best_k: Optional[int] = None
        self.metrics: Dict[int, Dict[str, float]] = {}
        
        # -- Models
        self.best_model = None
        self.scaler = RobustScaler(unit_variance=True)
        self.selector_ = None           # fitted VarianceThreshold
        self.skew_mask_: Optional[np.ndarray] = None  # bool mask from skewness filter
        self.ttest_mask_: Optional[np.ndarray] = None  # bool mask from t-test filter
        self.feature_mask_: Optional[np.ndarray] = None  # combined bool mask (variance, skewness, t-test)
        self.label_order_: Optional[np.ndarray] = None   # deterministic label permutation (by centroid norm)

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
            
    def _fit_selector(self, X: np.ndarray, var_threshold: float = 1E-10, skew_threshold: float = 2.0, y_true: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit feature selection on training data and return the filtered array.

        Three filters are applied sequentially:
        1. **Variance**: remove near-constant features (VarianceThreshold).
        2. **Skewness**: remove features with |skewness| > *skew_threshold*,
           which tend to be degenerate or dominated by outliers.
        3. **T-test** (optional): when *y_true* is provided with two classes,
           remove features that do not show a significant difference between
           groups (Welch's t-test, p >= 0.05).

        The fitted masks are stored as instance attributes so :meth:`predict`
        and :meth:`_apply_selector` can apply the same selection at inference.

        Args:
            X: Feature matrix ``(N, F)``.
            var_threshold: Minimum variance for VarianceThreshold.
            skew_threshold: Maximum absolute skewness allowed.
            y_true: Optional binary label array ``(N,)`` for t-test filtering.

        Returns:
            Filtered ``X`` with only selected columns.
        """
        logger.info(f"Filtering features with parameters: {skew_threshold = } and {var_threshold = :}")
        n_before = X.shape[1]

        # Step 1: variance filter
        self.selector_ = VarianceThreshold()
        X_var = self.selector_.fit_transform(X)
        var_mask = self.selector_.get_support()
        logger.debug(f"{var_mask}")

        # Step 2: skewness filter (on variance-surviving columns)
        skewness = skew(X_var, axis=0)
        self.skew_mask_ = np.abs(skewness) <= skew_threshold
        X_out = X_var[:, self.skew_mask_]
        logger.debug(f"{skewness = }")

        # Step 3: t-test filter (on skewness-surviving columns)
        self.ttest_mask_: Optional[np.ndarray] = None
        if y_true is not None and len(np.unique(y_true)) > 1:
            import pingouin as pg
            import pandas as pd

            p_values = np.ones(X_out.shape[1])
            for i in range(X_out.shape[1]):
                group1 = X_out[y_true == 0, i]
                group2 = X_out[y_true == 1, i]
                # Skip if either group is empty or both are constant
                if len(group1) == 0 or len(group2) == 0:
                    continue
                if np.std(group1) == 0 and np.std(group2) == 0:
                    continue
                res = pg.ttest(pd.Series(group1), pd.Series(group2), correction=True)
                p_values[i] = res['p-val'].iloc[0]

            self.ttest_mask_ = p_values < 0.2
            n_removed = int(np.sum(~self.ttest_mask_))
            X_out = X_out[:, self.ttest_mask_]
            logger.debug(f"T-test p-values: {p_values}")
            logger.info(f"Applied t-test filter: removed {n_removed}/{len(p_values)} non-significant features")

        # Combined mask in original feature space
        self.feature_mask_ = var_mask.copy()
        self.feature_mask_[var_mask] &= self.skew_mask_
        if self.ttest_mask_ is not None:
            combined = self.feature_mask_.copy()
            combined[self.feature_mask_] &= self.ttest_mask_
            self.feature_mask_ = combined

        n_after = X_out.shape[1]
        n_var_removed = n_before - var_mask.sum()
        n_skew_removed = int((~self.skew_mask_).sum())
        n_ttest_removed = int((~self.ttest_mask_).sum()) if self.ttest_mask_ is not None else 0
        if n_after < n_before:
            logger.info("\n"
                f"Feature selection: {n_before} → {n_after} features "
                f"(variance: removed {n_var_removed}, "
                f"skewness: removed {n_skew_removed}, "
                f"t-test: removed {n_ttest_removed})"
            )
        else:
            logger.info("No feature columns were removed.")

        return X_out

    def _apply_selector(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted feature selection masks (variance, skewness, t-test) to new data."""
        if self.selector_ is None:
            return X
        X_var = self.selector_.transform(X)
        X_out = X_var[:, self.skew_mask_]
        if self.ttest_mask_ is not None:
            X_out = X_out[:, self.ttest_mask_]
        return X_out

    def _predict_labels(self, model, X: np.ndarray) -> np.ndarray:
        if self.method == "gmm":
            return model.predict(X)
        return model.labels_

    def fit(self, X: np.ndarray, y_true: Optional[np.ndarray] = None) -> "HabitatClusterer":
        """Fit clustering models over the k range and select the best k. Note that there's a
        basic feature selector that removes features with bad statistical properties. This
        selector requires a rigid input order of the input array `X`. For it to work, the input
        order must be identical during fit and prediction.

        Args:
            X: Feature matrix ``(N_voxels, N_filters)``.  FP16 is upcast
                internally.
            y_true: Optional binary label array ``(N_voxels,)`` for supervised
                feature selection via t-test.  When provided, features that do
                not show a significant difference between the two groups
                (p >= 0.05, Welch's t-test) are removed before clustering.

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
            y_fit = y_true[idx] if y_true is not None else None
            logger.info(f"Subsampling {self.subsample} / {len(X)} voxels for clustering fit.")
        else:
            X_fit = X
            y_fit = y_true

        # Clip top & bot 2.5% per feature during clustering training
        self.clip_lo_ = np.percentile(X_fit, 2.5, axis=0).astype(X_fit.dtype)
        self.clip_hi_ = np.percentile(X_fit, 97.5, axis=0).astype(X_fit.dtype)
        X_fit = np.clip(X_fit, self.clip_lo_, self.clip_hi_)

        # Select only features with desired properties
        X_fit = self._fit_selector(X_fit, y_true=y_fit)

        # Scale
        X_scaled = self.scaler.fit_transform(X_fit)

        silhouettes, dbi_scores, chi_scores, costs = [], [], [], []

        for k in self.k_range:
            model = self._make_model(k)
            if self.method == "gmm":
                model.fit(X_scaled)
                labels = model.predict(X_scaled)
                # Negative log-likelihood (lower = better fit)
                cost = -model.score(X_scaled) * len(X_scaled)
            else:
                model.fit(X_scaled)
                labels = model.labels_
                cost = model.inertia_

            sil = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)), random_state=self.random_state)
            dbi = davies_bouldin_score(X_scaled, labels)
            chi = calinski_harabasz_score(X_scaled, labels)

            self.metrics[k] = {
                "silhouette": sil,
                "davies_bouldin": dbi,
                "calinski_harabasz": chi,
                "inertia": cost,
            }
            silhouettes.append(sil)
            dbi_scores.append(dbi)
            chi_scores.append(chi)
            costs.append(cost)

            logger.info(f"k={k}  sil={sil:.4f}  dbi={dbi:.4f}  chi={chi:.1f}  inertia={cost:.1f}")

        # Composite score (always computed for diagnostics)
        norm_sil = _normalise_scores(silhouettes)
        norm_dbi = _normalise_scores(dbi_scores, invert=True)
        norm_chi = _normalise_scores(chi_scores)

        composite = [(s + d + c) / 3.0 for s, d, c in zip(norm_sil, norm_dbi, norm_chi)]
        for i, k in enumerate(self.k_range):
            self.metrics[k]["composite"] = composite[i]

        # Select best k according to the chosen strategy
        if self.k_selection == "elbow":
            self.best_k = _find_elbow(self.k_range, costs)
            logger.info(
                f"Best k={self.best_k} (elbow method on "
                f"{'inertia' if self.method == 'kmeans' else 'NLL'})"
            )
        else:
            best_idx = int(np.argmax(composite))
            self.best_k = self.k_range[best_idx]
            logger.info(f"Best k={self.best_k} (composite={composite[best_idx]:.4f})")

        # Refit best model on full (unsubsampled, clipped, selected) dataset
        X_full_clipped = np.clip(X.astype(np.float32), self.clip_lo_, self.clip_hi_)
        X_full_selected = self._apply_selector(X_full_clipped)
        X_full_scaled = self.scaler.transform(X_full_selected)
        self.best_model = self._make_model(self.best_k)
        if self.method == "gmm":
            self.best_model.fit(X_full_scaled)
        else:
            self.best_model.fit(X_full_scaled)

        # Build a deterministic label permutation so cluster IDs are stable
        # across runs.  We sort centroids by L2 norm (ascending), so cluster 1
        # is always the "smallest signal" centroid, cluster 2 the next, etc.
        self.label_order_ = self._compute_label_order()
        logger.info(f"Deterministic label order (by centroid norm): {self.label_order_.tolist()}")

        self._is_fitted = True
        return self

    def _compute_label_order(self) -> np.ndarray:
        """Return a permutation array that maps raw 0-based labels to
        deterministic 1-based labels, sorted by centroid L2 norm (ascending).

        ``label_order_[raw_label]`` gives the new 1-based label.
        """
        if self.method == "gmm":
            centroids = self.best_model.means_
        else:
            centroids = self.best_model.cluster_centers_
        norms = np.linalg.norm(centroids, axis=1)
        # argsort gives indices that would sort norms ascending
        rank = np.empty_like(norms, dtype=np.int32)
        rank[np.argsort(norms)] = np.arange(1, len(norms) + 1)
        return rank

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels (1-indexed) for feature matrix *X*.

        Labels are deterministically ordered by centroid L2 norm (ascending),
        so cluster 1 always corresponds to the centroid with the smallest norm
        regardless of random initialisation.

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
        X_clipped = np.clip(X.astype(np.float32), self.clip_lo_, self.clip_hi_)
        X_selected = self._apply_selector(X_clipped)
        X_scaled = self.scaler.transform(X_selected)
        if self.method == "gmm":
            raw_labels = self.best_model.predict(X_scaled)
        else:
            raw_labels = self.best_model.predict(X_scaled)
        return self.label_order_[raw_labels]

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
            "k_selection": self.k_selection,
            "best_k": self.best_k,
            "metrics": self.metrics,
            "scaler": self.scaler,
            "best_model": self.best_model,
            "kmeans_n_init": self.kmeans_n_init,
            "random_state": self.random_state,
            "selector": self.selector_,
            "skew_mask": self.skew_mask_,
            "ttest_mask": getattr(self, "ttest_mask_", None),
            "feature_mask": self.feature_mask_,
            "label_order": getattr(self, "label_order_", None),
            "clip_lo": getattr(self, "clip_lo_", None),
            "clip_hi": getattr(self, "clip_hi_", None),
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
            k_selection=payload.get("k_selection", "composite"),
            kmeans_n_init=payload["kmeans_n_init"],
            random_state=payload["random_state"],
        )
        obj.best_k = payload["best_k"]
        obj.metrics = payload["metrics"]
        obj.scaler = payload["scaler"]
        obj.best_model = payload["best_model"]
        obj.selector_ = payload.get("selector")
        obj.skew_mask_ = payload.get("skew_mask")
        obj.ttest_mask_ = payload.get("ttest_mask")
        obj.feature_mask_ = payload.get("feature_mask")
        obj.clip_lo_ = payload.get("clip_lo")
        obj.clip_hi_ = payload.get("clip_hi")
        # Restore or recompute label ordering for backward compat with old saves
        saved_order = payload.get("label_order")
        if saved_order is not None:
            obj.label_order_ = saved_order
        else:
            obj.label_order_ = obj._compute_label_order()
        obj._is_fitted = True
        return obj

    def metrics_summary(self) -> str:
        """Return a human-readable summary of per-k evaluation metrics.

        Example output::

            k   silhouette  davies_bouldin  calinski_harabasz     inertia  composite
            2       0.3412          1.1023             892.4     98765.4     0.5120
            3       0.4891          0.8754            1243.7     76543.2     0.7865 *
            4       0.4203          0.9901            1102.1     65432.1     0.6534

        The ``*`` marker indicates the selected k (strategy: elbow or composite).
        """
        header = (
            f"k   silhouette  davies_bouldin  calinski_harabasz"
            f"     inertia  composite  (strategy: {self.k_selection})"
        )
        lines = [header]
        for k in sorted(self.metrics):
            m = self.metrics[k]
            marker = " *" if k == self.best_k else ""
            lines.append(
                f"{k:<4}{m['silhouette']:>10.4f}  {m['davies_bouldin']:>14.4f}  "
                f"{m['calinski_harabasz']:>17.1f}  {m.get('inertia', float('nan')):>10.1f}"
                f"  {m.get('composite', float('nan')):>9.4f}{marker}"
            )
        return "\n".join(lines)

    def visualize_cluster_results(
        self,
        X_scaled: np.ndarray,
        out_path: Union[str, Path],
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Visualize clustering via PCA projection and save as PNG.

        Args:
            X_scaled: Scaled feature matrix used for fitting.
            out_path: Destination PNG path.
            feature_names: Optional list of feature/column names matching
                columns of *X_scaled*.  When provided, a companion CSV of
                PCA loadings is saved alongside the PNG.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before visualize().")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
            plt.style.use('ggplot')

            out_path = Path(out_path)
            labels = self.best_model.predict(X_scaled)

            pca = PCA(n_components=2, random_state=self.random_state)
            X_2d = pca.fit_transform(X_scaled)

            # -- PCA loadings record --
            evr = pca.explained_variance_ratio_
            ax_labels = [f"PC{i+1} ({evr[i]:.1%})" for i in range(len(evr))]

            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(pca.components_.shape[1])]

            csv_path = out_path.with_suffix(".csv")
            import csv
            with open(csv_path, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["feature"] + ax_labels)
                for j, name in enumerate(feature_names):
                    writer.writerow([name] + [f"{pca.components_[i, j]:.6f}" for i in range(len(evr))])
                writer.writerow(["explained_variance_ratio"] + [f"{v:.6f}" for v in evr])
            logger.info(f"PCA loadings saved to {csv_path}")

            # -- scatter plot --
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=5, alpha=0.2)
            ax.set_title(f"Habitat Clusters (k={self.best_k}, method={self.method})")
            ax.set_xlabel(ax_labels[0])
            ax.set_ylabel(ax_labels[1])
            ax.legend(*scatter.legend_elements(), title="Cluster")
            ax.grid(True)

            fig.tight_layout()
            fig.savefig(str(out_path), dpi=300)
            plt.close(fig)
            logger.info(f"Cluster visualization saved to {out_path}")

        except ImportError:
            logger.warning("matplotlib or sklearn not available; cannot visualize clusters.")