"""Tests for HabitatClusterer."""

import numpy as np
import pytest

from habitat_analysis.clusterer import HabitatClusterer


def _make_feature_matrix(n=500, n_filters=6, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, n_filters)).astype(np.float16)


class TestHabitatClusterer:
    def test_fit_selects_valid_k(self):
        X = _make_feature_matrix()
        clust = HabitatClusterer(k_range=range(2, 5), random_state=0)
        clust.fit(X)
        assert clust.best_k in range(2, 5)
        assert clust._is_fitted

    def test_predict_label_range(self):
        X = _make_feature_matrix()
        clust = HabitatClusterer(k_range=range(2, 4), random_state=0)
        clust.fit(X)
        labels = clust.predict(X)
        assert labels.min() >= 1
        assert labels.max() <= clust.best_k
        assert labels.shape == (len(X),)

    def test_predict_before_fit_raises(self):
        X = _make_feature_matrix()
        clust = HabitatClusterer()
        with pytest.raises(RuntimeError, match="fit"):
            clust.predict(X)

    def test_save_load_roundtrip(self, tmp_path):
        X = _make_feature_matrix()
        clust = HabitatClusterer(k_range=range(2, 4), random_state=0)
        clust.fit(X)
        labels_before = clust.predict(X)

        path = tmp_path / "clusterer.joblib"
        clust.save(path)
        loaded = HabitatClusterer.load(path)

        labels_after = loaded.predict(X)
        np.testing.assert_array_equal(labels_before, labels_after)

    def test_metrics_summary_contains_all_k(self):
        X = _make_feature_matrix()
        clust = HabitatClusterer(k_range=range(2, 5), random_state=0)
        clust.fit(X)
        summary = clust.metrics_summary()
        for k in range(2, 5):
            assert str(k) in summary

    def test_gmm_method(self):
        X = _make_feature_matrix()
        clust = HabitatClusterer(method="gmm", k_range=range(2, 4), random_state=0)
        clust.fit(X)
        labels = clust.predict(X)
        assert 1 <= labels.min() and labels.max() <= clust.best_k
