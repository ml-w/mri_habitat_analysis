"""Tests for HabitatState save/load round-trip."""

import json
import numpy as np
import pytest

from habitat_analysis.clusterer import HabitatClusterer
from habitat_analysis.state import HabitatState


def _make_and_save_clusterer(tmp_path):
    X = np.random.default_rng(0).random((300, 4)).astype(np.float16)
    clust = HabitatClusterer(k_range=range(2, 4), random_state=0)
    clust.fit(X)
    p = tmp_path / "clusterer.joblib"
    clust.save(p)
    return p, clust


class TestHabitatState:
    def test_save_creates_zip(self, tmp_path):
        clusterer_path, _ = _make_and_save_clusterer(tmp_path)
        pyrad_config = tmp_path / "pyrad.yaml"
        pyrad_config.write_text("imageType:\n  Original: {}\n")
        norm_state = tmp_path / "norm_state"
        norm_state.mkdir()
        (norm_state / "dummy.json").write_text("{}")

        state = HabitatState(clusterer_path, pyrad_config, norm_state, metadata={"best_k": 3})
        archive = tmp_path / "state.zip"
        state.save(archive)
        assert archive.exists()

    def test_load_restores_metadata(self, tmp_path):
        clusterer_path, _ = _make_and_save_clusterer(tmp_path)
        pyrad_config = tmp_path / "pyrad.yaml"
        pyrad_config.write_text("imageType:\n  Original: {}\n")
        norm_state = tmp_path / "norm_state"
        norm_state.mkdir()

        meta = {"best_k": 3, "cluster_method": "kmeans"}
        state = HabitatState(clusterer_path, pyrad_config, norm_state, metadata=meta)
        archive = tmp_path / "state.zip"
        state.save(archive)

        loaded = HabitatState.load(archive, extract_dir=tmp_path / "extracted")
        assert loaded.metadata["best_k"] == 3
        assert loaded.metadata["cluster_method"] == "kmeans"

    def test_load_clusterer_predictions_match(self, tmp_path):
        clusterer_path, clust = _make_and_save_clusterer(tmp_path)
        pyrad_config = tmp_path / "pyrad.yaml"
        pyrad_config.write_text("imageType:\n  Original: {}\n")
        norm_state = tmp_path / "norm_state"
        norm_state.mkdir()

        state = HabitatState(clusterer_path, pyrad_config, norm_state)
        archive = tmp_path / "state.zip"
        state.save(archive)

        loaded_state = HabitatState.load(archive, extract_dir=tmp_path / "ext")
        loaded_clust = HabitatClusterer.load(loaded_state.clusterer_path)

        X = np.random.default_rng(1).random((100, 4)).astype(np.float16)
        np.testing.assert_array_equal(clust.predict(X), loaded_clust.predict(X))
