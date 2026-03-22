"""Tests for HabitatIOManager provenance tracking."""

import numpy as np
import pytest

from habitat_analysis.io_manager import HabitatIOManager, VoxelRecord


def _make_indices(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 20, size=(n, 3)).astype(int)


def _labels(seq: str, filters=("Original", "LBP2D", "LoG", "Gradient", "Exponential")) -> list:
    return [f"{seq}__{f}" for f in filters]


class FakePipeline:
    id_globber = r"case\d+"


def _mgr(tmp_path, sequences=("T1",)):
    return HabitatIOManager(
        pipeline=FakePipeline(),
        sequence_paths={s: tmp_path / s for s in sequences},
        segmentation_path=tmp_path / "masks",
    )


class TestVoxelRecord:
    def test_n_voxels(self):
        idx = _make_indices(50)
        rec = VoxelRecord("case01", {"T1": None}, None, idx, row_start=10, row_end=60)
        assert rec.n_voxels == 50

    def test_coord_at(self):
        idx = np.array([[2, 3, 4], [5, 6, 7]])
        rec = VoxelRecord("case01", {"T1": None}, None, idx, row_start=0, row_end=2)
        assert rec.coord_at(0) == (2, 3, 4)
        assert rec.coord_at(1) == (5, 6, 7)


class TestHabitatIOManagerInit:
    def test_dict_sequence_paths(self, tmp_path):
        mgr = HabitatIOManager(
            pipeline=FakePipeline(),
            sequence_paths={"T1": tmp_path / "T1", "T2": tmp_path / "T2"},
            segmentation_path=tmp_path / "masks",
        )
        assert mgr.sequence_names == ["T1", "T2"]

    def test_list_sequence_paths_auto_named(self, tmp_path):
        mgr = HabitatIOManager(
            pipeline=FakePipeline(),
            sequence_paths=[tmp_path / "a", tmp_path / "b"],
            segmentation_path=tmp_path / "masks",
        )
        assert mgr.sequence_names == ["seq_0", "seq_1"]


class TestHabitatIOManagerContext:
    def test_context_clears_state(self, tmp_path):
        mgr = _mgr(tmp_path)
        p = tmp_path / "case01.nii.gz"; p.touch()
        mgr.register({"T1": p}, tmp_path / "case01_mask.nii.gz", _make_indices(10))
        assert mgr.total_voxels == 10

        with mgr:
            assert mgr.total_voxels == 0
            assert mgr.feature_columns == []


class TestRegister:
    def test_accumulates_rows(self, tmp_path):
        mgr = _mgr(tmp_path)
        for i, n in enumerate([30, 50, 20]):
            p = tmp_path / f"case0{i}.nii.gz"; p.touch()
            mgr.register({"T1": p}, tmp_path / "mask.nii.gz", _make_indices(n, seed=i))
        assert mgr.total_voxels == 100

    def test_row_ranges_are_contiguous(self, tmp_path):
        mgr = _mgr(tmp_path)
        p0 = tmp_path / "case00.nii.gz"; p0.touch()
        p1 = tmp_path / "case01.nii.gz"; p1.touch()
        r0 = mgr.register({"T1": p0}, tmp_path / "m.nii.gz", _make_indices(40))
        r1 = mgr.register({"T1": p1}, tmp_path / "m.nii.gz", _make_indices(25))
        assert r0.row_start == 0 and r0.row_end == 40
        assert r1.row_start == 40 and r1.row_end == 65

    def test_column_labels_stored_on_first_registration(self, tmp_path):
        mgr = _mgr(tmp_path, sequences=("T1", "T2"))
        p = tmp_path / "case00.nii.gz"; p.touch()
        cols = _labels("T1") + _labels("T2")
        mgr.register({"T1": p, "T2": p}, tmp_path / "m.nii.gz", _make_indices(10), column_labels=cols)
        assert mgr.feature_columns == cols

    def test_column_labels_not_overwritten_on_subsequent_registrations(self, tmp_path):
        mgr = _mgr(tmp_path)
        p0 = tmp_path / "case00.nii.gz"; p0.touch()
        p1 = tmp_path / "case01.nii.gz"; p1.touch()
        first_cols = _labels("T1")
        mgr.register({"T1": p0}, tmp_path / "m.nii.gz", _make_indices(5), column_labels=first_cols)
        mgr.register({"T1": p1}, tmp_path / "m.nii.gz", _make_indices(5), column_labels=_labels("other"))
        assert mgr.feature_columns == first_cols  # unchanged

    def test_sequence_paths_stored_in_record(self, tmp_path):
        mgr = _mgr(tmp_path, sequences=("T1", "T2"))
        p_t1 = tmp_path / "case01_T1.nii.gz"; p_t1.touch()
        p_t2 = tmp_path / "case01_T2.nii.gz"; p_t2.touch()
        mask = tmp_path / "case01_mask.nii.gz"; mask.touch()
        rec = mgr.register({"T1": p_t1, "T2": p_t2}, mask, _make_indices(8))
        assert rec.sequence_paths["T1"] == p_t1
        assert rec.sequence_paths["T2"] == p_t2
        assert rec.segmentation_path == mask


class TestLookup:
    def test_lookup_correct_case_and_coord(self, tmp_path):
        mgr = _mgr(tmp_path)
        idx0 = _make_indices(20, seed=0)
        idx1 = _make_indices(15, seed=1)
        p0 = tmp_path / "case00.nii.gz"; p0.touch()
        p1 = tmp_path / "case01.nii.gz"; p1.touch()
        m = tmp_path / "m.nii.gz"
        mgr.register({"T1": p0}, m, idx0)
        mgr.register({"T1": p1}, m, idx1)

        case_id, coords = mgr.lookup(5)
        assert case_id == "case00"
        assert coords == tuple(int(v) for v in idx0[5])

        case_id, coords = mgr.lookup(20)
        assert case_id == "case01"
        assert coords == tuple(int(v) for v in idx1[0])

    def test_lookup_out_of_bounds(self, tmp_path):
        mgr = _mgr(tmp_path)
        p = tmp_path / "case00.nii.gz"; p.touch()
        mgr.register({"T1": p}, tmp_path / "m.nii.gz", _make_indices(10))
        with pytest.raises(IndexError):
            mgr.lookup(10)

    def test_lookup_no_records_raises(self, tmp_path):
        mgr = _mgr(tmp_path)
        with pytest.raises(RuntimeError):
            mgr.lookup(0)

    def test_lookup_column(self, tmp_path):
        mgr = _mgr(tmp_path, sequences=("T1", "T2"))
        p = tmp_path / "case00.nii.gz"; p.touch()
        cols = _labels("T1") + _labels("T2")  # 10 columns
        mgr.register({"T1": p, "T2": p}, tmp_path / "m.nii.gz", _make_indices(5), column_labels=cols)
        assert mgr.lookup_column(0) == ("T1", "Original")
        assert mgr.lookup_column(5) == ("T2", "Original")
        assert mgr.lookup_column(9) == ("T2", "Exponential")

    def test_lookup_column_no_labels_raises(self, tmp_path):
        mgr = _mgr(tmp_path)
        with pytest.raises(RuntimeError):
            mgr.lookup_column(0)

    def test_lookup_column_out_of_range_raises(self, tmp_path):
        mgr = _mgr(tmp_path)
        p = tmp_path / "case00.nii.gz"; p.touch()
        mgr.register({"T1": p}, tmp_path / "m.nii.gz", _make_indices(5), column_labels=_labels("T1"))
        with pytest.raises(IndexError):
            mgr.lookup_column(999)


class TestRecordsProperty:
    def test_snapshot_not_affected_by_later_registration(self, tmp_path):
        mgr = _mgr(tmp_path)
        p0 = tmp_path / "case00.nii.gz"; p0.touch()
        p1 = tmp_path / "case01.nii.gz"; p1.touch()
        m = tmp_path / "m.nii.gz"
        mgr.register({"T1": p0}, m, _make_indices(5))
        snap = mgr.records
        mgr.register({"T1": p1}, m, _make_indices(5))
        assert len(snap) == 1
        assert len(mgr.records) == 2
