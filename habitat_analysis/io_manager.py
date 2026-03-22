"""
IO manager for HabitatPipeline — multi-sequence path specification, provenance
tracking, and context management.

HabitatIOManager is initialised with the input data layout:

* ``sequence_paths`` — an ordered mapping of sequence name → NIfTI file or
  directory, one entry per MRI sequence (e.g. T1, T2, T1CE).
* ``segmentation_path`` — path to the ROI mask file or directory.

During a training or inference run every foreground voxel that enters the
combined feature matrix is registered together with its 3-D coordinate,
source NIfTI paths, and the column layout
``(sequence_name, filter_name)`` that describes each feature column.

Case IDs are extracted from filenames using :func:`mnts.utils.get_unique_IDs`
with the same id_globber regex as the wrapped pipeline.

Feature column labels use a double-underscore separator:
``"{sequence_name}__{filter_name}"`` (e.g. ``"T1__Original"``, ``"T2__LBP2D"``).

Example — single-sequence inference::

    pipeline = HabitatPipeline()

    with HabitatIOManager(
        pipeline=pipeline,
        sequence_paths={"T1": Path("images/T1/")},
        segmentation_path=Path("masks/"),
    ) as mgr:
        mgr.infer(img_dir, mask_dir, "model.zip", out_dir)
        mgr.print_summary()
        case_id, (z, y, x) = mgr.lookup(1234)

Example — multi-sequence with column lookup::

    with HabitatIOManager(
        pipeline=pipeline,
        sequence_paths={"T1": img_dir_t1, "T2": img_dir_t2, "T1CE": img_dir_t1ce},
        segmentation_path=mask_dir,
    ) as mgr:
        mgr.train(out_state="model.zip")
        mgr.print_summary()
        seq_name, filter_name = mgr.lookup_column(7)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from mnts.mnts_logger import MNTSLogger
from mnts.utils import get_unique_IDs

logger = MNTSLogger[__name__]


# ---------------------------------------------------------------------------
# Provenance record
# ---------------------------------------------------------------------------


@dataclass
class VoxelRecord:
    """Provenance for all foreground voxels in one NIfTI case.

    Attributes:
        case_id: String key extracted from the filename by id_globber.
        sequence_paths: Mapping of sequence name → path to the NIfTI file
            used for this case (may be a normalised copy in a temp directory).
        segmentation_path: Path to the segmentation / mask file for this case.
        voxel_indices: ``(N, 3)`` int array of ``(z, y, x)`` voxel coordinates
            within the 3-D volume (SimpleITK ``GetArrayFromImage`` ordering).
        row_start: First row index (inclusive) in the combined feature matrix.
        row_end: One-past-last row index (exclusive) in the combined matrix.
    """

    case_id: str
    sequence_paths: Dict[str, Path]
    segmentation_path: Path
    voxel_indices: np.ndarray  # (N, 3) — z, y, x
    row_start: int
    row_end: int

    @property
    def n_voxels(self) -> int:
        """Number of foreground voxels for this case."""
        return self.row_end - self.row_start

    def coord_at(self, local_idx: int) -> Tuple[int, int, int]:
        """Return the ``(z, y, x)`` coordinate for a case-local voxel index."""
        z, y, x = self.voxel_indices[local_idx]
        return int(z), int(y), int(x)


# ---------------------------------------------------------------------------
# IO manager
# ---------------------------------------------------------------------------


class HabitatIOManager:
    """Context manager wrapping :class:`~pipeline.HabitatPipeline` with
    multi-sequence path specification and voxel provenance tracking.

    The manager is initialised once with the structural description of the
    dataset (which sequences exist, where masks live).  During a pipeline run
    it accumulates one :class:`VoxelRecord` per case and tracks the feature
    column layout so any column index can be resolved back to a
    ``(sequence_name, filter_name)`` pair.

    When used as a context manager the internal registry is cleared on entry
    so that each ``with`` block starts from a clean state.

    Args:
        pipeline: A :class:`~pipeline.HabitatPipeline` instance to delegate to.
        sequence_paths: Ordered mapping ``{sequence_name: Path}`` or a plain
            list of paths (auto-named ``"seq_0"``, ``"seq_1"``, …).  Each path
            is the image file or directory for that sequence.
        segmentation_path: Path to the segmentation file or directory.
        id_globber: Regex passed to :func:`mnts.utils.get_unique_IDs` to
            extract the case key from each filename.  Defaults to the
            pipeline's ``id_globber``.
    """

    def __init__(
        self,
        pipeline,
        sequence_paths: Union[Dict[str, Path], List[Path]],
        segmentation_path: Union[str, Path],
        id_globber: Optional[str] = None,
    ):
        self.pipeline = pipeline
        self.segmentation_path = Path(segmentation_path)
        self.id_globber = id_globber or pipeline.id_globber

        # Normalise sequence_paths to an ordered dict
        if isinstance(sequence_paths, dict):
            self.sequence_paths: Dict[str, Path] = {k: Path(v) for k, v in sequence_paths.items()}
        else:
            self.sequence_paths = {f"seq_{i}": Path(p) for i, p in enumerate(sequence_paths)}

        self._records: Dict[str, VoxelRecord] = {}
        self._row_cursor: int = 0
        self._feature_columns: List[str] = []  # set on first registration with labels

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def __enter__(self) -> "HabitatIOManager":
        self._records.clear()
        self._row_cursor = 0
        self._feature_columns = []
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False  # never suppress exceptions

    # ------------------------------------------------------------------
    # Feature-column layout
    # ------------------------------------------------------------------

    @property
    def feature_columns(self) -> List[str]:
        """Ordered list of ``"{sequence}__{filter}"`` labels for each feature column.

        Populated after the first :meth:`register` call that carries column
        labels.  Empty until then.
        """
        return list(self._feature_columns)

    @property
    def sequence_names(self) -> List[str]:
        """Ordered list of sequence names as provided at construction."""
        return list(self.sequence_paths.keys())

    def lookup_column(self, col_idx: int) -> Tuple[str, str]:
        """Resolve a feature column index to ``(sequence_name, filter_name)``.

        Args:
            col_idx: Zero-based column index in the feature matrix.

        Returns:
            ``(sequence_name, filter_name)`` tuple.

        Raises:
            RuntimeError: If feature columns have not been set yet.
            IndexError: If *col_idx* is out of range.
        """
        if not self._feature_columns:
            raise RuntimeError(
                "Feature columns not yet set. Run a pipeline method first."
            )
        if col_idx < 0 or col_idx >= len(self._feature_columns):
            raise IndexError(
                f"col_idx {col_idx} out of range (n_columns={len(self._feature_columns)})"
            )
        label = self._feature_columns[col_idx]
        seq, filt = label.split("__", 1)
        return seq, filt

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        sequence_paths: Dict[str, Path],
        segmentation_path: Union[str, Path],
        voxel_indices: np.ndarray,
        column_labels: Optional[List[str]] = None,
    ) -> VoxelRecord:
        """Record the voxel origins for one case.

        Extracts the case ID from one of the sequence paths using
        :func:`mnts.utils.get_unique_IDs` and advances the internal row cursor
        by ``len(voxel_indices)``.

        Args:
            sequence_paths: Mapping of sequence name → NIfTI path actually
                used for this case (may differ from the constructor paths when
                normalised copies are in a temp directory).
            segmentation_path: Path to the mask file used for this case.
            voxel_indices: ``(N, 3)`` array of ``(z, y, x)`` foreground
                coordinates from :meth:`~feature_extractor.PixelwiseFeatureExtractor.extract`.
            column_labels: Optional list of ``"{seq}__{filter}"`` strings
                describing each feature column.  Updates :attr:`feature_columns`
                if provided and not yet set.

        Returns:
            The newly created :class:`VoxelRecord`.
        """
        sequence_paths = {k: Path(v) for k, v in sequence_paths.items()}
        segmentation_path = Path(segmentation_path)

        # Extract case ID from the first available sequence path
        reference_path = next(iter(sequence_paths.values()))
        ids = get_unique_IDs([reference_path], globber=self.id_globber)
        case_id = ids[0] if ids else reference_path.stem

        if case_id in self._records:
            logger.warning(f"Duplicate case_id '{case_id}'; overwriting previous record.")

        if column_labels and not self._feature_columns:
            self._feature_columns = list(column_labels)

        n = len(voxel_indices)
        record = VoxelRecord(
            case_id=case_id,
            sequence_paths=sequence_paths,
            segmentation_path=segmentation_path,
            voxel_indices=voxel_indices,
            row_start=self._row_cursor,
            row_end=self._row_cursor + n,
        )
        self._records[case_id] = record
        self._row_cursor += n
        seq_files = ", ".join(f"{k}={v.name}" for k, v in sequence_paths.items())
        logger.info(
            f"Registered '{case_id}': {n} voxels "
            f"(rows {record.row_start}–{record.row_end - 1})  [{seq_files}]"
        )
        return record

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(self, row_idx: int) -> Tuple[str, Tuple[int, int, int]]:
        """Trace a flat row index in the combined feature matrix to its source.

        Args:
            row_idx: Zero-based row index in the concatenated feature matrix.

        Returns:
            ``(case_id, (z, y, x))`` — the case that owns this row and the
            3-D coordinate of the corresponding voxel.

        Raises:
            RuntimeError: If no cases have been registered yet.
            IndexError: If *row_idx* is outside ``[0, total_voxels)``.
        """
        if not self._records:
            raise RuntimeError("No records registered. Run a pipeline method first.")
        for record in self._records.values():
            if record.row_start <= row_idx < record.row_end:
                return record.case_id, record.coord_at(row_idx - record.row_start)
        raise IndexError(f"row_idx {row_idx} out of bounds (total rows: {self._row_cursor})")

    def get_record(self, case_id: str) -> VoxelRecord:
        """Return the :class:`VoxelRecord` for *case_id*.

        Raises:
            KeyError: If *case_id* is not registered.
        """
        if case_id not in self._records:
            raise KeyError(
                f"case_id '{case_id}' not found. Available: {sorted(self._records)}"
            )
        return self._records[case_id]

    @property
    def records(self) -> Dict[str, VoxelRecord]:
        """Snapshot of all registered :class:`VoxelRecord` objects keyed by case ID."""
        return dict(self._records)

    @property
    def total_voxels(self) -> int:
        """Total number of registered voxels across all cases."""
        return self._row_cursor

    # ------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Log a structured summary: dataset layout, feature columns, and per-case table."""
        lines: List[str] = []

        # --- Dataset layout ---
        lines.append("=== IO Manager — Dataset Layout ===")
        lines.append(f"  Sequences ({len(self.sequence_paths)}):")
        for seq, path in self.sequence_paths.items():
            lines.append(f"    {seq:<12}  {path}")
        lines.append(f"  Segmentation:  {self.segmentation_path}")

        # --- Feature column layout ---
        if self._feature_columns:
            n_seq = len(self.sequence_paths)
            n_per_seq = len(self._feature_columns) // n_seq if n_seq else 0
            lines.append(
                f"  Feature columns: {len(self._feature_columns)} total "
                f"({n_seq} sequences × {n_per_seq} filters)"
            )
            for i, label in enumerate(self._feature_columns):
                lines.append(f"    [{i:>3}]  {label}")
        else:
            lines.append("  Feature columns: not yet determined")

        # --- Per-case table ---
        if self._records:
            lines.append("")
            lines.append("=== Registered Cases ===")
            col_id = max(len(r.case_id) for r in self._records.values())
            col_id = max(col_id, 7)
            header = f"  {'case_id':<{col_id}}  {'voxels':>8}  {'rows':>22}"
            sep = "  " + "-" * (col_id + 8 + 24)
            lines.append(header)
            lines.append(sep)
            for rec in self._records.values():
                row_range = f"{rec.row_start}–{rec.row_end - 1}"
                lines.append(f"  {rec.case_id:<{col_id}}  {rec.n_voxels:>8}  {row_range:>22}")
            lines.append(sep)
            lines.append(f"  {'TOTAL':<{col_id}}  {self.total_voxels:>8}")
        else:
            lines.append("  No cases registered.")

        logger.info("\n" + "\n".join(lines))

    # ------------------------------------------------------------------
    # Pipeline delegation
    # ------------------------------------------------------------------

    def train(
        self,
        img_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        out_state: Union[str, Path],
    ):
        """Delegate to :meth:`~pipeline.HabitatPipeline.train` with provenance tracking.

        Returns:
            The saved :class:`~state.HabitatState`.
        """
        return self.pipeline.train(img_dir, mask_dir, out_state, io_manager=self)

    def infer(
        self,
        img_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        state_path: Union[str, Path],
        out_dir: Union[str, Path],
        visualize: bool = True,
    ) -> List[Path]:
        """Delegate to :meth:`~pipeline.HabitatPipeline.infer` with provenance tracking.

        Returns:
            List of output NIfTI label-map paths.
        """
        return self.pipeline.infer(
            img_dir, mask_dir, state_path, out_dir,
            visualize=visualize, io_manager=self,
        )
