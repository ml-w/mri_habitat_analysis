"""
Visualisation utilities for habitat analysis outputs.

Provides:
    - label_map_to_nifti      — write a numpy label array as a NIfTI file
    - compute_display_params  — pick representative axial slice and ROI crop bounds
    - load_case_images        — load seq images + optional seg/habitat for one case ID
    - iter_cases              — iterate all cases found across multiple sequence dirs
    - plot_filter_grid        — multi-sequence × multi-filter panel figure
    - plot_habitat_overlay    — side-by-side raw / cluster-colour overlay figure
    - render_habitat_overlay  — save habitat overlay PNG (pipeline-facing API)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk

from mnts.mnts_logger import MNTSLogger
from mnts.utils import get_unique_IDs

logger = MNTSLogger[__name__]

# Type aliases
CropParams = Tuple[int, int, int, int, int]   # (best_z, y0, y1, x0, x1)
VolumeDict = Dict[str, np.ndarray]            # filter_label → array [Z,Y,X]
SeqVolumes = Dict[str, VolumeDict]            # seq_name → VolumeDict


# ---------------------------------------------------------------------------
# NIfTI I/O
# ---------------------------------------------------------------------------

def label_map_to_nifti(
    label_array: np.ndarray,
    reference_image: sitk.Image,
    out_path: Union[str, Path],
) -> None:
    """Write *label_array* as a NIfTI image sharing geometry with *reference_image*.

    Args:
        label_array: Integer array ``(Z, Y, X)`` with cluster labels (0 = background).
        reference_image: SimpleITK image whose origin, spacing, and direction are copied.
        out_path: Destination ``.nii.gz`` path.
    """
    out_path = Path(out_path)
    label_sitk = sitk.GetImageFromArray(label_array.astype(np.int32))
    label_sitk.CopyInformation(reference_image)
    sitk.WriteImage(label_sitk, str(out_path))
    logger.debug(f"Label map written to {out_path}")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def compute_display_params(
    seg_arr: np.ndarray,
    pad: int = 20,
) -> CropParams:
    """Return the best axial slice and a padded crop bounding box around the ROI.

    Args:
        seg_arr: Binary/label mask array ``(Z, Y, X)``.
        pad: Pixel padding added around the tight bounding box.

    Returns:
        ``(best_z, y0, y1, x0, x1)`` — slice index and crop coordinates.
    """
    counts = np.sum(seg_arr > 0, axis=(1, 2))
    best_z = int(np.argmax(counts))
    seg_sl = seg_arr[best_z]
    ys, xs = np.where(seg_sl > 0)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad, seg_sl.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad, seg_sl.shape[1])
    return best_z, y0, y1, x0, x1


def _crop(arr: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    return arr[y0:y1, x0:x1]


def _percentile_window(arr: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> Tuple[float, float]:
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------

def _find_file_for_id(directory: Path, case_id: str, globber: str) -> Optional[Path]:
    """Return the first file in *directory* whose name matches *case_id* via *globber*."""
    id_dict = get_unique_IDs(sorted(directory.glob("*.nii.gz")), globber=globber, return_dict=True)
    files = id_dict.get(case_id)
    if files:
        return Path(files[0])
    return None


def load_case_images(
    seq_dirs: Dict[str, Path],
    case_id: str,
    globber: str = r"([0-9a-zA-Z]+)",
    seg_dir: Optional[Path] = None,
    habitat_dir: Optional[Path] = None,
) -> Dict[str, sitk.Image]:
    """Load all available images for one *case_id*.

    Args:
        seq_dirs: Mapping of sequence name → directory of ``.nii.gz`` files
                  (e.g. ``{"T1": Path("..."), "T2": Path("...")}``).
        case_id: The case identifier to load.
        globber: Regex used by :func:`~mnts.utils.get_unique_IDs` to extract IDs
                 from filenames.
        seg_dir: Optional directory containing segmentation masks named
                 ``{case_id}.nii.gz``.
        habitat_dir: Optional directory containing habitat label maps named
                     ``{case_id}_habitat.nii.gz``.

    Returns:
        Dict with keys matching *seq_dirs* keys plus ``"seg"`` and ``"habitat"``
        where available.  Values are ``sitk.Image`` objects.

    Raises:
        FileNotFoundError: If a required sequence file cannot be found.
    """
    images: Dict[str, sitk.Image] = {}

    for seq_name, seq_dir in seq_dirs.items():
        path = _find_file_for_id(seq_dir, case_id, globber)
        if path is None:
            raise FileNotFoundError(f"No file matching ID '{case_id}' in {seq_dir}")
        images[seq_name] = sitk.ReadImage(str(path), sitk.sitkFloat32)

    if seg_dir is not None:
        seg_path = seg_dir / f"{case_id}.nii.gz"
        if seg_path.exists():
            images["seg"] = sitk.ReadImage(str(seg_path), sitk.sitkUInt8)
        else:
            logger.warning(f"Segmentation not found for {case_id} in {seg_dir}")

    if habitat_dir is not None:
        hab_path = habitat_dir / f"{case_id}_habitat.nii.gz"
        if hab_path.exists():
            images["habitat"] = sitk.ReadImage(str(hab_path), sitk.sitkUInt8)
        else:
            logger.warning(f"Habitat label map not found for {case_id} in {habitat_dir}")

    return images


def iter_cases(
    seq_dirs: Dict[str, Path],
    globber: str = r"([0-9a-zA-Z]+)",
    seg_dir: Optional[Path] = None,
    habitat_dir: Optional[Path] = None,
) -> Iterator[Tuple[str, Dict[str, sitk.Image]]]:
    """Iterate over all cases present in every sequence directory.

    Uses :func:`~mnts.utils.get_unique_IDs` to collect IDs from each sequence
    directory, takes the intersection, then yields ``(case_id, images)`` for
    each common ID in sorted order.

    Args:
        seq_dirs: Mapping of sequence name → directory of ``.nii.gz`` files.
        globber: Regex for ID extraction.
        seg_dir: Optional segmentation directory (passed to
                 :func:`load_case_images`).
        habitat_dir: Optional habitat label directory (passed to
                     :func:`load_case_images`).

    Yields:
        ``(case_id, images)`` tuples — same dict structure as
        :func:`load_case_images`.
    """
    id_sets: List[set] = []
    for seq_dir in seq_dirs.values():
        ids = get_unique_IDs(sorted(seq_dir.glob("*.nii.gz")), globber=globber)
        id_sets.append(set(ids))

    common_ids = sorted(id_sets[0].intersection(*id_sets[1:]))
    logger.info(f"Found {len(common_ids)} common case IDs across {list(seq_dirs.keys())}")

    for case_id in common_ids:
        try:
            yield case_id, load_case_images(seq_dirs, case_id, globber, seg_dir, habitat_dir)
        except FileNotFoundError as exc:
            logger.warning(str(exc))


# ---------------------------------------------------------------------------
# Plot: multi-sequence × multi-filter grid
# ---------------------------------------------------------------------------

def plot_filter_grid(
    seq_volumes: SeqVolumes,
    seg_arr: np.ndarray,
    case_id: str,
    crop_params: CropParams,
    panel_width_in: float = 2.5,
    label_width_in: float = 0.9,
) -> "plt.Figure":
    """Render a grid figure with one column per sequence and one row per filter.

    Args:
        seq_volumes: ``{seq_name: {filter_label: ndarray[Z,Y,X]}}`` — from
                     e.g. ``get_filter_volumes`` applied per sequence.
        seg_arr: Segmentation mask ``(Z, Y, X)`` used for the ROI contour.
        case_id: Case identifier for the figure title.
        crop_params: ``(best_z, y0, y1, x0, x1)`` from
                     :func:`compute_display_params`.
        panel_width_in: Width of each image panel in inches.
        label_width_in: Reserved left-margin width for row labels in inches.

    Returns:
        The :class:`matplotlib.figure.Figure` — caller is responsible for
        saving or displaying it.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    best_z, y0, y1, x0, x1 = crop_params
    seq_names = list(seq_volumes.keys())
    filter_labels = list(next(iter(seq_volumes.values())).keys())
    n_rows, n_cols = len(filter_labels), len(seq_names)
    seg_crop = _crop(seg_arr[best_z], y0, y1, x0, x1)

    crop_h, crop_w = y1 - y0, x1 - x0
    panel_h_in = panel_width_in * (crop_h / max(crop_w, 1))
    fig_w = label_width_in + panel_width_in * n_cols
    fig_h = panel_h_in * n_rows

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(
        f"Case {case_id}  —  slice z={best_z}",
        fontsize=10, fontweight="bold",
        x=(label_width_in + (fig_w - label_width_in) / 2) / fig_w,
        y=1.01,
    )

    left_frac = label_width_in / fig_w
    gs = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        left=left_frac, right=1.0, top=0.95, bottom=0.0,
        hspace=0.0, wspace=0.0,
    )

    for col, seq_name in enumerate(seq_names):
        for row, label in enumerate(filter_labels):
            ax = fig.add_subplot(gs[row, col])
            cropped = _crop(seq_volumes[seq_name][label][best_z], y0, y1, x0, x1)
            vmin, vmax = _percentile_window(cropped)
            ax.imshow(cropped, cmap="gray", vmin=vmin, vmax=vmax,
                      aspect="auto", interpolation="none")
            ax.contour(seg_crop, levels=[0.5], colors="red", linewidths=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(seq_name, fontsize=15, pad=3)
            if col == 0:
                ax.set_ylabel(label, fontsize=15, rotation=0,
                              ha="right", va="center", labelpad=4)

    return fig


# ---------------------------------------------------------------------------
# Plot: habitat overlay
# ---------------------------------------------------------------------------

def plot_habitat_overlay(
    seq_images: Dict[str, np.ndarray],
    habitat_arr: np.ndarray,
    seg_arr: np.ndarray,
    case_id: str,
    crop_params: CropParams,
    bg_seq: str = "T2",
    raw_seq: Optional[str] = None,
    slice_offset: int = 0,
    alpha: float = 0.55,
    cmap_name: Union[str, Dict[int, tuple]] = "tab10",
    label_map: Optional[Dict[int, str]] = None,
    contour_mode: bool = False,
) -> "plt.Figure":
    """Render a side-by-side figure: background sequence alone vs. habitat overlay.

    Args:
        seq_images: ``{seq_name: ndarray[Z,Y,X]}`` — raw image arrays
                    (float, any range; percentile windowing is applied).
        habitat_arr: Integer label array ``(Z, Y, X)`` — 0 = background.
        seg_arr: Binary mask ``(Z, Y, X)`` used for the ROI contour and to
                 restrict overlay to tumour voxels.
        case_id: Case identifier for the figure title.
        crop_params: ``(best_z, y0, y1, x0, x1)`` from
                     :func:`compute_display_params`.
        bg_seq: Key in *seq_images* to use as the greyscale background for
                the right (overlay) panel.  Falls back to the first available
                key if not found.
        raw_seq: Key in *seq_images* to use as the left (plain) panel.  When
                 ``None`` (default), the left panel uses *bg_seq* as before.
        alpha: Opacity of the cluster colour overlay (ignored in contour mode).
        cmap_name: Matplotlib colormap name for cluster colours, or a dictionary
                   mapping cluster IDs to RGB/RGBA tuples (e.g. ``{1: (1, 0, 0), 2: (0, 1, 0)}``).
        label_map: Optional ``{cluster_id: label_string}`` to rename clusters
                   in the legend (e.g. ``{1: "Hypoxic", 2: "Viable"}``).
                   Unmapped IDs fall back to ``"Cluster {id}"``.
        contour_mode: If ``True``, draw inner contour lines for each cluster
                      instead of filled colour patches.  Each cluster mask is
                      eroded by one pixel before contouring so borders between
                      adjacent segments do not overlap.

    Returns:
        The :class:`matplotlib.figure.Figure`.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import ListedColormap

    best_z, y0, y1, x0, x1 = crop_params
    n_slices = next(iter(seq_images.values())).shape[0]
    best_z = int(np.clip(best_z + slice_offset, 0, n_slices - 1))

    if bg_seq not in seq_images:
        bg_seq = next(iter(seq_images))
        logger.warning(f"bg_seq not found; using '{bg_seq}' as background.")

    raw_seq_key = raw_seq if (raw_seq and raw_seq in seq_images) else None
    if raw_seq and raw_seq not in seq_images:
        logger.warning(f"raw_seq '{raw_seq}' not found; skipping raw panel.")

    bg_crop  = _crop(seq_images[bg_seq][best_z], y0, y1, x0, x1)
    hab_crop = _crop(habitat_arr[best_z], y0, y1, x0, x1)
    seg_crop = _crop(seg_arr[best_z], y0, y1, x0, x1)

    cluster_ids = sorted(int(v) for v in np.unique(habitat_arr[best_z][seg_arr[best_z] > 0]) if v != 0)
    n_clusters  = len(cluster_ids)

    if isinstance(cmap_name, dict):
        colors = [cmap_name.get(cid, (0.5, 0.5, 0.5)) for cid in cluster_ids]
        cmap = ListedColormap(colors)
    else:
        tab_colors = plt.get_cmap(cmap_name).colors
        cmap = ListedColormap([tab_colors[i % len(tab_colors)] for i in range(n_clusters)])

    n_cols = 3 if raw_seq_key else 2
    crop_h, crop_w = y1 - y0, x1 - x0
    panel_w = 2.5
    panel_h = panel_w * (crop_h / max(crop_w, 1))
    fig_w = panel_w * n_cols
    fig = plt.figure(figsize=(fig_w, panel_h + 0.5))
    fig.suptitle(
        f"Case {case_id}  —  slice z={best_z}  —  habitat labels",
        fontsize=10, fontweight="bold",
    )
    gs = gridspec.GridSpec(
        1, n_cols, figure=fig,
        left=0.0, right=1.0, top=0.82, bottom=0.0,
        hspace=0.0, wspace=0.0,
    )

    col = 0
    if raw_seq_key:
        raw_crop = _crop(seq_images[raw_seq_key][best_z], y0, y1, x0, x1)
        raw_vmin, raw_vmax = _percentile_window(raw_crop)
        ax_raw = fig.add_subplot(gs[0, col])
        ax_raw.imshow(raw_crop, cmap="gray", vmin=raw_vmin, vmax=raw_vmax,
                      aspect="auto", interpolation="none")
        ax_raw.contour(seg_crop, levels=[0.5], colors="red", linewidths=0.8)
        ax_raw.set_title(f"{raw_seq_key}", fontsize=9)
        ax_raw.set_xticks([])
        ax_raw.set_yticks([])
        col += 1

    vmin, vmax = _percentile_window(bg_crop)

    ax_bg = fig.add_subplot(gs[0, col])
    ax_bg.imshow(bg_crop, cmap="gray", vmin=vmin, vmax=vmax,
                 aspect="auto", interpolation="none")
    ax_bg.contour(seg_crop, levels=[0.5], colors="red", linewidths=0.8)
    ax_bg.set_title(f"{bg_seq}", fontsize=9)
    ax_bg.set_xticks([])
    ax_bg.set_yticks([])
    col += 1

    ax_ov = fig.add_subplot(gs[0, col])
    ax_ov.imshow(bg_crop, cmap="gray", vmin=vmin, vmax=vmax,
                 aspect="auto", interpolation="none")

    if contour_mode:
        from scipy.ndimage import distance_transform_edt
        for idx, cid in enumerate(cluster_ids):
            cluster_mask = (hab_crop == cid) & (seg_crop > 0)
            if not cluster_mask.any():
                continue
            dist_inside  = distance_transform_edt(cluster_mask)
            dist_outside = distance_transform_edt(~cluster_mask)
            ax_ov.contour(dist_inside - dist_outside, levels=[0.5],
                          colors=[cmap(idx)[:3]], linewidths=1.5)
    else:
        rgba = np.zeros((*hab_crop.shape, 4), dtype=float)
        for idx, cid in enumerate(cluster_ids):
            mask = (hab_crop == cid) & (seg_crop > 0)
            rgba[mask] = (*cmap(idx)[:3], alpha)
        ax_ov.imshow(rgba, aspect="auto", interpolation="none")
        ax_ov.contour(seg_crop, levels=[0.5], colors="white", linewidths=0.8)

    ax_ov.set_title(f"{bg_seq} + clusters", fontsize=9)
    ax_ov.set_xticks([])
    ax_ov.set_yticks([])

    def _legend_label(cid: int) -> str:
        if label_map and cid in label_map:
            return label_map[cid]
        return f"Cluster {cid}"

    if isinstance(cmap_name, dict):
        color_map = {cid: cmap_name.get(cid, (0.5, 0.5, 0.5)) for cid in cluster_ids}
    else:
        color_map = {cid: cmap(i) for i, cid in enumerate(cluster_ids)}

    legend_order = [cid for cid in label_map.keys() if cid in cluster_ids] if label_map else cluster_ids
    legend_patches = [
        mpatches.Patch(color=color_map[cid], alpha=0.8, label=_legend_label(cid))
        for cid in legend_order
    ]
    ax_ov.legend(handles=legend_patches, loc="lower right", fontsize=7, framealpha=0.7)

    return fig


# ---------------------------------------------------------------------------
# Pipeline-facing save helper (original single-panel API)
# ---------------------------------------------------------------------------

def render_habitat_overlay(
    image_array: np.ndarray,
    label_array: np.ndarray,
    mask_array: np.ndarray,
    out_path: Union[str, Path],
    alpha: float = 0.4,
    cmap_name: str = "tab10",
) -> None:
    """Render a colour-coded habitat overlay on the axial MRI slice with the
    largest mask area and save it as a PNG.

    Args:
        image_array: Float image volume ``(Z, Y, X)``.
        label_array: Integer label volume ``(Z, Y, X)`` (0 = background).
        mask_array: Binary mask volume ``(Z, Y, X)``.
        out_path: Destination PNG path.
        alpha: Opacity of the cluster overlay (0 = transparent, 1 = opaque).
        cmap_name: Matplotlib colormap name for cluster colours.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    z = _pick_representative_slice(mask_array)

    img_slice = image_array[z].astype(np.float32)
    lbl_slice = label_array[z]

    # Normalise image to [0, 1] for display
    img_min, img_max = img_slice.min(), img_slice.max()
    if img_max > img_min:
        img_norm = (img_slice - img_min) / (img_max - img_min)
    else:
        img_norm = np.zeros_like(img_slice)

    k = int(lbl_slice.max())
    if k == 0:
        logger.warning(f"Label map slice {z} has no foreground labels — overlay will be empty.")

    cmap = plt.get_cmap(cmap_name)
    colours = [cmap(i / max(k, 1)) for i in range(1, k + 1)]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.imshow(img_norm, cmap="gray", interpolation="nearest")

    for cluster_id in range(1, k + 1):
        colour = colours[cluster_id - 1]
        overlay = np.zeros((*lbl_slice.shape, 4), dtype=np.float32)
        mask = lbl_slice == cluster_id
        overlay[mask, :3] = colour[:3]
        overlay[mask, 3] = alpha
        ax.imshow(overlay, interpolation="nearest")

    ax.set_title(f"Habitat overlay — axial slice {z}  (k={k})")
    ax.axis("off")

    patches = [
        plt.Rectangle((0, 0), 1, 1, color=colours[i], label=f"Habitat {i + 1}")
        for i in range(k)
    ]
    if patches:
        ax.legend(handles=patches, loc="lower right", fontsize=8, framealpha=0.7)

    fig.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    logger.debug(f"Overlay saved to {out_path}")
