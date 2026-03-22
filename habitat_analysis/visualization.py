"""
Visualisation utilities for habitat analysis outputs.

Provides:
    - label_map_to_nifti  — write a numpy label array as a NIfTI file
    - render_habitat_overlay — PNG overlay of cluster labels on an axial MRI slice
"""

import logging
from pathlib import Path
from typing import Union

import numpy as np
import SimpleITK as sitk

from mnts.mnts_logger import MNTSLogger

logger = MNTSLogger[__name__]


def label_map_to_nifti(
    label_array: np.ndarray,
    reference_image: sitk.Image,
    out_path: Union[str, Path],
) -> None:
    """Write *label_array* as a NIfTI image sharing geometry with *reference_image*.

    Note:
        This function does not perform inference.

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


def _pick_representative_slice(mask_array: np.ndarray) -> int:
    """Return the axial slice index (z) with the most foreground voxels."""
    counts = np.sum(mask_array > 0, axis=(1, 2))
    return int(np.argmax(counts))


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
    import matplotlib.colors as mcolors

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

    # Legend
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
