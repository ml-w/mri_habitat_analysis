"""
MRI normalisation wrapper using mri-normalization-tools (mnts).

The normalisation graph is defined in configs/normalization.yaml and follows:
    SpatialNorm -> HuangThresholding -> N4ITKBiasFieldCorrection -> NyulNormalizer

NyulNormalizer requires a training step to learn histogram landmarks from a
representative dataset; all other nodes are parameter-free.
"""

import logging
import tempfile
from pathlib import Path
from typing import Union

from mnts.mnts_logger import MNTSLogger

logger = MNTSLogger[__name__]


class HabitatNormalizer:
    """Thin wrapper around mnts MNTSFilterGraph for habitat analysis normalisation.

    Args:
        config_path: Path to the normalization YAML config.  Defaults to the
            bundled ``configs/normalization.yaml``.
        id_globber: Regex used by mnts to extract case IDs from file names.
    """

    _DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "normalization.yaml"

    def __init__(
        self,
        config_path: Union[str, Path, None] = None,
        id_globber: str = r"^[0-9a-zA-Z]+",
    ):
        self.config_path = Path(config_path) if config_path else self._DEFAULT_CONFIG
        self.id_globber = id_globber

    def _build_graph(self):
        """Instantiate the MNTSFilterGraph from YAML."""
        from mnts.filters import MNTSFilterGraph

        graph = MNTSFilterGraph.CreateGraphFromYAML(str(self.config_path))
        return graph

    def train(
        self,
        img_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        state_save_path: Union[str, Path],
    ) -> None:
        """Train the NyulNormalizer on a set of images and save the graph state.

        Args:
            img_dir: Directory containing input NIfTI images.
            mask_dir: Directory containing corresponding binary masks.
            state_save_path: Path to save the trained normaliser state (.yaml or directory).
        """
        from mnts.scripts.normalization import run_graph_train

        img_dir = Path(img_dir)
        mask_dir = Path(mask_dir)
        state_save_path = Path(state_save_path)
        state_save_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Training normalisation graph on {img_dir}")
        run_graph_train(
            str(self.config_path),
            str(img_dir),
            str(mask_dir),
            str(state_save_path),
            id_globber=self.id_globber,
        )
        logger.info(f"Normaliser state saved to {state_save_path}")

    def infer(
        self,
        img_dir: Union[str, Path],
        out_dir: Union[str, Path],
        state_path: Union[str, Path],
        mask_dir: Union[str, Path, None] = None,
    ) -> None:
        """Apply trained normalisation to images and write results to out_dir.

        Args:
            img_dir: Directory containing input NIfTI images.
            out_dir: Directory to write normalised images.
            state_path: Path to the trained normaliser state produced by :meth:`train`.
            mask_dir: Optional mask directory (passed to mnts if required).
        """
        from mnts.scripts.normalization import run_graph_inference

        img_dir = Path(img_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Applying normalisation from {state_path} to images in {img_dir}")
        run_graph_inference(
            str(self.config_path),
            str(img_dir),
            str(out_dir),
            str(state_path),
            *([str(mask_dir)] if mask_dir else []),
            id_globber=self.id_globber,
        )
        logger.info(f"Normalised images written to {out_dir}")

    def infer_single(
        self,
        image,  # SimpleITK.Image
        state_path: Union[str, Path],
        mask=None,  # SimpleITK.Image or None
    ):
        """Normalise a single SimpleITK image in memory and return the result.

        Writes the image to a temporary directory, runs inference, and reads
        the result back.  Useful for pipeline integration without disk I/O overhead.

        Args:
            image: SimpleITK image to normalise.
            state_path: Path to trained normaliser state.
            mask: Optional SimpleITK binary mask.

        Returns:
            Normalised SimpleITK image.
        """
        import SimpleITK as sitk

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            in_dir = tmp / "in"
            out_dir = tmp / "out"
            in_dir.mkdir()

            sitk.WriteImage(image, str(in_dir / "case.nii.gz"))
            mask_dir = None
            if mask is not None:
                mask_path = tmp / "masks"
                mask_path.mkdir()
                sitk.WriteImage(mask, str(mask_path / "case.nii.gz"))
                mask_dir = mask_path

            self.infer(in_dir, out_dir, state_path, mask_dir=mask_dir)

            result_files = list(out_dir.rglob("*.nii.gz"))
            if not result_files:
                raise RuntimeError("Normalisation produced no output files.")
            return sitk.ReadImage(str(result_files[0]))
