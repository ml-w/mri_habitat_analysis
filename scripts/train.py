"""
CLI entry point for habitat analysis training.

Usage — single sequence (backward-compatible)::

    habitat-train --img_dir /data/T1 --mask_dir /data/masks --out state.zip

Usage — multi-sequence (preferred)::

    habitat-train \\
        --seq T1:/data/T1 \\
        --seq T2:/data/T2 \\
        --seq T1CE:/data/T1CE \\
        --mask_dir /data/masks \\
        --out state.zip

The sequence names supplied via ``--seq`` are written into the state archive
metadata.  ``habitat-infer`` reads these names back to verify that the same
sequences are provided at inference time.
"""

from pathlib import Path
from typing import Dict, Optional

import click
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from mnts.mnts_logger import MNTSLogger

from habitat_analysis.pipeline import HabitatPipeline

stream_handler = MNTSLogger.shared_handlers.get('stream_handler')
console = stream_handler.console  # the RichHandler's Console


def _parse_seq_option(param, values) -> dict:
    """Parse repeatable ``NAME:DIR`` tokens into an ordered dict."""
    result = {}
    for token in values:
        if ":" not in token:
            raise click.BadParameter(
                f"Expected NAME:DIR, got '{token}'. Example: --seq T1:/data/T1",
                param=param,
            )
        name, _, dir_str = token.partition(":")
        name = name.strip()
        p = Path(dir_str.strip())
        if not p.is_dir():
            raise click.BadParameter(f"Directory does not exist: {p}", param=param)
        result[name] = p
    return result


@click.command()
@click.option(
    "--seq", "seq_tokens", multiple=True, metavar="NAME:DIR",
    help="Sequence name:directory pair (repeatable). "
         "E.g. --seq T1:/data/T1 --seq T2:/data/T2",
)
@click.option(
    "--img-dir", default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Single-sequence image directory (alias for --seq image:DIR).",
)
@click.option(
    "--mask-dir", required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory of binary mask NIfTI files.",
)
@click.option(
    "--out", required=True,
    type=click.Path(path_type=Path),
    help="Output state path — a .zip archive if the suffix is .zip, otherwise a plain directory.",
)
@click.option(
    "--norm-config", default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="mnts normalisation YAML (default: bundled).",
)
@click.option(
    "--pyrad-config", default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="PyRadiomics filter YAML (default: bundled).",
)
@click.option(
    "--method", default="kmeans",
    type=click.Choice(["kmeans", "gmm"]),
    help="Clustering algorithm.",
)
@click.option(
    "--k-min", default=2, type=int, show_default=True,
    help="Minimum number of clusters to evaluate."
)
@click.option(
    "--k-max", default=6, type=int, show_default=True,
    help="Maximum number of clusters to evaluate (inclusive)."
)
@click.option(
    "--k-selection", default="elbow", show_default=True,
    type=click.Choice(["elbow", "composite"]),
    help="Strategy for selecting best k: 'elbow' (recommended) or 'composite'.",
)
@click.option(
    "--subsample", default=200_000, type=int, show_default=True,
    help="Max voxels for clustering fit (0 = use all)."
)
@click.option("--seed", default=42, type=int, show_default=True,
              help="Random seed.")
@click.option("--id-globber", default=r"^[0-9a-zA-Z]+", show_default=True,
              help="Regex to extract case IDs from filenames.")
@click.option("--skip-norm", is_flag=True,
              help="Skip normalisation (use when images are already normalised).")
@click.option(
    "--seg-dir", default=None,
    type=click.Path(path_type=Path),
    help="Directory for output habitat segmentations (default: <out>/clustered_labels/).",
)
@click.option("--no-vis", is_flag=True,
              help="Disable cluster PCA visualization PNG.")
@click.option("--workers", "-n", default=1, type=int, show_default=True,
              help="Number of parallel workers for feature extraction.")
@click.option(
    "--y-true", default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="CSV or XLSX file for supervised feature selection (t-test). "
         "First column = case ID index, second column = binary label (0/1).",
)
@click.option("--force-extract", is_flag=True,
              help="Force feature re-extraction even if cached features.parquet exists.")
@click.option("--debug", is_flag=True,
              help="Debug mode: process only the first 3 cases.")
@click.option("--verbose", "-v", is_flag=True, help="Enable DEBUG logging.")
def main(
    seq_tokens: tuple,
    img_dir: Path,
    mask_dir: Path,
    out: Path,
    norm_config: Path,
    pyrad_config: Path,
    method: str,
    k_min: int,
    k_max: int,
    k_selection: str,
    subsample: int,
    seed: int,
    id_globber: str,
    skip_norm: bool,
    seg_dir: Optional[Path],
    no_vis: bool,
    workers: int,
    y_true: Optional[Path],
    force_extract: bool,
    debug: bool,
    verbose: bool,
):
    """Train the habitat analysis pipeline and save the state archive."""
    if verbose:
        MNTSLogger.set_global_log_level('debug')

    # ── Resolve sequence directories ──────────────────────────────────────────
    if seq_tokens and img_dir:
        raise click.UsageError("Use either --seq or --img_dir, not both.")
    if not seq_tokens and img_dir is None:
        raise click.UsageError("Provide at least one --seq NAME:DIR or --img_dir.")

    if img_dir is not None:
        seq_dirs = {"image": img_dir}
    else:
        seq_dirs = _parse_seq_option(None, seq_tokens)
        
    if not seq_dirs:
        raise click.UsageError("No valid sequence directories were parsed.")

    # ── Load y_true labels if provided ─────────────────────────────────────────
    y_true_map: Optional[Dict[str, int]] = None
    if y_true is not None:
        import re
        id_re = re.compile(id_globber)
        if y_true.suffix in (".xls", ".xlsx"):
            df_y = pd.read_excel(y_true, index_col=0)
        else:
            df_y = pd.read_csv(y_true, index_col=0)
        label_col = df_y.columns[0]
        y_true_map = {}
        for idx, val in df_y[label_col].items():
            # Apply the same id-globber regex to the index so it matches case IDs
            m = id_re.search(str(idx))
            key = m.group(0) if m else str(idx)
            y_true_map[key] = int(val)

        # Cross-reference y_true case IDs with discoverable imaging data
        from habitat_analysis.pipeline import _index_dir
        mask_index = _index_dir(mask_dir, id_globber)
        seq_id_sets = [set(_index_dir(d, id_globber)) for d in seq_dirs.values()]
        imaging_ids = set(mask_index) & set.intersection(*seq_id_sets) if seq_id_sets else set(mask_index)

        matched = set(y_true_map) & imaging_ids
        y_only = set(y_true_map) - imaging_ids
        img_only = imaging_ids - set(y_true_map)

        match_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        match_table.add_column(style="bold", no_wrap=True)
        match_table.add_column()
        match_table.add_row("Label file", str(y_true))
        match_table.add_row("Cases in label file", str(len(y_true_map)))
        match_table.add_row("Cases in imaging dirs", str(len(imaging_ids)))
        match_table.add_row("[green]Matched[/green]", f"[green]{len(matched)}[/green]")
        if y_only:
            match_table.add_row("[yellow]Labels without images[/yellow]",
                                f"[yellow]{len(y_only)}[/yellow]: {sorted(y_only)[:10]}{'...' if len(y_only) > 10 else ''}")
        if img_only:
            match_table.add_row("[yellow]Images without labels[/yellow]",
                                f"[yellow]{len(img_only)}[/yellow]: {sorted(img_only)[:10]}{'...' if len(img_only) > 10 else ''}")
        console.print(Panel(match_table, title="[bold]y_true Coverage", border_style="yellow"))

        if not matched:
            raise click.UsageError(
                "No y_true case IDs match the imaging data. "
                "Check the label file index and --id-globber."
            )

        # Keep only labels that have corresponding imaging data
        y_true_map = {k: v for k, v in y_true_map.items() if k in matched}

    # ── Print run configuration ───────────────────────────────────────────────
    cfg_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    cfg_table.add_column(style="bold cyan", no_wrap=True)
    cfg_table.add_column()

    for seq_name, seq_path in seq_dirs.items():
        cfg_table.add_row(f"Sequence [{seq_name}]", str(seq_path))
    cfg_table.add_row("Mask directory", str(mask_dir))
    cfg_table.add_row("Output state", str(out))
    cfg_table.add_row("Method", method)
    cfg_table.add_row("k range", f"{k_min}–{k_max}")
    cfg_table.add_row("Subsample", str(subsample) if subsample > 0 else "all")
    cfg_table.add_row("Skip norm", "yes" if skip_norm else "no")
    if force_extract:
        cfg_table.add_row("Force extract", "[bold yellow]yes[/bold yellow]")
    if y_true_map is not None:
        cfg_table.add_row("y_true", f"{y_true} ({len(y_true_map)} cases)")
    if debug:
        cfg_table.add_row("Debug mode", "[bold yellow]ON — first 3 cases only[/bold yellow]")

    console.print(Panel(cfg_table, title="[bold]Habitat Training", border_style="blue"))

    # ── Run pipeline ──────────────────────────────────────────────────────────
    pipeline = HabitatPipeline(
        norm_config=norm_config,
        pyrad_config=pyrad_config,
        id_globber=id_globber,
        cluster_method=method,
        k_range=range(k_min, k_max + 1),
        k_selection=k_selection,
        subsample=subsample if subsample > 0 else None,
        random_state=seed,
        visualize=not no_vis,
    )
    # Default seg_dir: alongside state for zip, inside out dir otherwise
    if seg_dir is None:
        if out.suffix == ".zip":
            seg_dir = out.parent / "clustered_labels"
        else:
            seg_dir = out / "clustered_labels"

    state = pipeline.train(
        seq_dirs=seq_dirs,
        mask_dir=mask_dir,
        out_state=out,
        out_seg_dir=seg_dir,
        skip_norm=skip_norm,
        max_cases=3 if debug else None,
        n_workers=workers,
        y_true=y_true_map,
        force_extract=force_extract,
    )

    # ── Print results ─────────────────────────────────────────────────────────
    res_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    res_table.add_column(style="bold green", no_wrap=True)
    res_table.add_column()
    res_table.add_row("State archive", str(out))
    res_table.add_row("Segmentations", str(seg_dir))
    res_table.add_row("Required sequences", ", ".join(state.required_sequences or ["-"]))
    res_table.add_row("Best k", str(state.metadata.get("best_k")))

    console.print(Panel(res_table, title="[bold]Training Complete", border_style="green"))


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
