"""
CLI entry point for habitat analysis inference.

Usage — single sequence (backward-compatible)::

    habitat-infer \\
        --img_dir /data/T1 \\
        --mask_dir /data/masks \\
        --state state.zip \\
        --out results/

Usage — multi-sequence (preferred)::

    habitat-infer \\
        --seq T1:/data/T1 \\
        --seq T2:/data/T2 \\
        --seq T1CE:/data/T1CE \\
        --mask_dir /data/masks \\
        --state state.zip \\
        --out results/

Sequence names are validated against those recorded at training time.
Use ``--override`` to skip this check (e.g. when testing with a subset
of sequences).
"""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from habitat_analysis.state import HabitatState

console = Console()


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
    "--state", required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the trained state archive (.zip).",
)
@click.option(
    "--out", required=True,
    type=click.Path(path_type=Path),
    help="Output path — a directory (created if needed) or a .zip archive.",
)
@click.option(
    "--norm_config", default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="mnts normalisation YAML (optional override).",
)
@click.option("--skip-norm", is_flag=True,
              help="Skip normalisation (for already-normalised data).")
@click.option("--no-vis", is_flag=True,
              help="Disable PNG overlay generation.")
@click.option("--override", is_flag=True,
              help="Skip sequence validation against training metadata.",)
@click.option("--workers", "-j", default=1, type=int, show_default=True,
              help="Number of parallel workers for inference.")
@click.option("--debug", is_flag=True,
              help="Debug mode: process only the first 3 cases.")
@click.option("--id-globber", default=r"^[0-9a-zA-Z]+", show_default=True,
              help="Regex to extract case IDs from filenames.")
@click.option("--verbose", "-v", is_flag=True, help="Enable DEBUG logging.")
def main(seq_tokens, img_dir, mask_dir, state, out, norm_config,
         skip_norm, no_vis, override, workers, debug, id_globber, verbose):
    """Run habitat analysis inference using a trained state archive."""
    from mnts.mnts_logger import MNTSLogger
    MNTSLogger("habitat_infer.log", logger_name="habitat_infer",
               verbose=verbose, log_level="debug" if verbose else "info")

    # ── Resolve sequence directories ──────────────────────────────────────────
    if seq_tokens and img_dir:
        raise click.UsageError("Use either --seq or --img_dir, not both.")
    if not seq_tokens and img_dir is None:
        raise click.UsageError("Provide at least one --seq NAME:DIR or --img_dir.")

    if img_dir is not None:
        seq_dirs = {"image": img_dir}
    else:
        seq_dirs = _parse_seq_option(None, seq_tokens)

    # ── Print run configuration ───────────────────────────────────────────────
    cfg_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    cfg_table.add_column(style="bold cyan", no_wrap=True)
    cfg_table.add_column()

    for seq_name, seq_path in seq_dirs.items():
        cfg_table.add_row(f"Sequence [{seq_name}]", str(seq_path))
    cfg_table.add_row("Mask directory", str(mask_dir))
    cfg_table.add_row("State archive", str(state))
    cfg_table.add_row("Output directory", str(out))
    cfg_table.add_row("Skip norm", "yes" if skip_norm else "no")
    cfg_table.add_row("Visualize", "no" if no_vis else "yes")
    if debug:
        cfg_table.add_row("Debug mode", "[bold yellow]ON — first 3 cases only[/bold yellow]")

    console.print(Panel(cfg_table, title="[bold]Habitat Inference", border_style="blue"))

    # ── Sequence validation ───────────────────────────────────────────────────
    loaded_state = HabitatState.load(state)
    required = loaded_state.required_sequences

    if required:
        console.print(f"[bold]Required sequences:[/bold] {', '.join(required)}")
        if override:
            console.print("[yellow]WARNING:[/yellow] --override set, skipping sequence validation.")
        else:
            try:
                warnings = loaded_state.validate_sequences(list(seq_dirs.keys()))
            except ValueError as exc:
                console.print(f"[bold red]ERROR:[/bold red] {exc}")
                sys.exit(1)
            for w in warnings:
                console.print(f"[yellow]WARNING:[/yellow] {w}")
    else:
        console.print("[dim]NOTE: State archive has no sequence metadata (legacy archive).[/dim]")

    # ── Run inference ─────────────────────────────────────────────────────────
    import shutil
    import tempfile
    from habitat_analysis.pipeline import HabitatPipeline

    use_zip = out.suffix == ".zip"
    if use_zip:
        _tmp_ctx = tempfile.TemporaryDirectory()
        out_dir = Path(_tmp_ctx.name) / "results"
    else:
        _tmp_ctx = None
        out_dir = out

    pipeline = HabitatPipeline(
        norm_config=norm_config,
        id_globber=id_globber,
    )
    outputs = pipeline.infer(
        seq_dirs=seq_dirs,
        mask_dir=mask_dir,
        state_path=state,
        out_dir=out_dir,
        visualize=not no_vis,
        skip_norm=skip_norm,
        max_cases=3 if debug else None,
        n_workers=workers,
    )

    if use_zip and _tmp_ctx is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.make_archive(str(out.with_suffix("")), "zip", out_dir)
        _tmp_ctx.cleanup()

    # ── Print results ─────────────────────────────────────────────────────────
    res_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    res_table.add_column(style="bold green", no_wrap=True)
    res_table.add_column()
    res_table.add_row("Cases written", str(len(outputs)))
    res_table.add_row("Output", str(out))

    console.print(Panel(res_table, title="[bold]Inference Complete", border_style="green"))


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
