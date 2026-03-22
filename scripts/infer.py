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
    "--img_dir", default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Single-sequence image directory (alias for --seq image:DIR).",
)
@click.option(
    "--mask_dir", required=True,
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
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output directory for segmentations and overlays.",
)
@click.option(
    "--norm_config", default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="mnts normalisation YAML (optional override).",
)
@click.option("--no_vis", is_flag=True, help="Disable PNG overlay generation.")
@click.option(
    "--override", is_flag=True,
    help="Skip sequence validation against training metadata.",
)
@click.option("--id_globber", default=r"^[0-9a-zA-Z]+", show_default=True,
              help="Regex to extract case IDs from filenames.")
@click.option("--verbose", "-v", is_flag=True, help="Enable DEBUG logging.")
def main(seq_tokens, img_dir, mask_dir, state, out, norm_config,
         no_vis, override, id_globber, verbose):
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

    primary_img_dir = next(iter(seq_dirs.values()))

    click.echo(f"Sequences            : {', '.join(f'{k}={v}' for k, v in seq_dirs.items())}")
    click.echo(f"State archive        : {state}")

    # ── Sequence validation ───────────────────────────────────────────────────
    from habitat_analysis.state import HabitatState

    loaded_state = HabitatState.load(state)
    required = loaded_state.required_sequences

    if required:
        click.echo(f"Required sequences   : {required}")
        if override:
            click.echo("WARNING: --override set, skipping sequence validation.", err=True)
        else:
            try:
                warnings = loaded_state.validate_sequences(list(seq_dirs.keys()))
            except ValueError as exc:
                click.echo(f"ERROR: {exc}", err=True)
                sys.exit(1)
            for w in warnings:
                click.echo(f"WARNING: {w}", err=True)
    else:
        click.echo("NOTE: State archive has no sequence metadata (legacy archive).")

    # ── Run inference ─────────────────────────────────────────────────────────
    from habitat_analysis.pipeline import HabitatPipeline

    pipeline = HabitatPipeline(
        norm_config=norm_config,
        id_globber=id_globber,
    )
    outputs = pipeline.infer(
        img_dir=primary_img_dir,
        mask_dir=mask_dir,
        state_path=state,
        out_dir=out,
        visualize=not no_vis,
    )
    click.echo(f"Written {len(outputs)} output files to {out}")


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
