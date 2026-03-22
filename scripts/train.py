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
    "--out", required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output state archive path (.zip).",
)
@click.option(
    "--norm_config", default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="mnts normalisation YAML (default: bundled).",
)
@click.option(
    "--pyrad_config", default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="PyRadiomics filter YAML (default: bundled).",
)
@click.option(
    "--method", default="kmeans",
    type=click.Choice(["kmeans", "gmm"]),
    help="Clustering algorithm.",
)
@click.option("--k_min", default=2, type=int, show_default=True,
              help="Minimum number of clusters to evaluate.")
@click.option("--k_max", default=6, type=int, show_default=True,
              help="Maximum number of clusters to evaluate (inclusive).")
@click.option("--subsample", default=200_000, type=int, show_default=True,
              help="Max voxels for clustering fit (0 = use all).")
@click.option("--seed", default=42, type=int, show_default=True,
              help="Random seed.")
@click.option("--id_globber", default=r"^[0-9a-zA-Z]+", show_default=True,
              help="Regex to extract case IDs from filenames.")
@click.option("--verbose", "-v", is_flag=True, help="Enable DEBUG logging.")
def main(seq_tokens, img_dir, mask_dir, out, norm_config, pyrad_config,
         method, k_min, k_max, subsample, seed, id_globber, verbose):
    """Train the habitat analysis pipeline and save the state archive."""
    from mnts.mnts_logger import MNTSLogger
    MNTSLogger("habitat_train.log", logger_name="habitat_train",
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

    if not seq_dirs:
        raise click.UsageError("No valid sequence directories were parsed.")

    primary_img_dir = next(iter(seq_dirs.values()))
    seq_names = list(seq_dirs.keys())

    click.echo(f"Sequences            : {', '.join(f'{k}={v}' for k, v in seq_dirs.items())}")
    click.echo(f"Mask directory       : {mask_dir}")
    click.echo(f"Output state         : {out}")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    from habitat_analysis.pipeline import HabitatPipeline

    pipeline = HabitatPipeline(
        norm_config=norm_config,
        pyrad_config=pyrad_config,
        id_globber=id_globber,
        cluster_method=method,
        k_range=range(k_min, k_max + 1),
        subsample=subsample if subsample > 0 else None,
        random_state=seed,
    )
    state = pipeline.train(
        img_dir=primary_img_dir,
        mask_dir=mask_dir,
        out_state=out,
        extra_metadata={"sequences": seq_names},
    )

    click.echo(f"\nState archive saved  : {out}")
    click.echo(f"Required sequences   : {state.required_sequences}")
    click.echo(f"Best k               : {state.metadata.get('best_k')}")


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
