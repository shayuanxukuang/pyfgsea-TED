import click
from ..api import run as run_pipeline
from ..io.anndata_io import load_adata
from ..io.meta_merge import merge_metadata_safe
from ..ted_mad.cli import cli as ted_mad_cli


@click.group()
def cli():
    pass


cli.add_command(ted_mad_cli, name="ted-mad")


@cli.command()
@click.option("--h5ad", required=True, help="Path to .h5ad file")
@click.option("--gmt", required=True, help="Path to .gmt file")
@click.option("--out", default="results", help="Output directory")
@click.option(
    "--pseudotime-key", default="dpt_pseudotime", help="Key for pseudotime in adata.obs"
)
@click.option("--meta", default=None, help="Optional metadata CSV to merge")
@click.option("--window-size", default=800, show_default=True, type=int)
@click.option("--step", default=50, show_default=True, type=int)
@click.option(
    "--ranker",
    default="mean_diff",
    show_default=True,
    type=click.Choice(
        [
            "mean_diff",
            "wilcoxon",
            "t_stat",
            "z_score",
            "cohens_d",
            "detection_weighted",
            "local_slope",
            "neighbor_contrast",
        ]
    ),
    help="Gene ranking statistic for each trajectory window",
)
@click.option(
    "--window-mode",
    default="cell_count",
    show_default=True,
    type=click.Choice(["cell_count", "pseudotime_span", "adaptive", "graph_adaptive"]),
)
@click.option("--min-cells", default=None, type=int)
@click.option("--max-cells", default=None, type=int)
@click.option("--target-span", default=None, type=float)
@click.option("--span-step", default=None, type=float)
@click.option("--cell-weight-key", default=None, help="Optional obs column of non-negative cell weights")
@click.option("--graph-key", default="connectivities", show_default=True, help="obsp graph key for graph_adaptive windows")
@click.option("--graph-radius", default=2, show_default=True, type=int)
@click.option("--bandwidth-pt", default=None, type=float)
@click.option("--bandwidth-graph", default=None, type=float)
@click.option("--branch-key", default=None, help="Optional obs column for branch purity diagnostics")
@click.option("--min-branch-purity", default=0.75, show_default=True, type=float)
@click.option("--experimental", is_flag=True, help="Mark experimental graph-aware window analyses")
@click.option(
    "--return-leading-edge",
    is_flag=True,
    help="Include leading-edge genes for each pathway and window",
)
@click.option("--layer", default=None, help="AnnData layer to use instead of X")
@click.option("--use-raw", is_flag=True, help="Use adata.raw.X when --layer is not provided")
@click.option(
    "--fail-on-missing-pseudotime",
    is_flag=True,
    help="Fail instead of dropping cells with non-finite pseudotime",
)
@click.option(
    "--make-var-names-unique",
    is_flag=True,
    help="Make duplicated gene names unique for this run",
)
@click.option(
    "--allow-positional-merge",
    is_flag=True,
    help="Allow merging metadata by position (DANGEROUS)",
)
def run(
    h5ad,
    gmt,
    out,
    pseudotime_key,
    meta,
    window_size,
    step,
    ranker,
    window_mode,
    min_cells,
    max_cells,
    target_span,
    span_step,
    cell_weight_key,
    graph_key,
    graph_radius,
    bandwidth_pt,
    bandwidth_graph,
    branch_key,
    min_branch_purity,
    experimental,
    return_leading_edge,
    layer,
    use_raw,
    fail_on_missing_pseudotime,
    make_var_names_unique,
    allow_positional_merge,
):
    """Run the Universal Trajectory GSEA pipeline."""
    print(f"Loading {h5ad}...")
    adata = load_adata(h5ad)

    if meta:
        print(f"Merging metadata from {meta}...")
        adata = merge_metadata_safe(
            adata, meta, allow_positional_merge=allow_positional_merge
        )

    run_pipeline(
        adata,
        gmt_path=gmt,
        pseudotime_key=pseudotime_key,
        output_dir=out,
        window_size=window_size,
        step=step,
        ranker=ranker,
        window_mode=window_mode,
        min_cells=min_cells,
        max_cells=max_cells,
        target_span=target_span,
        span_step=span_step,
        cell_weight_key=cell_weight_key,
        graph_key=graph_key,
        graph_radius=graph_radius,
        bandwidth_pt=bandwidth_pt,
        bandwidth_graph=bandwidth_graph,
        branch_key=branch_key,
        min_branch_purity=min_branch_purity,
        experimental=experimental,
        return_leading_edge=return_leading_edge,
        layer=layer,
        use_raw=use_raw,
        dropna=not fail_on_missing_pseudotime,
        make_var_names_unique=make_var_names_unique,
    )


if __name__ == "__main__":
    cli()
