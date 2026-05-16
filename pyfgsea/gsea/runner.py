from ..trajectory import run_trajectory_gsea
import pandas as pd
from .smooth import smooth_nes


def run_core(
    adata,
    gmt_path,
    out_csv=None,
    pseudotime_key="dpt_pseudotime",
    window_size=800,
    step=50,
    nperm=1000,
    smooth=True,
    min_size=15,
    ranker="mean_diff",
    window_mode="cell_count",
    min_cells=None,
    max_cells=None,
    target_span=None,
    span_step=None,
    return_leading_edge=False,
    gsea_param=1.0,
    layer=None,
    use_raw=False,
    dropna=True,
    make_var_names_unique=False,
    cell_weight_key=None,
    graph_key="connectivities",
    graph_radius=2,
    bandwidth_pt=None,
    bandwidth_graph=None,
    branch_key=None,
    min_branch_purity=0.75,
    experimental=False,
):
    print(
        f"[Core] Running Trajectory GSEA (Window={window_size}, Step={step}, "
        f"MinSize={min_size}, Ranker={ranker}, WindowMode={window_mode})..."
    )

    df = run_trajectory_gsea(
        adata,
        gmt_path=gmt_path,
        root_gene=None,
        window_size=window_size,
        step=step,
        out_csv=out_csv,
        nperm_nes=nperm,
        pseudotime_key=pseudotime_key,
        min_size=min_size,
        ranker=ranker,
        window_mode=window_mode,
        min_cells=min_cells,
        max_cells=max_cells,
        target_span=target_span,
        span_step=span_step,
        return_leading_edge=return_leading_edge,
        gsea_param=gsea_param,
        layer=layer,
        use_raw=use_raw,
        dropna=dropna,
        make_var_names_unique=make_var_names_unique,
        cell_weight_key=cell_weight_key,
        graph_key=graph_key,
        graph_radius=graph_radius,
        bandwidth_pt=bandwidth_pt,
        bandwidth_graph=bandwidth_graph,
        branch_key=branch_key,
        min_branch_purity=min_branch_purity,
        experimental=experimental,
    )

    if df is None or df.empty:
        print("  [Error] No results returned.")
        return pd.DataFrame()

    if smooth:
        print("  - Smoothing NES curves...")
        df = smooth_nes(df)

    return df
