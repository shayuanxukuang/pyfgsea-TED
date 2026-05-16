import pandas as pd
import logging
from pathlib import Path
from typing import Any, Optional

from .preprocess.pseudotime import ensure_pseudotime
from .gsea.runner import run_core
from .anchors.select import select_anchor_pair
from .anchors.switch import find_switch_point
from .plotting.overview import plot_overview_heatmap
from .plotting.fastproof import plot_fastproof
from .windows.binning import build_anchor_matrix_from_df

logger = logging.getLogger(__name__)


def run_pipeline(
    adata: Any,
    gmt_path: str,
    output_dir: str = "results/run1",
    pseudotime_key: str = "dpt_pseudotime",
    force_rerun: bool = False,
    window_size: int = 800,
    step: int = 50,
    min_size: int = 15,
    ranker: str = "mean_diff",
    window_mode: str = "cell_count",
    min_cells: Optional[int] = None,
    max_cells: Optional[int] = None,
    target_span: Optional[float] = None,
    span_step: Optional[float] = None,
    return_leading_edge: bool = False,
    gsea_param: float = 1.0,
    layer: Optional[str] = None,
    use_raw: bool = False,
    dropna: bool = True,
    make_var_names_unique: bool = False,
    cell_weight_key: Optional[str] = None,
    graph_key: str = "connectivities",
    graph_radius: int = 2,
    bandwidth_pt: Optional[float] = None,
    bandwidth_graph: Optional[float] = None,
    branch_key: Optional[str] = None,
    min_branch_purity: float = 0.75,
    experimental: bool = False,
) -> pd.DataFrame:
    """
    Orchestrates the full GSEA trajectory analysis pipeline.

    Args:
        adata: AnnData object or path to .h5ad
        gmt_path: Path to GMT file
        output_dir: Directory to save results
        pseudotime_key: Column name in adata.obs for pseudotime
        force_rerun: If True, ignore cached results and re-run GSEA
        ranker: Gene-level statistic used to rank genes per window
        window_mode: Window construction mode: cell_count, pseudotime_span,
            adaptive, or experimental graph_adaptive
        return_leading_edge: If True, include semicolon-delimited leading-edge genes
        layer: Optional AnnData layer to use instead of adata.X
        use_raw: Use adata.raw.X when layer is not provided
        graph_key: AnnData obsp key for graph_adaptive windows
        branch_key: Optional obs key used to skip low-purity graph windows
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Data Prep
    adata = ensure_pseudotime(adata, key=pseudotime_key)

    # Step 2: Compute or Load GSEA
    gsea_csv = out_path / "gsea_results_core.csv"
    if gsea_csv.exists() and not force_rerun:
        logger.info(f"Loading cached results from {gsea_csv}")
        gsea_df = pd.read_csv(gsea_csv)
    else:
        logger.info("Running core GSEA analysis...")
        gsea_df = run_core(
            adata,
            gmt_path,
            out_csv=str(gsea_csv),
            pseudotime_key=pseudotime_key,
            window_size=window_size,
            step=step,
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

    if gsea_df.empty:
        logger.warning("GSEA yielded no results. Check gene coverage or GMT file.")
        return gsea_df

    # Step 3: Analysis & Visualization
    _analyze_anchors_and_plot(gsea_df, out_path)

    return gsea_df


def _analyze_anchors_and_plot(df: pd.DataFrame, out_path: Path):
    """Internal helper to handle anchor selection and plotting."""
    early, late, score, stats = select_anchor_pair(
        df, out_report=str(out_path / "anchor_report.csv")
    )

    if not (early and late):
        logger.warning("Skipping plots: No valid anchor pair found.")
        return

    logger.info(f"Best Pair: {early} vs {late} (Score={score:.3f})")
    logger.info(
        f"Stats: Corr={stats['Corr']:.3f}, Sep={stats['Sep']:.3f}, Range={stats['Range']:.3f}"
    )

    # 4. Plots
    plot_overview_heatmap(df, str(out_path))
    plot_fastproof(df, early, late, str(out_path))

    # 5. Switch Point
    try:
        import numpy as np

        b_grid = np.linspace(0, 1, 61)
        mat, centers = build_anchor_matrix_from_df(
            df, [early, late], bins=b_grid, value_col="NES_smooth"
        )
        pt_switch, _ = find_switch_point(
            mat.loc[early].values, mat.loc[late].values, centers
        )
        logger.info(f"Switch Point: {pt_switch:.3f}")
    except Exception as e:
        logger.error(f"Failed to calculate switch point: {e}")


# Maintain backward compatibility alias
run = run_pipeline
