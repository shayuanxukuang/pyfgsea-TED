from collections import Counter
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _parse_gene_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        if not value:
            return []
        return [gene for gene in value.split(";") if gene]
    if isinstance(value, Iterable):
        return [str(gene) for gene in value if str(gene)]
    return []


def _format_gene_list(genes: Iterable[str]) -> str:
    return ";".join(str(gene) for gene in genes)


def _leading_edge_for_pathway(
    genes: np.ndarray,
    scores: np.ndarray,
    pathway_indices: list[int],
    gsea_param: float = 1.0,
) -> list[str]:
    if len(pathway_indices) == 0:
        return []

    scores = np.asarray(scores, dtype=np.float64)
    clean_scores = scores.copy()
    clean_scores[~np.isfinite(clean_scores)] = 0.0

    order = np.argsort(-clean_scores, kind="mergesort")
    ranked_scores = clean_scores[order]
    ranked_genes = genes[order]

    hit_mask_original = np.zeros(len(genes), dtype=bool)
    hit_mask_original[np.asarray(pathway_indices, dtype=int)] = True
    hits = hit_mask_original[order]

    n_hits = int(hits.sum())
    n_misses = len(hits) - n_hits
    if n_hits == 0 or n_misses == 0:
        return [str(gene) for gene in ranked_genes[hits]]

    hit_weights = np.abs(ranked_scores) ** gsea_param
    hit_weights = np.where(hits, hit_weights, 0.0)
    hit_weight_total = float(hit_weights.sum())
    if hit_weight_total <= 0:
        hit_weights = hits.astype(float)
        hit_weight_total = float(hit_weights.sum())

    running = np.cumsum(hit_weights / hit_weight_total)
    running -= np.cumsum((~hits).astype(float) / n_misses)

    peak_idx = int(np.argmax(running))
    trough_idx = int(np.argmin(running))
    if abs(running[peak_idx]) >= abs(running[trough_idx]):
        leading_mask = hits.copy()
        leading_mask[peak_idx + 1 :] = False
    else:
        leading_mask = hits.copy()
        leading_mask[:trough_idx] = False

    return [str(gene) for gene in ranked_genes[leading_mask]]


def compute_leading_edges(
    genes: np.ndarray,
    scores: np.ndarray,
    pathway_names: list[str],
    pathway_indices: list[list[int]],
    gsea_param: float = 1.0,
) -> dict[str, str]:
    """Compute semicolon-delimited leading-edge genes for each pathway."""
    genes = np.asarray(genes)
    return {
        name: _format_gene_list(
            _leading_edge_for_pathway(genes, scores, indices, gsea_param=gsea_param)
        )
        for name, indices in zip(pathway_names, pathway_indices)
    }


def leading_edge_dynamics(
    result: pd.DataFrame,
    pathway: Optional[str] = None,
    pathway_col: str = "Pathway",
    leading_edge_col: str = "leading_edge",
    time_col: str = "pt_mid",
    nes_col: str = "NES",
    core_fraction: float = 0.5,
) -> pd.DataFrame:
    """
    Summarize how leading-edge genes change across trajectory windows.

    ``run_trajectory_gsea(..., return_leading_edge=True)`` is required so that
    ``result`` contains a leading-edge column.
    """
    if result is None or result.empty:
        return pd.DataFrame()
    if pathway_col not in result.columns:
        raise ValueError(f"Missing pathway column '{pathway_col}'")
    if leading_edge_col not in result.columns:
        raise ValueError(
            f"Missing leading-edge column '{leading_edge_col}'. "
            "Run run_trajectory_gsea(..., return_leading_edge=True)."
        )
    if not (0 < core_fraction <= 1):
        raise ValueError("core_fraction must be in (0, 1]")

    df = result.copy()
    if pathway is not None:
        df = df[df[pathway_col] == pathway]
    if df.empty:
        return pd.DataFrame()

    sort_cols = [pathway_col]
    if time_col in df.columns:
        sort_cols.append(time_col)
    elif "window_id" in df.columns:
        sort_cols.append("window_id")
    df = df.sort_values(sort_cols)

    rows = []
    for pathway_name, group in df.groupby(pathway_col, sort=False):
        gene_sets = [_parse_gene_list(value) for value in group[leading_edge_col]]
        nonempty = [set(genes) for genes in gene_sets if genes]
        counter = Counter(gene for genes in nonempty for gene in genes)
        min_count = max(1, int(np.ceil(len(nonempty) * core_fraction)))
        core_genes = sorted(gene for gene, count in counter.items() if count >= min_count)
        core_gene_text = _format_gene_list(core_genes)

        previous = None
        for (_, row), genes in zip(group.iterrows(), gene_sets):
            current = set(genes)
            if previous is None:
                turnover = np.nan
            elif not previous and not current:
                turnover = 0.0
            else:
                turnover = 1.0 - (len(previous & current) / max(len(previous | current), 1))
            previous = current

            out = {
                pathway_col: pathway_name,
                "leading_edge_genes": _format_gene_list(genes),
                "leading_edge_size": len(genes),
                "core_genes": core_gene_text,
                "core_gene_count": len(core_genes),
                "turnover_score": turnover,
            }
            for col in ("window_id", time_col, nes_col, "padj"):
                if col in row.index:
                    out[col] = row[col]
            rows.append(out)

    return pd.DataFrame(rows)


get_dynamic_leading_edge = leading_edge_dynamics

