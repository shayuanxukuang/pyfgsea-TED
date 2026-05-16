from __future__ import annotations

import numpy as np


def _dense_row(block, row_idx: int) -> np.ndarray:
    row = block[row_idx]
    if hasattr(row, "toarray"):
        return np.asarray(row.toarray()).ravel()
    return np.asarray(row).ravel()


def rank_auc_scores(
    X,
    pathway_names: list[str],
    pathway_indices: list[list[int]],
    auc_top_fraction: float = 0.05,
) -> np.ndarray:
    """
    AUCell-style rank-AUC baseline scores per cell and pathway.

    Scores are based only on within-cell expression ranks. For each gene set,
    the score is the fraction of top-ranked genes recovered, weighted by how
    close they are to the top of the expression ranking.
    """
    if not 0 < auc_top_fraction <= 1:
        raise ValueError("auc_top_fraction must be in (0, 1]")
    n_cells, n_genes = X.shape
    max_rank = max(1, int(round(n_genes * auc_top_fraction)))
    pathway_sets = [set(map(int, indices)) for indices in pathway_indices]
    out = np.zeros((n_cells, len(pathway_names)), dtype=float)

    for cell_idx in range(n_cells):
        values = _dense_row(X, cell_idx)
        values = np.nan_to_num(values, nan=-np.inf)
        order = np.argsort(-values, kind="mergesort")[:max_rank]
        rank_weight = 1.0 - (np.arange(len(order), dtype=float) / max(max_rank, 1))
        for pathway_idx, gene_set in enumerate(pathway_sets):
            if not gene_set:
                continue
            hits = np.fromiter((gene in gene_set for gene in order), dtype=bool)
            if hits.any():
                denom = min(len(gene_set), max_rank)
                out[cell_idx, pathway_idx] = float(rank_weight[hits].sum() / denom)
    return out
