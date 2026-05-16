from __future__ import annotations

import numpy as np

from .rank_auc import rank_auc_scores


def _mean_zscore_scores(X, pathway_names: list[str], pathway_indices: list[list[int]]) -> np.ndarray:
    if hasattr(X, "toarray"):
        arr = X.toarray()
    else:
        arr = np.asarray(X)
    arr = arr.astype(float, copy=False)
    gene_mean = np.nanmean(arr, axis=0)
    gene_sd = np.nanstd(arr, axis=0)
    z = (arr - gene_mean) / np.maximum(gene_sd, 1e-12)
    out = np.zeros((z.shape[0], len(pathway_names)), dtype=float)
    for pathway_idx, indices in enumerate(pathway_indices):
        if indices:
            out[:, pathway_idx] = np.nanmean(z[:, indices], axis=1)
    return out


def gsva_like_scores(X, pathway_names: list[str], pathway_indices: list[list[int]]) -> np.ndarray:
    """
    Lightweight GSVA-style fallback.

    This is intentionally a baseline bridge, not a reimplementation of GSVA.
    It returns sample-wise standardized pathway activity with the same shape
    expected by the score-then-smooth runner.
    """
    return _mean_zscore_scores(X, pathway_names, pathway_indices)


def ssgsea_like_scores(
    X,
    pathway_names: list[str],
    pathway_indices: list[list[int]],
    auc_top_fraction: float = 1.0,
) -> np.ndarray:
    """Rank-based ssGSEA-style fallback using the built-in rank-AUC scorer."""
    return rank_auc_scores(
        X,
        pathway_names=pathway_names,
        pathway_indices=pathway_indices,
        auc_top_fraction=auc_top_fraction,
    )
