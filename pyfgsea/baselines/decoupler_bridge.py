from __future__ import annotations

import numpy as np


def decoupler_ulm_scores(
    X,
    pathway_names: list[str],
    pathway_indices: list[list[int]],
) -> np.ndarray:
    """
    Lightweight decoupler-ULM-style fallback activity scores.

    If users have a full decoupler workflow, they can compare its activity
    matrix separately. This built-in bridge keeps PyFgsea baselines runnable
    without adding decoupler as a hard dependency.
    """
    if hasattr(X, "toarray"):
        arr = X.toarray()
    else:
        arr = np.asarray(X)
    arr = arr.astype(float, copy=False)
    gene_mean = np.nanmean(arr, axis=0)
    gene_sd = np.nanstd(arr, axis=0)
    z = (arr - gene_mean) / np.maximum(gene_sd, 1e-12)

    global_mean = np.nanmean(z, axis=1)
    out = np.zeros((z.shape[0], len(pathway_names)), dtype=float)
    for pathway_idx, indices in enumerate(pathway_indices):
        if not indices:
            continue
        in_score = np.nanmean(z[:, indices], axis=1)
        out[:, pathway_idx] = in_score - global_mean
    return out
