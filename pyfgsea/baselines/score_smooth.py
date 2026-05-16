from __future__ import annotations

from typing import Optional
import time

import numpy as np
import pandas as pd

from ..trajectory import _make_windows, _prepare_gene_sets_for_mode
from ..validation import _expression_matrix
from ..wrapper import prepare_pathways
from .decoupler_bridge import decoupler_ulm_scores
from .gsva_bridge import gsva_like_scores, ssgsea_like_scores
from .rank_auc import rank_auc_scores


def _matrix_to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def _mean_zscore_scores(X, pathway_names, pathway_indices):
    arr = _matrix_to_dense(X).astype(float, copy=False)
    gene_mean = np.nanmean(arr, axis=0)
    gene_sd = np.nanstd(arr, axis=0)
    z = (arr - gene_mean) / np.maximum(gene_sd, 1e-12)
    out = np.zeros((z.shape[0], len(pathway_names)), dtype=float)
    for pathway_idx, indices in enumerate(pathway_indices):
        if indices:
            out[:, pathway_idx] = np.nanmean(z[:, indices], axis=1)
    return out


def _score_cells(
    X,
    pathway_names,
    pathway_indices,
    method: str,
    auc_top_fraction: float,
):
    method = method.lower().replace("-", "_")
    if method == "rank_auc":
        return rank_auc_scores(X, pathway_names, pathway_indices, auc_top_fraction)
    if method == "mean_zscore":
        return _mean_zscore_scores(X, pathway_names, pathway_indices)
    if method == "decoupler_ulm":
        return decoupler_ulm_scores(X, pathway_names, pathway_indices)
    if method == "gsva":
        return gsva_like_scores(X, pathway_names, pathway_indices)
    if method == "ssgsea":
        return ssgsea_like_scores(X, pathway_names, pathway_indices, auc_top_fraction=1.0)
    raise ValueError(
        "method must be one of rank_auc, mean_zscore, decoupler_ulm, gsva, or ssgsea"
    )


def _zscore_by_pathway(window_table: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for pathway, group in window_table.groupby("Pathway", sort=False):
        group = group.sort_values("pt_mid").copy()
        values = group["activity_score"].to_numpy(dtype=float)
        mean = float(np.nanmean(values)) if len(values) else 0.0
        sd = float(np.nanstd(values)) if len(values) else 0.0
        group["activity_z"] = (values - mean) / max(sd, 1e-12)
        frames.append(group)
    if not frames:
        return window_table.assign(activity_z=np.nan)
    return pd.concat(frames, ignore_index=True)


def _smooth_group(time_values: np.ndarray, values: np.ndarray, smoother: str, frac: float) -> np.ndarray:
    smoother = smoother.lower().replace("-", "_")
    if len(values) < 3 or smoother == "rolling":
        return values

    if smoother == "lowess":
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess

            return lowess(values, time_values, frac=frac, return_sorted=False)
        except Exception:
            return values

    if smoother == "spline":
        try:
            from scipy.interpolate import UnivariateSpline

            order = np.argsort(time_values)
            x = time_values[order]
            y = values[order]
            k = min(3, len(np.unique(x)) - 1)
            if k < 1:
                return values
            spline = UnivariateSpline(x, y, k=k, s=len(x) * np.nanvar(y) * 0.1)
            smoothed = np.empty_like(values, dtype=float)
            smoothed[order] = spline(x)
            return smoothed
        except Exception:
            return values

    raise ValueError("smoother must be one of rolling, lowess, or spline")


def _apply_smoother(window_table: pd.DataFrame, smoother: str, lowess_frac: float) -> pd.DataFrame:
    if window_table.empty:
        return window_table
    frames = []
    for _pathway, group in window_table.groupby("Pathway", sort=False):
        group = group.sort_values("pt_mid").copy()
        group["activity_score_raw"] = group["activity_score"].to_numpy(dtype=float)
        group["activity_score"] = _smooth_group(
            group["pt_mid"].to_numpy(dtype=float),
            group["activity_score"].to_numpy(dtype=float),
            smoother=smoother,
            frac=lowess_frac,
        )
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def run_score_then_smooth_baseline(
    adata,
    gene_sets,
    pseudotime_key: str = "dpt_pseudotime",
    method: str = "rank_auc",
    smoother: str = "rolling",
    window_size: int = 500,
    step: int = 100,
    window_mode: str = "cell_count",
    min_cells: Optional[int] = None,
    max_cells: Optional[int] = None,
    target_span: Optional[float] = None,
    span_step: Optional[float] = None,
    min_size: int = 15,
    max_size: int = 500,
    layer: Optional[str] = None,
    use_raw: bool = False,
    auc_top_fraction: float = 0.05,
    lowess_frac: float = 0.35,
    gene_set_mode: str = "standard",
    min_abs_gene_weight: float = 0.0,
) -> pd.DataFrame:
    """
    Run score-then-smooth pathway activity baselines along pseudotime.

    These baselines are intended as external consistency checks for TED events,
    not as replacements for fgsea-style trajectory event discovery.
    """
    if pseudotime_key not in adata.obs:
        raise ValueError(f"pseudotime_key '{pseudotime_key}' not found in adata.obs")

    started = time.time()
    pt = pd.to_numeric(adata.obs[pseudotime_key], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(pt)
    if not finite.all():
        adata = adata[finite].copy()
        pt = pt[finite]

    X, genes, expression_source = _expression_matrix(adata, layer=layer, use_raw=use_raw)
    gene_sets = _prepare_gene_sets_for_mode(
        gene_sets,
        gene_set_mode=gene_set_mode,
        min_abs_gene_weight=min_abs_gene_weight,
    )
    pathway_names, pathway_indices = prepare_pathways(genes, gene_sets, min_size, max_size)
    if not pathway_indices:
        return pd.DataFrame()

    cell_scores = _score_cells(
        X,
        pathway_names,
        pathway_indices,
        method=method,
        auc_top_fraction=auc_top_fraction,
    )
    order = np.argsort(pt)
    windows = _make_windows(
        order,
        window_size=window_size,
        step=step,
        pt=pt,
        window_mode=window_mode,
        min_cells=min_cells,
        max_cells=max_cells,
        target_span=target_span,
        span_step=span_step,
    )

    rows = []
    for window_id, (_start, _end, window_indices) in enumerate(windows):
        pt_vals = pt[window_indices]
        activity = np.nanmean(cell_scores[window_indices, :], axis=0)
        for pathway_idx, pathway in enumerate(pathway_names):
            rows.append(
                {
                    "Pathway": pathway,
                    "pathway": pathway,
                    "window_id": window_id,
                    "pt_start": float(np.nanmin(pt_vals)),
                    "pt_end": float(np.nanmax(pt_vals)),
                    "pt_mid": float((np.nanmin(pt_vals) + np.nanmax(pt_vals)) / 2.0),
                    "window_midpoint": float((np.nanmin(pt_vals) + np.nanmax(pt_vals)) / 2.0),
                    "n_cells": int(len(window_indices)),
                    "activity_score": float(activity[pathway_idx]),
                    "baseline_method": method,
                    "smoother": smoother,
                    "window_mode": window_mode,
                    "expression_source": expression_source,
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = _apply_smoother(result, smoother=smoother, lowess_frac=lowess_frac)
    result = _zscore_by_pathway(result)
    result["NES"] = result["activity_z"]
    result["ES"] = result["activity_score"]
    result.attrs["baseline"] = {
        "method": method,
        "smoother": smoother,
        "window_mode": window_mode,
        "runtime_seconds": float(time.time() - started),
        "role": "score_then_smooth_baseline",
    }
    return result.sort_values(["Pathway", "window_id"]).reset_index(drop=True)
