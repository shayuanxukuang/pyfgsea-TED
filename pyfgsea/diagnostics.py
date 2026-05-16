"""Diagnostics for technical confounding in trajectory pathway events."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _pathway_column(df: pd.DataFrame) -> str:
    if "Pathway" in df.columns:
        return "Pathway"
    if "pathway" in df.columns:
        return "pathway"
    raise KeyError("Expected a Pathway or pathway column")


def _corr(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def add_window_detection_metrics(
    result: pd.DataFrame,
    adata,
    *,
    pseudotime_key: str = "dpt_pseudotime",
    detection_key: str = "cell_detection_rate",
) -> pd.DataFrame:
    """Attach mean/median detection rate diagnostics to window-level results."""

    if pseudotime_key not in adata.obs:
        raise KeyError(f"pseudotime_key '{pseudotime_key}' is not present in adata.obs")

    obs = adata.obs.copy()
    if detection_key not in obs:
        matrix = adata.X
        if hasattr(matrix, "getnnz"):
            detected = np.asarray(matrix.getnnz(axis=1), dtype=float)
        else:
            detected = np.asarray((np.asarray(matrix) > 0).sum(axis=1), dtype=float)
        obs[detection_key] = detected / max(1, int(adata.n_vars))

    if not {"pt_start", "pt_end"}.issubset(result.columns):
        raise KeyError("result must contain pt_start and pt_end columns")

    pt = obs[pseudotime_key].to_numpy(dtype=float)
    det = obs[detection_key].to_numpy(dtype=float)
    unique_windows = result[["window_id", "pt_start", "pt_end"]].drop_duplicates()
    rows = []
    for row in unique_windows.itertuples(index=False):
        mask = (pt >= float(row.pt_start)) & (pt <= float(row.pt_end))
        values = det[mask]
        rows.append(
            {
                "window_id": row.window_id,
                "mean_cell_detection_rate": float(np.nanmean(values))
                if values.size
                else np.nan,
                "median_cell_detection_rate": float(np.nanmedian(values))
                if values.size
                else np.nan,
                "window_metric_n_cells": int(values.size),
            }
        )

    metrics = pd.DataFrame(rows)
    return result.merge(metrics, on="window_id", how="left")


def technical_confound_diagnostics(
    result: pd.DataFrame,
    adata=None,
    *,
    pseudotime_key: str = "dpt_pseudotime",
    detection_key: str = "cell_detection_rate",
    fdr_threshold: float = 0.05,
    low_detection_quantile: float = 0.25,
) -> pd.DataFrame:
    """Summarize whether pathway events track low-detection windows."""

    work = result.copy()
    if "mean_cell_detection_rate" not in work.columns:
        if adata is None:
            raise KeyError(
                "result needs mean_cell_detection_rate or adata must be provided"
            )
        work = add_window_detection_metrics(
            work,
            adata,
            pseudotime_key=pseudotime_key,
            detection_key=detection_key,
        )

    pathway_col = _pathway_column(work)
    q_col = "padj" if "padj" in work.columns else "pval"
    if q_col not in work.columns:
        raise KeyError("result must contain padj or pval")

    det = work["mean_cell_detection_rate"].to_numpy(dtype=float)
    low_threshold = float(np.nanquantile(det, low_detection_quantile))
    rows = []
    for pathway, group in work.groupby(pathway_col, sort=False):
        nes = group["NES"].to_numpy(dtype=float)
        detection = group["mean_cell_detection_rate"].to_numpy(dtype=float)
        padj = group[q_col].to_numpy(dtype=float)
        sig = np.isfinite(padj) & (padj <= fdr_threshold)
        low = detection <= low_threshold
        n_sig = int(sig.sum())
        n_low_sig = int((sig & low).sum())
        rows.append(
            {
                "Pathway": pathway,
                "nes_detection_corr": _corr(nes, detection),
                "abs_nes_detection_corr": _corr(np.abs(nes), detection),
                "n_windows": int(len(group)),
                "n_significant_windows": n_sig,
                "n_low_detection_significant_windows": n_low_sig,
                "low_detection_sig_fraction": float(n_low_sig / n_sig)
                if n_sig
                else 0.0,
                "min_window_fdr": float(np.nanmin(padj))
                if np.isfinite(padj).any()
                else np.nan,
                "median_detection_rate": float(np.nanmedian(detection)),
                "low_detection_threshold": low_threshold,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        out.attrs["summary"] = {
            "technical_confound_score": 0.0,
            "n_significant_windows": 0,
            "n_low_detection_significant_windows": 0,
            "low_detection_sig_fraction": 0.0,
            "median_abs_nes_detection_corr": np.nan,
        }
        return out

    total_sig = int(out["n_significant_windows"].sum())
    total_low_sig = int(out["n_low_detection_significant_windows"].sum())
    low_fraction = float(total_low_sig / total_sig) if total_sig else 0.0
    median_abs_corr = float(np.nanmedian(np.abs(out["abs_nes_detection_corr"])))
    confound_score = float(np.nanmean([low_fraction, median_abs_corr]))
    out.attrs["summary"] = {
        "technical_confound_score": confound_score,
        "n_significant_windows": total_sig,
        "n_low_detection_significant_windows": total_low_sig,
        "low_detection_sig_fraction": low_fraction,
        "median_abs_nes_detection_corr": median_abs_corr,
    }
    return out
