from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


_OUTPUT_FILENAMES = {
    "trajectory_alignment_functions": "trajectory_alignment_functions.tsv",
    "alignment_anchor_pathways": "alignment_anchor_pathways.tsv",
    "aligned_pathway_score_process": "aligned_pathway_score_process.tsv",
    "differential_event_table": "differential_event_table.tsv",
    "differential_event_fdr": "differential_event_fdr.tsv",
    "alignment_sensitivity_report": "alignment_sensitivity_report.tsv",
}


def _pick_column(df: pd.DataFrame, *candidates: Optional[str]) -> Optional[str]:
    lower_to_original = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate is None:
            continue
        if candidate in df.columns:
            return candidate
        lower = str(candidate).lower()
        if lower in lower_to_original:
            return lower_to_original[lower]
    return None


def _numeric(values) -> pd.Series:
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _trapz(values: np.ndarray, x: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(values) & np.isfinite(x)
    if finite.sum() < 2:
        return 0.0
    values = values[finite]
    x = x[finite]
    order = np.argsort(x)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values[order], x=x[order]))
    return float(np.trapz(values[order], x=x[order]))


def _bh_adjust(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(list(values), dtype=float)
    out = np.full(len(p), np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    pf = p[finite]
    order = np.argsort(pf)
    ranked = pf[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[finite] = restored
    return out


def _condition_values(results: pd.DataFrame, condition_col: str) -> list[str]:
    values = pd.Series(results[condition_col]).dropna().astype(str).unique().tolist()
    return sorted(values, key=str)


def _prepare_condition_process(
    results: pd.DataFrame,
    *,
    condition_col: str,
    condition,
    pathway_col: str,
    time_col: str,
    score_col: str,
    fdr_col: Optional[str],
) -> tuple[pd.DataFrame, tuple[float, float]]:
    sub = results[results[condition_col].astype(str) == str(condition)].copy()
    if sub.empty:
        raise ValueError(f"No rows found for condition '{condition}'")

    work = pd.DataFrame(
        {
            "pathway": sub[pathway_col].astype(str).to_numpy(),
            "time_raw": pd.to_numeric(sub[time_col], errors="coerce").to_numpy(dtype=float),
            "score": pd.to_numeric(sub[score_col], errors="coerce").to_numpy(dtype=float),
        }
    )
    if fdr_col is not None and fdr_col in sub.columns:
        work["window_q"] = pd.to_numeric(sub[fdr_col], errors="coerce").to_numpy(dtype=float)
    else:
        work["window_q"] = np.nan
    if "window_id" in sub.columns:
        work["window_id"] = pd.to_numeric(sub["window_id"], errors="coerce").to_numpy(dtype=float)

    work = work[np.isfinite(work["time_raw"]) & np.isfinite(work["score"])].copy()
    if work.empty:
        raise ValueError(f"Condition '{condition}' has no finite score/time rows")

    t_min = float(work["time_raw"].min())
    t_max = float(work["time_raw"].max())
    span = t_max - t_min
    if span <= 0:
        work["time_norm"] = 0.0
    else:
        work["time_norm"] = (work["time_raw"] - t_min) / span

    agg = {"score": "mean", "time_raw": "mean", "window_q": "min"}
    if "window_id" in work.columns:
        agg["window_id"] = "first"
    process = (
        work.groupby(["pathway", "time_norm"], as_index=False)
        .agg(agg)
        .sort_values(["pathway", "time_norm"])
        .reset_index(drop=True)
    )
    return process, (t_min, t_max)


def _norm_to_raw(time_norm: np.ndarray, time_range: tuple[float, float]) -> np.ndarray:
    t_min, t_max = time_range
    return float(t_min) + np.asarray(time_norm, dtype=float) * (float(t_max) - float(t_min))


def _pathway_curve(process: pd.DataFrame, pathway: str, value_col: str = "score") -> pd.DataFrame:
    curve = process[process["pathway"].astype(str) == str(pathway)][["time_norm", value_col]].copy()
    curve[value_col] = pd.to_numeric(curve[value_col], errors="coerce")
    curve["time_norm"] = pd.to_numeric(curve["time_norm"], errors="coerce")
    curve = curve[np.isfinite(curve["time_norm"]) & np.isfinite(curve[value_col])]
    if curve.empty:
        return curve
    return (
        curve.groupby("time_norm", as_index=False)[value_col]
        .mean()
        .sort_values("time_norm")
        .reset_index(drop=True)
    )


def _interp_curve(
    process: pd.DataFrame,
    pathway: str,
    x: np.ndarray,
    value_col: str = "score",
) -> np.ndarray:
    curve = _pathway_curve(process, pathway, value_col=value_col)
    x = np.asarray(x, dtype=float)
    if curve.empty:
        return np.full(len(x), np.nan)
    times = curve["time_norm"].to_numpy(dtype=float)
    values = curve[value_col].to_numpy(dtype=float)
    if len(times) == 1:
        return np.full(len(x), values[0], dtype=float)
    return np.interp(x, times, values, left=values[0], right=values[-1])


def _available_anchor_pathways(
    process_a: pd.DataFrame,
    process_b: pd.DataFrame,
    anchor_pathways: Optional[Sequence[str]],
    *,
    max_auto_anchors: int = 25,
) -> list[str]:
    common = sorted(set(process_a["pathway"].astype(str)) & set(process_b["pathway"].astype(str)))
    if anchor_pathways is None:
        return common[:max_auto_anchors]
    requested = [str(pathway) for pathway in anchor_pathways]
    return [pathway for pathway in requested if pathway in common]


def _anchor_matrix(process: pd.DataFrame, times: np.ndarray, anchors: Sequence[str]) -> np.ndarray:
    cols = [_interp_curve(process, pathway, times, value_col="score") for pathway in anchors]
    if not cols:
        return np.empty((len(times), 0), dtype=float)
    return np.vstack(cols).T.astype(float)


def _standardize_anchor_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()
    for col in range(a.shape[1]):
        combined = np.concatenate([a[:, col], b[:, col]])
        finite = combined[np.isfinite(combined)]
        if len(finite) == 0:
            a[:, col] = 0.0
            b[:, col] = 0.0
            continue
        center = float(np.nanmedian(finite))
        q75, q25 = np.nanpercentile(finite, [75, 25])
        scale = float(q75 - q25) / 1.349 if q75 > q25 else float(np.nanstd(finite))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        a[:, col] = (a[:, col] - center) / scale
        b[:, col] = (b[:, col] - center) / scale
        a[~np.isfinite(a[:, col]), col] = 0.0
        b[~np.isfinite(b[:, col]), col] = 0.0
    return a, b


def _dtw_path(cost: np.ndarray, gap_penalty: float) -> list[tuple[int, int]]:
    n, m = cost.shape
    dp = np.full((n, m), np.inf, dtype=float)
    prev = np.full((n, m, 2), -1, dtype=int)
    dp[0, 0] = cost[0, 0]
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            candidates = []
            if i > 0 and j > 0:
                candidates.append((dp[i - 1, j - 1], i - 1, j - 1))
            if i > 0:
                candidates.append((dp[i - 1, j] + gap_penalty, i - 1, j))
            if j > 0:
                candidates.append((dp[i, j - 1] + gap_penalty, i, j - 1))
            best, pi, pj = min(candidates, key=lambda item: item[0])
            dp[i, j] = cost[i, j] + best
            prev[i, j] = (pi, pj)

    path = []
    i, j = n - 1, m - 1
    while i >= 0 and j >= 0:
        path.append((i, j))
        pi, pj = prev[i, j]
        if pi < 0 or pj < 0:
            break
        i, j = int(pi), int(pj)
    path.reverse()
    return path


def _phi_from_path(
    path: Sequence[tuple[int, int]],
    times_a: np.ndarray,
    times_b: np.ndarray,
) -> np.ndarray:
    by_i: dict[int, list[float]] = {idx: [] for idx in range(len(times_a))}
    for i, j in path:
        by_i[int(i)].append(float(times_b[int(j)]))
    known_i = []
    known_phi = []
    for i, values in by_i.items():
        if values:
            known_i.append(i)
            known_phi.append(float(np.mean(values)))
    if not known_i:
        return np.interp(times_a, [times_a[0], times_a[-1]], [times_b[0], times_b[-1]])
    phi = np.interp(np.arange(len(times_a)), known_i, known_phi)
    phi = np.clip(phi, float(times_b[0]), float(times_b[-1]))
    return np.maximum.accumulate(phi)


def _alignment_quality_label(rmse_linear: float, rmse_aligned: float, n_anchors: int) -> str:
    if n_anchors <= 0 or not np.isfinite(rmse_aligned):
        return "failed"
    if not np.isfinite(rmse_linear) or rmse_linear <= 0:
        return "good" if rmse_aligned <= 0.5 else "weak"
    ratio = rmse_aligned / max(rmse_linear, 1e-12)
    if ratio <= 0.75:
        return "good"
    if ratio <= 1.05:
        return "ok"
    return "weak"


def _compute_alignment(
    process_a: pd.DataFrame,
    process_b: pd.DataFrame,
    *,
    condition_a: str,
    condition_b: str,
    time_range_a: tuple[float, float],
    time_range_b: tuple[float, float],
    anchor_pathways: Optional[Sequence[str]],
    anchor_set_name: str,
    alignment_lambda: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | str | int], list[str]]:
    times_a = np.sort(process_a["time_norm"].dropna().unique().astype(float))
    times_b = np.sort(process_b["time_norm"].dropna().unique().astype(float))
    if len(times_a) == 0 or len(times_b) == 0:
        raise ValueError("Both conditions must have at least one finite time point")

    anchors = _available_anchor_pathways(process_a, process_b, anchor_pathways)
    if not anchors:
        raise ValueError("No requested anchor pathways are present in both conditions")

    anchor_a = _anchor_matrix(process_a, times_a, anchors)
    anchor_b = _anchor_matrix(process_b, times_b, anchors)
    anchor_a, anchor_b = _standardize_anchor_pair(anchor_a, anchor_b)

    cost = np.zeros((len(times_a), len(times_b)), dtype=float)
    for i in range(len(times_a)):
        diff = anchor_b - anchor_a[i, :]
        cost[i, :] = np.nanmean(diff * diff, axis=1)
    path = _dtw_path(cost, gap_penalty=float(alignment_lambda))
    phi = _phi_from_path(path, times_a, times_b)

    aligned_b = np.vstack(
        [np.interp(phi, times_b, anchor_b[:, col], left=anchor_b[0, col], right=anchor_b[-1, col]) for col in range(anchor_b.shape[1])]
    ).T
    linear_b = np.vstack(
        [np.interp(times_a, times_b, anchor_b[:, col], left=anchor_b[0, col], right=anchor_b[-1, col]) for col in range(anchor_b.shape[1])]
    ).T
    rmse_aligned = float(np.sqrt(np.nanmean((anchor_a - aligned_b) ** 2)))
    rmse_linear = float(np.sqrt(np.nanmean((anchor_a - linear_b) ** 2)))
    improvement = (
        float((rmse_linear - rmse_aligned) / rmse_linear)
        if np.isfinite(rmse_linear) and rmse_linear > 0
        else np.nan
    )
    quality = _alignment_quality_label(rmse_linear, rmse_aligned, len(anchors))

    alignment_df = pd.DataFrame(
        {
            "condition_A": str(condition_a),
            "condition_B": str(condition_b),
            "alignment_anchor_set": anchor_set_name,
            "condition_A_time_norm": times_a,
            "condition_B_aligned_time_norm": phi,
            "condition_A_time_raw": _norm_to_raw(times_a, time_range_a),
            "condition_B_aligned_time_raw": _norm_to_raw(phi, time_range_b),
            "alignment_quality": quality,
            "anchor_rmse_linear": rmse_linear,
            "anchor_rmse_aligned": rmse_aligned,
            "anchor_rmse_improvement": improvement,
            "alignment_lambda": float(alignment_lambda),
            "n_anchor_pathways": int(len(anchors)),
            "dtw_path_length": int(len(path)),
        }
    )

    anchor_rows = []
    for idx, pathway in enumerate(anchors):
        raw_corr = np.nan
        aligned_corr = np.nan
        if len(times_a) >= 2:
            raw_b = np.interp(times_a, times_b, anchor_b[:, idx], left=anchor_b[0, idx], right=anchor_b[-1, idx])
            aligned = np.interp(phi, times_b, anchor_b[:, idx], left=anchor_b[0, idx], right=anchor_b[-1, idx])
            if np.nanstd(anchor_a[:, idx]) > 0 and np.nanstd(raw_b) > 0:
                raw_corr = float(np.corrcoef(anchor_a[:, idx], raw_b)[0, 1])
            if np.nanstd(anchor_a[:, idx]) > 0 and np.nanstd(aligned) > 0:
                aligned_corr = float(np.corrcoef(anchor_a[:, idx], aligned)[0, 1])
        anchor_rows.append(
            {
                "condition_A": str(condition_a),
                "condition_B": str(condition_b),
                "alignment_anchor_set": anchor_set_name,
                "pathway": pathway,
                "used_as_anchor": True,
                "anchor_correlation_raw": raw_corr,
                "anchor_correlation_aligned": aligned_corr,
                "anchor_rmse_linear": rmse_linear,
                "anchor_rmse_aligned": rmse_aligned,
                "anchor_rmse_improvement": improvement,
                "alignment_quality": quality,
            }
        )
    quality_meta: dict[str, float | str | int] = {
        "anchor_rmse_linear": rmse_linear,
        "anchor_rmse_aligned": rmse_aligned,
        "anchor_rmse_improvement": improvement,
        "alignment_quality": quality,
        "n_anchor_pathways": int(len(anchors)),
    }
    return alignment_df, pd.DataFrame(anchor_rows), quality_meta, anchors


def _linear_alignment(
    process_a: pd.DataFrame,
    *,
    condition_a: str,
    condition_b: str,
    time_range_a: tuple[float, float],
    time_range_b: tuple[float, float],
    anchor_set_name: str,
) -> pd.DataFrame:
    times_a = np.sort(process_a["time_norm"].dropna().unique().astype(float))
    phi = np.clip(times_a, 0.0, 1.0)
    return pd.DataFrame(
        {
            "condition_A": str(condition_a),
            "condition_B": str(condition_b),
            "alignment_anchor_set": anchor_set_name,
            "condition_A_time_norm": times_a,
            "condition_B_aligned_time_norm": phi,
            "condition_A_time_raw": _norm_to_raw(times_a, time_range_a),
            "condition_B_aligned_time_raw": _norm_to_raw(phi, time_range_b),
            "alignment_quality": "linear_time",
            "anchor_rmse_linear": np.nan,
            "anchor_rmse_aligned": np.nan,
            "anchor_rmse_improvement": 0.0,
            "alignment_lambda": 0.0,
            "n_anchor_pathways": 0,
            "dtw_path_length": int(len(times_a)),
        }
    )


def _aligned_score_process(
    process_a: pd.DataFrame,
    process_b: pd.DataFrame,
    alignment_df: pd.DataFrame,
    *,
    condition_a: str,
    condition_b: str,
    time_range_a: tuple[float, float],
    time_range_b: tuple[float, float],
    anchor_pathways: Sequence[str],
    anchor_set_name: str,
    quality_meta: dict[str, float | str | int],
    pathway_col_name: str = "Pathway",
) -> pd.DataFrame:
    common = sorted(set(process_a["pathway"].astype(str)) & set(process_b["pathway"].astype(str)))
    times_a = alignment_df["condition_A_time_norm"].to_numpy(dtype=float)
    phi = alignment_df["condition_B_aligned_time_norm"].to_numpy(dtype=float)

    rows = []
    for pathway in common:
        score_a = _interp_curve(process_a, pathway, times_a, value_col="score")
        score_b_aligned = _interp_curve(process_b, pathway, phi, value_col="score")
        score_b_linear = _interp_curve(process_b, pathway, times_a, value_col="score")
        q_a = _interp_curve(process_a, pathway, times_a, value_col="window_q")
        q_b_aligned = _interp_curve(process_b, pathway, phi, value_col="window_q")
        contrast = score_a - score_b_aligned
        raw_contrast = score_a - score_b_linear
        for idx, state_time in enumerate(times_a):
            rows.append(
                {
                    "condition_A": str(condition_a),
                    "condition_B": str(condition_b),
                    pathway_col_name: pathway,
                    "pathway": pathway,
                    "state_time": float(state_time),
                    "condition_A_time_norm": float(state_time),
                    "condition_B_aligned_time_norm": float(phi[idx]),
                    "condition_A_time_raw": float(_norm_to_raw(np.asarray([state_time]), time_range_a)[0]),
                    "condition_B_aligned_time_raw": float(_norm_to_raw(np.asarray([phi[idx]]), time_range_b)[0]),
                    "score_A": float(score_a[idx]),
                    "score_B_aligned": float(score_b_aligned[idx]),
                    "score_B_linear": float(score_b_linear[idx]),
                    "D_A_minus_B": float(contrast[idx]),
                    "raw_D_A_minus_B": float(raw_contrast[idx]),
                    "window_q_A": float(q_a[idx]) if np.isfinite(q_a[idx]) else np.nan,
                    "window_q_B_aligned": float(q_b_aligned[idx]) if np.isfinite(q_b_aligned[idx]) else np.nan,
                    "alignment_anchor_set": anchor_set_name,
                    "is_alignment_anchor": pathway in set(anchor_pathways),
                    "alignment_quality": quality_meta.get("alignment_quality", ""),
                    "anchor_rmse_linear": quality_meta.get("anchor_rmse_linear", np.nan),
                    "anchor_rmse_aligned": quality_meta.get("anchor_rmse_aligned", np.nan),
                    "anchor_rmse_improvement": quality_meta.get("anchor_rmse_improvement", np.nan),
                }
            )
    return pd.DataFrame(rows)


def _pathway_contrast_metrics(
    aligned_process: pd.DataFrame,
    *,
    time_tolerance: float,
    auc_tolerance: float,
    contrast_threshold: float,
) -> pd.DataFrame:
    if aligned_process is None or aligned_process.empty:
        return pd.DataFrame()
    rows = []
    for pathway, group in aligned_process.groupby("pathway", sort=False):
        group = group.sort_values("state_time")
        t = group["state_time"].to_numpy(dtype=float)
        a = group["score_A"].to_numpy(dtype=float)
        b_aligned = group["score_B_aligned"].to_numpy(dtype=float)
        b_linear = group["score_B_linear"].to_numpy(dtype=float)
        d = group["D_A_minus_B"].to_numpy(dtype=float)
        raw_d = group["raw_D_A_minus_B"].to_numpy(dtype=float)

        peak_a = float(t[int(np.nanargmax(np.abs(a)))]) if np.isfinite(a).any() else np.nan
        peak_b_raw = float(t[int(np.nanargmax(np.abs(b_linear)))]) if np.isfinite(b_linear).any() else np.nan
        peak_b_aligned = float(t[int(np.nanargmax(np.abs(b_aligned)))]) if np.isfinite(b_aligned).any() else np.nan
        raw_peak_shift = peak_b_raw - peak_a if np.isfinite(peak_a) and np.isfinite(peak_b_raw) else np.nan
        aligned_peak_shift = peak_b_aligned - peak_a if np.isfinite(peak_a) and np.isfinite(peak_b_aligned) else np.nan
        raw_auc_diff = _trapz(raw_d, t)
        aligned_auc_diff = _trapz(d, t)
        raw_abs_auc = abs(raw_auc_diff)
        aligned_abs_auc = abs(aligned_auc_diff)
        max_abs_aligned = float(np.nanmax(np.abs(d))) if len(d) else np.nan
        if (
            np.isfinite(raw_peak_shift)
            and abs(raw_peak_shift) > time_tolerance
            and aligned_abs_auc <= max(auc_tolerance, 0.5 * raw_abs_auc)
            and (not np.isfinite(max_abs_aligned) or max_abs_aligned < contrast_threshold)
        ):
            effect_type = "trajectory_speed_difference"
        elif np.isfinite(aligned_peak_shift) and abs(aligned_peak_shift) > time_tolerance and aligned_abs_auc > auc_tolerance:
            effect_type = "event_order_rewiring"
        elif aligned_abs_auc > auc_tolerance:
            effect_type = "amplitude_rewiring"
        elif np.isfinite(max_abs_aligned) and max_abs_aligned >= contrast_threshold:
            effect_type = "focal_rewiring"
        else:
            effect_type = "aligned_shared_program"
        if aligned_auc_diff > 0:
            direction = "condition_A_higher"
        elif aligned_auc_diff < 0:
            direction = "condition_B_higher"
        else:
            direction = "no_direction"
        rows.append(
            {
                "pathway": pathway,
                "raw_peak_shift": raw_peak_shift,
                "aligned_peak_shift": aligned_peak_shift,
                "raw_AUC_diff": raw_auc_diff,
                "aligned_AUC_diff": aligned_auc_diff,
                "max_abs_aligned_D": max_abs_aligned,
                "effect_type": effect_type,
                "contrast_direction": direction,
            }
        )
    return pd.DataFrame(rows)


def _component_slices(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    slices: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(mask):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            slices.append((start, idx))
            start = None
    if start is not None:
        slices.append((start, len(mask)))
    return slices


def _local_spacing(time: np.ndarray) -> float:
    time = np.asarray(time, dtype=float)
    finite = np.sort(time[np.isfinite(time)])
    if len(finite) < 2:
        return 0.0
    diffs = np.diff(finite)
    diffs = diffs[diffs > 0]
    return float(np.nanmedian(diffs)) if len(diffs) else 0.0


def _component_integral(values: np.ndarray, time: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    time = np.asarray(time, dtype=float)
    finite = np.isfinite(values) & np.isfinite(time)
    if finite.sum() == 0:
        return 0.0
    values = values[finite]
    time = time[finite]
    if len(values) == 1:
        return float(values[0] * _local_spacing(time))
    return _trapz(values, time)


def _phase_label(time_value: float, time_min: float, time_max: float) -> str:
    if not np.isfinite(time_value) or time_max <= time_min:
        return "unclear"
    rel = (time_value - time_min) / (time_max - time_min)
    if rel < 1.0 / 3.0:
        return "early"
    if rel > 2.0 / 3.0:
        return "late"
    return "mid"


def _contrast_events(
    aligned_process: pd.DataFrame,
    *,
    condition_a: str,
    condition_b: str,
    anchor_set_name: str,
    quality_meta: dict[str, float | str | int],
    contrast_threshold: float,
    min_consecutive: int,
    time_tolerance: float,
    auc_tolerance: float,
) -> pd.DataFrame:
    if aligned_process is None or aligned_process.empty:
        return pd.DataFrame()
    metrics = _pathway_contrast_metrics(
        aligned_process,
        time_tolerance=time_tolerance,
        auc_tolerance=auc_tolerance,
        contrast_threshold=contrast_threshold,
    )
    rows = []
    for pathway, group in aligned_process.groupby("pathway", sort=False):
        group = group.sort_values("state_time")
        time = pd.to_numeric(group["state_time"], errors="coerce").to_numpy(dtype=float)
        values = pd.to_numeric(group["D_A_minus_B"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(time) & np.isfinite(values)
        if not finite.any():
            continue
        time = time[finite]
        values = values[finite]
        active = np.abs(values) >= float(contrast_threshold)
        slices = _component_slices(active)
        if not slices:
            peak_idx = int(np.nanargmax(np.abs(values)))
            slices = [(peak_idx, peak_idx + 1)]
        spacing = _local_spacing(time)
        t_min = float(np.nanmin(time))
        t_max = float(np.nanmax(time))
        for event_idx, (start, end) in enumerate(slices, start=1):
            idx = np.arange(start, end)
            event_time = time[idx]
            event_values = values[idx]
            if len(event_time) == 0:
                continue
            peak_local = int(np.nanargmax(np.abs(event_values)))
            peak_value = float(event_values[peak_local])
            peak_time = float(event_time[peak_local])
            trough_local = int(np.nanargmin(event_values))
            positive = np.maximum(event_values, 0.0)
            negative = np.maximum(-event_values, 0.0)
            c_plus = _component_integral(positive, event_time)
            c_minus = _component_integral(negative, event_time)
            c_abs = _component_integral(np.abs(event_values), event_time)
            signed_auc = c_plus - c_minus
            duration = (
                float(event_time[-1] - event_time[0])
                if len(event_time) > 1
                else float(spacing)
            )
            direction = (
                "condition_A_higher"
                if c_plus >= c_minus
                else "condition_B_higher"
            )
            phase = _phase_label(peak_time, t_min, t_max)
            rows.append(
                {
                    "Pathway": pathway,
                    "pathway": pathway,
                    "event_index": int(event_idx),
                    "activation_onset": float(event_time[0]) if c_plus > 0 else np.nan,
                    "suppression_onset": float(event_time[0]) if c_minus > 0 else np.nan,
                    "peak_time": peak_time,
                    "peak_NES": peak_value,
                    "trough_time": float(event_time[trough_local]),
                    "trough_NES": float(event_values[trough_local]),
                    "duration": duration,
                    "AUC": signed_auc,
                    "integrated_NES": signed_auc,
                    "AUC_abs": c_abs,
                    "contrast_C": c_abs,
                    "contrast_C_plus": c_plus,
                    "contrast_C_minus": c_minus,
                    "contrast_direction": direction,
                    "sharpness": float(abs(peak_value) / max(c_abs, 1e-12)),
                    "direction_switch": np.nan,
                    "direction_switch_count": int(
                        np.any(np.sign(event_values[event_values != 0]) != np.sign(peak_value))
                    )
                    if np.any(event_values != 0)
                    else 0,
                    "recurrence": max(len(slices) - 1, 0),
                    "window_fdr_min": np.nan,
                    "significant_window_count": int(len(event_time)),
                    "event_window_count": int(len(event_time)),
                    "event_label": f"{phase} differential pathway event ({direction})",
                    "event_confidence_class": (
                        "multi_window_contrast" if len(event_time) >= max(2, min_consecutive) else "single_window_contrast"
                    ),
                    "event_confidence_reason": f"|D_A_minus_B| >= {float(contrast_threshold):g}",
                }
            )

    events = pd.DataFrame(rows)
    if events.empty:
        return events
    events = events.merge(metrics, on="pathway", how="left")
    events["condition_A"] = str(condition_a)
    events["condition_B"] = str(condition_b)
    events["alignment_anchor_set"] = anchor_set_name
    events["alignment_quality"] = quality_meta.get("alignment_quality", "")
    events["anchor_rmse_linear"] = quality_meta.get("anchor_rmse_linear", np.nan)
    events["anchor_rmse_aligned"] = quality_meta.get("anchor_rmse_aligned", np.nan)
    events["anchor_rmse_improvement"] = quality_meta.get("anchor_rmse_improvement", np.nan)
    events["event_statistic_used"] = "contrast_C_abs"
    events["observed_event_statistic"] = pd.to_numeric(events["contrast_C"], errors="coerce")
    events["event_id"] = [
        f"{condition_a}_vs_{condition_b}|{anchor_set_name}|{row.pathway}|{int(row.event_index):03d}"
        for row in events.itertuples()
    ]
    return events.sort_values(
        ["observed_event_statistic", "pathway", "event_index"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def _calibrate_contrast_events(
    events: pd.DataFrame,
    aligned_process: pd.DataFrame,
    *,
    condition_a: str,
    condition_b: str,
    anchor_set_name: str,
    quality_meta: dict[str, float | str | int],
    contrast_threshold: float,
    min_consecutive: int,
    time_tolerance: float,
    auc_tolerance: float,
    n_permutations: int,
    seed: int,
    global_null: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fdr_columns = [
        "condition_A",
        "condition_B",
        "pathway",
        "event_id",
        "event_statistic_used",
        "observed_event_statistic",
        "event_p",
        "event_q",
        "n_perm",
        "minimum_attainable_p",
        "minimum_attainable_q",
        "null_model",
        "calibration_status",
        "alignment_anchor_set",
        "alignment_quality",
    ]
    if events is None or events.empty:
        empty = pd.DataFrame(columns=fdr_columns)
        return pd.DataFrame(), empty

    calibrated = events.copy()
    if int(n_permutations) <= 0:
        calibrated["event_p"] = np.nan
        calibrated["event_q"] = np.nan
        calibrated["event_fdr"] = np.nan
        calibrated["n_perm"] = 0
        calibrated["minimum_attainable_p"] = np.nan
        calibrated["minimum_attainable_q"] = np.nan
        calibrated["null_model"] = "none"
        calibrated["calibration_status"] = "exploratory_no_null"
        return calibrated, calibrated[fdr_columns].copy()

    pathways = sorted(aligned_process["pathway"].astype(str).unique())
    grouped = {
        pathway: group.sort_values("state_time").reset_index(drop=True)
        for pathway, group in aligned_process.groupby("pathway", sort=False)
    }
    rng = np.random.default_rng(seed)
    null_frames = []
    for perm_id in range(int(n_permutations)):
        shuffled = rng.permutation(pathways)
        if len(pathways) > 1 and np.all(shuffled == np.asarray(pathways)):
            shuffled = np.roll(shuffled, 1)
        perm_parts = []
        for pathway, b_pathway in zip(pathways, shuffled):
            left = grouped[pathway].copy()
            right = grouped[str(b_pathway)]
            left["score_B_aligned"] = right["score_B_aligned"].to_numpy(dtype=float)
            left["score_B_linear"] = right["score_B_linear"].to_numpy(dtype=float)
            left["D_A_minus_B"] = left["score_A"].to_numpy(dtype=float) - left["score_B_aligned"].to_numpy(dtype=float)
            left["raw_D_A_minus_B"] = left["score_A"].to_numpy(dtype=float) - left["score_B_linear"].to_numpy(dtype=float)
            perm_parts.append(left)
        perm_process = pd.concat(perm_parts, ignore_index=True)
        perm_events = _contrast_events(
            perm_process,
            condition_a=condition_a,
            condition_b=condition_b,
            anchor_set_name=anchor_set_name,
            quality_meta=quality_meta,
            contrast_threshold=contrast_threshold,
            min_consecutive=min_consecutive,
            time_tolerance=time_tolerance,
            auc_tolerance=auc_tolerance,
        )
        if not perm_events.empty:
            perm_events = perm_events.copy()
            perm_events["perm_id"] = int(perm_id)
            null_frames.append(
                perm_events[["pathway", "observed_event_statistic", "perm_id"]]
            )

    null_events = pd.concat(null_frames, ignore_index=True) if null_frames else pd.DataFrame()
    p_values = []
    for _, row in calibrated.iterrows():
        if null_events.empty:
            p_values.append(np.nan)
            continue
        if global_null:
            null_values = (
                pd.to_numeric(null_events["observed_event_statistic"], errors="coerce")
                .dropna()
                .to_numpy(dtype=float)
            )
        else:
            null_values = (
                pd.to_numeric(
                    null_events.loc[
                        null_events["pathway"].astype(str) == str(row["pathway"]),
                        "observed_event_statistic",
                    ],
                    errors="coerce",
                )
                .dropna()
                .to_numpy(dtype=float)
            )
        observed = float(row["observed_event_statistic"])
        if len(null_values) == 0 or not np.isfinite(observed):
            p_values.append(np.nan)
        else:
            p_values.append((1.0 + float(np.sum(null_values >= observed))) / (1.0 + float(len(null_values))))

    q_values = _bh_adjust(p_values)
    min_p = 1.0 / (1.0 + float(len(null_events))) if not null_events.empty else np.nan
    calibrated["event_p"] = p_values
    calibrated["event_q"] = q_values
    calibrated["event_fdr"] = q_values
    calibrated["n_perm"] = int(n_permutations)
    calibrated["minimum_attainable_p"] = min_p
    calibrated["minimum_attainable_q"] = (
        np.minimum(1.0, len(calibrated) * min_p) if np.isfinite(min_p) else np.nan
    )
    calibrated["null_model"] = "pathway_label_permutation_after_alignment_screening"
    calibrated["calibration_status"] = (
        "screening_pathway_label_null" if not null_events.empty else "null_calibration_failed"
    )
    return calibrated, calibrated[fdr_columns].copy()


def _alignment_sensitivity(
    process_a: pd.DataFrame,
    process_b: pd.DataFrame,
    *,
    condition_a: str,
    condition_b: str,
    time_range_a: tuple[float, float],
    time_range_b: tuple[float, float],
    primary_events: pd.DataFrame,
    primary_aligned: pd.DataFrame,
    anchors: Sequence[str],
    anchor_set_name: str,
    alignment_lambda: float,
    contrast_threshold: float,
    min_consecutive: int,
    time_tolerance: float,
    auc_tolerance: float,
) -> pd.DataFrame:
    primary_metrics = _pathway_contrast_metrics(
        primary_aligned,
        time_tolerance=time_tolerance,
        auc_tolerance=auc_tolerance,
        contrast_threshold=contrast_threshold,
    )
    if primary_metrics.empty:
        return pd.DataFrame()
    primary_metrics = primary_metrics.set_index("pathway")
    variants: list[tuple[str, Optional[list[str]]]] = [("linear_time", None)]
    if len(anchors) >= 2:
        for anchor in anchors:
            variants.append((f"leave_one_out:{anchor}", [p for p in anchors if p != anchor]))

    rows = []
    for variant_name, variant_anchors in variants:
        if variant_name == "linear_time":
            alignment_df = _linear_alignment(
                process_a,
                condition_a=condition_a,
                condition_b=condition_b,
                time_range_a=time_range_a,
                time_range_b=time_range_b,
                anchor_set_name=f"{anchor_set_name}:{variant_name}",
            )
            quality_meta: dict[str, float | str | int] = {
                "alignment_quality": "linear_time",
                "anchor_rmse_linear": np.nan,
                "anchor_rmse_aligned": np.nan,
                "anchor_rmse_improvement": 0.0,
            }
            used_anchors = []
        else:
            alignment_df, _anchor_df, quality_meta, used_anchors = _compute_alignment(
                process_a,
                process_b,
                condition_a=condition_a,
                condition_b=condition_b,
                time_range_a=time_range_a,
                time_range_b=time_range_b,
                anchor_pathways=variant_anchors,
                anchor_set_name=f"{anchor_set_name}:{variant_name}",
                alignment_lambda=alignment_lambda,
            )
        aligned = _aligned_score_process(
            process_a,
            process_b,
            alignment_df,
            condition_a=condition_a,
            condition_b=condition_b,
            time_range_a=time_range_a,
            time_range_b=time_range_b,
            anchor_pathways=used_anchors,
            anchor_set_name=f"{anchor_set_name}:{variant_name}",
            quality_meta=quality_meta,
        )
        events = _contrast_events(
            aligned,
            condition_a=condition_a,
            condition_b=condition_b,
            anchor_set_name=f"{anchor_set_name}:{variant_name}",
            quality_meta=quality_meta,
            contrast_threshold=contrast_threshold,
            min_consecutive=min_consecutive,
            time_tolerance=time_tolerance,
            auc_tolerance=auc_tolerance,
        )
        metrics = _pathway_contrast_metrics(
            aligned,
            time_tolerance=time_tolerance,
            auc_tolerance=auc_tolerance,
            contrast_threshold=contrast_threshold,
        ).set_index("pathway")
        event_pathways = set(events["pathway"].astype(str)) if not events.empty else set()
        primary_event_pathways = set(primary_events["pathway"].astype(str)) if primary_events is not None and not primary_events.empty else set()
        for pathway, primary_row in primary_metrics.iterrows():
            if pathway not in metrics.index:
                continue
            variant_row = metrics.loc[pathway]
            primary_auc = float(primary_row["aligned_AUC_diff"])
            variant_auc = float(variant_row["aligned_AUC_diff"])
            sign_agreement = (
                bool(np.sign(primary_auc) == np.sign(variant_auc))
                if primary_auc != 0 and variant_auc != 0
                else np.nan
            )
            rows.append(
                {
                    "condition_A": str(condition_a),
                    "condition_B": str(condition_b),
                    "pathway": pathway,
                    "primary_anchor_set": anchor_set_name,
                    "sensitivity_variant": variant_name,
                    "variant_anchor_pathways": ";".join(used_anchors),
                    "primary_effect_type": primary_row["effect_type"],
                    "variant_effect_type": variant_row["effect_type"],
                    "primary_aligned_AUC_diff": primary_auc,
                    "variant_aligned_AUC_diff": variant_auc,
                    "aligned_AUC_diff_delta": variant_auc - primary_auc,
                    "primary_aligned_peak_shift": primary_row["aligned_peak_shift"],
                    "variant_aligned_peak_shift": variant_row["aligned_peak_shift"],
                    "effect_type_stable": bool(primary_row["effect_type"] == variant_row["effect_type"]),
                    "sign_agreement": sign_agreement,
                    "primary_event_called": pathway in primary_event_pathways,
                    "variant_event_called": pathway in event_pathways,
                    "alignment_quality": quality_meta.get("alignment_quality", ""),
                }
            )
    return pd.DataFrame(rows)


def run_aligned_trajectory_contrast(
    results: pd.DataFrame,
    *,
    condition_col: str = "condition",
    condition_a: Optional[str] = None,
    condition_b: Optional[str] = None,
    reference: Optional[str] = None,
    query: Optional[str] = None,
    anchor_pathways: Optional[Sequence[str]] = None,
    pathway_col: Optional[str] = None,
    time_col: str = "pt_mid",
    score_col: str = "NES",
    fdr_col: Optional[str] = "padj",
    contrast_threshold: float = 0.5,
    min_consecutive: int = 2,
    alignment_lambda: float = 1.0,
    anchor_set_name: str = "alignment_anchors",
    time_tolerance: float = 0.05,
    auc_tolerance: float = 0.25,
    n_permutations: int = 0,
    seed: int = 42,
    sensitivity: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Discover alignment-aware differential trajectory pathway events.

    The input is a long rolling-window pathway score table containing two
    conditions. TED-v3 first learns a monotone pathway-state alignment from
    anchor pathways, then detects events on ``S_A(t) - S_B(phi(t))`` instead of
    on either absolute score process.
    """
    if results is None or results.empty:
        raise ValueError("results must contain pathway score rows")
    if condition_col not in results.columns:
        raise ValueError(f"Missing condition column '{condition_col}'")
    pathway_col = pathway_col or _pick_column(results, "Pathway", "pathway")
    if pathway_col is None:
        raise ValueError("Could not find a pathway column")
    if time_col not in results.columns:
        raise ValueError(f"Missing time column '{time_col}'")
    if score_col not in results.columns:
        raise ValueError(f"Missing score column '{score_col}'")
    fdr_col = _pick_column(results, fdr_col) if fdr_col else None

    values = _condition_values(results, condition_col)
    if len(values) < 2:
        raise ValueError("At least two condition values are required")
    condition_a = str(reference if reference is not None else (condition_a if condition_a is not None else values[0]))
    condition_b = str(query if query is not None else (condition_b if condition_b is not None else values[1]))

    process_a, time_range_a = _prepare_condition_process(
        results,
        condition_col=condition_col,
        condition=condition_a,
        pathway_col=pathway_col,
        time_col=time_col,
        score_col=score_col,
        fdr_col=fdr_col,
    )
    process_b, time_range_b = _prepare_condition_process(
        results,
        condition_col=condition_col,
        condition=condition_b,
        pathway_col=pathway_col,
        time_col=time_col,
        score_col=score_col,
        fdr_col=fdr_col,
    )
    alignment_df, anchor_df, quality_meta, anchors = _compute_alignment(
        process_a,
        process_b,
        condition_a=condition_a,
        condition_b=condition_b,
        time_range_a=time_range_a,
        time_range_b=time_range_b,
        anchor_pathways=anchor_pathways,
        anchor_set_name=anchor_set_name,
        alignment_lambda=alignment_lambda,
    )
    aligned_process = _aligned_score_process(
        process_a,
        process_b,
        alignment_df,
        condition_a=condition_a,
        condition_b=condition_b,
        time_range_a=time_range_a,
        time_range_b=time_range_b,
        anchor_pathways=anchors,
        anchor_set_name=anchor_set_name,
        quality_meta=quality_meta,
        pathway_col_name=str(pathway_col),
    )
    events = _contrast_events(
        aligned_process,
        condition_a=condition_a,
        condition_b=condition_b,
        anchor_set_name=anchor_set_name,
        quality_meta=quality_meta,
        contrast_threshold=contrast_threshold,
        min_consecutive=min_consecutive,
        time_tolerance=time_tolerance,
        auc_tolerance=auc_tolerance,
    )
    calibrated_events, event_fdr = _calibrate_contrast_events(
        events,
        aligned_process,
        condition_a=condition_a,
        condition_b=condition_b,
        anchor_set_name=anchor_set_name,
        quality_meta=quality_meta,
        contrast_threshold=contrast_threshold,
        min_consecutive=min_consecutive,
        time_tolerance=time_tolerance,
        auc_tolerance=auc_tolerance,
        n_permutations=n_permutations,
        seed=seed,
    )
    sensitivity_report = (
        _alignment_sensitivity(
            process_a,
            process_b,
            condition_a=condition_a,
            condition_b=condition_b,
            time_range_a=time_range_a,
            time_range_b=time_range_b,
            primary_events=calibrated_events,
            primary_aligned=aligned_process,
            anchors=anchors,
            anchor_set_name=anchor_set_name,
            alignment_lambda=alignment_lambda,
            contrast_threshold=contrast_threshold,
            min_consecutive=min_consecutive,
            time_tolerance=time_tolerance,
            auc_tolerance=auc_tolerance,
        )
        if sensitivity
        else pd.DataFrame()
    )

    tables = {
        "trajectory_alignment_functions": alignment_df,
        "alignment_anchor_pathways": anchor_df,
        "aligned_pathway_score_process": aligned_process,
        "differential_event_table": calibrated_events,
        "differential_event_fdr": event_fdr,
        "alignment_sensitivity_report": sensitivity_report,
    }
    for table in tables.values():
        table.attrs["aligned_trajectory_contrast"] = {
            "condition_A": condition_a,
            "condition_B": condition_b,
            "anchor_set_name": anchor_set_name,
            "anchor_pathways": list(anchors),
            "contrast_threshold": float(contrast_threshold),
            "alignment_lambda": float(alignment_lambda),
            "n_permutations": int(n_permutations),
        }
    return tables


def write_aligned_trajectory_contrast(
    tables: dict[str, pd.DataFrame],
    outdir: str | Path,
    *,
    sep: str = "\t",
) -> dict[str, Path]:
    """Write TED-v3 alignment contrast tables using the standard filenames."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, filename in _OUTPUT_FILENAMES.items():
        table = tables.get(key, pd.DataFrame())
        path = out / filename
        table.to_csv(path, sep=sep, index=False, na_rep="NA")
        paths[key] = path
    return paths
