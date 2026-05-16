from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


_FATE_OUTPUT_FILENAMES = {
    "prebranch_fate_predictive_events": "prebranch_fate_predictive_events.tsv",
    "fate_prediction_model_performance": "fate_prediction_model_performance.tsv",
    "prebranch_event_fdr": "prebranch_event_fdr.tsv",
    "ted_prime_pseudotime_matched_null": "ted_prime_pseudotime_matched_null.tsv",
    "fate_predictive_driver_genes": "fate_predictive_driver_genes.tsv",
    "fate_predictive_leading_edge": "fate_predictive_leading_edge.tsv",
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


def _roc_auc_binary(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    finite = np.isfinite(scores)
    scores = scores[finite]
    labels = labels[finite]
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=float)
    rank_sum_pos = float(ranks[labels == 1].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _stratified_fold_ids(labels: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    fold = np.full(len(labels), -1, dtype=int)
    for value in sorted(pd.Series(labels).dropna().unique(), key=str):
        idx = np.where(labels == value)[0]
        rng.shuffle(idx)
        for pos, obs_idx in enumerate(idx):
            fold[obs_idx] = pos % max(n_splits, 1)
    return fold


def _cross_validated_auc(scores: np.ndarray, labels: np.ndarray, n_splits: int, seed: int) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    scores = scores[finite]
    labels = labels[finite]
    if len(scores) == 0:
        return np.nan
    counts = pd.Series(labels).value_counts()
    if len(counts) < 2:
        return np.nan
    max_splits = int(min(n_splits, counts.min()))
    if max_splits < 2:
        auc = _roc_auc_binary(scores, labels)
        return float(max(auc, 1.0 - auc)) if np.isfinite(auc) else np.nan
    folds = _stratified_fold_ids(labels, max_splits, seed)
    aucs = []
    for fold_id in range(max_splits):
        test = folds == fold_id
        if int(test.sum()) == 0:
            continue
        auc = _roc_auc_binary(scores[test], labels[test])
        if np.isfinite(auc):
            aucs.append(max(auc, 1.0 - auc))
    return float(np.mean(aucs)) if aucs else np.nan


def _time_bin_permutation(values: np.ndarray, pt: np.ndarray, n_bins: int, rng) -> np.ndarray:
    values = np.asarray(values).copy()
    pt = np.asarray(pt, dtype=float)
    finite = np.isfinite(pt)
    if finite.sum() == 0:
        rng.shuffle(values)
        return values
    ranks = pd.Series(pt[finite]).rank(method="first")
    bins = pd.qcut(ranks, q=min(int(n_bins), int(finite.sum())), labels=False, duplicates="drop")
    out = values.copy()
    finite_idx = np.where(finite)[0]
    for bin_id in pd.Series(bins).dropna().unique():
        idx = finite_idx[np.asarray(bins) == bin_id]
        out[idx] = rng.permutation(out[idx])
    return out


def _process_curves(
    results: pd.DataFrame,
    pathway_col: str,
    time_col: str,
    score_col: str,
) -> dict[str, pd.DataFrame]:
    work = pd.DataFrame(
        {
            "pathway": results[pathway_col].astype(str).to_numpy(),
            "time": pd.to_numeric(results[time_col], errors="coerce").to_numpy(dtype=float),
            "score": pd.to_numeric(results[score_col], errors="coerce").to_numpy(dtype=float),
        }
    )
    work = work[np.isfinite(work["time"]) & np.isfinite(work["score"])]
    curves = {}
    for pathway, group in work.groupby("pathway", sort=False):
        curve = (
            group.groupby("time", as_index=False)["score"]
            .mean()
            .sort_values("time")
            .reset_index(drop=True)
        )
        curves[pathway] = curve
    return curves


def _interp_curve(curve: pd.DataFrame, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if curve is None or curve.empty:
        return np.full(len(x), np.nan)
    times = curve["time"].to_numpy(dtype=float)
    scores = curve["score"].to_numpy(dtype=float)
    if len(times) == 1:
        return np.full(len(x), scores[0], dtype=float)
    return np.interp(x, times, scores, left=scores[0], right=scores[-1])


def _exposure_curve(curve: pd.DataFrame, x: np.ndarray, bandwidth: Optional[float]) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if curve is None or curve.empty:
        return np.full(len(x), np.nan)
    times = curve["time"].to_numpy(dtype=float)
    scores = curve["score"].to_numpy(dtype=float)
    keep = np.isfinite(times) & np.isfinite(scores)
    times = times[keep]
    scores = scores[keep]
    if len(times) == 0:
        return np.full(len(x), np.nan)
    order = np.argsort(times)
    times = times[order]
    scores = scores[order]
    if len(times) == 1:
        return np.where(x >= times[0], scores[0], 0.0)
    span = max(float(times[-1] - times[0]), 1e-12)
    bw = float(bandwidth) if bandwidth is not None else 0.15 * span
    if not np.isfinite(bw) or bw <= 0:
        bw = 0.15 * span
    dt = np.gradient(times)
    out = []
    for tau in x:
        if not np.isfinite(tau):
            out.append(np.nan)
            continue
        delta = tau - times
        active = delta >= 0
        if not active.any():
            out.append(0.0)
            continue
        weights = np.exp(-0.5 * (delta[active] / bw) ** 2)
        out.append(float(np.sum(weights * scores[active] * dt[active])))
    return np.asarray(out, dtype=float)


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    center = float(np.nanmean(values))
    scale = float(np.nanstd(values))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    out = (values - center) / scale
    out[~np.isfinite(out)] = 0.0
    return out


def _logistic_theta(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    x = _standardize(scores)
    finite = np.isfinite(x) & np.isfinite(labels)
    x = x[finite]
    labels = labels[finite]
    if len(np.unique(labels)) < 2:
        return np.nan
    try:
        import statsmodels.api as sm

        fit = sm.Logit(labels, sm.add_constant(x)).fit(disp=False, maxiter=100)
        return float(fit.params[1])
    except Exception:
        return float(np.nanmean(x[labels == 1]) - np.nanmean(x[labels == 0]))


def _binary_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    scores = scores[finite]
    labels = labels[finite]
    if len(scores) == 0 or len(np.unique(labels)) < 2:
        return {
            "balanced_accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "macro_F1": np.nan,
        }
    auc = _roc_auc_binary(scores, labels)
    oriented = scores if not np.isfinite(auc) or auc >= 0.5 else -scores
    threshold = float(np.nanmedian(oriented))
    pred = (oriented >= threshold).astype(int)
    tp = float(np.sum((pred == 1) & (labels == 1)))
    fp = float(np.sum((pred == 1) & (labels == 0)))
    tn = float(np.sum((pred == 0) & (labels == 0)))
    fn = float(np.sum((pred == 0) & (labels == 1)))
    recall_pos = tp / (tp + fn) if tp + fn > 0 else np.nan
    recall_neg = tn / (tn + fp) if tn + fp > 0 else np.nan
    precision = tp / (tp + fp) if tp + fp > 0 else np.nan
    f1_pos = 2 * precision * recall_pos / (precision + recall_pos) if precision + recall_pos > 0 else np.nan
    precision_neg = tn / (tn + fn) if tn + fn > 0 else np.nan
    f1_neg = (
        2 * precision_neg * recall_neg / (precision_neg + recall_neg)
        if precision_neg + recall_neg > 0
        else np.nan
    )
    return {
        "balanced_accuracy": float(np.nanmean([recall_pos, recall_neg])),
        "precision": float(precision),
        "recall": float(recall_pos),
        "macro_F1": float(np.nanmean([f1_pos, f1_neg])),
    }


def _event_summary_for_prebranch_curve(
    curve: pd.DataFrame,
    split_time: float,
    event_threshold: float,
) -> dict[str, float]:
    sub = curve[curve["time"] <= float(split_time)].copy()
    if sub.empty:
        return {
            "event_onset": np.nan,
            "event_peak": np.nan,
            "event_score": np.nan,
            "prebranch_fraction": np.nan,
        }
    time = sub["time"].to_numpy(dtype=float)
    score = sub["score"].to_numpy(dtype=float)
    peak_idx = int(np.nanargmax(np.abs(score)))
    active = np.abs(score) >= float(event_threshold)
    onset = float(time[np.where(active)[0][0]]) if active.any() else float(time[peak_idx])
    pre_auc = _trapz(np.abs(score), time)
    all_auc = _trapz(np.abs(curve["score"].to_numpy(dtype=float)), curve["time"].to_numpy(dtype=float))
    return {
        "event_onset": onset,
        "event_peak": float(time[peak_idx]),
        "event_score": float(score[peak_idx]),
        "prebranch_fraction": float(pre_auc / all_auc) if all_auc > 0 else np.nan,
    }


def _leading_edge_table(
    results: pd.DataFrame,
    events: pd.DataFrame,
    *,
    pathway_col: str,
    time_col: str,
    split_time: float,
    top_n: int,
) -> pd.DataFrame:
    leading_col = _pick_column(results, "leading_edge", "leading_edges", "core_genes")
    if leading_col is None or events is None or events.empty:
        return pd.DataFrame(
            columns=[
                "pathway",
                "future_fate",
                "gene",
                "leading_edge_frequency",
                "n_prebranch_windows",
            ]
        )
    rows = []
    pre = results[pd.to_numeric(results[time_col], errors="coerce") <= float(split_time)]
    for row in events.itertuples():
        group = pre[pre[pathway_col].astype(str) == str(row.pathway)]
        genes = []
        for raw in group[leading_col].dropna().astype(str):
            genes.extend([gene.strip() for gene in raw.replace(",", ";").split(";") if gene.strip()])
        if not genes:
            continue
        counts = pd.Series(genes).value_counts()
        denom = max(int(len(group)), 1)
        for gene, count in counts.head(int(top_n)).items():
            frequency = float(count) / denom
            theta = getattr(row, "theta", np.nan)
            event_score = getattr(row, "event_score", np.nan)
            rows.append(
                {
                    "dataset": getattr(row, "dataset", ""),
                    "pathway": row.pathway,
                    "pathway_or_module": getattr(row, "pathway_or_module", row.pathway),
                    "future_fate": row.future_fate,
                    "gene": gene,
                    "leading_edge_frequency": frequency,
                    "driver_score": frequency
                    * abs(float(theta) if np.isfinite(theta) else float(event_score) if np.isfinite(event_score) else 1.0),
                    "n_prebranch_windows": int(len(group)),
                    "evidence_level": getattr(row, "evidence_level", "screening"),
                }
            )
    return pd.DataFrame(rows)


def _driver_gene_strings(leading: pd.DataFrame, top_n: int) -> dict[tuple[str, str], str]:
    if leading is None or leading.empty:
        return {}
    out = {}
    for (pathway, fate), group in leading.groupby(["pathway", "future_fate"], sort=False):
        group = group.sort_values("leading_edge_frequency", ascending=False)
        out[(str(pathway), str(fate))] = ";".join(group["gene"].astype(str).head(int(top_n)))
    return out


def run_fate_predictive_events(
    results: pd.DataFrame,
    *,
    adata=None,
    obs: Optional[pd.DataFrame] = None,
    pseudotime_key: str = "dpt_pseudotime",
    fate_key: Optional[str] = None,
    fate_probability_cols: Optional[Sequence[str]] = None,
    pathway_col: Optional[str] = None,
    time_col: str = "pt_mid",
    score_col: str = "NES",
    split_time: Optional[float] = None,
    prebranch_quantile: float = 0.5,
    event_threshold: float = 0.5,
    min_prebranch_cells: int = 20,
    n_splits: int = 5,
    n_permutations: int = 0,
    permutation_bins: int = 5,
    seed: int = 42,
    top_driver_genes: int = 10,
    evidence_q_threshold: float = 0.1,
    dataset: Optional[str] = None,
    exposure_kernel_bandwidth: Optional[float] = None,
    prebranch_max_fate_probability: Optional[float] = 0.7,
    prebranch_entropy_quantile: Optional[float] = None,
) -> dict[str, pd.DataFrame]:
    """
    Score pre-branch pathway events for future fate prediction.

    This lightweight TED-v3 implementation interpolates each pathway score
    process onto pre-branch cells, computes one-vs-rest future-fate AUCs, and
    calibrates them by fate-label permutation within pseudotime bins.
    """
    if results is None or results.empty:
        raise ValueError("results must contain pathway score rows")
    if obs is None:
        if adata is None:
            raise ValueError("Pass either adata or obs")
        obs = adata.obs.copy()
    else:
        obs = obs.copy()
    if pseudotime_key not in obs:
        raise ValueError(f"pseudotime_key '{pseudotime_key}' not found in obs")
    if fate_key is None and not fate_probability_cols:
        raise ValueError("Pass fate_key or fate_probability_cols")
    if fate_key is not None and fate_key not in obs:
        raise ValueError(f"fate_key '{fate_key}' not found in obs")

    pathway_col = pathway_col or _pick_column(results, "Pathway", "pathway")
    if pathway_col is None:
        raise ValueError("Could not find a pathway column")
    if time_col not in results:
        raise ValueError(f"Missing time column '{time_col}'")
    if score_col not in results:
        raise ValueError(f"Missing score column '{score_col}'")

    pt = pd.to_numeric(obs[pseudotime_key], errors="coerce").to_numpy(dtype=float)
    finite_pt = np.isfinite(pt)
    if split_time is None:
        if not finite_pt.any():
            raise ValueError("No finite pseudotime values available")
        split_time = float(np.nanquantile(pt[finite_pt], float(prebranch_quantile)))
    if fate_key is not None:
        fate_values = obs[fate_key].astype(str).to_numpy()
        fate_probability_mask = np.ones(len(obs), dtype=bool)
    else:
        missing = [col for col in fate_probability_cols or [] if col not in obs]
        if missing:
            raise ValueError(f"Missing fate probability columns: {missing}")
        prob = obs[list(fate_probability_cols)].apply(pd.to_numeric, errors="coerce")
        fate_values = prob.idxmax(axis=1).astype(str).to_numpy()
        prob_values = prob.to_numpy(dtype=float)
        row_sum = np.nansum(prob_values, axis=1, keepdims=True)
        row_sum[~np.isfinite(row_sum) | (row_sum <= 0)] = 1.0
        prob_norm = np.clip(prob_values / row_sum, 1e-12, 1.0)
        entropy = -np.nansum(prob_norm * np.log(prob_norm), axis=1)
        max_prob = np.nanmax(prob_norm, axis=1)
        masks = []
        if prebranch_max_fate_probability is not None:
            masks.append(max_prob < float(prebranch_max_fate_probability))
        if prebranch_entropy_quantile is not None:
            finite_entropy = entropy[np.isfinite(entropy)]
            if len(finite_entropy):
                masks.append(entropy >= float(np.nanquantile(finite_entropy, float(prebranch_entropy_quantile))))
        fate_probability_mask = np.logical_or.reduce(masks) if masks else np.ones(len(obs), dtype=bool)

    pre_mask = finite_pt & (pt <= float(split_time)) & fate_probability_mask
    if int(pre_mask.sum()) < int(min_prebranch_cells):
        raise ValueError("Too few pre-branch cells for fate-predictive analysis")

    if fate_key is not None:
        fates = sorted(pd.Series(fate_values[pre_mask]).dropna().unique(), key=str)
    else:
        fates = [str(col) for col in fate_probability_cols or []]

    curves = _process_curves(results, pathway_col, time_col, score_col)
    pre_pt = pt[pre_mask]
    pre_fate = fate_values[pre_mask]
    rows = []
    for pathway, curve in curves.items():
        signal = _exposure_curve(curve, pre_pt, exposure_kernel_bandwidth)
        event_info = _event_summary_for_prebranch_curve(curve, float(split_time), event_threshold)
        for fate in fates:
            labels = (pre_fate.astype(str) == str(fate)).astype(int)
            if len(np.unique(labels)) < 2:
                continue
            auc = _roc_auc_binary(signal, labels)
            predictive_auc = float(max(auc, 1.0 - auc)) if np.isfinite(auc) else np.nan
            effect_size = float(np.nanmean(signal[labels == 1]) - np.nanmean(signal[labels == 0]))
            cv_auc = _cross_validated_auc(signal, labels, n_splits=n_splits, seed=seed)
            theta = _logistic_theta(signal, labels)
            stability = (
                float(np.clip(1.0 - abs(predictive_auc - cv_auc), 0.0, 1.0))
                if np.isfinite(predictive_auc) and np.isfinite(cv_auc)
                else np.nan
            )
            prebranch_fraction = event_info["prebranch_fraction"]
            fpes = (
                abs(theta)
                * (cv_auc if np.isfinite(cv_auc) else predictive_auc)
                * (prebranch_fraction if np.isfinite(prebranch_fraction) else 0.0)
                * (stability if np.isfinite(stability) else 1.0)
            )
            metrics = _binary_metrics(signal, labels)
            rows.append(
                {
                    "dataset": dataset or "",
                    "pathway": pathway,
                    "pathway_or_module": pathway,
                    "event_onset": event_info["event_onset"],
                    "event_peak": event_info["event_peak"],
                    "event_score": event_info["event_score"],
                    "prebranch_fraction": prebranch_fraction,
                    "future_fate": str(fate),
                    "theta": theta,
                    "effect_size": effect_size,
                    "apparent_AUC": predictive_auc,
                    "cross_validated_AUC": cv_auc,
                    "cv_auc": cv_auc,
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "macro_F1": metrics["macro_F1"],
                    "stability": stability,
                    "FPES": fpes,
                    "fate_predictive_event_score": fpes,
                    "n_prebranch_cells": int(pre_mask.sum()),
                    "split_time": float(split_time),
                    "event_q": np.nan,
                    "driver_genes": "",
                    "evidence_level": "screening",
                }
            )

    events = pd.DataFrame(rows)
    if events.empty:
        empty_fdr = pd.DataFrame()
        return {
            "prebranch_fate_predictive_events": events,
            "fate_prediction_model_performance": empty_fdr,
            "prebranch_event_fdr": empty_fdr,
            "fate_predictive_leading_edge": empty_fdr,
        }

    rng = np.random.default_rng(seed)
    null_values = []
    if int(n_permutations) > 0:
        pathways = list(curves)
        for perm_id in range(int(n_permutations)):
            perm_fate = _time_bin_permutation(pre_fate, pre_pt, int(permutation_bins), rng)
            for pathway in pathways:
                signal = _exposure_curve(curves[pathway], pre_pt, exposure_kernel_bandwidth)
                event_info = _event_summary_for_prebranch_curve(curves[pathway], float(split_time), event_threshold)
                for fate in fates:
                    labels = (perm_fate.astype(str) == str(fate)).astype(int)
                    if len(np.unique(labels)) < 2:
                        continue
                    auc = _roc_auc_binary(signal, labels)
                    theta = _logistic_theta(signal, labels)
                    if np.isfinite(auc):
                        auc_oriented = float(max(auc, 1.0 - auc))
                        prebranch_fraction = event_info["prebranch_fraction"]
                        null_values.append(
                            {
                                "pathway": pathway,
                                "future_fate": str(fate),
                                "perm_id": int(perm_id),
                                "null_AUC": auc_oriented,
                                "null_FPES": abs(theta)
                                * auc_oriented
                                * (prebranch_fraction if np.isfinite(prebranch_fraction) else 0.0),
                            }
                        )
    null = pd.DataFrame(null_values)
    p_values = []
    for row in events.itertuples():
        observed = float(row.FPES)
        if null.empty or not np.isfinite(observed):
            p_values.append(np.nan)
            continue
        subset = null[
            (null["pathway"].astype(str) == str(row.pathway))
            & (null["future_fate"].astype(str) == str(row.future_fate))
        ]
        values = pd.to_numeric(subset["null_FPES"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) == 0:
            p_values.append(np.nan)
        else:
            p_values.append((1.0 + float(np.sum(values >= observed))) / (1.0 + float(len(values))))

    events["event_p"] = p_values
    events["event_q"] = _bh_adjust(p_values)
    events["evidence_level"] = np.where(
        pd.to_numeric(events["event_q"], errors="coerce") <= float(evidence_q_threshold),
        "FDR_supported_prebranch_predictive_event",
        "screening_prebranch_predictive_event",
    )
    leading = _leading_edge_table(
        results,
        events,
        pathway_col=pathway_col,
        time_col=time_col,
        split_time=float(split_time),
        top_n=top_driver_genes,
    )
    driver_lookup = _driver_gene_strings(leading, top_driver_genes)
    events["driver_genes"] = [
        driver_lookup.get((str(row.pathway), str(row.future_fate)), "")
        for row in events.itertuples()
    ]

    event_fdr = events[
        [
            "pathway",
            "pathway_or_module",
            "future_fate",
            "apparent_AUC",
            "cross_validated_AUC",
            "theta",
            "FPES",
            "event_p",
            "event_q",
            "n_prebranch_cells",
            "split_time",
            "evidence_level",
        ]
    ].copy()
    event_fdr["null_model"] = (
        "future_fate_label_permutation_within_pseudotime_bins"
        if int(n_permutations) > 0
        else "none"
    )
    event_fdr["n_perm"] = int(n_permutations)
    if null.empty:
        prime_null = events[
            [
                "dataset",
                "future_fate",
                "pathway_or_module",
                "FPES",
                "event_p",
                "event_q",
            ]
        ].copy()
        prime_null["null_mean_FPES"] = np.nan
        prime_null["null_95pct_FPES"] = np.nan
    else:
        null_summary = (
            null.groupby(["pathway", "future_fate"], as_index=False)
            .agg(
                null_mean_FPES=("null_FPES", "mean"),
                null_95pct_FPES=("null_FPES", lambda x: float(np.nanquantile(x, 0.95))),
            )
            .rename(columns={"pathway": "pathway_or_module"})
        )
        prime_null = events[
            [
                "dataset",
                "future_fate",
                "pathway_or_module",
                "FPES",
                "event_p",
                "event_q",
            ]
        ].merge(null_summary, on=["pathway_or_module", "future_fate"], how="left")
    prime_null = prime_null.rename(columns={"FPES": "observed_FPES"})

    perf_rows = []
    for fate, group in events.groupby("future_fate", sort=False):
        best = group.sort_values("cross_validated_AUC", ascending=False).iloc[0]
        perf_rows.append(
            {
                "future_fate": fate,
                "best_pathway": best["pathway"],
                "best_cross_validated_AUC": best["cross_validated_AUC"],
                "best_macro_F1": best["macro_F1"],
                "best_balanced_accuracy": best["balanced_accuracy"],
                "mean_top5_cross_validated_AUC": float(
                    pd.to_numeric(
                        group.sort_values("cross_validated_AUC", ascending=False)
                        .head(5)["cross_validated_AUC"],
                        errors="coerce",
                    ).mean()
                ),
                "n_pathways": int(group["pathway"].nunique()),
                "n_prebranch_cells": int(pre_mask.sum()),
                "split_time": float(split_time),
                "model_type": "single_pathway_exposure_softmax_screen",
            }
        )
    performance = pd.DataFrame(perf_rows)
    events = events.sort_values(
        ["event_q", "cross_validated_AUC", "pathway"],
        ascending=[True, False, True],
        na_position="last",
    ).reset_index(drop=True)
    return {
        "prebranch_fate_predictive_events": events,
        "fate_prediction_model_performance": performance,
        "prebranch_event_fdr": event_fdr,
        "ted_prime_pseudotime_matched_null": prime_null,
        "fate_predictive_driver_genes": leading,
        "fate_predictive_leading_edge": leading,
    }


def write_fate_predictive_events(
    tables: dict[str, pd.DataFrame],
    outdir: str | Path,
    *,
    sep: str = "\t",
) -> dict[str, Path]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, filename in _FATE_OUTPUT_FILENAMES.items():
        table = tables.get(key, pd.DataFrame())
        path = out / filename
        table.to_csv(path, sep=sep, index=False, na_rep="NA")
        paths[key] = path
    return paths
