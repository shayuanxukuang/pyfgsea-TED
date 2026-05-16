from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .wrapper import load_gmt


_REPLICATION_OUTPUT_FILENAMES = {
    "event_match_matrix": "event_match_matrix.tsv",
    "cross_dataset_event_replication": "cross_dataset_event_replication.tsv",
    "meta_event_score": "meta_event_score.tsv",
    "dataset_event_coverage": "dataset_event_coverage.tsv",
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


def _safe_float(value: object, default: float = np.nan) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(out) if np.isfinite(out) else float(default)


def _label(row: pd.Series, label_col: Optional[str]) -> str:
    if label_col is not None and label_col in row.index and pd.notna(row[label_col]):
        return str(row[label_col])
    for col in ("pathway", "Pathway", "module", "event"):
        if col in row.index and pd.notna(row[col]):
            return str(row[col])
    return ""


def _direction(row: pd.Series) -> int:
    if "direction" in row.index and pd.notna(row["direction"]):
        text = str(row["direction"]).lower()
        if "negative" in text or "suppression" in text or "down" in text:
            return -1
        if "positive" in text or "activation" in text or "up" in text:
            return 1
    for col in ("peak_NES", "event_score", "effect_size", "AUC", "integrated_NES"):
        if col in row.index:
            value = _safe_float(row[col])
            if np.isfinite(value) and value != 0:
                return int(np.sign(value))
    return 0


def _interval(row: pd.Series) -> tuple[float, float, float]:
    peak = _safe_float(row.get("peak_time", row.get("event_peak", np.nan)))
    start_candidates = (
        row.get("event_onset", np.nan),
        row.get("activation_onset", np.nan),
        row.get("suppression_onset", np.nan),
        row.get("pt_start", np.nan),
    )
    starts = [_safe_float(value) for value in start_candidates]
    start = next((value for value in starts if np.isfinite(value)), peak)
    duration = abs(_safe_float(row.get("duration", np.nan), default=np.nan))
    if np.isfinite(duration) and duration > 0 and np.isfinite(start):
        end = start + duration
    else:
        end = _safe_float(row.get("pt_end", np.nan), default=np.nan)
        if not np.isfinite(end):
            end = peak
    if not np.isfinite(peak):
        peak = start if np.isfinite(start) else end
    if not np.isfinite(start):
        start = peak
    if not np.isfinite(end):
        end = peak
    if np.isfinite(start) and np.isfinite(end) and end < start:
        start, end = end, start
    return float(start), float(end), float(peak)


def _prepare_events(
    events: pd.DataFrame,
    *,
    dataset_col: str,
    event_id_col: Optional[str],
    label_col: Optional[str],
    q_col: Optional[str],
) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame()
    if dataset_col not in events:
        raise ValueError(f"Missing dataset column '{dataset_col}'")
    event_id_col = _pick_column(events, event_id_col, "event_id")
    label_col = label_col or _pick_column(events, "pathway", "Pathway", "module", "event")
    q_col = _pick_column(events, q_col, "event_q", "event_fdr", "q")
    rows = []
    for idx, row in events.reset_index(drop=True).iterrows():
        dataset = str(row[dataset_col])
        label = _label(row, label_col)
        event_id = str(row[event_id_col]) if event_id_col and pd.notna(row.get(event_id_col)) else f"{dataset}|{label}|{idx + 1:03d}"
        start, end, peak = _interval(row)
        rows.append(
            {
                "dataset": dataset,
                "event_id": event_id,
                "event_label": label,
                "event_start": start,
                "event_end": end,
                "event_peak": peak,
                "event_q": _safe_float(row.get(q_col, np.nan)) if q_col else np.nan,
                "direction": _direction(row),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.drop_duplicates(["dataset", "event_id"]).reset_index(drop=True)
    norm_rows = []
    for dataset, group in out.groupby("dataset", sort=False):
        values = pd.concat(
            [
                pd.to_numeric(group["event_start"], errors="coerce"),
                pd.to_numeric(group["event_end"], errors="coerce"),
                pd.to_numeric(group["event_peak"], errors="coerce"),
            ],
            ignore_index=True,
        ).dropna()
        low = float(values.min()) if not values.empty else 0.0
        high = float(values.max()) if not values.empty else 1.0
        span = max(high - low, 1e-12)
        sub = group.copy()
        for col in ("event_start", "event_end", "event_peak"):
            sub[f"{col}_norm"] = (pd.to_numeric(sub[col], errors="coerce") - low) / span
        norm_rows.append(sub)
    return pd.concat(norm_rows, ignore_index=True)


def _load_gene_sets(gene_sets: Optional[dict[str, list[str] | set[str]] | str | Path]) -> dict[str, set[str]]:
    if gene_sets is None:
        return {}
    raw = load_gmt(str(gene_sets)) if isinstance(gene_sets, (str, Path)) else gene_sets
    return {str(name): {str(gene).upper() for gene in genes} for name, genes in raw.items()}


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = {token for token in left.upper().replace("-", "_").split("_") if token}
    right_tokens = {token for token in right.upper().replace("-", "_").split("_") if token}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return np.nan
    return float(len(left & right) / len(left | right))


def _event_gene_set(label: str, gene_sets: dict[str, set[str]]) -> set[str]:
    if label in gene_sets:
        return gene_sets[label]
    lower = {name.lower(): genes for name, genes in gene_sets.items()}
    return lower.get(str(label).lower(), set())


def _driver_gene_sets(driver_scores: Optional[pd.DataFrame]) -> dict[str, set[str]]:
    if driver_scores is None or driver_scores.empty:
        return {}
    event_col = _pick_column(driver_scores, "event_id")
    gene_col = _pick_column(driver_scores, "gene", "target_gene")
    if event_col is None or gene_col is None:
        return {}
    out = {}
    for event_id, group in driver_scores.groupby(event_col, sort=False):
        out[str(event_id)] = {str(gene).upper() for gene in group[gene_col].dropna()}
    return out


def _score_curves(
    score_process: Optional[pd.DataFrame],
    *,
    dataset_col: str,
    label_col: Optional[str],
    score_time_col: Optional[str],
    score_col: Optional[str],
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    if score_process is None or score_process.empty:
        return {}
    dataset_col = _pick_column(score_process, dataset_col, "dataset")
    label_col = label_col or _pick_column(score_process, "pathway", "Pathway", "module", "event")
    score_time_col = _pick_column(score_process, score_time_col, "pt_mid", "center_time", "state_time")
    score_col = _pick_column(score_process, score_col, "NES", "module_score", "D_A_minus_B", "score")
    if dataset_col is None or label_col is None or score_time_col is None or score_col is None:
        return {}
    curves = {}
    for (dataset, label), group in score_process.groupby([dataset_col, label_col], sort=False):
        x = pd.to_numeric(group[score_time_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(group[score_col], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(x) & np.isfinite(y)
        if keep.sum() < 2:
            continue
        x = x[keep]
        y = y[keep]
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        span = max(float(x[-1] - x[0]), 1e-12)
        curves[(str(dataset), str(label))] = ((x - x[0]) / span, y)
    return curves


def _curve_correlation(
    left: pd.Series,
    right: pd.Series,
    curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
) -> float:
    left_curve = curves.get((left["dataset"], left["event_label"]))
    right_curve = curves.get((right["dataset"], right["event_label"]))
    if left_curve is None or right_curve is None:
        return np.nan
    grid = np.linspace(0.0, 1.0, 50)
    left_y = np.interp(grid, left_curve[0], left_curve[1])
    right_y = np.interp(grid, right_curve[0], right_curve[1])
    if np.nanstd(left_y) <= 0 or np.nanstd(right_y) <= 0:
        return np.nan
    corr = float(np.corrcoef(left_y, right_y)[0, 1])
    return (corr + 1.0) / 2.0 if np.isfinite(corr) else np.nan


def _time_iou(left: pd.Series, right: pd.Series) -> float:
    a0 = _safe_float(left["event_start_norm"])
    a1 = _safe_float(left["event_end_norm"])
    b0 = _safe_float(right["event_start_norm"])
    b1 = _safe_float(right["event_end_norm"])
    if not all(np.isfinite(value) for value in (a0, a1, b0, b1)):
        return np.nan
    if a1 < a0:
        a0, a1 = a1, a0
    if b1 < b0:
        b0, b1 = b1, b0
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    if union <= 0:
        return 1.0 if abs(float(left["event_peak_norm"]) - float(right["event_peak_norm"])) < 1e-9 else 0.0
    return float(inter / union)


def _direction_consistency(left: pd.Series, right: pd.Series) -> float:
    left_dir = int(left.get("direction", 0))
    right_dir = int(right.get("direction", 0))
    if left_dir == 0 or right_dir == 0:
        return 0.5
    return 1.0 if left_dir == right_dir else 0.0


def _neglog10_q(q: float) -> float:
    if not np.isfinite(q):
        return 0.0
    return float(-np.log10(max(q, 1e-300)))


def _weighted_score(components: dict[str, float], weights: dict[str, float]) -> float:
    total = 0.0
    denom = 0.0
    for name, value in components.items():
        if not np.isfinite(value):
            continue
        weight = float(weights.get(name, 0.0))
        total += weight * float(value)
        denom += weight
    return float(total / denom) if denom > 0 else 0.0


def _match_matrix(
    events: pd.DataFrame,
    *,
    gene_sets: dict[str, set[str]],
    driver_sets: dict[str, set[str]],
    curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    weights: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for i, left in events.iterrows():
        for j, right in events.iterrows():
            if i >= j or left["dataset"] == right["dataset"]:
                continue
            left_genes = _event_gene_set(left["event_label"], gene_sets)
            right_genes = _event_gene_set(right["event_label"], gene_sets)
            gene_jaccard = _jaccard(left_genes, right_genes)
            if not np.isfinite(gene_jaccard):
                gene_jaccard = 1.0 if left["event_label"] == right["event_label"] else _token_jaccard(left["event_label"], right["event_label"])
            leading_jaccard = _jaccard(
                driver_sets.get(left["event_id"], set()),
                driver_sets.get(right["event_id"], set()),
            )
            components = {
                "gene_jaccard": gene_jaccard,
                "time_iou": _time_iou(left, right),
                "score_correlation": _curve_correlation(left, right, curves),
                "leading_edge_jaccard": leading_jaccard,
            }
            match_score = _weighted_score(components, weights)
            rows.append(
                {
                    "dataset_1": left["dataset"],
                    "event_id_1": left["event_id"],
                    "event_1": left["event_label"],
                    "event_q_1": left["event_q"],
                    "dataset_2": right["dataset"],
                    "event_id_2": right["event_id"],
                    "event_2": right["event_label"],
                    "event_q_2": right["event_q"],
                    **components,
                    "direction_consistency": _direction_consistency(left, right),
                    "match_score": match_score,
                }
            )
    return pd.DataFrame(rows).sort_values("match_score", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def _replication_table(matches: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if matches is None or matches.empty:
        return pd.DataFrame()
    out = matches[pd.to_numeric(matches["match_score"], errors="coerce") >= float(threshold)].copy()
    if out.empty:
        return out
    out["replication_status"] = np.where(
        pd.to_numeric(out["direction_consistency"], errors="coerce") >= 0.5,
        "replicated_direction_supported",
        "matched_opposite_direction",
    )
    return out.reset_index(drop=True)


def _meta_scores(events: pd.DataFrame, replicated: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    match_lookup: dict[str, list[pd.Series]] = {event_id: [] for event_id in events["event_id"].astype(str)}
    if replicated is not None and not replicated.empty:
        for _, row in replicated.iterrows():
            match_lookup.setdefault(str(row["event_id_1"]), []).append(row)
            match_lookup.setdefault(str(row["event_id_2"]), []).append(row)
    event_by_id = events.set_index("event_id", drop=False)
    rows = []
    for event_id, event in event_by_id.iterrows():
        matched = match_lookup.get(str(event_id), [])
        datasets = {str(event["dataset"])}
        score = _neglog10_q(float(event["event_q"]))
        direction_values = []
        match_labels = []
        for match in matched:
            if str(match["event_id_1"]) == str(event_id):
                other_id = str(match["event_id_2"])
                other_dataset = str(match["dataset_2"])
                other_label = str(match["event_2"])
                other_q = _safe_float(match["event_q_2"])
            else:
                other_id = str(match["event_id_1"])
                other_dataset = str(match["dataset_1"])
                other_label = str(match["event_1"])
                other_q = _safe_float(match["event_q_1"])
            consistency = _safe_float(match["direction_consistency"], default=0.5)
            datasets.add(other_dataset)
            direction_values.append(consistency)
            score += _neglog10_q(other_q) * max(consistency, 0.0)
            match_labels.append(f"{other_dataset}:{other_label}:{other_id}")
        rows.append(
            {
                "dataset": event["dataset"],
                "event_id": event_id,
                "event": event["event_label"],
                "event_q": event["event_q"],
                "replicated_dataset_count": int(max(len(datasets) - 1, 0)),
                "datasets_with_event": ";".join(sorted(datasets)),
                "meta_event_score": float(score),
                "mean_direction_consistency": float(np.mean(direction_values)) if direction_values else np.nan,
                "matched_events": ";".join(match_labels[:20]),
                "evidence_level": "cross_dataset_replicated" if len(datasets) > 1 else "single_dataset_event",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["replicated_dataset_count", "meta_event_score"], ascending=[False, False]
    ).reset_index(drop=True)


def _coverage(events: pd.DataFrame, replicated: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in events.groupby("event_label", sort=False):
        rows.append(
            {
                "event": label,
                "observed_dataset_count": int(group["dataset"].nunique()),
                "observed_datasets": ";".join(sorted(group["dataset"].astype(str).unique())),
                "min_event_q": float(pd.to_numeric(group["event_q"], errors="coerce").min()),
                "replicated_match_count": 0,
                "best_match_score": np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if replicated is None or replicated.empty or out.empty:
        return out.sort_values(["observed_dataset_count", "event"], ascending=[False, True]).reset_index(drop=True)
    counts = {}
    best = {}
    for _, row in replicated.iterrows():
        for col in ("event_1", "event_2"):
            label = str(row[col])
            counts[label] = counts.get(label, 0) + 1
            best[label] = max(best.get(label, 0.0), _safe_float(row["match_score"], default=0.0))
    out["replicated_match_count"] = out["event"].map(counts).fillna(0).astype(int)
    out["best_match_score"] = out["event"].map(best)
    return out.sort_values(
        ["observed_dataset_count", "replicated_match_count", "best_match_score", "event"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def match_cross_dataset_events(
    events: pd.DataFrame,
    *,
    score_process: Optional[pd.DataFrame] = None,
    driver_scores: Optional[pd.DataFrame] = None,
    gene_sets: Optional[dict[str, list[str] | set[str]] | str | Path] = None,
    dataset_col: str = "dataset",
    event_id_col: Optional[str] = "event_id",
    label_col: Optional[str] = None,
    q_col: Optional[str] = "event_q",
    score_time_col: Optional[str] = None,
    score_col: Optional[str] = None,
    match_threshold: float = 0.6,
    weights: Optional[dict[str, float]] = None,
) -> dict[str, pd.DataFrame]:
    """
    Match trajectory events across datasets and compute meta-event evidence.

    The matching score combines pathway/module gene overlap, normalized timing
    interval IoU, score-curve correlation, and leading-edge/driver overlap. Any
    unavailable component is skipped and the remaining weights are renormalized.
    """
    prepared = _prepare_events(
        events,
        dataset_col=dataset_col,
        event_id_col=event_id_col,
        label_col=label_col,
        q_col=q_col,
    )
    if prepared.empty:
        raise ValueError("events must contain at least one event")
    weight_map = {
        "gene_jaccard": 0.35,
        "time_iou": 0.25,
        "score_correlation": 0.20,
        "leading_edge_jaccard": 0.20,
    }
    if weights:
        weight_map.update({str(key): float(value) for key, value in weights.items()})
    curves = _score_curves(
        score_process,
        dataset_col=dataset_col,
        label_col=label_col,
        score_time_col=score_time_col,
        score_col=score_col,
    )
    matches = _match_matrix(
        prepared,
        gene_sets=_load_gene_sets(gene_sets),
        driver_sets=_driver_gene_sets(driver_scores),
        curves=curves,
        weights=weight_map,
    )
    replicated = _replication_table(matches, match_threshold)
    meta = _meta_scores(prepared, replicated)
    coverage = _coverage(prepared, replicated)
    return {
        "event_match_matrix": matches,
        "cross_dataset_event_replication": replicated,
        "meta_event_score": meta,
        "dataset_event_coverage": coverage,
    }


def write_cross_dataset_replication(
    tables: dict[str, pd.DataFrame],
    outdir: str | Path,
    *,
    sep: str = "\t",
) -> dict[str, Path]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, filename in _REPLICATION_OUTPUT_FILENAMES.items():
        table = tables.get(key, pd.DataFrame())
        path = out / filename
        table.to_csv(path, sep=sep, index=False, na_rep="NA")
        paths[key] = path
    return paths
