from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


_GENERIC_TERMS = {
    "cell_cycle",
    "cycle",
    "ribosome",
    "ribosomal",
    "mitochondria",
    "mitochondrial",
    "oxidative_phosphorylation",
    "stress",
    "immune",
    "inflammatory",
    "interferon",
    "myc",
    "translation",
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


def _safe_num(value: object, default: float = np.nan) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(out) if np.isfinite(out) else float(default)


def _event_id(row: pd.Series, event_id_col: Optional[str], label_col: Optional[str], idx: int) -> str:
    if event_id_col and event_id_col in row.index and pd.notna(row[event_id_col]):
        return str(row[event_id_col])
    label = _label(row, label_col)
    return f"{label}|event_{idx + 1:03d}"


def _label(row: pd.Series, label_col: Optional[str]) -> str:
    if label_col and label_col in row.index and pd.notna(row[label_col]):
        return str(row[label_col])
    for col in ("pathway_or_module", "pathway", "Pathway", "module", "event"):
        if col in row.index and pd.notna(row[col]):
            return str(row[col])
    return ""


def _generic_penalty(label: str) -> float:
    text = str(label).lower().replace("-", "_").replace(" ", "_")
    penalty = 0.0
    for term in _GENERIC_TERMS:
        if term in text:
            penalty = max(penalty, 0.45)
    if "cell_cycle_exit" in text or "fate" in text:
        penalty = min(penalty, 0.25)
    return float(penalty)


def _mechanistic_lookup(driver_scores: Optional[pd.DataFrame]) -> dict[str, float]:
    if driver_scores is None or driver_scores.empty:
        return {}
    event_col = _pick_column(driver_scores, "event_id")
    score_col = _pick_column(driver_scores, "driver_score", "regulator_activity")
    if event_col is None or score_col is None:
        return {}
    grouped = pd.to_numeric(driver_scores[score_col], errors="coerce").groupby(driver_scores[event_col].astype(str)).max()
    finite = grouped[np.isfinite(grouped)]
    denom = float(np.nanmax(np.log1p(finite))) if len(finite) else 1.0
    if not np.isfinite(denom) or denom <= 0:
        denom = 1.0
    return {str(event_id): float(np.log1p(value) / denom) for event_id, value in grouped.items() if np.isfinite(value)}


def _replication_lookup(replication: Optional[pd.DataFrame]) -> dict[str, float]:
    if replication is None or replication.empty:
        return {}
    if "event_id" in replication and "replicated_dataset_count" in replication:
        counts = pd.to_numeric(replication["replicated_dataset_count"], errors="coerce")
        denom = max(float(np.nanmax(counts)), 1.0) if np.isfinite(counts).any() else 1.0
        return {
            str(row.event_id): float(min(_safe_num(row.replicated_dataset_count, 0.0) / denom, 1.0))
            for row in replication.itertuples()
        }
    return {}


def _phenotype_lookup(phenotype_report: Optional[pd.DataFrame]) -> dict[str, float]:
    if phenotype_report is None or phenotype_report.empty or "event_id" not in phenotype_report:
        return {}
    out = {}
    for row in phenotype_report.itertuples():
        q = _safe_num(getattr(row, "event_q", np.nan))
        value = min(-np.log10(max(q, 1e-300)) / 5.0, 1.0) if np.isfinite(q) else 0.0
        out[str(row.event_id)] = value
    return out


def _robustness(row: pd.Series) -> float:
    candidates = []
    for col in ("robustness_score", "bootstrap_support", "ranker_support", "window_support", "stability"):
        if col in row.index:
            value = _safe_num(row[col])
            if np.isfinite(value):
                candidates.append(float(np.clip(value, 0.0, 1.0)))
    if candidates:
        return float(np.nanmean(candidates))
    if _safe_num(row.get("event_q", np.nan)) <= 0.05:
        return 0.7
    return 0.4


def _predictiveness(row: pd.Series, phenotype: dict[str, float], event_id: str) -> float:
    values = []
    if "prebranch_fraction" in row.index:
        pre = _safe_num(row["prebranch_fraction"])
        auc = _safe_num(row.get("cross_validated_AUC", row.get("cv_auc", np.nan)))
        if np.isfinite(pre) and np.isfinite(auc):
            values.append(float(np.clip(pre, 0.0, 1.0) * np.clip(auc, 0.0, 1.0)))
    if "FPES" in row.index:
        fpes = _safe_num(row["FPES"])
        if np.isfinite(fpes):
            values.append(float(np.clip(np.log1p(abs(fpes)) / 3.0, 0.0, 1.0)))
    if event_id in phenotype:
        values.append(phenotype[event_id])
    return float(max(values)) if values else 0.35


def _differential_specificity(row: pd.Series) -> float:
    if "aligned_AUC_diff" in row.index:
        value = abs(_safe_num(row["aligned_AUC_diff"]))
        if np.isfinite(value):
            return float(np.clip(np.log1p(value) / 2.0, 0.0, 1.0))
    if "contrast_C" in row.index:
        value = abs(_safe_num(row["contrast_C"]))
        if np.isfinite(value):
            return float(np.clip(np.log1p(value) / 2.0, 0.0, 1.0))
    if "effect_type" in row.index and str(row["effect_type"]):
        return 0.8
    return 0.35


def score_biological_discovery(
    events: pd.DataFrame,
    *,
    driver_scores: Optional[pd.DataFrame] = None,
    replication: Optional[pd.DataFrame] = None,
    phenotype_report: Optional[pd.DataFrame] = None,
    event_id_col: Optional[str] = "event_id",
    label_col: Optional[str] = None,
    generic_terms: Optional[Sequence[str]] = None,
    q_col: Optional[str] = "event_q",
    q_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Compute a biological discovery score for trajectory events.

    ``BDS(E) = Q(E) * R(E) * N(E) * P(E) * D(E) * M(E)`` where Q is event-q
    evidence, R robustness, N non-generic novelty, P prebranch/phenotype
    predictiveness, D differential specificity, and M mechanistic driver
    support.
    """
    if events is None or events.empty:
        return pd.DataFrame()
    q_col = _pick_column(events, q_col, "event_q", "event_fdr", "q")
    event_id_col = _pick_column(events, event_id_col, "event_id")
    label_col = label_col or _pick_column(events, "pathway_or_module", "pathway", "Pathway", "module", "event")
    global _GENERIC_TERMS
    old_terms = _GENERIC_TERMS
    if generic_terms is not None:
        _GENERIC_TERMS = {str(term).lower().replace(" ", "_") for term in generic_terms}
    mechanistic = _mechanistic_lookup(driver_scores)
    replicated = _replication_lookup(replication)
    phenotype = _phenotype_lookup(phenotype_report)
    rows = []
    try:
        for idx, row in events.reset_index(drop=True).iterrows():
            event_id = _event_id(row, event_id_col, label_col, idx)
            label = _label(row, label_col)
            q = _safe_num(row.get(q_col, np.nan)) if q_col else np.nan
            q_component = min(-np.log10(max(q, 1e-300)) / 5.0, 1.0) if np.isfinite(q) else 0.25
            robustness = _robustness(row)
            generic_penalty = _generic_penalty(label)
            novelty = 1.0 - generic_penalty
            predictiveness = _predictiveness(row, phenotype, event_id)
            differential = _differential_specificity(row)
            mechanism = mechanistic.get(event_id, 0.0)
            replication_component = replicated.get(event_id, 0.0)
            mechanism = max(mechanism, replication_component * 0.5)
            bds = q_component * robustness * novelty * predictiveness * differential * max(mechanism, 0.05)
            rows.append(
                {
                    "event_id": event_id,
                    "event": label,
                    "event_q": q,
                    "Q_event_q": q_component,
                    "R_robustness": robustness,
                    "N_novelty": novelty,
                    "generic_penalty": generic_penalty,
                    "P_predictiveness": predictiveness,
                    "D_differential_specificity": differential,
                    "M_mechanistic_support": mechanism,
                    "replication_support": replication_component,
                    "biological_discovery_score": float(bds),
                    "meaningful_discovery": bool(
                        np.isfinite(q)
                        and q <= float(q_threshold)
                        and novelty >= 0.5
                        and (predictiveness >= 0.35 or differential >= 0.5)
                        and mechanism > 0
                    ),
                }
            )
    finally:
        _GENERIC_TERMS = old_terms
    return pd.DataFrame(rows).sort_values(
        ["meaningful_discovery", "biological_discovery_score"],
        ascending=[False, False],
    ).reset_index(drop=True)


def write_biological_discovery_score(
    table: pd.DataFrame,
    outdir: str | Path,
    *,
    filename: str = "biological_discovery_score.tsv",
    sep: str = "\t",
) -> Path:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    table.to_csv(path, sep=sep, index=False, na_rep="NA")
    return path
