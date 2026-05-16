from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import spatial, stats


TED_DEVELOPMENTAL_OUTPUTS = {
    "ted_time_event_table": "ted_time_event_table.tsv",
    "developmental_event_mode": "developmental_event_mode.tsv",
    "ot_cell_couplings": "ot_cell_couplings.tsv",
    "ot_event_flow": "ot_event_flow.tsv",
    "counterfactual_event_loss": "counterfactual_event_loss.tsv",
    "fate_probability_linked_event": "fate_probability_linked_event.tsv",
    "multiome_event_lag_table": "multiome_event_lag_table.tsv",
    "motif_to_target_event_concordance": "motif_to_target_event_concordance.tsv",
    "chromatin_first_mechanism_candidates": "chromatin_first_mechanism_candidates.tsv",
    "lineage_edge_event_table": "lineage_edge_event_table.tsv",
    "sister_branch_divergence_score": "sister_branch_divergence_score.tsv",
    "prebranch_event_candidates": "prebranch_event_candidates.tsv",
    "lineage_convergence_event": "lineage_convergence_event.tsv",
    "spatial_neighborhood_event_table": "spatial_neighborhood_event_table.tsv",
    "boundary_specific_event": "boundary_specific_event.tsv",
    "spatial_event_propagation": "spatial_event_propagation.tsv",
    "cross_kingdom_event_ontology": "cross_kingdom_event_ontology.tsv",
    "species_gene_set_mapping": "species_gene_set_mapping.tsv",
    "orthology_confidence": "orthology_confidence.tsv",
    "event_grammar_similarity": "event_grammar_similarity.tsv",
    "developmental_claim_ceiling": "developmental_claim_ceiling.tsv",
}


_GRAMMAR_ROWS = [
    (
        "cell_cycle_exit",
        "cell-cycle deceleration, mitotic exit, or G1/G0 entry before fate output",
        "cell cycle;mitosis;E2F;G2M;CYCB;CDK",
    ),
    (
        "chromatin_accessibility_priming",
        "regulatory accessibility or motif activity appears before RNA/fate change",
        "chromatin;accessibility;ATAC;motif;TF binding;enhancer",
    ),
    (
        "stress_response",
        "osmotic, oxidative, heat, immune, or injury stress response dynamics",
        "stress;osmotic;hypoxia;ROS;heat shock;immune",
    ),
    (
        "metabolic_maturation",
        "energy, biosynthetic, or organellar maturation accompanying development",
        "metabolism;mitochondria;glycolysis;translation;ribosome;photosynthesis",
    ),
    (
        "cell_wall_ecm_remodeling",
        "extracellular matrix or cell wall remodeling during morphogenesis",
        "cell wall;ECM;collagen;pectin;expansin;matrix",
    ),
    (
        "hormone_morphogen_response",
        "response to positional signaling, hormone, or morphogen gradients",
        "hormone;morphogen;auxin;cytokinin;BMP;WNT;FGF;Notch",
    ),
    (
        "lineage_output",
        "terminal lineage program, differentiation marker, or fate output",
        "lineage;differentiation;fate;maturation;terminal;identity",
    ),
]


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


def _numeric(values: Any) -> pd.Series:
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _as_frame(data: Any, *, index: Optional[pd.Index] = None) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame(index=index)
    if isinstance(data, pd.DataFrame):
        out = data.copy()
    else:
        out = pd.DataFrame(data)
    if index is not None and not out.index.equals(index):
        if len(out) == len(index):
            out = out.copy()
            out.index = index
        else:
            shared = index.intersection(out.index)
            if len(shared) != len(index):
                raise ValueError("table indices must match or have the same length")
            out = out.loc[index]
    return out


def _norm(value: float, values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0 or not np.isfinite(value):
        return np.nan
    span = float(np.nanmax(finite) - np.nanmin(finite))
    if span <= 0:
        return 0.0
    return float((value - np.nanmin(finite)) / span)


def _stable_onset(
    axis: Sequence[float],
    values: Sequence[float],
    *,
    q_values: Optional[Sequence[float]] = None,
    fdr_threshold: float = 0.05,
    score_threshold: float = 0.0,
    min_consecutive: int = 1,
    direction: Optional[int] = None,
) -> float:
    axis_arr = np.asarray(axis, dtype=float)
    value_arr = np.asarray(values, dtype=float)
    finite = np.isfinite(axis_arr) & np.isfinite(value_arr)
    if q_values is not None:
        q_arr = np.asarray(q_values, dtype=float)
        finite = finite & np.isfinite(q_arr)
    if finite.sum() == 0:
        return np.nan

    axis_arr = axis_arr[finite]
    value_arr = value_arr[finite]
    order = np.argsort(axis_arr)
    axis_arr = axis_arr[order]
    value_arr = value_arr[order]
    if q_values is not None:
        q_arr = np.asarray(q_values, dtype=float)[finite][order]
        supported = q_arr <= fdr_threshold
    else:
        supported = np.ones(len(value_arr), dtype=bool)

    if direction is None:
        max_idx = int(np.nanargmax(np.abs(value_arr)))
        direction = 1 if value_arr[max_idx] >= 0 else -1
    active = supported & (value_arr * int(direction) > float(score_threshold))
    run = 0
    start = 0
    for idx, flag in enumerate(active):
        if flag:
            if run == 0:
                start = idx
            run += 1
            if run >= int(min_consecutive):
                return float(axis_arr[start])
        else:
            run = 0
    return np.nan


def _curve_by_axis(
    table: pd.DataFrame,
    *,
    axis_col: str,
    score_col: str,
    fdr_col: Optional[str],
) -> pd.DataFrame:
    keep = [axis_col, score_col] + ([fdr_col] if fdr_col and fdr_col in table.columns else [])
    work = table[keep].copy()
    work[axis_col] = pd.to_numeric(work[axis_col], errors="coerce")
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work = work[np.isfinite(work[axis_col]) & np.isfinite(work[score_col])]
    if work.empty:
        return pd.DataFrame(columns=[axis_col, "score", "q"])
    agg: dict[str, Any] = {score_col: "mean"}
    if fdr_col and fdr_col in work.columns:
        work[fdr_col] = pd.to_numeric(work[fdr_col], errors="coerce")
        agg[fdr_col] = "min"
    out = work.groupby(axis_col, as_index=False).agg(agg).sort_values(axis_col)
    out = out.rename(columns={score_col: "score"})
    if fdr_col and fdr_col in out.columns:
        out = out.rename(columns={fdr_col: "q"})
    else:
        out["q"] = np.nan
    return out


def _dominant_direction(values: Sequence[float]) -> int:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return 1
    max_abs = finite[np.argmax(np.abs(finite))]
    return 1 if max_abs >= 0 else -1


def _event_onset(row: pd.Series) -> float:
    for col in (
        "real_time_onset",
        "pseudotime_onset",
        "event_onset",
        "activation_onset",
        "suppression_onset",
        "onset_time",
        "onset",
    ):
        if col in row.index:
            value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if np.isfinite(value):
                return float(value)
    return np.nan


def _event_auc(row: pd.Series) -> float:
    for col in ("AUC", "integrated_NES", "auc", "delta_auc", "effect_size"):
        if col in row.index:
            value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if np.isfinite(value):
                return float(value)
    return np.nan


def _event_peak_time(row: pd.Series) -> float:
    for col in ("peak_time", "time_peak", "real_time_peak", "pseudotime_peak"):
        if col in row.index:
            value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if np.isfinite(value):
                return float(value)
    return np.nan


def _event_terminal(row: pd.Series) -> float:
    for col in (
        "terminal_output",
        "terminal_score",
        "final_score",
        "terminal_NES",
        "last_score",
        "peak_NES",
    ):
        if col in row.index:
            value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if np.isfinite(value):
                return float(value)
    auc = _event_auc(row)
    return float(np.sign(auc) * abs(auc)) if np.isfinite(auc) else np.nan


def _infer_id_columns(
    left: pd.DataFrame,
    right: Optional[pd.DataFrame] = None,
    *,
    requested: Optional[Sequence[str]] = None,
) -> list[str]:
    if requested:
        return [col for col in requested if col in left.columns and (right is None or col in right.columns)]
    candidates = [
        "event_id",
        "pathway",
        "Pathway",
        "pathway_or_module",
        "family_id",
        "lineage",
        "trajectory",
        "cell_type",
    ]
    ids = [col for col in candidates if col in left.columns and (right is None or col in right.columns)]
    if not ids:
        left = left.copy()
        left["_event_row"] = np.arange(len(left))
        return ["_event_row"]
    first_feature = next(
        (col for col in ids if col.lower() in {"event_id", "pathway", "pathway_or_module", "family_id"}),
        ids[0],
    )
    lineage_like = [col for col in ids if col != first_feature and col.lower() in {"lineage", "trajectory", "cell_type"}]
    return [first_feature] + lineage_like[:1]


def _correlation(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    finite = np.isfinite(xv) & np.isfinite(yv)
    if finite.sum() < 3 or np.nanstd(xv[finite]) <= 0 or np.nanstd(yv[finite]) <= 0:
        return np.nan, np.nan
    r, p = stats.pearsonr(xv[finite], yv[finite])
    return float(r), float(p)


def _bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    idx = np.where(finite)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def run_ted_time(
    score_table: pd.DataFrame,
    *,
    pathway_col: Optional[str] = None,
    score_col: Optional[str] = None,
    real_time_col: str = "real_time",
    pseudotime_col: str = "pseudotime",
    lineage_col: Optional[str] = "lineage",
    fdr_col: Optional[str] = None,
    fdr_threshold: float = 0.05,
    score_threshold: float = 0.0,
    min_consecutive: int = 1,
    cell_metadata: Optional[pd.DataFrame] = None,
) -> dict[str, pd.DataFrame]:
    """Fuse real time and pseudotime event timing into a TED-Time table."""

    if score_table is None or len(score_table) == 0:
        return {"ted_time_event_table": pd.DataFrame()}
    table = score_table.copy()
    pathway_col = pathway_col or _pick_column(table, "Pathway", "pathway", "event_id", "pathway_or_module")
    score_col = score_col or _pick_column(table, "NES", "score", "module_score", "event_score")
    fdr_col = _pick_column(table, fdr_col, "padj", "q", "event_q", "window_q") if fdr_col or "q" in table.columns else _pick_column(table, "padj", "q")
    lineage_col = _pick_column(table, lineage_col, "trajectory", "branch", "lineage") if lineage_col else None
    if pathway_col is None:
        raise ValueError("Could not find a pathway/event column")
    missing = [col for col in (score_col, real_time_col, pseudotime_col) if col is None or col not in table.columns]
    if missing:
        raise ValueError(f"Missing required TED-Time columns: {missing}")

    group_cols = [pathway_col] + ([lineage_col] if lineage_col and lineage_col in table.columns else [])
    cell_meta = _as_frame(cell_metadata) if cell_metadata is not None else pd.DataFrame()
    rows = []
    for keys, group in table.groupby(group_cols, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        real_curve = _curve_by_axis(group, axis_col=real_time_col, score_col=score_col, fdr_col=fdr_col)
        pseudo_curve = _curve_by_axis(group, axis_col=pseudotime_col, score_col=score_col, fdr_col=fdr_col)
        if real_curve.empty and pseudo_curve.empty:
            continue
        direction = _dominant_direction(pd.concat([real_curve.get("score", pd.Series(dtype=float)), pseudo_curve.get("score", pd.Series(dtype=float))]))
        real_onset = _stable_onset(
            real_curve[real_time_col],
            real_curve["score"],
            q_values=real_curve["q"] if "q" in real_curve else None,
            fdr_threshold=fdr_threshold,
            score_threshold=score_threshold,
            min_consecutive=min_consecutive,
            direction=direction,
        )
        pseudo_onset = _stable_onset(
            pseudo_curve[pseudotime_col],
            pseudo_curve["score"],
            q_values=pseudo_curve["q"] if "q" in pseudo_curve else None,
            fdr_threshold=fdr_threshold,
            score_threshold=score_threshold,
            min_consecutive=min_consecutive,
            direction=direction,
        )
        real_norm = _norm(real_onset, real_curve[real_time_col]) if not real_curve.empty else np.nan
        pseudo_norm = _norm(pseudo_onset, pseudo_curve[pseudotime_col]) if not pseudo_curve.empty else np.nan
        disagreement = abs(real_norm - pseudo_norm) if np.isfinite(real_norm) and np.isfinite(pseudo_norm) else np.nan

        if not cell_meta.empty and real_time_col in cell_meta.columns and pseudotime_col in cell_meta.columns:
            async_source = cell_meta
            if lineage_col and lineage_col in cell_meta.columns and len(keys) > 1:
                async_source = async_source[async_source[lineage_col].astype(str) == str(keys[1])]
        else:
            async_source = group
        pseudo_span = pd.to_numeric(async_source[pseudotime_col], errors="coerce")
        span = float(np.nanmax(pseudo_span) - np.nanmin(pseudo_span)) if np.isfinite(pseudo_span).any() else np.nan
        iqr_by_real = []
        if span and np.isfinite(span) and span > 0:
            for _, real_group in async_source.groupby(real_time_col, dropna=True):
                vals = pd.to_numeric(real_group[pseudotime_col], errors="coerce").dropna()
                if len(vals) >= 2:
                    iqr_by_real.append(float((vals.quantile(0.75) - vals.quantile(0.25)) / span))
        asynchrony = float(np.nanmedian(iqr_by_real)) if iqr_by_real else np.nan

        has_real = np.isfinite(real_onset)
        has_pseudo = np.isfinite(pseudo_onset)
        if has_real and has_pseudo and np.isfinite(disagreement) and disagreement <= 0.15:
            confidence = "high_real_pseudo_concordant"
        elif has_real and has_pseudo:
            confidence = "medium_axis_disagreement"
        elif has_real or has_pseudo:
            confidence = "axis_specific"
        else:
            confidence = "low_no_stable_onset"

        record = {
            pathway_col: keys[0],
            "real_time_onset": real_onset,
            "pseudotime_onset": pseudo_onset,
            "real_time_peak": float(real_curve.loc[real_curve["score"].abs().idxmax(), real_time_col]) if not real_curve.empty else np.nan,
            "pseudotime_peak": float(pseudo_curve.loc[pseudo_curve["score"].abs().idxmax(), pseudotime_col]) if not pseudo_curve.empty else np.nan,
            "event_direction": "activation" if direction > 0 else "suppression",
            "asynchrony_score": asynchrony,
            "event_time_confidence": confidence,
            "real_vs_pseudo_disagreement": disagreement,
            "n_observations": int(len(group)),
        }
        if lineage_col and len(keys) > 1:
            record[lineage_col] = keys[1]
        rows.append(record)

    out = pd.DataFrame(rows)
    if not out.empty:
        sort_cols = ["event_time_confidence", "real_vs_pseudo_disagreement", pathway_col]
        out = out.sort_values(sort_cols, ascending=[True, True, True], na_position="last").reset_index(drop=True)
    return {"ted_time_event_table": out}


def classify_ted_delay_modes(
    reference_events: pd.DataFrame,
    comparison_events: pd.DataFrame,
    *,
    id_cols: Optional[Sequence[str]] = None,
    auc_threshold: float = 0.0,
    time_shift_threshold: float = 0.05,
    terminal_threshold: float = 0.0,
    recovery_fraction: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """Classify perturbation effects as delay, loss, pulse, redirection, or accumulation."""

    ref = _as_frame(reference_events)
    comp = _as_frame(comparison_events)
    if ref.empty or comp.empty:
        return {"developmental_event_mode": pd.DataFrame()}
    ids = _infer_id_columns(ref, comp, requested=id_cols)
    ref_work = ref.copy()
    comp_work = comp.copy()
    if ids == ["_event_row"]:
        ref_work["_event_row"] = np.arange(len(ref_work))
        comp_work["_event_row"] = np.arange(len(comp_work))
    merged = ref_work.merge(comp_work, on=ids, suffixes=("_reference", "_comparison"), how="inner")
    rows = []
    for _, row in merged.iterrows():
        ref_row = row.filter(regex="_reference$").rename(lambda x: x.removesuffix("_reference"))
        comp_row = row.filter(regex="_comparison$").rename(lambda x: x.removesuffix("_comparison"))
        ref_auc = _event_auc(ref_row)
        comp_auc = _event_auc(comp_row)
        ref_peak = _event_peak_time(ref_row)
        comp_peak = _event_peak_time(comp_row)
        ref_onset = _event_onset(ref_row)
        comp_onset = _event_onset(comp_row)
        ref_terminal = _event_terminal(ref_row)
        comp_terminal = _event_terminal(comp_row)
        delta_auc = comp_auc - ref_auc if np.isfinite(comp_auc) and np.isfinite(ref_auc) else np.nan
        delta_peak = comp_peak - ref_peak if np.isfinite(comp_peak) and np.isfinite(ref_peak) else np.nan
        delta_onset = comp_onset - ref_onset if np.isfinite(comp_onset) and np.isfinite(ref_onset) else np.nan
        delta_terminal = comp_terminal - ref_terminal if np.isfinite(comp_terminal) and np.isfinite(ref_terminal) else np.nan
        ref_duration = pd.to_numeric(pd.Series([ref_row.get("duration", np.nan)]), errors="coerce").iloc[0]
        comp_duration = pd.to_numeric(pd.Series([comp_row.get("duration", np.nan)]), errors="coerce").iloc[0]
        finite_durations = [value for value in (ref_duration, comp_duration) if np.isfinite(value)]
        duration = max(finite_durations) if finite_durations else np.nan
        intermediate_delta = pd.to_numeric(
            pd.Series([comp_row.get("intermediate_state_score", np.nan)]), errors="coerce"
        ).iloc[0] - pd.to_numeric(pd.Series([ref_row.get("intermediate_state_score", np.nan)]), errors="coerce").iloc[0]
        original_lineage_delta = pd.to_numeric(
            pd.Series([comp_row.get("original_lineage_score", np.nan)]), errors="coerce"
        ).iloc[0] - pd.to_numeric(pd.Series([ref_row.get("original_lineage_score", np.nan)]), errors="coerce").iloc[0]
        alternative_lineage_delta = pd.to_numeric(
            pd.Series([comp_row.get("alternative_lineage_score", np.nan)]), errors="coerce"
        ).iloc[0] - pd.to_numeric(pd.Series([ref_row.get("alternative_lineage_score", np.nan)]), errors="coerce").iloc[0]

        partly_recovers = (
            np.isfinite(delta_terminal)
            and np.isfinite(ref_terminal)
            and abs(delta_terminal) <= max(abs(ref_terminal) * recovery_fraction, terminal_threshold)
        )
        terminal_loss_threshold = max(
            abs(ref_terminal) * (1.0 - recovery_fraction) if np.isfinite(ref_terminal) else 0.0,
            abs(terminal_threshold),
        )
        no_later_recovery = np.isfinite(delta_terminal) and delta_terminal < -terminal_loss_threshold
        auc_loss = np.isfinite(delta_auc) and delta_auc < -abs(auc_threshold)
        peak_shift = np.isfinite(delta_peak) and delta_peak > abs(time_shift_threshold)
        onset_shift = np.isfinite(delta_onset) and delta_onset > abs(time_shift_threshold)
        modest_auc = np.isfinite(delta_auc) and abs(delta_auc) <= max(abs(ref_auc) * 0.25 if np.isfinite(ref_auc) else 0, auc_threshold)
        short_duration = np.isfinite(duration) and duration <= 2 * abs(time_shift_threshold)

        if np.isfinite(original_lineage_delta) and np.isfinite(alternative_lineage_delta) and original_lineage_delta < 0 < alternative_lineage_delta:
            mode = "fate_redirection"
        elif np.isfinite(intermediate_delta) and np.isfinite(delta_terminal) and intermediate_delta > 0 and delta_terminal < 0:
            mode = "transient_accumulation"
        elif auc_loss and no_later_recovery:
            mode = "true_loss"
        elif (peak_shift or onset_shift) and partly_recovers:
            mode = "developmental_delay"
        elif peak_shift and short_duration and modest_auc:
            mode = "pulse_shift"
        else:
            mode = "ambiguous_or_mixed"

        record = {col: row[col] for col in ids if col in row.index}
        record.update(
            {
                "delta_auc": delta_auc,
                "delta_peak_time": delta_peak,
                "delta_onset_time": delta_onset,
                "delta_terminal_output": delta_terminal,
                "terminal_output_reference": ref_terminal,
                "terminal_output_comparison": comp_terminal,
                "developmental_event_mode": mode,
                "event_mode_rule": (
                    "lineage_redirection"
                    if mode == "fate_redirection"
                    else "intermediate_gain_terminal_loss"
                    if mode == "transient_accumulation"
                    else "auc_loss_without_recovery"
                    if mode == "true_loss"
                    else "late_shift_with_partial_recovery"
                    if mode == "developmental_delay"
                    else "short_duration_peak_shift"
                    if mode == "pulse_shift"
                    else "heuristics_inconclusive"
                ),
            }
        )
        rows.append(record)
    return {"developmental_event_mode": pd.DataFrame(rows)}


def _sinkhorn_coupling(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epsilon: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    cost = spatial.distance.cdist(x, y, metric="sqeuclidean")
    finite = cost[np.isfinite(cost)]
    scale = float(np.nanmedian(finite[finite > 0])) if np.any(finite > 0) else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    cost = cost / scale
    eps = max(float(epsilon), 1e-6)
    kernel = np.exp(-cost / eps)
    kernel = np.maximum(kernel, 1e-300)
    a = np.full(x.shape[0], 1.0 / x.shape[0])
    b = np.full(y.shape[0], 1.0 / y.shape[0])
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(int(max_iter)):
        prev_u = u.copy()
        u = a / np.maximum(kernel @ v, 1e-300)
        v = b / np.maximum(kernel.T @ u, 1e-300)
        if np.nanmax(np.abs(u - prev_u)) < tol:
            break
    coupling = (u[:, None] * kernel) * v[None, :]
    total = float(coupling.sum())
    if total > 0:
        coupling = coupling / total
    return coupling, cost


def run_ted_ot_dynamic(
    feature_matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    time_col: str = "real_time",
    cell_id_col: Optional[str] = None,
    event_scores: Optional[pd.DataFrame] = None,
    condition_col: Optional[str] = None,
    reference_label: Optional[str] = None,
    case_label: Optional[str] = None,
    fate_probability: Optional[pd.DataFrame] = None,
    fate_probability_cols: Optional[Sequence[str]] = None,
    lineage_col: Optional[str] = None,
    epsilon: float = 0.05,
    top_k: int = 5,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> dict[str, pd.DataFrame]:
    """Run a lightweight TED optimal-transport dynamic matching analysis."""

    features = _as_frame(feature_matrix)
    meta = _as_frame(metadata, index=features.index)
    if time_col not in meta.columns:
        raise KeyError(f"time_col '{time_col}' not found in metadata")
    scores = _as_frame(event_scores, index=features.index) if event_scores is not None else pd.DataFrame(index=features.index)
    fate = _as_frame(fate_probability, index=features.index) if fate_probability is not None else pd.DataFrame(index=features.index)
    if fate_probability_cols:
        missing = [col for col in fate_probability_cols if col not in fate.columns and col not in meta.columns]
        if missing:
            raise KeyError(f"fate probability columns not found: {missing}")
        for col in fate_probability_cols:
            if col not in fate.columns and col in meta.columns:
                fate[col] = meta[col]

    numeric_features = features.apply(pd.to_numeric, errors="coerce")
    keep_feature_cols = numeric_features.columns[numeric_features.notna().any()].tolist()
    numeric_features = numeric_features[keep_feature_cols].fillna(numeric_features[keep_feature_cols].median()).fillna(0.0)
    if numeric_features.empty:
        raise ValueError("feature_matrix must contain at least one numeric column")
    feature_sd = numeric_features.std(axis=0).replace(0, 1.0)
    numeric_features = (numeric_features - numeric_features.mean(axis=0)) / feature_sd
    cell_ids = meta[cell_id_col].astype(str) if cell_id_col and cell_id_col in meta.columns else pd.Series(features.index.astype(str), index=features.index)
    times = sorted(pd.to_numeric(meta[time_col], errors="coerce").dropna().unique())
    lineage_groups: list[tuple[str, pd.Index]]
    if lineage_col and lineage_col in meta.columns:
        lineage_groups = [(str(label), idx) for label, idx in meta.groupby(lineage_col, sort=False).groups.items()]
    else:
        lineage_groups = [("all", meta.index)]

    coupling_rows = []
    flow_rows = []
    for lineage, lineage_idx in lineage_groups:
        meta_l = meta.loc[lineage_idx]
        for t0, t1 in zip(times[:-1], times[1:]):
            src_idx = meta_l.index[pd.to_numeric(meta_l[time_col], errors="coerce") == t0]
            tgt_idx = meta_l.index[pd.to_numeric(meta_l[time_col], errors="coerce") == t1]
            if len(src_idx) == 0 or len(tgt_idx) == 0:
                continue
            coupling, cost = _sinkhorn_coupling(
                numeric_features.loc[src_idx].to_numpy(dtype=float),
                numeric_features.loc[tgt_idx].to_numpy(dtype=float),
                epsilon=epsilon,
                max_iter=max_iter,
                tol=tol,
            )
            row_mass = coupling.sum(axis=1)
            col_mass = coupling.sum(axis=0)
            for i, src in enumerate(src_idx):
                order = np.argsort(coupling[i])[::-1][: max(int(top_k), 1)]
                for j in order:
                    if coupling[i, j] <= 0:
                        continue
                    coupling_rows.append(
                        {
                            "lineage": lineage,
                            "source_cell_id": cell_ids.loc[src],
                            "target_cell_id": cell_ids.loc[tgt_idx[j]],
                            "source_time": t0,
                            "target_time": t1,
                            "transport_weight": float(coupling[i, j]),
                            "transport_cost": float(cost[i, j]),
                        }
                    )
            if not scores.empty:
                score_numeric = scores.apply(pd.to_numeric, errors="coerce")
                for event in score_numeric.columns:
                    src_values = score_numeric.loc[src_idx, event].to_numpy(dtype=float)
                    tgt_values = score_numeric.loc[tgt_idx, event].to_numpy(dtype=float)
                    if not (np.isfinite(src_values).any() and np.isfinite(tgt_values).any()):
                        continue
                    src_mean = float(np.nansum(row_mass * np.nan_to_num(src_values, nan=np.nanmean(src_values))))
                    expected_future = float(np.nansum(col_mass * np.nan_to_num(tgt_values, nan=np.nanmean(tgt_values))))
                    flow_rows.append(
                        {
                            "lineage": lineage,
                            "event": event,
                            "source_time": t0,
                            "target_time": t1,
                            "transported_source_score": src_mean,
                            "expected_future_score": expected_future,
                            "event_transport_delta": expected_future - src_mean,
                            "n_source_cells": int(len(src_idx)),
                            "n_target_cells": int(len(tgt_idx)),
                        }
                    )

    counterfactual_rows = []
    if (
        condition_col
        and condition_col in meta.columns
        and reference_label is not None
        and case_label is not None
        and not scores.empty
    ):
        score_numeric = scores.apply(pd.to_numeric, errors="coerce")
        for t0, t1 in zip(times[:-1], times[1:]):
            case_src = meta.index[(pd.to_numeric(meta[time_col], errors="coerce") == t0) & (meta[condition_col].astype(str) == str(case_label))]
            case_future = meta.index[(pd.to_numeric(meta[time_col], errors="coerce") == t1) & (meta[condition_col].astype(str) == str(case_label))]
            ref_future = meta.index[(pd.to_numeric(meta[time_col], errors="coerce") == t1) & (meta[condition_col].astype(str) == str(reference_label))]
            if len(case_src) == 0 or len(case_future) == 0 or len(ref_future) == 0:
                continue
            coupling, _ = _sinkhorn_coupling(
                numeric_features.loc[case_src].to_numpy(dtype=float),
                numeric_features.loc[ref_future].to_numpy(dtype=float),
                epsilon=epsilon,
                max_iter=max_iter,
                tol=tol,
            )
            ref_mass = coupling.sum(axis=0)
            for event in score_numeric.columns:
                obs_case = float(np.nanmean(score_numeric.loc[case_future, event]))
                matched_ref = float(np.nansum(ref_mass * np.nan_to_num(score_numeric.loc[ref_future, event].to_numpy(dtype=float), nan=0.0)))
                counterfactual_rows.append(
                    {
                        "event": event,
                        "source_time": t0,
                        "future_time": t1,
                        "case_label": case_label,
                        "reference_label": reference_label,
                        "observed_case_future_score": obs_case,
                        "matched_reference_future_score": matched_ref,
                        "counterfactual_event_loss": matched_ref - obs_case,
                        "n_case_source_cells": int(len(case_src)),
                        "n_case_future_cells": int(len(case_future)),
                        "n_reference_future_cells": int(len(ref_future)),
                    }
                )

    fate_rows = []
    if not scores.empty and not fate.empty:
        score_numeric = scores.apply(pd.to_numeric, errors="coerce")
        fate_numeric = fate.apply(pd.to_numeric, errors="coerce")
        for event in score_numeric.columns:
            for fate_col in fate_numeric.columns:
                r, p = _correlation(score_numeric[event], fate_numeric[fate_col])
                fate_rows.append(
                    {
                        "event": event,
                        "future_fate": fate_col,
                        "event_fate_correlation": r,
                        "event_fate_p": p,
                        "n_cells": int((np.isfinite(score_numeric[event]) & np.isfinite(fate_numeric[fate_col])).sum()),
                    }
                )
        fate_df = pd.DataFrame(fate_rows)
        if not fate_df.empty:
            fate_df["event_fate_q"] = _bh_fdr(fate_df["event_fate_p"])
            fate_rows = fate_df.to_dict("records")

    return {
        "ot_cell_couplings": pd.DataFrame(coupling_rows),
        "ot_event_flow": pd.DataFrame(flow_rows),
        "counterfactual_event_loss": pd.DataFrame(counterfactual_rows),
        "fate_probability_linked_event": pd.DataFrame(fate_rows),
    }


def _standard_event_table(table: Optional[pd.DataFrame], *, modality: str, event_col: Optional[str] = None) -> pd.DataFrame:
    if table is None or len(table) == 0:
        return pd.DataFrame(columns=["event_id", "onset", "direction", "direction_stability", "modality"])
    df = table.copy()
    event_col = event_col or _pick_column(df, "event_id", "pathway", "Pathway", "motif", "feature", "gene", "pathway_or_module")
    onset_col = _pick_column(df, "onset", "onset_time", "activation_onset", "real_time_onset", "pseudotime_onset", "peak_time")
    direction_col = _pick_column(df, "direction", "event_direction", "family_direction", "effect_direction")
    stability_col = _pick_column(df, "direction_stability", "event_stability", "replicate_direction_stability")
    if event_col is None or onset_col is None:
        raise ValueError(f"{modality} event table needs event and onset columns")
    out = pd.DataFrame(
        {
            "event_id": df[event_col].astype(str),
            "onset": pd.to_numeric(df[onset_col], errors="coerce"),
            "direction": df[direction_col].astype(str) if direction_col else "unknown",
            "direction_stability": pd.to_numeric(df[stability_col], errors="coerce") if stability_col else 1.0,
            "modality": modality,
        }
    )
    return out


def _direction_sign(value: Any) -> int:
    text = str(value).lower()
    if any(token in text for token in ("down", "loss", "suppression", "decrease", "negative")):
        return -1
    if any(token in text for token in ("up", "gain", "activation", "increase", "positive")):
        return 1
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if np.isfinite(numeric):
        return 1 if numeric >= 0 else -1
    return 0


def run_ted_multiome_lag(
    *,
    atac_events: Optional[pd.DataFrame] = None,
    motif_events: Optional[pd.DataFrame] = None,
    rna_events: Optional[pd.DataFrame] = None,
    pathway_events: Optional[pd.DataFrame] = None,
    cell_state_events: Optional[pd.DataFrame] = None,
    target_links: Optional[pd.DataFrame] = None,
    min_driver_lag_score: float = 0.25,
) -> dict[str, pd.DataFrame]:
    """Compute ATAC/RNA/fate lag tables and chromatin-first mechanism candidates."""

    atac = _standard_event_table(atac_events, modality="ATAC")
    rna = _standard_event_table(rna_events, modality="RNA")
    motif = _standard_event_table(motif_events, modality="motif") if motif_events is not None else pd.DataFrame()
    pathway = _standard_event_table(pathway_events, modality="pathway") if pathway_events is not None else pd.DataFrame()
    cell_state = _standard_event_table(cell_state_events, modality="cell_state") if cell_state_events is not None else pd.DataFrame()

    lag_rows = []
    if not atac.empty and not rna.empty:
        merged = atac.merge(rna, on="event_id", suffixes=("_ATAC", "_RNA"), how="outer")
        span_values = pd.concat([merged["onset_ATAC"], merged["onset_RNA"]], ignore_index=True)
        span = float(span_values.max() - span_values.min()) if np.isfinite(span_values).any() else 1.0
        if not np.isfinite(span) or span <= 0:
            span = 1.0
        for _, row in merged.iterrows():
            lag = row["onset_RNA"] - row["onset_ATAC"] if np.isfinite(row.get("onset_RNA", np.nan)) and np.isfinite(row.get("onset_ATAC", np.nan)) else np.nan
            concordance = int(_direction_sign(row.get("direction_ATAC", "")) == _direction_sign(row.get("direction_RNA", ""))) if pd.notna(row.get("direction_ATAC", np.nan)) and pd.notna(row.get("direction_RNA", np.nan)) else 0
            stability = float(np.nanmean([row.get("direction_stability_ATAC", np.nan), row.get("direction_stability_RNA", np.nan)]))
            if not np.isfinite(stability):
                stability = 1.0
            lead = max(float(lag), 0.0) / span if np.isfinite(lag) else 0.0
            lag_rows.append(
                {
                    "event_id": row["event_id"],
                    "ATAC_onset": row.get("onset_ATAC", np.nan),
                    "RNA_onset": row.get("onset_RNA", np.nan),
                    "Lag_ATAC_to_RNA": lag,
                    "ATAC_RNA_direction_concordance": bool(concordance),
                    "direction_stability": stability,
                    "DriverLagScore": float(lead * concordance * stability),
                    "claim_grade": "ATAC->RNA supportive" if concordance and np.isfinite(lag) and lag >= 0 else "ATAC-only candidate",
                }
            )
    lag_table = pd.DataFrame(lag_rows)

    if target_links is not None and not motif.empty:
        links = target_links.copy()
        motif_col = _pick_column(links, "motif", "motif_id", "TF", "tf", "regulator")
        target_col = _pick_column(links, "target_event", "target", "gene", "event_id", "pathway")
        if motif_col is None or target_col is None:
            concordance = pd.DataFrame()
        else:
            concordance = links[[motif_col, target_col]].rename(columns={motif_col: "motif_id", target_col: "target_event"}).copy()
            concordance = concordance.merge(motif.rename(columns={"event_id": "motif_id"}), on="motif_id", how="left")
            concordance = concordance.merge(rna.rename(columns={"event_id": "target_event"}), on="target_event", how="left", suffixes=("_motif", "_RNA"))
            concordance["motif_leads_target_RNA"] = concordance["onset_RNA"] - concordance["onset_motif"]
            concordance["target_RNA_concordance"] = [
                _direction_sign(a) == _direction_sign(b)
                for a, b in zip(concordance.get("direction_motif", []), concordance.get("direction_RNA", []))
            ]
            concordance["DriverLagScore"] = (
                concordance["motif_leads_target_RNA"].clip(lower=0).fillna(0)
                * concordance["target_RNA_concordance"].astype(float)
                * concordance.get("direction_stability_motif", 1.0)
            )
    else:
        concordance = pd.DataFrame()

    candidate = lag_table.copy()
    if not candidate.empty:
        pathway_supported = set(pathway["event_id"].astype(str)) if "event_id" in pathway.columns else set()
        cell_state_supported = set(cell_state["event_id"].astype(str)) if "event_id" in cell_state.columns else set()
        fate_supported = pathway_supported | cell_state_supported
        candidate["fate_or_pathway_supported"] = candidate["event_id"].astype(str).isin(fate_supported)
        candidate["claim_grade"] = np.where(
            candidate["fate_or_pathway_supported"] & (candidate["DriverLagScore"] >= min_driver_lag_score),
            "ATAC->RNA->fate strong candidate",
            candidate["claim_grade"],
        )
        candidate = candidate[
            (candidate["Lag_ATAC_to_RNA"] >= 0)
            & (candidate["DriverLagScore"] >= min_driver_lag_score)
        ].reset_index(drop=True)

    return {
        "multiome_event_lag_table": lag_table,
        "motif_to_target_event_concordance": concordance,
        "chromatin_first_mechanism_candidates": candidate,
    }


def run_ted_lineage_tree(
    score_matrix: pd.DataFrame,
    lineage_edges: pd.DataFrame,
    *,
    parent_col: str = "parent",
    child_col: str = "child",
    cell_metadata: Optional[pd.DataFrame] = None,
    terminal_state_col: Optional[str] = None,
    lineage_col: Optional[str] = None,
    divergence_quantile: float = 0.75,
) -> dict[str, pd.DataFrame]:
    """Score events on true lineage-tree edges and sister branches."""

    scores = _as_frame(score_matrix)
    edges = lineage_edges.copy()
    if parent_col not in edges.columns or child_col not in edges.columns:
        raise KeyError("lineage_edges must contain parent and child columns")
    scores.index = scores.index.astype(str)
    edges[parent_col] = edges[parent_col].astype(str)
    edges[child_col] = edges[child_col].astype(str)
    numeric_scores = scores.apply(pd.to_numeric, errors="coerce")
    edge_rows = []
    for _, edge in edges.iterrows():
        parent = str(edge[parent_col])
        child = str(edge[child_col])
        if parent not in numeric_scores.index or child not in numeric_scores.index:
            continue
        delta = numeric_scores.loc[child] - numeric_scores.loc[parent]
        for event, value in delta.items():
            edge_rows.append(
                {
                    "parent_cell": parent,
                    "child_cell": child,
                    "event": event,
                    "parent_score": float(numeric_scores.loc[parent, event]),
                    "child_score": float(numeric_scores.loc[child, event]),
                    "edge_delta": float(value),
                    "branch_divergence_event": "gain" if value > 0 else "loss" if value < 0 else "flat",
                }
            )
    edge_table = pd.DataFrame(edge_rows)

    sister_rows = []
    for parent, group in edges.groupby(parent_col, sort=False):
        children = [str(child) for child in group[child_col] if str(child) in numeric_scores.index]
        parent = str(parent)
        if len(children) < 2 or parent not in numeric_scores.index:
            continue
        child_scores = numeric_scores.loc[children]
        for event in numeric_scores.columns:
            values = child_scores[event].to_numpy(dtype=float)
            divergence = float(np.nanmax(values) - np.nanmin(values)) if np.isfinite(values).any() else np.nan
            sister_rows.append(
                {
                    "parent_cell": parent,
                    "event": event,
                    "daughter_cells": ";".join(children),
                    "parent_score": float(numeric_scores.loc[parent, event]),
                    "sister_cell_asymmetry_event": divergence,
                    "highest_scoring_daughter": children[int(np.nanargmax(values))] if np.isfinite(values).any() else "",
                    "lowest_scoring_daughter": children[int(np.nanargmin(values))] if np.isfinite(values).any() else "",
                }
            )
    sister_table = pd.DataFrame(sister_rows)
    if not sister_table.empty:
        threshold = float(sister_table["sister_cell_asymmetry_event"].quantile(divergence_quantile))
        parent_scores = sister_table["parent_score"].replace([np.inf, -np.inf], np.nan)
        center = float(parent_scores.median())
        scale = float(parent_scores.std()) if float(parent_scores.std()) > 0 else 1.0
        prebranch = sister_table[sister_table["sister_cell_asymmetry_event"] >= threshold].copy()
        prebranch["prebranch_priming_z"] = (prebranch["parent_score"] - center) / scale
        prebranch["prebranch_priming_event"] = prebranch["prebranch_priming_z"] > 0
    else:
        prebranch = pd.DataFrame()

    convergence = pd.DataFrame()
    if cell_metadata is not None and terminal_state_col and lineage_col:
        meta = _as_frame(cell_metadata, index=scores.index)
        meta.index = meta.index.astype(str)
        if terminal_state_col in meta.columns and lineage_col in meta.columns:
            rows = []
            global_var = numeric_scores.var(axis=0).replace(0, np.nan)
            for state, state_idx in meta.groupby(terminal_state_col, sort=False).groups.items():
                lineages = meta.loc[state_idx, lineage_col].dropna().astype(str).unique()
                idx = pd.Index(state_idx).intersection(numeric_scores.index)
                if len(lineages) < 2 or len(idx) < 2:
                    continue
                within_var = numeric_scores.loc[idx].var(axis=0)
                for event in numeric_scores.columns:
                    gv = float(global_var[event])
                    wv = float(within_var[event])
                    score = float(1.0 - min(wv / gv, 1.0)) if np.isfinite(gv) and gv > 0 and np.isfinite(wv) else np.nan
                    rows.append(
                        {
                            "terminal_state": state,
                            "event": event,
                            "n_lineages": int(len(lineages)),
                            "n_cells": int(len(idx)),
                            "within_state_variance": wv,
                            "global_variance": gv,
                            "lineage_convergence_event": score,
                        }
                    )
            convergence = pd.DataFrame(rows)

    return {
        "lineage_edge_event_table": edge_table,
        "sister_branch_divergence_score": sister_table,
        "prebranch_event_candidates": prebranch,
        "lineage_convergence_event": convergence,
    }


def _build_knn_edges(coords: pd.DataFrame, *, k: int) -> pd.DataFrame:
    tree = spatial.cKDTree(coords.to_numpy(dtype=float))
    distances, indices = tree.query(coords.to_numpy(dtype=float), k=min(k + 1, len(coords)))
    rows = []
    for i, source in enumerate(coords.index):
        idx_values = np.atleast_1d(indices[i])[1:]
        dist_values = np.atleast_1d(distances[i])[1:]
        for j, dist in zip(idx_values, dist_values):
            rows.append({"source": source, "target": coords.index[int(j)], "distance": float(dist)})
    return pd.DataFrame(rows)


def run_ted_spatial_neighborhood(
    score_matrix: pd.DataFrame,
    spatial_metadata: pd.DataFrame,
    *,
    x_col: str = "x",
    y_col: str = "y",
    neighborhood_col: Optional[str] = "cell_type",
    neighbor_edges: Optional[pd.DataFrame] = None,
    k_neighbors: int = 6,
    axis_cols: Optional[Sequence[str]] = None,
    time_col: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """Score spatial neighborhood, boundary, axis, and propagation events."""

    scores = _as_frame(score_matrix)
    meta = _as_frame(spatial_metadata, index=scores.index)
    scores.index = scores.index.astype(str)
    meta.index = meta.index.astype(str)
    numeric_scores = scores.apply(pd.to_numeric, errors="coerce")
    coords = meta[[x_col, y_col]].apply(pd.to_numeric, errors="coerce") if x_col in meta.columns and y_col in meta.columns else pd.DataFrame()
    if neighbor_edges is None:
        if coords.empty:
            raise ValueError("Either neighbor_edges or numeric x/y coordinates are required")
        edges = _build_knn_edges(coords.dropna(), k=k_neighbors)
    else:
        edges = neighbor_edges.copy()
        edges = edges.rename(
            columns={
                _pick_column(edges, "source", "cell", "spot", "from"): "source",
                _pick_column(edges, "target", "neighbor", "to"): "target",
            }
        )
    neighborhood_col = _pick_column(meta, neighborhood_col, "cell_type", "neighborhood", "organ", "region") if neighborhood_col else None

    neighborhood_rows = []
    if neighborhood_col:
        global_mean = numeric_scores.mean(axis=0)
        for label, idx in meta.groupby(neighborhood_col, sort=False).groups.items():
            idx = pd.Index(idx).intersection(numeric_scores.index)
            if len(idx) == 0:
                continue
            source_mask = edges["source"].astype(str).isin(idx.astype(str))
            neighbor_idx = pd.Index(edges.loc[source_mask, "target"].astype(str)).intersection(numeric_scores.index)
            for event in numeric_scores.columns:
                inside = float(numeric_scores.loc[idx, event].mean())
                neighbor = float(numeric_scores.loc[neighbor_idx, event].mean()) if len(neighbor_idx) else np.nan
                neighborhood_rows.append(
                    {
                        "neighborhood": label,
                        "event": event,
                        "inside_mean_score": inside,
                        "neighbor_mean_score": neighbor,
                        "global_mean_score": float(global_mean[event]),
                        "neighborhood_event": inside - float(global_mean[event]),
                        "neighborhood_boundary_delta": inside - neighbor if np.isfinite(neighbor) else np.nan,
                        "n_cells": int(len(idx)),
                        "n_neighbor_cells": int(len(neighbor_idx)),
                    }
                )

    boundary_rows = []
    if neighborhood_col:
        labels = meta[neighborhood_col].astype(str)
        for _, edge in edges.iterrows():
            source = str(edge["source"])
            target = str(edge["target"])
            if source not in labels.index or target not in labels.index or labels.loc[source] == labels.loc[target]:
                continue
            pair = "|".join(sorted([labels.loc[source], labels.loc[target]]))
            diff = numeric_scores.loc[source] - numeric_scores.loc[target]
            for event, value in diff.items():
                boundary_rows.append({"boundary": pair, "event": event, "signed_edge_delta": float(value), "abs_edge_delta": float(abs(value))})
    boundary = pd.DataFrame(boundary_rows)
    if not boundary.empty:
        boundary = (
            boundary.groupby(["boundary", "event"], as_index=False)
            .agg(boundary_specific_event=("signed_edge_delta", "mean"), boundary_abs_delta=("abs_edge_delta", "mean"), n_boundary_edges=("abs_edge_delta", "size"))
            .sort_values(["boundary_abs_delta", "boundary", "event"], ascending=[False, True, True])
            .reset_index(drop=True)
        )

    propagation_rows = []
    axis_candidates = list(axis_cols or [])
    for col in (x_col, y_col):
        if col in meta.columns and col not in axis_candidates:
            axis_candidates.append(col)
    if not coords.empty:
        center = coords.mean(axis=0)
        dist = np.sqrt(((coords - center) ** 2).sum(axis=1))
        meta["_distance_from_center"] = dist
        axis_candidates.append("_distance_from_center")
    for event in numeric_scores.columns:
        for axis in axis_candidates:
            if axis not in meta.columns:
                continue
            r, p = _correlation(meta[axis], numeric_scores[event])
            propagation_rows.append(
                {
                    "event": event,
                    "spatial_axis": axis,
                    "axis_correlation": r,
                    "axis_p": p,
                    "spatial_propagation_event": "increasing_along_axis" if np.isfinite(r) and r > 0 else "decreasing_along_axis" if np.isfinite(r) and r < 0 else "unclear",
                }
            )
        if time_col and time_col in meta.columns:
            r, p = _correlation(meta[time_col], numeric_scores[event])
            propagation_rows.append(
                {
                    "event": event,
                    "spatial_axis": time_col,
                    "axis_correlation": r,
                    "axis_p": p,
                    "spatial_propagation_event": "time_linked_spatial_wave" if np.isfinite(r) else "unclear",
                }
            )
    propagation = pd.DataFrame(propagation_rows)
    if not propagation.empty:
        propagation["axis_q"] = _bh_fdr(propagation["axis_p"])

    return {
        "spatial_neighborhood_event_table": pd.DataFrame(neighborhood_rows),
        "boundary_specific_event": boundary,
        "spatial_event_propagation": propagation,
    }


def _grammar_for_text(text: str) -> str:
    lowered = str(text).lower()
    best = "unmapped_event_grammar"
    best_hits = 0
    for grammar, _definition, keywords in _GRAMMAR_ROWS:
        tokens = [token.strip().lower() for token in keywords.split(";")]
        hits = sum(token in lowered for token in tokens if token)
        if hits > best_hits:
            best_hits = hits
            best = grammar
    return best


def build_cross_kingdom_event_ontology(
    event_tables: Optional[Mapping[str, pd.DataFrame] | Sequence[pd.DataFrame]] = None,
    *,
    species_gene_sets: Optional[pd.DataFrame | Mapping[str, Mapping[str, Sequence[str]]]] = None,
    orthology_table: Optional[pd.DataFrame] = None,
    event_col: Optional[str] = None,
    species_col: str = "species",
) -> dict[str, pd.DataFrame]:
    """Build cross-kingdom event grammar tables without forcing one-to-one gene matches."""

    ontology = pd.DataFrame(
        [
            {
                "event_grammar": grammar,
                "mapping_level": "Level 3 analogous event grammar",
                "definition": definition,
                "keyword_examples": keywords,
            }
            for grammar, definition, keywords in _GRAMMAR_ROWS
        ]
    )

    if species_gene_sets is None:
        mapping = pd.DataFrame()
    elif isinstance(species_gene_sets, Mapping):
        rows = []
        for species, sets in species_gene_sets.items():
            for gene_set, genes in sets.items():
                rows.append(
                    {
                        "species": species,
                        "gene_set": gene_set,
                        "event_grammar": _grammar_for_text(gene_set),
                        "genes": ";".join(map(str, genes)),
                        "mapping_level": "Level 2 conserved pathway/process",
                    }
                )
        mapping = pd.DataFrame(rows)
    else:
        mapping = species_gene_sets.copy()
        gene_set_col = _pick_column(mapping, "gene_set", "pathway", "Pathway", "term", "name")
        if gene_set_col:
            mapping["event_grammar"] = mapping.get("event_grammar", mapping[gene_set_col].map(_grammar_for_text))
        if "mapping_level" not in mapping.columns:
            mapping["mapping_level"] = "Level 2 conserved pathway/process"

    if orthology_table is None:
        orthology = pd.DataFrame()
    else:
        orthology = orthology_table.copy()
        confidence_col = _pick_column(orthology, "orthology_confidence", "confidence", "score", "identity")
        if confidence_col:
            orthology["orthology_confidence"] = pd.to_numeric(orthology[confidence_col], errors="coerce")
        elif "mapping_level" in orthology.columns:
            orthology["orthology_confidence"] = np.where(
                orthology["mapping_level"].astype(str).str.contains("exact|ortholog|Level 1", case=False, regex=True),
                1.0,
                0.5,
            )
        else:
            orthology["orthology_confidence"] = np.nan

    similarity_rows = []
    if event_tables is not None:
        if isinstance(event_tables, Mapping):
            iterable = event_tables.items()
        else:
            iterable = [(f"dataset_{idx}", table) for idx, table in enumerate(event_tables)]
        normalized = []
        for species_name, table in iterable:
            df = table.copy()
            event_name_col = event_col or _pick_column(df, "event_id", "pathway", "Pathway", "event_label", "pathway_or_module")
            if event_name_col is None:
                continue
            species_values = df[species_col].astype(str) if species_col in df.columns else pd.Series(str(species_name), index=df.index)
            for event_name, species in zip(df[event_name_col].astype(str), species_values):
                normalized.append({"species": species, "event": event_name, "grammar": _grammar_for_text(event_name)})
        norm_df = pd.DataFrame(normalized)
        if not norm_df.empty:
            by_grammar = norm_df.groupby("grammar")
            for grammar, group in by_grammar:
                species = sorted(group["species"].unique())
                if len(species) < 2:
                    continue
                similarity_rows.append(
                    {
                        "event_grammar": grammar,
                        "species": ";".join(species),
                        "n_species": int(len(species)),
                        "n_events": int(len(group)),
                        "event_grammar_similarity": 1.0,
                        "comparison_basis": "shared analogous event grammar",
                    }
                )
    similarity = pd.DataFrame(similarity_rows)

    return {
        "cross_kingdom_event_ontology": ontology,
        "species_gene_set_mapping": mapping,
        "orthology_confidence": orthology,
        "event_grammar_similarity": similarity,
    }


def assign_developmental_claim_ceiling(
    event_table: pd.DataFrame,
    *,
    event_q_col: Optional[str] = None,
    robustness_q_col: Optional[str] = None,
    replicate_support_col: Optional[str] = None,
    perturbation_support_col: Optional[str] = None,
    multiome_support_col: Optional[str] = None,
    functional_validation_col: Optional[str] = None,
    cross_system_col: Optional[str] = None,
    q_threshold: float = 0.05,
) -> dict[str, pd.DataFrame]:
    """Assign TED developmental claim ceilings from descriptive through causal levels."""

    table = _as_frame(event_table)
    if table.empty:
        return {"developmental_claim_ceiling": pd.DataFrame()}
    event_q_col = _pick_column(table, event_q_col, "event_q", "q", "padj", "window_fdr_min")
    robustness_q_col = _pick_column(table, robustness_q_col, "block_q", "block_perm_q", "family_block_q", "robustness_q")
    replicate_support_col = _pick_column(table, replicate_support_col, "replicate_support", "time_robust", "block_robust", "direction_stability")
    perturbation_support_col = _pick_column(table, perturbation_support_col, "perturbation_support", "driver_dependency_pass", "validation_pass")
    multiome_support_col = _pick_column(table, multiome_support_col, "multiome_support", "ATAC_RNA_direction_concordance", "fate_or_pathway_supported")
    functional_validation_col = _pick_column(table, functional_validation_col, "functional_rescue", "wet_lab_rescue_pass", "functional_validation")
    cross_system_col = _pick_column(table, cross_system_col, "cross_system_support", "cross_system_causal", "event_grammar_similarity")

    def truthy(value: Any) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if np.isfinite(numeric):
            return bool(numeric > 0)
        return str(value).strip().lower() in {"true", "yes", "pass", "supported", "validated", "robust"}

    rows = []
    for idx, row in table.iterrows():
        level = 1.0
        reasons = ["dynamic trend"]
        event_q = pd.to_numeric(pd.Series([row.get(event_q_col, np.nan)]), errors="coerce").iloc[0] if event_q_col else np.nan
        robust_q = pd.to_numeric(pd.Series([row.get(robustness_q_col, np.nan)]), errors="coerce").iloc[0] if robustness_q_col else np.nan
        replicate = truthy(row.get(replicate_support_col, False)) if replicate_support_col else False
        perturbation = truthy(row.get(perturbation_support_col, False)) if perturbation_support_col else False
        multiome = truthy(row.get(multiome_support_col, False)) if multiome_support_col else False
        functional = truthy(row.get(functional_validation_col, False)) if functional_validation_col else False
        cross_system = truthy(row.get(cross_system_col, False)) if cross_system_col else False
        if np.isfinite(event_q) and event_q <= q_threshold:
            level = max(level, 2.0)
            reasons.append("event-FDR supported")
        if (np.isfinite(robust_q) and robust_q <= q_threshold) or replicate:
            level = max(level, 3.0)
            reasons.append("block / replicate / time robust")
        if perturbation or multiome:
            level = max(level, 3.5)
            reasons.append("perturbation or multiome-supported mechanism candidate")
        if functional:
            level = max(level, 4.0)
            reasons.append("functional rescue / perturbation validation")
        if cross_system and functional:
            level = max(level, 5.0)
            reasons.append("cross-system causal mechanism")
        label = {
            1.0: "Level 1: dynamic trend",
            2.0: "Level 2: event-FDR supported",
            3.0: "Level 3: block / replicate / time robust",
            3.5: "Level 3.5: perturbation or multiome-supported mechanism candidate",
            4.0: "Level 4: functional rescue / perturbation validation",
            5.0: "Level 5: cross-system causal mechanism",
        }[level]
        rows.append(
            {
                "source_row": idx,
                "claim_level_numeric": level,
                "claim_level": label,
                "claim_ceiling": label.split(": ", 1)[1],
                "claim_rationale": "; ".join(dict.fromkeys(reasons)),
            }
        )
    ceiling = pd.concat([table.reset_index(drop=True), pd.DataFrame(rows).drop(columns=["source_row"])], axis=1)
    return {"developmental_claim_ceiling": ceiling}


def write_ted_developmental_tables(
    tables: Mapping[str, pd.DataFrame],
    outdir: str | Path,
    *,
    sep: str = "\t",
) -> dict[str, Path]:
    """Write TED developmental module tables using stable output filenames."""

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, table in tables.items():
        filename = TED_DEVELOPMENTAL_OUTPUTS.get(key, f"{key}.tsv")
        path = out / filename
        table.to_csv(path, sep=sep, index=False, na_rep="NA")
        paths[key] = path
    return paths
