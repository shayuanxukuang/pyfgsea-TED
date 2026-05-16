from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame()


def _bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return q
    idx = np.where(finite)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.clip(ranked, 0.0, 1.0)
    return q


def _align_metadata(score_matrix: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    if score_matrix.index.equals(metadata.index):
        return metadata.copy()
    shared = score_matrix.index.intersection(metadata.index)
    if len(shared) == len(score_matrix):
        return metadata.loc[score_matrix.index].copy()
    if len(metadata) == len(score_matrix):
        aligned = metadata.copy()
        aligned.index = score_matrix.index
        return aligned
    raise ValueError("metadata must share the score_matrix index or have the same number of rows")


def _as_binary_series(
    values: pd.Series,
    *,
    column: str,
    level_map: Optional[Mapping[Any, int]] = None,
) -> pd.Series:
    if level_map is not None:
        mapped = values.map(level_map)
        if mapped.isna().any():
            missing = sorted(map(str, values[mapped.isna()].unique()))
            raise ValueError(f"Unmapped factor levels in {column}: {missing}")
        return mapped.astype(float)

    if pd.api.types.is_bool_dtype(values):
        return values.astype(float)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all() and set(numeric.unique()).issubset({0, 1, 0.0, 1.0}):
        return numeric.astype(float)

    positives = {
        "1",
        "true",
        "yes",
        "case",
        "treated",
        "mut",
        "mutant",
        "t21",
        "gata1s",
        "sigata1",
        "knockdown",
        "kd",
    }
    negatives = {
        "0",
        "false",
        "no",
        "control",
        "reference",
        "euploid",
        "wt",
        "wtgata1",
        "sicon",
        "untreated",
    }
    lowered = values.astype(str).str.strip().str.lower()
    mapped = pd.Series(np.nan, index=values.index, dtype=float)
    mapped[lowered.isin(positives)] = 1.0
    mapped[lowered.isin(negatives)] = 0.0
    if mapped.isna().any():
        levels = sorted(map(str, values.dropna().unique()))
        if len(levels) == 2:
            level_to_value = {levels[0]: 0.0, levels[1]: 1.0}
            return values.astype(str).map(level_to_value).astype(float)
        raise ValueError(
            f"{column} must be binary or have an explicit factor_level_map; observed levels: {levels}"
        )
    return mapped


def _trajectory_groups(metadata: pd.DataFrame, trajectory_col: Optional[str]) -> list[tuple[str, pd.Index]]:
    if trajectory_col is None:
        return [("all", metadata.index)]
    if trajectory_col not in metadata.columns:
        raise KeyError(f"trajectory_col '{trajectory_col}' not found in metadata")
    groups: list[tuple[str, pd.Index]] = []
    for label, group in metadata.groupby(trajectory_col, sort=False):
        groups.append((str(label), group.index))
    return groups


def _make_design(
    metadata: pd.DataFrame,
    *,
    factor_columns: tuple[str, str],
    factor_level_map: Optional[Mapping[str, Mapping[Any, int]]] = None,
    strata_cols: Sequence[str] = (),
    include_strata_fixed_effects: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    missing = [col for col in factor_columns if col not in metadata.columns]
    if missing:
        raise KeyError(f"factor columns missing from metadata: {missing}")
    factor_a, factor_b = factor_columns
    maps = factor_level_map or {}
    a = _as_binary_series(metadata[factor_a], column=factor_a, level_map=maps.get(factor_a))
    b = _as_binary_series(metadata[factor_b], column=factor_b, level_map=maps.get(factor_b))
    design = pd.DataFrame(
        {
            "intercept": 1.0,
            factor_a: a,
            factor_b: b,
            "interaction": a * b,
        },
        index=metadata.index,
    )
    effect_names = [factor_a, factor_b, "interaction"]
    if include_strata_fixed_effects and strata_cols:
        missing_strata = [col for col in strata_cols if col not in metadata.columns]
        if missing_strata:
            raise KeyError(f"strata columns missing from metadata: {missing_strata}")
        strata = metadata[list(strata_cols)].astype(str).agg("|".join, axis=1)
        dummies = pd.get_dummies(strata, prefix="stratum", drop_first=True, dtype=float)
        design = pd.concat([design, dummies], axis=1)
    return design, effect_names


def _fit_ols_effects(
    y: pd.Series,
    design: pd.DataFrame,
    *,
    effect_names: Sequence[str],
) -> pd.DataFrame:
    data = pd.concat([y.rename("y"), design], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    rows = []
    if len(data) <= len(design.columns):
        for effect in effect_names:
            rows.append({"effect_type": effect, "beta": np.nan, "se": np.nan, "p": np.nan, "n": len(data)})
        return pd.DataFrame(rows)

    yv = data["y"].to_numpy(dtype=float)
    xv = data[design.columns].to_numpy(dtype=float)
    rank = int(np.linalg.matrix_rank(xv))
    beta = np.linalg.pinv(xv) @ yv
    resid = yv - xv @ beta
    df = len(yv) - rank
    mse = float(np.dot(resid, resid) / df) if df > 0 else np.nan
    cov = mse * np.linalg.pinv(xv.T @ xv) if np.isfinite(mse) else np.full((xv.shape[1], xv.shape[1]), np.nan)
    se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    coef = pd.Series(beta, index=design.columns)
    coef_se = pd.Series(se, index=design.columns)
    for effect in effect_names:
        estimate = float(coef.get(effect, np.nan))
        err = float(coef_se.get(effect, np.nan))
        if np.isfinite(err) and err > 0 and df > 0:
            p_value = float(2.0 * stats.t.sf(abs(estimate / err), df))
        else:
            p_value = np.nan
        rows.append(
            {
                "effect_type": effect,
                "beta": estimate,
                "se": err,
                "p": p_value,
                "n": len(data),
                "df": df,
            }
        )
    return pd.DataFrame(rows)


def _normalize_families(
    pathways: Sequence[str],
    pathway_families: Optional[Mapping[str, Sequence[str] | Mapping[str, float]]] = None,
) -> dict[str, dict[str, float]]:
    pathway_set = set(map(str, pathways))
    if pathway_families is None:
        return {str(pathway): {str(pathway): 1.0} for pathway in pathways}

    normalized: dict[str, dict[str, float]] = {}
    for family_id, members in pathway_families.items():
        if isinstance(members, Mapping):
            weights = {
                str(pathway): float(weight)
                for pathway, weight in members.items()
                if str(pathway) in pathway_set and np.isfinite(float(weight)) and float(weight) != 0
            }
        else:
            weights = {str(pathway): 1.0 for pathway in members if str(pathway) in pathway_set}
        total = float(np.sum(np.abs(list(weights.values())))) if weights else 0.0
        if total > 0:
            normalized[str(family_id)] = {pathway: weight / total for pathway, weight in weights.items()}
    if not normalized:
        raise ValueError("pathway_families did not match any score_matrix columns")
    return normalized


def _family_scores(score_matrix: pd.DataFrame, families: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    data = {}
    for family_id, weights in families.items():
        cols = list(weights)
        w = np.asarray([weights[col] for col in cols], dtype=float)
        data[family_id] = score_matrix[cols].to_numpy(dtype=float) @ w
    return pd.DataFrame(data, index=score_matrix.index)


def _parse_contrasts(
    contrasts: Optional[Mapping[str, Sequence[str] | Mapping[str, str]]],
) -> dict[str, tuple[str, str]]:
    parsed: dict[str, tuple[str, str]] = {}
    if not contrasts:
        return parsed
    for name, spec in contrasts.items():
        if isinstance(spec, Mapping):
            case = spec.get("case")
            reference = spec.get("reference") or spec.get("ref")
        else:
            if len(spec) != 2:
                raise ValueError(f"contrast '{name}' must contain case and reference")
            case, reference = spec
        if case is None or reference is None:
            raise ValueError(f"contrast '{name}' must contain case and reference")
        parsed[str(name)] = (str(case), str(reference))
    return parsed


def _condition_series(
    metadata: pd.DataFrame,
    *,
    condition_col: Optional[str],
    factor_columns: tuple[str, str],
    factor_level_map: Optional[Mapping[str, Mapping[Any, int]]] = None,
) -> pd.Series:
    if condition_col is not None:
        if condition_col not in metadata.columns:
            raise KeyError(f"condition_col '{condition_col}' not found in metadata")
        return metadata[condition_col].astype(str)
    factor_a, factor_b = factor_columns
    maps = factor_level_map or {}
    a = _as_binary_series(metadata[factor_a], column=factor_a, level_map=maps.get(factor_a)).astype(int).astype(str)
    b = _as_binary_series(metadata[factor_b], column=factor_b, level_map=maps.get(factor_b)).astype(int).astype(str)
    return factor_a + a + "_" + factor_b + b


def _aggregate_block_deltas(
    family_scores: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    trajectory: str,
    condition: pd.Series,
    contrast_name: str,
    case: str,
    reference: str,
    strata_cols: Sequence[str],
    min_block_n: int,
) -> pd.DataFrame:
    block_cols = [col for col in strata_cols if col in metadata.columns]
    if block_cols:
        block_id = metadata[block_cols].astype(str).agg("|".join, axis=1)
    else:
        block_id = pd.Series("all", index=metadata.index)

    rows = []
    for block, idx in metadata.groupby(block_id, sort=False).groups.items():
        idx = pd.Index(idx)
        labels = condition.loc[idx]
        case_idx = idx[labels == case]
        ref_idx = idx[labels == reference]
        if len(case_idx) < min_block_n or len(ref_idx) < min_block_n:
            continue
        weight = min(len(case_idx), len(ref_idx))
        case_mean = family_scores.loc[case_idx].mean(axis=0)
        ref_mean = family_scores.loc[ref_idx].mean(axis=0)
        for family_id in family_scores.columns:
            rows.append(
                {
                    "trajectory": trajectory,
                    "contrast": contrast_name,
                    "family_id": family_id,
                    "block_id": str(block),
                    "case": case,
                    "reference": reference,
                    "n_case": len(case_idx),
                    "n_reference": len(ref_idx),
                    "block_weight": weight,
                    "block_delta": float(case_mean[family_id] - ref_mean[family_id]),
                }
            )
    return pd.DataFrame(rows)


def _block_permutation_fdr(
    family_scores: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    trajectory: str,
    condition: pd.Series,
    contrasts: Mapping[str, tuple[str, str]],
    strata_cols: Sequence[str],
    min_block_n: int,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_block_rows = []
    result_rows = []
    if not contrasts:
        return pd.DataFrame(), pd.DataFrame()

    block_cols = [col for col in strata_cols if col in metadata.columns]
    if block_cols:
        block_id = metadata[block_cols].astype(str).agg("|".join, axis=1)
    else:
        block_id = pd.Series("all", index=metadata.index)

    for contrast_name, (case, reference) in contrasts.items():
        eligible_idx = metadata.index[condition.isin([case, reference])]
        if len(eligible_idx) == 0:
            continue
        observed_blocks = _aggregate_block_deltas(
            family_scores.loc[eligible_idx],
            metadata.loc[eligible_idx],
            trajectory=trajectory,
            condition=condition.loc[eligible_idx],
            contrast_name=contrast_name,
            case=case,
            reference=reference,
            strata_cols=strata_cols,
            min_block_n=min_block_n,
        )
        if observed_blocks.empty:
            continue
        all_block_rows.append(observed_blocks)
        by_family = observed_blocks.groupby("family_id", sort=False)
        observed = {}
        direction_stability = {}
        n_blocks = {}
        for family_id, group in by_family:
            weights = group["block_weight"].to_numpy(dtype=float)
            deltas = group["block_delta"].to_numpy(dtype=float)
            obs = float(np.average(deltas, weights=weights))
            observed[family_id] = obs
            n_blocks[family_id] = int(len(group))
            if obs == 0:
                direction_stability[family_id] = np.nan
            else:
                direction_stability[family_id] = float(np.mean(np.sign(deltas) == np.sign(obs)))

        perm_stats = {family_id: [] for family_id in observed}
        contrast_idx = metadata.index[condition.isin([case, reference])]
        for _ in range(int(n_perm)):
            perm_condition = condition.loc[contrast_idx].copy()
            for _block, raw_idx in metadata.loc[contrast_idx].groupby(block_id.loc[contrast_idx], sort=False).groups.items():
                idx = pd.Index(raw_idx)
                labels = perm_condition.loc[idx].to_numpy(copy=True)
                if np.isin(labels, [case, reference]).all():
                    perm_condition.loc[idx] = rng.permutation(labels)
            perm_blocks = _aggregate_block_deltas(
                family_scores.loc[contrast_idx],
                metadata.loc[contrast_idx],
                trajectory=trajectory,
                condition=perm_condition,
                contrast_name=contrast_name,
                case=case,
                reference=reference,
                strata_cols=strata_cols,
                min_block_n=min_block_n,
            )
            if perm_blocks.empty:
                for family_id in observed:
                    perm_stats[family_id].append(np.nan)
                continue
            for family_id, group in perm_blocks.groupby("family_id", sort=False):
                weights = group["block_weight"].to_numpy(dtype=float)
                deltas = group["block_delta"].to_numpy(dtype=float)
                perm_stats[family_id].append(float(np.average(deltas, weights=weights)))

        for family_id, obs in observed.items():
            null = np.asarray(perm_stats[family_id], dtype=float)
            null = null[np.isfinite(null)]
            if len(null) == 0:
                p_value = np.nan
            else:
                p_value = float((1.0 + np.sum(np.abs(null) >= abs(obs))) / (1.0 + len(null)))
            result_rows.append(
                {
                    "family_id": family_id,
                    "trajectory": trajectory,
                    "contrast": contrast_name,
                    "case": case,
                    "reference": reference,
                    "observed_family_delta_auc": obs,
                    "n_blocks": n_blocks[family_id],
                    "n_perm": int(n_perm),
                    "block_perm_p": p_value,
                    "direction_stability": direction_stability[family_id],
                }
            )

    block_detail = pd.concat(all_block_rows, ignore_index=True) if all_block_rows else pd.DataFrame()
    block_null = pd.DataFrame(result_rows)
    if not block_null.empty:
        block_null["block_perm_q"] = _bh_fdr(block_null["block_perm_p"])
    return block_null, block_detail


def _fit_factorial_table(
    scores: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    trajectory: str,
    family_or_pathway_col: str,
    factor_columns: tuple[str, str],
    factor_level_map: Optional[Mapping[str, Mapping[Any, int]]],
    strata_cols: Sequence[str],
    include_strata_fixed_effects: bool,
) -> pd.DataFrame:
    design, effect_names = _make_design(
        metadata,
        factor_columns=factor_columns,
        factor_level_map=factor_level_map,
        strata_cols=strata_cols,
        include_strata_fixed_effects=include_strata_fixed_effects,
    )
    rows = []
    for feature in scores.columns:
        fitted = _fit_ols_effects(scores[feature], design, effect_names=effect_names)
        fitted.insert(0, family_or_pathway_col, feature)
        fitted.insert(1, "trajectory", trajectory)
        rows.append(fitted)
    table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not table.empty:
        table["q"] = _bh_fdr(table["p"])
        table["effect_direction"] = np.where(table["beta"] > 0, "gain", np.where(table["beta"] < 0, "loss", "flat"))
    return table


def _pairwise_family_table(
    family_scores: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    trajectory: str,
    condition: pd.Series,
    contrasts: Mapping[str, tuple[str, str]],
) -> pd.DataFrame:
    rows = []
    for contrast_name, (case, reference) in contrasts.items():
        case_idx = metadata.index[condition == case]
        ref_idx = metadata.index[condition == reference]
        if len(case_idx) == 0 or len(ref_idx) == 0:
            continue
        for family_id in family_scores.columns:
            delta = float(family_scores.loc[case_idx, family_id].mean() - family_scores.loc[ref_idx, family_id].mean())
            rows.append(
                {
                    "family_id": family_id,
                    "trajectory": trajectory,
                    "contrast_or_effect": contrast_name,
                    "effect_kind": "pairwise_contrast",
                    "family_delta_auc": delta,
                    "family_direction": "gain" if delta > 0 else "loss" if delta < 0 else "flat",
                    "case": case,
                    "reference": reference,
                    "n_case": len(case_idx),
                    "n_reference": len(ref_idx),
                }
            )
    return pd.DataFrame(rows)


def _specificity_rows(
    family_scores: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    trajectory: str,
    factor_columns: tuple[str, str],
    factor_level_map: Optional[Mapping[str, Mapping[Any, int]]],
    condition: pd.Series,
    contrasts: Mapping[str, tuple[str, str]],
    proliferation_family_id: Optional[str],
    raw_effects: pd.DataFrame,
    specificity_ratio_threshold: float,
    retained_effect_threshold: float,
) -> pd.DataFrame:
    if proliferation_family_id is None or proliferation_family_id not in family_scores.columns:
        return pd.DataFrame()

    rows = []
    eps = 1e-9
    design, effect_names = _make_design(
        metadata,
        factor_columns=factor_columns,
        factor_level_map=factor_level_map,
        include_strata_fixed_effects=False,
    )
    for family_id in family_scores.columns:
        if family_id == proliferation_family_id:
            continue
        for effect in effect_names:
            raw_row = raw_effects[
                (raw_effects["family_id"] == family_id)
                & (raw_effects["trajectory"] == trajectory)
                & (raw_effects["effect_type"] == effect)
            ]
            prolif_row = raw_effects[
                (raw_effects["family_id"] == proliferation_family_id)
                & (raw_effects["trajectory"] == trajectory)
                & (raw_effects["effect_type"] == effect)
            ]
            if raw_row.empty or prolif_row.empty:
                continue
            raw = float(raw_row.iloc[0]["beta"])
            prolif = float(prolif_row.iloc[0]["beta"])
            adjusted_design = design.copy()
            adjusted_design["proliferation_family_score"] = family_scores[proliferation_family_id]
            fitted = _fit_ols_effects(family_scores[family_id], adjusted_design, effect_names=[effect])
            adjusted = float(fitted.iloc[0]["beta"]) if not fitted.empty else np.nan
            rows.append(
                _specificity_record(
                    family_id=family_id,
                    trajectory=trajectory,
                    contrast_or_effect=effect,
                    effect_kind="factorial_effect",
                    raw_delta=raw,
                    proliferation_delta=prolif,
                    adjusted_delta=adjusted,
                    specificity_ratio_threshold=specificity_ratio_threshold,
                    retained_effect_threshold=retained_effect_threshold,
                    eps=eps,
                )
            )

        for contrast_name, (case, reference) in contrasts.items():
            idx = metadata.index[condition.isin([case, reference])]
            if len(idx) == 0:
                continue
            labels = (condition.loc[idx] == case).astype(float)
            base_design = pd.DataFrame(
                {
                    "intercept": 1.0,
                    "pairwise_case": labels,
                    "proliferation_family_score": family_scores.loc[idx, proliferation_family_id],
                },
                index=idx,
            )
            raw = float(
                family_scores.loc[idx[labels == 1], family_id].mean()
                - family_scores.loc[idx[labels == 0], family_id].mean()
            )
            prolif = float(
                family_scores.loc[idx[labels == 1], proliferation_family_id].mean()
                - family_scores.loc[idx[labels == 0], proliferation_family_id].mean()
            )
            fitted = _fit_ols_effects(family_scores.loc[idx, family_id], base_design, effect_names=["pairwise_case"])
            adjusted = float(fitted.iloc[0]["beta"]) if not fitted.empty else np.nan
            rows.append(
                _specificity_record(
                    family_id=family_id,
                    trajectory=trajectory,
                    contrast_or_effect=contrast_name,
                    effect_kind="pairwise_contrast",
                    raw_delta=raw,
                    proliferation_delta=prolif,
                    adjusted_delta=adjusted,
                    specificity_ratio_threshold=specificity_ratio_threshold,
                    retained_effect_threshold=retained_effect_threshold,
                    eps=eps,
                )
            )
    return pd.DataFrame(rows)


def _specificity_record(
    *,
    family_id: str,
    trajectory: str,
    contrast_or_effect: str,
    effect_kind: str,
    raw_delta: float,
    proliferation_delta: float,
    adjusted_delta: float,
    specificity_ratio_threshold: float,
    retained_effect_threshold: float,
    eps: float,
) -> dict[str, Any]:
    ratio = float(abs(raw_delta) / (abs(proliferation_delta) + eps))
    retained = float(abs(adjusted_delta) / (abs(raw_delta) + eps)) if np.isfinite(adjusted_delta) else np.nan
    direction_preserved = bool(np.sign(adjusted_delta) == np.sign(raw_delta)) if np.isfinite(adjusted_delta) else False
    ratio_pass = ratio >= specificity_ratio_threshold
    retained_pass = retained >= retained_effect_threshold and direction_preserved
    specificity_pass = bool(ratio_pass or retained_pass)
    if ratio_pass:
        classification = "erythroid_specific"
    elif retained_pass:
        classification = "partly_proliferation_associated"
    else:
        classification = "proliferation_dominated"
    return {
        "family_id": family_id,
        "trajectory": trajectory,
        "contrast_or_effect": contrast_or_effect,
        "effect_kind": effect_kind,
        "family_delta_auc": raw_delta,
        "proliferation_family_delta_auc": proliferation_delta,
        "specificity_ratio": ratio,
        "proliferation_adjusted_delta_auc": adjusted_delta,
        "adjusted_effect_retained_fraction": retained,
        "adjusted_direction_preserved": direction_preserved,
        "specificity_classification": classification,
        "specificity_pass": specificity_pass,
    }


def _prepare_support_table(table: Optional[pd.DataFrame], columns: Sequence[str]) -> pd.DataFrame:
    if table is None:
        return pd.DataFrame(columns=list(columns))
    prepared = table.copy()
    for col in columns:
        if col not in prepared.columns:
            prepared[col] = np.nan
    return prepared


def _support_pass(table: pd.DataFrame, family_id: str) -> bool:
    if table.empty or "family_id" not in table.columns:
        return False
    subset = table[table["family_id"].astype(str) == str(family_id)]
    if subset.empty:
        return False
    boolean_cols = [
        col
        for col in (
            "validation_pass",
            "direction_consistent",
            "driver_dependency_pass",
            "mechanism_pass",
            "enhancer_specificity_support",
            "motif_support",
            "peak_gene_link_support",
            "wet_lab_rescue_pass",
        )
        if col in subset.columns
    ]
    for col in boolean_cols:
        values = subset[col]
        if values.astype(str).str.lower().isin({"true", "1", "yes", "pass", "supported"}).any():
            return True
        if values.dtype == bool and bool(values.any()):
            return True
    status_cols = [col for col in ("validation_status", "support_level", "claim_level") if col in subset.columns]
    for col in status_cols:
        values = subset[col].dropna().astype(str).str.lower()
        if values.str.contains("support|validated|multiome|external|orthogonal|mechanism", regex=True).any():
            return True
    return False


def _build_claim_ceiling(
    family_table: pd.DataFrame,
    *,
    driver_table: pd.DataFrame,
    external_support_table: pd.DataFrame,
    block_q_threshold: float,
) -> pd.DataFrame:
    if family_table.empty:
        return pd.DataFrame()
    rows = []
    for _, row in family_table.iterrows():
        family_id = str(row["family_id"])
        family_q = pd.to_numeric(pd.Series([row.get("family_q", np.nan)]), errors="coerce").iloc[0]
        block_q = pd.to_numeric(pd.Series([row.get("family_block_q", np.nan)]), errors="coerce").iloc[0]
        specificity_pass = bool(row.get("specificity_pass", False))
        driver_pass = _support_pass(driver_table, family_id)
        external_pass = _support_pass(external_support_table, family_id)
        claim_levels = []
        if np.isfinite(family_q) and family_q <= 0.05:
            claim_levels.append("family_level_candidate")
        if np.isfinite(block_q) and block_q <= block_q_threshold and specificity_pass:
            claim_levels.append("internal_block_robust")
        if external_pass:
            claim_levels.append("external_supported")
        if driver_pass:
            claim_levels.append("mechanism_candidate")
        if (
            "internal_block_robust" in claim_levels
            and external_pass
            and driver_pass
            and specificity_pass
        ):
            claim_levels.append("causal_ready_for_experiment")

        if row.get("specificity_classification") == "proliferation_dominated":
            ceiling = "candidate_only_generic_perturbation"
        elif "internal_block_robust" in claim_levels:
            ceiling = "block_robust_family_discovery"
        elif "family_level_candidate" in claim_levels:
            ceiling = "family_level_candidate"
        else:
            ceiling = "screening_only"
        rows.append(
            {
                "family_id": family_id,
                "trajectory": row.get("trajectory", "all"),
                "contrast_or_effect": row.get("contrast_or_effect", row.get("effect_type", "")),
                "effect_kind": row.get("effect_kind", ""),
                "family_delta_auc": row.get("family_delta_auc", row.get("beta", np.nan)),
                "family_q": family_q,
                "family_block_q": block_q,
                "specificity_classification": row.get("specificity_classification", ""),
                "specificity_pass": specificity_pass,
                "driver_dependency_pass": driver_pass,
                "external_support_pass": external_pass,
                "claim_level": ";".join(claim_levels) if claim_levels else "screening_only",
                "claim_ceiling": ceiling,
            }
        )
    return pd.DataFrame(rows)


@dataclass
class PerturbationEventResult:
    event_table: pd.DataFrame = field(default_factory=_empty_df)
    family_table: pd.DataFrame = field(default_factory=_empty_df)
    factorial_effect_table: pd.DataFrame = field(default_factory=_empty_df)
    block_null_table: pd.DataFrame = field(default_factory=_empty_df)
    driver_table: pd.DataFrame = field(default_factory=_empty_df)
    external_support_table: pd.DataFrame = field(default_factory=_empty_df)
    claim_ceiling_table: pd.DataFrame = field(default_factory=_empty_df)
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence_layers: dict[str, str] = field(default_factory=dict)

    def to_tables(self) -> dict[str, pd.DataFrame]:
        return {
            "event_table": self.event_table,
            "family_table": self.family_table,
            "factorial_effect_table": self.factorial_effect_table,
            "block_null_table": self.block_null_table,
            "driver_table": self.driver_table,
            "external_support_table": self.external_support_table,
            "claim_ceiling_table": self.claim_ceiling_table,
        }

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"table": name, "rows": int(len(table)), "columns": int(len(table.columns))}
                for name, table in self.to_tables().items()
            ]
        )

    def write(self, output_dir: str | Path, *, prefix: str = "ted_perturbation") -> dict[str, Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for name, table in self.to_tables().items():
            path = output / f"{prefix}_{name}.tsv"
            table.to_csv(path, sep="\t", index=False)
            written[name] = path
        metadata_path = output / f"{prefix}_metadata.tsv"
        pd.DataFrame(
            [{"key": key, "value": value} for key, value in sorted(self.metadata.items())]
        ).to_csv(metadata_path, sep="\t", index=False)
        written["metadata"] = metadata_path
        return written


def run_ted_perturbation(
    score_matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    trajectory_col: Optional[str] = None,
    condition_col: Optional[str] = None,
    factor_columns: tuple[str, str] = ("T21", "GATA1s"),
    factor_level_map: Optional[Mapping[str, Mapping[Any, int]]] = None,
    strata_cols: Sequence[str] = (),
    pathway_families: Optional[Mapping[str, Sequence[str] | Mapping[str, float]]] = None,
    contrasts: Optional[Mapping[str, Sequence[str] | Mapping[str, str]]] = None,
    driver_table: Optional[pd.DataFrame] = None,
    external_support_table: Optional[pd.DataFrame] = None,
    proliferation_family_id: Optional[str] = None,
    n_perm: int = 999,
    random_state: int = 0,
    min_block_n: int = 2,
    include_strata_fixed_effects: bool = False,
    specificity_ratio_threshold: float = 1.2,
    retained_effect_threshold: float = 0.5,
    block_q_threshold: float = 0.05,
) -> PerturbationEventResult:
    """Run family-first TED-Perturbation inference.

    The main statistical unit is an event family. Single pathways are retained
    in ``event_table`` as screening evidence, while ``family_table`` and
    ``block_null_table`` carry the discovery-grade perturbation claims.
    """

    if not isinstance(score_matrix, pd.DataFrame):
        score_matrix = pd.DataFrame(score_matrix)
    score_matrix = score_matrix.copy()
    score_matrix.columns = score_matrix.columns.astype(str)
    metadata = _align_metadata(score_matrix, metadata)
    parsed_contrasts = _parse_contrasts(contrasts)
    families = _normalize_families(score_matrix.columns, pathway_families)
    family_scores_all = _family_scores(score_matrix, families)
    rng = np.random.default_rng(random_state)

    event_tables = []
    factorial_tables = []
    pairwise_tables = []
    block_tables = []
    specificity_tables = []

    for trajectory, idx in _trajectory_groups(metadata, trajectory_col):
        idx = pd.Index(idx)
        meta_t = metadata.loc[idx].copy()
        scores_t = score_matrix.loc[idx]
        family_t = family_scores_all.loc[idx]

        event = _fit_factorial_table(
            scores_t,
            meta_t,
            trajectory=trajectory,
            family_or_pathway_col="pathway",
            factor_columns=factor_columns,
            factor_level_map=factor_level_map,
            strata_cols=strata_cols,
            include_strata_fixed_effects=include_strata_fixed_effects,
        )
        event_tables.append(event)

        factorial = _fit_factorial_table(
            family_t,
            meta_t,
            trajectory=trajectory,
            family_or_pathway_col="family_id",
            factor_columns=factor_columns,
            factor_level_map=factor_level_map,
            strata_cols=strata_cols,
            include_strata_fixed_effects=include_strata_fixed_effects,
        )
        factorial_tables.append(factorial)

        condition = _condition_series(
            meta_t,
            condition_col=condition_col,
            factor_columns=factor_columns,
            factor_level_map=factor_level_map,
        )
        pairwise = _pairwise_family_table(
            family_t,
            meta_t,
            trajectory=trajectory,
            condition=condition,
            contrasts=parsed_contrasts,
        )
        if not pairwise.empty:
            pairwise_tables.append(pairwise)
        block_null, _block_detail = _block_permutation_fdr(
            family_t,
            meta_t,
            trajectory=trajectory,
            condition=condition,
            contrasts=parsed_contrasts,
            strata_cols=strata_cols,
            min_block_n=min_block_n,
            n_perm=n_perm,
            rng=rng,
        )
        if not block_null.empty:
            block_tables.append(block_null)

        spec = _specificity_rows(
            family_t,
            meta_t,
            trajectory=trajectory,
            factor_columns=factor_columns,
            factor_level_map=factor_level_map,
            condition=condition,
            contrasts=parsed_contrasts,
            proliferation_family_id=proliferation_family_id,
            raw_effects=factorial,
            specificity_ratio_threshold=specificity_ratio_threshold,
            retained_effect_threshold=retained_effect_threshold,
        )
        if not spec.empty:
            specificity_tables.append(spec)

    event_table = pd.concat(event_tables, ignore_index=True) if event_tables else pd.DataFrame()
    factorial_effect_table = (
        pd.concat(factorial_tables, ignore_index=True) if factorial_tables else pd.DataFrame()
    )
    block_null_table = pd.concat(block_tables, ignore_index=True) if block_tables else pd.DataFrame()
    specificity_table = (
        pd.concat(specificity_tables, ignore_index=True) if specificity_tables else pd.DataFrame()
    )

    family_factorial = factorial_effect_table.rename(
        columns={"beta": "family_delta_auc", "q": "family_q"}
    ).copy()
    if not family_factorial.empty:
        family_factorial["contrast_or_effect"] = family_factorial["effect_type"]
        family_factorial["effect_kind"] = "factorial_effect"
        family_factorial["family_direction"] = family_factorial["effect_direction"]

    family_pairwise = pd.concat(pairwise_tables, ignore_index=True) if pairwise_tables else pd.DataFrame()
    if not family_pairwise.empty and not block_null_table.empty:
        family_pairwise = family_pairwise.merge(
            block_null_table[
                [
                    "family_id",
                    "trajectory",
                    "contrast",
                    "observed_family_delta_auc",
                    "block_perm_p",
                    "block_perm_q",
                    "direction_stability",
                    "n_blocks",
                ]
            ],
            left_on=["family_id", "trajectory", "contrast_or_effect"],
            right_on=["family_id", "trajectory", "contrast"],
            how="left",
        )
        family_pairwise["family_delta_auc"] = family_pairwise["observed_family_delta_auc"].fillna(
            family_pairwise["family_delta_auc"]
        )
        family_pairwise["family_q"] = family_pairwise["block_perm_q"]
        family_pairwise["family_block_q"] = family_pairwise["block_perm_q"]
        family_pairwise = family_pairwise.drop(columns=["contrast"], errors="ignore")

    family_table = pd.concat(
        [
            family_factorial[
                [
                    col
                    for col in (
                        "family_id",
                        "trajectory",
                        "contrast_or_effect",
                        "effect_kind",
                        "family_delta_auc",
                        "family_q",
                        "family_direction",
                        "p",
                        "se",
                        "n",
                        "df",
                    )
                    if col in family_factorial.columns
                ]
            ],
            family_pairwise,
        ],
        ignore_index=True,
    )
    if not specificity_table.empty and not family_table.empty:
        family_table = family_table.merge(
            specificity_table,
            on=["family_id", "trajectory", "contrast_or_effect", "effect_kind"],
            how="left",
            suffixes=("", "_specificity"),
        )
        if "family_delta_auc_specificity" in family_table.columns:
            family_table = family_table.drop(columns=["family_delta_auc_specificity"])
    else:
        family_table["specificity_pass"] = False
        family_table["specificity_classification"] = ""

    driver = _prepare_support_table(
        driver_table,
        [
            "family_id",
            "driver_module",
            "driver_genes",
            "signed_driver_support",
            "driver_dependency_score",
            "driver_dependency_pass",
            "mechanism_layer",
        ],
    )
    external = _prepare_support_table(
        external_support_table,
        [
            "family_id",
            "external_dataset",
            "validation_status",
            "validation_pass",
            "enhancer_specificity_support",
            "motif_support",
            "peak_gene_link_support",
            "support_level",
        ],
    )
    claim_ceiling_table = _build_claim_ceiling(
        family_table,
        driver_table=driver,
        external_support_table=external,
        block_q_threshold=block_q_threshold,
    )

    metadata_out = {
        "algorithm": "TED-Perturbation",
        "family_level_primary": True,
        "block_level_primary": True,
        "specificity_gate": True,
        "factor_columns": ",".join(factor_columns),
        "trajectory_col": trajectory_col,
        "condition_col": condition_col,
        "strata_cols": ",".join(strata_cols),
        "n_perm": int(n_perm),
        "random_state": int(random_state),
        "n_pathways": int(score_matrix.shape[1]),
        "n_event_families": int(len(families)),
    }
    evidence_layers = {
        "cell_or_window_q": "screening only",
        "pathway_event_q": "candidate pathway event",
        "family_q": "family-level candidate",
        "block_permutation_q": "robust perturbation discovery",
        "specificity_gate": "guards against proliferation or generic-state dominated claims",
        "external_validation": "orthogonal support, not a substitute for block-level statistics",
        "wet_lab_rescue": "causal validation layer",
    }
    return PerturbationEventResult(
        event_table=event_table,
        family_table=family_table,
        factorial_effect_table=factorial_effect_table,
        block_null_table=block_null_table,
        driver_table=driver,
        external_support_table=external,
        claim_ceiling_table=claim_ceiling_table,
        metadata=metadata_out,
        evidence_layers=evidence_layers,
    )
