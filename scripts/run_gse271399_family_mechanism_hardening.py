"""Family-level attribution, block robustness, specificity, and driver dependency.

This is a post-processing hardening pass for GSE271399. It turns the v2
family-level TED-Perturbation output into mechanistic tables:

- perturbation attribution index (T21 vs GATA1s vs interaction)
- block bootstrap / jackknife family effects
- family specificity score
- erythroid driver-module ablation
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from run_gse271399_first_round import module_score_from_z
from run_gse271399_hard_validation_addons import (
    CONTRAST_TO_CASE_REF,
    TRAJECTORY_TARGET,
    add_scores_and_states,
    load_cached_adata,
    read_gmt,
    sign_int,
    sign_label,
)


EFFECT_TYPES = ["T21", "GATA1s", "interaction"]
CONDITIONS = ["Euploid_wtGATA1", "Euploid_GATA1s", "T21_wtGATA1", "T21_GATA1s"]
PAIRWISE_CONTRASTS = list(CONTRAST_TO_CASE_REF)
RNG_SEED = 271399

DRIVER_MODULES = {
    "regulatory_axis": ["KLF1", "TAL1", "ZFPM1", "NFE2", "EPOR", "LMO2", "GFI1B"],
    "maturation_membrane_axis": ["ANK1", "GYPC", "GYPB", "GYPA", "RHAG", "RHD", "ERMAP"],
    "heme_iron_axis": [
        "ALAS2",
        "TFRC",
        "SLC25A37",
        "FECH",
        "ABCB10",
        "STEAP3",
        "SLC25A38",
        "SLC11A2",
        "FTH1",
        "FTL",
    ],
}
DRIVER_MODULES["all_erythroid_drivers"] = sorted({g for genes in DRIVER_MODULES.values() for g in genes})


def parse_profile(text: object) -> np.ndarray:
    vals = []
    for part in str(text).split(","):
        part = part.strip()
        vals.append(np.nan if part in {"", "NA", "nan"} else float(part))
    return np.asarray(vals, dtype=float)


def profile_text(values: np.ndarray) -> str:
    return ",".join("NA" if not np.isfinite(v) else f"{float(v):.6g}" for v in values)


def abs_auc_from_profile(text: object) -> float:
    values = parse_profile(text)
    return float(np.nanmean(np.abs(values))) if np.isfinite(values).any() else np.nan


def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not ok.any():
        return np.nan
    return float(np.average(values[ok], weights=weights[ok]))


def bootstrap_weighted_mean(
    values: np.ndarray,
    weights: np.ndarray,
    n_boot: int = 1000,
    seed: int = RNG_SEED,
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[ok]
    weights = weights[ok]
    if len(values) == 0:
        return {
            "mean": np.nan,
            "sd": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "direction_stability": np.nan,
            "bootstrap_p": np.nan,
        }
    observed = weighted_average(values, weights)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    n = len(values)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[i] = weighted_average(values[idx], weights[idx])
    obs_sign = sign_int(observed)
    direction_stability = float(np.mean([sign_int(v) == obs_sign for v in draws])) if obs_sign else 0.0
    if obs_sign > 0:
        bootstrap_p = float((np.sum(draws <= 0) + 1) / (len(draws) + 1))
    elif obs_sign < 0:
        bootstrap_p = float((np.sum(draws >= 0) + 1) / (len(draws) + 1))
    else:
        bootstrap_p = 1.0
    return {
        "mean": float(np.nanmean(draws)),
        "sd": float(np.nanstd(draws, ddof=1)) if len(draws) > 1 else 0.0,
        "ci_lower": float(np.nanquantile(draws, 0.025)),
        "ci_upper": float(np.nanquantile(draws, 0.975)),
        "direction_stability": direction_stability,
        "bootstrap_p": bootstrap_p,
    }


def weighted_lstsq_beta(y: np.ndarray, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(y) & np.isfinite(x).all(axis=1) & np.isfinite(weights) & (weights > 0)
    if ok.sum() < x.shape[1]:
        return np.full(x.shape[1], np.nan)
    x = x[ok]
    y = y[ok]
    weights = weights[ok]
    if np.linalg.matrix_rank(x) < x.shape[1]:
        return np.full(x.shape[1], np.nan)
    root_w = np.sqrt(weights)
    beta, *_ = np.linalg.lstsq(x * root_w[:, None], y * root_w, rcond=None)
    return beta


def bh_fdr(p_values: list[float] | pd.Series | np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    idx = np.where(finite)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    m = len(ranked)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out[order] = np.minimum(q, 1.0)
    return out


def load_inputs(root: Path) -> dict[str, pd.DataFrame | Path]:
    dataset_dir = root / "GSE271399_T21_GATA1s"
    family_dir = dataset_dir / "deliverables_family_level_inference_v2"
    ted_dir = dataset_dir / "ted"
    external_dir = root / "external_validation" / "deliverables_external_validation"
    return {
        "dataset_dir": dataset_dir,
        "ted_dir": ted_dir,
        "family_dir": family_dir,
        "family_def": pd.read_csv(family_dir / "gse271399_event_family_definition_v2.tsv", sep="\t"),
        "family_effect": pd.read_csv(family_dir / "gse271399_family_level_effect_decomposition.tsv", sep="\t"),
        "family_fdr": pd.read_csv(family_dir / "gse271399_family_level_perturbation_fdr.tsv", sep="\t"),
        "driver_summary": pd.read_csv(family_dir / "gse271399_family_level_driver_summary.tsv", sep="\t"),
        "external_summary": pd.read_csv(external_dir / "external_validation_summary.tsv", sep="\t")
        if (external_dir / "external_validation_summary.tsv").exists()
        else pd.DataFrame(),
    }


def family_members_and_weights(family_def: pd.DataFrame) -> dict[str, tuple[list[str], dict[str, float]]]:
    out: dict[str, tuple[list[str], dict[str, float]]] = {}
    for family_id, sub in family_def.groupby("family_id", observed=True):
        sub = sub.copy()
        sub["redundancy_aware_weight"] = pd.to_numeric(sub["redundancy_aware_weight"], errors="coerce")
        members = sub["pathway"].astype(str).tolist()
        weights = dict(zip(sub["pathway"].astype(str), sub["redundancy_aware_weight"].astype(float)))
        if not np.isfinite(list(weights.values())).all() or sum(weights.values()) <= 0:
            weights = {p: 1.0 / len(members) for p in members}
        else:
            total = float(sum(weights.values()))
            weights = {p: float(w / total) for p, w in weights.items()}
        out[str(family_id)] = (members, weights)
    return out


def perturbation_attribution_index(inputs: dict[str, pd.DataFrame | Path], out_dir: Path) -> pd.DataFrame:
    effect = inputs["family_effect"].copy()  # type: ignore[index]
    effect = effect[effect["effect_type"].isin(EFFECT_TYPES)].copy()
    rows = []
    for (family_id, trajectory), sub in effect.groupby(["family_id", "trajectory"], observed=True):
        aucs = {}
        profiles = {}
        for effect_type in EFFECT_TYPES:
            hit = sub[sub["effect_type"] == effect_type]
            if hit.empty:
                aucs[effect_type] = np.nan
                profiles[effect_type] = ""
            else:
                row = hit.iloc[0]
                aucs[effect_type] = abs_auc_from_profile(row["family_profile"])
                profiles[effect_type] = row["family_profile"]
        total = sum(v for v in aucs.values() if np.isfinite(v))
        eps = 1e-12
        pai_t21 = aucs["T21"] / (total + eps) if np.isfinite(aucs["T21"]) else np.nan
        pai_gata = aucs["GATA1s"] / (total + eps) if np.isfinite(aucs["GATA1s"]) else np.nan
        pai_int = aucs["interaction"] / (total + eps) if np.isfinite(aucs["interaction"]) else np.nan
        pai_values = {"T21": pai_t21, "GATA1s": pai_gata, "interaction": pai_int}
        dominant = max(pai_values, key=lambda k: -np.inf if not np.isfinite(pai_values[k]) else pai_values[k])
        sorted_vals = sorted([v for v in pai_values.values() if np.isfinite(v)], reverse=True)
        if len(sorted_vals) >= 2 and sorted_vals[0] - sorted_vals[1] < 0.15:
            dominant_effect = "mixed"
            interpretation = "combined perturbation architecture"
        elif dominant == "GATA1s":
            dominant_effect = "GATA1s"
            interpretation = "GATA1s main-effect disruption"
        elif dominant == "interaction":
            dominant_effect = "interaction"
            interpretation = "T21-context-specific GATA1s disruption"
        else:
            dominant_effect = "T21"
            interpretation = "T21 background effect"
        rows.append(
            {
                "family_id": family_id,
                "trajectory": trajectory,
                "A_T21": aucs["T21"],
                "A_GATA1s": aucs["GATA1s"],
                "A_interaction": aucs["interaction"],
                "PAI_T21": pai_t21,
                "PAI_GATA1s": pai_gata,
                "PAI_interaction": pai_int,
                "dominant_effect": dominant_effect,
                "interpretation": interpretation,
                "T21_profile": profiles["T21"],
                "GATA1s_profile": profiles["GATA1s"],
                "interaction_profile": profiles["interaction"],
            }
        )
    df = pd.DataFrame(rows).sort_values(["family_id", "trajectory"])
    df.to_csv(out_dir / "gse271399_perturbation_attribution_index.tsv", sep="\t", index=False)
    return df


def bin_labels(values: np.ndarray, n_bins: int = 8) -> np.ndarray:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    labels = np.digitize(values, bins[1:-1], right=False)
    labels[~np.isfinite(values)] = -1
    return labels.astype(int)


def build_family_scores(scores: pd.DataFrame, families: dict[str, tuple[list[str], dict[str, float]]]) -> pd.DataFrame:
    out = {}
    for family_id, (members, weights) in families.items():
        cols = [p for p in members if p in scores.columns]
        if not cols:
            continue
        matrix = scores[cols].to_numpy(dtype=float)
        w = np.asarray([weights[p] for p in cols], dtype=float)
        w = w / max(w.sum(), 1e-12)
        out[family_id] = np.average(matrix, axis=1, weights=w)
    return pd.DataFrame(out, index=scores.index)


def block_table_for_family(adata, family_score: np.ndarray, trajectory: str) -> pd.DataFrame:
    obs = adata.obs
    pt_bin = bin_labels(obs[f"pseudotime_{trajectory}"].to_numpy(dtype=float), 8)
    traj = obs[f"in_trajectory_{trajectory}"].to_numpy(dtype=bool)
    work = pd.DataFrame(
        {
            "condition": obs["condition"].astype(str).to_numpy(),
            "day": obs["day"].astype(str).to_numpy(),
            "pt_bin": pt_bin,
            "coarse_state": obs["coarse_cell_type"].astype(str).to_numpy(),
            "score": family_score,
            "traj": traj,
        }
    )
    work = work[work["traj"] & (work["pt_bin"] >= 0) & np.isfinite(work["score"])]
    return (
        work.groupby(["condition", "day", "pt_bin", "coarse_state"], observed=True)
        .agg(block_score=("score", "mean"), n_cells=("score", "size"))
        .reset_index()
    )


def pairwise_blocks(blocks: pd.DataFrame, contrast: str, min_cells: int = 5) -> pd.DataFrame:
    case, ref = CONTRAST_TO_CASE_REF[contrast]
    piv = blocks.pivot_table(
        index=["day", "pt_bin", "coarse_state"],
        columns="condition",
        values=["block_score", "n_cells"],
        aggfunc="first",
    )
    rows = []
    for idx, row in piv.iterrows():
        if ("block_score", case) not in row.index or ("block_score", ref) not in row.index:
            continue
        case_score = row.get(("block_score", case), np.nan)
        ref_score = row.get(("block_score", ref), np.nan)
        case_n = row.get(("n_cells", case), np.nan)
        ref_n = row.get(("n_cells", ref), np.nan)
        if not all(np.isfinite(v) for v in [case_score, ref_score, case_n, ref_n]):
            continue
        if case_n < min_cells or ref_n < min_cells:
            continue
        day, pt_bin, coarse_state = idx
        rows.append(
            {
                "day": day,
                "pt_bin": int(pt_bin),
                "coarse_state": coarse_state,
                "block_delta": float(case_score - ref_score),
                "weight": float(min(case_n, ref_n)),
                "case_n": int(case_n),
                "ref_n": int(ref_n),
            }
        )
    return pd.DataFrame(rows)


def factorial_blocks(blocks: pd.DataFrame, min_cells: int = 5) -> pd.DataFrame:
    rows = []
    design = {
        "Euploid_wtGATA1": (0.0, 0.0, 0.0),
        "Euploid_GATA1s": (0.0, 1.0, 0.0),
        "T21_wtGATA1": (1.0, 0.0, 0.0),
        "T21_GATA1s": (1.0, 1.0, 1.0),
    }
    for key, sub in blocks.groupby(["day", "pt_bin", "coarse_state"], observed=True):
        sub = sub[sub["condition"].isin(CONDITIONS) & (sub["n_cells"] >= min_cells)].copy()
        if sub["condition"].nunique() < 4:
            continue
        x_rows = []
        y = []
        w = []
        for _, r in sub.iterrows():
            t21, gata, inter = design[str(r["condition"])]
            x_rows.append([1.0, t21, gata, inter])
            y.append(float(r["block_score"]))
            w.append(float(r["n_cells"]))
        beta = weighted_lstsq_beta(np.asarray(y), np.asarray(x_rows), np.asarray(w))
        if not np.isfinite(beta).all():
            continue
        day, pt_bin, coarse_state = key
        min_n = int(sub["n_cells"].min())
        for idx, effect_type in enumerate(EFFECT_TYPES, start=1):
            rows.append(
                {
                    "day": day,
                    "pt_bin": int(pt_bin),
                    "coarse_state": coarse_state,
                    "effect_type": effect_type,
                    "block_delta": float(beta[idx]),
                    "weight": float(min_n),
                }
            )
    return pd.DataFrame(rows)


def jackknife_from_blocks(blocks: pd.DataFrame, observed: float) -> dict[str, object]:
    obs_sign = sign_int(observed)
    rows = []
    for axis in ["day", "pt_bin", "coarse_state"]:
        values = sorted(blocks[axis].dropna().unique().tolist())
        matches = []
        for value in values:
            sub = blocks[blocks[axis] != value]
            estimate = weighted_average(sub["block_delta"].to_numpy(float), sub["weight"].to_numpy(float))
            match = bool(sign_int(estimate) == obs_sign and obs_sign != 0)
            matches.append(match)
            rows.append((axis, value, estimate, match, len(sub)))
        pass_rate = float(np.mean(matches)) if matches else np.nan
        if axis == "day":
            day_pass = bool(pass_rate == 1.0) if np.isfinite(pass_rate) else False
        elif axis == "pt_bin":
            pt_pass = bool(pass_rate >= 0.90) if np.isfinite(pass_rate) else False
        else:
            state_pass = bool(pass_rate >= 0.90) if np.isfinite(pass_rate) else False
    return {
        "leave_one_day_out_pass": day_pass if "day_pass" in locals() else False,
        "leave_one_state_bin_out_pass": pt_pass if "pt_pass" in locals() else False,
        "leave_one_coarse_state_out_pass": state_pass if "state_pass" in locals() else False,
        "jackknife_rows": rows,
    }


def block_uncertainty_outputs(
    adata,
    family_scores: pd.DataFrame,
    out_dir: Path,
    n_boot: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bootstrap_rows = []
    jackknife_rows = []
    for family_id in family_scores.columns:
        score = family_scores[family_id].to_numpy(dtype=float)
        for trajectory in TRAJECTORY_TARGET:
            blocks = block_table_for_family(adata, score, trajectory)
            for contrast in PAIRWISE_CONTRASTS:
                pb = pairwise_blocks(blocks, contrast)
                observed = weighted_average(pb["block_delta"].to_numpy(float), pb["weight"].to_numpy(float)) if not pb.empty else np.nan
                boot = bootstrap_weighted_mean(
                    pb["block_delta"].to_numpy(float) if not pb.empty else np.asarray([]),
                    pb["weight"].to_numpy(float) if not pb.empty else np.asarray([]),
                    n_boot=n_boot,
                    seed=RNG_SEED + len(bootstrap_rows),
                )
                jack = jackknife_from_blocks(pb, observed) if not pb.empty and np.isfinite(observed) else {}
                claim = bool(
                    np.isfinite(boot["ci_lower"])
                    and boot["direction_stability"] >= 0.9
                    and not (boot["ci_lower"] <= 0 <= boot["ci_upper"])
                    and jack.get("leave_one_day_out_pass", False)
                    and jack.get("leave_one_state_bin_out_pass", False)
                )
                bootstrap_rows.append(
                    {
                        "family_id": family_id,
                        "trajectory": trajectory,
                        "contrast": contrast,
                        "effect_type": "pairwise_contrast",
                        "delta_auc": observed,
                        "bootstrap_mean": boot["mean"],
                        "bootstrap_sd": boot["sd"],
                        "ci_lower": boot["ci_lower"],
                        "ci_upper": boot["ci_upper"],
                        "direction_stability": boot["direction_stability"],
                        "bootstrap_p": boot["bootstrap_p"],
                        "n_blocks": int(len(pb)),
                        "claim_stability": claim,
                        "block_source": "condition-matched day x pseudotime_bin x coarse_state block means",
                    }
                )
                for axis, value, estimate, match, n_remaining in jack.get("jackknife_rows", []):
                    jackknife_rows.append(
                        {
                            "family_id": family_id,
                            "trajectory": trajectory,
                            "contrast": contrast,
                            "effect_type": "pairwise_contrast",
                            "jackknife_axis": axis,
                            "left_out_block": value,
                            "delta_auc_leave_one_out": estimate,
                            "direction_preserved": match,
                            "n_blocks_remaining": n_remaining,
                        }
                    )
            fb = factorial_blocks(blocks)
            for effect_type in EFFECT_TYPES:
                sub = fb[fb["effect_type"] == effect_type] if not fb.empty else pd.DataFrame()
                observed = weighted_average(sub["block_delta"].to_numpy(float), sub["weight"].to_numpy(float)) if not sub.empty else np.nan
                boot = bootstrap_weighted_mean(
                    sub["block_delta"].to_numpy(float) if not sub.empty else np.asarray([]),
                    sub["weight"].to_numpy(float) if not sub.empty else np.asarray([]),
                    n_boot=n_boot,
                    seed=RNG_SEED + len(bootstrap_rows),
                )
                jack = jackknife_from_blocks(sub, observed) if not sub.empty and np.isfinite(observed) else {}
                claim = bool(
                    np.isfinite(boot["ci_lower"])
                    and boot["direction_stability"] >= 0.9
                    and not (boot["ci_lower"] <= 0 <= boot["ci_upper"])
                    and jack.get("leave_one_day_out_pass", False)
                    and jack.get("leave_one_state_bin_out_pass", False)
                )
                bootstrap_rows.append(
                    {
                        "family_id": family_id,
                        "trajectory": trajectory,
                        "contrast": "NA",
                        "effect_type": effect_type,
                        "delta_auc": observed,
                        "bootstrap_mean": boot["mean"],
                        "bootstrap_sd": boot["sd"],
                        "ci_lower": boot["ci_lower"],
                        "ci_upper": boot["ci_upper"],
                        "direction_stability": boot["direction_stability"],
                        "bootstrap_p": boot["bootstrap_p"],
                        "n_blocks": int(len(sub)),
                        "claim_stability": claim,
                        "block_source": "factorial WLS beta from day x pseudotime_bin x coarse_state condition block means",
                    }
                )
                for axis, value, estimate, match, n_remaining in jack.get("jackknife_rows", []):
                    jackknife_rows.append(
                        {
                            "family_id": family_id,
                            "trajectory": trajectory,
                            "contrast": "NA",
                            "effect_type": effect_type,
                            "jackknife_axis": axis,
                            "left_out_block": value,
                            "delta_auc_leave_one_out": estimate,
                            "direction_preserved": match,
                            "n_blocks_remaining": n_remaining,
                        }
                    )
    boot_df = pd.DataFrame(bootstrap_rows)
    jack_df = pd.DataFrame(jackknife_rows)
    boot_df.to_csv(out_dir / "gse271399_block_bootstrap_family_effects.tsv", sep="\t", index=False)
    jack_df.to_csv(out_dir / "gse271399_block_jackknife_family_effects.tsv", sep="\t", index=False)
    return boot_df, jack_df


def family_specificity_score(inputs: dict[str, pd.DataFrame | Path], out_dir: Path) -> pd.DataFrame:
    effect = inputs["family_effect"].copy()  # type: ignore[index]
    effect["contrast"] = effect["contrast"].fillna("NA").astype(str)
    rows = []
    for key, sub in effect.groupby(["trajectory", "effect_type", "contrast"], observed=True):
        trajectory, effect_type, contrast = key
        sub = sub.copy()
        sub["abs_effect"] = sub["family_delta_auc"].astype(float).abs()
        global_mean = float(sub["abs_effect"].mean())
        ranks = sub["abs_effect"].rank(pct=True, method="average")
        for idx, row in sub.iterrows():
            family_id = str(row["family_id"])
            generic_penalty = 0.35 if any(x in family_id for x in ["CELL_CYCLE", "RIBOSOME", "MYC", "E2F"]) else 0.0
            if "INFLAMMATORY" in family_id:
                generic_penalty = 0.10
            driver_specificity = float(row.get("signed_driver_support", np.nan))
            external_supported = str(row.get("external_supported", "False")).lower() == "true"
            external_bonus = 0.10 if external_supported else 0.0
            psi = float(row["abs_effect"] / (global_mean + 1e-12))
            psi_percentile = float(ranks.loc[idx])
            bio_score = max(
                0.0,
                min(1.0, 0.45 * psi_percentile + 0.35 * driver_specificity + 0.20 * external_bonus - generic_penalty),
            )
            rows.append(
                {
                    "family_id": family_id,
                    "trajectory": trajectory,
                    "contrast": contrast,
                    "effect_type": effect_type,
                    "family_delta_auc": float(row["family_delta_auc"]),
                    "global_abs_effect_mean": global_mean,
                    "PSI": psi,
                    "PSI_percentile": psi_percentile,
                    "generic_penalty": generic_penalty,
                    "driver_specificity": driver_specificity,
                    "external_supported": external_supported,
                    "biological_specificity_score": bio_score,
                    "specificity_pass": bool(
                        psi_percentile >= 0.75 and generic_penalty <= 0.20 and driver_specificity >= 0.70
                    ),
                }
            )
    df = pd.DataFrame(rows).sort_values(["specificity_pass", "PSI_percentile"], ascending=[False, False])
    df.to_csv(out_dir / "gse271399_family_specificity_score.tsv", sep="\t", index=False)
    return df


def family_score_from_gene_sets(
    Z: np.ndarray,
    var_names: pd.Index,
    gene_sets: dict[str, list[str]],
    members: list[str],
    weights: dict[str, float],
    remove_genes: set[str] | None = None,
) -> np.ndarray:
    remove_genes = {g.upper() for g in remove_genes or set()}
    cols = []
    w = []
    for pathway in members:
        genes = [g.upper() for g in gene_sets.get(pathway, []) if g.upper() not in remove_genes]
        present = [g for g in genes if g in var_names]
        if len(present) < 2:
            continue
        cols.append(module_score_from_z(Z, var_names, present))
        w.append(float(weights.get(pathway, 0.0)))
    if not cols:
        return np.zeros(Z.shape[0], dtype=np.float32)
    matrix = np.column_stack(cols)
    w_arr = np.asarray(w, dtype=float)
    w_arr = w_arr / max(w_arr.sum(), 1e-12)
    return np.average(matrix, axis=1, weights=w_arr).astype(np.float32)


def effect_delta_from_score(adata, score: np.ndarray, trajectory: str, effect_type: str, contrast: str) -> float:
    obs = adata.obs
    mask = obs[f"in_trajectory_{trajectory}"].to_numpy(dtype=bool)
    pt_bin = bin_labels(obs[f"pseudotime_{trajectory}"].to_numpy(dtype=float), 8)
    if effect_type == "pairwise_contrast":
        case, ref = CONTRAST_TO_CASE_REF[contrast]
        deltas = []
        for b in range(8):
            case_mask = mask & (pt_bin == b) & (obs["condition"].to_numpy() == case)
            ref_mask = mask & (pt_bin == b) & (obs["condition"].to_numpy() == ref)
            if case_mask.sum() >= 5 and ref_mask.sum() >= 5:
                deltas.append(float(np.nanmean(score[case_mask]) - np.nanmean(score[ref_mask])))
        return float(np.nanmean(deltas)) if deltas else np.nan
    t21 = (obs["chr21_status"].to_numpy() == "T21").astype(float)
    gata = (obs["gata1_status"].to_numpy() == "GATA1s").astype(float)
    inter = t21 * gata
    beta_idx = {"T21": 1, "GATA1s": 2, "interaction": 3}[effect_type]
    betas = []
    for b in range(8):
        m = mask & (pt_bin == b)
        if m.sum() < 20:
            continue
        x = np.column_stack([np.ones(m.sum()), t21[m], gata[m], inter[m]])
        beta = weighted_lstsq_beta(score[m], x, np.ones(m.sum()))
        if np.isfinite(beta[beta_idx]):
            betas.append(float(beta[beta_idx]))
    return float(np.nanmean(betas)) if betas else np.nan


def matched_random_genes(
    var_names: pd.Index,
    gene_sets: dict[str, list[str]],
    family_members: list[str],
    removed_present: list[str],
    expression_rank: pd.Series,
    rng: np.random.Generator,
) -> list[str]:
    if not removed_present:
        return []
    family_genes = {g.upper() for p in family_members for g in gene_sets.get(p, [])}
    present = set(var_names.astype(str).str.upper().tolist())
    # Random ablation must remove genes from the same family score definition;
    # otherwise the random removal leaves the family score unchanged.
    candidates = sorted((family_genes & present) - {g.upper() for g in removed_present})
    if not candidates:
        return []
    ranks = expression_rank.reindex(candidates).dropna().sort_values()
    if ranks.empty:
        return list(rng.choice(candidates, size=min(len(removed_present), len(candidates)), replace=False))
    picked = []
    for gene in removed_present:
        target = expression_rank.get(gene, np.nan)
        if not np.isfinite(target):
            target = float(ranks.median())
        nearest = (ranks - target).abs().sort_values().index.tolist()
        nearest = [g for g in nearest[:50] if g not in picked]
        if nearest:
            picked.append(str(rng.choice(nearest)))
    return picked


def driver_module_ablation(
    adata,
    Z: np.ndarray,
    gene_sets: dict[str, list[str]],
    families: dict[str, tuple[list[str], dict[str, float]]],
    out_dir: Path,
    n_random: int,
) -> pd.DataFrame:
    family_id = "ERYTHROID_EVENT_LOSS_FAMILY"
    members, weights = families[family_id]
    baseline = family_score_from_gene_sets(Z, adata.var_names, gene_sets, members, weights)
    x = adata.X
    mean_expr = np.asarray(x.mean(axis=0)).ravel()
    expression_rank = pd.Series(pd.Series(mean_expr, index=adata.var_names.astype(str).str.upper()).rank(), dtype=float)
    rng = np.random.default_rng(RNG_SEED)
    tests = [(e, "NA") for e in EFFECT_TYPES] + [("pairwise_contrast", c) for c in PAIRWISE_CONTRASTS]
    rows = []
    for trajectory in TRAJECTORY_TARGET:
        before_by_test = {
            (effect_type, contrast): effect_delta_from_score(adata, baseline, trajectory, effect_type, contrast)
            for effect_type, contrast in tests
        }
        for module_name, genes in DRIVER_MODULES.items():
            removed_present = [g for g in [x.upper() for x in genes] if g in adata.var_names]
            after_score = family_score_from_gene_sets(Z, adata.var_names, gene_sets, members, weights, set(removed_present))
            random_scores = []
            for _ in range(n_random):
                rand = matched_random_genes(adata.var_names, gene_sets, members, removed_present, expression_rank, rng)
                random_scores.append(
                    family_score_from_gene_sets(Z, adata.var_names, gene_sets, members, weights, set(rand))
                )
            for effect_type, contrast in tests:
                before = before_by_test[(effect_type, contrast)]
                after = effect_delta_from_score(adata, after_score, trajectory, effect_type, contrast)
                delta_loss = abs(before) - abs(after) if np.isfinite(before) and np.isfinite(after) else np.nan
                random_losses = []
                for random_score in random_scores:
                    rand_after = effect_delta_from_score(adata, random_score, trajectory, effect_type, contrast)
                    if np.isfinite(before) and np.isfinite(rand_after):
                        random_losses.append(abs(before) - abs(rand_after))
                random_mean = float(np.nanmean(random_losses)) if random_losses else np.nan
                random_sd = float(np.nanstd(random_losses, ddof=1)) if len(random_losses) > 1 else 0.0
                dep = (delta_loss - random_mean) / (abs(before) + 1e-12) if np.isfinite(delta_loss) and np.isfinite(random_mean) else np.nan
                rows.append(
                    {
                        "family_id": family_id,
                        "trajectory": trajectory,
                        "contrast": contrast,
                        "effect_type": effect_type,
                        "driver_module": module_name,
                        "removed_genes": ",".join(removed_present),
                        "family_delta_auc_before": before,
                        "family_delta_auc_after": after,
                        "delta_loss": delta_loss,
                        "matched_random_delta_loss_mean": random_mean,
                        "matched_random_delta_loss_sd": random_sd,
                        "dependency_score": dep,
                        "dependency_pass": bool(np.isfinite(dep) and dep > 0.05 and delta_loss > max(random_mean, 0)),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "gse271399_driver_module_ablation.tsv", sep="\t", index=False)
    return df


def write_manifest(out_dir: Path) -> None:
    rows = []
    for file in sorted(out_dir.glob("*.tsv")):
        if file.name == "bundle_manifest.tsv":
            continue
        rows.append({"file_name": file.name, "bytes": file.stat().st_size, "path": str(file.resolve())})
    pd.DataFrame(rows).to_csv(out_dir / "bundle_manifest.tsv", sep="\t", index=False)


def copy_outputs(out_dir: Path, root: Path) -> None:
    dataset_dir = root / "GSE271399_T21_GATA1s"
    deliverables = dataset_dir / "deliverables_family_mechanism_hardening"
    bundle = root / "deliverables_all_ted_rounds" / "GSE271399_T21_GATA1s"
    deliverables.mkdir(parents=True, exist_ok=True)
    bundle.mkdir(parents=True, exist_ok=True)
    for file in sorted(out_dir.glob("*.tsv")):
        shutil.copy2(file, deliverables / file.name)
        shutil.copy2(file, bundle / file.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data_external")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--n-random", type=int, default=30)
    args = parser.parse_args()

    root = Path(args.root)
    inputs = load_inputs(root)
    dataset_dir: Path = inputs["dataset_dir"]  # type: ignore[assignment]
    ted_dir: Path = inputs["ted_dir"]  # type: ignore[assignment]
    out_dir = dataset_dir / "ted_family_mechanism_hardening"
    out_dir.mkdir(parents=True, exist_ok=True)

    perturbation_attribution_index(inputs, out_dir)
    family_specificity_score(inputs, out_dir)

    gene_sets = read_gmt(dataset_dir / "metadata" / "gse271399_ted_gene_sets.gmt")
    families = family_members_and_weights(inputs["family_def"])  # type: ignore[arg-type]
    adata = load_cached_adata(dataset_dir)
    scores, Z = add_scores_and_states(adata, gene_sets)
    family_scores = build_family_scores(scores, families)

    block_uncertainty_outputs(adata, family_scores, out_dir, n_boot=args.n_boot)
    driver_module_ablation(adata, Z, gene_sets, families, out_dir, n_random=args.n_random)

    for file in out_dir.glob("gse271399_*.tsv"):
        shutil.copy2(file, ted_dir / file.name)
    write_manifest(out_dir)
    copy_outputs(out_dir, root)
    print("[family-mechanism-hardening] done")


if __name__ == "__main__":
    main()
