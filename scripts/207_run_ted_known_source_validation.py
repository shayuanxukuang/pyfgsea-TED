from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "config" / "ted_upgrade_benchmark_registry.tsv"
DEFAULT_CARDS = ROOT / "config" / "ted_preregistration_cards.tsv"
DEFAULT_PROCESSED = ROOT / "data" / "processed" / "ted_known_source"
DEFAULT_OUT = ROOT / "results" / "ted_known_source_validation" / "tables"

IFNG_PDL1_AXIS = ["CD274", "IRF1", "STAT1", "STAT2", "JAK2", "IFNGR1", "IFNGR2", "CXCL10", "GBP1", "TAP1"]
NEGATIVE_AXES = {
    "stress_axis": ["HSPA1A", "HSPA1B", "HSP90AA1", "DNAJB1", "JUN", "FOS", "ATF3", "DDIT3"],
    "ribosome_axis": ["RPL3", "RPL4", "RPL5", "RPL7", "RPL8", "RPL10", "RPL13A", "RPS3", "RPS6", "RPS8"],
    "mitochondrial_axis": ["MT-CO1", "MT-CO2", "MT-CO3", "MT-CYB", "MT-ND1", "MT-ND2", "MT-ATP6"],
}


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def bh_fdr(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return q
    idx = np.where(finite)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.clip(ranked, 0, 1)
    return q


def read_selected_rows(path: Path, features: list[str]) -> pd.DataFrame:
    wanted = {feature.upper(): feature for feature in features}
    rows: dict[str, list[float]] = {}
    cells: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        cells = [cell.strip().strip('"') for cell in header[1:]]
        for row in reader:
            if not row:
                continue
            feature = row[0].strip().strip('"')
            key = feature.upper()
            if key in wanted:
                rows[wanted[key]] = [float(x) if x not in {"", "NA"} else 0.0 for x in row[1:]]
    if not rows:
        return pd.DataFrame(index=cells)
    return pd.DataFrame(rows, index=cells)


def zscore_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    values = np.log1p(df.astype(float))
    return (values - values.mean(axis=0)) / values.std(axis=0, ddof=0).replace(0, np.nan)


def axis_score(matrix: pd.DataFrame, genes: list[str]) -> pd.Series:
    available = [gene for gene in genes if gene in matrix.columns]
    if not available:
        return pd.Series(dtype=float)
    z = zscore_columns(matrix[available])
    return z.mean(axis=1).rename("axis_score")


def group_effects(scores: pd.Series, metadata: pd.DataFrame, group_col: str = "perturbed_gene") -> pd.DataFrame:
    data = pd.concat([scores, metadata[[group_col, "replicate", "is_non_targeting"]]], axis=1).dropna()
    nt = data[data["is_non_targeting"].astype(bool)]
    nt_values = nt["axis_score"].astype(float)
    nt_mean = float(nt_values.mean()) if len(nt_values) else np.nan
    rows = []
    for group, sub in data.groupby(group_col):
        vals = sub["axis_score"].astype(float)
        if len(vals) < 20:
            continue
        if len(nt_values) > 1 and len(vals) > 1:
            p = float(stats.ttest_ind(vals, nt_values, equal_var=False, nan_policy="omit").pvalue)
        else:
            p = np.nan
        rep_effects = []
        for rep, rep_sub in sub.groupby("replicate"):
            nt_rep = nt[nt["replicate"].astype(str).eq(str(rep))]["axis_score"].astype(float)
            if len(nt_rep):
                rep_effects.append(float(rep_sub["axis_score"].mean() - nt_rep.mean()))
        effect = float(vals.mean() - nt_mean) if np.isfinite(nt_mean) else np.nan
        sign = np.sign(effect)
        stability = float(np.mean([np.sign(x) == sign for x in rep_effects])) if rep_effects and sign != 0 else np.nan
        rows.append(
            {
                "perturbation": group,
                "n_cells": len(vals),
                "event_effect_size": effect,
                "event_direction": "up" if effect > 0 else "down" if effect < 0 else "flat",
                "p_value": p,
                "block_support": stability,
                "direction_stability": stability,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["event_q_value"] = bh_fdr(out["p_value"].tolist())
    return out


def run_gse153056(processed_root: Path, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = processed_root / "GSE153056"
    meta = pd.read_csv(root / "cell_metadata.tsv.gz", sep="\t")
    meta = meta.set_index("cell_id", drop=False)
    expression = read_selected_rows(root / "expression_matrix.tsv.gz", IFNG_PDL1_AXIS + sum(NEGATIVE_AXES.values(), []))
    protein = read_selected_rows(root / "protein_matrix.tsv.gz", ["PDL1"])
    shared = meta.index.intersection(expression.index).intersection(protein.index)
    meta = meta.loc[shared].copy()
    expression = expression.loc[shared]
    protein = protein.loc[shared]

    event = axis_score(expression, IFNG_PDL1_AXIS)
    event_scores = group_effects(event, meta)
    pdl1 = np.log1p(protein["PDL1"].astype(float)).rename("axis_score")
    outcome = group_effects(pdl1, meta)
    outcome = outcome.rename(
        columns={
            "event_effect_size": "pdl1_protein_effect_size",
            "event_direction": "pdl1_protein_direction",
            "p_value": "pdl1_p_value",
            "event_q_value": "pdl1_q_value",
        }
    )
    alignment = event_scores.merge(
        outcome[["perturbation", "pdl1_protein_effect_size", "pdl1_protein_direction", "pdl1_p_value", "pdl1_q_value"]],
        on="perturbation",
        how="inner",
    )
    alignment["event_outcome_direction_match"] = np.sign(alignment["event_effect_size"]) == np.sign(
        alignment["pdl1_protein_effect_size"]
    )
    finite = alignment[["event_effect_size", "pdl1_protein_effect_size"]].replace([np.inf, -np.inf], np.nan).dropna()
    corr = float(finite.corr(method="spearman").iloc[0, 1]) if len(finite) >= 3 else np.nan

    arrayed_meta = pd.read_csv(root / "arrayed_cell_metadata.tsv.gz", sep="\t").set_index("cell_id", drop=False)
    arrayed_expression = read_selected_rows(root / "arrayed_expression_matrix.tsv.gz", IFNG_PDL1_AXIS)
    shared_arrayed = arrayed_meta.index.intersection(arrayed_expression.index)
    baseline_score = axis_score(arrayed_expression.loc[shared_arrayed], IFNG_PDL1_AXIS)
    baseline = pd.concat([baseline_score, arrayed_meta.loc[shared_arrayed, ["condition"]]], axis=1).dropna()
    ifng = baseline[baseline["condition"].eq("IFNg")]["axis_score"]
    ctrl = baseline[baseline["condition"].eq("control")]["axis_score"]
    baseline_effect = float(ifng.mean() - ctrl.mean()) if len(ifng) and len(ctrl) else np.nan
    baseline_p = (
        float(stats.ttest_ind(ifng, ctrl, equal_var=False, nan_policy="omit").pvalue)
        if len(ifng) > 1 and len(ctrl) > 1
        else np.nan
    )

    negative_rows = []
    for axis, genes in NEGATIVE_AXES.items():
        score = axis_score(expression, genes)
        effects = group_effects(score, meta) if not score.empty else pd.DataFrame()
        max_abs = float(effects["event_effect_size"].abs().max()) if not effects.empty else np.nan
        negative_rows.append(
            {
                "dataset": "GSE153056",
                "negative_control": axis,
                "max_abs_effect_size": max_abs,
                "negative_control_pass": bool(np.isnan(max_abs) or max_abs < alignment["event_effect_size"].abs().quantile(0.9)),
            }
        )
    negative = pd.DataFrame(negative_rows)

    direction_match_fraction = float(alignment["event_outcome_direction_match"].mean()) if len(alignment) else np.nan
    outcome_pass = bool(np.isfinite(corr) and corr > 0.25 and direction_match_fraction >= 0.6)
    baseline_pass = bool(np.isfinite(baseline_effect) and baseline_effect > 0 and (not np.isfinite(baseline_p) or baseline_p <= 0.05))
    neg_pass = bool(negative["negative_control_pass"].all()) if not negative.empty else False
    claim = pd.DataFrame(
        [
            {
                "dataset": "GSE153056",
                "claim_boundary": "outcome_supported_event" if outcome_pass and baseline_pass and neg_pass else "known_source_event_partial",
                "status": "pass" if outcome_pass and baseline_pass and neg_pass else "partial",
                "reason": f"baseline_effect={baseline_effect:.4g}; spearman_event_pdl1={corr:.4g}; direction_match={direction_match_fraction:.3g}; negative_controls={neg_pass}",
                "matched_rescue_design": False,
                "same_system": False,
                "level4_causal_rescue": False,
            }
        ]
    )
    event_scores.insert(0, "dataset", "GSE153056")
    event_scores.insert(1, "candidate_event", "IFNg_JAK_STAT_PDL1")
    event_scores["baseline_ifng_vs_control_effect"] = baseline_effect
    event_scores["baseline_ifng_vs_control_p"] = baseline_p
    alignment.insert(0, "dataset", "GSE153056")
    alignment["outcome_correlation"] = corr
    alignment["outcome_alignment_pass"] = outcome_pass

    write_tsv(event_scores, out / "gse153056_ted_event_scores.tsv")
    write_tsv(alignment, out / "gse153056_pdl1_outcome_alignment.tsv")
    write_tsv(negative, out / "gse153056_negative_control_results.tsv")
    write_tsv(claim, out / "gse153056_claim_boundary.tsv")
    return event_scores, alignment, negative, claim


def run_gse93735(processed_root: Path, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = processed_root / "GSE93735"
    effects = pd.read_csv(root / "gse93735_matched_intervention_effects.tsv", sep="\t")
    summary = pd.read_csv(root / "gse93735_matched_intervention_positive_stratum_summary.tsv", sep="\t")
    effects["dataset"] = "GSE93735"
    effects["candidate_event"] = "LPS_inflammatory_activation"
    effects["activation_score"] = pd.to_numeric(effects["LPS10h_vs_baseline_delta"], errors="coerce")
    effects["reversal_score"] = -pd.to_numeric(effects["DexLate10h_vs_LPS10h_delta"], errors="coerce")
    effects["reversal_fraction"] = pd.to_numeric(effects["recovery_fraction"], errors="coerce")
    effects["reversal_class"] = pd.cut(
        effects["reversal_fraction"],
        bins=[-np.inf, 0.2, 0.5, np.inf],
        labels=["weak_or_no_reversal", "partial_reversal", "strong_reversal"],
    ).astype(str)

    activation = effects[effects["role"].str.contains("positive|primary|secondary", case=False, na=False)][
        ["dataset", "axis", "candidate_event", "role", "activation_score", "n_detected_genes"]
    ].copy()
    reversal = effects[effects["role"].str.contains("positive|primary|secondary", case=False, na=False)][
        ["dataset", "axis", "candidate_event", "role", "reversal_score", "reversal_fraction", "reversal_class"]
    ].copy()
    neg = effects[effects["role"].eq("negative_control_axis")][
        ["dataset", "axis", "activation_score", "reversal_score", "reversal_fraction", "reversal_class"]
    ].copy()
    neg = neg.rename(columns={"axis": "negative_control"})
    primary_recovery = float(summary.iloc[0].get("primary_recovery_fraction", np.nan))
    neg_max = float(summary.iloc[0].get("max_negative_control_recovery_fraction", np.nan))
    neg["negative_control_pass"] = neg["reversal_fraction"] < primary_recovery
    neg["primary_recovery_fraction"] = primary_recovery
    neg["max_negative_control_recovery_fraction"] = neg_max
    claim = pd.DataFrame(
        [
            {
                "dataset": "GSE93735",
                "claim_boundary": "reversal_supported_event",
                "status": "pass" if str(summary.iloc[0].get("matched_validation_positive_stratum")).lower() == "pass" else "fail",
                "reason": f"primary_recovery={primary_recovery:.4g}; max_negative_control={neg_max:.4g}",
                "matched_rescue_design": False,
                "same_system": False,
                "level4_causal_rescue": False,
            }
        ]
    )
    write_tsv(activation, out / "gse93735_lps_activation_event.tsv")
    write_tsv(reversal, out / "gse93735_dex_reversal_event.tsv")
    write_tsv(effects[["dataset", "axis", "role", "activation_score", "reversal_score", "reversal_fraction", "reversal_class"]], out / "gse93735_reversal_index.tsv")
    write_tsv(neg, out / "gse93735_negative_control_results.tsv")
    write_tsv(claim, out / "gse93735_claim_boundary.tsv")
    event = reversal.rename(
        columns={"axis": "perturbation", "reversal_score": "event_effect_size"}
    )
    event["event_direction"] = "reversal"
    event["event_q_value"] = np.nan
    event["block_support"] = summary.iloc[0].get("primary_gate_pass", "")
    event["direction_stability"] = summary.iloc[0].get("engagement_gate_pass", "")
    align = effects[["dataset", "axis", "reversal_fraction", "reversal_class"]].copy()
    align["outcome_alignment_pass"] = effects["reversal_fraction"] > 0.2
    align["outcome_correlation"] = np.nan
    return event, align, neg, claim


def pending_dataset(dataset: str, reason: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event = pd.DataFrame([{"dataset": dataset, "candidate_event": "pending_access", "status": "pending", "reason": reason}])
    align = pd.DataFrame([{"dataset": dataset, "status": "pending", "reason": reason}])
    neg = pd.DataFrame([{"dataset": dataset, "negative_control_pass": False, "status": "pending", "reason": reason}])
    claim = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "claim_boundary": "not_evaluable",
                "status": "pending",
                "reason": reason,
                "matched_rescue_design": False,
                "same_system": False,
                "level4_causal_rescue": False,
            }
        ]
    )
    return event, align, neg, claim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--cards", type=Path, default=DEFAULT_CARDS)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    events = []
    alignments = []
    negatives = []
    claims = []

    e, a, n, c = run_gse153056(args.processed_root, args.out)
    events.append(e)
    alignments.append(a)
    negatives.append(n)
    claims.append(c)

    e, a, n, c = run_gse93735(args.processed_root, args.out)
    events.append(e)
    alignments.append(a)
    negatives.append(n)
    claims.append(c)

    for dataset, reason in [
        ("GSE90546", "GSE90546 raw expression matrix not downloaded in first-pass audit"),
        ("GSE90063", "supplementary adapter prepared from WT UMI summaries only"),
        ("GSE133344", "supplementary scalability adapter has metadata but no expression matrix in first-pass audit"),
    ]:
        e, a, n, c = pending_dataset(dataset, reason)
        events.append(e)
        alignments.append(a)
        negatives.append(n)
        claims.append(c)

    write_tsv(pd.concat(events, ignore_index=True, sort=False), args.out / "ted_event_recovery_summary.tsv")
    write_tsv(pd.concat(alignments, ignore_index=True, sort=False), args.out / "ted_outcome_alignment_summary.tsv")
    write_tsv(pd.concat(negatives, ignore_index=True, sort=False), args.out / "ted_negative_control_summary.tsv")
    write_tsv(pd.concat(claims, ignore_index=True, sort=False), args.out / "ted_dataset_level_claim_boundary.tsv")
    print(f"Wrote known-source validation tables to {args.out}")


if __name__ == "__main__":
    main()
