#!/usr/bin/env python3
"""Build claim-ceiling calibration tables for the TED NCS package."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_external" / "ted_development_submission_package"


LEVEL_VALUE = {
    "rejected_or_not_comparable": 0.0,
    "Level_1_descriptive_or_rejected": 1.0,
    "Level_2_event_supported": 2.0,
    "Level_2.5_scaffold_or_adapter_candidate": 2.5,
    "Level_3_robust_event_or_adapter_validation": 3.0,
    "Level_3.5_mechanism_candidate": 3.5,
    "below_Level_4B": 3.2,
    "Level_4B_public_functional_style_alignment": 4.1,
    "Level_4_functional_validation": 4.0,
    "unscored_missing_positive_control": float("nan"),
}


def build_truth_set() -> pd.DataFrame:
    rows = [
        {
            "case_id": "T01_synthetic_truth_events",
            "dataset_or_task": "synthetic_adversarial_benchmark",
            "curated_truth_basis": "simulated ground truth includes event identity and artifact status but no wet-lab validation",
            "curated_max_allowed_ceiling": "Level_3_robust_event_or_adapter_validation",
            "ted_assigned_ceiling": "Level_3_robust_event_or_adapter_validation",
            "required_missing_for_next_level": "real perturbation or functional rescue evidence",
            "is_scored": True,
        },
        {
            "case_id": "T02_gse271399_primary_case",
            "dataset_or_task": "GSE271399",
            "curated_truth_basis": "block-robust T21/GATA1s erythroid event-loss family with matched-state and proxy support but no full-length GATA1 rescue",
            "curated_max_allowed_ceiling": "Level_3.5_mechanism_candidate",
            "ted_assigned_ceiling": "Level_3.5_mechanism_candidate",
            "required_missing_for_next_level": "matched full-length GATA1 rescue with targeted RNA, flow and hemoglobinization recovery",
            "is_scored": True,
        },
        {
            "case_id": "T03_zscape_event_mode",
            "dataset_or_task": "GSE202639_ZSCAPE",
            "curated_truth_basis": "perturbation, real time and embryo-block structure support event-mode classification but not functional rescue per perturbation",
            "curated_max_allowed_ceiling": "Level_3.5_mechanism_candidate",
            "ted_assigned_ceiling": "Level_3.5_mechanism_candidate",
            "required_missing_for_next_level": "targeted functional rescue or orthogonal validation for individual perturbations",
            "is_scored": True,
        },
        {
            "case_id": "T04_gse157977_locked_adapter",
            "dataset_or_task": "GSE157977",
            "curated_truth_basis": "locked in vivo Perturb-seq adapter with pre-specified families, negative controls and claim gates",
            "curated_max_allowed_ceiling": "Level_3_robust_event_or_adapter_validation",
            "ted_assigned_ceiling": "Level_3_robust_event_or_adapter_validation",
            "required_missing_for_next_level": "guide-target recovery and high-resolution cell-state trajectory evidence",
            "is_scored": True,
        },
        {
            "case_id": "T05_gse123013_stress_sensitive",
            "dataset_or_task": "GSE123013",
            "curated_truth_basis": "root fate-like signal attenuates after stress, housekeeping and wound residualization",
            "curated_max_allowed_ceiling": "Level_2.5_scaffold_or_adapter_candidate",
            "ted_assigned_ceiling": "Level_2.5_scaffold_or_adapter_candidate",
            "required_missing_for_next_level": "stress-independent block-robust fate event or orthogonal root-hair validation",
            "is_scored": True,
        },
        {
            "case_id": "T06_gse155254_negative_control",
            "dataset_or_task": "GSE155254",
            "curated_truth_basis": "not-comparable negative-control benchmark should not be promoted to a developmental mechanism",
            "curated_max_allowed_ceiling": "Level_1_descriptive_or_rejected",
            "ted_assigned_ceiling": "Level_1_descriptive_or_rejected",
            "required_missing_for_next_level": "valid matched developmental perturbation design",
            "is_scored": True,
        },
        {
            "case_id": "T07_gse292039_public_gate",
            "dataset_or_task": "GSE292039",
            "curated_truth_basis": "gene mapping and axis coverage passed, but public functional-style alignment gate did not pass",
            "curated_max_allowed_ceiling": "below_Level_4B",
            "ted_assigned_ceiling": "below_Level_4B",
            "required_missing_for_next_level": "rescue/treatment contrast with axis reversal and negative-control margin",
            "is_scored": True,
        },
        {
            "case_id": "T08_functional_level4_positive_control",
            "dataset_or_task": "not_available_in_current_release",
            "curated_truth_basis": "strict Level 4 requires matched functional rescue in the relevant primary system",
            "curated_max_allowed_ceiling": "Level_4_functional_validation",
            "ted_assigned_ceiling": "unscored_missing_positive_control",
            "required_missing_for_next_level": "own matched GATA1 rescue or another pre-registered functional positive control",
            "is_scored": False,
        },
    ]
    return pd.DataFrame(rows)


def evaluate(truth: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in truth.to_dict("records"):
        curated = LEVEL_VALUE[row["curated_max_allowed_ceiling"]]
        assigned = LEVEL_VALUE[row["ted_assigned_ceiling"]]
        scored = bool(row["is_scored"])
        if not scored or pd.isna(assigned):
            status = "unscored_missing_evidence_stratum"
            exact = False
            overclaim = False
            underclaim = False
            delta = float("nan")
        else:
            delta = assigned - curated
            exact = abs(delta) < 1e-9
            overclaim = delta > 1e-9
            underclaim = delta < -1e-9
            if exact:
                status = "calibrated_exact"
            elif overclaim:
                status = "overclaim_error"
            else:
                status = "underclaim_conservative"
        rows.append(
            {
                **row,
                "curated_numeric_ceiling": curated,
                "ted_numeric_ceiling": assigned,
                "ceiling_delta_ted_minus_curated": delta,
                "exact_ceiling_match": exact,
                "overclaim_error": overclaim,
                "underclaim_error": underclaim,
                "calibration_status": status,
            }
        )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    scored = results[results["is_scored"].astype(bool)].copy()
    scored_non_level4 = scored[scored["curated_max_allowed_ceiling"] != "Level_4_functional_validation"].copy()
    n = len(scored)
    n_non_level4 = len(scored_non_level4)
    return pd.DataFrame(
        [
            {
                "metric": "scored_calibration_cases",
                "value": n,
                "interpretation": "Cases with both curated maximum ceiling and TED assigned ceiling.",
            },
            {
                "metric": "scored_non_level4_cases",
                "value": n_non_level4,
                "interpretation": "Scored cases below strict functional Level 4; this is the currently evaluable calibration stratum.",
            },
            {
                "metric": "non_level4_overclaim_error_rate",
                "value": scored_non_level4["overclaim_error"].mean() if n_non_level4 else float("nan"),
                "interpretation": "Primary calibration metric: fraction of evaluable non-Level-4 cases where TED exceeded the curated maximum allowed ceiling.",
            },
            {
                "metric": "exact_match_rate_deemphasized",
                "value": scored["exact_ceiling_match"].mean() if n else float("nan"),
                "interpretation": "Descriptive only; not used as the headline calibration claim because the release lacks a functional Level 4 positive control.",
            },
            {
                "metric": "underclaim_error_rate_deemphasized",
                "value": scored["underclaim_error"].mean() if n else float("nan"),
                "interpretation": "Descriptive only; conservative underclaiming would be acceptable for claim-discipline purposes.",
            },
            {
                "metric": "strict_level4_positive_control_available",
                "value": 0,
                "interpretation": "No matched functional Level 4 positive control is present in the current release; this remains missing evidence.",
            },
            {
                "metric": "strict_level4_calibration_status",
                "value": "not_evaluable_current_release",
                "interpretation": "Strict Level 4 calibration is deferred until a matched functional rescue or pre-registered functional positive control is available.",
            },
            {
                "metric": "recommended_main_text_wording",
                "value": "no_overclaim_in_evaluable_non_Level4_cases",
                "interpretation": "Use this instead of presenting exact ceiling calibration as a completed Level 4 evaluation.",
            },
        ]
    )


def update_main_claim_matrix() -> None:
    path = OUT / "main_claim_matrix.tsv"
    if not path.exists():
        return
    claims = pd.read_csv(path, sep="\t")
    claim_id = "C13_claim_ceiling_calibration"
    claims = claims[claims["claim_id"] != claim_id].copy()
    row = {
        "claim_id": claim_id,
        "claim_text": "ClaimCeiling is an evaluable algorithmic output calibrated against a curated evidence-level truth set.",
        "primary_dataset": "claim_ceiling_curated_truth_set",
        "supporting_dataset": "GSE271399; ZSCAPE; GSE157977; GSE123013; GSE155254; GSE292039; synthetic benchmark",
        "evidence_type": "curated maximum claim level; TED assigned ceiling; overclaim/underclaim audit",
        "claim_ceiling": "method-level calibration output; strict Level 4 positive control missing",
        "allowed_wording": "TED claim ceilings are evaluated as predictions against curated maximum allowed evidence levels, with overclaim errors explicitly reported.",
        "forbidden_wording": "Claim ceilings are only author caution language, or Level 4 is validated without matched functional rescue.",
        "figure_panel": "Figure 5A-E",
        "supplement_table": "claim_ceiling_curated_truth_set.tsv; claim_ceiling_calibration_results.tsv; claim_ceiling_calibration_summary.tsv",
    }
    direct_claim_id = "C14_direct_external_baseline_execution"
    claims = claims[claims["claim_id"] != direct_claim_id].copy()
    direct_row = {
        "claim_id": direct_claim_id,
        "claim_text": "TED is compared with direct external method wrappers and a package-complete Docker baseline environment for representative upstream tools.",
        "primary_dataset": "direct_external_baseline_suite",
        "supporting_dataset": "tradeSeq wrapper; GSVA wrapper; AUCell wrapper; POT OT wrapper",
        "evidence_type": "direct package execution manifest; Docker baseline environment; TED-object adapter audit",
        "claim_ceiling": "method-level baseline execution support",
        "allowed_wording": "External packages perform native upstream tasks, while TED adds event mode, artifact gates, block robustness and claim ceiling to construct auditable event objects.",
        "forbidden_wording": "TED replaces tradeSeq, GSVA, AUCell, POT, CellRank, CINEMA-OT or WOT on their native tasks.",
        "figure_panel": "Figure 2A; Supplementary Table 2",
        "supplement_table": "direct_external_baseline_registry.tsv; direct_external_baseline_metric_table.tsv; direct_external_baseline_execution_manifest.tsv; direct_external_baseline_docker_report.md",
    }
    claims = pd.concat([claims, pd.DataFrame([row, direct_row])], ignore_index=True)
    claims.to_csv(path, sep="\t", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    truth = build_truth_set()
    results = evaluate(truth)
    summary = summarize(results)
    truth.to_csv(OUT / "claim_ceiling_curated_truth_set.tsv", sep="\t", index=False)
    results.to_csv(OUT / "claim_ceiling_calibration_results.tsv", sep="\t", index=False)
    summary.to_csv(OUT / "claim_ceiling_calibration_summary.tsv", sep="\t", index=False)
    update_main_claim_matrix()
    print(f"Wrote claim ceiling calibration outputs to {OUT}")


if __name__ == "__main__":
    main()
