"""Benchmarks for TED-MAD/ARD mechanism adjudication.

These benchmarks are intentionally lightweight and table driven. They are not a
replacement for real biological validation; they test whether TED-MAD behaves
as a reusable mechanism-decision protocol rather than a GATA1/T21 case script.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .core import (
    adjudicate_mechanism,
    design_rescue_experiments,
    load_ted_mad_yaml,
)


HYPOTHESES = [
    ("H0_noise_batch", "noise/batch/family artifact"),
    ("H1_state_composition", "state/composition artifact"),
    ("H2_proliferation_confounded", "proliferation-confounded mechanism"),
    ("H3_TF_regulatory", "TF regulatory driver"),
    ("H4_downstream_maturation", "downstream maturation/heme mechanism"),
    ("H5_background_chromatin", "background-specific chromatin interaction"),
]

ARTIFACT_HYPOTHESES = {"H0_noise_batch", "H1_state_composition", "H2_proliferation_confounded"}
MECHANISM_HYPOTHESES = {"H3_TF_regulatory", "H4_downstream_maturation", "H5_background_chromatin"}


SCENARIOS = {
    "sim-H0_batch_family_artifact": {
        "truth": "H0_noise_batch",
        "best_experiment": "A0_family_block_validation",
        "claim_ceiling": 2.0,
        "families": ["family_block_robustness", "negative_mediator_controls"],
    },
    "sim-H1_state_composition_artifact": {
        "truth": "H1_state_composition",
        "best_experiment": "A1_state_matched_sort",
        "claim_ceiling": 2.0,
        "families": ["state_matched", "counterfactual_ot", "negative_mediator_controls"],
    },
    "sim-H2_proliferation_confounded_event": {
        "truth": "H2_proliferation_confounded",
        "best_experiment": "A2_proliferation_normalized_condition",
        "claim_ceiling": 2.0,
        "families": ["proliferation_adjusted_mediation", "negative_mediator_controls"],
    },
    "sim-H3_TF_regulatory_driver": {
        "truth": "H3_TF_regulatory",
        "best_experiment": "A3_TF_rescue",
        "claim_ceiling": 3.5,
        "families": [
            "family_block_robustness",
            "proliferation_adjusted_mediation",
            "counterfactual_ot",
            "day_stratified_timing",
            "external_GATA1_KD",
        ],
    },
    "sim-H4_downstream_maturation_heme": {
        "truth": "H4_downstream_maturation",
        "best_experiment": "A4_hemin_rescue",
        "claim_ceiling": 3.0,
        "families": ["rescue_prediction", "day_stratified_timing", "negative_mediator_controls"],
    },
    "sim-H5_background_chromatin_interaction": {
        "truth": "H5_background_chromatin",
        "best_experiment": "A5_chromatin_validation",
        "claim_ceiling": 3.5,
        "families": [
            "family_block_robustness",
            "counterfactual_ot",
            "external_T21_multiome",
            "day_stratified_timing",
        ],
    },
}


def make_benchmark_hypotheses() -> dict[str, Any]:
    """Return schema-compliant generic benchmark hypotheses."""

    return {
        "schema_version": "0.1",
        "model": {
            "base_likelihood_ratio": 2.0,
            "dependency_aggregation": "mean",
            "max_family_abs_log_lr": 2.5,
        },
        "hypotheses": [
            {
                "hypothesis_id": hyp_id,
                "label": label,
                "prior": 1.0 / len(HYPOTHESES),
                "description": f"Synthetic benchmark hypothesis: {label}.",
                "expected_evidence": {"family_block_robustness": "unknown"},
                "falsifiers": [f"Evidence pattern is inconsistent with {hyp_id}."],
            }
            for hyp_id, label in HYPOTHESES
        ],
    }


def make_benchmark_experiments() -> dict[str, Any]:
    """Return a generic experiment library for mechanism benchmark scenarios."""

    return {
        "schema_version": "0.1",
        "design_model": {
            "lambda_claim": 1.0,
            "gamma_falsification": 0.6,
            "cost_weight": 0.15,
            "risk_weight": 0.25,
        },
        "experiments": [
            _experiment(
                "A0_family_block_validation",
                "Family/block replication and batch audit",
                0.25,
                0.15,
                ["family_block_signal", "batch_association", "replicate_consistency"],
                ["H0_noise_batch", "H3_TF_regulatory"],
                "H0_noise_batch",
                {"H0_noise_batch": "batch_signal_persists", "H3_TF_regulatory": "mechanism_signal_unstable"},
                claim=2.0,
            ),
            _experiment(
                "A1_state_matched_sort",
                "Sorted state-matched progenitor comparison",
                0.65,
                0.35,
                ["state_matched_event", "state_composition", "TED_event_score"],
                ["H1_state_composition", "H3_TF_regulatory", "H5_background_chromatin"],
                "H1_state_composition",
                {"H1_state_composition": "event_disappears", "H3_TF_regulatory": "event_persists"},
                claim=3.0,
            ),
            _experiment(
                "A2_proliferation_normalized_condition",
                "Proliferation-normalized condition",
                0.55,
                0.30,
                ["proliferation_module", "TED_event_score", "maturation_module"],
                ["H2_proliferation_confounded", "H3_TF_regulatory"],
                "H2_proliferation_confounded",
                {"H2_proliferation_confounded": "event_normalizes", "H3_TF_regulatory": "event_persists"},
                claim=3.0,
            ),
            _experiment(
                "A3_TF_rescue",
                "TF rescue at early regulatory timepoint",
                0.70,
                0.40,
                ["early_regulatory_module", "TF_target_module", "TED_event_score", "maturation_module"],
                ["H3_TF_regulatory", "H4_downstream_maturation", "H5_background_chromatin"],
                "H3_TF_regulatory",
                {
                    "H3_TF_regulatory": "strong_rescue",
                    "H4_downstream_maturation": "weak_early_rescue",
                    "H5_background_chromatin": "partial_rescue",
                },
                claim=4.0,
            ),
            _experiment(
                "A4_hemin_rescue",
                "Hemin or downstream maturation rescue",
                0.30,
                0.20,
                ["heme_module", "hemoglobinization", "early_regulatory_module"],
                ["H4_downstream_maturation", "H3_TF_regulatory"],
                "H4_downstream_maturation",
                {"H4_downstream_maturation": "strong_rescue", "H3_TF_regulatory": "weak_early_rescue"},
                claim=4.0,
            ),
            _experiment(
                "A5_chromatin_validation",
                "Background-specific ATAC/CUT&Tag validation",
                0.80,
                0.45,
                ["chromatin_accessibility", "TF_enhancer_usage", "TED_event_score"],
                ["H5_background_chromatin", "H3_TF_regulatory"],
                "H5_background_chromatin",
                {"H5_background_chromatin": "chromatin_specific_residual", "H3_TF_regulatory": "full_rescue"},
                claim=4.0,
            ),
        ],
    }


def _experiment(
    experiment_id: str,
    label: str,
    cost: float,
    risk: float,
    readouts: Sequence[str],
    distinguishes: Sequence[str],
    support_hypothesis: str,
    signature: Mapping[str, str],
    *,
    claim: float,
) -> dict[str, Any]:
    expected = {}
    for hyp_id, _ in HYPOTHESES:
        expected[hyp_id] = {
            readout: signature.get(hyp_id, "nonspecific_pattern") for readout in readouts[:2]
        }
    return {
        "experiment_id": experiment_id,
        "label": label,
        "description": label,
        "cost": cost,
        "risk": risk,
        "claim_level_if_success": claim,
        "claim_upgrade_evidence": "direct_rescue" if support_hypothesis in MECHANISM_HYPOTHESES else "",
        "supports_hypotheses": [support_hypothesis],
        "readouts": list(readouts),
        "minimal_readout_panel": {
            "required": list(readouts[: min(3, len(readouts))]),
            "optional": list(readouts[3:]),
            "negative_controls": ["shuffled_mediator_control", "wrong_lineage_module"],
        },
        "distinguishes": list(distinguishes),
        "expected_patterns": expected,
        "falsifiers": [
            {
                "hypothesis": support_hypothesis,
                "rule": f"Expected {label} pattern is absent.",
            }
        ],
    }


def make_synthetic_evidence(
    scenario_name: str,
    *,
    replicate: int = 0,
    correlated_evidence: bool = True,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Generate a schema-compliant synthetic evidence ledger for one mechanism."""

    if scenario_name not in SCENARIOS:
        raise KeyError(f"unknown synthetic scenario: {scenario_name}")
    rng = rng or np.random.default_rng(20260513 + replicate)
    spec = SCENARIOS[scenario_name]
    truth = spec["truth"]
    rows = []
    for i, family in enumerate(spec["families"]):
        rows.append(
            _evidence_row(
                f"{scenario_name}_E{i + 1}",
                family,
                f"{scenario_name}_group_{i + 1}",
                truth,
                rng,
                weight=0.8 + 0.2 * float(rng.random()),
            )
        )

    if correlated_evidence:
        for j in range(3):
            rows.append(
                _evidence_row(
                    f"{scenario_name}_correlated_{j + 1}",
                    spec["families"][0],
                    f"{scenario_name}_shared_correlated_source",
                    truth,
                    rng,
                    weight=0.75,
                )
            )

    distractors = [hyp for hyp, _ in HYPOTHESES if hyp != truth]
    distractor = str(rng.choice(distractors))
    rows.append(
        _evidence_row(
            f"{scenario_name}_weak_distractor",
            "rescue_prediction",
            f"{scenario_name}_distractor",
            distractor,
            rng,
            weight=0.25,
        )
    )
    return {"schema_version": "0.1", "evidence": rows}


def _evidence_row(
    evidence_id: str,
    family: str,
    dependency_group: str,
    truth: str,
    rng: np.random.Generator,
    *,
    weight: float,
) -> dict[str, Any]:
    weakens = {hyp_id: 0.35 for hyp_id, _ in HYPOTHESES if hyp_id != truth}
    effect = 0.75 + 0.35 * float(rng.random())
    se = 0.45 + 0.20 * float(rng.random())
    return {
        "evidence_id": evidence_id,
        "evidence_family": family,
        "dependency_group": dependency_group,
        "target_event": "synthetic_event",
        "direction": "supports",
        "effect_size": effect,
        "standard_error": se,
        "p_value": float(max(0.001, min(0.2, 0.05 * se / max(effect, 1e-6)))),
        "weight": float(weight),
        "assumptions": [f"{family} synthetic assumption holds"],
        "failure_modes": [f"{family} synthetic failure mode"],
        "supports": {truth: 0.85},
        "weakens": weakens,
        "data_source": "synthetic TED-MAD benchmark",
    }


def run_synthetic_mechanism_benchmark(
    *,
    n_replicates: int = 5,
    correlated_evidence: bool = True,
    random_seed: int = 20260513,
) -> dict[str, pd.DataFrame]:
    """Run synthetic mechanism recovery and naive-vs-aware fusion benchmark."""

    hypotheses = make_benchmark_hypotheses()
    experiments = make_benchmark_experiments()
    rng = np.random.default_rng(random_seed)
    rows = []
    for scenario_name, spec in SCENARIOS.items():
        for replicate in range(n_replicates):
            evidence = make_synthetic_evidence(
                scenario_name,
                replicate=replicate,
                correlated_evidence=correlated_evidence,
                rng=rng,
            )
            adjudication = adjudicate_mechanism(
                evidence,
                hypotheses,
                event="synthetic_event",
                strict=True,
                compare_naive=True,
                leave_one_family_out=True,
            )
            posterior = adjudication["posterior"]
            top = str(posterior.iloc[0]["hypothesis"])
            top2 = set(posterior.head(2)["hypothesis"])
            truth = spec["truth"]
            design = design_rescue_experiments(
                {
                    "posterior": posterior.to_dict(orient="records"),
                    "claim_ceiling": adjudication["claim_ceiling"],
                },
                experiments,
                strict=True,
            )
            best = str(design["next_best_experiment"]["experiment_id"])
            truth_post = float(posterior[posterior["hypothesis"] == truth].iloc[0]["posterior"])
            brier = _brier_score(posterior, truth)
            naive_truth = _fusion_posterior(
                adjudication["fusion_comparison"], "naive", truth
            )
            aware_truth = _fusion_posterior(
                adjudication["fusion_comparison"], "dependency_aware", truth
            )
            rows.append(
                {
                    "scenario": scenario_name,
                    "replicate": replicate,
                    "truth_hypothesis": truth,
                    "truth_best_experiment": spec["best_experiment"],
                    "truth_claim_ceiling": spec["claim_ceiling"],
                    "top_hypothesis": top,
                    "truth_in_top2": truth in top2,
                    "truth_posterior": truth_post,
                    "brier_score": brier,
                    "claim_level_numeric": adjudication["claim_ceiling"]["current_level_numeric"],
                    "false_claim_upgrade": float(adjudication["claim_ceiling"]["current_level_numeric"])
                    > float(spec["claim_ceiling"]),
                    "recommended_experiment": best,
                    "best_experiment_recovered": best == spec["best_experiment"],
                    "falsifier_recovered": _best_experiment_falsifies(
                        design["experiment_contrast_matrix"], best, truth
                    ),
                    "aware_truth_posterior": aware_truth,
                    "naive_truth_posterior": naive_truth,
                    "naive_overconfidence_delta": naive_truth - aware_truth,
                }
            )
    case_df = pd.DataFrame(rows)
    metrics = _synthetic_metrics(case_df)
    naive = _naive_dependency_summary(case_df)
    return {
        "synthetic_case_results": case_df,
        "synthetic_metrics": metrics,
        "naive_dependency_summary": naive,
    }


def _fusion_posterior(df: pd.DataFrame, model: str, hypothesis: str) -> float:
    row = df[(df["fusion_model"] == model) & (df["hypothesis"] == hypothesis)]
    return float(row.iloc[0]["posterior"]) if not row.empty else 0.0


def _brier_score(posterior: pd.DataFrame, truth: str) -> float:
    total = 0.0
    for row in posterior.to_dict(orient="records"):
        expected = 1.0 if row["hypothesis"] == truth else 0.0
        total += (float(row["posterior"]) - expected) ** 2
    return total / max(len(posterior), 1)


def _best_experiment_falsifies(contrast: pd.DataFrame, experiment_id: str, truth: str) -> bool:
    col = f"falsifies_{truth}"
    row = contrast[contrast["experiment_id"] == experiment_id]
    return bool(not row.empty and col in row.columns and row.iloc[0][col])


def _synthetic_metrics(case_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "n_cases": int(len(case_df)),
                "top1_hypothesis_recovery": float(
                    (case_df["top_hypothesis"] == case_df["truth_hypothesis"]).mean()
                ),
                "top2_hypothesis_coverage": float(case_df["truth_in_top2"].mean()),
                "mean_truth_posterior": float(case_df["truth_posterior"].mean()),
                "mean_brier_score": float(case_df["brier_score"].mean()),
                "false_claim_upgrade_rate": float(case_df["false_claim_upgrade"].mean()),
                "best_experiment_recovery": float(case_df["best_experiment_recovered"].mean()),
                "falsifier_recovery": float(case_df["falsifier_recovered"].mean()),
                "mean_naive_overconfidence_delta": float(
                    case_df["naive_overconfidence_delta"].mean()
                ),
            }
        ]
    )


def _naive_dependency_summary(case_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, group in case_df.groupby("scenario"):
        rows.append(
            {
                "scenario": scenario,
                "aware_truth_posterior_mean": float(group["aware_truth_posterior"].mean()),
                "naive_truth_posterior_mean": float(group["naive_truth_posterior"].mean()),
                "naive_overconfidence_delta_mean": float(
                    group["naive_overconfidence_delta"].mean()
                ),
                "dependency_aware_more_conservative": bool(
                    group["naive_overconfidence_delta"].mean() > 0
                ),
            }
        )
    return pd.DataFrame(rows)


def run_retrospective_evidence_hiding_benchmark(
    evidence_input: Mapping[str, Any] | str | Path,
    hypotheses_input: Mapping[str, Any] | str | Path,
    experiments_input: Mapping[str, Any] | str | Path,
    *,
    event: str | None = None,
    hide_families: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Hide each evidence family, run adjudication/design, then reveal it."""

    evidence = _load_if_path(evidence_input)
    hypotheses = _load_if_path(hypotheses_input)
    experiments = _load_if_path(experiments_input)
    families = hide_families or sorted({row["evidence_family"] for row in evidence["evidence"]})
    full = adjudicate_mechanism(evidence, hypotheses, event=event, strict=True)
    full_design = design_rescue_experiments(
        {"posterior": full["posterior"].to_dict(orient="records"), "claim_ceiling": full["claim_ceiling"]},
        experiments,
        strict=True,
    )
    full_leading = str(full["posterior"].iloc[0]["hypothesis"])
    rows = []
    for family in families:
        hidden_evidence = {
            **evidence,
            "evidence": [row for row in evidence["evidence"] if row["evidence_family"] != family],
        }
        if not hidden_evidence["evidence"]:
            continue
        before = adjudicate_mechanism(hidden_evidence, hypotheses, event=event, strict=True)
        before_design = design_rescue_experiments(
            {
                "posterior": before["posterior"].to_dict(orient="records"),
                "claim_ceiling": before["claim_ceiling"],
            },
            experiments,
            strict=True,
        )
        before_full = _posterior_value(before["posterior"], full_leading)
        after_full = _posterior_value(full["posterior"], full_leading)
        rows.append(
            {
                "hidden_family": family,
                "before_leading_hypothesis": str(before["posterior"].iloc[0]["hypothesis"]),
                "after_leading_hypothesis": full_leading,
                "full_leading_posterior_before_reveal": before_full,
                "full_leading_posterior_after_reveal": after_full,
                "posterior_delta_after_reveal": after_full - before_full,
                "claim_ceiling_before_reveal": before["claim_ceiling"]["current_level"],
                "claim_ceiling_after_reveal": full["claim_ceiling"]["current_level"],
                "recommended_experiment_before_reveal": before_design["next_best_experiment"][
                    "experiment_id"
                ],
                "recommended_experiment_after_reveal": full_design["next_best_experiment"][
                    "experiment_id"
                ],
                "hidden_evidence_family_revealed": family,
            }
        )
    return pd.DataFrame(rows)


def _posterior_value(posterior: pd.DataFrame, hypothesis: str) -> float:
    row = posterior[posterior["hypothesis"] == hypothesis]
    return float(row.iloc[0]["posterior"]) if not row.empty else 0.0


def run_negative_control_benchmark(
    *,
    n_controls: int = 6,
    random_seed: int = 20260513,
    promotion_threshold: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """Run negative-control evidence scenarios and estimate false promotion rate."""

    hypotheses = make_benchmark_hypotheses()
    experiments = make_benchmark_experiments()
    rng = np.random.default_rng(random_seed)
    controls = [
        "shuffled_mediator",
        "wrong_lineage_external_perturbation",
        "random_TF_KD",
        "non_erythroid_module",
        "permuted_day_labels",
        "family_label_shuffled",
    ][:n_controls]
    rows = []
    for i, control in enumerate(controls):
        evidence = _negative_control_evidence(control, rng, i)
        adjudication = adjudicate_mechanism(
            evidence,
            hypotheses,
            event="negative_control_event",
            strict=True,
            compare_naive=True,
        )
        design = design_rescue_experiments(
            {
                "posterior": adjudication["posterior"].to_dict(orient="records"),
                "claim_ceiling": adjudication["claim_ceiling"],
            },
            experiments,
            strict=True,
        )
        mech_post = max(
            _posterior_value(adjudication["posterior"], hyp) for hyp in MECHANISM_HYPOTHESES
        )
        promoted = mech_post > promotion_threshold or float(
            adjudication["claim_ceiling"]["current_level_numeric"]
        ) > 2.0
        rows.append(
            {
                "negative_control": control,
                "leading_hypothesis": str(adjudication["posterior"].iloc[0]["hypothesis"]),
                "max_mechanism_posterior": mech_post,
                "claim_level_numeric": adjudication["claim_ceiling"]["current_level_numeric"],
                "recommended_experiment": design["next_best_experiment"]["experiment_id"],
                "false_mechanism_promotion": promoted,
                "warning": "false mechanism promotion" if promoted else "passed negative control",
            }
        )
    control_df = pd.DataFrame(rows)
    metrics = pd.DataFrame(
        [
            {
                "n_negative_controls": int(len(control_df)),
                "false_mechanism_promotion_rate": float(
                    control_df["false_mechanism_promotion"].mean()
                ),
                "max_mechanism_posterior_mean": float(
                    control_df["max_mechanism_posterior"].mean()
                ),
            }
        ]
    )
    return {"negative_control_results": control_df, "negative_control_metrics": metrics}


def _negative_control_evidence(control: str, rng: np.random.Generator, index: int) -> dict[str, Any]:
    families = [
        "negative_mediator_controls",
        "external_GATA1_KD",
        "rescue_prediction",
        "day_stratified_timing",
        "family_block_robustness",
    ]
    artifact = str(rng.choice(["H0_noise_batch", "H1_state_composition", "H2_proliferation_confounded"]))
    rows = []
    for j, family in enumerate(families[:3]):
        rows.append(
            _evidence_row(
                f"{control}_E{j + 1}",
                family,
                f"{control}_dependency_{j + 1}",
                artifact,
                rng,
                weight=0.25 + 0.10 * index,
            )
        )
        rows[-1]["target_event"] = "negative_control_event"
        rows[-1]["assumptions"] = [f"{control} should not support a mechanism claim"]
        rows[-1]["failure_modes"] = ["negative-control evidence accidentally aligned with mechanism"]
    return {"schema_version": "0.1", "evidence": rows}


def write_benchmark_outputs(results: Mapping[str, pd.DataFrame], outdir: str | Path) -> dict[str, str]:
    """Write benchmark result tables to disk."""

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, table in results.items():
        if isinstance(table, pd.DataFrame):
            path = out / f"{name}.csv"
            table.to_csv(path, index=False)
            paths[name] = str(path)
    summary = {
        name: table.to_dict(orient="records")
        for name, table in results.items()
        if isinstance(table, pd.DataFrame)
    }
    summary_path = out / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["benchmark_summary"] = str(summary_path)
    return paths


def _load_if_path(value: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(value, (str, Path)):
        return load_ted_mad_yaml(value)
    return value
