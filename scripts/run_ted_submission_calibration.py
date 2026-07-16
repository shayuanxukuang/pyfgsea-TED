from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


SEED = 20260715
PACKET_CLASSES = [
    "activation",
    "suppression",
    "developmental_delay",
    "true_loss",
    "fate_redirection",
    "composition_artifact",
    "stress_dominated",
    "not_identifiable",
    "outcome_supported",
    "reversal_supported",
]

BIOLOGICAL_EVENT_MODES = {
    "activation": "activation",
    "suppression": "suppression",
    "developmental_delay": "developmental_delay",
    "true_loss": "true_loss",
    "fate_redirection": "fate_redirection",
    # These two packets encode an orthogonal provenance tag in addition to the
    # underlying biological event.  They are retained for backwards-compatible
    # packet-class scoring, not presented as additional biological modes.
    "outcome_supported": "activation",
    "reversal_supported": "suppression",
}


def packet_class_components(packet_class: str) -> dict[str, object]:
    """Factor a legacy packet class into the independent TED target domains."""
    return {
        "biological_event_mode": BIOLOGICAL_EVENT_MODES.get(packet_class, "not_assigned"),
        "artifact_class": (
            "composition"
            if packet_class == "composition_artifact"
            else "stress"
            if packet_class == "stress_dominated"
            else "none"
        ),
        "identifiability_status": "not_identifiable" if packet_class == "not_identifiable" else "identifiable",
        "v_outcome": packet_class == "outcome_supported",
        "v_reversal": packet_class == "reversal_supported",
        "v_rescue": False,
        "v_replication": False,
    }


def controlled_packet_class_factorization() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for packet_class in PACKET_CLASSES:
        rows.append({"packet_class": packet_class, **packet_class_components(packet_class)})
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class RulePerturbationProfile:
    profile_id: str
    identifiability_threshold: float
    artifact_threshold: float
    outcome_threshold: float
    packet_class_margin: float
    tier_bias: float
    feature_jitter_sd: float = 0.025


RULE_PERTURBATION_PROFILES = [
    RulePerturbationProfile("rule_perturbation_01", 0.58, 0.62, 0.45, 0.10, -0.10),
    RulePerturbationProfile("rule_perturbation_02", 0.52, 0.70, 0.50, 0.08, 0.05),
    RulePerturbationProfile("rule_perturbation_03", 0.62, 0.58, 0.40, 0.12, -0.15),
    RulePerturbationProfile("rule_perturbation_04", 0.55, 0.65, 0.55, 0.15, 0.00),
    RulePerturbationProfile("rule_perturbation_05", 0.68, 0.55, 0.60, 0.10, -0.25),
]


def clip(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def make_packets(packets_per_class: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    base = {
        "activation": (0.85, 0.00, 0.85, 0.05, 0.08, 0.08, 0.10, 0.05),
        "suppression": (-0.85, 0.00, 0.20, 0.05, 0.08, 0.08, 0.10, 0.05),
        "developmental_delay": (-0.20, 0.34, 0.72, 0.08, 0.08, 0.08, 0.10, 0.05),
        "true_loss": (-0.82, 0.02, 0.12, 0.08, 0.08, 0.08, 0.10, 0.05),
        "fate_redirection": (-0.58, 0.08, 0.25, 0.72, 0.08, 0.12, 0.10, 0.05),
        "composition_artifact": (0.45, 0.02, 0.45, 0.12, 0.88, 0.10, 0.05, 0.04),
        "stress_dominated": (0.48, 0.04, 0.45, 0.08, 0.10, 0.88, 0.05, 0.04),
        "not_identifiable": (0.05, 0.05, 0.48, 0.10, 0.30, 0.30, 0.05, 0.04),
        "outcome_supported": (0.62, 0.03, 0.78, 0.08, 0.08, 0.10, 0.72, 0.05),
        "reversal_supported": (-0.55, 0.02, 0.76, 0.08, 0.08, 0.10, 0.18, 0.76),
    }
    truth_tier = {
        "composition_artifact": 1.0,
        "stress_dominated": 2.0,
        "not_identifiable": 1.0,
        "outcome_supported": 3.5,
        "reversal_supported": 3.5,
    }
    rows: list[dict[str, object]] = []
    for packet_class_index, packet_class in enumerate(PACKET_CLASSES):
        for replicate in range(packets_per_class):
            effect, peak_shift, recovery, alt_gain, composition, stress_score, outcome, reversal = base[packet_class]
            shifted = replicate >= max(1, int(packets_per_class * 0.75))
            noise = 0.13 if shifted else 0.07
            block_stability = clip(rng.normal(0.84 if packet_class not in {"composition_artifact", "not_identifiable"} else 0.48, noise))
            matched_balance = clip(rng.normal(0.82 if packet_class not in {"composition_artifact", "stress_dominated"} else 0.38, noise))
            negative_margin = clip(rng.normal(0.75 if packet_class not in {"composition_artifact", "stress_dominated"} else 0.25, noise))
            missing_fraction = clip(rng.normal(0.18 if packet_class != "not_identifiable" else 0.72, noise))
            components = packet_class_components(packet_class)
            rows.append(
                {
                    "packet_id": f"P{packet_class_index + 1:02d}_{replicate + 1:03d}",
                    "synthetic_source_group": f"heldout_group_{replicate % 4 + 1}",
                    "distribution_shift": shifted,
                    "n_blocks": int(rng.choice([2, 3, 5, 10], p=[0.12, 0.24, 0.38, 0.26])),
                    "curve_effect": float(effect + rng.normal(0, noise)),
                    "peak_shift": float(peak_shift + rng.normal(0, noise)),
                    "terminal_recovery": clip(recovery + rng.normal(0, noise)),
                    "alternative_fate_gain": clip(alt_gain + rng.normal(0, noise)),
                    "composition_score": clip(composition + rng.normal(0, noise)),
                    "stress_score": clip(stress_score + rng.normal(0, noise)),
                    "outcome_alignment": clip(outcome + rng.normal(0, noise)),
                    "reversal_score": clip(reversal + rng.normal(0, noise)),
                    "block_direction_stability": block_stability,
                    "matched_state_balance": matched_balance,
                    "negative_control_margin": negative_margin,
                    "missing_evidence_fraction": missing_fraction,
                    "truth_event_present": packet_class not in {"composition_artifact", "not_identifiable"},
                    "truth_artifact_sensitive": packet_class in {"composition_artifact", "stress_dominated"},
                    "truth_packet_class": packet_class,
                    "truth_biological_event_mode": components["biological_event_mode"],
                    "truth_artifact_class": components["artifact_class"],
                    "truth_identifiability_status": components["identifiability_status"],
                    "truth_v_outcome": components["v_outcome"],
                    "truth_v_reversal": components["v_reversal"],
                    "truth_v_rescue": components["v_rescue"],
                    "truth_v_replication": components["v_replication"],
                    "truth_evidence_tier": truth_tier.get(packet_class, 3.0),
                }
            )
    return pd.DataFrame(rows)


def identifiability(row: pd.Series) -> float:
    blocks = min(float(row["n_blocks"]) / 5.0, 1.0)
    return clip(
        0.24 * float(row["block_direction_stability"])
        + 0.20 * float(row["matched_state_balance"])
        + 0.20 * float(row["negative_control_margin"])
        + 0.16 * blocks
        + 0.20 * (1.0 - float(row["missing_evidence_fraction"]))
    )


def classify_packet(row: pd.Series, profile: RulePerturbationProfile | None = None) -> tuple[str, str]:
    ident_threshold = profile.identifiability_threshold if profile else 0.58
    artifact_threshold = profile.artifact_threshold if profile else 0.62
    outcome_threshold = profile.outcome_threshold if profile else 0.48
    margin = profile.packet_class_margin if profile else 0.10
    ident = identifiability(row)
    if ident < ident_threshold:
        return "not_identifiable", "not_identifiable"
    if float(row["composition_score"]) >= artifact_threshold:
        return "composition_artifact", "composition_artifact;not_identifiable"
    if float(row["stress_score"]) >= artifact_threshold:
        return "stress_dominated", "stress_dominated;composition_artifact"
    if float(row["outcome_alignment"]) >= outcome_threshold + margin:
        return "outcome_supported", "outcome_supported;activation"
    if float(row["reversal_score"]) >= outcome_threshold + margin:
        return "reversal_supported", "reversal_supported;suppression"
    if float(row["alternative_fate_gain"]) >= 0.55:
        return "fate_redirection", "fate_redirection;true_loss"
    if float(row["peak_shift"]) >= 0.20 and float(row["terminal_recovery"]) >= 0.45:
        return "developmental_delay", "developmental_delay;true_loss"
    if float(row["curve_effect"]) <= -0.35 and float(row["terminal_recovery"]) < 0.40:
        return "true_loss", "true_loss;suppression"
    if float(row["curve_effect"]) <= -0.25:
        return "suppression", "suppression;true_loss"
    if float(row["curve_effect"]) >= 0.25:
        return "activation", "activation;outcome_supported"
    return "not_identifiable", "not_identifiable"


def tier_for(row: pd.Series, predicted_mode: str, bias: float = 0.0) -> float:
    if predicted_mode in {"not_identifiable", "composition_artifact"}:
        return 1.0
    tier = 2.0
    if float(row["block_direction_stability"]) >= 0.75 and int(row["n_blocks"]) >= 3:
        tier = 3.0
    if predicted_mode in {"outcome_supported", "reversal_supported"} and float(row["outcome_alignment"] if predicted_mode == "outcome_supported" else row["reversal_score"]) >= 0.55:
        tier = 3.5
    return float(np.clip(tier + bias, 1.0, 3.5) * 2 // 1 / 2)


def annotate_packets(packets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ted_rows = []
    profile_rows = []
    for _, row in packets.iterrows():
        packet_class, ambiguity = classify_packet(row)
        ident = identifiability(row)
        components = packet_class_components(packet_class)
        ted_rows.append(
            {
                "packet_id": row["packet_id"],
                "ted_event_present": packet_class not in {"not_identifiable", "composition_artifact"},
                "ted_packet_class": packet_class,
                "ted_biological_event_mode": components["biological_event_mode"],
                "ted_artifact_class": components["artifact_class"],
                "ted_identifiability_status": components["identifiability_status"],
                "ted_v_outcome": components["v_outcome"],
                "ted_v_reversal": components["v_reversal"],
                "ted_v_rescue": components["v_rescue"],
                "ted_v_replication": components["v_replication"],
                "ted_artifact_sensitive": packet_class in {"composition_artifact", "stress_dominated"},
                "ted_evidence_tier": tier_for(row, packet_class),
                "ted_identifiability": ident,
                "ted_packet_class_compatibility_set": ambiguity if ident < 0.78 or ";" in ambiguity else packet_class,
            }
        )
        for profile_index, profile in enumerate(RULE_PERTURBATION_PROFILES):
            local = row.copy()
            rng = np.random.default_rng(SEED + profile_index * 10_000 + int(str(row["packet_id"])[1:3]) * 100 + int(str(row["packet_id"])[4:]))
            for field in [
                "curve_effect",
                "peak_shift",
                "terminal_recovery",
                "alternative_fate_gain",
                "composition_score",
                "stress_score",
                "outcome_alignment",
                "reversal_score",
                "block_direction_stability",
                "matched_state_balance",
                "negative_control_margin",
            ]:
                local[field] = float(local[field]) + rng.normal(0, profile.feature_jitter_sd)
            perturbed_packet_class, perturbed_ambiguity = classify_packet(local, profile)
            profile_rows.append(
                {
                    "packet_id": row["packet_id"],
                    "profile_id": profile.profile_id,
                    "perturbed_event_present": perturbed_packet_class not in {"not_identifiable", "composition_artifact"},
                    "perturbed_packet_class": perturbed_packet_class,
                    "perturbed_artifact_sensitive": perturbed_packet_class in {"composition_artifact", "stress_dominated"},
                    "perturbed_evidence_tier": tier_for(local, perturbed_packet_class, profile.tier_bias),
                    "perturbed_packet_class_compatibility_set": perturbed_ambiguity,
                }
            )
    return pd.DataFrame(ted_rows), pd.DataFrame(profile_rows)


def rule_perturbation_profile_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "profile_id": profile.profile_id,
                "identifiability_threshold": profile.identifiability_threshold,
                "artifact_threshold": profile.artifact_threshold,
                "outcome_threshold": profile.outcome_threshold,
                "packet_class_margin": profile.packet_class_margin,
                "tier_bias": profile.tier_bias,
                "feature_jitter_sd": profile.feature_jitter_sd,
                "seed_policy": "SEED + profile_index*10000 + packet_class_index*100 + replicate_index",
            }
            for profile in RULE_PERTURBATION_PROFILES
        ]
    )


def summarize_rule_perturbations(ted: pd.DataFrame, calls: pd.DataFrame) -> pd.DataFrame:
    joined = calls.merge(ted, on="packet_id", validate="many_to_one")
    rows: list[dict[str, object]] = []
    for profile_id, group in joined.groupby("profile_id", sort=True):
        rows.append(
            {
                "profile_id": profile_id,
                "n_packets": len(group),
                "packet_class_change_rate": float((group["perturbed_packet_class"] != group["ted_packet_class"]).mean()),
                "event_present_change_rate": float((group["perturbed_event_present"] != group["ted_event_present"]).mean()),
                "artifact_flag_change_rate": float((group["perturbed_artifact_sensitive"] != group["ted_artifact_sensitive"]).mean()),
                "evidence_tier_mae_vs_reference": float(
                    np.mean(np.abs(group["perturbed_evidence_tier"] - group["ted_evidence_tier"]))
                ),
            }
        )
    return pd.DataFrame(rows)


def controlled_truth_metrics(
    packets: pd.DataFrame, ted: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    joined = packets.merge(ted, on="packet_id")
    labels = PACKET_CLASSES
    metrics = pd.DataFrame(
        [
            {"metric": "ted_vs_truth_packet_class_macro_f1", "value": f1_score(joined["truth_packet_class"], joined["ted_packet_class"], labels=labels, average="macro", zero_division=0)},
            {"metric": "ted_vs_truth_packet_class_balanced_accuracy", "value": balanced_accuracy_score(joined["truth_packet_class"], joined["ted_packet_class"])},
            {
                "metric": "tier_weighted_kappa_vs_truth",
                "value": cohen_kappa_score(
                    (joined["truth_evidence_tier"] * 2).round().astype(int),
                    (joined["ted_evidence_tier"] * 2).round().astype(int),
                    weights="quadratic",
                ),
            },
            {"metric": "tier_mean_absolute_error_vs_truth", "value": float(np.mean(np.abs(joined["truth_evidence_tier"] - joined["ted_evidence_tier"])))},
            {"metric": "false_upgrade_rate", "value": float(np.mean(joined["ted_evidence_tier"] > joined["truth_evidence_tier"]))},
            {"metric": "false_downgrade_rate", "value": float(np.mean(joined["ted_evidence_tier"] < joined["truth_evidence_tier"]))},
        ]
    )
    confusion = pd.DataFrame(
        confusion_matrix(joined["truth_packet_class"], joined["ted_packet_class"], labels=labels),
        index=labels,
        columns=labels,
    ).rename_axis("truth_packet_class").reset_index()
    coverage_rows = []
    for threshold in np.linspace(0.45, 0.90, 10):
        selected = joined[joined["ted_identifiability"] >= threshold]
        coverage_rows.append(
            {
                "identifiability_threshold": threshold,
                "selective_coverage": len(selected) / len(joined),
                "packet_class_error": float((selected["truth_packet_class"] != selected["ted_packet_class"]).mean()) if len(selected) else np.nan,
                "tier_mae": float(np.mean(np.abs(selected["truth_evidence_tier"] - selected["ted_evidence_tier"]))) if len(selected) else np.nan,
            }
        )
    return metrics, confusion, pd.DataFrame(coverage_rows)


def bh(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.clip(ranked, 0, 1)
    return out


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(np.nanmean(arr))
    se = float(np.nanstd(arr, ddof=1) / np.sqrt(np.isfinite(arr).sum()))
    return mean, max(0.0, mean - 1.96 * se), min(1.0, mean + 1.96 * se)


def event_fdr_calibration(
    n_replicates: int,
    q01_replicates: int,
    q05_replicates: int,
    relative_tolerance: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 11)
    rows = []
    for n_pathways in [20, 50, 100]:
        n_alt = max(1, int(0.20 * n_pathways))
        truth = np.zeros(n_pathways, dtype=bool)
        truth[:n_alt] = True
        for overlap in [0.0, 0.35, 0.70]:
            family = np.arange(n_pathways) // 10
            for target_q in [0.01, 0.05, 0.10, 0.20]:
                local_replicates = (
                    q01_replicates
                    if target_q == 0.01
                    else q05_replicates
                    if target_q == 0.05
                    else n_replicates
                )
                fdp_values: list[float] = []
                power_values: list[float] = []
                for _ in range(local_replicates):
                    common = rng.normal(size=family.max() + 1)
                    z = np.sqrt(overlap) * common[family] + np.sqrt(1 - overlap) * rng.normal(size=n_pathways)
                    z[truth] += 2.35
                    p = 2 * stats.norm.sf(np.abs(z))
                    called = bh(p) <= target_q
                    fdp_values.append(float((called & ~truth).sum() / max(called.sum(), 1)))
                    power_values.append(float((called & truth).sum() / truth.sum()))
                fdp, fdp_low, fdp_high = mean_ci(fdp_values)
                power, power_low, power_high = mean_ci(power_values)
                rows.append(
                    {
                        "target_q": target_q,
                        "n_pathways": n_pathways,
                        "pathway_overlap_rho": overlap,
                        "n_replicates": local_replicates,
                        "empirical_fdp": fdp,
                        "empirical_fdp_ci95_low": fdp_low,
                        "empirical_fdp_ci95_high": fdp_high,
                        "power": power,
                        "power_ci95_low": power_low,
                        "power_ci95_high": power_high,
                        "relative_tolerance": relative_tolerance,
                        "calibration_upper_limit": target_q * (1.0 + relative_tolerance),
                        "calibration_pass": fdp_high <= target_q * (1.0 + relative_tolerance),
                    }
                )
    return pd.DataFrame(rows)


CONFOUNDING_SCENARIOS = {
    "donor_batch_time_confounding": (0.85, 0.35),
    "condition_dependent_library_size": (0.65, 0.45),
    "cell_cycle_enrichment": (0.60, 0.48),
    "stress_response_spike_in": (0.90, 0.35),
    "lineage_frequency_shift": (0.75, 0.40),
    "composition_only_change": (0.85, 0.30),
    "rare_state_depletion": (0.70, 0.35),
    "missing_intermediate_timepoints": (0.50, 0.50),
    "pseudotime_reversal": (0.80, 0.38),
    "matched_state_overlap_decline": (0.65, 0.42),
}


def confounded_null_calibration(n_replicates: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 23)
    rows = []
    for scenario, (shift, base_ident) in CONFOUNDING_SCENARIOS.items():
        for n_blocks in [2, 3, 5, 10]:
            false_raw = []
            false_gated = []
            downgrade = []
            for _ in range(n_replicates):
                z = rng.normal(size=60)
                affected = rng.choice(60, size=12, replace=False)
                z[affected] += shift * np.sqrt(5 / max(n_blocks, 1))
                q = bh(2 * stats.norm.sf(np.abs(z)))
                raw = q <= 0.05
                ident = clip(base_ident + 0.06 * np.log2(n_blocks) + rng.normal(0, 0.04))
                gated = raw & (ident >= 0.60)
                false_raw.append(float(raw.mean()))
                false_gated.append(float(gated.mean()))
                downgrade.append(float(raw.any() and not gated.any()))
            rows.append(
                {
                    "scenario": scenario,
                    "n_blocks": n_blocks,
                    "n_replicates": n_replicates,
                    "raw_false_event_rate": float(np.mean(false_raw)),
                    "gated_false_escalation_rate": float(np.mean(false_gated)),
                    "claim_downgrade_fraction": float(np.mean(downgrade)),
                    "fail_closed_pass": float(np.mean(false_gated)) <= float(np.mean(false_raw)),
                }
            )
    return pd.DataFrame(rows)


def confounded_signal_calibration(n_replicates: int) -> pd.DataFrame:
    """Joint true-event/confounding sensitivity with explicit demotion accounting.

    This is a post-review controlled sensitivity analysis.  Twelve of 60
    pathways carry a true event, a disjoint set of 12 null pathways receives a
    scenario-specific confounding shift, and the fail-closed gate acts on
    pathway-level identifiability and control-margin diagnostics.
    """
    rng = np.random.default_rng(SEED + 29)
    rows: list[dict[str, object]] = []
    n_pathways = 60
    truth = np.zeros(n_pathways, dtype=bool)
    truth[:12] = True
    confounded_null = np.zeros(n_pathways, dtype=bool)
    confounded_null[12:24] = True
    for scenario, (shift, base_ident) in CONFOUNDING_SCENARIOS.items():
        for n_blocks in [2, 3, 5, 10]:
            raw_power_values: list[float] = []
            gated_power_values: list[float] = []
            false_demotion_values: list[float] = []
            conditional_demotion_values: list[float] = []
            gated_false_promotion_values: list[float] = []
            for _ in range(n_replicates):
                z = rng.normal(size=n_pathways)
                z[truth] += 2.35
                z[confounded_null] += shift * np.sqrt(5 / max(n_blocks, 1))
                q_value = bh(2 * stats.norm.sf(np.abs(z)))
                raw = q_value <= 0.05
                ident = np.clip(
                    base_ident
                    + 0.06 * np.log2(n_blocks)
                    + rng.normal(0, 0.08, size=n_pathways),
                    0,
                    1,
                )
                # True events receive an estimability increment, whereas the
                # deliberately confounded nulls retain the scenario-specific
                # low-identifiability distribution.
                ident[truth] = np.clip(ident[truth] + 0.22, 0, 1)
                control_margin = rng.normal(0.18, 0.16, size=n_pathways)
                control_margin[truth] += 0.34
                control_margin[confounded_null] -= 0.22
                gated = raw & (ident >= 0.60) & (control_margin > 0)
                raw_true = raw & truth
                gated_true = gated & truth
                demoted_true = raw_true & ~gated_true
                raw_power_values.append(float(raw_true.sum() / truth.sum()))
                gated_power_values.append(float(gated_true.sum() / truth.sum()))
                false_demotion_values.append(float(demoted_true.sum() / truth.sum()))
                conditional_demotion_values.append(float(demoted_true.sum() / max(raw_true.sum(), 1)))
                gated_false_promotion_values.append(float((gated & ~truth).sum() / (~truth).sum()))
            rows.append(
                {
                    "scenario": scenario,
                    "n_blocks": n_blocks,
                    "n_replicates": n_replicates,
                    "target_q": 0.05,
                    "n_true_events": int(truth.sum()),
                    "n_confounded_nulls": int(confounded_null.sum()),
                    "raw_event_power": float(np.mean(raw_power_values)),
                    "gated_event_power": float(np.mean(gated_power_values)),
                    "false_demotion_rate_all_true": float(np.mean(false_demotion_values)),
                    "false_demotion_rate_given_raw_detection": float(np.mean(conditional_demotion_values)),
                    "gated_false_promotion_rate": float(np.mean(gated_false_promotion_values)),
                }
            )
    return pd.DataFrame(rows)


def ambiguity_calibration(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for shifted, sub in joined.groupby("distribution_shift"):
        covered = []
        sizes = []
        for row in sub.itertuples(index=False):
            event_set = set(str(row.ted_packet_class_compatibility_set).split(";"))
            covered.append(row.truth_packet_class in event_set)
            sizes.append(len(event_set))
        rows.append(
            {
                "distribution_shift": bool(shifted),
                "n_packets": len(sub),
                "empirical_coverage": float(np.mean(covered)),
                "mean_prediction_set_size": float(np.mean(sizes)),
                "singleton_fraction": float(np.mean(np.asarray(sizes) == 1)),
                "terminology": "packet-class compatibility set; not conformal",
            }
        )
    return pd.DataFrame(rows)


def failure_modes(packets: pd.DataFrame, confounded: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "failure_mode": scenario,
                "detectability": "directly simulated",
                "required_behavior": "downgrade identifiability and prevent evidence-tier escalation",
                "observed_gated_false_escalation_max": float(sub["gated_false_escalation_rate"].max()),
                "applicability_condition": "Interpret only at dataset/contrast or biological-block level; never use cells as independent replicates.",
            }
            for scenario, sub in confounded.groupby("scenario")
        ]
        + [
            {
                "failure_mode": "matched_functional_rescue_absent",
                "detectability": "protocol gate",
                "required_behavior": "cap claim at Level 3.5",
                "observed_gated_false_escalation_max": 0.0,
                "applicability_condition": "GSE271399/T21/GATA1 remains a computational mechanism candidate.",
            },
            {
                "failure_mode": "small_block_count",
                "detectability": "n_blocks field",
                "required_behavior": "report limited identifiability",
                "observed_gated_false_escalation_max": float(confounded[confounded["n_blocks"].eq(2)]["gated_false_escalation_rate"].max()),
                "applicability_condition": "Level 3 block-robust language requires at least three usable blocks.",
            },
        ]
    )


def write_report(outdir: Path, tables: dict[str, pd.DataFrame], args: argparse.Namespace) -> None:
    metrics = tables["controlled_truth_metrics"].set_index("metric")["value"]
    fdr = tables["event_fdr_calibration"]
    ambiguity = tables["ambiguity_calibration"]
    report = [
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: run + validate",
        "- Origin Date: 2026-07-15",
        "- Verification Status: ANALYZED",
        "- Version Label: ted_submission_calibration_v1",
        "",
        "# TED submission calibration report",
        "",
        f"- Packets: {len(tables['packets_internal'])}; five rule-perturbation sensitivity profiles.",
        f"- TED vs controlled synthetic packet-class truth macro-F1: {metrics['ted_vs_truth_packet_class_macro_f1']:.3f}.",
        f"- Evidence-tier false-upgrade rate: {metrics['false_upgrade_rate']:.3f}.",
        "- Event-FDR is reported by nominal q with empirical FDP, 95% Monte Carlo intervals and the worst configuration; pass counts are secondary.",
        f"- Relative calibration criterion: upper 95% bound <= (1 + {args.fdr_relative_tolerance:.2f}) x q.",
        f"- Packet-class compatibility-set coverage under shift: {ambiguity.loc[ambiguity['distribution_shift'], 'empirical_coverage'].iloc[0]:.3f}.",
        "",
        "This is an internal benchmark with controlled synthetic truth. The rule-perturbation sensitivity profiles are supplementary sensitivity analyses only, not an independent truth source or external validation.",
        "The ten legacy labels are controlled packet classes spanning biological mode, artifact, identifiability and V-provenance domains; they are not ten biological event modes.",
        "The packet-class compatibility sets are rule-defined sets, not conformal prediction sets.",
        "No result in this package upgrades GSE271399/T21/GATA1 above Level 3.5.",
        "",
        f"Simulation settings: packets_per_class={args.packets_per_class}, FDR replicates={args.fdr_q01_replicates} (q=0.01), {args.fdr_q05_replicates} (q=0.05), or {args.fdr_replicates} (q=0.10/0.20); confounded-null replicates={args.null_replicates}, confounded-signal replicates={args.confounded_signal_replicates}, seed={SEED}.",
    ]
    (outdir / "ted_submission_calibration_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled-truth, rule-sensitivity, event-FDR, evidence-tier, and packet-class compatibility-set calibration")
    parser.add_argument("--outdir", type=Path, default=Path("results/ted_submission_calibration"))
    parser.add_argument("--packets-per-class", type=int, default=16)
    parser.add_argument("--fdr-replicates", type=int, default=500, help="Replicates for q=0.10 and q=0.20")
    parser.add_argument("--fdr-q01-replicates", type=int, default=10_000)
    parser.add_argument("--fdr-q05-replicates", type=int, default=5_000)
    parser.add_argument("--fdr-relative-tolerance", type=float, default=0.50)
    parser.add_argument("--null-replicates", type=int, default=500)
    parser.add_argument("--confounded-signal-replicates", type=int, default=500)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    packets = make_packets(args.packets_per_class)
    ted, sensitivity_calls = annotate_packets(packets)
    sensitivity_profiles = rule_perturbation_profile_registry()
    sensitivity_summary = summarize_rule_perturbations(ted, sensitivity_calls)
    metrics, confusion, selective = controlled_truth_metrics(packets, ted)
    joined = packets.merge(ted, on="packet_id")
    fdr = event_fdr_calibration(
        args.fdr_replicates,
        args.fdr_q01_replicates,
        args.fdr_q05_replicates,
        args.fdr_relative_tolerance,
    )
    confounded = confounded_null_calibration(args.null_replicates)
    confounded_signal = confounded_signal_calibration(args.confounded_signal_replicates)
    ambiguity = ambiguity_calibration(joined)
    failures = failure_modes(packets, confounded)
    public_evidence = packets.drop(columns=[column for column in packets if column.startswith("truth_") or column == "synthetic_source_group"])
    truth_key = packets[[column for column in packets.columns if column == "packet_id" or column.startswith("truth_")]]

    tables = {
        "controlled_packet_features": public_evidence,
        "rule_perturbation_sensitivity_profiles": sensitivity_profiles,
        "rule_perturbation_sensitivity_calls": sensitivity_calls,
        "rule_perturbation_sensitivity_summary": sensitivity_summary,
        "ted_packet_predictions": ted,
        "controlled_packet_class_factorization": controlled_packet_class_factorization(),
        "controlled_truth_metrics": metrics,
        "packet_class_confusion_matrix": confusion,
        "evidence_tier_selective_coverage": selective,
        "event_fdr_calibration": fdr,
        "confounded_null_calibration": confounded,
        "confounded_signal_calibration": confounded_signal,
        "ambiguity_calibration": ambiguity,
        "failure_modes_and_applicability": failures,
        "packets_internal": packets,
    }
    written_paths: list[Path] = []
    for name, table in tables.items():
        if name == "packets_internal":
            continue
        path = args.outdir / f"{name}.tsv"
        table.to_csv(path, sep="\t", index=False)
        written_paths.append(path)
    truth_key_path = args.outdir / "controlled_truth_key.tsv"
    truth_key.to_csv(truth_key_path, sep="\t", index=False)
    written_paths.append(truth_key_path)
    write_report(args.outdir, tables, args)
    report_path = args.outdir / "ted_submission_calibration_report.md"
    written_paths.append(report_path)
    run_config_path = args.outdir / "run_config.json"
    run_config_path.write_text(
        json.dumps(vars(args) | {"seed": SEED}, default=str, indent=2), encoding="utf-8"
    )
    written_paths.append(run_config_path)
    manifest_rows = [
        {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(written_paths)
    ]
    pd.DataFrame(manifest_rows).to_csv(args.outdir / "manifest.tsv", sep="\t", index=False)
    print(f"TED submission calibration complete: {args.outdir.resolve()}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
