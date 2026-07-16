from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import yaml

from .t21_calibration_profile import (
    BINDINGS_SCHEMA_VERSION,
    load_calibration_design_profile,
    validate_calibration_design_profile,
)
from .trajectory_covariate_simulation import (
    ArrayFreedmanLanePlan,
    make_array_freedman_lane_plan,
    run_array_freedman_lane_calibration_batch,
)
from .trajectory_covariate_pseudobulk import (
    _bin_design_row,
    _make_pattern_blocks,
    _select_segment,
    _space_sizes,
)
from .trajectory_decomposition import (
    calibrate_compositional_component_arrays_batch,
    fate_response_from_masses,
    occupancy_response_from_counts,
)
from .trajectory_events import estimate_half_rise_onset, stable_event_timing
from .t21_covariate_design import (
    build_t21_canonical_donor_design,
    build_t21_sensitivity_signature_matrix,
    canonical_t21_donor_design_spec_sha256,
    covariate_matrix_sha256,
)


SCENARIOS = (
    "complete_null",
    "occupancy_only",
    "fate_only",
    "trajectory_speed_or_mapping_only",
    "covariate_condition_association",
    "chr21_dosage_only",
)

REQUIRED_BINDING_KEYS = frozenset(
    {
        "analysis_plan_sha256",
        "calibration_policy_sha256",
        "scrna_sha256",
        "donor_design_sha256",
        "fates_sha256",
        "scrna_cell_id_set_hash",
        "scrna_gene_order_hash",
        "donor_set_hash",
        "scrna_donor_set_hash",
        "trajectory_tree_digest_sha256",
        "trajectory_grid_hash",
        "pathway_universe_relative_path",
        "pathway_universe_sha256",
        "pathway_universe_logical_sha256",
        "code_commit",
        "code_dirty",
    }
)
OPTIONAL_BINDING_KEYS = frozenset({"code_patch_sha256"})
_SHA256_KEYS = REQUIRED_BINDING_KEYS - {
    "pathway_universe_relative_path",
    "code_commit",
    "code_dirty",
}
REQUIRED_BINDING_KEYS_V2 = frozenset(
    {
        "bindings_schema_version",
        "analysis_plan_sha256",
        "calibration_policy_sha256",
        "runner_spec_sha256",
        "calibration_report_schema_sha256",
        "design_profile_sha256",
        "design_profile_payload_sha256",
        "scrna_sha256",
        "donor_design_sha256",
        "fates_sha256",
        "scrna_cell_id_set_hash",
        "scrna_gene_order_hash",
        "donor_set_hash",
        "scrna_donor_set_hash",
        "expression_contract_sha256",
        "expression_implementation_source_sha256",
        "expression_csr_semantic_sha256",
        "formal_analysis_cell_set_hash",
        "formal_analysis_cell_order_hash",
        "formal_analysis_cell_count",
        "formal_gene_order_bound_support_sha256",
        "formal_support_contract_sha256",
        "formal_support_mask_sha256_uint8",
        "formal_analysis_cell_mask_sha256_uint8",
        "trajectory_tree_digest_sha256",
        "trajectory_grid_hash",
        "trajectory_primary_draw_id_sha256",
        "pathway_universe_sha256",
        "pathway_universe_logical_sha256",
        "supported_pathway_universe_logical_sha256",
        "code_commit",
        "code_dirty",
    }
)
_SHA256_KEYS_V2 = REQUIRED_BINDING_KEYS_V2 - {
    "bindings_schema_version",
    "formal_analysis_cell_count",
    "code_commit",
    "code_dirty",
}


@dataclass(frozen=True)
class CalibrationRunPlan:
    """Frozen replicate counts for one calibration phase."""

    phase: str
    complete_null_replicates: int
    scenario_replicates: int
    power_replicates_per_point: int
    development_override: bool = False


@dataclass
class CalibrationRunResult:
    """Auditable six-scenario and power simulation tables."""

    scenario_replicates: pd.DataFrame
    scenario_metrics: pd.DataFrame
    power_replicates: pd.DataFrame
    power_curve: pd.DataFrame
    onset_power_curve: pd.DataFrame
    loco_replicates: pd.DataFrame
    loco_power: pd.DataFrame
    power_metrics: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _ExactContrastPlan:
    residual_maker: np.ndarray
    residual_conditions: np.ndarray
    denominators: np.ndarray
    observed_assignment_index: int
    degrees_of_freedom: int
    n_assignments: int


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA256 digest of a local file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_power_and_phases(spec: Mapping[str, Any]) -> None:
    phases = spec.get("phase_replicates")
    if not isinstance(phases, Mapping) or set(phases) != {"smoke", "screen", "final"}:
        raise ValueError("Runner must define smoke, screen, and final phases")
    for phase, row in phases.items():
        if not isinstance(row, Mapping) or any(
            int(row.get(key, 0)) < 1
            for key in ("complete_null", "scenario", "power_per_point")
        ):
            raise ValueError(f"Runner replicate phase {phase!r} is invalid")
    power = spec.get("power")
    if not isinstance(power, Mapping):
        raise ValueError("Runner power specification must be a mapping")
    target_power = float(power.get("target_power", math.nan))
    target_effect = float(power.get("target_effect_standardized", math.nan))
    effect_grid = tuple(float(value) for value in power["effect_grid_standardized"])
    onset_grid = tuple(float(value) for value in power["onset_shift_grid"])
    if not 0.0 < target_power < 1.0 or target_effect <= 0:
        raise ValueError("Runner power targets are invalid")
    if (
        effect_grid != tuple(sorted(set(effect_grid)))
        or onset_grid != tuple(sorted(set(onset_grid)))
        or effect_grid[0] != 0.0
        or onset_grid[0] != 0.0
        or not any(math.isclose(value, target_effect) for value in effect_grid)
        or float(power.get("onset_shift_standardized_unit", math.nan)) <= 0
        or not 0
        < float(power.get("onset_recovery_tolerance", math.nan))
        <= 0.10
        or power.get("leave_one_control_out_at_target_effect") is not True
    ):
        raise ValueError("Runner power grids must be sorted, unique, and anchored")


def _validate_runner_spec_v2(spec: Mapping[str, Any]) -> None:
    if spec.get("design_source") != "outcome_blind_design_profile_v1":
        raise ValueError("Runner v2 must derive its design from a blind profile")
    blind_gate = spec.get("blind_gate")
    required_flags = {
        "design_profile_required",
        "candidate_expression_matrices_forbidden_after_profile",
        "candidate_pathway_scores_forbidden",
        "observed_pathway_outcomes_forbidden",
        "raw_identifiers_forbidden",
    }
    if not isinstance(blind_gate, Mapping) or any(
        blind_gate.get(flag) is not True for flag in required_flags
    ):
        raise ValueError("Runner v2 blind gate is incomplete")
    exact_bindings = spec.get("exact_product_bindings")
    if (
        not isinstance(exact_bindings, Mapping)
        or exact_bindings.get("bindings_schema_version") != BINDINGS_SCHEMA_VERSION
        or exact_bindings.get("design_profile_sha256_required") is not True
        or exact_bindings.get("design_profile_payload_sha256_required") is not True
        or exact_bindings.get("candidate_paths_forbidden") is not True
    ):
        raise ValueError("Runner v2 exact-product bindings are incomplete")
    design = spec.get("design")
    if not isinstance(design, Mapping):
        raise ValueError("Runner v2 design must be a mapping")
    if int(design.get("n_disomy_controls", 0)) != 3 or int(
        design.get("n_t21_cases", 0)
    ) != 14:
        raise ValueError("Runner v2 must preserve the frozen 3 + 14 donor design")
    if int(design.get("required_grid_bins", 0)) != 20:
        raise ValueError("Runner v2 must require the frozen 20-bin common grid")
    alpha = float(design.get("alpha", math.nan))
    if not 0.0 < alpha < 1.0:
        raise ValueError("Runner v2 alpha must lie between zero and one")
    parameterization = design.get("profile_parameterization")
    required_parameters = {
        "donor_covariate_signatures",
        "sensitivity_signature_association",
        "donor_cell_count_precision",
        "fixed_grid_missingness",
        "trajectory_draw_dispersion",
        "fate_probability_dispersion",
        "pathway_overlap",
        "chr21_overlap",
        "pooled_log_expression_dispersion",
        "production_support_selection",
        "pathway_family_topology",
        "production_occupancy_fate_decomposition",
        "gene_level_chr21_total_trans_projection",
        "selected_support_design_validation",
    }
    if not isinstance(parameterization, Mapping) or any(
        parameterization.get(key) is not True for key in required_parameters
    ):
        raise ValueError("Runner v2 does not consume every frozen profile component")
    scales = design.get("scale_contract")
    scale_keys = {
        "base_regulation_projection_residual_sd",
        "base_timing_projection_residual_sd",
        "base_duration_projection_residual_sd",
        "covariate_slope_sd",
        "sensitivity_signature_slope_sd",
        "extreme_donor_noise_scale",
        "chr21_cis_effect_standardized",
        "occupancy_effect_standardized",
        "fate_effect_standardized",
        "occupancy_pseudocount",
        "fate_pseudocount",
        "occupancy_logistic_normal_base_sd",
        "occupancy_signature_logit_scale",
        "fate_logistic_normal_base_sd",
        "fate_signature_logit_scale",
        "mean_variance_log_correlation_tolerance",
        "minimum_noise_multiplier",
        "maximum_noise_multiplier",
        "minimum_mapping_shift_sd",
        "mapping_shift_multiplier",
    }
    if not isinstance(scales, Mapping) or any(
        not math.isfinite(float(scales.get(key, math.nan)))
        or float(scales.get(key, math.nan)) <= 0
        for key in scale_keys
    ):
        raise ValueError("Runner v2 scale contract is incomplete")
    standardization = design.get("effect_standardization_contract")
    if (
        not isinstance(standardization, Mapping)
        or set(standardization) != {"chr21_cis", "occupancy", "fate"}
        or any(len(str(standardization[key]).strip()) < 40 for key in standardization)
    ):
        raise ValueError("Runner v2 effect standardization contract is incomplete")
    inference = spec.get("inference")
    if (
        not isinstance(inference, Mapping)
        or inference.get("whole_donor_assignment_enumeration") != "exhaustive"
        or inference.get("label_space_exhaustive") is not True
        or int(inference.get("condition_label_space_size", -1)) != 680
        or inference.get("covariate_adjustment")
        != "freedman_lane_style_residualized_condition_contrast"
        or inference.get("finite_sample_exactness_with_continuous_covariates_claimed")
        is not False
        or inference.get("pathway_family_control") != "permutation_maxT"
        or inference.get("level_2_control") != "Benjamini_Yekutieli"
        or inference.get("coverage_metric")
        != "complete_null_pointwise_condition_curve_95pct_t_interval_coverage"
        or inference.get("onset_interval_coverage_evaluated") is not False
        or inference.get("timing_event_estimator")
        != "production_consecutive_significant_windows"
        or int(inference.get("timing_min_consecutive_windows", 0)) < 1
        or not 0.0
        < float(inference.get("timing_min_duration_fraction", math.nan))
        <= 1.0
        or inference.get("fixed_common_grid") is not True
        or int(inference.get("max_exhaustive_residual_mappings", 0)) != 20000
        or inference.get("residual_reference_mode") != "auto"
        or int(inference.get("monte_carlo_residual_mappings", 0)) != 999
        or int(inference.get("residual_mapping_seed", -1))
        != 3713135434119673626
        or inference.get("residual_mapping_seed_rule")
        != "sha256_uint64_little_endian_of_20260713_colon_shared-plan"
    ):
        raise ValueError("Runner v2 inference/exactness contract is invalid")
    statement = str(inference.get("exactness_statement", "")).lower()
    if (
        "680" not in statement
        or "diagnostic" not in statement
        or "residual" not in statement
        or "monte carlo" not in statement
        or "not" not in statement
        or "finite-sample exact" not in statement
    ):
        raise ValueError("Runner v2 exactness statement is not sufficiently explicit")
    support = inference.get("support_selection")
    if not isinstance(support, Mapping) or any(
        int(support.get(key, 0)) < 1
        for key in (
            "min_cells_per_donor_bin",
            "min_donors_per_condition",
            "min_common_bins",
            "min_residual_df",
        )
    ):
        raise ValueError("Runner v2 production support-selection contract is incomplete")
    if not 2 <= int(support["min_common_bins"]) <= 20 or float(
        support.get("max_condition_vif", math.nan)
    ) < 1.0:
        raise ValueError("Runner v2 support-selection thresholds are invalid")
    loco_support = inference.get("leave_one_control_out_support_gate")
    if (
        not isinstance(loco_support, Mapping)
        or loco_support.get("fixed_full_design_selected_segment") is not True
        or loco_support.get("segment_reselection_forbidden") is not True
        or int(loco_support.get("expected_controls_after_omission", -1)) != 2
        or int(loco_support.get("min_controls_per_selected_bin", -1)) != 2
        or int(loco_support.get("min_cases_per_selected_bin", -1)) < 3
        or int(loco_support.get("min_residual_df", 0)) < 1
        or float(loco_support.get("max_condition_vif", math.nan)) < 1.0
        or loco_support.get("condition_and_permutation_information_required")
        is not True
    ):
        raise ValueError("Runner v2 LOCO support gate is incomplete")
    performance = spec.get("performance_contract")
    if (
        not isinstance(performance, Mapping)
        or performance.get("vectorized_shared_freedman_lane_batch") is not True
        or performance.get("scalar_kernel_parity_required") is not True
        or int(performance.get("mapping_batch_size", -1)) != 16
        or not 1 <= int(performance.get("maximum_replicate_chunk_size", 0)) <= 32
        or performance.get("publication_runner_benchmark_required") is not True
    ):
        raise ValueError("Runner v2 performance/parity contract is incomplete")
    publication = spec.get("publication_execution_contract")
    if (
        not isinstance(publication, Mapping)
        or publication.get("phase") != "final"
        or int(publication.get("seed", -1)) != 20260713
        or int(publication.get("chunk_size", -1)) != 32
        or publication.get("development_override") is not False
        or publication.get("publication_minima_required") is not True
        or int(publication.get("complete_null_replicates", -1)) != 10000
        or int(publication.get("scenario_replicates", -1)) != 2000
        or int(publication.get("power_replicates_per_point", -1)) != 1000
    ):
        raise ValueError("Runner v2 publication execution contract changed")
    _validate_power_and_phases(spec)


def load_runner_spec(path: str | Path) -> dict[str, Any]:
    """Load and validate the outcome-blind runner specification."""
    source = Path(path)
    spec = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("Calibration runner specification must be a mapping")
    if spec.get("schema_name") != "t21_pre_unblinding_calibration_runner_spec":
        raise ValueError("Unexpected calibration runner specification schema")
    if spec.get("outcome_blinded_at_freeze") is not True:
        raise ValueError("Calibration runner specification was not frozen blind")
    if spec.get("real_pathway_results_may_be_read") is not False:
        raise ValueError("Calibration runner specification permits real outcomes")
    scenarios = spec.get("scenario_truth")
    if not isinstance(scenarios, Mapping) or set(scenarios) != set(SCENARIOS):
        raise ValueError("Runner specification must define exactly six scenarios")
    if str(spec.get("schema_version", "")).startswith("2."):
        _validate_runner_spec_v2(spec)
        return spec
    blind_gate = spec.get("blind_gate")
    required_blind_flags = {
        "bindings_only_input",
        "candidate_expression_matrices_forbidden",
        "candidate_pathway_scores_forbidden",
        "observed_pathway_outcomes_forbidden",
    }
    if not isinstance(blind_gate, Mapping) or any(
        blind_gate.get(flag) is not True for flag in required_blind_flags
    ):
        raise ValueError("Runner specification blind gate is incomplete")
    exact_bindings = spec.get("exact_product_bindings")
    if (
        not isinstance(exact_bindings, Mapping)
        or exact_bindings.get("required_sha256")
        != ["scrna_sha256", "donor_design_sha256", "fates_sha256"]
        or exact_bindings.get("trajectory_binding") != "trajectory_tree_digest_sha256"
    ):
        raise ValueError("Runner specification exact-product bindings are incomplete")
    design = spec.get("design")
    if not isinstance(design, Mapping):
        raise ValueError("Runner specification design must be a mapping")
    integer_fields = (
        "n_disomy_controls",
        "n_t21_cases",
        "n_pathways",
        "n_chr21_cis_pathways",
        "n_curve_bins",
)

    if any(int(design.get(key, 0)) < 1 for key in integer_fields):
        raise ValueError("Runner design integer counts must be positive")
    if int(design["n_chr21_cis_pathways"]) >= int(design["n_pathways"]):
        raise ValueError("At least one simulated trans pathway is required")
    alpha = float(design.get("alpha", math.nan))
    if not 0.0 < alpha < 1.0:
        raise ValueError("Runner alpha must be between zero and one")
    for key in ("pathway_correlation", "bin_ar1_correlation"):
        value = float(design.get(key, math.nan))
        if not 0.0 <= value < 1.0:
            raise ValueError(f"Runner design {key} must be in [0, 1)")
    positive_fields = (
        "regulation_projection_residual_sd",
        "timing_projection_residual_sd",
        "duration_projection_residual_sd",
        "covariate_slope_sd",
        "case_age_shift_sd",
        "extreme_donor_noise_scale",
        "occupancy_case_variance_ratio",
        "fate_case_variance_ratio",
        "chr21_cis_effect_standardized",
    )
    positive_values = [float(design.get(key, math.nan)) for key in positive_fields]
    if any(not math.isfinite(value) or value <= 0 for value in positive_values):
        raise ValueError("Runner design scale parameters must be positive")
    mapping_shift = float(design.get("residual_mapping_shift_sd", math.nan))
    if not math.isfinite(mapping_shift) or mapping_shift < 0:
        raise ValueError("residual_mapping_shift_sd must be finite and non-negative")
    label_space = math.comb(
        int(design["n_disomy_controls"]) + int(design["n_t21_cases"]),
        int(design["n_disomy_controls"]),
    )
    inference = spec.get("inference")
    if (
        not isinstance(inference, Mapping)
        or inference.get("whole_donor_assignment_enumeration") != "exact"
        or int(inference.get("condition_label_space_size", -1)) != label_space
        or inference.get("covariate_adjustment") != "residualized_condition_contrast"
        or inference.get("pathway_family_control") != "permutation_maxT"
        or inference.get("level_2_control") != "Benjamini_Yekutieli"
        or inference.get("confidence_interval") != "pointwise_t_95pct"
        or inference.get("fixed_common_grid") is not True
    ):
        raise ValueError("Runner inference contract is not exact fixed-grid")
    power = spec.get("power")
    if not isinstance(power, Mapping):
        raise ValueError("Runner power specification must be a mapping")
    target_power = float(power.get("target_power", math.nan))
    target_effect = float(power.get("target_effect_standardized", math.nan))
    effect_grid = tuple(float(value) for value in power["effect_grid_standardized"])
    onset_grid = tuple(float(value) for value in power["onset_shift_grid"])
    if not 0.0 < target_power < 1.0 or target_effect <= 0:
        raise ValueError("Runner power targets are invalid")
    if (
        effect_grid != tuple(sorted(set(effect_grid)))
        or onset_grid != tuple(sorted(set(onset_grid)))
        or effect_grid[0] != 0.0
        or onset_grid[0] != 0.0
        or not any(math.isclose(value, target_effect) for value in effect_grid)
        or float(power.get("onset_shift_standardized_unit", math.nan)) <= 0
        or power.get("leave_one_control_out_at_target_effect") is not True
    ):
        raise ValueError("Runner power grids must be sorted, unique, and anchored")
    return spec


def build_run_plan(
    spec: Mapping[str, Any],
    phase: str,
    *,
    development_replicates: int | None = None,
    allow_development_override: bool = False,
) -> CalibrationRunPlan:
    """Resolve smoke, screen, or final publication-scale replicate counts."""
    phase = str(phase).lower()
    phases = spec.get("phase_replicates")
    if not isinstance(phases, Mapping) or phase not in phases:
        raise ValueError("phase must be one of smoke, screen, or final")
    row = phases[phase]
    if not isinstance(row, Mapping):
        raise ValueError(f"Replicate plan for {phase!r} must be a mapping")
    counts = {
        "complete_null": int(row["complete_null"]),
        "scenario": int(row["scenario"]),
        "power_per_point": int(row["power_per_point"]),
    }
    development_override = development_replicates is not None
    if development_override:
        if not allow_development_override:
            raise ValueError(
                "Underpowered development counts require an explicit override"
            )
        counts = dict.fromkeys(counts, int(development_replicates))
    if any(value < 1 for value in counts.values()):
        raise ValueError("All calibration replicate counts must be positive")
    return CalibrationRunPlan(
        phase=phase,
        complete_null_replicates=counts["complete_null"],
        scenario_replicates=counts["scenario"],
        power_replicates_per_point=counts["power_per_point"],
        development_override=development_override,
    )


def plan_meets_acceptance_minima(
    plan: CalibrationRunPlan, policy: Mapping[str, Any]
) -> bool:
    """Return whether a plan reaches every frozen publication minimum."""
    minima = policy.get("replicate_minima")
    if not isinstance(minima, Mapping):
        raise ValueError("Calibration policy replicate_minima must be a mapping")
    return bool(
        not plan.development_override
        and plan.complete_null_replicates >= int(minima["final_null"])
        and plan.scenario_replicates >= int(minima["scenario_screen"])
        and plan.power_replicates_per_point >= int(minima["power_per_point"])
    )


def validate_blind_bindings(
    bindings: Mapping[str, Any],
    *,
    analysis_plan_path: str | Path,
    calibration_policy_path: str | Path,
    runner_spec_path: str | Path | None = None,
    design_profile_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Accept only scalar candidate hashes; never accept outcome paths or tables."""
    if not isinstance(bindings, Mapping):
        raise TypeError("Calibration bindings must be a JSON object")
    if bindings.get("bindings_schema_version") == BINDINGS_SCHEMA_VERSION:
        keys = set(bindings)
        missing = REQUIRED_BINDING_KEYS_V2 - keys
        extra = keys - REQUIRED_BINDING_KEYS_V2 - OPTIONAL_BINDING_KEYS
        if missing:
            raise ValueError(f"Calibration bindings are missing keys: {sorted(missing)}")
        if extra:
            raise ValueError(
                f"Calibration bindings contain forbidden non-binding keys: {sorted(extra)}"
            )
        normalized = dict(bindings)
        for key in _SHA256_KEYS_V2:
            value = normalized.get(key)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"Calibration binding {key!r} is not a SHA256 digest")
        for key, value in normalized.items():
            if isinstance(value, str) and ("/" in value or "\\" in value):
                raise ValueError(f"Calibration binding {key!r} leaks a filesystem path")
        if (
            isinstance(normalized.get("formal_analysis_cell_count"), bool)
            or not isinstance(normalized.get("formal_analysis_cell_count"), int)
            or int(normalized["formal_analysis_cell_count"]) < 1
        ):
            raise ValueError("formal_analysis_cell_count must be a positive integer")
        commit = normalized.get("code_commit")
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("Calibration code_commit must be a full lowercase commit")
        if not isinstance(normalized.get("code_dirty"), bool):
            raise ValueError("Calibration code_dirty binding must be boolean")
        patch_digest = normalized.get("code_patch_sha256")
        if normalized["code_dirty"]:
            if not isinstance(patch_digest, str) or not re.fullmatch(
                r"[0-9a-f]{64}", patch_digest
            ):
                raise ValueError("Dirty calibration code requires code_patch_sha256")
        elif patch_digest not in {None, ""}:
            raise ValueError("Clean calibration code must not declare a dirty patch")
        if normalized["analysis_plan_sha256"] != sha256_file(analysis_plan_path):
            raise ValueError("analysis_plan_sha256 does not match the frozen plan")
        if normalized["calibration_policy_sha256"] != sha256_file(
            calibration_policy_path
        ):
            raise ValueError("calibration_policy_sha256 does not match the policy")
        if runner_spec_path is None or design_profile_path is None:
            raise ValueError("Runner v2 bindings require runner-spec and design-profile files")
        if normalized["runner_spec_sha256"] != sha256_file(runner_spec_path):
            raise ValueError("runner_spec_sha256 does not match the frozen runner")
        root = (
            Path(repository_root).resolve()
            if repository_root is not None
            else Path(__file__).resolve().parents[1]
        )
        report_schema = root / "schemas" / "t21_calibration_report_v2.schema.json"
        if normalized["calibration_report_schema_sha256"] != sha256_file(
            report_schema
        ):
            raise ValueError(
                "calibration_report_schema_sha256 does not match the frozen schema"
            )
        profile = load_calibration_design_profile(
            design_profile_path, repository_root=repository_root
        )
        if normalized["design_profile_sha256"] != sha256_file(design_profile_path):
            raise ValueError("design_profile_sha256 does not match the profile file")
        if normalized["design_profile_payload_sha256"] != profile["integrity"][
            "profile_payload_sha256"
        ]:
            raise ValueError("design_profile_payload_sha256 does not match the profile")
        inputs = profile["input_bindings"]
        expected_profile_bindings = {
            "scrna_sha256": inputs["scrna"]["file_sha256"],
            "donor_design_sha256": inputs["donor_design"]["file_sha256"],
            "fates_sha256": inputs["fates"]["file_sha256"],
            "scrna_cell_id_set_hash": inputs["scrna"]["cell_set_hash"],
            "scrna_gene_order_hash": inputs["scrna"]["gene_order_hash"],
            "donor_set_hash": inputs["donor_design"]["donor_set_hash"],
            "scrna_donor_set_hash": inputs["scrna"]["donor_set_hash"],
            "expression_contract_sha256": inputs["scrna"][
                "expression_contract_sha256"
            ],
            "expression_implementation_source_sha256": inputs["scrna"][
                "expression_implementation_sha256"
            ],
            "expression_csr_semantic_sha256": inputs["scrna"][
                "x_semantic_sha256"
            ],
            "formal_analysis_cell_set_hash": inputs["scrna"][
                "formal_analysis_cell_set_hash"
            ],
            "formal_analysis_cell_order_hash": inputs["scrna"][
                "formal_analysis_cell_order_hash"
            ],
            "formal_analysis_cell_count": inputs["scrna"][
                "formal_analysis_cell_count"
            ],
            "formal_gene_order_bound_support_sha256": inputs["scrna"][
                "formal_gene_order_bound_support_sha256"
            ],
            "formal_support_contract_sha256": inputs["scrna"][
                "formal_support_contract_sha256"
            ],
            "formal_support_mask_sha256_uint8": inputs["scrna"][
                "formal_support_mask_sha256_uint8"
            ],
            "formal_analysis_cell_mask_sha256_uint8": inputs["scrna"][
                "formal_analysis_cell_mask_sha256_uint8"
            ],
            "trajectory_tree_digest_sha256": inputs["trajectory"][
                "tree_digest_sha256"
            ],
            "trajectory_grid_hash": inputs["trajectory"]["grid_hash"],
            "trajectory_primary_draw_id_sha256": inputs["trajectory"][
                "primary_draw_id_sha256"
            ],
            "pathway_universe_sha256": inputs["pathway_universe"]["file_sha256"],
            "pathway_universe_logical_sha256": inputs["pathway_universe"][
                "logical_sha256"
            ],
            "supported_pathway_universe_logical_sha256": inputs[
                "pathway_universe"
            ]["supported_logical_sha256"],
        }
        for key, expected in expected_profile_bindings.items():
            if normalized.get(key) != expected:
                raise ValueError(f"Calibration binding {key!r} differs from the profile")
        return normalized
    keys = set(bindings)
    missing = REQUIRED_BINDING_KEYS - keys
    extra = keys - REQUIRED_BINDING_KEYS - OPTIONAL_BINDING_KEYS
    if missing:
        raise ValueError(f"Calibration bindings are missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(
            f"Calibration bindings contain forbidden non-binding keys: {sorted(extra)}"
        )
    normalized = dict(bindings)
    for key in _SHA256_KEYS:
        value = normalized.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"Calibration binding {key!r} is not a SHA256 digest")
    if normalized["pathway_universe_relative_path"] != (
        "reference/t21_pathway_universe_v1.tsv"
    ):
        raise ValueError("Calibration must bind the canonical pathway universe")
    commit = normalized.get("code_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Calibration code_commit must be a full lowercase commit")
    if not isinstance(normalized.get("code_dirty"), bool):
        raise ValueError("Calibration code_dirty binding must be boolean")
    patch_digest = normalized.get("code_patch_sha256")
    if normalized["code_dirty"]:
        if not isinstance(patch_digest, str) or not re.fullmatch(
            r"[0-9a-f]{64}", patch_digest
        ):
            raise ValueError("Dirty calibration code requires code_patch_sha256")
    elif patch_digest not in {None, ""}:
        raise ValueError("Clean calibration code must not declare a dirty patch")
    expected_plan = sha256_file(analysis_plan_path)
    if normalized["analysis_plan_sha256"] != expected_plan:
        raise ValueError("analysis_plan_sha256 does not match the frozen plan")
    expected_policy = sha256_file(calibration_policy_path)
    if normalized["calibration_policy_sha256"] != expected_policy:
        raise ValueError("calibration_policy_sha256 does not match the policy")
    return normalized


def enumerate_whole_donor_assignments(n_controls: int, n_cases: int) -> np.ndarray:
    """Enumerate all case labels while preserving whole-donor group counts."""
    n_controls = int(n_controls)
    n_cases = int(n_cases)
    if n_controls < 1 or n_cases < 1:
        raise ValueError("Both donor groups must be non-empty")
    n_donors = n_controls + n_cases
    rows = []
    for control_indices in itertools.combinations(range(n_donors), n_controls):
        case = np.ones(n_donors, dtype=float)
        case[list(control_indices)] = 0.0
        rows.append(case)
    return np.asarray(rows, dtype=float)


def _make_exact_contrast_plan(
    age: np.ndarray, observed_case: np.ndarray
) -> _ExactContrastPlan:
    age = np.asarray(age, dtype=float)
    observed_case = np.asarray(observed_case, dtype=float)
    if age.ndim != 1 or observed_case.shape != age.shape:
        raise ValueError("age and observed_case must be matching vectors")
    if not np.isfinite(age).all() or np.std(age) <= 0:
        raise ValueError("age must contain finite variation")
    return _make_covariate_contrast_plan(age[:, None], observed_case)


def _make_covariate_contrast_plan(
    covariates: np.ndarray, observed_case: np.ndarray
) -> _ExactContrastPlan:
    covariates = np.asarray(covariates, dtype=float)
    observed_case = np.asarray(observed_case, dtype=float)
    if covariates.ndim == 1:
        covariates = covariates[:, None]
    if (
        covariates.ndim != 2
        or observed_case.ndim != 1
        or covariates.shape[0] != len(observed_case)
        or not np.isfinite(covariates).all()
    ):
        raise ValueError("covariates and observed_case must be finite donor-aligned arrays")
    values, counts = np.unique(observed_case, return_counts=True)
    if not np.array_equal(values, np.asarray([0.0, 1.0])):
        raise ValueError("observed_case must be binary")
    n_controls, n_cases = int(counts[0]), int(counts[1])
    assignments = enumerate_whole_donor_assignments(n_controls, n_cases)
    centered_columns = []
    for column in covariates.T:
        centered = column - np.mean(column)
        scale = float(np.std(centered))
        if scale > np.finfo(float).eps:
            centered_columns.append(centered / scale)
    if not centered_columns:
        raise ValueError("At least one varying donor covariate is required")
    candidate = np.column_stack([np.ones(len(observed_case)), *centered_columns])
    independent = [candidate[:, 0]]
    rank = 1
    for column in candidate[:, 1:].T:
        trial = np.column_stack([*independent, column])
        trial_rank = int(np.linalg.matrix_rank(trial))
        if trial_rank > rank:
            independent.append(column)
            rank = trial_rank
    reduced = np.column_stack(independent)
    residual_maker = np.eye(len(observed_case)) - reduced @ np.linalg.pinv(reduced)
    residual_conditions = assignments @ residual_maker
    denominators = np.einsum("mn,mn->m", residual_conditions, residual_conditions)
    if np.any(denominators <= np.finfo(float).eps):
        raise ValueError("At least one donor assignment is covariate-confounded")
    matches = np.flatnonzero(np.all(assignments == observed_case, axis=1))
    if len(matches) != 1:
        raise ValueError("Observed donor assignment is absent or duplicated")
    degrees_of_freedom = len(observed_case) - np.linalg.matrix_rank(reduced) - 1
    if degrees_of_freedom < 2:
        raise ValueError("Donor design has too few residual degrees of freedom")
    return _ExactContrastPlan(
        residual_maker=residual_maker,
        residual_conditions=residual_conditions,
        denominators=denominators,
        observed_assignment_index=int(matches[0]),
        degrees_of_freedom=int(degrees_of_freedom),
        n_assignments=int(len(assignments)),
    )


def _reduced_covariate_design(covariates: np.ndarray) -> np.ndarray:
    values = np.asarray(covariates, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    candidate = np.column_stack([np.ones(len(values)), values])
    independent = [candidate[:, 0]]
    rank = 1
    for column in candidate[:, 1:].T:
        trial = np.column_stack([*independent, column])
        trial_rank = int(np.linalg.matrix_rank(trial))
        if trial_rank > rank:
            independent.append(column)
            rank = trial_rank
    return np.column_stack(independent)


def _by_adjust_rows(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values, axis=1)
    ordered = np.take_along_axis(p_values, order, axis=1)
    n_tests = p_values.shape[1]
    harmonic = float(np.sum(1.0 / np.arange(1, n_tests + 1)))
    scaled = ordered * harmonic * n_tests / np.arange(1, n_tests + 1)
    monotone = np.minimum.accumulate(scaled[:, ::-1], axis=1)[:, ::-1]
    adjusted_ordered = np.clip(monotone, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_ordered)
    np.put_along_axis(adjusted, order, adjusted_ordered, axis=1)
    return adjusted


def _exact_test(
    values: np.ndarray, plan: _ExactContrastPlan, alpha: float
) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 3 or values.shape[2] != plan.residual_maker.shape[0]:
        raise ValueError("values must have shape replicate x pathway x donor")
    residual_values = values @ plan.residual_maker
    numerators = np.einsum(
        "rpn,mn->rpm", residual_values, plan.residual_conditions, optimize=True
    )
    betas = numerators / plan.denominators[None, None, :]
    sum_squares = np.einsum("rpn,rpn->rp", residual_values, residual_values)
    residual_sse = sum_squares[:, :, None] - (
        numerators**2 / plan.denominators[None, None, :]
    )
    residual_sse = np.maximum(residual_sse, np.finfo(float).tiny)
    standard_errors = np.sqrt(
        residual_sse / plan.degrees_of_freedom / plan.denominators[None, None, :]
    )
    statistics = betas / standard_errors
    observed_index = plan.observed_assignment_index
    observed_absolute = np.abs(statistics[:, :, observed_index])
    absolute_reference = np.abs(statistics)
    p_raw = np.mean(absolute_reference >= observed_absolute[:, :, None] - 1e-12, axis=2)
    max_reference = np.max(absolute_reference, axis=1)
    p_max_t = np.mean(
        max_reference[:, None, :] >= observed_absolute[:, :, None] - 1e-12,
        axis=2,
    )
    q_by = _by_adjust_rows(p_raw)
    return {
        "beta": betas[:, :, observed_index],
        "standard_error": standard_errors[:, :, observed_index],
        "p_raw": p_raw,
        "p_maxT": p_max_t,
        "q_by": q_by,
        "maxT_reject": p_max_t <= alpha,
        "by_reject": q_by <= alpha,
    }


def _seed_for(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


_CURVE_RANDOM_STREAM_NAMES = (
    "independent",
    "common",
    "nuisance",
    "sensitivity",
    "mapping",
)


def _curve_random_streams(
    seed: int, label: str
) -> dict[str, np.random.Generator]:
    """Return persistent named streams so replicate draws ignore chunk boundaries."""
    return {
        name: np.random.default_rng(_seed_for(seed, f"{label}:{name}"))
        for name in _CURVE_RANDOM_STREAM_NAMES
    }


def _donor_design(
    n_controls: int,
    n_cases: int,
    *,
    confounded: bool,
    case_age_shift_sd: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    observed_case = np.r_[np.zeros(n_controls), np.ones(n_cases)]
    control_age = np.linspace(-1.0, 1.0, n_controls)
    case_age = np.linspace(-1.3, 1.3, n_cases)
    if confounded:
        control_age -= case_age_shift_sd / 2.0
        case_age += case_age_shift_sd / 2.0
    return np.r_[control_age, case_age], observed_case


def _weighted_median(values: Sequence[float], weights: Sequence[int]) -> float:
    array = np.asarray(values, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if array.shape != weight.shape or not len(array) or np.sum(weight) <= 0:
        raise ValueError("Weighted median inputs are invalid")
    order = np.argsort(array, kind="stable")
    cumulative = np.cumsum(weight[order])
    index = int(np.searchsorted(cumulative, np.sum(weight) / 2.0, side="left"))
    return float(array[order[min(index, len(order) - 1)]])


def _positive_log_correlation(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
    *,
    label: str,
) -> float:
    """Correlation on a multiplicative, scale-invariant positive log scale."""

    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if (
        left_array.ndim != 1
        or right_array.shape != left_array.shape
        or len(left_array) < 2
        or np.any(~np.isfinite(left_array))
        or np.any(~np.isfinite(right_array))
        or np.any(left_array <= 0)
        or np.any(right_array <= 0)
    ):
        raise ValueError(f"{label} requires paired finite positive vectors")
    log_left = np.log(left_array)
    log_right = np.log(right_array)
    if np.std(log_left) <= np.finfo(float).eps or np.std(log_right) <= np.finfo(float).eps:
        return 0.0
    correlation = float(np.corrcoef(log_left, log_right)[0, 1])
    if not np.isfinite(correlation) or correlation < -1.0 - 1e-12 or correlation > 1.0 + 1e-12:
        raise ValueError(f"{label} produced an invalid correlation")
    return float(np.clip(correlation, -1.0, 1.0))


def _fate_precision_denominators(
    donor_rows: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Return exact eligible counts without a second eligible-fraction discount."""
    denominators = np.asarray(
        [int(row["analysis_cell_count"]) for row in donor_rows], dtype=int
    )
    if np.any(denominators < 1):
        raise ValueError("Fate precision denominators must be positive")
    return denominators


def derive_profile_simulation_parameters(
    spec: Mapping[str, Any], profile: Mapping[str, Any]
) -> dict[str, Any]:
    """Resolve every v2 simulation parameter from the frozen blind profile."""
    if not str(spec.get("schema_version", "")).startswith("2."):
        raise ValueError("Profile-derived simulation parameters require runner spec v2")
    _validate_runner_spec_v2(spec)
    validation = validate_calibration_design_profile(profile)
    design = profile["design"]
    donor_rows = list(design["donor_rows"])
    assignments = np.asarray([row["assignment_code"] for row in donor_rows], dtype=float)
    if int(np.sum(assignments == 0)) != 3 or int(np.sum(assignments == 1)) != 14:
        raise ValueError("Design profile changed the frozen 3 + 14 assignment")
    canonical_design = build_t21_canonical_donor_design(
        donor_ids=[row["donor_slot"] for row in donor_rows],
        conditions=[
            "T21" if int(row["assignment_code"]) == 1 else "disomy"
            for row in donor_rows
        ],
        pcw=[float(row["pcw"]) for row in donor_rows],
        technical_batch=[row["formal_batch_code"] for row in donor_rows],
        control="disomy",
        case="T21",
        expected_primary_batch_status=str(
            design["canonical_formal_design"]["technical_batch_status"]
        ),
        donor_order_mode="provided_frozen_slots",
    )
    if (
        canonical_design.audit_manifest() != design["canonical_formal_design"]
        or canonical_design.spec_sha256
        != design["canonical_formal_design_spec_sha256"]
        or canonical_design.spec_sha256
        != canonical_t21_donor_design_spec_sha256()
    ):
        raise ValueError("Calibration canonical donor design differs from the profile")
    covariates = canonical_design.nuisance_matrix
    sensitivity_covariates, sensitivity_components = (
        build_t21_sensitivity_signature_matrix(
            sex_signature=np.asarray(
                [row["sex_signature"] for row in donor_rows], dtype=float
            ),
            batch_signature=np.asarray(
                [row["batch_signature"] for row in donor_rows], dtype=float
            ),
        )
    )
    if sensitivity_covariates.shape[1] < 1:
        raise ValueError(
            "The covariate-condition stress scenario requires at least one "
            "outcome-blind sex/batch sensitivity component"
        )

    grid_rows = pd.DataFrame(profile["fixed_grid"]["fixed_donor_bin_rows"])
    primary_draw_index = int(profile["fixed_grid"]["primary_draw_index"])
    primary_mask_rows = []
    primary_count_rows = []
    for donor_row in donor_rows:
        slot = str(donor_row["donor_slot"])
        primary = grid_rows.loc[
            grid_rows["donor_slot"].eq(slot)
            & grid_rows["draw_index"].eq(primary_draw_index)
        ].sort_values("bin_index")
        if len(primary) != int(profile["fixed_grid"]["n_bins"]):
            raise ValueError("Profile primary-draw donor mask is incomplete")
        primary_mask_rows.append((~primary["missing"].astype(bool)).tolist())
        primary_count_rows.append(primary["cell_count"].astype(int).tolist())
    primary_mask_hash = hashlib.sha256(
        json.dumps(
            primary_mask_rows, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    primary_count_hash = hashlib.sha256(
        json.dumps(
            primary_count_rows, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if (
        primary_mask_hash
        != profile["fixed_grid"]["primary_draw_available_mask_sha256"]
        or primary_count_hash
        != profile["fixed_grid"]["primary_draw_cell_count_sha256"]
    ):
        raise ValueError("Profile primary-draw mask/count hashes changed")
    source_counts = np.asarray(primary_count_rows, dtype=int)
    source_n_bins = int(profile["fixed_grid"]["n_bins"])
    edges = np.concatenate(
        [
            np.asarray(profile["fixed_grid"]["bin_left"], dtype=float)[:1],
            np.asarray(profile["fixed_grid"]["bin_right"], dtype=float),
        ]
    )
    support_contract = spec["inference"]["support_selection"]
    support_donors = canonical_design.donor_frame.copy()
    (
        selected_bins,
        included_donors,
        encoded_support,
        support_groups,
        support_signatures,
        support_blocks,
        segment_diagnostics,
        design_diagnostics,
    ) = _select_segment(
        support_donors,
        source_counts,
        edges,
        min_cells_per_donor_bin=int(support_contract["min_cells_per_donor_bin"]),
        min_donors_per_condition=int(support_contract["min_donors_per_condition"]),
        min_common_bins=int(support_contract["min_common_bins"]),
        min_residual_df=int(support_contract["min_residual_df"]),
        max_condition_vif=float(support_contract["max_condition_vif"]),
        continuous_covariate_keys=canonical_design.continuous_covariate_keys,
        categorical_covariate_keys=canonical_design.categorical_covariate_keys,
        strata_keys=canonical_design.strata_keys,
    )
    if not np.asarray(included_donors, dtype=bool).all():
        raise ValueError(
            "Production support selection excluded a frozen primary-frame donor; "
            "the 17-donor calibration design cannot be preserved"
        )
    if (
        not np.array_equal(encoded_support.reduced, canonical_design.reduced_design)
        or tuple(encoded_support.terms) != canonical_design.terms
        or tuple(dict(value) for value in encoded_support.encoding)
        != canonical_design.encoding
    ):
        raise ValueError(
            "Production support selector did not reproduce the canonical donor design"
        )
    occupancy_baseline_counts = source_counts[:, selected_bins]
    support_available = occupancy_baseline_counts >= int(
        support_contract["min_cells_per_donor_bin"]
    )
    support_counts = np.where(support_available, occupancy_baseline_counts, 0)
    primary_draw_median_positive_cell_count = float(
        np.median(support_counts[support_available])
    )
    selected_bin_mask = np.zeros(source_n_bins, dtype=bool)
    selected_bin_mask[selected_bins] = True
    support_mask_hash = hashlib.sha256(
        json.dumps(
            support_available.tolist(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    support_count_hash = hashlib.sha256(
        json.dumps(
            support_counts.tolist(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    occupancy_count_hash = hashlib.sha256(
        json.dumps(
            occupancy_baseline_counts.tolist(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    selected_bin_mask_hash = hashlib.sha256(selected_bin_mask.tobytes()).hexdigest()
    included_donor_mask_hash = hashlib.sha256(
        np.asarray(included_donors, dtype=bool).tobytes()
    ).hexdigest()
    fate_rows = {
        str(row["donor_slot"]): row
        for row in profile["fate_probability_distribution"]["donor_level"]
    }
    trajectory_rows = {
        str(row["donor_slot"]): row
        for row in profile["trajectory_draw_dispersion"]["donor_level"]
    }
    analysis_counts = np.asarray(
        [max(int(row["analysis_cell_count"]), 1) for row in donor_rows], dtype=float
    )
    median_cells = float(np.median(analysis_counts))
    cell_precision_scale = np.sqrt(median_cells / analysis_counts)
    occupancy_scale = []
    fate_scale = []
    mapping_scale = []
    occupancy_signature = []
    fate_signature = []
    for row in donor_rows:
        slot = str(row["donor_slot"])
        local = grid_rows.loc[grid_rows["donor_slot"].eq(slot)]
        local_counts = local["cell_count"].to_numpy(dtype=float)
        available_fraction = 1.0 - float(local["missing"].astype(bool).mean())
        positive = local_counts[local_counts > 0]
        count_cv = (
            float(np.std(positive) / np.mean(positive))
            if len(positive) and np.mean(positive) > 0
            else 1.0
        )
        occupancy_scale.append(
            math.sqrt(1.0 / max(available_fraction, 0.10)) * (1.0 + min(count_cv, 2.0) / 4.0)
        )
        occupancy_signature.append(
            math.log1p(float(np.sum(local_counts))) + available_fraction
        )
        fate = fate_rows[slot]
        fate_scale.append(
            1.0
            + float(fate["mean_component_variance"])
            + (1.0 - float(fate["eligible_fraction"])) / 2.0
        )
        fate_signature.append(
            float(fate["mean_entropy"]) - float(fate["mean_component_variance"])
        )
        mapping_scale.append(float(trajectory_rows[slot]["mean_draw_dispersion"]))
    occupancy_scale_array = np.asarray(occupancy_scale, dtype=float)
    occupancy_scale_array /= np.median(occupancy_scale_array)
    fate_scale_array = np.asarray(fate_scale, dtype=float)
    fate_scale_array /= np.median(fate_scale_array)
    component_rows = profile["fate_probability_distribution"][
        "component_distributions"
    ]
    fate_baseline_probability = np.asarray(
        [
            max(float(row["probability_quantiles"]["q50"]), 1e-8)
            for row in component_rows
        ],
        dtype=float,
    )
    if fate_baseline_probability.ndim != 1 or len(fate_baseline_probability) < 2:
        raise ValueError("Profile must retain at least two anonymous fate components")
    fate_baseline_probability /= fate_baseline_probability.sum()
    fate_denominators = _fate_precision_denominators(donor_rows)
    if np.any(fate_denominators < 2):
        raise ValueError(
            "Profile fate-eligible denominators must contain at least two cells per donor"
        )
    donor_noise_scale = cell_precision_scale / np.median(cell_precision_scale)
    occupancy_signature_array = np.asarray(occupancy_signature, dtype=float)
    fate_signature_array = np.asarray(fate_signature, dtype=float)
    for values, label in (
        (occupancy_signature_array, "occupancy"),
        (fate_signature_array, "fate"),
    ):
        scale = float(np.std(values))
        if scale <= np.finfo(float).eps:
            raise ValueError(f"Profile {label} signature has no donor variation")
        values -= np.mean(values)
        values /= scale

    mean_variance_bins = profile["pooled_anonymous_log_expression_dispersion"]["bins"]
    pooled_log_expression_dispersion = _weighted_median(
        [
            float(row["log_expression_variance_to_mean_median"])
            for row in mean_variance_bins
        ],
        [int(row["n_features"]) for row in mean_variance_bins],
    )
    scale_contract = spec["design"]["scale_contract"]
    minimum_multiplier = float(scale_contract["minimum_noise_multiplier"])
    maximum_multiplier = float(scale_contract["maximum_noise_multiplier"])
    noise_multiplier = float(
        np.clip(
            math.sqrt(max(pooled_log_expression_dispersion, 1e-6)),
            minimum_multiplier,
            maximum_multiplier,
        )
    )
    pathway = profile["pathway_structure"]
    n_pathways = int(pathway["n_pathways"])
    cis_mask = np.asarray(pathway["chr21_pathway_mask"], dtype=bool)
    family_index = np.asarray(
        pathway["level_1_family_index_by_pathway"], dtype=int
    )
    pathway_dependence = np.asarray(
        pathway["pathway_dependence_correlation"], dtype=float
    )
    member_indices = [
        np.asarray(values, dtype=int)
        for values in pathway["pathway_member_feature_indices"]
    ]
    chr21_member_feature_mask = np.asarray(
        pathway["chr21_member_feature_mask"], dtype=bool
    )
    if (
        cis_mask.shape != (n_pathways,)
        or family_index.shape != (n_pathways,)
        or pathway_dependence.shape != (n_pathways, n_pathways)
        or not np.isfinite(pathway_dependence).all()
        or not np.allclose(pathway_dependence, pathway_dependence.T, atol=1e-12)
        or np.any(family_index < -1)
        or np.min(np.linalg.eigvalsh(pathway_dependence)) < -1e-8
        or len(member_indices) != n_pathways
        or chr21_member_feature_mask.shape
        != (int(pathway["n_unique_member_features"]),)
    ):
        raise ValueError("Profile pathway topology is invalid")
    trans_member_count = np.asarray(
        [int(np.sum(~chr21_member_feature_mask[indices])) for indices in member_indices],
        dtype=int,
    )
    trans_pathway_eligible = trans_member_count > 0
    if not np.any(trans_pathway_eligible):
        raise ValueError(
            "At least one pathway requires non-chr21 members for the trans null"
        )
    n_cis = int(cis_mask.sum())
    if n_cis < 1 or n_cis >= n_pathways:
        raise ValueError("Profile pathway structure must retain cis and trans projections")
    pathway_correlation = float(
        np.clip(
            0.05 + 4.0 * float(pathway["pairwise_jaccard_mean"]),
            0.05,
            0.75,
        )
    )
    anonymous_mean = np.asarray(
        [
            max(float(row["mean_log_expression_median"]), 1e-8)
            for row in mean_variance_bins
        ],
        dtype=float,
    )
    anonymous_variance = np.asarray(
        [
            max(float(row["log_expression_variance_median"]), 1e-8)
            for row in mean_variance_bins
        ],
        dtype=float,
    )
    mean_order = np.argsort(anonymous_mean, kind="stable")
    anonymous_mean = anonymous_mean[mean_order]
    anonymous_variance = anonymous_variance[mean_order]
    quantile_positions = np.linspace(0.0, 1.0, len(anonymous_mean))
    target_positions = np.linspace(0.0, 1.0, n_pathways)
    pathway_baseline_mean = np.interp(
        target_positions, quantile_positions, anonymous_mean
    )
    pathway_noise_scale = np.sqrt(
        np.interp(target_positions, quantile_positions, anonymous_variance)
    )
    mv_permutation = np.random.default_rng(20260714).permutation(n_pathways)
    pathway_baseline_mean = pathway_baseline_mean[mv_permutation]
    pathway_noise_scale = pathway_noise_scale[mv_permutation]
    pathway_noise_scale /= np.median(pathway_noise_scale)
    pathway_noise_scale = np.clip(
        pathway_noise_scale, minimum_multiplier, maximum_multiplier
    )
    pathway_noise_scale /= np.median(pathway_noise_scale)
    mean_variance_log_correlation = _positive_log_correlation(
        anonymous_mean,
        anonymous_variance,
        label="Pooled expression mean-variance relation",
    )
    simulated_mean_variance_log_correlation = _positive_log_correlation(
        pathway_baseline_mean,
        pathway_noise_scale**2,
        label="Simulated pathway mean-variance relation",
    )
    if abs(
        simulated_mean_variance_log_correlation - mean_variance_log_correlation
    ) > float(scale_contract["mean_variance_log_correlation_tolerance"]):
        raise ValueError(
            "Frozen pathway simulation does not preserve the profile mean-variance relation"
        )
    assigned_pathways = np.flatnonzero(family_index >= 0)
    family_sizes = {
        int(family): int(np.sum(family_index == family))
        for family in np.unique(family_index[family_index >= 0])
    }
    mean_overlap = (
        np.sum(pathway_dependence, axis=1) - np.diag(pathway_dependence)
    ) / max(n_pathways - 1, 1)
    power_target_index = max(
        assigned_pathways.tolist(),
        key=lambda index: (
            family_sizes[int(family_index[index])],
            float(mean_overlap[index]),
            -int(index),
        ),
    )
    bin_correlation = float(
        np.clip(profile["fixed_grid"]["adjacent_log_count_correlation"], 0.0, 0.95)
    )
    source_n_curve_bins = int(profile["fixed_grid"]["n_bins"])
    if source_n_curve_bins != int(spec["design"]["required_grid_bins"]):
        raise ValueError(
            "Design profile fixed-grid bin count differs from the frozen runner contract"
        )
    mapping_median = float(
        profile["trajectory_draw_dispersion"]["cell_level_draw_sd_quantiles"]["q50"]
    )
    mapping_shift = max(
        float(scale_contract["minimum_mapping_shift_sd"]),
        mapping_median * float(scale_contract["mapping_shift_multiplier"]),
    )
    regulation_projection_residual_sd = float(
        scale_contract["base_regulation_projection_residual_sd"]
    ) * noise_multiplier
    power_target_effect_unit = float(
        regulation_projection_residual_sd * pathway_noise_scale[power_target_index]
    )
    derived = {
        "profile_payload_sha256": str(validation["profile_payload_sha256"]),
        "n_disomy_controls": 3,
        "n_t21_cases": 14,
        "n_pathways": n_pathways,
        "n_chr21_cis_pathways": n_cis,
        "chr21_pathway_mask": cis_mask.tolist(),
        "level_1_family_index_by_pathway": family_index.tolist(),
        "n_level_1_families": int(pathway["n_level_1_families"]),
        "power_target_pathway_index": int(power_target_index),
        "power_target_selection_rule": (
            "assigned pathway in largest Level-1 family, then highest anonymous "
            "mean overlap, then lowest anonymous index"
        ),
        "power_target_effect_standardization_sd": power_target_effect_unit,
        "power_target_effect_standardization_rule": (
            "profile-derived regulation projection residual SD multiplied by the "
            "frozen anonymous target-pathway noise scale"
        ),
        "anonymous_pathway_order_sha256": str(
            pathway["anonymous_pathway_order_sha256"]
        ),
        "pathway_dependence_correlation": pathway_dependence.tolist(),
        "pathway_member_feature_indices": [
            values.astype(int).tolist() for values in member_indices
        ],
        "chr21_member_feature_mask": chr21_member_feature_mask.tolist(),
        "anonymous_member_feature_order_sha256": str(
            pathway["anonymous_member_feature_order_sha256"]
        ),
        "trans_member_count_by_pathway": trans_member_count.tolist(),
        "trans_pathway_eligible_mask": trans_pathway_eligible.tolist(),
        "pathway_noise_scale": pathway_noise_scale.tolist(),
        "pathway_baseline_mean": pathway_baseline_mean.tolist(),
        "pooled_mean_variance_log_correlation": mean_variance_log_correlation,
        "simulated_mean_variance_log_correlation": (
            simulated_mean_variance_log_correlation
        ),
        "mean_variance_joint_draw_rule": (
            "joint interpolation of anonymous mean-expression and variance medians "
            "followed by one frozen identity-independent permutation"
        ),
        "source_n_curve_bins": source_n_bins,
        "n_curve_bins": int(len(selected_bins)),
        "selected_bin_indices": selected_bins.astype(int).tolist(),
        "selected_bin_mask": selected_bin_mask.tolist(),
        "selected_bin_mask_sha256": selected_bin_mask_hash,
        "included_donor_mask": np.asarray(included_donors, dtype=bool).tolist(),
        "included_donor_mask_sha256": included_donor_mask_hash,
        "fixed_20_bin_source_grid_verified": True,
        "selected_support_design_valid": True,
        "support_reduced_design": encoded_support.reduced.tolist(),
        "support_reduced_design_sha256": canonical_design.reduced_design_sha256,
        "support_reduced_design_terms": list(canonical_design.terms),
        "support_reduced_design_terms_sha256": canonical_design.terms_sha256,
        "support_reduced_design_encoding": _json_ready(encoded_support.encoding),
        "support_reduced_design_encoding_sha256": canonical_design.encoding_sha256,
        "canonical_donor_design_spec_sha256": canonical_design.spec_sha256,
        "canonical_technical_batch_status": canonical_design.technical_batch_status,
        "support_availability_signatures": [
            str(value) for value in support_signatures
        ],
        "support_permutation_blocks": [str(value) for value in support_blocks],
        "support_residual_groups": [
            np.asarray(group, dtype=int).tolist() for group in support_groups
        ],
        "support_bin_design_diagnostics": _json_ready(
            design_diagnostics.to_dict("records")
        ),
        "support_selected_segment_diagnostic": _json_ready(
            segment_diagnostics.loc[
                segment_diagnostics["selected_segment"].fillna(False)
            ].iloc[0].to_dict()
        ),
        "support_selection_contract": dict(support_contract),
        "bin_widths": (
            np.asarray(profile["fixed_grid"]["bin_right"], dtype=float)
            - np.asarray(profile["fixed_grid"]["bin_left"], dtype=float)
        )[selected_bins].tolist(),
        "selected_bin_left": np.asarray(
            profile["fixed_grid"]["bin_left"], dtype=float
        )[selected_bins].tolist(),
        "selected_bin_right": np.asarray(
            profile["fixed_grid"]["bin_right"], dtype=float
        )[selected_bins].tolist(),
        "alpha": float(spec["design"]["alpha"]),
        "observed_assignment": assignments.astype(int).tolist(),
        "covariate_matrix": covariates.tolist(),
        "covariate_matrix_sha256": covariate_matrix_sha256(covariates),
        "covariate_rank": int(np.linalg.matrix_rank(covariates - covariates.mean(axis=0))),
        "sensitivity_signature_matrix": sensitivity_covariates.tolist(),
        "sensitivity_signature_matrix_sha256": covariate_matrix_sha256(
            sensitivity_covariates
        ),
        "sensitivity_signature_components": sensitivity_components,
        "donor_noise_scale": donor_noise_scale.tolist(),
        "occupancy_noise_scale": occupancy_scale_array.tolist(),
        "fate_noise_scale": fate_scale_array.tolist(),
        "occupancy_detector_signature": occupancy_signature_array.tolist(),
        "fate_detector_signature": fate_signature_array.tolist(),
        "fate_baseline_probability": fate_baseline_probability.tolist(),
        "fate_eligible_denominator": fate_denominators.tolist(),
        "trajectory_mapping_dispersion_by_donor": mapping_scale,
        "primary_draw_available_mask": support_available.tolist(),
        "primary_draw_cell_count": support_counts.tolist(),
        "source_primary_draw_nonmissing_mask": np.asarray(
            primary_mask_rows, dtype=bool
        ).tolist(),
        "source_primary_draw_cell_count": source_counts.tolist(),
        "source_support_available_mask": (
            source_counts >= int(support_contract["min_cells_per_donor_bin"])
        ).tolist(),
        "primary_draw_median_positive_cell_count": (
            primary_draw_median_positive_cell_count
        ),
        "primary_draw_available_mask_sha256": support_mask_hash,
        "primary_draw_cell_count_sha256": support_count_hash,
        "occupancy_baseline_cell_count": occupancy_baseline_counts.tolist(),
        "occupancy_baseline_cell_count_sha256": occupancy_count_hash,
        "source_primary_draw_available_mask_sha256": primary_mask_hash,
        "source_primary_draw_cell_count_sha256": primary_count_hash,
        "shared_freedman_lane_kernel_sha256": profile["code_bindings"][
            "shared_freedman_lane_kernel_sha256"
        ],
        "covariate_pseudobulk_core_sha256": profile["code_bindings"][
            "covariate_pseudobulk_core_sha256"
        ],
        "pathway_family_inference_core_sha256": profile["code_bindings"][
            "pathway_family_inference_core_sha256"
        ],
        "trajectory_decomposition_core_sha256": profile["code_bindings"][
            "trajectory_decomposition_core_sha256"
        ],
        "trajectory_event_timing_core_sha256": profile["code_bindings"][
            "trajectory_event_timing_core_sha256"
        ],
        "t21_covariate_design_core_sha256": profile["code_bindings"][
            "t21_covariate_design_core_sha256"
        ],
        "pathway_correlation": pathway_correlation,
        "bin_ar1_correlation": bin_correlation,
        "regulation_projection_residual_sd": regulation_projection_residual_sd,
        "timing_projection_residual_sd": float(
            scale_contract["base_timing_projection_residual_sd"]
        )
        * noise_multiplier,
        "duration_projection_residual_sd": float(
            scale_contract["base_duration_projection_residual_sd"]
        )
        * noise_multiplier,
        "covariate_slope_sd": float(scale_contract["covariate_slope_sd"]),
        "sensitivity_signature_slope_sd": float(
            scale_contract["sensitivity_signature_slope_sd"]
        ),
        "extreme_donor_noise_scale": float(
            scale_contract["extreme_donor_noise_scale"]
        ),
        "residual_mapping_shift_sd": mapping_shift,
        "chr21_cis_effect_standardized": float(
            scale_contract["chr21_cis_effect_standardized"]
        ),
        "occupancy_effect_standardized": float(
            scale_contract["occupancy_effect_standardized"]
        ),
        "occupancy_condition_logit_effect": float(
            scale_contract["occupancy_effect_standardized"]
        )
        * float(scale_contract["occupancy_logistic_normal_base_sd"]),
        "occupancy_effect_standardization_rule": (
            "multiples of the reference donor per-component occupancy "
            "logistic-normal base SD on the logit scale"
        ),
        "fate_effect_standardized": float(
            scale_contract["fate_effect_standardized"]
        ),
        "fate_condition_logit_effect": float(
            scale_contract["fate_effect_standardized"]
        )
        * float(scale_contract["fate_logistic_normal_base_sd"]),
        "fate_effect_standardization_rule": (
            "multiples of the reference donor per-component fate "
            "logistic-normal base SD on the logit scale"
        ),
        "occupancy_pseudocount": float(scale_contract["occupancy_pseudocount"]),
        "fate_pseudocount": float(scale_contract["fate_pseudocount"]),
        "occupancy_logistic_normal_base_sd": float(
            scale_contract["occupancy_logistic_normal_base_sd"]
        ),
        "occupancy_signature_logit_scale": float(
            scale_contract["occupancy_signature_logit_scale"]
        ),
        "fate_logistic_normal_base_sd": float(
            scale_contract["fate_logistic_normal_base_sd"]
        ),
        "fate_signature_logit_scale": float(
            scale_contract["fate_signature_logit_scale"]
        ),
        "pooled_log_expression_dispersion_median": pooled_log_expression_dispersion,
        "noise_multiplier": noise_multiplier,
        "parameter_sources": {
            "donor_covariates": (
                "design.canonical_formal_design via shared canonical T21 builder; "
                "PCW primary, technical batch omitted_not_identifiable, sex sensitivity-only"
            ),
            "sensitivity_covariates": (
                "design.donor_rows sex_signature rank<=1 plus batch_signature "
                "rank<=2; injected only in the covariate-condition stress scenario "
                "and excluded from the primary reduced design"
            ),
            "precision": (
                "raw integer layers[counts] design.donor_rows.analysis_cell_count+"
                "fixed_grid; counts are not used as the expression response"
            ),
            "support": "production _select_segment on fixed_grid primary-draw counts",
            "mapping": "trajectory_draw_dispersion",
            "fate": "fate_probability_distribution",
            "fate_precision": (
                "design.donor_rows.analysis_cell_count; this is already the exact "
                "fate-eligible count and is never multiplied by eligible_fraction again"
            ),
            "pathway_dependence": (
                "pathway_structure.pathway_dependence_correlation+"
                "level_1_family_index_by_pathway"
            ),
            "log_expression_dispersion_heteroskedasticity": (
                "pooled_anonymous_log_expression_dispersion joint mean/variance bins "
                "computed from formal log1p(CP10K) X after frozen support; "
                "identity-independent frozen permutation preserving joint pairs"
            ),
            "chr21_panel": "pathway_structure.n_pathways_with_chr21",
            "curve_dependence": "fixed_grid.adjacent_log_count_correlation",
            "projection_noise": "pooled_anonymous_log_expression_dispersion",
        },
    }
    return _json_ready(derived)


def derived_profile_parameters_sha256(parameters: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            _json_ready(dict(parameters)),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _correlated_errors(
    rng: np.random.Generator,
    n_replicates: int,
    n_pathways: int,
    n_donors: int,
    *,
    residual_sd: float,
    pathway_correlation: float,
    n_curve_bins: int,
    bin_ar1_correlation: float,
    projection_kind: str,
    observed_case: np.ndarray,
    case_variance_ratio: float = 1.0,
    extreme_donor_noise_scale: float = 1.0,
    donor_noise_scale: Sequence[float] | None = None,
    donor_bin_available: np.ndarray | None = None,
    donor_bin_cell_count: np.ndarray | None = None,
) -> np.ndarray:
    grid = (np.arange(n_curve_bins, dtype=float) + 0.5) / n_curve_bins
    logistic = 1.0 / (1.0 + np.exp(-(grid - 0.5) / 0.08))
    if projection_kind == "amplitude":
        basis = logistic
    elif projection_kind == "onset":
        basis = logistic * (1.0 - logistic)
        basis -= np.mean(basis)
    elif projection_kind == "duration":
        basis = np.gradient(logistic * (1.0 - logistic), grid)
    else:
        raise ValueError("projection_kind must be amplitude, onset, or duration")
    basis /= np.linalg.norm(basis)
    indices = np.arange(n_curve_bins)
    bin_covariance = bin_ar1_correlation ** np.abs(indices[:, None] - indices[None, :])
    cholesky = np.linalg.cholesky(bin_covariance)
    if donor_bin_available is None or donor_bin_cell_count is None:
        projection_weights = np.repeat(basis[None, :], n_donors, axis=0)
        information_scale = np.ones(n_donors, dtype=float)
    else:
        available = np.asarray(donor_bin_available, dtype=bool)
        cell_count = np.asarray(donor_bin_cell_count, dtype=float)
        expected_shape = (n_donors, n_curve_bins)
        if available.shape != expected_shape or cell_count.shape != expected_shape:
            raise ValueError("Primary-draw donor-bin masks/counts have the wrong shape")
        if np.any(cell_count < 0) or not np.array_equal(available, cell_count > 0):
            raise ValueError("Primary-draw donor-bin masks must equal cell_count > 0")
        if np.any(available.sum(axis=1) < 2):
            raise ValueError("Every donor requires at least two primary-draw bins")
        positive = cell_count[cell_count > 0]
        median_bin_count = float(np.median(positive))
        precision = np.sqrt(
            np.divide(
                cell_count,
                median_bin_count,
                out=np.zeros_like(cell_count),
                where=available,
            )
        )
        projection_weights = basis[None, :] * precision * available
        donor_total = cell_count.sum(axis=1)
        median_total = float(np.median(donor_total[donor_total > 0]))
        information_scale = (
            available.mean(axis=1)
            * np.divide(
                donor_total,
                median_total,
                out=np.zeros_like(donor_total),
                where=donor_total > 0,
            )
        )
        information_scale = np.clip(information_scale, 0.05, None)
    projected_variance = np.einsum(
        "nb,bc,nc->n", projection_weights, bin_covariance, projection_weights
    )
    if np.any(projected_variance <= np.finfo(float).eps):
        raise ValueError("A primary-draw mask makes the curve projection undefined")
    common_curves = (
        rng.normal(size=(n_replicates, 1, n_donors, n_curve_bins)) @ cholesky.T
    )
    independent_curves = (
        rng.normal(size=(n_replicates, n_pathways, n_donors, n_curve_bins)) @ cholesky.T
    )
    normalizer = np.sqrt(projected_variance * information_scale)
    common = np.einsum("rpnb,nb->rpn", common_curves, projection_weights) / normalizer[
        None, None, :
    ]
    independent = np.einsum(
        "rpnb,nb->rpn", independent_curves, projection_weights
    ) / normalizer[None, None, :]
    errors = residual_sd * (
        math.sqrt(pathway_correlation) * common
        + math.sqrt(1.0 - pathway_correlation) * independent
    )
    errors[:, :, observed_case.astype(bool)] *= math.sqrt(case_variance_ratio)
    if donor_noise_scale is not None:
        donor_scale = np.asarray(donor_noise_scale, dtype=float)
        if donor_scale.shape != (n_donors,) or np.any(~np.isfinite(donor_scale)) or np.any(
            donor_scale <= 0
        ):
            raise ValueError("donor_noise_scale must contain one positive value per donor")
        errors *= donor_scale[None, None, :]
    if extreme_donor_noise_scale != 1.0:
        errors[:, :, -1] *= extreme_donor_noise_scale
    return errors


def _simulate_projection_values(
    rng: np.random.Generator,
    n_replicates: int,
    n_pathways: int,
    age: np.ndarray,
    observed_case: np.ndarray,
    *,
    residual_sd: float,
    pathway_correlation: float,
    n_curve_bins: int,
    bin_ar1_correlation: float,
    projection_kind: str,
    covariate_slope_sd: float,
    case_variance_ratio: float = 1.0,
    extreme_donor_noise_scale: float = 1.0,
    condition_effects: Sequence[float] | None = None,
    covariates: np.ndarray | None = None,
    donor_noise_scale: Sequence[float] | None = None,
    donor_bin_available: np.ndarray | None = None,
    donor_bin_cell_count: np.ndarray | None = None,
) -> np.ndarray:
    errors = _correlated_errors(
        rng,
        n_replicates,
        n_pathways,
        len(age),
        residual_sd=residual_sd,
        pathway_correlation=pathway_correlation,
        n_curve_bins=n_curve_bins,
        bin_ar1_correlation=bin_ar1_correlation,
        projection_kind=projection_kind,
        observed_case=observed_case,
        case_variance_ratio=case_variance_ratio,
        extreme_donor_noise_scale=extreme_donor_noise_scale,
        donor_noise_scale=donor_noise_scale,
        donor_bin_available=donor_bin_available,
        donor_bin_cell_count=donor_bin_cell_count,
    )
    covariate_values = np.asarray(age, dtype=float)[:, None] if covariates is None else np.asarray(
        covariates, dtype=float
    )
    if covariate_values.ndim == 1:
        covariate_values = covariate_values[:, None]
    if covariate_values.ndim != 2 or covariate_values.shape[0] != len(age):
        raise ValueError("covariates must have one row per donor")
    slopes = rng.normal(
        0.0,
        covariate_slope_sd / math.sqrt(covariate_values.shape[1]),
        size=(n_replicates, n_pathways, covariate_values.shape[1]),
    )
    values = errors + np.einsum("rpc,nc->rpn", slopes, covariate_values)
    if condition_effects is not None:
        effects = np.asarray(condition_effects, dtype=float)
        if effects.shape != (n_pathways,):
            raise ValueError("condition_effects must have one value per pathway")
        values += effects[None, :, None] * observed_case[None, None, :]
    return values


def _softmax_vector(values: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=float) - float(np.max(values))
    exponential = np.exp(centered)
    return exponential / exponential.sum()


def _simulate_profile_occupancy_counts(
    rng: np.random.Generator,
    n_replicates: int,
    baseline_counts: np.ndarray,
    regulation_available: np.ndarray,
    observed_case: np.ndarray,
    *,
    donor_noise_scale: Sequence[float],
    donor_signature: Sequence[float],
    condition_effect: float,
    logistic_normal_base_sd: float,
    signature_logit_scale: float,
    min_cells_per_available_bin: int,
) -> np.ndarray:
    """Generate one shared state-population draw for occupancy and regulation."""
    baseline = np.asarray(baseline_counts, dtype=int)
    available = np.asarray(regulation_available, dtype=bool)
    condition = np.asarray(observed_case, dtype=float)
    noise_scale = np.asarray(donor_noise_scale, dtype=float)
    signature = np.asarray(donor_signature, dtype=float)
    if (
        baseline.ndim != 2
        or available.shape != baseline.shape
        or condition.shape != (baseline.shape[0],)
        or noise_scale.shape != condition.shape
        or signature.shape != condition.shape
        or np.any(baseline < 0)
        or np.any(available & (baseline <= 0))
        or not np.isfinite(noise_scale).all()
        or np.any(noise_scale <= 0)
        or not np.isfinite(signature).all()
        or not math.isfinite(float(logistic_normal_base_sd))
        or float(logistic_normal_base_sd) <= 0
        or not math.isfinite(float(signature_logit_scale))
        or float(signature_logit_scale) <= 0
    ):
        raise ValueError("Occupancy DGP arrays are not aligned with fixed support")
    n_donors, n_bins = baseline.shape
    contrast = np.linspace(-1.0, 1.0, n_bins)
    contrast -= np.mean(contrast)
    contrast /= float(np.std(contrast))
    result = np.empty((n_replicates, n_donors, n_bins), dtype=int)
    if int(min_cells_per_available_bin) < 1:
        raise ValueError("min_cells_per_available_bin must be positive")
    minimum = int(min_cells_per_available_bin)
    donor_totals = baseline.sum(axis=1)
    # Randomly reassign each anonymous composition template together with its
    # noise scale and detector signature on every replicate. This preserves
    # their empirical joint distribution without carrying a real
    # condition-associated occupancy pattern into a null DGP.
    for replicate in range(n_replicates):
        template_order = rng.permutation(n_donors)
        for donor in range(n_donors):
            total = int(donor_totals[donor])
            recipient_available = available[donor]
            required = minimum * int(recipient_available.sum())
            if total < required or required < minimum:
                raise ValueError(
                    "Occupancy DGP donor totals are too small for fixed support"
                )
            source_donor = int(template_order[donor])
            template = baseline[source_donor]
            baseline_probability = (template + 0.5) / (
                template.sum() + 0.5 * n_bins
            )
            # The recipient donor's frozen, outcome-blind support is the shared
            # >=minimum threshold contract. Low-support bins may still contain
            # 0..minimum-1 cells for the occupancy composition, while regulation
            # treats them as unavailable exactly as the production selector does.
            baseline_probability[~recipient_available] = 0.0
            logistic_normal = rng.normal(
                0.0,
                float(logistic_normal_base_sd) * noise_scale[source_donor],
                size=n_bins,
            )
            logistic_normal -= np.mean(logistic_normal)
            probability = _softmax_vector(
                np.log(np.clip(baseline_probability, 1e-300, None))
                + logistic_normal
                + float(signature_logit_scale)
                * signature[source_donor]
                * contrast
                + float(condition_effect) * condition[donor] * contrast
            )
            probability[~recipient_available] = 0.0
            probability /= probability.sum()
            low_support = np.minimum(template, minimum - 1).astype(int, copy=True)
            low_support[recipient_available] = 0
            low_support_budget = total - required
            if int(low_support.sum()) > low_support_budget:
                scaled = low_support * (low_support_budget / float(low_support.sum()))
                low_support = np.floor(scaled).astype(int)
                remainder = low_support_budget - int(low_support.sum())
                if remainder:
                    candidates = np.flatnonzero(
                        (~recipient_available) & (low_support < minimum - 1)
                    )
                    for index in rng.permutation(candidates)[:remainder]:
                        low_support[int(index)] += 1
            draw = low_support + rng.multinomial(
                total - required - int(low_support.sum()), probability
            )
            draw[recipient_available] += minimum
            result[replicate, donor, :] = draw
    if not np.array_equal(
        result >= minimum, np.broadcast_to(available, result.shape)
    ):
        raise RuntimeError("Occupancy draw violated the frozen support threshold")
    return result


def _simulate_profile_fate_masses(
    rng: np.random.Generator,
    n_replicates: int,
    baseline_probability: Sequence[float],
    denominators: Sequence[int],
    donor_dispersion_scale: Sequence[float],
    donor_signature: Sequence[float],
    observed_case: np.ndarray,
    *,
    condition_effect: float,
    logistic_normal_base_sd: float,
    signature_logit_scale: float,
) -> np.ndarray:
    baseline = np.asarray(baseline_probability, dtype=float)
    denominator = np.asarray(denominators, dtype=int)
    dispersion = np.asarray(donor_dispersion_scale, dtype=float)
    signature = np.asarray(donor_signature, dtype=float)
    condition = np.asarray(observed_case, dtype=float)
    if (
        baseline.ndim != 1
        or len(baseline) < 2
        or np.any(baseline <= 0)
        or not np.isclose(baseline.sum(), 1.0)
        or denominator.shape != condition.shape
        or dispersion.shape != condition.shape
        or signature.shape != condition.shape
        or not np.isfinite(dispersion).all()
        or np.any(dispersion <= 0)
        or not np.isfinite(signature).all()
        or np.any(denominator < 2)
        or not math.isfinite(float(logistic_normal_base_sd))
        or float(logistic_normal_base_sd) <= 0
        or not math.isfinite(float(signature_logit_scale))
        or float(signature_logit_scale) <= 0
    ):
        raise ValueError("Fate DGP arrays are invalid")
    contrast = np.linspace(-1.0, 1.0, len(baseline))
    contrast -= np.mean(contrast)
    contrast /= float(np.std(contrast))
    result = np.empty(
        (n_replicates, len(denominator), len(baseline)), dtype=float
    )
    for replicate in range(n_replicates):
        # Dispersion and detector-signature templates are reassigned jointly and
        # independently of condition on every replicate.
        template_order = rng.permutation(len(dispersion))
        reassigned_dispersion = dispersion[template_order]
        reassigned_signature = signature[template_order]
        for donor in range(len(denominator)):
            logistic_normal = rng.normal(
                0.0,
                float(logistic_normal_base_sd) * reassigned_dispersion[donor],
                size=len(baseline),
            )
            logistic_normal -= np.mean(logistic_normal)
            probability = _softmax_vector(
                np.log(baseline)
                + logistic_normal
                + float(signature_logit_scale)
                * reassigned_signature[donor]
                * contrast
                + float(condition_effect) * condition[donor] * contrast
            )
            result[replicate, donor, :] = rng.multinomial(
                int(denominator[donor]), probability
            )
    return result


def _chr21_gene_projection_contract(
    design: Mapping[str, Any],
) -> dict[str, Any]:
    """Construct total and trans pathway projections from anonymous gene weights."""
    membership = [
        np.asarray(values, dtype=int)
        for values in design["pathway_member_feature_indices"]
    ]
    chr21 = np.asarray(design["chr21_member_feature_mask"], dtype=bool)
    eligible = np.asarray(design["trans_pathway_eligible_mask"], dtype=bool)
    n_pathways = len(membership)
    if eligible.shape != (n_pathways,) or chr21.ndim != 1:
        raise ValueError("Gene-level chr21 projection topology is invalid")
    total_weights = np.zeros((n_pathways, len(chr21)), dtype=float)
    trans_weights = np.zeros((int(eligible.sum()), len(chr21)), dtype=float)
    trans_source = np.flatnonzero(eligible)
    total_effect = np.zeros(n_pathways, dtype=float)
    for pathway, indices in enumerate(membership):
        if not len(indices):
            raise ValueError("Anonymous pathway membership may not be empty")
        total_weights[pathway, indices] = 1.0 / len(indices)
        total_effect[pathway] = float(np.mean(chr21[indices]))
    for local, pathway in enumerate(trans_source):
        indices = membership[int(pathway)]
        trans_indices = indices[~chr21[indices]]
        if not len(trans_indices):  # pragma: no cover - protected by profile validation
            raise RuntimeError("Eligible trans projection has no non-chr21 members")
        trans_weights[local, trans_indices] = 1.0 / len(trans_indices)
    weights = np.vstack([total_weights, trans_weights])
    covariance = weights @ weights.T
    projection_sd = np.sqrt(np.diag(covariance))
    if np.any(projection_sd <= 0):
        raise ValueError("Gene-level pathway projection has zero variance")
    correlation = covariance / np.outer(projection_sd, projection_sd)
    correlation = 0.5 * (correlation + correlation.T)
    # Numerical clipping retains the exact linear-projection PSD contract.
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    correlation = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    scale = np.sqrt(np.diag(correlation))
    correlation /= np.outer(scale, scale)
    projection_sd /= np.median(projection_sd[:n_pathways])
    maximum_total_effect = float(np.max(total_effect))
    if maximum_total_effect <= 0:
        raise ValueError("Chr21 projection has no total-pathway positive control")
    positive_control_pathway_index = int(np.argmax(total_effect))
    standardized_total_effect = (
        total_effect / maximum_total_effect
    ) * float(design["chr21_cis_effect_standardized"])
    pathway_scale = np.asarray(design["pathway_noise_scale"], dtype=float)
    if pathway_scale.shape != (n_pathways,) or np.any(pathway_scale <= 0):
        raise ValueError("Chr21 projection pathway noise scales are invalid")
    total_residual_scale = (
        float(design["regulation_projection_residual_sd"])
        * pathway_scale
        * projection_sd[:n_pathways]
    )
    return {
        "dependence_correlation": correlation,
        "noise_scale": projection_sd,
        "total_condition_effect": standardized_total_effect * total_residual_scale,
        "total_condition_effect_standardized": standardized_total_effect,
        "total_residual_scale": total_residual_scale,
        "trans_source_pathway_index": trans_source,
        "positive_control_pathway_index": positive_control_pathway_index,
        "positive_control_selection_rule": (
            "largest anonymous chr21-member fraction, then lowest pathway index"
        ),
    }


def _run_profile_component_detector_batch(
    responses: np.ndarray,
    plan: ArrayFreedmanLanePlan,
    *,
    component: str,
    axis_left: Sequence[float],
    axis_right: Sequence[float],
    alpha: float,
) -> np.ndarray:
    """Calibrate component positives through the production decomposition path."""
    values = np.asarray(responses, dtype=float)
    if values.ndim != 4:
        raise ValueError("Component detector responses require replicate/donor/axis/feature")
    feature_names = [f"F{index:03d}" for index in range(values.shape[3])]
    axis_ids = np.arange(values.shape[2], dtype=int)
    calibrated = calibrate_compositional_component_arrays_batch(
        values,
        feature_names,
        component=component,
        axis_ids=axis_ids,
        axis_left=axis_left,
        axis_right=axis_right,
        reduced_design=plan.reduced_design,
        condition=plan.condition,
        null_mappings=plan.null_mappings,
        statistic="max_absolute_effect",
        tail="greater",
        calibration_scale="studentized",
        alpha=float(alpha),
        mapping_batch_size=16,
    )
    return np.any(calibrated["component_maxT_reject"], axis=1)


def _simulate_profile_curve_scores(
    rng: np.random.Generator,
    n_replicates: int,
    n_pathways: int,
    *,
    covariates: np.ndarray,
    observed_case: np.ndarray,
    donor_bin_available: np.ndarray,
    donor_bin_cell_count: np.ndarray,
    residual_sd: float,
    pathway_correlation: float,
    pathway_dependence_correlation: np.ndarray | None = None,
    pathway_noise_scale: Sequence[float] | None = None,
    mapping_shift_by_donor: Sequence[float] | None = None,
    bin_ar1_correlation: float,
    covariate_slope_sd: float,
    sensitivity_covariates: np.ndarray | None = None,
    sensitivity_slope_sd: float = 0.0,
    donor_noise_scale: Sequence[float],
    condition_effects: Sequence[float] | None = None,
    effect_profile: Sequence[float] | None = None,
    baseline_curves: np.ndarray | None = None,
    random_streams: Mapping[str, np.random.Generator] | None = None,
    precision_reference_count: float | None = None,
) -> np.ndarray:
    streams = (
        {name: rng for name in _CURVE_RANDOM_STREAM_NAMES}
        if random_streams is None
        else dict(random_streams)
    )
    if any(name not in streams for name in _CURVE_RANDOM_STREAM_NAMES):
        raise ValueError("Profile curve simulation random streams are incomplete")
    covariate_array = np.asarray(covariates, dtype=float)
    condition = np.asarray(observed_case, dtype=float)
    available = np.asarray(donor_bin_available, dtype=bool)
    cell_count = np.asarray(donor_bin_cell_count, dtype=float)
    donor_scale = np.asarray(donor_noise_scale, dtype=float)
    n_donors, n_bins = available.shape
    if (
        covariate_array.shape[0] != n_donors
        or condition.shape != (n_donors,)
        or donor_scale.shape != (n_donors,)
    ):
        raise ValueError("Profile curve simulation arrays are not donor-grid aligned")
    if cell_count.ndim == 2:
        if cell_count.shape != available.shape or not np.array_equal(
            available, cell_count > 0
        ):
            raise ValueError("Profile curve counts differ from the frozen availability")
        cell_count_batch = np.broadcast_to(
            cell_count[None, :, :], (n_replicates, n_donors, n_bins)
        )
    elif cell_count.ndim == 3:
        if cell_count.shape != (n_replicates, n_donors, n_bins) or not np.all(
            (cell_count > 0) == available[None, :, :]
        ):
            raise ValueError(
                "Replicate-specific curve counts differ from the frozen availability"
            )
        cell_count_batch = cell_count
    else:
        raise ValueError("Profile curve counts must be donor-grid or replicate-donor-grid")
    indices = np.arange(n_bins)
    covariance = bin_ar1_correlation ** np.abs(indices[:, None] - indices[None, :])
    cholesky = np.linalg.cholesky(covariance)
    independent = streams["independent"].normal(
        size=(n_replicates, n_donors, n_bins, n_pathways)
    )
    independent = np.einsum("rdbp,bc->rdcp", independent, cholesky.T)
    if pathway_dependence_correlation is None:
        common = streams["common"].normal(
            size=(n_replicates, n_donors, n_bins, 1)
        )
        common = np.einsum("rdbp,bc->rdcp", common, cholesky.T)
        errors = residual_sd * (
            math.sqrt(pathway_correlation) * common
            + math.sqrt(1.0 - pathway_correlation) * independent
        )
    else:
        dependence = np.asarray(pathway_dependence_correlation, dtype=float)
        if dependence.shape != (n_pathways, n_pathways):
            raise ValueError("Pathway-dependence matrix is not pathway aligned")
        eigenvalues, eigenvectors = np.linalg.eigh(dependence)
        if float(np.min(eigenvalues)) < -1e-8:
            raise ValueError("Pathway-dependence matrix must be positive semidefinite")
        factor = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
        errors = residual_sd * np.einsum(
            "rdbp,qp->rdbq", independent, factor
        )
    positive = cell_count_batch[cell_count_batch > 0]
    median_count = (
        float(np.median(positive))
        if precision_reference_count is None
        else float(precision_reference_count)
    )
    if not math.isfinite(median_count) or median_count <= 0:
        raise ValueError("Curve precision reference count must be positive")
    bin_precision_scale = np.sqrt(
        np.divide(
            median_count,
            cell_count_batch,
            out=np.zeros_like(cell_count_batch),
            where=available[None, :, :],
        )
    )
    errors *= donor_scale[None, :, None, None]
    errors *= bin_precision_scale[:, :, :, None]
    if pathway_noise_scale is not None:
        pathway_scale = np.asarray(pathway_noise_scale, dtype=float)
        if pathway_scale.shape != (n_pathways,) or np.any(pathway_scale <= 0):
            raise ValueError("Pathway noise scale must be positive and pathway aligned")
        errors *= pathway_scale[None, None, None, :]
    nuisance = streams["nuisance"].normal(
        0.0,
        covariate_slope_sd / math.sqrt(covariate_array.shape[1]),
        size=(n_replicates, covariate_array.shape[1], n_bins, n_pathways),
    )
    scores = errors + np.einsum("dc,rcbp->rdbp", covariate_array, nuisance)
    if sensitivity_covariates is not None:
        sensitivity = np.asarray(sensitivity_covariates, dtype=float)
        if (
            sensitivity.ndim != 2
            or sensitivity.shape[0] != n_donors
            or not np.isfinite(sensitivity).all()
            or sensitivity.shape[1] < 1
            or not math.isfinite(float(sensitivity_slope_sd))
            or float(sensitivity_slope_sd) <= 0
        ):
            raise ValueError("Sensitivity signature covariates are invalid")
        sensitivity_slopes = streams["sensitivity"].normal(
            0.0,
            float(sensitivity_slope_sd) / math.sqrt(sensitivity.shape[1]),
            size=(n_replicates, sensitivity.shape[1], n_bins, n_pathways),
        )
        scores += np.einsum(
            "dc,rcbp->rdbp", sensitivity, sensitivity_slopes
        )
    if baseline_curves is not None:
        baseline = np.asarray(baseline_curves, dtype=float)
        if baseline.shape != (n_bins, n_pathways) or not np.isfinite(
            baseline
        ).all():
            raise ValueError("Baseline curves must be finite and bin/pathway aligned")
        scores += baseline[None, None, :, :]
    if mapping_shift_by_donor is not None:
        mapping_shift = np.asarray(mapping_shift_by_donor, dtype=float)
        if mapping_shift.shape != (n_donors,) or not np.isfinite(mapping_shift).all():
            raise ValueError("Mapping-shift vector must be finite and donor aligned")
        latent = streams["mapping"].normal(
            size=(n_replicates, n_bins, n_pathways)
        )
        latent = np.einsum("rbp,bc->rcp", latent, cholesky.T) * residual_sd
        for donor_index in range(n_donors):
            positions = np.clip(
                np.arange(n_bins, dtype=float)
                + condition[donor_index] * mapping_shift[donor_index] * n_bins,
                0.0,
                n_bins - 1.0,
            )
            lower = np.floor(positions).astype(int)
            upper = np.ceil(positions).astype(int)
            weight = positions - lower
            warped = (
                latent[:, lower, :] * (1.0 - weight)[None, :, None]
                + latent[:, upper, :] * weight[None, :, None]
            )
            scores[:, donor_index, :, :] += warped
    if condition_effects is not None:
        effects = np.asarray(condition_effects, dtype=float)
        if effects.shape != (n_pathways,):
            raise ValueError("Curve condition effects must have one value per pathway")
        profile = (
            np.ones(n_bins, dtype=float)
            if effect_profile is None
            else np.asarray(effect_profile, dtype=float)
        )
        if profile.shape != (n_bins,):
            raise ValueError("Curve effect profile must have one value per fixed bin")
        scores += (
            condition[None, :, None, None]
            * profile[None, None, :, None]
            * effects[None, None, None, :]
        )
    return np.where(available[None, :, :, None], scores, np.nan)


def _run_shared_curve_kernel_batch(
    scores: np.ndarray,
    plan: ArrayFreedmanLanePlan,
    *,
    alpha: float,
    family_index: Sequence[int] | None = None,
) -> list[dict[str, np.ndarray]]:
    batch = run_array_freedman_lane_calibration_batch(
        scores,
        plan,
        statistic="max_absolute_effect",
        tail="greater",
        calibration_scale="studentized",
        alpha=alpha,
        family_index=family_index,
        mapping_batch_size=16,
    )
    shared_keys = {"residual_df_by_bin"}
    return [
        {
            key: value if key in shared_keys else np.asarray(value)[index]
            for key, value in batch.items()
        }
        for index in range(scores.shape[0])
    ]


def _formal_event_decisions(
    result: Mapping[str, np.ndarray],
    bin_centers: Sequence[float],
    *,
    alpha: float,
    min_consecutive: int,
    min_duration_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the production consecutive-window event rule after global control."""
    beta = np.asarray(result["beta_curve"], dtype=float)
    t_curve = np.asarray(result["calibration_curve"], dtype=float)
    residual_df = np.asarray(result["residual_df_by_bin"], dtype=int)
    globally_rejected = np.asarray(
        result["production_family_hierarchical_reject"], dtype=bool
    )
    if beta.shape != t_curve.shape or beta.shape[0] != len(bin_centers):
        raise ValueError("Formal event decision arrays are not aligned")
    critical = student_t.ppf(1.0 - float(alpha) / 2.0, residual_df)
    pointwise = np.abs(t_curve) >= critical[:, None]
    onset_call = np.zeros(beta.shape[1], dtype=bool)
    duration_call = np.zeros(beta.shape[1], dtype=bool)
    for pathway in range(beta.shape[1]):
        timing = stable_event_timing(
            np.asarray(bin_centers, dtype=float),
            beta[:, pathway],
            pointwise[:, pathway],
            min_consecutive=int(min_consecutive),
        )
        has_onset = bool(
            np.isfinite(timing["activation_onset"])
            or np.isfinite(timing["suppression_onset"])
        )
        onset_call[pathway] = globally_rejected[pathway] and has_onset
        grid_span = float(np.max(bin_centers) - np.min(bin_centers))
        duration_call[pathway] = (
            globally_rejected[pathway]
            and has_onset
            and float(timing["duration"])
            >= float(min_duration_fraction) * grid_span
        )
    return onset_call, duration_call


def _curve_effect_profile(kind: str, n_bins: int) -> np.ndarray:
    """Return a frozen unit-scale curve perturbation for one estimand."""
    grid = (np.arange(int(n_bins), dtype=float) + 0.5) / int(n_bins)
    if kind == "amplitude":
        return np.ones_like(grid)
    logistic = 1.0 / (1.0 + np.exp(-(grid - 0.5) / 0.08))
    if kind == "onset":
        profile = logistic * (1.0 - logistic)
    elif kind == "duration":
        profile = np.abs(grid - 0.5) * logistic * (1.0 - logistic)
    else:
        raise ValueError("Curve effect-profile kind must be amplitude, onset, or duration")
    maximum = float(np.max(np.abs(profile)))
    if maximum <= 0:  # pragma: no cover - protected by the fixed grid construction
        raise RuntimeError("Curve effect profile is degenerate")
    return profile / maximum


def _onset_logistic_profiles(
    bin_centers: Sequence[float], shift: float
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.asarray(bin_centers, dtype=float)
    if grid.ndim != 1 or len(grid) < 2 or not np.all(np.diff(grid) > 0):
        raise ValueError("Onset recovery requires increasing fixed-grid centers")
    control = 1.0 / (1.0 + np.exp(-(grid - 0.5) / 0.08))
    case = 1.0 / (1.0 + np.exp(-(grid - (0.5 + float(shift))) / 0.08))
    return control, case


def _estimate_half_rise_onset(
    bin_centers: Sequence[float], curve: Sequence[float]
) -> float:
    return estimate_half_rise_onset(
        np.asarray(bin_centers, dtype=float), np.asarray(curve, dtype=float)
    )


def _make_profile_shared_curve_plan(
    spec: Mapping[str, Any],
    design: Mapping[str, Any],
    *,
    seed: int,
) -> ArrayFreedmanLanePlan:
    expected_seed = int(spec["inference"]["residual_mapping_seed"])
    if seed != expected_seed:
        raise ValueError("Shared residual mapping seed differs from the frozen spec")
    return make_array_freedman_lane_plan(
        reduced_design=np.asarray(design["support_reduced_design"], dtype=float),
        condition=np.asarray(design["observed_assignment"], dtype=bool),
        available=np.asarray(design["primary_draw_available_mask"], dtype=bool),
        widths=np.asarray(design["bin_widths"], dtype=float),
        max_exact_permutations=int(
            spec["inference"]["max_exhaustive_residual_mappings"]
        ),
        permutation_mode=str(spec["inference"]["residual_reference_mode"]),
        n_permutations=int(
            spec["inference"]["monte_carlo_residual_mappings"]
        ),
        seed=seed,
    )


def _scenario_replicates(
    scenario: str,
    n_replicates: int,
    spec: Mapping[str, Any],
    *,
    seed: int,
    chunk_size: int,
    design_profile: Mapping[str, Any] | None = None,
    derived_parameters: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    confounded = scenario == "covariate_condition_association"
    if str(spec.get("schema_version", "")).startswith("2."):
        if design_profile is None:
            raise ValueError("Runner v2 requires an outcome-blind design profile")
        design = dict(
            derived_parameters
            if derived_parameters is not None
            else derive_profile_simulation_parameters(spec, design_profile)
        )
        n_controls = int(design["n_disomy_controls"])
        n_cases = int(design["n_t21_cases"])
        n_pathways = int(design["n_pathways"])
        n_cis = int(design["n_chr21_cis_pathways"])
        alpha = float(design["alpha"])
        observed_case = np.asarray(design["observed_assignment"], dtype=float)
        covariates = np.asarray(design["covariate_matrix"], dtype=float)
        sensitivity_signature_matrix = np.asarray(
            design["sensitivity_signature_matrix"], dtype=float
        )
        sensitivity_covariates = (
            sensitivity_signature_matrix
            if confounded and sensitivity_signature_matrix.shape[1] > 0
            else None
        )
        age = covariates[:, 0]
        plan = _make_covariate_contrast_plan(covariates, observed_case)
        donor_noise = np.asarray(design["donor_noise_scale"], dtype=float)
        donor_bin_available = np.asarray(
            design["primary_draw_available_mask"], dtype=bool
        )
        donor_bin_cell_count = np.asarray(design["primary_draw_cell_count"], dtype=float)
        shared_curve_plan = _make_profile_shared_curve_plan(
            spec, design, seed=int(spec["inference"]["residual_mapping_seed"])
        )
        pathway_family_index = np.asarray(
            design["level_1_family_index_by_pathway"], dtype=int
        )
        cis_pathway_mask = np.asarray(design["chr21_pathway_mask"], dtype=bool)
        selected_bin_left = np.asarray(design["selected_bin_left"], dtype=float)
        selected_bin_right = np.asarray(design["selected_bin_right"], dtype=float)
        fate_baseline_probability = np.asarray(
            design["fate_baseline_probability"], dtype=float
        )
        fate_eligible_denominator = np.asarray(
            design["fate_eligible_denominator"], dtype=int
        )
        if confounded:
            extreme_index = int(np.argmax(donor_noise))
            donor_noise[extreme_index] *= float(design["extreme_donor_noise_scale"])
        ratio = 1.0
        extreme_scale = 1.0
    else:
        design = spec["design"]
        n_controls = int(design["n_disomy_controls"])
        n_cases = int(design["n_t21_cases"])
        n_pathways = int(design["n_pathways"])
        n_cis = int(design["n_chr21_cis_pathways"])
        alpha = float(design["alpha"])
        age, observed_case = _donor_design(
            n_controls,
            n_cases,
            confounded=confounded,
            case_age_shift_sd=float(design["case_age_shift_sd"]),
        )
        covariates = age[:, None]
        plan = _make_exact_contrast_plan(age, observed_case)
        ratio = 1.0
        if scenario == "occupancy_only":
            ratio = float(design["occupancy_case_variance_ratio"])
        elif scenario == "fate_only":
            ratio = float(design["fate_case_variance_ratio"])
        extreme_scale = (
            float(design["extreme_donor_noise_scale"]) if confounded else 1.0
        )
        donor_noise = None
        donor_bin_available = None
        donor_bin_cell_count = None
        shared_curve_plan = None
        pathway_family_index = None
        cis_pathway_mask = np.zeros(n_pathways, dtype=bool)
        cis_pathway_mask[:n_cis] = True
        selected_bin_left = None
        selected_bin_right = None
        fate_baseline_probability = None
        fate_eligible_denominator = None
        sensitivity_covariates = None
    regulation_effects = np.zeros(n_pathways)
    chr21_projection_contract = None
    if scenario == "chr21_dosage_only":
        if shared_curve_plan is not None:
            chr21_projection_contract = _chr21_gene_projection_contract(design)
            regulation_effects = np.asarray(
                chr21_projection_contract["total_condition_effect"], dtype=float
            )
        else:
            regulation_effects[cis_pathway_mask] = float(
                design["chr21_cis_effect_standardized"]
            )
    mapping_effects = np.zeros(n_pathways)
    mapping_shift_by_donor = None
    if scenario == "trajectory_speed_or_mapping_only":
        if design_profile is not None:
            mapping_dispersion = np.asarray(
                design["trajectory_mapping_dispersion_by_donor"], dtype=float
            )
            positive_mapping = mapping_dispersion[mapping_dispersion > 0]
            mapping_reference = (
                float(np.median(positive_mapping)) if len(positive_mapping) else 1.0
            )
            mapping_shift_by_donor = float(
                design["residual_mapping_shift_sd"]
            ) * np.clip(mapping_dispersion / mapping_reference, 0.25, 4.0)
        else:
            mapping_effects[:] = float(design["residual_mapping_shift_sd"])

    rng = np.random.default_rng(_seed_for(seed, f"scenario:{scenario}"))
    occupancy_rng = (
        np.random.default_rng(_seed_for(seed, f"scenario:{scenario}:occupancy"))
        if shared_curve_plan is not None
        else rng
    )
    fate_rng = (
        np.random.default_rng(_seed_for(seed, f"scenario:{scenario}:fate"))
        if shared_curve_plan is not None
        else rng
    )
    formal_curve_rng = np.random.default_rng(
        _seed_for(seed, f"scenario:{scenario}:formal-curve")
    )
    formal_curve_streams = _curve_random_streams(
        seed, f"scenario:{scenario}:formal-curve"
    )
    frames: list[pd.DataFrame] = []
    t_critical = float(student_t.ppf(0.975, plan.degrees_of_freedom))
    for start in range(0, n_replicates, chunk_size):
        size = min(chunk_size, n_replicates - start)
        occupancy_count_draws = None
        occupancy_responses = None
        fate_responses = None
        if shared_curve_plan is not None:
            occupancy_count_draws = _simulate_profile_occupancy_counts(
                occupancy_rng,
                size,
                np.asarray(design["occupancy_baseline_cell_count"], dtype=int),
                donor_bin_available,
                observed_case,
                donor_noise_scale=np.asarray(
                    design["occupancy_noise_scale"], dtype=float
                ),
                donor_signature=np.asarray(
                    design["occupancy_detector_signature"], dtype=float
                ),
                condition_effect=(
                    float(design["occupancy_condition_logit_effect"])
                    if scenario == "occupancy_only"
                    else 0.0
                ),
                logistic_normal_base_sd=float(
                    design["occupancy_logistic_normal_base_sd"]
                ),
                signature_logit_scale=float(
                    design["occupancy_signature_logit_scale"]
                ),
                min_cells_per_available_bin=int(
                    design["support_selection_contract"][
                        "min_cells_per_donor_bin"
                    ]
                ),
            )
            occupancy_responses = np.stack(
                [
                    occupancy_response_from_counts(
                        occupancy_count_draws[index],
                        pseudocount=float(design["occupancy_pseudocount"]),
                    )[0]
                    for index in range(size)
                ],
                axis=0,
            )
            fate_mass_draws = _simulate_profile_fate_masses(
                fate_rng,
                size,
                fate_baseline_probability,
                fate_eligible_denominator,
                np.asarray(design["fate_noise_scale"], dtype=float),
                np.asarray(design["fate_detector_signature"], dtype=float),
                observed_case,
                condition_effect=(
                    float(design["fate_condition_logit_effect"])
                    if scenario == "fate_only"
                    else 0.0
                ),
                logistic_normal_base_sd=float(
                    design["fate_logistic_normal_base_sd"]
                ),
                signature_logit_scale=float(
                    design["fate_signature_logit_scale"]
                ),
            )
            fate_responses = np.stack(
                [
                    fate_response_from_masses(
                        fate_mass_draws[index],
                        fate_eligible_denominator,
                        pseudocount=float(design["fate_pseudocount"]),
                    )[0]
                    for index in range(size)
                ],
                axis=0,
            )
        regulation = None
        onset = None
        duration = None
        regulation_test: dict[str, np.ndarray] = {}
        if shared_curve_plan is None:
            regulation = _simulate_projection_values(
                rng,
                size,
                n_pathways,
                age,
                observed_case,
                residual_sd=float(design["regulation_projection_residual_sd"]),
                pathway_correlation=float(design["pathway_correlation"]),
                n_curve_bins=int(design["n_curve_bins"]),
                bin_ar1_correlation=float(design["bin_ar1_correlation"]),
                projection_kind="amplitude",
                covariate_slope_sd=float(design["covariate_slope_sd"]),
                case_variance_ratio=ratio,
                extreme_donor_noise_scale=extreme_scale,
                condition_effects=regulation_effects,
                covariates=covariates,
                donor_noise_scale=donor_noise,
                donor_bin_available=donor_bin_available,
                donor_bin_cell_count=donor_bin_cell_count,
            )
            onset = _simulate_projection_values(
                rng,
                size,
                n_pathways,
                age,
                observed_case,
                residual_sd=float(design["timing_projection_residual_sd"]),
                pathway_correlation=float(design["pathway_correlation"]),
                n_curve_bins=int(design["n_curve_bins"]),
                bin_ar1_correlation=float(design["bin_ar1_correlation"]),
                projection_kind="onset",
                covariate_slope_sd=float(design["covariate_slope_sd"]),
                case_variance_ratio=ratio,
                extreme_donor_noise_scale=extreme_scale,
                condition_effects=mapping_effects,
                covariates=covariates,
                donor_noise_scale=donor_noise,
                donor_bin_available=donor_bin_available,
                donor_bin_cell_count=donor_bin_cell_count,
            )
            duration = _simulate_projection_values(
                rng,
                size,
                n_pathways,
                age,
                observed_case,
                residual_sd=float(design["duration_projection_residual_sd"]),
                pathway_correlation=float(design["pathway_correlation"]),
                n_curve_bins=int(design["n_curve_bins"]),
                bin_ar1_correlation=float(design["bin_ar1_correlation"]),
                projection_kind="duration",
                covariate_slope_sd=float(design["covariate_slope_sd"]),
                case_variance_ratio=ratio,
                extreme_donor_noise_scale=extreme_scale,
                covariates=covariates,
                donor_noise_scale=donor_noise,
                donor_bin_available=donor_bin_available,
                donor_bin_cell_count=donor_bin_cell_count,
            )
            regulation_test = _exact_test(regulation, plan, alpha)
        shared_curve_scores = None
        shared_curve_results = None
        shared_curve_coverage = None
        formal_trans_results = None
        if shared_curve_plan is not None:
            curve_counts = (
                np.where(
                    donor_bin_available[None, :, :], occupancy_count_draws, 0
                )
                if occupancy_count_draws is not None
                else donor_bin_cell_count
            )
            if chr21_projection_contract is not None:
                trans_source = np.asarray(
                    chr21_projection_contract["trans_source_pathway_index"],
                    dtype=int,
                )
                joint_effects = np.concatenate(
                    [regulation_effects, np.zeros(len(trans_source), dtype=float)]
                )
                base_pathway_scale = np.asarray(
                    design["pathway_noise_scale"], dtype=float
                )
                joint_noise_scale = np.concatenate(
                    [base_pathway_scale, base_pathway_scale[trans_source]]
                ) * np.asarray(chr21_projection_contract["noise_scale"], dtype=float)
                base_pathway_mean = np.asarray(
                    design["pathway_baseline_mean"], dtype=float
                )
                joint_baseline = np.broadcast_to(
                    np.concatenate(
                        [base_pathway_mean, base_pathway_mean[trans_source]]
                    )[None, :],
                    (int(design["n_curve_bins"]), len(joint_effects)),
                )
                joint_curve_scores = _simulate_profile_curve_scores(
                    formal_curve_rng,
                    size,
                    len(joint_effects),
                    covariates=covariates,
                    observed_case=observed_case,
                    donor_bin_available=donor_bin_available,
                    donor_bin_cell_count=curve_counts,
                    residual_sd=float(design["regulation_projection_residual_sd"]),
                    pathway_correlation=float(design["pathway_correlation"]),
                    pathway_dependence_correlation=np.asarray(
                        chr21_projection_contract["dependence_correlation"],
                        dtype=float,
                    ),
                    pathway_noise_scale=joint_noise_scale,
                    mapping_shift_by_donor=mapping_shift_by_donor,
                    bin_ar1_correlation=float(design["bin_ar1_correlation"]),
                    covariate_slope_sd=float(design["covariate_slope_sd"]),
                    sensitivity_covariates=sensitivity_covariates,
                    sensitivity_slope_sd=float(
                        design["sensitivity_signature_slope_sd"]
                    ),
                    donor_noise_scale=donor_noise,
                    condition_effects=joint_effects,
                    baseline_curves=joint_baseline,
                    random_streams=formal_curve_streams,
                    precision_reference_count=float(
                        design["primary_draw_median_positive_cell_count"]
                    ),
                )
                shared_curve_scores = joint_curve_scores[:, :, :, :n_pathways]
                trans_curve_scores = joint_curve_scores[:, :, :, n_pathways:]
                formal_trans_results = _run_shared_curve_kernel_batch(
                    trans_curve_scores,
                    shared_curve_plan,
                    alpha=alpha,
                    family_index=pathway_family_index[trans_source],
                )
            else:
                shared_curve_scores = _simulate_profile_curve_scores(
                    formal_curve_rng,
                    size,
                    n_pathways,
                    covariates=covariates,
                    observed_case=observed_case,
                    donor_bin_available=donor_bin_available,
                    donor_bin_cell_count=curve_counts,
                    residual_sd=float(design["regulation_projection_residual_sd"]),
                    pathway_correlation=float(design["pathway_correlation"]),
                    pathway_dependence_correlation=np.asarray(
                        design["pathway_dependence_correlation"], dtype=float
                    ),
                    pathway_noise_scale=design["pathway_noise_scale"],
                    mapping_shift_by_donor=mapping_shift_by_donor,
                    bin_ar1_correlation=float(design["bin_ar1_correlation"]),
                    covariate_slope_sd=float(design["covariate_slope_sd"]),
                    sensitivity_covariates=sensitivity_covariates,
                    sensitivity_slope_sd=float(
                        design["sensitivity_signature_slope_sd"]
                    ),
                    donor_noise_scale=donor_noise,
                    condition_effects=regulation_effects,
                    baseline_curves=np.broadcast_to(
                        np.asarray(design["pathway_baseline_mean"], dtype=float)[
                            None, :
                        ],
                        (int(design["n_curve_bins"]), n_pathways),
                    ),
                    random_streams=formal_curve_streams,
                    precision_reference_count=float(
                        design["primary_draw_median_positive_cell_count"]
                    ),
                )
            shared_curve_results = _run_shared_curve_kernel_batch(
                shared_curve_scores,
                shared_curve_plan,
                alpha=alpha,
                family_index=pathway_family_index,
            )
            for key, shared_key in (
                ("p_raw", "p_raw"),
                ("p_maxT", "p_maxT"),
                ("q_by", "q_by"),
                (
                    "maxT_reject",
                    "production_family_hierarchical_reject",
                ),
                ("by_reject", "by_reject"),
            ):
                regulation_test[key] = np.vstack(
                    [result[shared_key] for result in shared_curve_results]
                )
            curve_truth = np.broadcast_to(
                regulation_effects[None, :],
                (int(design["n_curve_bins"]), n_pathways),
            )
            shared_curve_coverage = np.asarray(
                [
                    float(
                        np.mean(
                            np.abs(result["beta_curve"] - curve_truth)
                            <= student_t.ppf(
                                0.975, result["residual_df_by_bin"]
                            )[:, None]
                            * result["standard_error_curve"]
                        )
                    )
                    for result in shared_curve_results
                ],
                dtype=float,
            )
        if shared_curve_plan is None:
            if onset is None or duration is None:  # pragma: no cover - defensive
                raise RuntimeError("Legacy timing projections were not simulated")
            onset_test = _exact_test(onset, plan, alpha)
            duration_test = _exact_test(duration, plan, alpha)
        else:
            onset_test: dict[str, np.ndarray] = {}
            duration_test: dict[str, np.ndarray] = {}
        timing_shared_results = None
        duration_shared_results = None
        if shared_curve_plan is not None:
            timing_shared_results = shared_curve_results
            duration_shared_results = shared_curve_results
            centers = 0.5 * (selected_bin_left + selected_bin_right)
            event_calls = [
                _formal_event_decisions(
                    result,
                    centers,
                    alpha=alpha,
                    min_consecutive=int(
                        spec["inference"]["timing_min_consecutive_windows"]
                    ),
                    min_duration_fraction=float(
                        spec["inference"]["timing_min_duration_fraction"]
                    ),
                )
                for result in shared_curve_results
            ]
            onset_test["maxT_reject"] = np.vstack(
                [calls[0] for calls in event_calls]
            )
            duration_test["maxT_reject"] = np.vstack(
                [calls[1] for calls in event_calls]
            )
        occupancy_detection = None
        fate_detection = None
        if occupancy_responses is not None and fate_responses is not None:
            occupancy_detection = _run_profile_component_detector_batch(
                occupancy_responses,
                shared_curve_plan,
                component="occupancy",
                axis_left=selected_bin_left,
                axis_right=selected_bin_right,
                alpha=alpha,
            )
            fate_detection = _run_profile_component_detector_batch(
                fate_responses,
                shared_curve_plan,
                component="fate",
                axis_left=[0.0],
                axis_right=[1.0],
                alpha=alpha,
            )
        null_mask = np.ones(n_pathways, dtype=bool)
        if scenario == "chr21_dosage_only":
            null_mask[cis_pathway_mask] = False
        false_by = regulation_test["by_reject"][:, null_mask].sum(axis=1)
        total_by = regulation_test["by_reject"].sum(axis=1)
        fdp = np.divide(
            false_by,
            total_by,
            out=np.zeros(size, dtype=float),
            where=total_by > 0,
        )
        if null_mask.all():
            family_false = regulation_test["maxT_reject"].any(axis=1)
            trans_test = None
        else:
            if formal_trans_results is not None:
                trans_test = {
                    "maxT_reject": np.vstack(
                        [
                            result["production_family_hierarchical_reject"]
                            for result in formal_trans_results
                        ]
                    ),
                    "by_reject": np.vstack(
                        [result["by_reject"] for result in formal_trans_results]
                    ),
                }
            elif shared_curve_plan is not None and shared_curve_scores is not None:
                trans_results = _run_shared_curve_kernel_batch(
                    shared_curve_scores[:, :, :, null_mask],
                    shared_curve_plan,
                    alpha=alpha,
                    family_index=pathway_family_index[null_mask],
                )
                trans_test = {
                    "maxT_reject": np.vstack(
                        [
                            result["production_family_hierarchical_reject"]
                            for result in trans_results
                        ]
                    ),
                    "by_reject": np.vstack(
                        [result["by_reject"] for result in trans_results]
                    ),
                }
            else:
                trans_test = _exact_test(regulation[:, null_mask, :], plan, alpha)
            family_false = trans_test["maxT_reject"].any(axis=1)
        if shared_curve_coverage is not None:
            coverage = shared_curve_coverage
        else:
            coverage = np.mean(
                np.abs(regulation_test["beta"] - regulation_effects[None, :])
                <= t_critical * regulation_test["standard_error"],
                axis=1,
            )
        trans_fdp = np.full(size, np.nan)
        cis_detection = np.full(size, np.nan)
        chr21_positive_control_detected = np.full(size, np.nan)
        if scenario == "chr21_dosage_only":
            if trans_test is None:  # pragma: no cover - protected by null_mask
                raise RuntimeError("Trans-only calibration result was not computed")
            trans_discoveries = trans_test["by_reject"].sum(axis=1)
            trans_fdp = (trans_discoveries > 0).astype(float)
            cis_detection = regulation_test["maxT_reject"][:, cis_pathway_mask].mean(
                axis=1
            )
            positive_control_index = (
                int(chr21_projection_contract["positive_control_pathway_index"])
                if chr21_projection_contract is not None
                else int(np.flatnonzero(cis_pathway_mask)[0])
            )
            chr21_positive_control_detected = regulation_test["maxT_reject"][
                :,
                positive_control_index,
            ].astype(float)
        row_data: dict[str, Any] = {
                    "scenario": scenario,
                    "replicate_id": np.arange(start, start + size),
                    "any_maxT_false_rejection": family_false,
                    "by_false_discovery_proportion": fdp,
                    "onset_false_positive": onset_test["maxT_reject"].any(axis=1),
                    "duration_false_positive": duration_test["maxT_reject"].any(axis=1),
                    (
                        "pointwise_curve_coverage_fraction"
                        if design_profile is not None
                        else "confidence_coverage_fraction"
                    ): coverage,
                    "regulation_false_discovery_proportion": fdp,
                    "false_timing_shift": onset_test["maxT_reject"].any(axis=1),
                    "trans_false_discovery_proportion": trans_fdp,
                    "cis_detection_fraction": cis_detection,
                    "chr21_total_positive_control_detected": (
                        chr21_positive_control_detected
                    ),
                    "whole_donor_assignment_space": plan.n_assignments,
                }
        if design_profile is not None:
            profile_hash = str(design_profile["integrity"]["profile_payload_sha256"])
            parameter_hash = derived_profile_parameters_sha256(design)
            row_data["design_profile_payload_sha256"] = profile_hash
            row_data["derived_design_parameters_sha256"] = parameter_hash
            if occupancy_detection is None or fate_detection is None:
                raise RuntimeError("Profile detector results were not computed")
            row_data["occupancy_signal_detected"] = occupancy_detection
            row_data["fate_signal_detected"] = fate_detection
            if (
                shared_curve_plan is None
                or shared_curve_results is None
                or timing_shared_results is None
                or duration_shared_results is None
            ):
                raise RuntimeError("Formal shared donor-curve kernel was not used")
            row_data["formal_regulation_shared_kernel_used"] = True
            row_data["formal_timing_shared_kernel_used"] = True
            row_data["formal_occupancy_fate_decomposition_kernel_used"] = True
            row_data["shared_state_population_draw_used"] = True
            row_data["fixed_20_bin_source_grid_verified"] = bool(
                design["fixed_20_bin_source_grid_verified"]
            )
            row_data["selected_support_design_valid"] = bool(
                design["selected_support_design_valid"]
            )
            row_data["sensitivity_signature_effect_injected"] = bool(
                confounded and sensitivity_covariates is not None
            )
            row_data["sensitivity_signature_matrix_sha256"] = str(
                design["sensitivity_signature_matrix_sha256"]
            )
            row_data["n_sensitivity_signature_components"] = int(
                np.asarray(design["sensitivity_signature_matrix"], dtype=float).shape[1]
            )
            row_data["chr21_gene_level_total_trans_projection_used"] = bool(
                scenario == "chr21_dosage_only"
                and chr21_projection_contract is not None
                and formal_trans_results is not None
            )
            row_data["residual_mapping_space_size"] = int(
                shared_curve_plan.residual_space_size
            )
            row_data["restricted_label_space_size"] = int(
                shared_curve_plan.restricted_label_space_size
            )
            row_data["shared_kernel_availability_mask_sha256"] = (
                shared_curve_plan.availability_mask_sha256
            )
            row_data["residual_reference_actual_mode"] = shared_curve_plan.actual_mode
            row_data["residual_reference_enumeration"] = (
                shared_curve_plan.reference_enumeration
            )
            row_data["residual_null_mappings"] = shared_curve_plan.n_null_mappings
            row_data["residual_p_resolution"] = (
                shared_curve_plan.monte_carlo_p_resolution
            )
        frames.append(pd.DataFrame(row_data))
    return pd.concat(frames, ignore_index=True)


def summarize_scenario_replicates(replicates: pd.DataFrame) -> pd.DataFrame:
    """Recompute all report-level scenario metrics from replicate rows."""
    required = {
        "scenario",
        "replicate_id",
        "any_maxT_false_rejection",
        "by_false_discovery_proportion",
        "onset_false_positive",
        "duration_false_positive",
        "regulation_false_discovery_proportion",
        "false_timing_shift",
        "trans_false_discovery_proportion",
    }
    missing = required - set(replicates)
    if missing:
        raise ValueError(f"Scenario replicate table is missing: {sorted(missing)}")
    coverage_columns = {
        "pointwise_curve_coverage_fraction",
        "confidence_coverage_fraction",
    } & set(replicates)
    if len(coverage_columns) != 1:
        raise ValueError(
            "Scenario replicates must contain exactly one pointwise coverage column"
        )
    coverage_column = next(iter(coverage_columns))
    coverage_metric = (
        "pointwise_curve_coverage"
        if coverage_column == "pointwise_curve_coverage_fraction"
        else "confidence_coverage"
    )
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        group = replicates.loc[replicates["scenario"].eq(scenario)]
        if group.empty:
            raise ValueError(f"Scenario replicate table has no rows for {scenario}")
        if group["replicate_id"].duplicated().any():
            raise ValueError(f"Scenario {scenario} has duplicate replicate IDs")
        row: dict[str, Any] = {
            "scenario": scenario,
            "n_replicates": int(len(group)),
            "empirical_fwer": float(group["any_maxT_false_rejection"].mean()),
            "empirical_fdr": float(group["by_false_discovery_proportion"].mean()),
            "onset_false_positive_rate": float(group["onset_false_positive"].mean()),
            "duration_false_positive_rate": float(
                group["duration_false_positive"].mean()
            ),
            coverage_metric: float(group[coverage_column].mean()),
            "regulation_false_discovery_rate": float(
                group["regulation_false_discovery_proportion"].mean()
            ),
            "false_timing_shift_rate": float(group["false_timing_shift"].mean()),
            "trans_false_discovery_rate": float(
                group["trans_false_discovery_proportion"].mean()
            ),
        }
        cis = group["cis_detection_fraction"].dropna()
        if not cis.empty:
            row["chr21_cis_detection_rate"] = float(cis.mean())
        chr21_positive = group["chr21_total_positive_control_detected"].dropna()
        if not chr21_positive.empty:
            row["chr21_total_positive_control_detection_rate"] = float(
                chr21_positive.mean()
            )
        if {
            "occupancy_signal_detected",
            "fate_signal_detected",
        }.issubset(group.columns):
            row["occupancy_detection_rate"] = float(
                group["occupancy_signal_detected"].mean()
            )
            row["fate_detection_rate"] = float(group["fate_signal_detected"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _power_replicates(
    spec: Mapping[str, Any],
    n_replicates: int,
    *,
    seed: int,
    chunk_size: int,
    kind: str,
    design_profile: Mapping[str, Any] | None = None,
    derived_parameters: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    power = spec["power"]
    if str(spec.get("schema_version", "")).startswith("2."):
        if design_profile is None:
            raise ValueError("Runner v2 power requires a design profile")
        design = dict(
            derived_parameters
            if derived_parameters is not None
            else derive_profile_simulation_parameters(spec, design_profile)
        )
        observed_case = np.asarray(design["observed_assignment"], dtype=float)
        covariates = np.asarray(design["covariate_matrix"], dtype=float)
        age = covariates[:, 0]
        plan = _make_covariate_contrast_plan(covariates, observed_case)
        donor_noise = np.asarray(design["donor_noise_scale"], dtype=float)
        donor_bin_available = np.asarray(
            design["primary_draw_available_mask"], dtype=bool
        )
        donor_bin_cell_count = np.asarray(design["primary_draw_cell_count"], dtype=float)
        shared_curve_plan = _make_profile_shared_curve_plan(
            spec,
            design,
            seed=int(spec["inference"]["residual_mapping_seed"]),
        )
        pathway_family_index = np.asarray(
            design["level_1_family_index_by_pathway"], dtype=int
        )
        target_pathway_index = int(design["power_target_pathway_index"])
        selected_bin_centers = 0.5 * (
            np.asarray(design["selected_bin_left"], dtype=float)
            + np.asarray(design["selected_bin_right"], dtype=float)
        )
    else:
        design = spec["design"]
        n_controls = int(design["n_disomy_controls"])
        n_cases = int(design["n_t21_cases"])
        age, observed_case = _donor_design(n_controls, n_cases, confounded=False)
        covariates = age[:, None]
        plan = _make_exact_contrast_plan(age, observed_case)
        donor_noise = None
        donor_bin_available = None
        donor_bin_cell_count = None
        shared_curve_plan = None
        pathway_family_index = None
        target_pathway_index = 0
        selected_bin_centers = None
    n_pathways = int(design["n_pathways"])
    alpha = float(design["alpha"])
    if kind == "amplitude":
        grid = [float(value) for value in power["effect_grid_standardized"]]
        residual_sd = float(design["regulation_projection_residual_sd"])
        unit_scale = float(
            design.get("power_target_effect_standardization_sd", 1.0)
        )
    elif kind == "onset":
        grid = [float(value) for value in power["onset_shift_grid"]]
        residual_sd = float(design["timing_projection_residual_sd"])
        unit_scale = float(power["onset_shift_standardized_unit"])
    else:
        raise ValueError("kind must be amplitude or onset")
    rng = np.random.default_rng(_seed_for(seed, f"power:{kind}"))
    frames: list[pd.DataFrame] = []
    for grid_index, grid_value in enumerate(grid):
        formal_curve_rng = np.random.default_rng(
            _seed_for(seed, f"power:{kind}:{grid_index}:formal-curve")
        )
        formal_curve_streams = _curve_random_streams(
            seed, f"power:{kind}:{grid_index}:formal-curve"
        )
        effects = np.zeros(n_pathways)
        effects[target_pathway_index] = (
            grid_value * unit_scale
            if kind == "amplitude"
            else grid_value / unit_scale
        )
        for start in range(0, n_replicates, chunk_size):
            size = min(chunk_size, n_replicates - start)
            test: dict[str, np.ndarray] = {}
            if shared_curve_plan is None:
                values = _simulate_projection_values(
                    rng,
                    size,
                    n_pathways,
                    age,
                    observed_case,
                    residual_sd=residual_sd,
                    pathway_correlation=float(design["pathway_correlation"]),
                    n_curve_bins=int(design["n_curve_bins"]),
                    bin_ar1_correlation=float(design["bin_ar1_correlation"]),
                    projection_kind=(
                        "amplitude" if kind == "amplitude" else "onset"
                    ),
                    covariate_slope_sd=float(design["covariate_slope_sd"]),
                    condition_effects=effects,
                    covariates=covariates,
                    donor_noise_scale=donor_noise,
                    donor_bin_available=donor_bin_available,
                    donor_bin_cell_count=donor_bin_cell_count,
                )
                test = _exact_test(values, plan, alpha)
            onset_shift_estimate = np.full(size, np.nan, dtype=float)
            onset_shift_recovered = np.full(size, False, dtype=bool)
            formal_onset_event_called = np.full(size, False, dtype=bool)
            if shared_curve_plan is not None:
                formal_effects = effects.copy()
                baseline_curves = np.broadcast_to(
                    np.asarray(design["pathway_baseline_mean"], dtype=float)[
                        None, :
                    ],
                    (int(design["n_curve_bins"]), n_pathways),
                ).copy()
                if kind == "onset":
                    control_profile, case_profile = _onset_logistic_profiles(
                        selected_bin_centers, grid_value
                    )
                    curve_profile = case_profile - control_profile
                    formal_effects[:] = 0.0
                    formal_effects[target_pathway_index] = 1.0
                    baseline_curves[:, target_pathway_index] += control_profile
                else:
                    curve_profile = _curve_effect_profile(
                        kind, int(design["n_curve_bins"])
                    )
                curve_scores = _simulate_profile_curve_scores(
                    formal_curve_rng,
                    size,
                    n_pathways,
                    covariates=covariates,
                    observed_case=observed_case,
                    donor_bin_available=donor_bin_available,
                    donor_bin_cell_count=donor_bin_cell_count,
                    residual_sd=residual_sd,
                    pathway_correlation=float(design["pathway_correlation"]),
                    pathway_dependence_correlation=np.asarray(
                        design["pathway_dependence_correlation"], dtype=float
                    ),
                    pathway_noise_scale=design["pathway_noise_scale"],
                    bin_ar1_correlation=float(design["bin_ar1_correlation"]),
                    covariate_slope_sd=float(design["covariate_slope_sd"]),
                    donor_noise_scale=donor_noise,
                    condition_effects=formal_effects,
                    effect_profile=curve_profile,
                    baseline_curves=baseline_curves,
                    random_streams=formal_curve_streams,
                    precision_reference_count=float(
                        design["primary_draw_median_positive_cell_count"]
                    ),
                )
                shared_results = _run_shared_curve_kernel_batch(
                    curve_scores,
                    shared_curve_plan,
                    alpha=alpha,
                    family_index=pathway_family_index,
                )
                test["maxT_reject"] = np.vstack(
                    [
                        result["production_family_hierarchical_reject"]
                        for result in shared_results
                    ]
                )
                test["by_reject"] = np.vstack(
                    [result["by_reject"] for result in shared_results]
                )
                if kind == "onset":
                    formal_onset_call = np.asarray(
                        [
                            _formal_event_decisions(
                                result,
                                selected_bin_centers,
                                alpha=alpha,
                                min_consecutive=int(
                                    spec["inference"][
                                        "timing_min_consecutive_windows"
                                    ]
                                ),
                                min_duration_fraction=float(
                                    spec["inference"][
                                        "timing_min_duration_fraction"
                                    ]
                                ),
                            )[0][target_pathway_index]
                            for result in shared_results
                        ],
                        dtype=bool,
                    )
                    formal_onset_event_called = formal_onset_call
                    onset_shift_estimate = np.asarray(
                        [
                            _estimate_half_rise_onset(
                                selected_bin_centers,
                                result["adjusted_case_curve"][:, target_pathway_index],
                            )
                            - _estimate_half_rise_onset(
                                selected_bin_centers,
                                result["adjusted_control_curve"][:, target_pathway_index],
                            )
                            for result in shared_results
                        ],
                        dtype=float,
                    )
                    tolerance = float(power["onset_recovery_tolerance"])
                    onset_shift_recovered = (
                        formal_onset_call
                        & np.isfinite(onset_shift_estimate)
                        & (np.abs(onset_shift_estimate - grid_value) <= tolerance)
                    )
            row_data: dict[str, Any] = {
                        "power_kind": kind,
                        "grid_value": grid_value,
                        "replicate_id": np.arange(start, start + size),
                        "signal_detected_maxT": test["maxT_reject"][:, target_pathway_index],
                        "signal_detected_BY": test["by_reject"][:, target_pathway_index],
                        "whole_donor_assignment_space": plan.n_assignments,
                    }
            if design_profile is not None:
                row_data["design_profile_payload_sha256"] = str(
                    design_profile["integrity"]["profile_payload_sha256"]
                )
                row_data["derived_design_parameters_sha256"] = (
                    derived_profile_parameters_sha256(design)
                )
                row_data["onset_shift_estimate"] = onset_shift_estimate
                row_data["onset_shift_recovered_within_tolerance"] = (
                    onset_shift_recovered
                )
                row_data["formal_onset_event_called"] = formal_onset_event_called
                row_data["power_target_effect_standardization_sd"] = float(
                    design["power_target_effect_standardization_sd"]
                )
                if shared_curve_plan is None:
                    raise RuntimeError("Formal shared power kernel was not used")
                row_data["formal_power_shared_kernel_used"] = True
                row_data["residual_mapping_space_size"] = int(
                    shared_curve_plan.residual_space_size
                )
                row_data["restricted_label_space_size"] = int(
                    shared_curve_plan.restricted_label_space_size
                )
                row_data["shared_kernel_availability_mask_sha256"] = (
                    shared_curve_plan.availability_mask_sha256
                )
                row_data["residual_reference_actual_mode"] = (
                    shared_curve_plan.actual_mode
                )
                row_data["residual_reference_enumeration"] = (
                    shared_curve_plan.reference_enumeration
                )
                row_data["residual_null_mappings"] = (
                    shared_curve_plan.n_null_mappings
                )
                row_data["residual_p_resolution"] = (
                    shared_curve_plan.monte_carlo_p_resolution
                )
            frames.append(pd.DataFrame(row_data))
    return pd.concat(frames, ignore_index=True)


def summarize_power_replicates(replicates: pd.DataFrame) -> pd.DataFrame:
    """Summarize a raw amplitude or onset power table."""
    rows = []
    for (kind, value), group in replicates.groupby(
        ["power_kind", "grid_value"], sort=True
    ):
        detection_power = float(group["signal_detected_maxT"].mean())
        row = {
            "power_kind": str(kind),
            "grid_value": float(value),
            "n_replicates": int(len(group)),
            "maxT_power": detection_power,
            "BY_power": float(group["signal_detected_BY"].mean()),
        }
        if str(kind) == "onset" and "formal_power_shared_kernel_used" in group:
            recovered = group[
                "onset_shift_recovered_within_tolerance"
            ].astype(bool)
            row["detection_only_maxT_power"] = detection_power
            row["onset_recovery_power"] = float(recovered.mean())
            # Resolution is defined by both formal detection and quantitative
            # recovery, not merely by rejection of a curve-null hypothesis.
            row["maxT_power"] = float(recovered.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _minimum_resolved_value(
    curve: pd.DataFrame, *, target_power: float
) -> tuple[float, bool]:
    ordered = curve.sort_values("grid_value")
    grid = ordered["grid_value"].to_numpy(dtype=float)
    observed = ordered["maxT_power"].to_numpy(dtype=float)
    monotone = np.maximum.accumulate(observed)
    reached = np.flatnonzero(monotone >= target_power)
    if len(reached):
        index = int(reached[0])
        if index == 0:
            return float(grid[0]), False
        x0, x1 = grid[index - 1], grid[index]
        y0, y1 = monotone[index - 1], monotone[index]
        if y1 <= y0:
            return float(x1), False
        fraction = (target_power - y0) / (y1 - y0)
        return float(x0 + fraction * (x1 - x0)), False
    spacing = grid[-1] - grid[-2] if len(grid) > 1 else max(grid[-1], 1.0)
    return float(grid[-1] + spacing), True


def _profile_loco_design_gate(
    spec: Mapping[str, Any],
    design_profile: Mapping[str, Any],
    derived: Mapping[str, Any],
    *,
    omitted_index: int,
) -> dict[str, Any]:
    """Recheck production identifiability on fixed S after one control omission."""
    donor_rows = list(design_profile["design"]["donor_rows"])
    full_case = np.asarray(derived["observed_assignment"], dtype=bool)
    if omitted_index < 0 or omitted_index >= len(full_case) or full_case[omitted_index]:
        raise ValueError("LOCO omission must identify one frozen disomy control")
    keep = np.ones(len(full_case), dtype=bool)
    keep[omitted_index] = False
    kept_rows = [row for row, retained in zip(donor_rows, keep) if retained]
    canonical = build_t21_canonical_donor_design(
        donor_ids=[row["donor_slot"] for row in kept_rows],
        conditions=[
            "T21" if int(row["assignment_code"]) == 1 else "disomy"
            for row in kept_rows
        ],
        pcw=[float(row["pcw"]) for row in kept_rows],
        technical_batch=[row["formal_batch_code"] for row in kept_rows],
        control="disomy",
        case="T21",
        expected_primary_batch_status=str(
            design_profile["design"]["canonical_formal_design"][
                "technical_batch_status"
            ]
        ),
        donor_order_mode="provided_frozen_subset",
    )
    available = np.asarray(derived["primary_draw_available_mask"], dtype=bool)[keep]
    selected_bins = np.arange(available.shape[1], dtype=int)
    groups, signatures, blocks = _make_pattern_blocks(
        canonical.donor_frame, available, selected_bins
    )
    condition = canonical.donor_frame["observed_case"].to_numpy(dtype=bool)
    residual_space, label_space = _space_sizes(condition, groups)
    contract = spec["inference"]["leave_one_control_out_support_gate"]
    if int(np.sum(~condition)) != int(contract["expected_controls_after_omission"]):
        raise ValueError("LOCO retained-control count differs from the frozen contract")
    diagnostics: list[dict[str, Any]] = []
    invalid: list[str] = []
    for bin_index in selected_bins:
        row = _bin_design_row(
            canonical.reduced_design,
            condition,
            available[:, bin_index],
            groups,
            max_condition_vif=float(contract["max_condition_vif"]),
            min_residual_df=int(contract["min_residual_df"]),
            min_donors_per_condition=int(contract["min_controls_per_selected_bin"]),
        )
        reasons = [value for value in str(row["rejection_reason"]).split("|") if value]
        if int(row["n_control_available"]) < int(
            contract["min_controls_per_selected_bin"]
        ):
            reasons.append("insufficient_loco_controls")
        if int(row["n_case_available"]) < int(contract["min_cases_per_selected_bin"]):
            reasons.append("insufficient_loco_cases")
        row["bin_index"] = int(bin_index)
        row["design_gate_pass"] = not reasons
        row["rejection_reason"] = "|".join(sorted(set(reasons)))
        diagnostics.append(_json_ready(row))
        if reasons:
            invalid.append(f"bin_{bin_index}:{row['rejection_reason']}")
    if residual_space <= 1:
        invalid.append("residual_permutation_space_size_one")
    if label_space <= 1:
        invalid.append("condition_label_space_size_one")
    payload = {
        "omitted_control_slot": str(donor_rows[omitted_index]["donor_slot"]),
        "fixed_selected_bin_indices": list(range(available.shape[1])),
        "segment_reselected": False,
        "technical_batch_status": canonical.technical_batch_status,
        "reduced_design": canonical.reduced_design.tolist(),
        "reduced_design_sha256": canonical.reduced_design_sha256,
        "terms": list(canonical.terms),
        "terms_sha256": canonical.terms_sha256,
        "encoding": [dict(value) for value in canonical.encoding],
        "encoding_sha256": canonical.encoding_sha256,
        "availability_signatures": list(signatures),
        "permutation_blocks": list(blocks),
        "residual_groups": [group.astype(int).tolist() for group in groups],
        "residual_mapping_space_size": int(residual_space),
        "restricted_label_space_size": int(label_space),
        "bin_diagnostics": diagnostics,
        "design_gate_pass": not invalid,
        "rejection_reasons": sorted(set(invalid)),
    }
    payload["diagnostics_sha256"] = hashlib.sha256(
        json.dumps(
            _json_ready(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    if invalid:
        raise ValueError(
            "LOCO fixed-S production design gate failed: " + ";".join(invalid)
        )
    payload["canonical_design"] = canonical
    payload["keep_mask"] = keep
    return payload


def _loco_replicates(
    spec: Mapping[str, Any],
    n_replicates: int,
    *,
    seed: int,
    chunk_size: int,
    design_profile: Mapping[str, Any] | None = None,
    derived_parameters: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    target = float(spec["power"]["target_effect_standardized"])
    if str(spec.get("schema_version", "")).startswith("2."):
        if design_profile is None:
            raise ValueError("Runner v2 LOCO power requires a design profile")
        design = dict(
            derived_parameters
            if derived_parameters is not None
            else derive_profile_simulation_parameters(spec, design_profile)
        )
        full_case = np.asarray(design["observed_assignment"], dtype=float)
        full_covariates = np.asarray(design["covariate_matrix"], dtype=float)
        full_age = full_covariates[:, 0]
        full_donor_noise = np.asarray(design["donor_noise_scale"], dtype=float)
        full_donor_bin_available = np.asarray(
            design["primary_draw_available_mask"], dtype=bool
        )
        full_donor_bin_cell_count = np.asarray(
            design["primary_draw_cell_count"], dtype=float
        )
        pathway_family_index = np.asarray(
            design["level_1_family_index_by_pathway"], dtype=int
        )
        target_pathway_index = int(design["power_target_pathway_index"])
        control_indices = np.flatnonzero(full_case == 0)
    else:
        design = spec["design"]
        n_controls = int(design["n_disomy_controls"])
        n_cases = int(design["n_t21_cases"])
        full_age, full_case = _donor_design(n_controls, n_cases, confounded=False)
        full_covariates = full_age[:, None]
        full_donor_noise = None
        full_donor_bin_available = None
        full_donor_bin_cell_count = None
        pathway_family_index = None
        target_pathway_index = 0
        control_indices = np.arange(n_controls)
    n_pathways = int(design["n_pathways"])
    alpha = float(design["alpha"])
    frames: list[pd.DataFrame] = []
    for omitted_control, omitted_index in enumerate(control_indices):
        loco_gate = None
        if design_profile is not None:
            loco_gate = _profile_loco_design_gate(
                spec,
                design_profile,
                design,
                omitted_index=int(omitted_index),
            )
            keep = np.asarray(loco_gate["keep_mask"], dtype=bool)
            loco_canonical = loco_gate["canonical_design"]
            covariates = np.asarray(loco_canonical.nuisance_matrix, dtype=float)
            age = covariates[:, 0]
            observed_case = loco_canonical.donor_frame[
                "observed_case"
            ].to_numpy(dtype=float)
            loco_reduced_design = np.asarray(
                loco_canonical.reduced_design, dtype=float
            )
        else:
            keep = np.ones(len(full_case), dtype=bool)
            keep[int(omitted_index)] = False
            age = full_age[keep]
            observed_case = full_case[keep]
            covariates = full_covariates[keep]
            loco_reduced_design = _reduced_covariate_design(covariates)
        plan = _make_covariate_contrast_plan(covariates, observed_case)
        donor_noise = full_donor_noise[keep] if full_donor_noise is not None else None
        donor_bin_available = (
            full_donor_bin_available[keep]
            if full_donor_bin_available is not None
            else None
        )
        donor_bin_cell_count = (
            full_donor_bin_cell_count[keep]
            if full_donor_bin_cell_count is not None
            else None
        )
        shared_curve_plan = (
            make_array_freedman_lane_plan(
                reduced_design=loco_reduced_design,
                condition=np.asarray(observed_case, dtype=bool),
                available=np.asarray(donor_bin_available, dtype=bool),
                widths=np.asarray(design["bin_widths"], dtype=float),
                max_exact_permutations=int(
                    spec["inference"]["max_exhaustive_residual_mappings"]
                ),
                permutation_mode=str(spec["inference"]["residual_reference_mode"]),
                n_permutations=int(
                    spec["inference"]["monte_carlo_residual_mappings"]
                ),
                seed=_seed_for(seed, f"loco-plan:{omitted_control}"),
            )
            if donor_bin_available is not None
            else None
        )
        rng = np.random.default_rng(_seed_for(seed, f"loco:{omitted_control}"))
        formal_curve_rng = np.random.default_rng(
            _seed_for(seed, f"loco:{omitted_control}:formal-curve")
        )
        formal_curve_streams = _curve_random_streams(
            seed, f"loco:{omitted_control}:formal-curve"
        )
        effects = np.zeros(n_pathways)
        effects[target_pathway_index] = target * float(
            design.get("power_target_effect_standardization_sd", 1.0)
        )
        for start in range(0, n_replicates, chunk_size):
            size = min(chunk_size, n_replicates - start)
            test: dict[str, np.ndarray] = {}
            if shared_curve_plan is None:
                values = _simulate_projection_values(
                    rng,
                    size,
                    n_pathways,
                    age,
                    observed_case,
                    residual_sd=float(design["regulation_projection_residual_sd"]),
                    pathway_correlation=float(design["pathway_correlation"]),
                    n_curve_bins=int(design["n_curve_bins"]),
                    bin_ar1_correlation=float(design["bin_ar1_correlation"]),
                    projection_kind="amplitude",
                    covariate_slope_sd=float(design["covariate_slope_sd"]),
                    condition_effects=effects,
                    covariates=covariates,
                    donor_noise_scale=donor_noise,
                    donor_bin_available=donor_bin_available,
                    donor_bin_cell_count=donor_bin_cell_count,
                )
                test = _exact_test(values, plan, alpha)
            if shared_curve_plan is not None:
                curve_scores = _simulate_profile_curve_scores(
                    formal_curve_rng,
                    size,
                    n_pathways,
                    covariates=covariates,
                    observed_case=observed_case,
                    donor_bin_available=donor_bin_available,
                    donor_bin_cell_count=donor_bin_cell_count,
                    residual_sd=float(design["regulation_projection_residual_sd"]),
                    pathway_correlation=float(design["pathway_correlation"]),
                    pathway_dependence_correlation=np.asarray(
                        design["pathway_dependence_correlation"], dtype=float
                    ),
                    pathway_noise_scale=design["pathway_noise_scale"],
                    bin_ar1_correlation=float(design["bin_ar1_correlation"]),
                    covariate_slope_sd=float(design["covariate_slope_sd"]),
                    donor_noise_scale=donor_noise,
                    condition_effects=effects,
                    effect_profile=_curve_effect_profile(
                        "amplitude", int(design["n_curve_bins"])
                    ),
                    baseline_curves=np.broadcast_to(
                        np.asarray(design["pathway_baseline_mean"], dtype=float)[
                            None, :
                        ],
                        (int(design["n_curve_bins"]), n_pathways),
                    ),
                    random_streams=formal_curve_streams,
                    precision_reference_count=float(
                        design["primary_draw_median_positive_cell_count"]
                    ),
                )
                shared_results = _run_shared_curve_kernel_batch(
                    curve_scores,
                    shared_curve_plan,
                    alpha=alpha,
                    family_index=pathway_family_index,
                )
                test["maxT_reject"] = np.vstack(
                    [
                        result["production_family_hierarchical_reject"]
                        for result in shared_results
                    ]
                )
            row_data: dict[str, Any] = {
                        "omitted_control_index": omitted_control,
                        "replicate_id": np.arange(start, start + size),
                        "target_effect_standardized": target,
                        "signal_detected_maxT": test["maxT_reject"][:, target_pathway_index],
                        "whole_donor_assignment_space": plan.n_assignments,
                    }
            if design_profile is not None:
                row_data["design_profile_payload_sha256"] = str(
                    design_profile["integrity"]["profile_payload_sha256"]
                )
                row_data["derived_design_parameters_sha256"] = (
                    derived_profile_parameters_sha256(design)
                )
                row_data["power_target_effect_standardization_sd"] = float(
                    design["power_target_effect_standardization_sd"]
                )
                if shared_curve_plan is None:
                    raise RuntimeError("Formal shared LOCO power kernel was not used")
                row_data["formal_loco_shared_kernel_used"] = True
                if loco_gate is None:
                    raise RuntimeError("Formal LOCO production design gate was not run")
                row_data["loco_support_design_valid"] = True
                row_data["loco_support_diagnostics_sha256"] = str(
                    loco_gate["diagnostics_sha256"]
                )
                row_data["loco_reduced_design_sha256"] = str(
                    loco_gate["reduced_design_sha256"]
                )
                row_data["loco_terms_sha256"] = str(loco_gate["terms_sha256"])
                row_data["loco_encoding_sha256"] = str(
                    loco_gate["encoding_sha256"]
                )
                row_data["residual_mapping_space_size"] = int(
                    shared_curve_plan.residual_space_size
                )
                row_data["restricted_label_space_size"] = int(
                    shared_curve_plan.restricted_label_space_size
                )
                row_data["shared_kernel_availability_mask_sha256"] = (
                    shared_curve_plan.availability_mask_sha256
                )
                row_data["residual_reference_actual_mode"] = (
                    shared_curve_plan.actual_mode
                )
                row_data["residual_reference_enumeration"] = (
                    shared_curve_plan.reference_enumeration
                )
                row_data["residual_null_mappings"] = (
                    shared_curve_plan.n_null_mappings
                )
                row_data["residual_p_resolution"] = (
                    shared_curve_plan.monte_carlo_p_resolution
                )
            frames.append(pd.DataFrame(row_data))
    return pd.concat(frames, ignore_index=True)


def summarize_loco_replicates(replicates: pd.DataFrame) -> pd.DataFrame:
    """Return power for each leave-one-control-out refit."""
    return (
        replicates.groupby("omitted_control_index", as_index=False)
        .agg(
            n_replicates=("replicate_id", "size"),
            target_effect_standardized=("target_effect_standardized", "first"),
            maxT_power=("signal_detected_maxT", "mean"),
            whole_donor_assignment_space=(
                "whole_donor_assignment_space",
                "first",
            ),
        )
        .sort_values("omitted_control_index")
        .reset_index(drop=True)
    )


def summarize_calibration_power_metrics(
    *,
    scenario_metrics: pd.DataFrame,
    power_curve: pd.DataFrame,
    onset_power_curve: pd.DataFrame,
    loco_power: pd.DataFrame,
    spec: Mapping[str, Any],
    n_replicates_per_point: int,
) -> dict[str, Any]:
    """Recompute report-level power and resolution metrics from summaries."""
    target_power = float(spec["power"]["target_power"])
    minimum_effect, effect_extrapolated = _minimum_resolved_value(
        power_curve, target_power=target_power
    )
    minimum_onset, onset_extrapolated = _minimum_resolved_value(
        onset_power_curve, target_power=target_power
    )
    target_effect = float(spec["power"]["target_effect_standardized"])
    target_rows = power_curve.loc[np.isclose(power_curve["grid_value"], target_effect)]
    if len(target_rows) != 1:
        raise ValueError("Power grid must contain target_effect_standardized once")
    covariate_row = scenario_metrics.set_index("scenario").loc[
        "covariate_condition_association"
    ]
    return {
        "n_replicates_per_point": int(n_replicates_per_point),
        "target_effect_standardized": target_effect,
        "power_at_target_effect": float(target_rows.iloc[0]["maxT_power"]),
        "minimum_detectable_effect_standardized": minimum_effect,
        "minimum_detectable_effect_extrapolated": effect_extrapolated,
        "minimum_resolvable_onset_shift": minimum_onset,
        "minimum_resolvable_onset_shift_extrapolated": onset_extrapolated,
        "leave_one_control_out_power": float(loco_power["maxT_power"].min()),
        "leave_one_control_out_power_mean": float(loco_power["maxT_power"].mean()),
        "extreme_donor_empirical_fwer": float(covariate_row["empirical_fwer"]),
    }


def run_pre_unblinding_calibration(
    spec: Mapping[str, Any],
    plan: CalibrationRunPlan,
    *,
    seed: int = 20260713,
    chunk_size: int = 32,
    design_profile: Mapping[str, Any] | None = None,
) -> CalibrationRunResult:
    """Run the blinded six-scenario whole-donor calibration harness."""
    benchmark_started = time.perf_counter()
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    uses_profile = str(spec.get("schema_version", "")).startswith("2.")
    if uses_profile and chunk_size > int(
        spec["performance_contract"]["maximum_replicate_chunk_size"]
    ):
        raise ValueError("Runner v2 chunk_size exceeds the frozen memory contract")
    if uses_profile and plan.phase == "final":
        publication = spec["publication_execution_contract"]
        if (
            int(seed) != int(publication["seed"])
            or int(chunk_size) != int(publication["chunk_size"])
            or plan.development_override
            or plan.complete_null_replicates
            != int(publication["complete_null_replicates"])
            or plan.scenario_replicates
            != int(publication["scenario_replicates"])
            or plan.power_replicates_per_point
            != int(publication["power_replicates_per_point"])
        ):
            raise ValueError(
                "Runner v2 final execution requires the frozen seed, chunk size, "
                "replicate counts, and no development override"
            )
    if uses_profile and design_profile is None:
        raise ValueError("Runner spec v2 requires a validated design profile")
    if not uses_profile and design_profile is not None:
        raise ValueError("Runner spec v1 does not accept a design profile")
    derived_parameters = (
        derive_profile_simulation_parameters(spec, design_profile)
        if design_profile is not None
        else None
    )
    scenario_frames = []
    for scenario in SCENARIOS:
        n_replicates = (
            plan.complete_null_replicates
            if scenario == "complete_null"
            else plan.scenario_replicates
        )
        scenario_frames.append(
            _scenario_replicates(
                scenario,
                n_replicates,
                spec,
                seed=seed,
                chunk_size=chunk_size,
                design_profile=design_profile,
                derived_parameters=derived_parameters,
            )
        )
    scenario_replicates = pd.concat(scenario_frames, ignore_index=True)
    scenario_metrics = summarize_scenario_replicates(scenario_replicates)

    amplitude = _power_replicates(
        spec,
        plan.power_replicates_per_point,
        seed=seed,
        chunk_size=chunk_size,
        kind="amplitude",
        design_profile=design_profile,
        derived_parameters=derived_parameters,
    )
    onset = _power_replicates(
        spec,
        plan.power_replicates_per_point,
        seed=seed,
        chunk_size=chunk_size,
        kind="onset",
        design_profile=design_profile,
        derived_parameters=derived_parameters,
    )
    power_replicates = pd.concat([amplitude, onset], ignore_index=True)
    power_summary = summarize_power_replicates(power_replicates)
    power_curve = power_summary.loc[
        power_summary["power_kind"].eq("amplitude")
    ].reset_index(drop=True)
    onset_power_curve = power_summary.loc[
        power_summary["power_kind"].eq("onset")
    ].reset_index(drop=True)

    loco_replicates = _loco_replicates(
        spec,
        plan.power_replicates_per_point,
        seed=seed,
        chunk_size=chunk_size,
        design_profile=design_profile,
        derived_parameters=derived_parameters,
    )
    loco_power = summarize_loco_replicates(loco_replicates)
    power_metrics = summarize_calibration_power_metrics(
        scenario_metrics=scenario_metrics,
        power_curve=power_curve,
        onset_power_curve=onset_power_curve,
        loco_power=loco_power,
        spec=spec,
        n_replicates_per_point=plan.power_replicates_per_point,
    )
    benchmark_elapsed = float(time.perf_counter() - benchmark_started)
    benchmark_work_units = int(
        len(scenario_replicates) + len(power_replicates) + len(loco_replicates)
    )
    effective_design = derived_parameters if derived_parameters is not None else spec["design"]
    inference = spec["inference"]
    metadata = {
        "runner": "t21_pre_unblinding_calibration",
        "runner_schema_version": str(spec.get("schema_version", "1.0.0")),
        "phase": plan.phase,
        "development_override": plan.development_override,
        "seed": int(seed),
        "chunk_size": int(chunk_size),
        "outcome_blinded": True,
        "real_pathway_results_read": False,
        "simulation_input_class": (
            "profile_parameterized_synthetic_donor_curve_projections"
            if uses_profile
            else "synthetic_donor_curve_projections_only"
        ),
        "design_profile_used": uses_profile,
        "whole_donor_assignment_enumeration": str(
            inference["whole_donor_assignment_enumeration"]
        ),
        "label_space_exhaustive": True,
        "finite_sample_exactness_with_continuous_covariates_claimed": (
            False if uses_profile else True
        ),
        "exactness_statement": (
            str(inference["exactness_statement"])
            if uses_profile
            else "All 680 whole-donor labels are enumerated under the v1 synthetic design."
        ),
        "condition_label_space_size": math.comb(
            int(effective_design["n_disomy_controls"])
            + int(effective_design["n_t21_cases"]),
            int(effective_design["n_disomy_controls"]),
        ),
        "fixed_common_grid": bool(inference["fixed_common_grid"]),
        "n_curve_bins": int(
            effective_design.get("source_n_curve_bins", effective_design["n_curve_bins"])
        ),
        "selected_n_curve_bins": int(effective_design["n_curve_bins"]),
        "onset_interval_coverage_evaluated": False,
        "onset_confidence_interval_claim_unlocked": False,
        "metric_definitions": {
            "empirical_fwer": "mean(any permutation-maxT false rejection)",
            "empirical_fdr": "mean(BY false-discovery proportion)",
            "onset_false_positive_rate": "mean(any onset maxT rejection under null)",
            "duration_false_positive_rate": "mean(any duration maxT rejection under null)",
            (
                "pointwise_curve_coverage"
                if uses_profile
                else "confidence_coverage"
            ): "mean pointwise 95pct t-interval coverage of the fitted condition curve",
            "minimum_detectable_effect_standardized": (
                "linear interpolation to 80pct maxT power on frozen grid"
            ),
            "minimum_resolvable_onset_shift": (
                "linear interpolation to 80pct probability of formal maxT detection "
                "and absolute onset-shift recovery error within the frozen tolerance"
            ),
            "leave_one_control_out_power": (
                "minimum target-effect maxT power over three control omissions"
            ),
        },
        "publication_runner_benchmark": {
            "wall_clock_seconds": benchmark_elapsed,
            "replicate_work_units": benchmark_work_units,
            "replicate_work_units_per_second": (
                benchmark_work_units / benchmark_elapsed
            ),
            "seed": int(seed),
            "chunk_size": int(chunk_size),
            "vectorized_shared_freedman_lane_batch_used": bool(uses_profile),
            "mapping_batch_size": (
                int(spec["performance_contract"]["mapping_batch_size"])
                if uses_profile
                else 1
            ),
        },
    }
    if design_profile is not None and derived_parameters is not None:
        shared_plan = _make_profile_shared_curve_plan(
            spec,
            derived_parameters,
            seed=int(spec["inference"]["residual_mapping_seed"]),
        )
        metadata["design_profile_payload_sha256"] = str(
            design_profile["integrity"]["profile_payload_sha256"]
        )
        metadata["derived_design_parameters_sha256"] = (
            derived_profile_parameters_sha256(derived_parameters)
        )
        metadata["profile_parameter_sources"] = derived_parameters["parameter_sources"]
        metadata["formal_regulation_shared_kernel_used"] = True
        metadata["formal_timing_shared_kernel_used"] = True
        metadata["formal_power_shared_kernel_used"] = True
        metadata["formal_loco_shared_kernel_used"] = True
        metadata["formal_loco_fixed_s_support_gate_used"] = True
        metadata["vectorized_shared_freedman_lane_batch_used"] = True
        metadata["mapping_batch_size"] = int(
            spec["performance_contract"]["mapping_batch_size"]
        )
        metadata["maximum_replicate_chunk_size"] = int(
            spec["performance_contract"]["maximum_replicate_chunk_size"]
        )
        metadata["formal_occupancy_fate_decomposition_kernel_used"] = True
        metadata["shared_state_population_draw_used"] = True
        metadata["chr21_gene_level_total_trans_projection_used"] = True
        metadata["fixed_20_bin_source_grid_verified"] = bool(
            derived_parameters["fixed_20_bin_source_grid_verified"]
        )
        metadata["selected_support_design_valid"] = bool(
            derived_parameters["selected_support_design_valid"]
        )
        metadata["canonical_donor_design_spec_sha256"] = derived_parameters[
            "canonical_donor_design_spec_sha256"
        ]
        metadata["canonical_reduced_design_sha256"] = derived_parameters[
            "support_reduced_design_sha256"
        ]
        metadata["canonical_terms_sha256"] = derived_parameters[
            "support_reduced_design_terms_sha256"
        ]
        metadata["canonical_encoding_sha256"] = derived_parameters[
            "support_reduced_design_encoding_sha256"
        ]
        metadata["sensitivity_signature_matrix_sha256"] = derived_parameters[
            "sensitivity_signature_matrix_sha256"
        ]
        metadata["n_sensitivity_signature_components"] = int(
            np.asarray(
                derived_parameters["sensitivity_signature_matrix"], dtype=float
            ).shape[1]
        )
        metadata["covariate_stress_signature_effect_injected"] = bool(
            metadata["n_sensitivity_signature_components"] > 0
        )
        metadata["onset_recovery_tolerance"] = float(
            spec["power"]["onset_recovery_tolerance"]
        )
        metadata["power_target_effect_standardization_sd"] = float(
            derived_parameters["power_target_effect_standardization_sd"]
        )
        metadata["power_target_effect_standardization_rule"] = str(
            derived_parameters["power_target_effect_standardization_rule"]
        )
        metadata["shared_freedman_lane_kernel_sha256"] = derived_parameters[
            "shared_freedman_lane_kernel_sha256"
        ]
        for key in (
            "covariate_pseudobulk_core_sha256",
            "pathway_family_inference_core_sha256",
            "trajectory_decomposition_core_sha256",
            "trajectory_event_timing_core_sha256",
            "t21_covariate_design_core_sha256",
        ):
            metadata[key] = derived_parameters[key]
        metadata["shared_kernel_availability_mask_sha256"] = (
            shared_plan.availability_mask_sha256
        )
        metadata["source_primary_draw_available_mask_sha256"] = derived_parameters[
            "source_primary_draw_available_mask_sha256"
        ]
        metadata["selected_bin_mask_sha256"] = derived_parameters[
            "selected_bin_mask_sha256"
        ]
        metadata["included_donor_mask_sha256"] = derived_parameters[
            "included_donor_mask_sha256"
        ]
        metadata["residual_mapping_space_size"] = shared_plan.residual_space_size
        metadata["restricted_label_space_size"] = (
            shared_plan.restricted_label_space_size
        )
        metadata["residual_reference_enumeration"] = (
            shared_plan.reference_enumeration
        )
        metadata["residual_reference_actual_mode"] = shared_plan.actual_mode
        metadata["residual_null_mappings"] = shared_plan.n_null_mappings
        metadata["residual_p_resolution"] = shared_plan.monte_carlo_p_resolution
        metadata["condition_label_reference_role"] = (
            "exhaustive_680_assignment_design_and_label_space_diagnostic_only"
        )
        metadata["formal_curve_reference_role"] = (
            "availability_restricted_freedman_lane_residual_reference_for_all_formal_p_values"
        )
        metadata["residual_reference_exactness_status"] = shared_plan.exactness_status
    return CalibrationRunResult(
        scenario_replicates=scenario_replicates,
        scenario_metrics=scenario_metrics,
        power_replicates=power_replicates,
        power_curve=power_curve,
        onset_power_curve=onset_power_curve,
        loco_replicates=loco_replicates,
        loco_power=loco_power,
        power_metrics=power_metrics,
        metadata=metadata,
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_tsv(table: pd.DataFrame, path: Path) -> None:
    text = table.to_csv(sep="\t", index=False, lineterminator="\n")
    _atomic_text(path, text)


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    text = json.dumps(
        _json_ready(dict(payload)),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    _atomic_text(path, text + "\n")


def _calibration_artifact_role_paths(
    report: Mapping[str, Any], repository_root: Path
) -> dict[str, Path]:
    artifacts = report.get("output_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Calibration report output_artifacts must be a non-empty list")
    roles: dict[str, Path] = {}
    relative_paths: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise ValueError("Every calibration artifact record must be an object")
        role = str(artifact.get("role", ""))
        if not role or role in roles:
            raise ValueError("Calibration artifact roles must be non-empty and unique")
        relative_text = str(artifact.get("relative_path", ""))
        relative = Path(relative_text)
        if (
            not relative_text
            or relative.is_absolute()
            or ".." in relative.parts
            or relative_text in relative_paths
        ):
            raise ValueError(
                "Calibration artifact paths must be unique and repository-relative"
            )
        local = (repository_root / relative).resolve()
        try:
            local.relative_to(repository_root)
        except ValueError as exc:
            raise ValueError("Calibration artifact escapes repository_root") from exc
        if not local.is_file():
            raise ValueError(f"Calibration artifact is missing: {relative_text}")
        if int(artifact.get("bytes", -1)) != local.stat().st_size:
            raise ValueError(
                f"Calibration artifact byte count changed: {relative_text}"
            )
        digest = str(artifact.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Calibration artifact digest is invalid: {relative_text}")
        if sha256_file(local) != digest:
            raise ValueError(f"Calibration artifact digest changed: {relative_text}")
        roles[role] = local
        relative_paths.add(relative_text)
    required_roles = {
        "raw_scenario_replicates",
        "raw_power_replicates",
        "raw_leave_one_control_out_replicates",
    }
    missing = required_roles - set(roles)
    if missing:
        raise ValueError(
            f"Calibration report is missing raw artifact roles: {sorted(missing)}"
        )
    return roles


def _read_strict_tsv(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    if list(frame.columns) != list(columns):
        raise ValueError(
            f"Calibration raw artifact {path.name} has an unexpected column schema"
        )
    if frame.empty:
        raise ValueError(f"Calibration raw artifact {path.name} is empty")
    return frame


def _strict_integer_column(frame: pd.DataFrame, key: str) -> np.ndarray:
    values = frame[key].astype(str)
    if not values.str.fullmatch(r"0|[1-9][0-9]*").all():
        raise ValueError(
            f"Calibration raw column {key!r} must contain non-negative integers"
        )
    return values.to_numpy(dtype=np.int64)


def _strict_boolean_column(frame: pd.DataFrame, key: str) -> np.ndarray:
    values = frame[key].astype(str)
    if not values.isin({"True", "False"}).all():
        raise ValueError(f"Calibration raw column {key!r} must contain True/False")
    return values.eq("True").to_numpy(dtype=bool)


def _strict_float_column(
    frame: pd.DataFrame, key: str, *, allow_blank: bool = False
) -> np.ndarray:
    raw = frame[key].astype(str)
    blank = raw.eq("")
    if blank.any() and not allow_blank:
        raise ValueError(f"Calibration raw column {key!r} may not be blank")
    parsed = pd.to_numeric(raw.where(~blank, np.nan), errors="coerce").to_numpy(
        dtype=float
    )
    if np.any(~blank.to_numpy() & ~np.isfinite(parsed)):
        raise ValueError(f"Calibration raw column {key!r} must contain finite numbers")
    return parsed


def _require_probability(
    values: np.ndarray, key: str, *, allow_nan: bool = False
) -> None:
    values = np.asarray(values, dtype=float)
    if not allow_nan and not np.isfinite(values).all():
        raise ValueError(f"Calibration probability column {key!r} must be finite")
    finite = values[np.isfinite(values)]
    if np.any((finite < 0.0) | (finite > 1.0)):
        raise ValueError(f"Calibration probability column {key!r} is outside [0, 1]")


def _require_complete_replicate_ids(
    frame: pd.DataFrame, group_keys: Sequence[str]
) -> int:
    counts: set[int] = set()
    for name, group in frame.groupby(list(group_keys), sort=False, dropna=False):
        ids = group["replicate_id"].to_numpy(dtype=np.int64)
        if len(np.unique(ids)) != len(ids) or not np.array_equal(
            np.sort(ids), np.arange(len(ids), dtype=np.int64)
        ):
            raise ValueError(
                f"Calibration replicate IDs for {name!r} must be unique and contiguous from zero"
            )
        counts.add(len(ids))
    if len(counts) != 1:
        raise ValueError("Calibration replicate counts differ across equivalent groups")
    return counts.pop()


def _profile_raw_columns(spec: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        ("design_profile_payload_sha256", "derived_design_parameters_sha256")
        if str(spec.get("schema_version", "")).startswith("2.")
        else ()
    )


def _validate_raw_profile_bindings(
    raw: pd.DataFrame,
    typed: pd.DataFrame,
    *,
    expected_profile_payload_sha256: str | None,
    expected_parameters_sha256: str | None,
) -> None:
    if expected_profile_payload_sha256 is None or expected_parameters_sha256 is None:
        return
    for key, expected in (
        ("design_profile_payload_sha256", expected_profile_payload_sha256),
        ("derived_design_parameters_sha256", expected_parameters_sha256),
    ):
        values = raw[key].astype(str)
        if not values.map(lambda value: bool(re.fullmatch(r"[0-9a-f]{64}", value))).all():
            raise ValueError(f"Raw calibration profile binding {key!r} is invalid")
        if set(values) != {expected}:
            raise ValueError(f"Raw calibration profile binding {key!r} changed")
        typed[key] = values


def _load_raw_scenario_replicates(
    path: Path,
    spec: Mapping[str, Any],
    *,
    expected_profile_payload_sha256: str | None = None,
    expected_parameters_sha256: str | None = None,
    expected_shared_mask_sha256: str | None = None,
    expected_residual_mapping_space_size: int | None = None,
    expected_restricted_label_space_size: int | None = None,
    expected_residual_reference_actual_mode: str | None = None,
    expected_residual_reference_enumeration: str | None = None,
    expected_residual_null_mappings: int | None = None,
    expected_residual_p_resolution: float | None = None,
    expected_sensitivity_signature_matrix_sha256: str | None = None,
    expected_n_sensitivity_signature_components: int | None = None,
) -> pd.DataFrame:
    columns = (
        "scenario",
        "replicate_id",
        "any_maxT_false_rejection",
        "by_false_discovery_proportion",
        "onset_false_positive",
        "duration_false_positive",
        (
            "pointwise_curve_coverage_fraction"
            if str(spec.get("schema_version", "")).startswith("2.")
            else "confidence_coverage_fraction"
        ),
        "regulation_false_discovery_proportion",
        "false_timing_shift",
        "trans_false_discovery_proportion",
        "cis_detection_fraction",
        "chr21_total_positive_control_detected",
        "whole_donor_assignment_space",
        *_profile_raw_columns(spec),
        *(
            (
                "occupancy_signal_detected",
                "fate_signal_detected",
                "formal_regulation_shared_kernel_used",
                "formal_timing_shared_kernel_used",
                "formal_occupancy_fate_decomposition_kernel_used",
                "shared_state_population_draw_used",
                "fixed_20_bin_source_grid_verified",
                "selected_support_design_valid",
                "sensitivity_signature_effect_injected",
                "sensitivity_signature_matrix_sha256",
                "n_sensitivity_signature_components",
                "chr21_gene_level_total_trans_projection_used",
                "residual_mapping_space_size",
                "restricted_label_space_size",
                "shared_kernel_availability_mask_sha256",
                "residual_reference_actual_mode",
                "residual_reference_enumeration",
                "residual_null_mappings",
                "residual_p_resolution",
            )
            if str(spec.get("schema_version", "")).startswith("2.")
            else ()
        ),
    )
    raw = _read_strict_tsv(path, columns)
    if set(raw["scenario"]) != set(SCENARIOS):
        raise ValueError(
            "Raw scenario artifact must contain exactly the frozen six scenarios"
        )
    typed = pd.DataFrame({"scenario": raw["scenario"].astype(str)})
    typed["replicate_id"] = _strict_integer_column(raw, "replicate_id")
    boolean_columns = (
        "any_maxT_false_rejection",
        "onset_false_positive",
        "duration_false_positive",
        "false_timing_shift",
    )
    for key in boolean_columns:
        typed[key] = _strict_boolean_column(raw, key)
    finite_probability_columns = (
        "by_false_discovery_proportion",
        (
            "pointwise_curve_coverage_fraction"
            if str(spec.get("schema_version", "")).startswith("2.")
            else "confidence_coverage_fraction"
        ),
        "regulation_false_discovery_proportion",
    )
    for key in finite_probability_columns:
        values = _strict_float_column(raw, key)
        _require_probability(values, key)
        typed[key] = values
    optional_probability_columns = (
        "trans_false_discovery_proportion",
        "cis_detection_fraction",
        "chr21_total_positive_control_detected",
    )
    for key in optional_probability_columns:
        values = _strict_float_column(raw, key, allow_blank=True)
        _require_probability(values, key, allow_nan=True)
        typed[key] = values
    assignment_space = _strict_integer_column(raw, "whole_donor_assignment_space")
    expected_space = math.comb(
        int(spec["design"]["n_disomy_controls"]) + int(spec["design"]["n_t21_cases"]),
        int(spec["design"]["n_disomy_controls"]),
    )
    if not np.all(assignment_space == expected_space):
        raise ValueError(
            "Raw scenario artifact changed the whole-donor assignment space"
        )
    typed["whole_donor_assignment_space"] = assignment_space
    _validate_raw_profile_bindings(
        raw,
        typed,
        expected_profile_payload_sha256=expected_profile_payload_sha256,
        expected_parameters_sha256=expected_parameters_sha256,
    )
    if expected_shared_mask_sha256 is not None:
        for key in (
            "occupancy_signal_detected",
            "fate_signal_detected",
            "formal_regulation_shared_kernel_used",
            "formal_timing_shared_kernel_used",
            "formal_occupancy_fate_decomposition_kernel_used",
            "shared_state_population_draw_used",
            "fixed_20_bin_source_grid_verified",
            "selected_support_design_valid",
            "chr21_gene_level_total_trans_projection_used",
        ):
            typed[key] = _strict_boolean_column(raw, key)
        if not (
            typed["formal_regulation_shared_kernel_used"].all()
            and typed["formal_timing_shared_kernel_used"].all()
            and typed["formal_occupancy_fate_decomposition_kernel_used"].all()
            and typed["shared_state_population_draw_used"].all()
            and typed["fixed_20_bin_source_grid_verified"].all()
            and typed["selected_support_design_valid"].all()
        ):
            raise ValueError("Raw scenario rows did not all use the shared curve kernel")
        typed["sensitivity_signature_effect_injected"] = _strict_boolean_column(
            raw, "sensitivity_signature_effect_injected"
        )
        expected_injected = typed["scenario"].eq(
            "covariate_condition_association"
        ).to_numpy() & bool(expected_n_sensitivity_signature_components)
        if not np.array_equal(
            typed["sensitivity_signature_effect_injected"].to_numpy(dtype=bool),
            expected_injected,
        ):
            raise ValueError("Raw sensitivity-signature injection flag changed")
        signature_hashes = raw["sensitivity_signature_matrix_sha256"].astype(str)
        if (
            expected_sensitivity_signature_matrix_sha256 is None
            or set(signature_hashes)
            != {str(expected_sensitivity_signature_matrix_sha256)}
        ):
            raise ValueError("Raw sensitivity-signature matrix hash changed")
        typed["sensitivity_signature_matrix_sha256"] = signature_hashes
        signature_count = _strict_integer_column(
            raw, "n_sensitivity_signature_components"
        )
        if (
            expected_n_sensitivity_signature_components is None
            or not np.all(
                signature_count == int(expected_n_sensitivity_signature_components)
            )
        ):
            raise ValueError("Raw sensitivity-signature component count changed")
        typed["n_sensitivity_signature_components"] = signature_count
        chr21_projection = typed[
            "chr21_gene_level_total_trans_projection_used"
        ].to_numpy(dtype=bool)
        if not np.array_equal(
            chr21_projection,
            typed["scenario"].eq("chr21_dosage_only").to_numpy(),
        ):
            raise ValueError(
                "Raw scenario chr21 gene-level total/trans projection flag changed"
            )
        for key, expected in (
            ("residual_mapping_space_size", expected_residual_mapping_space_size),
            ("restricted_label_space_size", expected_restricted_label_space_size),
        ):
            values = _strict_integer_column(raw, key)
            if expected is None or not np.all(values == int(expected)):
                raise ValueError(f"Raw scenario {key!r} changed")
            typed[key] = values
        masks = raw["shared_kernel_availability_mask_sha256"].astype(str)
        if set(masks) != {expected_shared_mask_sha256}:
            raise ValueError("Raw scenario shared-kernel availability mask changed")
        typed["shared_kernel_availability_mask_sha256"] = masks
        for key, expected in (
            ("residual_reference_actual_mode", expected_residual_reference_actual_mode),
            (
                "residual_reference_enumeration",
                expected_residual_reference_enumeration,
            ),
        ):
            values = raw[key].astype(str)
            if expected is None or set(values) != {str(expected)}:
                raise ValueError(f"Raw scenario {key!r} changed")
            typed[key] = values
        null_mappings = _strict_integer_column(raw, "residual_null_mappings")
        if expected_residual_null_mappings is None or not np.all(
            null_mappings == int(expected_residual_null_mappings)
        ):
            raise ValueError("Raw scenario residual-null mapping count changed")
        typed["residual_null_mappings"] = null_mappings
        resolution = _strict_float_column(raw, "residual_p_resolution")
        if expected_residual_p_resolution is None or not np.allclose(
            resolution,
            float(expected_residual_p_resolution),
            rtol=0.0,
            atol=1e-15,
        ):
            raise ValueError("Raw scenario residual p-value resolution changed")
        typed["residual_p_resolution"] = resolution
    if typed.duplicated(["scenario", "replicate_id"]).any():
        raise ValueError("Raw scenario artifact contains duplicate replicate rows")
    for scenario, group in typed.groupby("scenario", sort=False):
        ids = group["replicate_id"].to_numpy(dtype=np.int64)
        if not np.array_equal(np.sort(ids), np.arange(len(ids), dtype=np.int64)):
            raise ValueError(
                f"Scenario {scenario!r} replicate IDs must be unique and contiguous from zero"
            )
    chr21 = typed["scenario"].eq("chr21_dosage_only").to_numpy()
    for key in optional_probability_columns:
        values = typed[key].to_numpy(dtype=float)
        if not np.isfinite(values[chr21]).all() or np.isfinite(values[~chr21]).any():
            raise ValueError(
                f"Raw scenario column {key!r} is only defined for chr21_dosage_only"
            )
    return typed


def _canonical_grid(
    values: np.ndarray, expected: Sequence[float], *, label: str
) -> np.ndarray:
    expected_array = np.asarray(expected, dtype=float)
    canonical = np.empty(len(values), dtype=float)
    for index, value in enumerate(np.asarray(values, dtype=float)):
        matches = np.flatnonzero(
            np.isclose(expected_array, value, rtol=0.0, atol=1e-12)
        )
        if len(matches) != 1:
            raise ValueError(f"Raw {label} grid contains an unfrozen value")
        canonical[index] = expected_array[int(matches[0])]
    if set(canonical) != set(expected_array):
        raise ValueError(f"Raw {label} grid is incomplete")
    return canonical


def _load_raw_power_replicates(
    path: Path,
    spec: Mapping[str, Any],
    *,
    expected_profile_payload_sha256: str | None = None,
    expected_parameters_sha256: str | None = None,
    expected_shared_mask_sha256: str | None = None,
    expected_residual_mapping_space_size: int | None = None,
    expected_restricted_label_space_size: int | None = None,
    expected_residual_reference_actual_mode: str | None = None,
    expected_residual_reference_enumeration: str | None = None,
    expected_residual_null_mappings: int | None = None,
    expected_residual_p_resolution: float | None = None,
    expected_power_target_effect_standardization_sd: float | None = None,
) -> tuple[pd.DataFrame, int]:
    columns = (
        "power_kind",
        "grid_value",
        "replicate_id",
        "signal_detected_maxT",
        "signal_detected_BY",
        "whole_donor_assignment_space",
        *_profile_raw_columns(spec),
        *(
            (
                "onset_shift_estimate",
                "onset_shift_recovered_within_tolerance",
                "formal_onset_event_called",
                "power_target_effect_standardization_sd",
                "formal_power_shared_kernel_used",
                "residual_mapping_space_size",
                "restricted_label_space_size",
                "shared_kernel_availability_mask_sha256",
                "residual_reference_actual_mode",
                "residual_reference_enumeration",
                "residual_null_mappings",
                "residual_p_resolution",
            )
            if str(spec.get("schema_version", "")).startswith("2.")
            else ()
        ),
    )
    raw = _read_strict_tsv(path, columns)
    if set(raw["power_kind"]) != {"amplitude", "onset"}:
        raise ValueError("Raw power artifact must contain amplitude and onset grids")
    typed = pd.DataFrame({"power_kind": raw["power_kind"].astype(str)})
    raw_grid = _strict_float_column(raw, "grid_value")
    typed["grid_value"] = np.nan
    for kind, expected in (
        ("amplitude", spec["power"]["effect_grid_standardized"]),
        ("onset", spec["power"]["onset_shift_grid"]),
    ):
        mask = typed["power_kind"].eq(kind).to_numpy()
        typed.loc[mask, "grid_value"] = _canonical_grid(
            raw_grid[mask], expected, label=kind
        )
    typed["replicate_id"] = _strict_integer_column(raw, "replicate_id")
    typed["signal_detected_maxT"] = _strict_boolean_column(raw, "signal_detected_maxT")
    typed["signal_detected_BY"] = _strict_boolean_column(raw, "signal_detected_BY")
    assignment_space = _strict_integer_column(raw, "whole_donor_assignment_space")
    expected_space = math.comb(
        int(spec["design"]["n_disomy_controls"]) + int(spec["design"]["n_t21_cases"]),
        int(spec["design"]["n_disomy_controls"]),
    )
    if not np.all(assignment_space == expected_space):
        raise ValueError("Raw power artifact changed the whole-donor assignment space")
    typed["whole_donor_assignment_space"] = assignment_space
    _validate_raw_profile_bindings(
        raw,
        typed,
        expected_profile_payload_sha256=expected_profile_payload_sha256,
        expected_parameters_sha256=expected_parameters_sha256,
    )
    if expected_shared_mask_sha256 is not None:
        onset_estimate = _strict_float_column(
            raw, "onset_shift_estimate", allow_blank=True
        )
        onset_recovered = _strict_boolean_column(
            raw, "onset_shift_recovered_within_tolerance"
        )
        formal_onset_event_called = _strict_boolean_column(
            raw, "formal_onset_event_called"
        )
        onset_mask = typed["power_kind"].eq("onset").to_numpy()
        amplitude_mask = ~onset_mask
        if (
            np.isfinite(onset_estimate[amplitude_mask]).any()
            or onset_recovered[amplitude_mask].any()
            or formal_onset_event_called[amplitude_mask].any()
        ):
            raise ValueError(
                "Amplitude power rows may not carry onset-recovery outcomes"
            )
        expected_recovered = (
            formal_onset_event_called[onset_mask]
            & np.isfinite(onset_estimate[onset_mask])
            & (
                np.abs(
                    onset_estimate[onset_mask]
                    - typed["grid_value"].to_numpy(dtype=float)[onset_mask]
                )
                <= float(spec["power"]["onset_recovery_tolerance"]) + 1e-12
            )
        )
        if not np.array_equal(onset_recovered[onset_mask], expected_recovered):
            raise ValueError(
                "Raw onset recovery flags differ from detection and frozen tolerance"
            )
        if np.any(
            formal_onset_event_called[onset_mask]
            & ~typed["signal_detected_maxT"].to_numpy(dtype=bool)[onset_mask]
        ):
            raise ValueError("Formal onset event calls require global curve detection")
        typed["onset_shift_estimate"] = onset_estimate
        typed["onset_shift_recovered_within_tolerance"] = onset_recovered
        typed["formal_onset_event_called"] = formal_onset_event_called
        standardization = _strict_float_column(
            raw, "power_target_effect_standardization_sd"
        )
        if (
            expected_power_target_effect_standardization_sd is None
            or not np.allclose(
                standardization,
                float(expected_power_target_effect_standardization_sd),
                rtol=0.0,
                atol=1e-15,
            )
        ):
            raise ValueError("Raw power target-effect standardization changed")
        typed["power_target_effect_standardization_sd"] = standardization
        typed["formal_power_shared_kernel_used"] = _strict_boolean_column(
            raw, "formal_power_shared_kernel_used"
        )
        if not typed["formal_power_shared_kernel_used"].all():
            raise ValueError("Raw power rows did not all use the shared curve kernel")
        for key, expected in (
            ("residual_mapping_space_size", expected_residual_mapping_space_size),
            ("restricted_label_space_size", expected_restricted_label_space_size),
        ):
            values = _strict_integer_column(raw, key)
            if expected is None or not np.all(values == int(expected)):
                raise ValueError(f"Raw power {key!r} changed")
            typed[key] = values
        masks = raw["shared_kernel_availability_mask_sha256"].astype(str)
        if set(masks) != {expected_shared_mask_sha256}:
            raise ValueError("Raw power shared-kernel availability mask changed")
        typed["shared_kernel_availability_mask_sha256"] = masks
        for key, expected in (
            ("residual_reference_actual_mode", expected_residual_reference_actual_mode),
            (
                "residual_reference_enumeration",
                expected_residual_reference_enumeration,
            ),
        ):
            values = raw[key].astype(str)
            if expected is None or set(values) != {str(expected)}:
                raise ValueError(f"Raw power {key!r} changed")
            typed[key] = values
        null_mappings = _strict_integer_column(raw, "residual_null_mappings")
        if expected_residual_null_mappings is None or not np.all(
            null_mappings == int(expected_residual_null_mappings)
        ):
            raise ValueError("Raw power residual-null mapping count changed")
        typed["residual_null_mappings"] = null_mappings
        resolution = _strict_float_column(raw, "residual_p_resolution")
        if expected_residual_p_resolution is None or not np.allclose(
            resolution,
            float(expected_residual_p_resolution),
            rtol=0.0,
            atol=1e-15,
        ):
            raise ValueError("Raw power residual p-value resolution changed")
        typed["residual_p_resolution"] = resolution
    if typed.duplicated(["power_kind", "grid_value", "replicate_id"]).any():
        raise ValueError("Raw power artifact contains duplicate replicate rows")
    n_per_point = _require_complete_replicate_ids(typed, ("power_kind", "grid_value"))
    return typed, n_per_point


def _load_raw_loco_replicates(
    path: Path,
    spec: Mapping[str, Any],
    *,
    expected_profile_payload_sha256: str | None = None,
    expected_parameters_sha256: str | None = None,
    expected_plan_bindings: Mapping[int, Mapping[str, Any]] | None = None,
    expected_power_target_effect_standardization_sd: float | None = None,
) -> tuple[pd.DataFrame, int]:
    columns = (
        "omitted_control_index",
        "replicate_id",
        "target_effect_standardized",
        "signal_detected_maxT",
        "whole_donor_assignment_space",
        *_profile_raw_columns(spec),
        *(
            (
                "power_target_effect_standardization_sd",
                "formal_loco_shared_kernel_used",
                "loco_support_design_valid",
                "loco_support_diagnostics_sha256",
                "loco_reduced_design_sha256",
                "loco_terms_sha256",
                "loco_encoding_sha256",
                "residual_mapping_space_size",
                "restricted_label_space_size",
                "shared_kernel_availability_mask_sha256",
                "residual_reference_actual_mode",
                "residual_reference_enumeration",
                "residual_null_mappings",
                "residual_p_resolution",
            )
            if str(spec.get("schema_version", "")).startswith("2.")
            else ()
        ),
    )
    raw = _read_strict_tsv(path, columns)
    typed = pd.DataFrame()
    typed["omitted_control_index"] = _strict_integer_column(
        raw, "omitted_control_index"
    )
    expected_omissions = set(range(int(spec["design"]["n_disomy_controls"])))
    if set(typed["omitted_control_index"]) != expected_omissions:
        raise ValueError("Raw LOCO artifact must contain every control omission once")
    typed["replicate_id"] = _strict_integer_column(raw, "replicate_id")
    target = _strict_float_column(raw, "target_effect_standardized")
    expected_target = float(spec["power"]["target_effect_standardized"])
    if not np.all(np.isclose(target, expected_target, rtol=0.0, atol=1e-12)):
        raise ValueError("Raw LOCO artifact changed the target effect")
    typed["target_effect_standardized"] = target
    typed["signal_detected_maxT"] = _strict_boolean_column(raw, "signal_detected_maxT")
    assignment_space = _strict_integer_column(raw, "whole_donor_assignment_space")
    expected_space = math.comb(
        int(spec["design"]["n_disomy_controls"])
        + int(spec["design"]["n_t21_cases"])
        - 1,
        int(spec["design"]["n_disomy_controls"]) - 1,
    )
    if not np.all(assignment_space == expected_space):
        raise ValueError("Raw LOCO artifact changed the whole-donor assignment space")
    typed["whole_donor_assignment_space"] = assignment_space
    _validate_raw_profile_bindings(
        raw,
        typed,
        expected_profile_payload_sha256=expected_profile_payload_sha256,
        expected_parameters_sha256=expected_parameters_sha256,
    )
    if expected_plan_bindings is not None:
        standardization = _strict_float_column(
            raw, "power_target_effect_standardization_sd"
        )
        if (
            expected_power_target_effect_standardization_sd is None
            or not np.allclose(
                standardization,
                float(expected_power_target_effect_standardization_sd),
                rtol=0.0,
                atol=1e-15,
            )
        ):
            raise ValueError("Raw LOCO target-effect standardization changed")
        typed["power_target_effect_standardization_sd"] = standardization
        typed["formal_loco_shared_kernel_used"] = _strict_boolean_column(
            raw, "formal_loco_shared_kernel_used"
        )
        if not typed["formal_loco_shared_kernel_used"].all():
            raise ValueError("Raw LOCO rows did not all use the shared curve kernel")
        typed["loco_support_design_valid"] = _strict_boolean_column(
            raw, "loco_support_design_valid"
        )
        if not typed["loco_support_design_valid"].all():
            raise ValueError("Raw LOCO rows did not pass the fixed-S design gate")
        for key in (
            "loco_support_diagnostics_sha256",
            "loco_reduced_design_sha256",
            "loco_terms_sha256",
            "loco_encoding_sha256",
        ):
            values = raw[key].astype(str)
            if not values.map(
                lambda value: bool(re.fullmatch(r"[0-9a-f]{64}", value))
            ).all():
                raise ValueError(f"Raw LOCO {key!r} is invalid")
            typed[key] = values
        typed["residual_mapping_space_size"] = _strict_integer_column(
            raw, "residual_mapping_space_size"
        )
        typed["restricted_label_space_size"] = _strict_integer_column(
            raw, "restricted_label_space_size"
        )
        typed["shared_kernel_availability_mask_sha256"] = raw[
            "shared_kernel_availability_mask_sha256"
        ].astype(str)
        typed["residual_reference_actual_mode"] = raw[
            "residual_reference_actual_mode"
        ].astype(str)
        typed["residual_reference_enumeration"] = raw[
            "residual_reference_enumeration"
        ].astype(str)
        typed["residual_null_mappings"] = _strict_integer_column(
            raw, "residual_null_mappings"
        )
        typed["residual_p_resolution"] = _strict_float_column(
            raw, "residual_p_resolution"
        )
        if set(expected_plan_bindings) != expected_omissions:
            raise ValueError("Expected LOCO shared-kernel plan set is incomplete")
        for omitted, group in typed.groupby("omitted_control_index", sort=False):
            expected = expected_plan_bindings[int(omitted)]
            if set(group["shared_kernel_availability_mask_sha256"]) != {
                str(expected["availability_mask_sha256"])
            }:
                raise ValueError("Raw LOCO shared-kernel availability mask changed")
            for key in (
                "loco_support_diagnostics_sha256",
                "loco_reduced_design_sha256",
                "loco_terms_sha256",
                "loco_encoding_sha256",
            ):
                if set(group[key]) != {str(expected[key])}:
                    raise ValueError(f"Raw LOCO {key!r} changed")
            for key in ("residual_mapping_space_size", "restricted_label_space_size"):
                if not np.all(group[key].to_numpy(dtype=np.int64) == int(expected[key])):
                    raise ValueError(f"Raw LOCO {key!r} changed")
            for key in (
                "residual_reference_actual_mode",
                "residual_reference_enumeration",
            ):
                if set(group[key]) != {str(expected[key])}:
                    raise ValueError(f"Raw LOCO {key!r} changed")
            if not np.all(
                group["residual_null_mappings"].to_numpy(dtype=np.int64)
                == int(expected["residual_null_mappings"])
            ):
                raise ValueError("Raw LOCO residual-null mapping count changed")
            if not np.allclose(
                group["residual_p_resolution"].to_numpy(dtype=float),
                float(expected["residual_p_resolution"]),
                rtol=0.0,
                atol=1e-15,
            ):
                raise ValueError("Raw LOCO residual p-value resolution changed")
    if typed.duplicated(["omitted_control_index", "replicate_id"]).any():
        raise ValueError("Raw LOCO artifact contains duplicate replicate rows")
    n_per_omission = _require_complete_replicate_ids(typed, ("omitted_control_index",))
    return typed, n_per_omission


def _strict_metric_mapping_match(
    actual: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    if set(actual) != set(expected):
        raise ValueError(f"{label} metric keys differ from raw-table recomputation")
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if expected_value is None or (
            isinstance(expected_value, (float, np.floating))
            and not math.isfinite(float(expected_value))
        ):
            if actual_value is not None:
                raise ValueError(f"{label} metric {key!r} must be null")
        elif isinstance(expected_value, (bool, np.bool_)):
            if not isinstance(actual_value, bool) or actual_value != bool(
                expected_value
            ):
                raise ValueError(f"{label} metric {key!r} differs from raw rows")
        elif isinstance(expected_value, (int, np.integer)):
            if (
                isinstance(actual_value, bool)
                or not isinstance(actual_value, int)
                or actual_value != int(expected_value)
            ):
                raise ValueError(f"{label} metric {key!r} differs from raw rows")
        elif isinstance(expected_value, str):
            if actual_value != expected_value:
                raise ValueError(f"{label} metric {key!r} differs from raw rows")
        else:
            if isinstance(actual_value, bool):
                raise ValueError(f"{label} metric {key!r} differs from raw rows")
            try:
                numeric = float(actual_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label} metric {key!r} is not numeric") from exc
            if not math.isfinite(numeric) or not math.isclose(
                numeric, float(expected_value), rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(f"{label} metric {key!r} differs from raw rows")


def validate_pre_unblinding_calibration_artifacts(
    report: Mapping[str, Any],
    *,
    repository_root: str | Path,
    runner_spec_path: str | Path,
) -> dict[str, Any]:
    """Recompute every report metric from raw replicate artifacts.

    Only the three ``raw_*`` artifact roles are used as statistical inputs.
    Reported summary TSVs are integrity-checked as registered artifacts but
    never trusted when recomputing ``scenario_metrics`` or ``power_metrics``.
    """
    root = Path(repository_root).resolve()
    spec_path = Path(runner_spec_path).resolve()
    try:
        spec_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "Calibration runner specification must be repository-local"
        ) from exc
    spec = load_runner_spec(spec_path)
    if report.get("outcome_blinded") is not True:
        raise ValueError("Calibration artifact recomputation requires a blind report")
    profile_runner = str(spec.get("schema_version", "")).startswith("2.")
    if profile_runner and (
        report.get("calibration_stage") != "final"
        or report.get("publication_minima_satisfied") is not True
    ):
        raise ValueError("Calibration v2 raw artifacts are not from a final publication run")
    execution = report.get("execution")
    if (
        not isinstance(execution, Mapping)
        or execution.get("real_pathway_results_read") is not False
    ):
        raise ValueError("Calibration report does not preserve the outcome blind")
    if profile_runner:
        publication = spec["publication_execution_contract"]
        if (
            execution.get("phase") != publication["phase"]
            or execution.get("development_override")
            is not publication["development_override"]
            or int(execution.get("seed", -1)) != int(publication["seed"])
            or int(execution.get("chunk_size", -1))
            != int(publication["chunk_size"])
        ):
            raise ValueError("Calibration v2 execution differs from the frozen final run")
    bindings = report.get("input_bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("Calibration report input_bindings must be an object")
    if bindings.get("runner_spec_sha256") != sha256_file(spec_path):
        raise ValueError("Calibration report runner-spec binding changed")
    for key in ("scrna_sha256", "donor_design_sha256", "fates_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(bindings.get(key, ""))):
            raise ValueError(
                f"Calibration report exact-product binding {key!r} is invalid"
            )

    roles = _calibration_artifact_role_paths(report, root)
    publication_benchmark: Mapping[str, Any] | None = None
    if profile_runner:
        if "publication_runner_benchmark" not in roles:
            raise ValueError("Calibration v2 lacks the required runner benchmark artifact")
        benchmark_payload = json.loads(
            roles["publication_runner_benchmark"].read_text(encoding="utf-8")
        )
        if (
            not isinstance(benchmark_payload, Mapping)
            or benchmark_payload != execution.get("publication_runner_benchmark")
            or not math.isfinite(
                float(benchmark_payload.get("wall_clock_seconds", math.nan))
            )
            or float(benchmark_payload.get("wall_clock_seconds", 0.0)) <= 0
            or not math.isfinite(
                float(
                    benchmark_payload.get(
                        "replicate_work_units_per_second", math.nan
                    )
                )
            )
            or float(benchmark_payload.get("replicate_work_units_per_second", 0.0))
            <= 0
            or not math.isclose(
                float(benchmark_payload.get("replicate_work_units_per_second", 0.0)),
                int(benchmark_payload.get("replicate_work_units", 0))
                / float(benchmark_payload.get("wall_clock_seconds", math.inf)),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or int(benchmark_payload.get("seed", -1))
            != int(spec["publication_execution_contract"]["seed"])
            or int(benchmark_payload.get("chunk_size", -1))
            != int(spec["publication_execution_contract"]["chunk_size"])
            or benchmark_payload.get(
                "vectorized_shared_freedman_lane_batch_used"
            )
            is not True
            or int(benchmark_payload.get("mapping_batch_size", -1))
            != int(spec["performance_contract"]["mapping_batch_size"])
        ):
            raise ValueError("Calibration v2 runner benchmark record is invalid")
        publication_benchmark = benchmark_payload
    expected_profile_payload_sha256: str | None = None
    expected_parameters_sha256: str | None = None
    expected_shared_mask_sha256: str | None = None
    expected_residual_mapping_space_size: int | None = None
    expected_restricted_label_space_size: int | None = None
    expected_residual_reference_actual_mode: str | None = None
    expected_residual_reference_enumeration: str | None = None
    expected_residual_null_mappings: int | None = None
    expected_residual_p_resolution: float | None = None
    expected_power_target_effect_standardization_sd: float | None = None
    expected_sensitivity_signature_matrix_sha256: str | None = None
    expected_n_sensitivity_signature_components: int | None = None
    expected_loco_plan_bindings: dict[int, dict[str, Any]] | None = None
    profile_usage_verified = False
    if profile_runner:
        required_profile_roles = {
            "bound_calibration_design_profile",
            "profile_derived_parameters",
        }
        missing_profile_roles = required_profile_roles - set(roles)
        if missing_profile_roles:
            raise ValueError(
                "Calibration report lacks profile evidence roles: "
                f"{sorted(missing_profile_roles)}"
            )
        profile_path = roles["bound_calibration_design_profile"]
        profile = load_calibration_design_profile(profile_path, repository_root=root)
        expected_profile_payload_sha256 = str(
            profile["integrity"]["profile_payload_sha256"]
        )
        if bindings.get("design_profile_sha256") != sha256_file(profile_path):
            raise ValueError("Calibration report design-profile file binding changed")
        if (
            bindings.get("design_profile_payload_sha256")
            != expected_profile_payload_sha256
        ):
            raise ValueError("Calibration report design-profile payload binding changed")
        if bindings.get("trajectory_grid_hash") != profile["input_bindings"][
            "trajectory"
        ]["grid_hash"]:
            raise ValueError("Calibration profile trajectory-grid hash binding changed")
        derived = derive_profile_simulation_parameters(spec, profile)
        expected_parameters_sha256 = derived_profile_parameters_sha256(derived)
        expected_power_target_effect_standardization_sd = float(
            derived["power_target_effect_standardization_sd"]
        )
        expected_sensitivity_signature_matrix_sha256 = str(
            derived["sensitivity_signature_matrix_sha256"]
        )
        expected_n_sensitivity_signature_components = int(
            np.asarray(derived["sensitivity_signature_matrix"], dtype=float).shape[1]
        )
        execution_seed = int(execution.get("seed", -1))
        if execution_seed < 0:
            raise ValueError("Calibration report seed is invalid")
        shared_plan = _make_profile_shared_curve_plan(
            spec,
            derived,
            seed=int(spec["inference"]["residual_mapping_seed"]),
        )
        expected_shared_mask_sha256 = shared_plan.availability_mask_sha256
        expected_residual_mapping_space_size = shared_plan.residual_space_size
        expected_restricted_label_space_size = shared_plan.restricted_label_space_size
        expected_residual_reference_actual_mode = shared_plan.actual_mode
        expected_residual_reference_enumeration = shared_plan.reference_enumeration
        expected_residual_null_mappings = shared_plan.n_null_mappings
        expected_residual_p_resolution = shared_plan.monte_carlo_p_resolution
        expected_loco_plan_bindings = {}
        full_case = np.asarray(derived["observed_assignment"], dtype=bool)
        full_available = np.asarray(
            derived["primary_draw_available_mask"], dtype=bool
        )
        for omitted_control, omitted_index in enumerate(np.flatnonzero(~full_case)):
            loco_gate = _profile_loco_design_gate(
                spec,
                profile,
                derived,
                omitted_index=int(omitted_index),
            )
            keep = np.asarray(loco_gate["keep_mask"], dtype=bool)
            loco_canonical = loco_gate["canonical_design"]
            loco_plan = make_array_freedman_lane_plan(
                reduced_design=np.asarray(
                    loco_canonical.reduced_design, dtype=float
                ),
                condition=full_case[keep],
                available=full_available[keep],
                widths=np.asarray(derived["bin_widths"], dtype=float),
                max_exact_permutations=int(
                    spec["inference"]["max_exhaustive_residual_mappings"]
                ),
                permutation_mode=str(spec["inference"]["residual_reference_mode"]),
                n_permutations=int(
                    spec["inference"]["monte_carlo_residual_mappings"]
                ),
                seed=_seed_for(
                    execution_seed, f"loco-plan:{omitted_control}"
                ),
            )
            expected_loco_plan_bindings[omitted_control] = {
                "availability_mask_sha256": loco_plan.availability_mask_sha256,
                "residual_mapping_space_size": loco_plan.residual_space_size,
                "restricted_label_space_size": loco_plan.restricted_label_space_size,
                "residual_reference_actual_mode": loco_plan.actual_mode,
                "residual_reference_enumeration": loco_plan.reference_enumeration,
                "residual_null_mappings": loco_plan.n_null_mappings,
                "residual_p_resolution": loco_plan.monte_carlo_p_resolution,
                "loco_support_diagnostics_sha256": loco_gate[
                    "diagnostics_sha256"
                ],
                "loco_reduced_design_sha256": loco_gate[
                    "reduced_design_sha256"
                ],
                "loco_terms_sha256": loco_gate["terms_sha256"],
                "loco_encoding_sha256": loco_gate["encoding_sha256"],
            }
        registered_derived = json.loads(
            roles["profile_derived_parameters"].read_text(encoding="utf-8")
        )
        if registered_derived != derived:
            raise ValueError("Profile-derived calibration parameters changed")
        required_bins = int(spec["design"]["required_grid_bins"])
        if (
            int(profile["fixed_grid"]["n_bins"]) != required_bins
            or int(derived["source_n_curve_bins"]) != required_bins
            or int(execution.get("n_curve_bins", -1)) != required_bins
            or int(execution.get("selected_n_curve_bins", -1))
            != int(derived["n_curve_bins"])
            or not 2 <= int(derived["n_curve_bins"]) <= required_bins
            or derived["selected_bin_indices"]
            != list(
                range(
                    int(derived["selected_bin_indices"][0]),
                    int(derived["selected_bin_indices"][0])
                    + int(derived["n_curve_bins"]),
                )
            )
            or int(sum(bool(value) for value in derived["selected_bin_mask"]))
            != int(derived["n_curve_bins"])
            or derived["fixed_20_bin_source_grid_verified"] is not True
            or derived["selected_support_design_valid"] is not True
        ):
            raise ValueError(
                "Calibration profile, derived parameters, and report must all use "
                "the frozen 20-bin common grid"
            )
        if (
            execution.get("design_profile_used") is not True
            or execution.get("design_profile_payload_sha256")
            != expected_profile_payload_sha256
            or execution.get("derived_design_parameters_sha256")
            != expected_parameters_sha256
            or execution.get("label_space_exhaustive") is not True
            or execution.get(
                "finite_sample_exactness_with_continuous_covariates_claimed"
            )
            is not False
            or execution.get("formal_regulation_shared_kernel_used") is not True
            or execution.get("formal_timing_shared_kernel_used") is not True
            or execution.get("formal_power_shared_kernel_used") is not True
            or execution.get("formal_loco_shared_kernel_used") is not True
            or execution.get("formal_loco_fixed_s_support_gate_used") is not True
            or execution.get("vectorized_shared_freedman_lane_batch_used") is not True
            or execution.get("mapping_batch_size")
            != int(spec["performance_contract"]["mapping_batch_size"])
            or execution.get("maximum_replicate_chunk_size")
            != int(
                spec["performance_contract"]["maximum_replicate_chunk_size"]
            )
            or not 1
            <= int(execution.get("chunk_size", 0))
            <= int(
                spec["performance_contract"]["maximum_replicate_chunk_size"]
            )
            or execution.get(
                "formal_occupancy_fate_decomposition_kernel_used"
            )
            is not True
            or execution.get("shared_state_population_draw_used") is not True
            or execution.get("chr21_gene_level_total_trans_projection_used")
            is not True
            or execution.get("fixed_20_bin_source_grid_verified") is not True
            or execution.get("selected_support_design_valid") is not True
            or execution.get("onset_interval_coverage_evaluated") is not False
            or execution.get("onset_confidence_interval_claim_unlocked") is not False
            or execution.get("canonical_donor_design_spec_sha256")
            != derived["canonical_donor_design_spec_sha256"]
            or execution.get("canonical_reduced_design_sha256")
            != derived["support_reduced_design_sha256"]
            or execution.get("canonical_terms_sha256")
            != derived["support_reduced_design_terms_sha256"]
            or execution.get("canonical_encoding_sha256")
            != derived["support_reduced_design_encoding_sha256"]
            or execution.get("sensitivity_signature_matrix_sha256")
            != derived["sensitivity_signature_matrix_sha256"]
            or int(execution.get("n_sensitivity_signature_components", -1))
            != int(
                np.asarray(
                    derived["sensitivity_signature_matrix"], dtype=float
                ).shape[1]
            )
            or execution.get("covariate_stress_signature_effect_injected") is not True
            or not math.isclose(
                float(execution.get("onset_recovery_tolerance", math.nan)),
                float(spec["power"]["onset_recovery_tolerance"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or not math.isclose(
                float(
                    execution.get(
                        "power_target_effect_standardization_sd", math.nan
                    )
                ),
                float(derived["power_target_effect_standardization_sd"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or execution.get("shared_kernel_availability_mask_sha256")
            != expected_shared_mask_sha256
            or execution.get("source_primary_draw_available_mask_sha256")
            != derived["source_primary_draw_available_mask_sha256"]
            or execution.get("selected_bin_mask_sha256")
            != derived["selected_bin_mask_sha256"]
            or execution.get("included_donor_mask_sha256")
            != derived["included_donor_mask_sha256"]
            or execution.get("residual_mapping_space_size")
            != expected_residual_mapping_space_size
            or execution.get("restricted_label_space_size")
            != expected_restricted_label_space_size
            or execution.get("residual_reference_enumeration")
            != shared_plan.reference_enumeration
            or execution.get("residual_reference_actual_mode")
            != shared_plan.actual_mode
            or execution.get("residual_null_mappings")
            != shared_plan.n_null_mappings
            or not math.isclose(
                float(execution.get("residual_p_resolution", math.nan)),
                shared_plan.monte_carlo_p_resolution,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or execution.get("residual_reference_exactness_status")
            != shared_plan.exactness_status
        ):
            raise ValueError("Calibration report does not prove profile use/exactness scope")
        for key in (
            "shared_freedman_lane_kernel_sha256",
            "covariate_pseudobulk_core_sha256",
            "pathway_family_inference_core_sha256",
            "trajectory_decomposition_core_sha256",
            "trajectory_event_timing_core_sha256",
            "t21_covariate_design_core_sha256",
        ):
            if execution.get(key) != derived[key]:
                raise ValueError(f"Calibration report code binding {key!r} changed")
        profile_usage_verified = True
    scenario_replicates = _load_raw_scenario_replicates(
        roles["raw_scenario_replicates"],
        spec,
        expected_profile_payload_sha256=expected_profile_payload_sha256,
        expected_parameters_sha256=expected_parameters_sha256,
        expected_shared_mask_sha256=expected_shared_mask_sha256,
        expected_residual_mapping_space_size=expected_residual_mapping_space_size,
        expected_restricted_label_space_size=expected_restricted_label_space_size,
        expected_residual_reference_actual_mode=expected_residual_reference_actual_mode,
        expected_residual_reference_enumeration=expected_residual_reference_enumeration,
        expected_residual_null_mappings=expected_residual_null_mappings,
        expected_residual_p_resolution=expected_residual_p_resolution,
        expected_sensitivity_signature_matrix_sha256=(
            expected_sensitivity_signature_matrix_sha256
        ),
        expected_n_sensitivity_signature_components=(
            expected_n_sensitivity_signature_components
        ),
    )
    power_replicates, power_replicates_per_point = _load_raw_power_replicates(
        roles["raw_power_replicates"],
        spec,
        expected_profile_payload_sha256=expected_profile_payload_sha256,
        expected_parameters_sha256=expected_parameters_sha256,
        expected_shared_mask_sha256=expected_shared_mask_sha256,
        expected_residual_mapping_space_size=expected_residual_mapping_space_size,
        expected_restricted_label_space_size=expected_restricted_label_space_size,
        expected_residual_reference_actual_mode=expected_residual_reference_actual_mode,
        expected_residual_reference_enumeration=expected_residual_reference_enumeration,
        expected_residual_null_mappings=expected_residual_null_mappings,
        expected_residual_p_resolution=expected_residual_p_resolution,
        expected_power_target_effect_standardization_sd=(
            expected_power_target_effect_standardization_sd
        ),
    )
    loco_replicates, loco_replicates_per_omission = _load_raw_loco_replicates(
        roles["raw_leave_one_control_out_replicates"],
        spec,
        expected_profile_payload_sha256=expected_profile_payload_sha256,
        expected_parameters_sha256=expected_parameters_sha256,
        expected_plan_bindings=expected_loco_plan_bindings,
        expected_power_target_effect_standardization_sd=(
            expected_power_target_effect_standardization_sd
        ),
    )
    if power_replicates_per_point != loco_replicates_per_omission:
        raise ValueError("Power and LOCO raw artifacts use different replicate counts")
    if profile_runner:
        publication = spec["publication_execution_contract"]
        scenario_counts = scenario_replicates.groupby("scenario").size().to_dict()
        for scenario in SCENARIOS:
            expected_count = int(
                publication["complete_null_replicates"]
                if scenario == "complete_null"
                else publication["scenario_replicates"]
            )
            if int(scenario_counts.get(scenario, -1)) != expected_count:
                raise ValueError(
                    f"Raw scenario {scenario!r} changed its frozen replicate count"
                )
        expected_power_count = int(publication["power_replicates_per_point"])
        if (
            power_replicates_per_point != expected_power_count
            or loco_replicates_per_omission != expected_power_count
        ):
            raise ValueError("Raw power artifacts changed their frozen replicate count")
        expected_work_units = int(
            len(scenario_replicates) + len(power_replicates) + len(loco_replicates)
        )
        if (
            publication_benchmark is None
            or int(publication_benchmark.get("replicate_work_units", -1))
            != expected_work_units
        ):
            raise ValueError("Runner benchmark work units differ from raw artifacts")

    scenario_metrics = summarize_scenario_replicates(scenario_replicates)
    power_summary = summarize_power_replicates(power_replicates)
    power_curve = power_summary.loc[
        power_summary["power_kind"].eq("amplitude")
    ].reset_index(drop=True)
    onset_power_curve = power_summary.loc[
        power_summary["power_kind"].eq("onset")
    ].reset_index(drop=True)
    loco_power = summarize_loco_replicates(loco_replicates)
    power_metrics = summarize_calibration_power_metrics(
        scenario_metrics=scenario_metrics,
        power_curve=power_curve,
        onset_power_curve=onset_power_curve,
        loco_power=loco_power,
        spec=spec,
        n_replicates_per_point=power_replicates_per_point,
    )

    reported_scenarios = report.get("scenario_metrics")
    if not isinstance(reported_scenarios, list) or len(reported_scenarios) != len(
        SCENARIOS
    ):
        raise ValueError("Calibration report scenario_metrics must contain six rows")
    if [str(row.get("scenario", "")) for row in reported_scenarios] != list(SCENARIOS):
        raise ValueError(
            "Calibration report scenario order differs from the frozen spec"
        )
    expected_scenarios = scenario_metrics.to_dict("records")
    for actual, expected in zip(reported_scenarios, expected_scenarios):
        if not isinstance(actual, Mapping):
            raise ValueError("Calibration report scenario rows must be objects")
        _strict_metric_mapping_match(
            actual,
            expected,
            label=f"scenario {expected['scenario']!r}",
        )
    reported_power = report.get("power_metrics")
    if not isinstance(reported_power, Mapping):
        raise ValueError("Calibration report power_metrics must be an object")
    _strict_metric_mapping_match(reported_power, power_metrics, label="power")
    return {
        "status": "pass",
        "metric_source": "raw_replicate_tsv_only",
        "trusted_artifact_roles": [
            "raw_scenario_replicates",
            "raw_power_replicates",
            "raw_leave_one_control_out_replicates",
        ],
        "reported_summary_tsvs_trusted": False,
        "design_profile_usage_verified": profile_usage_verified,
        "design_profile_payload_sha256": expected_profile_payload_sha256,
        "derived_design_parameters_sha256": expected_parameters_sha256,
        "n_scenario_replicate_rows": int(len(scenario_replicates)),
        "n_power_replicate_rows": int(len(power_replicates)),
        "n_loco_replicate_rows": int(len(loco_replicates)),
        "complete_null_replicates": int(
            scenario_metrics.set_index("scenario").loc["complete_null", "n_replicates"]
        ),
        "power_replicates_per_point": int(power_replicates_per_point),
        "condition_label_space_size": math.comb(
            int(spec["design"]["n_disomy_controls"])
            + int(spec["design"]["n_t21_cases"]),
            int(spec["design"]["n_disomy_controls"]),
        ),
    }


def write_calibration_outputs(
    result: CalibrationRunResult,
    *,
    output_dir: str | Path,
    repository_root: str | Path,
    bindings: Mapping[str, Any],
    runner_spec_path: str | Path,
    calibration_policy: Mapping[str, Any],
    calibration_policy_path: str | Path,
    analysis_plan_path: str | Path,
    design_profile_path: str | Path | None = None,
    report_id: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Write auditable tables and a report consumable by the formal validator."""
    root = Path(repository_root).resolve()
    output = Path(output_dir).resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "Calibration output_dir must be inside repository_root"
        ) from exc
    report_path = output / "t21_pre_unblinding_calibration_report.json"
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"Calibration output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    table_paths = {
        "scenario_replicates": output / "scenario_replicates.tsv",
        "scenario_metrics": output / "scenario_metrics.tsv",
        "power_replicates": output / "power_replicates.tsv",
        "power_curve": output / "power_curve.tsv",
        "onset_power_curve": output / "onset_power_curve.tsv",
        "loco_replicates": output / "leave_one_control_out_replicates.tsv",
        "loco_power": output / "leave_one_control_out_power.tsv",
        "power_metrics": output / "power_metrics.tsv",
    }
    artifact_roles = {
        "scenario_replicates": "raw_scenario_replicates",
        "scenario_metrics": "reported_scenario_metrics",
        "power_replicates": "raw_power_replicates",
        "power_curve": "reported_amplitude_power_curve",
        "onset_power_curve": "reported_onset_power_curve",
        "loco_replicates": "raw_leave_one_control_out_replicates",
        "loco_power": "reported_leave_one_control_out_power",
        "power_metrics": "reported_power_metrics",
    }
    _write_tsv(result.scenario_replicates, table_paths["scenario_replicates"])
    _write_tsv(result.scenario_metrics, table_paths["scenario_metrics"])
    _write_tsv(result.power_replicates, table_paths["power_replicates"])
    _write_tsv(result.power_curve, table_paths["power_curve"])
    _write_tsv(result.onset_power_curve, table_paths["onset_power_curve"])
    _write_tsv(result.loco_replicates, table_paths["loco_replicates"])
    _write_tsv(result.loco_power, table_paths["loco_power"])
    _write_tsv(pd.DataFrame([result.power_metrics]), table_paths["power_metrics"])

    artifact_rows = []
    for key, path in table_paths.items():
        artifact_rows.append(
            {
                "role": artifact_roles[key],
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    uses_profile = bool(result.metadata.get("design_profile_used", False))
    profile_usage: dict[str, Any] | None = None
    if uses_profile:
        benchmark = result.metadata.get("publication_runner_benchmark")
        if not isinstance(benchmark, Mapping):
            raise ValueError("Runner v2 outputs require a benchmark record")
        benchmark_path = output / "publication_runner_benchmark.json"
        _write_json(benchmark, benchmark_path)
        artifact_rows.append(
            {
                "role": "publication_runner_benchmark",
                "relative_path": benchmark_path.relative_to(root).as_posix(),
                "bytes": benchmark_path.stat().st_size,
                "sha256": sha256_file(benchmark_path),
            }
        )
        if design_profile_path is None:
            raise ValueError("Runner v2 outputs require the bound design-profile path")
        profile_path = Path(design_profile_path).resolve()
        try:
            profile_relative = profile_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("Calibration design profile must be repository-local") from exc
        profile = load_calibration_design_profile(profile_path, repository_root=root)
        profile_payload = str(profile["integrity"]["profile_payload_sha256"])
        if bindings.get("design_profile_sha256") != sha256_file(profile_path):
            raise ValueError("Output bindings differ from the design-profile file")
        if bindings.get("design_profile_payload_sha256") != profile_payload:
            raise ValueError("Output bindings differ from the design-profile payload")
        derived = derive_profile_simulation_parameters(
            load_runner_spec(runner_spec_path), profile
        )
        derived_hash = derived_profile_parameters_sha256(derived)
        if (
            result.metadata.get("design_profile_payload_sha256") != profile_payload
            or result.metadata.get("derived_design_parameters_sha256") != derived_hash
        ):
            raise ValueError("Calibration result metadata differs from the bound profile")
        derived_path = output / "profile_derived_parameters.json"
        _write_json(derived, derived_path)
        artifact_rows.extend(
            [
                {
                    "role": "profile_derived_parameters",
                    "relative_path": derived_path.relative_to(root).as_posix(),
                    "bytes": derived_path.stat().st_size,
                    "sha256": sha256_file(derived_path),
                },
                {
                    "role": "bound_calibration_design_profile",
                    "relative_path": profile_relative.as_posix(),
                    "bytes": profile_path.stat().st_size,
                    "sha256": sha256_file(profile_path),
                },
            ]
        )
        profile_usage = {
            "profile_file_sha256": sha256_file(profile_path),
            "profile_payload_sha256": profile_payload,
            "derived_parameters_sha256": derived_hash,
            "fixed_common_grid_bins": int(derived["source_n_curve_bins"]),
            "selected_common_grid_bins": int(derived["n_curve_bins"]),
            "selected_bin_mask_sha256": str(derived["selected_bin_mask_sha256"]),
            "fixed_20_bin_source_grid_verified": bool(
                derived["fixed_20_bin_source_grid_verified"]
            ),
            "selected_support_design_valid": bool(
                derived["selected_support_design_valid"]
            ),
            "canonical_donor_design_spec_sha256": str(
                derived["canonical_donor_design_spec_sha256"]
            ),
            "canonical_reduced_design_sha256": str(
                derived["support_reduced_design_sha256"]
            ),
            "canonical_terms_sha256": str(
                derived["support_reduced_design_terms_sha256"]
            ),
            "canonical_encoding_sha256": str(
                derived["support_reduced_design_encoding_sha256"]
            ),
            "sensitivity_signature_matrix_sha256": str(
                derived["sensitivity_signature_matrix_sha256"]
            ),
            "n_sensitivity_signature_components": int(
                np.asarray(
                    derived["sensitivity_signature_matrix"], dtype=float
                ).shape[1]
            ),
            "covariate_stress_signature_effect_injected": True,
            "trajectory_grid_hash": str(bindings["trajectory_grid_hash"]),
            "raw_rows_carry_profile_and_parameter_hashes": True,
        }
    elif design_profile_path is not None:
        raise ValueError("Runner v1 outputs may not register a design profile")
    manifest_path = output / "calibration_run_manifest.json"
    manifest = {
        "schema_name": "t21_pre_unblinding_calibration_run_manifest",
        "schema_version": "2.0.0" if uses_profile else "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner_spec_relative_path": Path(runner_spec_path)
        .resolve()
        .relative_to(root)
        .as_posix(),
        "runner_spec_sha256": sha256_file(runner_spec_path),
        "calibration_policy_relative_path": Path(calibration_policy_path)
        .resolve()
        .relative_to(root)
        .as_posix(),
        "calibration_policy_sha256": sha256_file(calibration_policy_path),
        "analysis_plan_relative_path": Path(analysis_plan_path)
        .resolve()
        .relative_to(root)
        .as_posix(),
        "analysis_plan_sha256": sha256_file(analysis_plan_path),
        "blind_gate": {
            "outcome_blinded": True,
            "real_pathway_results_read": False,
            "bindings_only_candidate_input": True,
            "accepted_binding_keys": sorted(bindings),
        },
        "execution": result.metadata,
        "table_artifacts": artifact_rows,
    }
    _write_json(manifest, manifest_path)
    artifact_rows.append(
        {
            "role": "calibration_run_manifest",
            "relative_path": manifest_path.relative_to(root).as_posix(),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
        }
    )
    scenario_records = _json_ready(result.scenario_metrics.to_dict("records"))
    report_bindings = dict(bindings)
    report_bindings["runner_spec_sha256"] = sha256_file(runner_spec_path)
    created = datetime.now(timezone.utc).isoformat()
    publication_v2_report = bool(
        uses_profile
        and result.metadata.get("phase") == "final"
        and result.metadata.get("development_override") is False
    )
    report = {
        "schema_name": (
            "t21_pre_unblinding_calibration_report"
            if (not uses_profile or publication_v2_report)
            else "t21_pre_unblinding_calibration_development_report"
        ),
        "schema_version": (
            "2.0.0"
            if publication_v2_report
            else ("2.0.0-development" if uses_profile else "1.1.0")
        ),
        "report_id": report_id
        or f"t21_six_scenario_{result.metadata['phase']}_{created[:10]}",
        "created_at_utc": created,
        "outcome_blinded": True,
        "calibration_stage": result.metadata["phase"],
        "publication_minima_satisfied": plan_meets_acceptance_minima(
            CalibrationRunPlan(
                phase=str(result.metadata["phase"]),
                complete_null_replicates=int(
                    result.scenario_metrics.set_index("scenario").loc[
                        "complete_null", "n_replicates"
                    ]
                ),
                scenario_replicates=int(
                    result.scenario_metrics.loc[
                        ~result.scenario_metrics["scenario"].eq("complete_null"),
                        "n_replicates",
                    ].min()
                ),
                power_replicates_per_point=int(
                    result.power_metrics["n_replicates_per_point"]
                ),
                development_override=bool(result.metadata["development_override"]),
            ),
            calibration_policy,
        ),
        "input_bindings": report_bindings,
        "scenario_metrics": scenario_records,
        "power_metrics": _json_ready(result.power_metrics),
        "execution": result.metadata,
        "output_artifacts": artifact_rows,
    }
    if profile_usage is not None:
        report["design_profile_usage"] = profile_usage
    _write_json(report, report_path)
    return report_path
