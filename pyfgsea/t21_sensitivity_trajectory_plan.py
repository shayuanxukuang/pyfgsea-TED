"""Fail-closed plan validation for the CD235a-negative trajectory audit.

Only frozen configuration and structural method evidence are read here.  The
module has no pathway-score, differential-expression, or condition-effect input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from .t21_data_product import sha256_file
from .t21_sensitivity_candidate import ANALYSIS_ROLE, SAMPLING_FRAME_ID
from .t21_trajectory_fate import load_trajectory_fate_plan


SENSITIVITY_TRAJECTORY_PLAN_ID = "t21_cd235a_neg_condition_blind_trajectory_v1"
SENSITIVITY_ANALYSIS_PLAN_ID = (
    "t21_cd235a_neg_terminal_amplitude_rescue_analysis_plan_v1"
)
SENSITIVITY_ANALYSIS_PLAN_SCHEMA = (
    "t21_outcome_blind_terminal_amplitude_rescue_analysis_plan"
)
SENSITIVITY_TRAJECTORY_PLAN_RELATIVE_PATH = (
    "config/t21_trajectory_fate_plan_cd235a_neg_sensitivity_v1.yaml"
)
TERMINAL_AMPLITUDE_DESIGN_CONTRACT_RELATIVE_PATH = (
    "config/t21_terminal_amplitude_design_atlas_v1.yaml"
)
SENSITIVITY_TRAJECTORY_VALIDATOR_RELATIVE_PATH = (
    "pyfgsea/t21_sensitivity_trajectory_plan.py"
)


def validate_sensitivity_trajectory_configuration(
    repository_root: str | Path,
    *,
    plan_path: str | Path,
    analysis_plan_path: str | Path,
) -> dict[str, Any]:
    """Validate the frozen sensitivity trajectory and terminal-amplitude binding."""
    root = Path(repository_root).resolve()
    trajectory_path = Path(plan_path).resolve()
    rescue_path = Path(analysis_plan_path).resolve()
    for path, label in (
        (trajectory_path, "sensitivity trajectory plan"),
        (rescue_path, "terminal-amplitude rescue analysis plan"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")

    plan = load_trajectory_fate_plan(trajectory_path, repository_root=root)
    scope = plan.get("scope")
    interpretation = plan.get("interpretation")
    if not isinstance(scope, Mapping) or not isinstance(interpretation, Mapping):
        raise ValueError("Sensitivity trajectory plan lacks its frozen scope")
    expected_scope = {
        "sampling_frame_id": SAMPLING_FRAME_ID,
        "analysis_role": ANALYSIS_ROLE,
        "condition_used_for_inference": False,
        "candidate_pathway_genes_used_for_inference": False,
        "selection_uses_pathway_outcomes": False,
        "pooling_with_primary_allowed": False,
        "primary_discovery_claim_allowed": False,
        "formal_release_allowed": False,
    }
    if (
        plan.get("plan_id") != SENSITIVITY_TRAJECTORY_PLAN_ID
        or plan.get("real_pathway_outcomes_read") is not False
        or any(scope.get(key) != value for key, value in expected_scope.items())
    ):
        raise ValueError("Sensitivity trajectory plan violates its diagnostic scope")
    expected_interpretation = {
        "trajectory_role": "condition_blind_sampling_frame_and_support_axis_only",
        "allowed_estimand_class": "terminal_window_amplitude_only",
        "timing_or_dynamics_claims_allowed": False,
        (
            "onset_duration_phase_shift_early_late_transient_sustained_"
            "or_heterochrony_allowed"
        ): False,
        "unsupported_region_extrapolation_allowed": False,
    }
    if any(
        interpretation.get(key) != value
        for key, value in expected_interpretation.items()
    ):
        raise ValueError("Sensitivity trajectory interpretation reopens timing")

    draw_ids = [str(draw.get("trajectory_draw_id", "")) for draw in plan["draws"]]
    expected_draw_ids = [
        "dpt_primary_k15",
        "dpt_root_centroid_k15",
        "dpt_graph_k10",
        "dpt_graph_k30",
        "dpt_donor_bootstrap",
        "dpt_reference_resample_map30",
        "dpt_terminal_expanded_q50",
        "knn_geodesic_second_method",
    ]
    grid = plan["fixed_common_pseudotime_grid"]
    fate_order = [str(value) for value in plan["fate_model"]["fate_order"]]
    if (
        draw_ids != expected_draw_ids
        or len(grid["bin_left"]) != 20
        or len(grid["bin_center"]) != 20
        or len(grid["bin_right"]) != 20
        or fate_order != ["erythroid", "megakaryocyte", "myeloid", "other"]
    ):
        raise ValueError("Sensitivity trajectory changed the frozen draw/grid/fate design")

    analysis_plan = yaml.safe_load(rescue_path.read_text(encoding="utf-8"))
    if not isinstance(analysis_plan, Mapping):
        raise ValueError("Terminal-amplitude rescue analysis plan must be a mapping")
    if (
        analysis_plan.get("schema_name") != SENSITIVITY_ANALYSIS_PLAN_SCHEMA
        or str(analysis_plan.get("schema_version")) != "1.0.0"
        or analysis_plan.get("plan_id") != SENSITIVITY_ANALYSIS_PLAN_ID
        or analysis_plan.get("outcome_blinded_at_freeze") is not True
        or analysis_plan.get("real_pathway_outcomes_read") is not False
        or analysis_plan.get("selection_uses_pathway_outcomes") is not False
    ):
        raise ValueError("Terminal-amplitude rescue analysis plan is not outcome-blind")

    analysis_scope = analysis_plan.get("scope")
    trajectory_binding = analysis_plan.get("trajectory")
    estimand = analysis_plan.get("estimand")
    design_binding = analysis_plan.get("design_contract")
    validation_binding = analysis_plan.get("validation_implementation")
    firewall = analysis_plan.get("outcome_firewall")
    if not all(
        isinstance(value, Mapping)
        for value in (
            analysis_scope,
            trajectory_binding,
            estimand,
            design_binding,
            validation_binding,
            firewall,
        )
    ):
        raise ValueError("Terminal-amplitude rescue analysis plan is incomplete")
    expected_analysis_scope = {
        "sampling_frame_id": SAMPLING_FRAME_ID,
        "analysis_role": ANALYSIS_ROLE,
        "pooling_with_primary_allowed": False,
        "primary_discovery_claim_allowed": False,
        "formal_release_allowed": False,
    }
    if any(
        analysis_scope.get(key) != value
        for key, value in expected_analysis_scope.items()
    ):
        raise ValueError("Terminal-amplitude rescue analysis scope is not diagnostic-only")

    trajectory_sha256 = sha256_file(trajectory_path)
    expected_trajectory_binding = {
        "frozen_implementation_plan_id": SENSITIVITY_TRAJECTORY_PLAN_ID,
        "frozen_implementation_plan_path": SENSITIVITY_TRAJECTORY_PLAN_RELATIVE_PATH,
        "frozen_implementation_plan_sha256": trajectory_sha256,
        "exact_author_reproduction_claim_allowed": False,
        "condition_information_used_for_inference": False,
        "candidate_pathway_genes_used_for_inference": False,
        "role": "condition_blind_sampling_frame_and_support_axis_only",
    }
    if any(
        trajectory_binding.get(key) != value
        for key, value in expected_trajectory_binding.items()
    ):
        raise ValueError("Terminal-amplitude analysis plan has a stale trajectory binding")

    forbidden_estimands = {
        "onset",
        "duration",
        "phase_shift",
        "early_late",
        "transient_sustained",
        "heterochrony",
    }
    expected_public_labels = [
        "terminal-window conditional regulation",
        "terminal-window state occupancy",
        "terminal fate composition",
    ]
    if (
        estimand.get("allowed_class") != "terminal_window_amplitude_only"
        or estimand.get("donor_level_target")
        != "average_donor_terminal_window_condition_difference"
        or estimand.get("allowed_public_labels") != expected_public_labels
        or estimand.get("timing_or_dynamics_estimands_allowed") is not False
        or set(estimand.get("forbidden_estimands", [])) != forbidden_estimands
    ):
        raise ValueError("Terminal-amplitude analysis plan permits another estimand")

    design_contract_path = root / TERMINAL_AMPLITUDE_DESIGN_CONTRACT_RELATIVE_PATH
    if (
        design_binding.get("contract_id")
        != "t21_terminal_amplitude_sampling_frame_design_atlas_v1"
        or design_binding.get("path")
        != TERMINAL_AMPLITUDE_DESIGN_CONTRACT_RELATIVE_PATH
        or design_binding.get("sha256") != sha256_file(design_contract_path)
        or design_binding.get("maximum_blind_method_amendments") != 1
        or design_binding.get("amendment_id")
        != "t21_terminal_amplitude_additive_variance_v1"
    ):
        raise ValueError("Terminal-amplitude design-contract binding changed")
    validation_path = root / SENSITIVITY_TRAJECTORY_VALIDATOR_RELATIVE_PATH
    if (
        validation_binding.get("path")
        != SENSITIVITY_TRAJECTORY_VALIDATOR_RELATIVE_PATH
        or validation_binding.get("sha256") != sha256_file(validation_path)
    ):
        raise ValueError("Terminal-amplitude validation implementation binding changed")
    expected_firewall = {
        "pathway_condition_effects_may_be_computed_during_trajectory_build": False,
        "pathway_scores_may_be_read_during_sampling_frame_selection": False,
        "differential_expression_may_be_read_during_sampling_frame_selection": False,
        "unsupported_pseudotime_regions_may_be_extrapolated": False,
    }
    if dict(firewall) != expected_firewall:
        raise ValueError("Terminal-amplitude outcome firewall must remain closed")

    return {
        "status": "pass_outcome_blind_terminal_amplitude_trajectory_contract",
        "plan_id": SENSITIVITY_TRAJECTORY_PLAN_ID,
        "sampling_frame_id": SAMPLING_FRAME_ID,
        "analysis_role": ANALYSIS_ROLE,
        "allowed_estimand_class": "terminal_window_amplitude_only",
        "n_draws": len(draw_ids),
        "n_bins": len(grid["bin_left"]),
        "fate_order": fate_order,
        "trajectory_plan_sha256": trajectory_sha256,
        "analysis_plan_sha256": sha256_file(rescue_path),
        "real_pathway_outcomes_read": False,
    }
