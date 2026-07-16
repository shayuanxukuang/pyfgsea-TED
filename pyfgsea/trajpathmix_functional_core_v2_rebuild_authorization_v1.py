from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import yaml


SCHEMA_NAME = "trajpathmix_functional_core_v2_rebuild_authorization"
SCHEMA_VERSION = "1.0.0"
PROJECT_ID = "COREBENCH_FUNCTIONAL_CORE_V2_REBUILD_AUTHORIZATION_v1"
FROZEN_PAYLOAD_SHA256 = (
    "7cdbeab4eec95b602025781f3f86123495915b82b667eae48034ec23eeb6c246"
)

CONFIG_FILE = "config/trajpathmix_functional_core_v2_rebuild_authorization_v1.yaml"
MODULE_FILE = "pyfgsea/trajpathmix_functional_core_v2_rebuild_authorization_v1.py"
RUNNER_FILE = "scripts/freeze_trajpathmix_functional_core_v2_rebuild_authorization_v1.py"
TEST_FILE = "tests/test_trajpathmix_functional_core_v2_rebuild_authorization_v1.py"
OUTPUT_DIR = "data_external/trajpathmix_functional_core_v2_rebuild_authorization_v1"

AUTHORIZATION_FILE = "trajpathmix_functional_core_v2_rebuild_authorization_v1.json"
PASSPORT_FILE = (
    "trajpathmix_functional_core_v2_rebuild_authorization_material_passport_v1.json"
)
BUILD_RECORD_FILE = (
    "trajpathmix_functional_core_v2_rebuild_authorization_build_record_v1.json"
)
EXACT_OUTPUT_FILES = (AUTHORIZATION_FILE, PASSPORT_FILE, BUILD_RECORD_FILE)

RETIRED_COMPONENTS = {
    "availability_restricted_freedman_lane_as_formal_primary_engine",
    "full_model_residual_simultaneous_bands",
    "nonstudentized_l1_integrated_endpoint",
    "cb2_v1_curve_p_values_for_formal_use",
    "cb2_v1_integrated_p_values_for_formal_use",
    "cb2_v1_family_maxT_p_values_for_formal_use",
    "cb2_v1_by_q_values_for_formal_discovery",
    "cb2_v1_simultaneous_bands_for_formal_use",
    "dynamic_leading_edge_significance_from_cb2_v1",
    "all_cb2_v1_timing_fields_for_formal_use",
    "by_zero_rejections_as_calibration_evidence",
}

V2_0_CHECK_IDS = (
    "lodo_jackknife_matches_direct_ols_leave_one_out_refits",
    "full_sample_fwl_matches_direct_full_ols",
    "centered_influence_sums_zero_and_sumsq_equals_jackknife_variance",
    "donor_order_invariance",
    "pathway_order_invariance",
    "bin_order_invariance",
    "shared_multiplier_across_all_coordinates",
    "multiplier_bank_batching_worker_and_pathway_chunk_invariance",
    "heteroskedastic_null_lodo_jackknife_scale",
    "missing_donor_bin_remains_na",
    "signed_auc_includes_cross_bin_covariance",
    "global_50_x_20_band_scope",
    "simple_randomized_design_near_direct_label_oracle",
    "experiment_fixed_effect_rank_is_deterministic",
    "experiment_overlap_component_webb_sensitivity_is_present",
    "retired_engine_and_timing_firewall",
)


class V2RebuildAuthorizationError(ValueError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _payload_hash(config: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in config.items()
        if key != "frozen_payload_sha256" and not str(key).startswith("_")
    }
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_file(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise V2RebuildAuthorizationError(
            "Authorization bindings must be repository-local"
        ) from exc
    return path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V2RebuildAuthorizationError(message)


def _require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise V2RebuildAuthorizationError(
            f"V2 rebuild authorization mismatch for {label}: "
            f"expected {expected!r}, observed {observed!r}"
        )


def _assert_finite_json(value: Any, label: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_json(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite_json(item, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise V2RebuildAuthorizationError(f"Non-finite JSON value at {label}")


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    _assert_finite_json(value)
    encoded = (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _strict_json_load(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            V2RebuildAuthorizationError(
                f"Non-finite JSON constant {value!r} in {path}"
            )
        ),
    )


def load_v2_rebuild_authorization_config(
    path: str | Path,
) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise V2RebuildAuthorizationError("Authorization config must be a mapping")
    observed = deepcopy(dict(config))
    validate_v2_rebuild_authorization_config(observed)
    return observed


def validate_v2_rebuild_authorization_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    _require_equal(config.get("schema_name"), SCHEMA_NAME, "schema_name")
    _require_equal(config.get("schema_version"), SCHEMA_VERSION, "schema_version")
    _require_equal(config.get("project_id"), PROJECT_ID, "project_id")
    _require_equal(
        config.get("frozen_payload_sha256"),
        FROZEN_PAYLOAD_SHA256,
        "frozen_payload_sha256",
    )
    _require_equal(_payload_hash(config), FROZEN_PAYLOAD_SHA256, "payload_sha256")

    semantics = config["append_only_semantics"]
    for key, expected in {
        "append_only": True,
        "modifies_parent_corebench_contract": False,
        "modifies_cb1_artifacts": False,
        "modifies_cb2_v1_artifacts": False,
        "modifies_cb2_500_failure_localization_v1": False,
        "overwrites_any_prior_evidence": False,
    }.items():
        _require_equal(semantics.get(key), expected, f"append_only_semantics.{key}")

    precedence = config["precedence_contract"]
    for key in (
        "prior_failure_localization_decision_remains_historically_immutable",
        "prior_functional_core_v2_implementation_authorized_false_remains_true_at_its_timestamp",
        "this_artifact_prospectively_authorizes_only_the_scope_stated_here",
        "this_artifact_does_not_reclassify_any_prior_cb2_result",
    ):
        _require_equal(precedence.get(key), True, f"precedence_contract.{key}")

    retirement = config["retirement_decision"]
    _require_equal(
        retirement["current_inferential_engine"]["status"],
        "permanently_closed",
        "current engine status",
    )
    _require_equal(
        retirement["current_inferential_engine"]["may_be_reactivated"],
        False,
        "current engine reactivation",
    )
    _require_equal(
        set(retirement["permanently_retired_components"]),
        RETIRED_COMPONENTS,
        "permanently_retired_components",
    )
    for key in (
        "empirical_scale_correction_of_l1_allowed",
        "variance_floor_patch_allowed",
        "critical_value_factor_patch_allowed",
        "mapping_count_only_patch_allowed",
        "acceptance_threshold_change_allowed",
        "difficult_pathway_donor_or_bin_deletion_allowed",
    ):
        _require_equal(retirement.get(key), False, f"retirement_decision.{key}")

    rebuild = config["rebuild_authorization"]
    for key, expected in {
        "trajpathmix_v2_rebuild_authorized": True,
        "maximum_number_of_rebuilds": 1,
        "current_rebuild_number": 1,
        "v3_inferential_engine_allowed": False,
        "theory_driven_reconstruction_only": True,
        "v2_0_analytic_and_pure_synthetic_implementation_authorized": True,
        "v2_0_real_expression_read_authorized": False,
        "v2_0_current_cb2_500_may_be_used_as_acceptance_set": False,
        "v2_1_new_holdout_assignment_bank_materialization_authorized": False,
        "v2_1_holdout_500_execution_authorized": False,
        "v2_2_screen_or_partial_null_authorized": False,
    }.items():
        _require_equal(rebuild.get(key), expected, f"rebuild_authorization.{key}")

    engine = config["v2_primary_engine"]
    for key, expected in {
        "type": "donor_score_multiplier_bootstrap",
        "influence_correction": "direct_leave_one_donor_out_jackknife",
        "hc3_primary_allowed": False,
        "correction_may_vary_by_bin": False,
        "nuisance_rank_rule": "retain_singular_value_strictly_greater_than_largest_singular_value_times_1e_minus_10",
        "delete_one_condition_information_must_exceed": 1.0e-12,
        "delete_one_minimum_residual_df": 3,
        "delete_one_requires_both_condition_groups_present": True,
        "nuisance_span_may_naturally_lose_rank_after_deletion": True,
        "outcome_or_condition_based_nuisance_column_selection_allowed": False,
        "lodo_rank_or_estimability_failure_action": "fail_closed",
        "full_model_pseudoinverse_or_leverage_clipping_allowed": False,
        "full_sample_point_estimate": "nuisance_span_fwl_ols_condition_coefficient",
        "jackknife_point_estimate_bias_correction_applied": False,
        "jackknife_influence_formula": "sqrt_n_minus_1_over_n_times_mean_leave_one_out_minus_leave_one_out",
        "jackknife_standard_error_formula": "sqrt_sum_squared_centered_jackknife_influence",
        "extra_inverse_sqrt_n_factor_applied": False,
        "donor_multiplier_distribution": "rademacher",
        "formal_future_multiplier_draws": 999,
        "formal_future_multiplier_bank_shape": "999_x_75",
        "multiplier_rows_must_be_unique": False,
        "all_plus_one_or_all_minus_one_rows_excluded": False,
        "studentization": "fixed_observed_jackknife_standard_error",
        "draw_wise_refit_or_studentization": False,
        "plus_one_p_value_rule": "one_plus_exceedances_divided_by_one_plus_draws",
        "p_value_tie_rule": "null_greater_than_or_equal_to_observed",
        "one_multiplier_per_donor_per_draw": True,
        "same_multiplier_shared_across_all_pathways_bins_families_and_signed_auc": True,
        "residual_curve_permutation_used": False,
        "full_model_residual_refit_used": False,
        "missing_donor_bin_score_contribution": "structural_zero_not_outcome_imputation",
    }.items():
        _require_equal(engine.get(key), expected, f"v2_primary_engine.{key}")

    structural = config["frozen_structural_preflight"]
    for key, expected in {
        "assignment_bin_designs_audited": 10000,
        "delete_one_refits_audited": 591500,
        "hc3_one_minus_h_at_or_below_1e_12_count": 41500,
        "assignment_bin_designs_with_hc3_undefined": 9500,
        "all_zero_column_only_lodo_rank_failures": 10000,
        "all_zero_column_only_lodo_failed_assignment_bin_designs": 5000,
        "nuisance_span_fwl_lodo_estimable_count": 591500,
        "nuisance_span_fwl_lodo_failure_count": 0,
        "minimum_delete_one_residual_df": 15,
        "minimum_delete_one_case_donors": 14,
        "minimum_delete_one_control_donors": 15,
        "rank_relative_tolerance": 1.0e-10,
    }.items():
        _require_equal(structural.get(key), expected, f"frozen_structural_preflight.{key}")

    sensitivity = config["mandatory_sensitivity"]
    for key, expected in {
        "type": "experiment_overlap_connected_component_wild_multiplier_sensitivity",
        "cluster_assignment": "connected_components_of_frozen_donor_experiment_bipartite_graph",
        "expected_component_count": 7,
        "expected_component_donor_sizes_descending": [39, 12, 6, 6, 6, 4, 2],
        "donors_with_one_experiment": 49,
        "donors_with_two_experiments": 25,
        "donors_with_three_experiments": 1,
        "dominant_or_primary_experiment_assignment_allowed": False,
        "multiplier_distribution": "webb_six_point",
        "finite_sample_scale": "sqrt(G/(G-1))",
        "same_cluster_multiplier_shared_across_all_coordinates": True,
        "formal_standard_one_way_experiment_cluster_inference_claimed": False,
        "sensitivity_may_be_noninformative_due_to_seven_components": True,
        "standard_experiment_cluster_inference_requires_donor_experiment_bin_reaggregation": True,
        "selecting_more_significant_engine_allowed": False,
        "severe_primary_sensitivity_disagreement_action": "fail_closed",
        "multiway_bootstrap_authorized": False,
    }.items():
        _require_equal(sensitivity.get(key), expected, f"mandatory_sensitivity.{key}")
    _require_equal(
        sensitivity.get("multiplier_probabilities"),
        [1.0 / 6.0] * 6,
        "Webb probabilities",
    )

    endpoints = config["endpoint_contract"]
    _require_equal(
        endpoints.get("primary_curve_statistic"),
        "studentized_supremum_over_supported_bins",
        "primary_curve_statistic",
    )
    signed = endpoints["signed_auc"]
    _require_equal(signed.get("primary"), True, "signed_auc.primary")
    _require_equal(signed.get("absolute_value_applied"), False, "signed_auc absolute")
    _require_equal(
        signed.get("influence_formula"),
        "sum_supported_bin_weight_times_donor_bin_influence",
        "signed_auc influence",
    )
    _require_equal(
        signed.get("cross_bin_covariance_source"),
        "shared_donor_influence_vector",
        "signed_auc covariance",
    )
    _require_equal(signed.get("formal_bin_count"), 20, "signed_auc bins")
    _require_equal(signed.get("formal_supported_bin_weight"), 0.05, "signed_auc weight")
    _require_equal(signed.get("formal_support_coordinates"), 1000, "signed_auc support")
    _require_equal(
        signed.get("pathway_specific_p_reference"),
        "absolute_signed_auc_studentized_against_same_draw_absolute_signed_auc_studentized",
        "signed_auc p reference",
    )
    _require_equal(
        endpoints.get("family_curve_statistic"),
        "max_over_frozen_family_member_pathways_and_supported_bins_of_absolute_studentized_effect",
        "family curve statistic",
    )
    _require_equal(
        endpoints.get("family_curve_reference"),
        "same_draw_max_over_frozen_family_member_pathways_and_supported_bins",
        "family curve reference",
    )
    _require_equal(
        endpoints["simultaneous_band"]["scope"],
        "global_50_pathways_x_20_supported_bins",
        "simultaneous band scope",
    )
    _require_equal(
        endpoints["simultaneous_band"].get("test_and_band_share_same_covariance_process"),
        True,
        "simultaneous covariance process",
    )
    _require_equal(endpoints.get("by_in_primary_acceptance"), False, "BY gate")
    _require_equal(endpoints.get("timing_in_primary_acceptance"), False, "timing gate")
    _require_equal(
        endpoints.get("formal_q95_order_with_999_draws_1based"),
        950,
        "formal q95 order",
    )

    independence = config["independence_contract"]
    _require_equal(
        independence.get("mandatory_sensitivity_unit"),
        "donor_experiment_bipartite_connected_component",
        "sensitivity independence unit",
    )

    checks = config["v2_0_required_checks"]
    _require_equal(tuple(checks["check_ids"]), V2_0_CHECK_IDS, "v2_0 check ids")
    _require_equal(checks.get("all_checks_required"), True, "all checks required")
    _require_equal(
        checks.get("failure_action"),
        "stop_before_holdout_generation",
        "v2_0 failure action",
    )

    holdout = config["future_holdout_contract"]
    for key, expected in {
        "current_cb2_500_role": "development_set_only",
        "new_assignment_bank_required": True,
        "new_assignment_seed_required": True,
        "new_multiplier_stream_required": True,
        "generation_after_v2_implementation_and_freeze_only": True,
        "legal_assignment_requires_all_20_bins_and_all_lodo_deletions_estimable_before_bank_admission": True,
        "target_distribution_is_conditional_on_predeclared_legal_assignment_set": True,
        "assignment_bank_used_as_randomization_p_value_reference": False,
        "interim_method_change_after_holdout_generation_allowed": False,
        "old_cb2_residual_mapping_bank_or_stream_may_be_reused": False,
        "old_cb2_formal_p_q_or_band_outputs_may_be_reused": False,
        "old_cb2_factorized_design_cache_allowed_in_v2_1_runtime": False,
        "old_cb2_assignment_bank_role": "overlap_rejection_list_only",
    }.items():
        _require_equal(holdout.get(key), expected, f"future_holdout_contract.{key}")

    gates = config["future_v2_1_acceptance_gates"]
    _require_equal(gates.get("global_50_curve_weak_fwer_max"), 0.06, "curve gate")
    _require_equal(gates.get("family_13_macro_weak_fwer_max"), 0.06, "family gate")
    _require_equal(
        gates.get("family_13_macro_weak_fwer_denominator"),
        "6500_family_by_assignment_decisions",
        "family gate denominator",
    )
    _require_equal(gates.get("signed_auc_type_i_error_max"), 0.06, "AUC gate")
    _require_equal(
        gates.get("signed_auc_type_i_error_denominator"),
        "25000_pathway_by_assignment_decisions",
        "AUC gate denominator",
    )
    _require_equal(
        gates.get("global_zero_curve_simultaneous_band_coverage_min"),
        0.93,
        "band gate",
    )
    _require_equal(gates.get("timing_participates"), False, "timing acceptance")
    _require_equal(gates.get("by_participates"), False, "BY acceptance")

    state = config["claim_state"]
    false_keys = (
        "functional_core_v2_calibrated",
        "curve_inference_allowed",
        "signed_auc_inference_allowed",
        "simultaneous_band_inference_allowed",
        "family_inference_allowed",
        "timing_claim_allowed",
        "biological_discovery_authorized",
        "real_condition_unblinding_authorized",
        "injection_recovery_authorized",
        "release_authorized",
        "manuscript_method_claim_authorized",
    )
    for key in false_keys:
        _require_equal(state.get(key), False, f"claim_state.{key}")
    _require_equal(
        state.get("pathway_score_preprocessing_uncertainty_propagated"),
        False,
        "preprocessing uncertainty",
    )
    _require_equal(
        state.get("future_interpretation_conditioned_on_frozen_preprocessed_scores"),
        True,
        "conditional preprocessing interpretation",
    )
    _require_equal(state.get("universal_timing_claim"), "closed", "timing claim")
    _require_equal(
        state.get("next_authorized_stage"),
        "v2_0_analytic_and_pure_synthetic_validation_only",
        "next authorized stage",
    )

    output = config["output_contract"]
    _require_equal(output.get("default_output_dir"), OUTPUT_DIR, "output dir")
    _require_equal(output.get("create_only"), True, "create-only")
    _require_equal(output.get("alternate_output_dir_allowed"), False, "alternate output")
    _require_equal(tuple(output["exact_output_files"]), EXACT_OUTPUT_FILES, "output files")

    passport = config["material_passport"]
    for key in (
        "raw_expression_read",
        "real_condition_labels_read",
        "injection_recovery_read",
        "new_assignments_generated",
        "new_multiplier_stream_generated",
        "timing_computed",
        "biological_interpretation_performed",
    ):
        _require_equal(passport.get(key), False, f"material_passport.{key}")
    return {
        "valid": True,
        "project_id": PROJECT_ID,
        "frozen_payload_sha256": FROZEN_PAYLOAD_SHA256,
        "n_retired_components": len(RETIRED_COMPONENTS),
        "n_v2_0_checks": len(V2_0_CHECK_IDS),
    }


def verify_v2_rebuild_authorization_bindings(
    root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    repository = Path(root).resolve()
    observed: dict[str, Any] = {}
    for binding_id, record in config["bindings"].items():
        path = _repo_file(repository, str(record["relative_path"]))
        _require(path.is_file(), f"Missing authorization binding: {path}")
        digest = _hash_file(path)
        _require_equal(digest, record["sha256"], f"binding {binding_id} sha256")
        observed[binding_id] = {
            "relative_path": str(record["relative_path"]),
            "sha256": digest,
            "size_bytes": int(path.stat().st_size),
        }

    failure_decision = _strict_json_load(
        _repo_file(
            repository,
            config["bindings"]["cb2_500_failure_localization_decision_v1"][
                "relative_path"
            ],
        )
    )
    _require_equal(
        failure_decision.get("current_inferential_engine_closed"),
        True,
        "bound failure decision engine closure",
    )
    _require_equal(
        failure_decision.get("causal_mechanism_adjudicated"),
        False,
        "bound causal mechanism status",
    )
    _require_equal(
        failure_decision.get("reference_distribution_mismatch_supported"),
        True,
        "bound reference mismatch",
    )
    _require_equal(
        failure_decision.get("next_stage_authorized"),
        "none",
        "bound prior next stage",
    )
    return observed


def _authorization_payload(
    config: Mapping[str, Any], binding_audit: Mapping[str, Any]
) -> dict[str, Any]:
    keys = (
        "append_only_semantics",
        "precedence_contract",
        "retirement_decision",
        "retained_components",
        "rebuild_authorization",
        "v2_primary_engine",
        "frozen_structural_preflight",
        "mandatory_sensitivity",
        "endpoint_contract",
        "independence_contract",
        "v2_0_required_checks",
        "future_holdout_contract",
        "future_v2_1_acceptance_gates",
        "hard_stop",
        "claim_state",
    )
    payload = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "project_id": PROJECT_ID,
        "frozen_at_utc": config["frozen_at_utc"],
        "frozen_payload_sha256": FROZEN_PAYLOAD_SHA256,
        "authorization_effective": True,
        "binding_audit": deepcopy(dict(binding_audit)),
    }
    payload.update({key: deepcopy(config[key]) for key in keys})
    return payload


def _passport_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    passport = deepcopy(config["material_passport"])
    passport.update(
        {
            "schema_name": "trajpathmix_functional_core_v2_rebuild_authorization_material_passport",
            "schema_version": SCHEMA_VERSION,
            "project_id": PROJECT_ID,
            "frozen_payload_sha256": FROZEN_PAYLOAD_SHA256,
            "verification_status": "ANALYZED",
            "next_stage_authorized": (
                "v2_0_analytic_and_pure_synthetic_validation_only"
            ),
        }
    )
    return passport


def _source_hashes(root: Path) -> dict[str, dict[str, str]]:
    source_paths = {
        "config": CONFIG_FILE,
        "module": MODULE_FILE,
        "runner": RUNNER_FILE,
        "test": TEST_FILE,
    }
    return {
        key: {
            "relative_path": relative_path,
            "sha256": _hash_file(_repo_file(root, relative_path)),
        }
        for key, relative_path in source_paths.items()
    }


def _build_record_payload(
    *,
    config: Mapping[str, Any],
    binding_audit: Mapping[str, Any],
    artifact_hashes: Mapping[str, str],
    source_hashes: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    return {
        "schema_name": "trajpathmix_functional_core_v2_rebuild_authorization_build_record",
        "schema_version": SCHEMA_VERSION,
        "project_id": PROJECT_ID,
        "frozen_at_utc": config["frozen_at_utc"],
        "frozen_payload_sha256": FROZEN_PAYLOAD_SHA256,
        "artifact_sha256": deepcopy(dict(artifact_hashes)),
        "source_sha256": deepcopy(dict(source_hashes)),
        "binding_sha256": {
            key: value["sha256"] for key, value in binding_audit.items()
        },
        "raw_expression_read": False,
        "real_condition_labels_read": False,
        "new_assignments_generated": False,
        "new_multiplier_stream_generated": False,
        "automatic_retry_used": False,
        "automatic_resume_used": False,
        "verification_status": "ANALYZED",
    }


def materialize_v2_rebuild_authorization(
    *,
    repository_root: str | Path,
    config_path: str | Path,
    explicit_execution_authorization: bool,
) -> dict[str, Any]:
    _require(
        explicit_execution_authorization,
        "Explicit v2 rebuild authorization materialization flag is required",
    )
    root = Path(repository_root).resolve()
    expected_config = _repo_file(root, CONFIG_FILE)
    observed_config = Path(config_path).resolve()
    _require_equal(observed_config, expected_config, "config path")
    config = load_v2_rebuild_authorization_config(observed_config)
    binding_audit = verify_v2_rebuild_authorization_bindings(root, config)

    target = _repo_file(root, OUTPUT_DIR)
    staging = target.with_name(target.name + ".incomplete")
    _require(not target.exists(), f"Authorization output already exists: {target}")
    _require(not staging.exists(), f"Authorization staging output already exists: {staging}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        authorization = _authorization_payload(config, binding_audit)
        passport = _passport_payload(config)
        _write_json(authorization, staging / AUTHORIZATION_FILE)
        _write_json(passport, staging / PASSPORT_FILE)

        artifact_hashes = {
            name: _hash_file(staging / name)
            for name in (AUTHORIZATION_FILE, PASSPORT_FILE)
        }
        source_hashes = _source_hashes(root)
        build_record = _build_record_payload(
            config=config,
            binding_audit=binding_audit,
            artifact_hashes=artifact_hashes,
            source_hashes=source_hashes,
        )
        _write_json(build_record, staging / BUILD_RECORD_FILE)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return validate_v2_rebuild_authorization_output(
        repository_root=root,
        config_path=observed_config,
    )


def validate_v2_rebuild_authorization_output(
    *, repository_root: str | Path, config_path: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = load_v2_rebuild_authorization_config(config_path)
    binding_audit = verify_v2_rebuild_authorization_bindings(root, config)
    target = _repo_file(root, OUTPUT_DIR)
    _require(target.is_dir(), f"Missing authorization output: {target}")
    observed_files = tuple(sorted(path.name for path in target.iterdir() if path.is_file()))
    _require_equal(observed_files, tuple(sorted(EXACT_OUTPUT_FILES)), "output file set")
    _require(
        not any(path.is_dir() for path in target.iterdir()),
        "Authorization output must not contain subdirectories",
    )

    authorization = _strict_json_load(target / AUTHORIZATION_FILE)
    passport = _strict_json_load(target / PASSPORT_FILE)
    build = _strict_json_load(target / BUILD_RECORD_FILE)
    expected_authorization = _authorization_payload(config, binding_audit)
    expected_passport = _passport_payload(config)
    _require_equal(
        authorization,
        expected_authorization,
        "complete published authorization payload",
    )
    _require_equal(
        passport,
        expected_passport,
        "complete published material passport",
    )

    expected_artifact_hashes = {
        name: _hash_file(target / name)
        for name in (AUTHORIZATION_FILE, PASSPORT_FILE)
    }
    expected_source_hashes = _source_hashes(root)
    expected_build = _build_record_payload(
        config=config,
        binding_audit=binding_audit,
        artifact_hashes=expected_artifact_hashes,
        source_hashes=expected_source_hashes,
    )
    _require_equal(build, expected_build, "complete published build record")
    return {
        "valid": True,
        "project_id": PROJECT_ID,
        "output_dir": str(target),
        "frozen_payload_sha256": FROZEN_PAYLOAD_SHA256,
        "authorization_sha256": _hash_file(target / AUTHORIZATION_FILE),
        "build_record_sha256": _hash_file(target / BUILD_RECORD_FILE),
    }


__all__ = [
    "AUTHORIZATION_FILE",
    "BUILD_RECORD_FILE",
    "CONFIG_FILE",
    "EXACT_OUTPUT_FILES",
    "FROZEN_PAYLOAD_SHA256",
    "OUTPUT_DIR",
    "PASSPORT_FILE",
    "PROJECT_ID",
    "V2RebuildAuthorizationError",
    "load_v2_rebuild_authorization_config",
    "materialize_v2_rebuild_authorization",
    "validate_v2_rebuild_authorization_config",
    "validate_v2_rebuild_authorization_output",
    "verify_v2_rebuild_authorization_bindings",
]
