from __future__ import annotations

import json

import pytest

from pyfgsea.ted_evidence import (
    EventSupportInputs,
    EventSupportThresholds,
    ParallelEvidenceRecord,
    ReplicationFacetInputs,
    ValidationProvenanceInputs,
    assign_event_support,
    assign_evidence_boundary,
    assign_replication_facets,
    assign_validation_provenance,
)


def test_parallel_outcome_record_does_not_promote_failed_event() -> None:
    event = assign_event_support(
        EventSupportInputs(
            event_family_declared=True,
            defensible_null_specified=True,
            biological_units_present=True,
            condition_batch_confounded=False,
            identifiability_status="identifiable",
            artifact_dominated=True,
            event_q=0.20,
            retained_module=True,
            basic_controls_pass=False,
        )
    )
    outcome = ParallelEvidenceRecord(
        record_id="bnt162b2_cd64_cd169",
        evidence_type="orthogonal_outcome",
        status="passed",
        independence_context="same_study_same_cells",
        outcome_type="protein",
        contrast="early post-dose minus baseline/late",
        controls_pass=True,
        replication_status="not_tested",
        reason_codes=("PROTEIN_OUTCOME_GATES_PASS",),
    )
    assert event.code == "E0"
    assert outcome.as_dict()["status"] == "passed"
    assert outcome.as_dict()["replication_status"] == "not_tested"


@pytest.mark.parametrize(
    "evidence_type",
    ["orthogonal_outcome", "intervention_reversal", "matched_rescue"],
)
def test_parallel_evidence_types_never_upgrade_event_e_code(
    evidence_type: str,
) -> None:
    event = assign_event_support(
        EventSupportInputs(
            event_family_declared=True,
            defensible_null_specified=True,
            biological_units_present=True,
            condition_batch_confounded=False,
            identifiability_status="identifiable",
            artifact_dominated=True,
            event_q=0.20,
            retained_module=True,
            basic_controls_pass=False,
        )
    )
    record = ParallelEvidenceRecord(
        record_id=f"{evidence_type}_record",
        evidence_type=evidence_type,  # type: ignore[arg-type]
        status="passed",
        independence_context="independent_experiment",
        controls_pass=True,
        reason_codes=("PARALLEL_EVIDENCE_PASSED",),
    )

    assert event.code == "E0"
    assert record.as_dict()["status"] == "passed"


def test_passed_parallel_record_requires_controls() -> None:
    with pytest.raises(ValueError, match="controls_pass=True"):
        ParallelEvidenceRecord(
            record_id="record",
            evidence_type="orthogonal_outcome",
            status="passed",
            independence_context="same_study_same_cells",
            controls_pass=False,
        )


def test_tested_parallel_record_replication_requires_dataset() -> None:
    with pytest.raises(ValueError, match="replication_dataset_id"):
        ParallelEvidenceRecord(
            record_id="record",
            evidence_type="orthogonal_outcome",
            status="failed",
            independence_context="same_study_same_cells",
            controls_pass=False,
            replication_status="failed",
        )


def test_replication_facets_keep_event_and_protein_outcome_separate() -> None:
    result = assign_replication_facets(
        ReplicationFacetInputs(
            event_analysis_complete=True,
            event_replication_tested=True,
            independent_cohort=True,
            same_event_family=True,
            early_activation_same_direction=True,
            recovery_same_direction=True,
            evaluable_donor_direction_fraction=5 / 6,
            family_adjusted_p=0.08,
            gates_frozen=True,
            additional_declared_gates_pass=True,
            outcome_analysis_complete=True,
            outcome_replication_tested=False,
            outcome_modality_compatible=False,
            outcome_type="protein",
        )
    )
    assert result.event_replication_eligibility_status == "passed"
    assert result.event_replication_test_status == "run_supported"
    assert result.event_replication_status == "passed"
    assert result.outcome_replication_status == "not_tested"
    assert result.display("E2", within_study_outcome_status="passed") == (
        "E2 | protein outcome passed | event independently replicated"
    )


def test_replication_display_requires_independent_outcome_pass_for_strong_text() -> (
    None
):
    result = assign_replication_facets(
        ReplicationFacetInputs(
            event_analysis_complete=True,
            event_replication_tested=True,
            independent_cohort=True,
            same_event_family=True,
            early_activation_same_direction=True,
            recovery_same_direction=True,
            evaluable_donor_direction_fraction=1.0,
            family_adjusted_p=0.02,
            gates_frozen=True,
            additional_declared_gates_pass=True,
            outcome_analysis_complete=True,
            outcome_replication_tested=True,
            outcome_modality_compatible=True,
            same_outcome_contrast=True,
            outcome_replication_pass=True,
            outcome_type="protein",
        )
    )
    assert result.event_replication_eligibility_status == "passed"
    assert result.event_replication_test_status == "run_supported"
    assert result.outcome_replication_status == "passed"
    assert result.display("E2", within_study_outcome_status="passed") == (
        "E2 | protein outcome passed and independently replicated"
    )


def test_outcome_replication_is_independent_of_event_replication_status() -> None:
    result = assign_replication_facets(
        ReplicationFacetInputs(
            event_analysis_complete=True,
            event_replication_tested=True,
            independent_cohort=True,
            same_event_family=True,
            early_activation_same_direction=False,
            recovery_same_direction=False,
            evaluable_donor_direction_fraction=0.50,
            family_adjusted_p=0.40,
            gates_frozen=True,
            additional_declared_gates_pass=False,
            outcome_analysis_complete=True,
            outcome_replication_tested=True,
            outcome_modality_compatible=True,
            same_outcome_contrast=True,
            outcome_replication_pass=True,
            outcome_type="protein",
        )
    )
    assert result.event_replication_eligibility_status == "passed"
    assert result.event_replication_test_status == "run_not_supported"
    assert result.event_replication_status == "failed"
    assert result.outcome_replication_status == "passed"
    assert result.display("E2", within_study_outcome_status="passed") == (
        "E2 | protein outcome passed and independently replicated"
    )


def test_replication_facet_analysis_is_pending_until_completed() -> None:
    result = assign_replication_facets(ReplicationFacetInputs())
    assert result.event_replication_eligibility_status == "pending"
    assert result.event_replication_test_status == "not_run"
    assert result.event_replication_status == "pending"
    assert result.outcome_replication_status == "pending"


def test_replication_prerequisite_failure_is_not_evaluable_and_not_run() -> None:
    result = assign_replication_facets(
        ReplicationFacetInputs(
            event_analysis_complete=True,
            event_replication_tested=False,
            additional_declared_gates_pass=False,
            outcome_analysis_complete=True,
            outcome_replication_tested=False,
            outcome_modality_compatible=False,
        )
    )
    assert result.event_replication_eligibility_status == "failed"
    assert result.event_replication_test_status == "not_run"
    assert result.event_replication_status == "not_evaluable"
    assert "EVENT_REPLICATION_FAILED_AT_ELIGIBILITY_PREREQUISITE" in result.reason_codes


def e1_inputs(**updates: object) -> EventSupportInputs:
    values = {
        "event_family_declared": True,
        "defensible_null_specified": True,
        "biological_units_present": True,
        "condition_batch_confounded": False,
        "identifiability_status": "identifiable",
        "artifact_dominated": False,
        "event_q": 0.05,
        "retained_module": True,
        "basic_controls_pass": True,
    }
    values.update(updates)
    return EventSupportInputs(**values)


def e2_inputs(**updates: object) -> EventSupportInputs:
    values = {
        "effective_blocks": 3,
        "block_support_method": "monte_carlo_block_permutation",
        "minimum_attainable_p": 0.001,
        "permutation_resolution_pass": True,
        "block_q": 0.08,
        "direction_stability": 0.80,
        "mode_identifiable": True,
        "negative_control_pass": True,
        "negative_control_margin": 0.01,
    }
    values.update(updates)
    return e1_inputs(**values)


def test_missing_mandatory_event_input_fails_closed_to_e0() -> None:
    result = assign_event_support(e1_inputs(event_q=None))

    assert result.code == "E0"
    assert "EVENT_Q_MISSING" in result.reason_codes
    assert any(
        gate.gate == "event_q" and gate.status == "missing"
        for gate in result.audit_trail
    )


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"identifiability_status": "not_identifiable"}, "EVENT_NOT_IDENTIFIABLE"),
        ({"artifact_dominated": True}, "EVENT_ARTIFACT_DOMINATED"),
        ({"condition_batch_confounded": True}, "CONDITION_BATCH_COMPLETELY_CONFOUNDED"),
        ({"event_q": 0.10001}, "EVENT_Q_ABOVE_THRESHOLD"),
        ({"event_q": float("nan")}, "EVENT_Q_INVALID"),
    ],
)
def test_hard_event_gates_return_e0(updates: dict[str, object], reason: str) -> None:
    result = assign_event_support(e2_inputs(**updates))

    assert result.code == "E0"
    assert reason in result.reason_codes


def test_supported_event_without_robustness_inputs_is_e1() -> None:
    result = assign_event_support(e1_inputs())

    assert result.code == "E1"
    assert "EFFECTIVE_BLOCKS_MISSING" in result.reason_codes
    assert "BLOCK_SUPPORT_MISSING" in result.reason_codes
    assert "NEGATIVE_CONTROL_STATUS_MISSING" in result.reason_codes


def test_complete_default_robustness_gates_assign_e2() -> None:
    result = assign_event_support(e2_inputs())

    assert result.code == "E2"
    assert result.reason_codes == ("E2_ALL_GATES_PASS",)


def test_exact_three_block_sign_permutation_is_not_testable_at_q_point_10() -> None:
    result = assign_event_support(
        e2_inputs(
            block_support_method="exact_paired_sign_permutation",
            minimum_attainable_p=0.125,
            permutation_resolution_pass=False,
        )
    )

    assert result.code == "E0"
    assert "INSUFFICIENT_PERMUTATION_RESOLUTION" in result.reason_codes


def test_exact_four_block_sign_permutation_can_attain_q_point_10() -> None:
    result = assign_event_support(
        e2_inputs(
            effective_blocks=4,
            block_support_method="exact_paired_sign_permutation",
            minimum_attainable_p=0.0625,
            permutation_resolution_pass=True,
        )
    )

    assert result.code == "E2"


def test_three_blocks_remain_allowed_for_selection_aware_ci() -> None:
    result = assign_event_support(
        e2_inputs(
            block_support_method="selection_aware_ci",
            minimum_attainable_p=None,
            permutation_resolution_pass=None,
            block_q=None,
            block_ci_excludes_zero=True,
        )
    )

    assert result.code == "E2"


def test_block_confidence_interval_is_an_allowed_alternative_to_block_q() -> None:
    result = assign_event_support(e2_inputs(block_q=None, block_ci_excludes_zero=True))

    assert result.code == "E2"
    assert any(
        gate.gate == "block_support" and gate.status == "pass"
        for gate in result.audit_trail
    )


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"effective_blocks": 2}, "INSUFFICIENT_EFFECTIVE_BLOCKS"),
        ({"block_q": 0.11}, "BLOCK_SUPPORT_FAIL"),
        ({"direction_stability": 0.79}, "DIRECTION_STABILITY_BELOW_THRESHOLD"),
        ({"mode_identifiable": False}, "MODE_NOT_IDENTIFIABLE_FOR_E2"),
        ({"negative_control_pass": False}, "NEGATIVE_CONTROL_GATES_FAIL"),
        ({"negative_control_margin": 0.0}, "NEGATIVE_CONTROL_MARGIN_NONPOSITIVE"),
    ],
)
def test_failed_robustness_gate_caps_support_at_e1(
    updates: dict[str, object], reason: str
) -> None:
    result = assign_event_support(e2_inputs(**updates))

    assert result.code == "E1"
    assert reason in result.reason_codes


def test_required_matched_state_overlap_fails_closed_and_attenuation_caps_e2() -> None:
    no_overlap = assign_event_support(
        e2_inputs(matched_state_required=True, matched_state_overlap_pass=False)
    )
    attenuated = assign_event_support(
        e2_inputs(
            matched_state_required=True,
            matched_state_overlap_pass=True,
            matched_state_attenuation=0.50,
        )
    )

    assert no_overlap.code == "E0"
    assert "MATCHED_STATE_OVERLAP_FAIL" in no_overlap.reason_codes
    assert attenuated.code == "E1"
    assert "MATCHED_STATE_ATTENUATION_TOO_LARGE" in attenuated.reason_codes


def test_thresholds_are_explicit_and_configurable() -> None:
    strict = EventSupportThresholds(event_q_max=0.05)

    assert assign_event_support(e2_inputs(event_q=0.05), thresholds=strict).code == "E2"
    failed = assign_event_support(e2_inputs(event_q=0.05001), thresholds=strict)
    assert failed.code == "E0"
    assert "EVENT_Q_ABOVE_THRESHOLD" in failed.reason_codes


def test_no_observed_validation_provenance_is_v0() -> None:
    result = assign_validation_provenance(ValidationProvenanceInputs())

    assert result.code == "V0"
    assert result.reason_codes == ("V0_COMPUTATIONAL_ONLY",)


def test_invalid_observation_flag_fails_closed_with_reason() -> None:
    result = assign_validation_provenance(
        ValidationProvenanceInputs(orthogonal_outcome_observed=None)  # type: ignore[arg-type]
    )

    assert result.code == "V0"
    assert "V1_OBSERVATION_FLAG_INVALID" in result.reason_codes


def test_incomplete_observed_outcome_fails_closed_to_v0() -> None:
    result = assign_validation_provenance(
        ValidationProvenanceInputs(
            orthogonal_outcome_observed=True,
            outcome_assessment_prespecified=True,
            outcome_aligned=True,
            outcome_controls_pass=None,
        )
    )

    assert result.code == "V0"
    assert "V1_OUTCOME_CONTROLS_STATUS_MISSING" in result.reason_codes


def test_complete_outcome_and_intervention_gates_assign_v1_and_v2() -> None:
    v1 = assign_validation_provenance(
        ValidationProvenanceInputs(
            orthogonal_outcome_observed=True,
            outcome_assessment_prespecified=True,
            outcome_aligned=True,
            outcome_controls_pass=True,
        )
    )
    v2 = assign_validation_provenance(
        ValidationProvenanceInputs(
            intervention_reversal_observed=True,
            intervention_contrast_prespecified=True,
            intervention_reversal_pass=True,
            matched_intervention_controls_pass=True,
        )
    )

    assert v1.code == "V1"
    assert v2.code == "V2"


def test_v3_requires_same_system_predicted_recovery_and_controls() -> None:
    incomplete = assign_validation_provenance(
        ValidationProvenanceInputs(
            matched_rescue_observed=True,
            rescue_same_system=False,
            predicted_readout_recovered=True,
            matched_rescue_controls_pass=True,
        )
    )
    complete = assign_validation_provenance(
        ValidationProvenanceInputs(
            matched_rescue_observed=True,
            rescue_same_system=True,
            predicted_readout_recovered=True,
            matched_rescue_controls_pass=True,
        )
    )

    assert incomplete.code == "V0"
    assert "V3_RESCUE_NOT_SAME_SYSTEM" in incomplete.reason_codes
    assert complete.code == "V3"


def test_v4_requires_independence_success_and_recorded_basis() -> None:
    missing_basis = assign_validation_provenance(
        ValidationProvenanceInputs(
            independent_replication_observed=True,
            replication_independent=True,
            independent_replication_pass=True,
        )
    )
    complete = assign_validation_provenance(
        ValidationProvenanceInputs(
            independent_replication_observed=True,
            replication_independent=True,
            independent_replication_pass=True,
            replicated_validation_basis="V2",
        )
    )

    assert missing_basis.code == "V0"
    assert "V4_REPLICATED_BASIS_MISSING" in missing_basis.reason_codes
    assert complete.code == "V4"
    assert complete.replicated_validation_basis == "V2"


def test_failed_higher_candidate_does_not_erase_complete_lower_provenance() -> None:
    result = assign_validation_provenance(
        ValidationProvenanceInputs(
            orthogonal_outcome_observed=True,
            outcome_assessment_prespecified=True,
            outcome_aligned=True,
            outcome_controls_pass=True,
            matched_rescue_observed=True,
            rescue_same_system=None,
            predicted_readout_recovered=True,
            matched_rescue_controls_pass=True,
        )
    )

    assert result.code == "V1"
    assert "V3_RESCUE_SYSTEM_STATUS_MISSING" in result.reason_codes


def test_e_and_v_axes_remain_independent_in_combined_boundary() -> None:
    result = assign_evidence_boundary(
        e1_inputs(event_q=None),
        ValidationProvenanceInputs(
            intervention_reversal_observed=True,
            intervention_contrast_prespecified=True,
            intervention_reversal_pass=True,
            matched_intervention_controls_pass=True,
        ),
    )

    assert result.event_support.code == "E0"
    assert result.validation_provenance.code == "V2"
    assert result.boundary == "E0-V2"


def test_assignments_are_deterministic_and_json_serializable() -> None:
    first = assign_evidence_boundary(e2_inputs(), ValidationProvenanceInputs())
    second = assign_evidence_boundary(e2_inputs(), ValidationProvenanceInputs())

    assert first == second
    assert json.loads(json.dumps(first.as_dict()))["evidence_boundary"] == "E2-V0"

    invalid = assign_event_support(e1_inputs(event_q=float("nan")))
    json.dumps(invalid.as_dict(), allow_nan=False)
