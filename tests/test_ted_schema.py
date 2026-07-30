from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from click.testing import CliRunner
from jsonschema import Draft202012Validator

from pyfgsea.cli.main import cli
from pyfgsea.ted_schema import ted_table_is_valid, validate_ted_table


ROOT = Path(__file__).resolve().parents[1]


def _schema(name: str, *, packaged: bool = False) -> dict[str, object]:
    base = ROOT / "pyfgsea" / "schemas" if packaged else ROOT / "schemas"
    return json.loads((base / name).read_text(encoding="utf-8"))


def _event_v2(**updates: object) -> pd.DataFrame:
    row: dict[str, object] = {
        "dataset_id": "d",
        "event_id": "e",
        "pathway": "p",
        "direction": "up",
        "event_mode": "activation",
        "event_test_status": "run_supported",
        "event_q": 0.01,
        "event_q_missing_reason": None,
        "e0_reason_code": None,
        "event_support_code": "E2",
        "validation_provenance_code": "V1",
        "evidence_boundary": "E2\u2013V1",
        "supported_interpretation": "Block-robust event with an orthogonal outcome.",
        "unsupported_interpretation_current_evidence": "No intervention reversal or rescue.",
        "identifiability_status": "identifiable",
        "block_support_method": "selection_aware_ci",
        "minimum_attainable_p": None,
        "minimum_attainable_q": None,
        "permutation_resolution_pass": None,
        "resampling_selection_frequency": None,
        "discovery_stability_status": "not_evaluable",
        "seed": 1,
        # Legacy v1 fields are accepted but do not define the v2 boundary.
        "evidence_tier": 3.0,
        "claim_ceiling": "Level 3 block-robust event",
        "matched_functional_rescue": False,
    }
    row.update(updates)
    return pd.DataFrame([row])


def test_activity_table_schema_passes() -> None:
    table = pd.DataFrame(
        {
            "dataset_id": ["d", "d"],
            "block_id": ["b", "b"],
            "time": [0.0, 1.0],
            "pathway": ["p", "p"],
            "activity": [0.0, 1.0],
        }
    )
    assert ted_table_is_valid(validate_ted_table(table, "activity"))


def test_event_schema_fails_closed_on_unmatched_level4() -> None:
    event = pd.DataFrame(
        [
            {
                "dataset_id": "d",
                "event_id": "e",
                "pathway": "p",
                "direction": "up",
                "event_mode": "activation",
                "event_q": 0.01,
                "evidence_tier": 4.0,
                "claim_ceiling": "Level 4 functional support",
                "identifiability_status": "identifiable",
                "matched_functional_rescue": False,
                "seed": 1,
            }
        ]
    )
    report = validate_ted_table(event, "event")
    assert not ted_table_is_valid(report)
    assert "claim_ceiling_gate" in set(report["check"])


def test_event_schema_v1_remains_the_legacy_fallback() -> None:
    event = pd.DataFrame(
        [
            {
                "dataset_id": "d",
                "event_id": "e",
                "pathway": "p",
                "direction": "up",
                "event_mode": "activation",
                "event_q": 0.01,
                "evidence_tier": 3.0,
                "claim_ceiling": "Level 3 block-robust event",
                "identifiability_status": "identifiable",
                "matched_functional_rescue": False,
                "seed": 1,
            }
        ]
    )
    report = validate_ted_table(event, "event")
    assert ted_table_is_valid(report)
    assert "passed v1 schema" in report.iloc[-1]["message"]


def test_event_schema_v2_is_auto_detected_and_accepts_legacy_fields() -> None:
    report = validate_ted_table(_event_v2(), "event")
    assert ted_table_is_valid(report)
    assert "passed v2 schema" in report.iloc[-1]["message"]


def test_event_schema_v2_accepts_ascii_boundary() -> None:
    report = validate_ted_table(_event_v2(evidence_boundary="E2-V1"), "event")
    assert ted_table_is_valid(report)


def test_event_schema_v2_accepts_true_unicode_en_dash() -> None:
    boundary = "E2\u2013V1"
    assert ord(boundary[2]) == 0x2013
    report = validate_ted_table(_event_v2(evidence_boundary=boundary), "event")
    assert ted_table_is_valid(report)


def test_event_schema_v2_accepts_event_only_replication_facets() -> None:
    report = validate_ted_table(
        _event_v2(
            event_replication_eligibility_status="passed",
            event_replication_test_status="run_supported",
            event_replication_status="passed",
            outcome_replication_status="not_tested",
            outcome_type="protein",
            replication_dataset_id="GSE171964",
        ),
        "event",
    )
    assert ted_table_is_valid(report), report.to_dict("records")


def test_event_schema_v2_accepts_parallel_records_without_legacy_v_fields() -> None:
    event = _event_v2(
        parallel_evidence_records=[
            {
                "record_id": "bnt162b2_cd64_cd169",
                "evidence_type": "orthogonal_outcome",
                "status": "passed",
                "independence_context": "same_study_same_cells",
                "outcome_type": "protein",
                "contrast": "early post-dose minus baseline/late",
                "controls_pass": True,
                "replication_status": "not_tested",
                "replication_dataset_id": None,
                "reason_codes": ["PROTEIN_OUTCOME_GATES_PASS"],
            }
        ]
    ).drop(columns=["validation_provenance_code", "evidence_boundary"])
    report = validate_ted_table(event, "event")
    assert ted_table_is_valid(report), report.to_dict("records")


def test_canonical_parallel_evidence_schema_matches_embedded_v2_contract() -> None:
    name = "parallel_evidence_record_v1.schema.json"
    assert (ROOT / "schemas" / name).read_bytes() == (
        ROOT / "pyfgsea" / "schemas" / name
    ).read_bytes()

    canonical = _schema(name)
    Draft202012Validator.check_schema(canonical)
    embedded = _schema("ted_event_report_v2.schema.json")["properties"][
        "parallel_evidence_records"
    ]["items"]
    for key in ("type", "required", "properties", "allOf", "additionalProperties"):
        assert canonical[key] == embedded[key]


def test_canonical_parallel_evidence_schema_enforces_controls_and_replication_id() -> (
    None
):
    validator = Draft202012Validator(_schema("parallel_evidence_record_v1.schema.json"))
    record = {
        "record_id": "protein_outcome",
        "evidence_type": "orthogonal_outcome",
        "status": "passed",
        "independence_context": "same_study_different_assay",
        "outcome_type": "protein",
        "contrast": "early minus baseline",
        "controls_pass": True,
        "replication_status": "not_tested",
        "replication_dataset_id": None,
        "reason_codes": ["OUTCOME_GATES_PASS"],
    }
    assert validator.is_valid(record)

    no_controls = {**record, "controls_pass": False}
    no_replication_id = {
        **record,
        "replication_status": "passed",
        "replication_dataset_id": None,
    }
    assert not validator.is_valid(no_controls)
    assert not validator.is_valid(no_replication_id)


def test_canonical_replication_facets_schema_matches_embedded_v2_contract() -> None:
    name = "replication_facets_v1.schema.json"
    assert (ROOT / "schemas" / name).read_bytes() == (
        ROOT / "pyfgsea" / "schemas" / name
    ).read_bytes()

    canonical = _schema(name)
    Draft202012Validator.check_schema(canonical)
    integrated = _schema("ted_event_report_v2.schema.json")
    facet_fields = (
        "event_replication_eligibility_status",
        "event_replication_test_status",
        "event_replication_status",
        "outcome_replication_status",
        "outcome_type",
        "replication_dataset_id",
    )
    for field in facet_fields:
        assert canonical["properties"][field] == integrated["properties"][field]
    assert canonical["allOf"][:4] == integrated["allOf"][1:5]


def test_canonical_replication_facets_keep_event_and_outcome_results_separate() -> None:
    validator = Draft202012Validator(_schema("replication_facets_v1.schema.json"))
    facets = {
        "event_replication_eligibility_status": "passed",
        "event_replication_test_status": "run_not_supported",
        "event_replication_status": "failed",
        "outcome_replication_status": "passed",
        "outcome_type": "protein",
        "replication_dataset_id": "GSE171964",
        "replication_reason_codes": [
            "EVENT_REPLICATION_GATES_FAIL",
            "OUTCOME_REPLICATION_GATES_PASS",
        ],
    }
    assert validator.is_valid(facets)

    eligibility_misreported_as_test_failure = {
        **facets,
        "event_replication_eligibility_status": "failed",
    }
    assert not validator.is_valid(eligibility_misreported_as_test_failure)


def test_event_schema_v2_rejects_passed_record_without_controls() -> None:
    event = _event_v2(
        parallel_evidence_records=[
            {
                "record_id": "outcome",
                "evidence_type": "orthogonal_outcome",
                "status": "passed",
                "independence_context": "same_study_same_cells",
                "controls_pass": False,
                "replication_status": "not_tested",
                "reason_codes": [],
            }
        ]
    )
    report = validate_ted_table(event, "event")
    assert not ted_table_is_valid(report)
    assert "json_schema" in set(report["check"])


def test_event_schema_v2_requires_complete_replication_facets() -> None:
    report = validate_ted_table(
        _event_v2(
            event_replication_eligibility_status="passed",
            event_replication_test_status="run_supported",
            event_replication_status="passed",
            outcome_type="protein",
            replication_dataset_id="GSE171964",
        ),
        "event",
    )
    assert not ted_table_is_valid(report)


def test_event_schema_v2_keeps_outcome_replication_independent_of_event_pass() -> None:
    report = validate_ted_table(
        _event_v2(
            event_replication_eligibility_status="passed",
            event_replication_test_status="run_not_supported",
            event_replication_status="failed",
            outcome_replication_status="passed",
            outcome_type="protein",
            replication_dataset_id="GSE171964",
        ),
        "event",
    )
    assert ted_table_is_valid(report), report.to_dict("records")


def test_event_schema_v2_replication_eligibility_failure_is_not_evaluable() -> None:
    report = validate_ted_table(
        _event_v2(
            event_replication_eligibility_status="failed",
            event_replication_test_status="not_run",
            event_replication_status="not_evaluable",
            outcome_replication_status="not_tested",
            outcome_type="protein",
            replication_dataset_id="GSE171964",
        ),
        "event",
    )
    assert ted_table_is_valid(report), report.to_dict("records")


def test_event_schema_v2_rejects_eligibility_failure_as_tested_failure() -> None:
    report = validate_ted_table(
        _event_v2(
            event_replication_eligibility_status="failed",
            event_replication_test_status="run_not_supported",
            event_replication_status="failed",
            outcome_replication_status="not_tested",
            outcome_type="protein",
            replication_dataset_id="GSE171964",
        ),
        "event",
    )
    assert not ted_table_is_valid(report)


def test_event_schema_v2_rejects_inconsistent_boundary() -> None:
    report = validate_ted_table(_event_v2(evidence_boundary="E1-V1"), "event")
    assert not ted_table_is_valid(report)
    assert "evidence_boundary_consistency" in set(report["check"])


def test_event_schema_v2_requires_matched_rescue_for_v3() -> None:
    report = validate_ted_table(
        _event_v2(validation_provenance_code="V3", evidence_boundary="E2-V3"),
        "event",
    )
    assert not ted_table_is_valid(report)
    assert "validation_provenance_gate" in set(report["check"])


def test_event_schema_v2_e0_requires_stable_reason_code() -> None:
    event = _event_v2(
        event_q=0.20,
        event_test_status="run_not_supported",
        e0_reason_code=None,
        event_support_code="E0",
        evidence_boundary="E0-V1",
        identifiability_status="limited",
    )
    report = validate_ted_table(event, "event")
    assert not ted_table_is_valid(report)
    assert "e0_reason_code" in set(report["check"])


def test_event_schema_v2_allows_null_q_when_test_not_run() -> None:
    event = _event_v2(
        event_test_status="not_run",
        event_q=None,
        event_q_missing_reason="insufficient_blocks",
        e0_reason_code="E0_not_estimable",
        event_support_code="E0",
        evidence_boundary="E0-V2",
        validation_provenance_code="V2",
        identifiability_status="limited",
    )
    report = validate_ted_table(event, "event")
    assert ted_table_is_valid(report), report.to_dict("records")


def test_event_schema_v2_rejects_null_q_after_test_run() -> None:
    event = _event_v2(
        event_test_status="run_not_supported",
        event_q=None,
        e0_reason_code="E0_not_supported",
        event_support_code="E0",
        validation_provenance_code="V0",
        evidence_boundary="E0-V0",
        identifiability_status="limited",
    )
    report = validate_ted_table(event, "event")
    assert not ted_table_is_valid(report)
    assert "event_q_range" in set(report["check"])


def test_event_schema_v2_allows_numeric_q_for_other_e0_reason() -> None:
    event = _event_v2(
        event_test_status="run_not_supported",
        event_q=0.25,
        e0_reason_code="E0_not_supported",
        event_support_code="E0",
        validation_provenance_code="V0",
        evidence_boundary="E0-V0",
        identifiability_status="limited",
    )
    report = validate_ted_table(event, "event")
    assert ted_table_is_valid(report), report.to_dict("records")


def test_event_schema_v2_rejects_null_q_for_supported_event() -> None:
    report = validate_ted_table(_event_v2(event_q=None), "event")
    assert not ted_table_is_valid(report)
    assert "event_q_range" in set(report["check"])


def test_event_schema_v2_requires_missing_reason_when_test_not_run() -> None:
    event = _event_v2(
        event_test_status="not_run",
        event_q=None,
        event_q_missing_reason=None,
        e0_reason_code="E0_missing_required_design",
        event_support_code="E0",
        validation_provenance_code="V0",
        evidence_boundary="E0-V0",
        identifiability_status="not_identifiable",
    )
    report = validate_ted_table(event, "event")
    assert not ted_table_is_valid(report)
    assert "event_q_missing_reason" in set(report["check"])


def test_event_schema_v2_rejects_missing_reason_after_test_run() -> None:
    report = validate_ted_table(_event_v2(event_q_missing_reason="other"), "event")
    assert not ted_table_is_valid(report)
    assert "event_q_missing_reason" in set(report["check"])


def test_event_schema_v2_rejects_e0_reason_on_supported_event() -> None:
    report = validate_ted_table(_event_v2(e0_reason_code="E0_not_supported"), "event")
    assert not ted_table_is_valid(report)
    assert "e0_reason_code" in set(report["check"])


def test_event_schema_v2_accepts_consistent_resampling_stability() -> None:
    event = _event_v2(
        resampling_selection_frequency=0.83,
        discovery_stability_status="stable_core",
        upstream_disagreement_flag=False,
        upstream_method_agreement=0.91,
    )
    report = validate_ted_table(event, "event")
    assert ted_table_is_valid(report), report.to_dict("records")


def test_event_schema_v2_rejects_inconsistent_resampling_stability() -> None:
    report = validate_ted_table(
        _event_v2(
            resampling_selection_frequency=0.49,
            discovery_stability_status="stable_core",
        ),
        "event",
    )
    assert not ted_table_is_valid(report)
    assert "discovery_stability_consistency" in set(report["check"])


def test_event_schema_v2_requires_not_evaluable_when_resampling_is_absent() -> None:
    report = validate_ted_table(
        _event_v2(
            resampling_selection_frequency=None,
            discovery_stability_status="stable_core",
        ),
        "event",
    )
    assert not ted_table_is_valid(report)
    assert "discovery_stability_consistency" in set(report["check"])


def test_event_schema_v2_failed_permutation_resolution_is_not_run_e0() -> None:
    event = _event_v2(
        block_support_method="exact_paired_sign_permutation",
        minimum_attainable_p=0.125,
        permutation_resolution_pass=False,
        event_test_status="not_run",
        event_q=None,
        event_q_missing_reason="insufficient_permutation_resolution",
        e0_reason_code="E0_not_estimable",
        event_support_code="E0",
        evidence_boundary="E0-V1",
        identifiability_status="limited",
    )
    report = validate_ted_table(event, "event")
    assert ted_table_is_valid(report), report.to_dict("records")


def test_event_schema_v2_upstream_disagreement_forbids_e2() -> None:
    report = validate_ted_table(
        _event_v2(upstream_disagreement_flag=True, upstream_method_agreement=0.50),
        "event",
    )
    assert not ted_table_is_valid(report)
    assert "upstream_disagreement_gate" in set(report["check"])


def test_cli_validate_writes_report(tmp_path) -> None:
    path = tmp_path / "activity.tsv"
    report = tmp_path / "report.tsv"
    pd.DataFrame(
        {
            "dataset_id": ["d", "d"],
            "block_id": ["b", "b"],
            "time": [0.0, 1.0],
            "pathway": ["p", "p"],
            "activity": [0.0, 1.0],
        }
    ).to_csv(path, sep="\t", index=False)
    result = CliRunner().invoke(
        cli, ["validate", str(path), "--kind", "activity", "--report", str(report)]
    )
    assert result.exit_code == 0, result.output
    assert report.exists()


def test_cli_validate_recognizes_event_v2(tmp_path) -> None:
    path = tmp_path / "events_v2.tsv"
    report = tmp_path / "report.tsv"
    _event_v2().to_csv(path, sep="\t", index=False)
    result = CliRunner().invoke(
        cli,
        [
            "validate",
            str(path),
            "--kind",
            "event",
            "--schema-version",
            "v2",
            "--report",
            str(report),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "passed v2 schema" in result.output
    assert report.exists()
