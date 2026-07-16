from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping
from urllib.parse import urlparse

import pandas as pd
import yaml


SCHEMA_NAME = "trajpathmix_metadata_first_dataset_screen_contract"
SCHEMA_VERSION = "1.0.0"
SCREEN_ID = "trajpathmix_dataset_screen_v1"
FROZEN_CONFIG_PAYLOAD_SHA256 = (
    "6718e4fb9533f32a77413a3337c133676a91f8f196ec548c941fe4eeb4a2dca6"
)
EXPECTED_CANDIDATE_IDS = (
    "hipsci_endoderm_125_v1",
    "scbloodnl_pathogen_120_v1",
)
EXPECTED_SOURCE_URLS = (
    "https://www.nature.com/articles/s41467-020-14457-z",
    "https://zenodo.org/records/3625024",
    "https://ega-archive.org/datasets/EGAD00001005741",
    "https://www.ebi.ac.uk/ena/browser/view/ERP016000",
    "https://www.nature.com/articles/s41467-022-30893-5",
    "https://ega-archive.org/datasets/EGAD00001007764",
)

ACCESS_MANIFEST_FILE = "candidate_access_manifest_v1.tsv"
METADATA_INVENTORY_FILE = "candidate_metadata_inventory_v1.tsv"
DESIGN_PREFLIGHT_FILE = "candidate_design_preflight_v1.tsv"
SELECTION_DECISION_FILE = "candidate_selection_decision_v1.json"
BUILD_RECORD_FILE = "trajpathmix_dataset_screen_build_record_v1.json"

ACCESS_COLUMNS = (
    "candidate_id",
    "metadata_audit_rank",
    "source_id",
    "source_kind",
    "source_authority",
    "source_url",
    "accession",
    "access_class",
    "requires_data_access_approval",
    "supports_fields",
    "verified_at_freeze",
    "source_verification_date",
    "source_verification_status",
    "file_inventory_status",
    "reported_file_count",
    "reported_total_size",
    "local_file_path",
    "downloaded_bytes",
    "downloaded_sha256",
    "expression_matrix_downloaded",
    "large_file_downloaded",
    "full_matrix_acquisition_authorized",
)

METADATA_COLUMNS = (
    "candidate_id",
    "metadata_audit_rank",
    "display_name",
    "intended_role",
    "study_accessions",
    "n_individuals_reported",
    "donors_per_timepoint_reported",
    "n_timepoints_reported",
    "timepoints_reported",
    "n_pooled_experiments_reported",
    "n_condition_combinations_reported",
    "n_cells_profiled_reported",
    "n_cells_qc_retained_reported",
    "perturbations_reported",
    "sequencing_protocols_reported",
    "external_time_reference_status",
    "auxiliary_validation_status",
    "donor_time_completeness_status",
    "donor_condition_completeness_status",
    "group_definition_status",
    "donor_bin_support_status",
    "condition_study_confounding_status",
    "condition_batch_confounding_status",
    "condition_gate_confounding_status",
    "full_matrix_access_status",
    "outcome_blind_mde_status",
    "precision_weight_status",
    "independent_validation_status",
    "metadata_only",
    "expression_matrix_loaded",
    "pathway_outcomes_read",
)

PREFLIGHT_COLUMNS = (
    "candidate_id",
    "metadata_audit_rank",
    "reported_individual_scale_screen_pass",
    "primary_official_source_gate_pass",
    "external_time_reference_gate_pass",
    "donor_time_completeness_gate_pass",
    "donor_condition_completeness_gate_pass",
    "group_definition_gate_pass",
    "donor_bin_support_gate_pass",
    "study_condition_confounding_gate_pass",
    "batch_condition_confounding_gate_pass",
    "gate_condition_confounding_gate_pass",
    "full_matrix_access_gate_pass",
    "outcome_blind_mde_gate_pass",
    "precision_weight_gate_pass",
    "independent_validation_gate_pass",
    "method_benchmark_preflight_pass",
    "main_biological_discovery_preflight_pass",
    "preferred_metadata_audit_target",
    "method_benchmark_suitability",
    "main_biological_discovery_suitability",
    "full_matrix_acquisition_allowed",
    "pathway_scoring_allowed",
    "allowed_next_action",
    "reason_codes",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _config_payload_sha256(config: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in config.items()
        if key != "frozen_payload_sha256" and not str(key).startswith("_")
    }
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def _require_exact(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise ValueError(f"Frozen dataset-screen contract mismatch for {label}")


def validate_dataset_screen_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen metadata-only screen and its primary-source firewall."""

    _require_exact(config.get("schema_name"), SCHEMA_NAME, "schema_name")
    _require_exact(config.get("schema_version"), SCHEMA_VERSION, "schema_version")
    _require_exact(config.get("screen_id"), SCREEN_ID, "screen_id")
    _require_exact(
        config.get("frozen_payload_sha256"),
        FROZEN_CONFIG_PAYLOAD_SHA256,
        "frozen_payload_sha256",
    )
    _require_exact(
        _config_payload_sha256(config),
        FROZEN_CONFIG_PAYLOAD_SHA256,
        "recomputed_payload_sha256",
    )

    scope = config.get("scope", {})
    for key, expected in {
        "metadata_only": True,
        "expression_matrices_downloaded": False,
        "large_files_downloaded": False,
        "pathway_outcomes_read": False,
        "selection_uses_pathway_outcomes": False,
        "network_requests_during_build": False,
    }.items():
        _require_exact(scope.get(key), expected, f"scope.{key}")

    source_policy = config.get("source_policy", {})
    _require_exact(
        source_policy.get("expression_download_during_screen_forbidden"),
        True,
        "source_policy.expression_download_during_screen_forbidden",
    )
    allowed_hosts = set(source_policy.get("allowed_https_hosts", []))
    selection = config.get("selection_policy", {})
    _require_exact(
        selection.get("preferred_metadata_audit_candidate_id"),
        EXPECTED_CANDIDATE_IDS[0],
        "preferred_metadata_audit_candidate_id",
    )
    for key in (
        "preference_uses_only_reported_design_metadata",
        "method_benchmark_approval_requires_all_design_gates",
        "main_discovery_approval_requires_all_design_and_validation_gates",
        "full_matrix_acquisition_requires_preflight_pass",
        "pathway_scoring_requires_preflight_pass",
    ):
        _require_exact(selection.get(key), True, f"selection_policy.{key}")
    _require_exact(
        selection.get("unresolved_gate_action"),
        "fail_closed",
        "selection_policy.unresolved_gate_action",
    )

    candidates = config.get("candidates", [])
    _require_exact(
        tuple(candidate.get("candidate_id") for candidate in candidates),
        EXPECTED_CANDIDATE_IDS,
        "candidate_ids",
    )
    _require_exact(
        tuple(candidate.get("metadata_audit_rank") for candidate in candidates),
        (1, 2),
        "metadata_audit_rank",
    )
    urls: list[str] = []
    source_ids: set[str] = set()
    for candidate in candidates:
        for source in candidate.get("sources", []):
            source_id = str(source.get("source_id", ""))
            if not source_id or source_id in source_ids:
                raise ValueError("Source IDs must be unique and non-empty")
            source_ids.add(source_id)
            url = str(source.get("source_url", ""))
            parsed = urlparse(url)
            if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
                raise ValueError("Source URL is not an allowed primary/official host")
            _require_exact(source.get("verified_at_freeze"), True, source_id)
            urls.append(url)
    _require_exact(tuple(urls), EXPECTED_SOURCE_URLS, "source_urls")

    result = deepcopy(dict(config))
    result["_config_payload_sha256"] = FROZEN_CONFIG_PAYLOAD_SHA256
    return result


def load_dataset_screen_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Dataset-screen config must be a YAML mapping")
    return validate_dataset_screen_config(value)


def _build_access_manifest(config: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    verification_date = config["source_policy"]["source_verification_date"]
    for candidate in config["candidates"]:
        for source in candidate["sources"]:
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "metadata_audit_rank": candidate["metadata_audit_rank"],
                    "source_id": source["source_id"],
                    "source_kind": source["source_kind"],
                    "source_authority": source["source_authority"],
                    "source_url": source["source_url"],
                    "accession": source["accession"],
                    "access_class": source["access_class"],
                    "requires_data_access_approval": bool(
                        source["requires_data_access_approval"]
                    ),
                    "supports_fields": source["supports_fields"],
                    "verified_at_freeze": bool(source["verified_at_freeze"]),
                    "source_verification_date": verification_date,
                    "source_verification_status": (
                        "verified_primary_or_official_url_at_freeze"
                    ),
                    "file_inventory_status": (
                        "not_audited_in_metadata_only_freeze"
                    ),
                    "reported_file_count": "not_frozen_in_metadata_screen",
                    "reported_total_size": "not_frozen_in_metadata_screen",
                    "local_file_path": "not_downloaded",
                    "downloaded_bytes": 0,
                    "downloaded_sha256": "not_downloaded",
                    "expression_matrix_downloaded": False,
                    "large_file_downloaded": False,
                    "full_matrix_acquisition_authorized": False,
                }
            )
    return pd.DataFrame(rows, columns=list(ACCESS_COLUMNS))


def _build_metadata_inventory(config: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in config["candidates"]:
        reported = candidate["reported_design"]
        unresolved = candidate["unresolved_design"]
        row = {
            "candidate_id": candidate["candidate_id"],
            "metadata_audit_rank": candidate["metadata_audit_rank"],
            "display_name": candidate["display_name"],
            "intended_role": candidate["intended_role"],
            **reported,
            **unresolved,
            "metadata_only": True,
            "expression_matrix_loaded": False,
            "pathway_outcomes_read": False,
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=list(METADATA_COLUMNS))


def _is_resolved_pass(value: Any) -> bool:
    return str(value) in {
        "resolved_exact_pass",
        "resolved_no_confounding_pass",
        "resolved_identifiable_pass",
        "authorized_and_schema_audited_pass",
        "evaluated_pass",
        "independent_validation_pass",
    }


def _candidate_preflight(
    candidate: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    reported = candidate["reported_design"]
    unresolved = candidate["unresolved_design"]
    sources = candidate["sources"]
    source_pass = bool(sources) and all(
        source["verified_at_freeze"]
        and source["source_kind"]
        in config["source_policy"]["allowed_source_kinds"]
        for source in sources
    )
    scale_pass = int(reported["n_individuals_reported"]) >= int(
        config["entry_policy"]["preferred_minimum_independent_donors_per_group"]
    )
    external_time_pass = reported["external_time_reference_status"] == (
        "reported_pass"
    )
    gate_values = {
        "donor_time_completeness_gate_pass": _is_resolved_pass(
            unresolved["donor_time_completeness_status"]
        ),
        "donor_condition_completeness_gate_pass": _is_resolved_pass(
            unresolved["donor_condition_completeness_status"]
        ),
        "group_definition_gate_pass": _is_resolved_pass(
            unresolved["group_definition_status"]
        ),
        "donor_bin_support_gate_pass": _is_resolved_pass(
            unresolved["donor_bin_support_status"]
        ),
        "study_condition_confounding_gate_pass": _is_resolved_pass(
            unresolved["condition_study_confounding_status"]
        ),
        "batch_condition_confounding_gate_pass": _is_resolved_pass(
            unresolved["condition_batch_confounding_status"]
        ),
        "gate_condition_confounding_gate_pass": _is_resolved_pass(
            unresolved["condition_gate_confounding_status"]
        ),
        "full_matrix_access_gate_pass": _is_resolved_pass(
            unresolved["full_matrix_access_status"]
        ),
        "outcome_blind_mde_gate_pass": _is_resolved_pass(
            unresolved["outcome_blind_mde_status"]
        ),
        "precision_weight_gate_pass": _is_resolved_pass(
            unresolved["precision_weight_status"]
        ),
        "independent_validation_gate_pass": _is_resolved_pass(
            unresolved["independent_validation_status"]
        ),
    }
    method_gate_names = (
        "donor_time_completeness_gate_pass",
        "donor_condition_completeness_gate_pass",
        "group_definition_gate_pass",
        "donor_bin_support_gate_pass",
        "study_condition_confounding_gate_pass",
        "batch_condition_confounding_gate_pass",
        "gate_condition_confounding_gate_pass",
        "full_matrix_access_gate_pass",
        "outcome_blind_mde_gate_pass",
        "precision_weight_gate_pass",
    )
    method_pass = bool(
        scale_pass
        and source_pass
        and external_time_pass
        and all(gate_values[name] for name in method_gate_names)
    )
    main_pass = bool(
        method_pass and gate_values["independent_validation_gate_pass"]
    )

    reason_map = (
        (
            "donor_time_completeness_gate_pass",
            "DONOR_TIME_COMPLETENESS_UNRESOLVED",
        ),
        (
            "donor_condition_completeness_gate_pass",
            "DONOR_CONDITION_COMPLETENESS_UNRESOLVED",
        ),
        ("group_definition_gate_pass", "PRIMARY_GROUP_DEFINITION_UNRESOLVED"),
        ("donor_bin_support_gate_pass", "DONOR_BIN_SUPPORT_NOT_EVALUATED"),
        (
            "study_condition_confounding_gate_pass",
            "STUDY_CONDITION_CONFOUNDING_UNRESOLVED",
        ),
        (
            "batch_condition_confounding_gate_pass",
            "BATCH_CONDITION_CONFOUNDING_UNRESOLVED",
        ),
        (
            "gate_condition_confounding_gate_pass",
            "GATE_CONDITION_CONFOUNDING_UNRESOLVED",
        ),
        ("full_matrix_access_gate_pass", "FULL_MATRIX_ACCESS_NOT_AUTHORIZED"),
        ("outcome_blind_mde_gate_pass", "OUTCOME_BLIND_MDE_NOT_EVALUATED"),
        ("precision_weight_gate_pass", "PRECISION_WEIGHT_STABILITY_NOT_EVALUATED"),
        (
            "independent_validation_gate_pass",
            "INDEPENDENT_VALIDATION_NOT_RESOLVED",
        ),
    )
    reasons = [code for gate, code in reason_map if not gate_values[gate]]
    if not scale_pass:
        reasons.insert(0, "REPORTED_INDIVIDUAL_SCALE_BELOW_SCREEN")
    if not source_pass:
        reasons.insert(0, "PRIMARY_OFFICIAL_SOURCE_GATE_FAILED")
    if not external_time_pass:
        reasons.insert(0, "EXTERNAL_TIME_REFERENCE_NOT_REPORTED")
    if candidate["candidate_id"] == "scbloodnl_pathogen_120_v1":
        reasons.append("THREE_LEVEL_TIME_ESTIMAND_REQUIRES_PREFREEZE_REVIEW")

    return {
        "candidate_id": candidate["candidate_id"],
        "metadata_audit_rank": candidate["metadata_audit_rank"],
        "reported_individual_scale_screen_pass": scale_pass,
        "primary_official_source_gate_pass": source_pass,
        "external_time_reference_gate_pass": external_time_pass,
        **gate_values,
        "method_benchmark_preflight_pass": method_pass,
        "main_biological_discovery_preflight_pass": main_pass,
        "preferred_metadata_audit_target": bool(
            candidate["suitability"]["preferred_metadata_audit_target"]
        ),
        "method_benchmark_suitability": candidate["suitability"][
            "method_benchmark_suitability"
        ],
        "main_biological_discovery_suitability": candidate["suitability"][
            "main_biological_discovery_suitability"
        ],
        "full_matrix_acquisition_allowed": False,
        "pathway_scoring_allowed": False,
        "allowed_next_action": candidate["suitability"]["allowed_next_action"],
        "reason_codes": ";".join(reasons),
    }


def build_dataset_screen_artifacts(
    config: Mapping[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Build deterministic metadata-only screen tables and a fail-closed decision."""

    validated = validate_dataset_screen_config(config)
    access = _build_access_manifest(validated)
    metadata = _build_metadata_inventory(validated)
    preflight = pd.DataFrame(
        [
            _candidate_preflight(candidate, validated)
            for candidate in validated["candidates"]
        ],
        columns=list(PREFLIGHT_COLUMNS),
    )
    method_approved = preflight.loc[
        preflight["method_benchmark_preflight_pass"], "candidate_id"
    ].tolist()
    main_approved = preflight.loc[
        preflight["main_biological_discovery_preflight_pass"], "candidate_id"
    ].tolist()
    preferred = validated["selection_policy"][
        "preferred_metadata_audit_candidate_id"
    ]
    decision = {
        "schema_name": "trajpathmix_dataset_screen_selection_decision",
        "schema_version": "1.0.0",
        "screen_id": SCREEN_ID,
        "config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
        "created_at_utc": validated["frozen_at_utc"],
        "metadata_only": True,
        "expression_matrices_downloaded": False,
        "large_files_downloaded": False,
        "pathway_outcomes_read": False,
        "selection_uses_pathway_outcomes": False,
        "preferred_metadata_audit_candidate_id": preferred,
        "preferred_metadata_audit_target_only": True,
        "preferred_target_full_matrix_acquisition_allowed": False,
        "approved_method_benchmark_candidate_ids": method_approved,
        "approved_main_biological_discovery_candidate_ids": main_approved,
        "full_matrix_acquisition_allowed": False,
        "pathway_scoring_allowed": False,
        "selection_status": "metadata_audit_priority_only_fail_closed",
        "candidate_decisions": [
            {
                "candidate_id": str(row["candidate_id"]),
                "metadata_audit_rank": int(row["metadata_audit_rank"]),
                "preferred_metadata_audit_target": bool(
                    row["preferred_metadata_audit_target"]
                ),
                "method_benchmark_preflight_pass": bool(
                    row["method_benchmark_preflight_pass"]
                ),
                "main_biological_discovery_preflight_pass": bool(
                    row["main_biological_discovery_preflight_pass"]
                ),
                "full_matrix_acquisition_allowed": False,
                "allowed_next_action": str(row["allowed_next_action"]),
                "reason_codes": str(row["reason_codes"]).split(";"),
            }
            for row in preflight.to_dict("records")
        ],
        "global_reason_codes": [
            "NO_CANDIDATE_PASSED_METHOD_BENCHMARK_PREFLIGHT",
            "NO_CANDIDATE_PASSED_MAIN_DISCOVERY_PREFLIGHT",
            "FULL_MATRIX_ACQUISITION_NOT_AUTHORIZED",
            "PATHWAY_SCORING_NOT_AUTHORIZED",
        ],
        "authorized_next_actions": [
            "inspect_small_public_metadata_and_processed_data_schemas",
            "construct_exact_donor_condition_time_experiment_tables",
            "run_outcome_blind_support_confounding_and_mde_preflight",
        ],
        "forbidden_actions": [
            "download_full_expression_matrix_before_preflight_pass",
            "score_pathways_before_candidate_selection_freeze",
            "select_candidate_using_pathway_outcomes",
            "call_metadata_audit_priority_a_method_benchmark_approval",
            "call_method_benchmark_suitability_main_biological_discovery_suitability",
        ],
    }
    if method_approved or main_approved:
        raise ValueError("Frozen metadata screen unexpectedly approved a candidate")
    tables = {
        ACCESS_MANIFEST_FILE: access,
        METADATA_INVENTORY_FILE: metadata,
        DESIGN_PREFLIGHT_FILE: preflight,
    }
    return tables, decision


def _write_table(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, sep="\t", index=False, lineterminator="\n")


def write_dataset_screen_artifacts(
    config: Mapping[str, Any],
    *,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create one complete screen directory atomically and never overwrite it."""

    validated = validate_dataset_screen_config(config)
    tables, decision = build_dataset_screen_artifacts(validated)
    config_file = Path(config_path).resolve()
    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Dataset-screen output already exists: {output}")

    lock_path = output.parent / f".{output.name}.create.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Dataset-screen output creation is locked: {lock_path}"
        ) from exc

    temporary_output: Path | None = None
    try:
        os.close(lock_fd)
        if output.exists():
            raise FileExistsError(f"Dataset-screen output already exists: {output}")
        temporary_output = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
        )
        for filename, table in tables.items():
            _write_table(table, temporary_output / filename)
        (temporary_output / SELECTION_DECISION_FILE).write_text(
            json.dumps(decision, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

        artifacts: dict[str, Any] = {}
        for filename, table in tables.items():
            artifacts[filename] = {
                "sha256": _sha256_file(temporary_output / filename),
                "rows": int(len(table)),
                "columns": list(table.columns),
            }
        artifacts[SELECTION_DECISION_FILE] = {
            "sha256": _sha256_file(temporary_output / SELECTION_DECISION_FILE),
            "schema_name": decision["schema_name"],
        }
        build_record = {
            "schema_name": "trajpathmix_dataset_screen_build_record",
            "schema_version": "1.0.0",
            "screen_id": SCREEN_ID,
            "created_at_utc": validated["frozen_at_utc"],
            "config_file": "config/trajpathmix_dataset_screen_v1.yaml",
            "config_file_sha256": _sha256_file(config_file),
            "config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
            "implementation_file": "pyfgsea/trajpathmix_dataset_screen.py",
            "implementation_sha256": _sha256_file(Path(__file__).resolve()),
            "artifacts": artifacts,
            "primary_official_source_urls": list(EXPECTED_SOURCE_URLS),
            "metadata_only": True,
            "network_requests_during_build": False,
            "expression_matrices_downloaded": False,
            "large_files_downloaded": False,
            "pathway_outcomes_read": False,
            "selection_uses_pathway_outcomes": False,
            "full_matrix_acquisition_allowed": False,
            "pathway_scoring_allowed": False,
        }
        (temporary_output / BUILD_RECORD_FILE).write_text(
            json.dumps(build_record, indent=2, sort_keys=True, ensure_ascii=True)
            + "\n",
            encoding="utf-8",
        )
        if output.exists():
            raise FileExistsError(f"Dataset-screen output already exists: {output}")
        os.rename(temporary_output, output)
        temporary_output = None
        result = dict(build_record)
        result["output_dir"] = str(output.resolve())
        result["build_record_sha256"] = _sha256_file(output / BUILD_RECORD_FILE)
        return result
    finally:
        if temporary_output is not None and temporary_output.exists():
            shutil.rmtree(temporary_output)
        lock_path.unlink(missing_ok=True)


def build_and_write_dataset_screen(
    *,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config = load_dataset_screen_config(config_path)
    return write_dataset_screen_artifacts(
        config,
        config_path=config_path,
        output_dir=output_dir,
    )


def _normalized_table(table: pd.DataFrame) -> pd.DataFrame:
    return table.fillna("").astype(str).reset_index(drop=True)


def validate_dataset_screen_output(
    *,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Rebuild expected metadata in memory and verify every output and hash."""

    config_file = Path(config_path).resolve()
    config = load_dataset_screen_config(config_file)
    output = Path(output_dir).resolve()
    expected_names = {
        ACCESS_MANIFEST_FILE,
        METADATA_INVENTORY_FILE,
        DESIGN_PREFLIGHT_FILE,
        SELECTION_DECISION_FILE,
        BUILD_RECORD_FILE,
    }
    if not output.is_dir():
        raise FileNotFoundError(f"Dataset-screen output is missing: {output}")
    observed_names = {path.name for path in output.iterdir() if path.is_file()}
    _require_exact(observed_names, expected_names, "output_file_set")

    record = json.loads((output / BUILD_RECORD_FILE).read_text(encoding="utf-8"))
    _require_exact(
        record.get("schema_name"),
        "trajpathmix_dataset_screen_build_record",
        "build_record.schema_name",
    )
    _require_exact(
        record.get("config_payload_sha256"),
        FROZEN_CONFIG_PAYLOAD_SHA256,
        "build_record.config_payload_sha256",
    )
    _require_exact(
        record.get("config_file_sha256"),
        _sha256_file(config_file),
        "build_record.config_file_sha256",
    )
    for key, expected in {
        "metadata_only": True,
        "network_requests_during_build": False,
        "expression_matrices_downloaded": False,
        "large_files_downloaded": False,
        "pathway_outcomes_read": False,
        "selection_uses_pathway_outcomes": False,
        "full_matrix_acquisition_allowed": False,
        "pathway_scoring_allowed": False,
    }.items():
        _require_exact(record.get(key), expected, f"build_record.{key}")
    _require_exact(
        tuple(record.get("primary_official_source_urls", [])),
        EXPECTED_SOURCE_URLS,
        "build_record.primary_official_source_urls",
    )

    expected_tables, expected_decision = build_dataset_screen_artifacts(config)
    for filename, expected_table in expected_tables.items():
        artifact = record.get("artifacts", {}).get(filename, {})
        path = output / filename
        _require_exact(artifact.get("sha256"), _sha256_file(path), filename)
        observed = pd.read_csv(path, sep="\t", keep_default_na=False)
        _require_exact(tuple(observed.columns), tuple(expected_table.columns), filename)
        pd.testing.assert_frame_equal(
            _normalized_table(observed),
            _normalized_table(expected_table),
            check_dtype=False,
        )

    decision_path = output / SELECTION_DECISION_FILE
    _require_exact(
        record.get("artifacts", {})
        .get(SELECTION_DECISION_FILE, {})
        .get("sha256"),
        _sha256_file(decision_path),
        SELECTION_DECISION_FILE,
    )
    observed_decision = json.loads(decision_path.read_text(encoding="utf-8"))
    _require_exact(observed_decision, expected_decision, "selection_decision")

    result = dict(record)
    result["output_dir"] = str(output)
    result["build_record_sha256"] = _sha256_file(output / BUILD_RECORD_FILE)
    result["validation_status"] = "pass_fail_closed_metadata_only"
    return result


__all__ = [
    "ACCESS_COLUMNS",
    "ACCESS_MANIFEST_FILE",
    "BUILD_RECORD_FILE",
    "DESIGN_PREFLIGHT_FILE",
    "EXPECTED_CANDIDATE_IDS",
    "EXPECTED_SOURCE_URLS",
    "FROZEN_CONFIG_PAYLOAD_SHA256",
    "METADATA_COLUMNS",
    "METADATA_INVENTORY_FILE",
    "PREFLIGHT_COLUMNS",
    "SCREEN_ID",
    "SELECTION_DECISION_FILE",
    "build_and_write_dataset_screen",
    "build_dataset_screen_artifacts",
    "load_dataset_screen_config",
    "validate_dataset_screen_config",
    "validate_dataset_screen_output",
    "write_dataset_screen_artifacts",
]
