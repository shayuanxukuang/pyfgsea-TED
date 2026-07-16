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


SCHEMA_NAME = "trajpathmix_dataset_portfolio_contract"
SCHEMA_VERSION = "2.0.0"
PORTFOLIO_ID = "trajpathmix_dataset_portfolio_v2"
FROZEN_CONFIG_PAYLOAD_SHA256 = (
    "0e45b67168685613ae1fa4642a581995856fe1db0c6ce122020468c727ab852e"
)

EXPECTED_STAGE_IDS = ("A", "B", "C", "D", "E")
EXPECTED_CANDIDATE_IDS = (
    "hipsci_endoderm_125_v2",
    "hipsci_dopaminergic_215_v2",
    "sound_life_flu_96_v2",
    "scbloodnl_pathogen_120_v2",
    "sarscov2_human_challenge_16_v2",
)
EXPECTED_ROLES = (
    "method_benchmark",
    "advanced_branching_genetic_benchmark",
    "primary_biological_discovery_candidate",
    "paired_repeated_measures_extension_driver",
    "small_sample_dense_time_stress_test",
)

ROLE_MATRIX_FILE = "dataset_role_matrix_v2.tsv"
SOURCE_MANIFEST_FILE = "dataset_source_manifest_v2.tsv"
AUTHORIZATION_DECISION_FILE = "dataset_authorization_decision_v2.json"
ENDODERM_CONTRACT_FILE = "endoderm_benchmark_contract_v1.json"
SOUND_LIFE_PREREG_FILE = "sound_life_discovery_preregistration_v1.json"
PAIRED_DEFERRAL_FILE = "paired_extension_deferral_v1.json"
BUILD_RECORD_FILE = "trajpathmix_dataset_portfolio_build_record_v2.json"

ROLE_COLUMNS = (
    "stage_id",
    "priority_order",
    "candidate_id",
    "display_name",
    "role",
    "independent_unit",
    "n_independent_units_reported",
    "timepoints_reported",
    "n_cells_reported",
    "statistical_core_required",
    "acquisition_decision",
    "full_expression_matrix_authorized_now",
    "metadata_acquisition_authorized_now",
    "pathway_scoring_authorized",
    "biological_discovery_claims_status",
    "fine_calendar_onset_claims_allowed",
    "pathway_outcomes_read",
    "selection_uses_pathway_outcomes",
    "next_gate",
)

SOURCE_COLUMNS = (
    "stage_id",
    "candidate_id",
    "source_id",
    "source_kind",
    "source_authority",
    "source_url",
    "accession",
    "access_class",
    "supports_fields",
    "verified_at_freeze",
    "file_id",
    "file_name",
    "file_url",
    "size_bytes",
    "publisher_checksum_algorithm",
    "publisher_checksum",
    "source_audit_sha256",
    "content_contract",
    "download_authorized_now",
    "pathway_outcomes_read",
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
        raise ValueError(
            f"Frozen dataset-portfolio contract mismatch for {label}: "
            f"expected {expected!r}, observed {value!r}"
        )


def _find_dataset(config: Mapping[str, Any], candidate_id: str) -> Mapping[str, Any]:
    for dataset in config.get("datasets", []):
        if dataset.get("candidate_id") == candidate_id:
            return dataset
    raise ValueError(f"Missing frozen dataset: {candidate_id}")


def _all_files(dataset: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for source in dataset.get("sources", []):
        for file_info in source.get("files", []):
            file_id = str(file_info.get("file_id", ""))
            if not file_id or file_id in result:
                raise ValueError("Dataset file IDs must be unique and non-empty")
            result[file_id] = file_info
    return result


def validate_dataset_portfolio_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the immutable, outcome-blind A--E portfolio contract."""

    _require_exact(config.get("schema_name"), SCHEMA_NAME, "schema_name")
    _require_exact(config.get("schema_version"), SCHEMA_VERSION, "schema_version")
    _require_exact(config.get("portfolio_id"), PORTFOLIO_ID, "portfolio_id")
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
        "selection_uses_pathway_outcomes": False,
        "pathway_outcomes_read": False,
        "expression_values_read_during_portfolio_build": False,
        "network_requests_during_portfolio_build": False,
        "t21_timing_no_go_unchanged": True,
        "t21_formal_discovery_no_go_unchanged": True,
        "prior_dataset_screen_v1_retained_as_provenance": True,
        "portfolio_v2_supersedes_v1_candidate_roles_only": True,
    }.items():
        _require_exact(scope.get(key), expected, f"scope.{key}")

    policy = config.get("portfolio_policy", {})
    _require_exact(
        tuple(policy.get("stage_order", [])), EXPECTED_STAGE_IDS, "stage_order"
    )
    for key in (
        "current_core_requires_donor_constant_condition",
        "paired_repeated_measures_extension_deferred",
        "full_matrix_acquisition_requires_explicit_dataset_authorization",
        "pathway_scoring_requires_structural_preflight_and_separate_freeze",
        "biological_discovery_requires_predeclared_discovery_and_validation_sets",
        "true_time_labels_must_be_hidden_when_declared_validation_only",
        "normalized_expression_must_not_be_treated_as_raw_counts",
        "fine_calendar_onset_from_three_nominal_visits_forbidden",
    ):
        _require_exact(policy.get(key), True, f"portfolio_policy.{key}")

    datasets = config.get("datasets", [])
    _require_exact(
        tuple(dataset.get("stage_id") for dataset in datasets),
        EXPECTED_STAGE_IDS,
        "dataset_stage_ids",
    )
    _require_exact(
        tuple(dataset.get("candidate_id") for dataset in datasets),
        EXPECTED_CANDIDATE_IDS,
        "candidate_ids",
    )
    _require_exact(
        tuple(dataset.get("role") for dataset in datasets),
        EXPECTED_ROLES,
        "dataset_roles",
    )
    _require_exact(
        tuple(dataset.get("priority_order") for dataset in datasets),
        (1, 2, 3, 4, 5),
        "priority_order",
    )

    source_ids: set[str] = set()
    authorized_full_matrices: list[str] = []
    for dataset in datasets:
        files = _all_files(dataset)
        acquisition = dataset.get("acquisition", {})
        authorized_file_ids = tuple(acquisition.get("authorized_file_ids", []))
        if set(authorized_file_ids) - set(files):
            raise ValueError(
                "Every authorized file ID must exist in its source manifest"
            )
        if bool(
            acquisition.get("full_processed_primary_matrix_authorized", False)
            or acquisition.get("full_expression_matrix_authorized", False)
            or acquisition.get("b_plasma_matrix_authorized", False)
        ):
            authorized_full_matrices.append(str(dataset["candidate_id"]))
        _require_exact(
            acquisition.get("pathway_scoring_authorized"),
            False,
            f"{dataset['candidate_id']}.pathway_scoring_authorized",
        )
        claims = dataset.get("claim_policy", {})
        _require_exact(
            claims.get("fine_calendar_onset_claims", False),
            False,
            f"{dataset['candidate_id']}.fine_calendar_onset_claims",
        )
        for source in dataset.get("sources", []):
            source_id = str(source.get("source_id", ""))
            if not source_id or source_id in source_ids:
                raise ValueError("Source IDs must be globally unique and non-empty")
            source_ids.add(source_id)
            parsed = urlparse(str(source.get("source_url", "")))
            if parsed.scheme != "https" or not parsed.hostname:
                raise ValueError("All frozen sources must use an HTTPS primary URL")
            _require_exact(source.get("verified_at_freeze"), True, source_id)
            for file_info in source.get("files", []):
                file_parsed = urlparse(str(file_info.get("file_url", "")))
                if file_parsed.scheme != "https" or not file_parsed.hostname:
                    raise ValueError("All frozen files must use an HTTPS URL")
                if int(file_info.get("size_bytes", 0)) <= 0:
                    raise ValueError("Frozen file sizes must be positive")

    _require_exact(
        tuple(authorized_full_matrices),
        ("hipsci_endoderm_125_v2",),
        "full_matrix_authorizations",
    )

    endoderm = _find_dataset(config, "hipsci_endoderm_125_v2")
    _require_exact(
        endoderm["acquisition"]["decision"], "authorized_now", "endoderm.decision"
    )
    _require_exact(
        tuple(endoderm["acquisition"]["authorized_file_ids"]),
        ("endoderm_raw_counts", "endoderm_cell_metadata"),
        "endoderm.authorized_file_ids",
    )
    dopamine = _find_dataset(config, "hipsci_dopaminergic_215_v2")
    _require_exact(
        dopamine["acquisition"]["decision"],
        "metadata_only_authorized",
        "dopaminergic.decision",
    )
    sound_life = _find_dataset(config, "sound_life_flu_96_v2")
    _require_exact(
        sound_life["acquisition"]["b_plasma_matrix_authorized"],
        False,
        "sound_life.b_plasma_matrix_authorized",
    )
    scblood = _find_dataset(config, "scbloodnl_pathogen_120_v2")
    _require_exact(
        scblood["acquisition"]["decision"],
        "deferred_until_paired_v2",
        "scbloodnl.decision",
    )

    benchmark = config.get("endoderm_benchmark_contract", {})
    for key, expected in {
        "role": "method_benchmark",
        "true_time_labels_used_for_trajectory": False,
        "true_time_labels_used_for_validation": True,
        "existing_trajectory_fields_used_for_trajectory": False,
        "primary_estimand": "pathway_curve_and_event_recovery",
        "biological_discovery_claims": False,
        "donor_equal_estimand_primary": True,
    }.items():
        _require_exact(benchmark.get(key), expected, f"endoderm_contract.{key}")
    _require_exact(
        tuple(benchmark.get("downsample_donor_counts", [])),
        (125, 60, 30, 20, 10, 6, 3),
        "endoderm_contract.downsample_donor_counts",
    )

    prereg = config.get("sound_life_discovery_preregistration", {})
    _require_exact(
        prereg.get("status"),
        "conditional_go_pending_metadata_preflight",
        "sound_life.status",
    )
    _require_exact(
        prereg.get("discovery", {}).get("series"), "Flu Year 1", "sound_life.discovery"
    )
    _require_exact(
        prereg.get("temporal_reproducibility", {}).get("independent_donor_replication"),
        False,
        "sound_life.independent_replication",
    )
    _require_exact(
        prereg.get("fine_calendar_onset_claims"), False, "sound_life.fine_onset"
    )

    paired = config.get("paired_extension_deferral", {})
    _require_exact(
        paired.get("development_status"), "deferred", "paired.development_status"
    )
    _require_exact(
        paired.get("development_authorized_now"),
        False,
        "paired.development_authorized_now",
    )

    result = deepcopy(dict(config))
    result["_config_payload_sha256"] = FROZEN_CONFIG_PAYLOAD_SHA256
    return result


def load_dataset_portfolio_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Dataset-portfolio config must be a YAML mapping")
    return validate_dataset_portfolio_config(value)


def _reported_value(reported: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in reported:
            return reported[key]
    return "not_applicable"


def _build_role_matrix(config: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in config["datasets"]:
        reported = dataset["reported_design"]
        acquisition = dataset["acquisition"]
        claims = dataset["claim_policy"]
        full_authorized = bool(
            acquisition.get("full_processed_primary_matrix_authorized", False)
            or acquisition.get("full_expression_matrix_authorized", False)
            or acquisition.get("b_plasma_matrix_authorized", False)
        )
        metadata_authorized = bool(
            acquisition.get("metadata_only_authorized", False)
            or acquisition.get("small_public_metadata_authorized", False)
            or acquisition.get("decision") == "metadata_only_stress_test"
            or full_authorized
        )
        rows.append(
            {
                "stage_id": dataset["stage_id"],
                "priority_order": dataset["priority_order"],
                "candidate_id": dataset["candidate_id"],
                "display_name": dataset["display_name"],
                "role": dataset["role"],
                "independent_unit": reported["independent_unit"],
                "n_independent_units_reported": _reported_value(
                    reported, "n_independent_units", "n_cell_lines"
                ),
                "timepoints_reported": _reported_value(
                    reported,
                    "timepoints",
                    "response_times",
                    "nasopharyngeal_timepoints",
                ),
                "n_cells_reported": _reported_value(
                    reported,
                    "n_qc_cells",
                    "n_qc_cells_primary_report",
                    "n_single_cells",
                ),
                "statistical_core_required": dataset["statistical_core_required"],
                "acquisition_decision": acquisition["decision"],
                "full_expression_matrix_authorized_now": full_authorized,
                "metadata_acquisition_authorized_now": metadata_authorized,
                "pathway_scoring_authorized": False,
                "biological_discovery_claims_status": claims.get(
                    "biological_discovery_claims", "not_applicable"
                ),
                "fine_calendar_onset_claims_allowed": False,
                "pathway_outcomes_read": False,
                "selection_uses_pathway_outcomes": False,
                "next_gate": acquisition["next_gate"],
            }
        )
    return pd.DataFrame(rows, columns=list(ROLE_COLUMNS))


def _build_source_manifest(config: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in config["datasets"]:
        authorized = set(dataset["acquisition"].get("authorized_file_ids", []))
        for source in dataset["sources"]:
            files = source.get("files", []) or [None]
            for file_info in files:
                info = file_info or {}
                file_id = str(info.get("file_id", "not_applicable"))
                rows.append(
                    {
                        "stage_id": dataset["stage_id"],
                        "candidate_id": dataset["candidate_id"],
                        "source_id": source["source_id"],
                        "source_kind": source["source_kind"],
                        "source_authority": source["source_authority"],
                        "source_url": source["source_url"],
                        "accession": source["accession"],
                        "access_class": source["access_class"],
                        "supports_fields": source["supports_fields"],
                        "verified_at_freeze": True,
                        "file_id": file_id,
                        "file_name": info.get("file_name", "not_applicable"),
                        "file_url": info.get("file_url", "not_applicable"),
                        "size_bytes": info.get("size_bytes", 0),
                        "publisher_checksum_algorithm": info.get(
                            "publisher_checksum_algorithm", "not_applicable"
                        ),
                        "publisher_checksum": info.get(
                            "publisher_checksum", "not_applicable"
                        ),
                        "source_audit_sha256": info.get(
                            "source_audit_sha256", "not_frozen"
                        ),
                        "content_contract": info.get(
                            "content_contract", "not_applicable"
                        ),
                        "download_authorized_now": file_id in authorized,
                        "pathway_outcomes_read": False,
                    }
                )
    return pd.DataFrame(rows, columns=list(SOURCE_COLUMNS))


def build_dataset_portfolio_artifacts(
    config: Mapping[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    validated = validate_dataset_portfolio_config(config)
    role_matrix = _build_role_matrix(validated)
    source_manifest = _build_source_manifest(validated)
    authorized = source_manifest.loc[
        source_manifest["download_authorized_now"].astype(bool), "file_id"
    ].tolist()
    decision = {
        "schema_name": "trajpathmix_dataset_authorization_decision",
        "schema_version": "2.0.0",
        "portfolio_id": PORTFOLIO_ID,
        "config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
        "created_at_utc": validated["frozen_at_utc"],
        "decision_basis": "primary_and_official_design_metadata_only",
        "selection_uses_pathway_outcomes": False,
        "pathway_outcomes_read": False,
        "t21_decisions_reopened": False,
        "authorized_full_matrix_candidate_ids": ["hipsci_endoderm_125_v2"],
        "authorized_file_ids": authorized,
        "pathway_scoring_authorized_candidate_ids": [],
        "current_primary_method_benchmark": "hipsci_endoderm_125_v2",
        "current_primary_biological_discovery_candidate": "sound_life_flu_96_v2",
        "primary_biological_discovery_matrix_download_authorized": False,
        "paired_extension_development_authorized_now": False,
        "candidate_decisions": [
            {
                "stage_id": dataset["stage_id"],
                "candidate_id": dataset["candidate_id"],
                "role": dataset["role"],
                "acquisition_decision": dataset["acquisition"]["decision"],
                "pathway_scoring_authorized": False,
                "next_gate": dataset["acquisition"]["next_gate"],
            }
            for dataset in validated["datasets"]
        ],
        "forbidden_actions": [
            "reopen_t21_timing_or_formal_discovery_without_new_donors_or_full_trajectory_data",
            "use_true_day_or_existing_trajectory_fields_to_build_endoderm_trajectory",
            "treat_dopaminergic_normalized_expression_as_raw_counts",
            "download_sound_life_b_plasma_before_metadata_preflight_pass",
            "call_sound_life_year2_independent_donor_replication",
            "develop_or_analyze_scbloodnl_with_the_current_independent_group_core",
            "use_sarscov2_challenge_as_the_main_biological_discovery",
            "score_pathways_before_dataset_specific_structural_and_mde_freeze",
        ],
    }
    contracts = {
        AUTHORIZATION_DECISION_FILE: decision,
        ENDODERM_CONTRACT_FILE: {
            "schema_name": "trajpathmix_endoderm_benchmark_contract",
            "schema_version": "1.0.0",
            "portfolio_id": PORTFOLIO_ID,
            "config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
            "pathway_outcomes_read": False,
            **deepcopy(validated["endoderm_benchmark_contract"]),
        },
        SOUND_LIFE_PREREG_FILE: {
            "schema_name": "trajpathmix_sound_life_discovery_preregistration",
            "schema_version": "1.0.0",
            "portfolio_id": PORTFOLIO_ID,
            "config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
            "selection_uses_pathway_outcomes": False,
            "pathway_outcomes_read": False,
            **deepcopy(validated["sound_life_discovery_preregistration"]),
        },
        PAIRED_DEFERRAL_FILE: {
            "schema_name": "trajpathmix_paired_extension_deferral",
            "schema_version": "1.0.0",
            "portfolio_id": PORTFOLIO_ID,
            "config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
            "pathway_outcomes_read": False,
            **deepcopy(validated["paired_extension_deferral"]),
        },
    }
    return {
        ROLE_MATRIX_FILE: role_matrix,
        SOURCE_MANIFEST_FILE: source_manifest,
    }, contracts


def _write_table(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, sep="\t", index=False, lineterminator="\n")


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _build_record(
    config: Mapping[str, Any],
    *,
    config_file: Path,
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    source_urls = [
        source["source_url"]
        for dataset in config["datasets"]
        for source in dataset["sources"]
    ]
    return {
        "schema_name": "trajpathmix_dataset_portfolio_build_record",
        "schema_version": "2.0.0",
        "portfolio_id": PORTFOLIO_ID,
        "portfolio_frozen_at_utc": config["frozen_at_utc"],
        "config_file": "config/trajpathmix_dataset_portfolio_v2.yaml",
        "config_file_sha256": _sha256_file(config_file),
        "config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
        "implementation_file": "pyfgsea/trajpathmix_dataset_portfolio.py",
        "implementation_sha256": _sha256_file(Path(__file__).resolve()),
        "artifacts": dict(artifacts),
        "primary_official_source_urls": source_urls,
        "network_requests_during_build": False,
        "expression_values_read_during_build": False,
        "pathway_outcomes_read": False,
        "selection_uses_pathway_outcomes": False,
        "t21_timing_no_go_unchanged": True,
        "t21_formal_discovery_no_go_unchanged": True,
        "authorized_full_matrix_candidate_ids": ["hipsci_endoderm_125_v2"],
        "pathway_scoring_authorized_candidate_ids": [],
    }


def write_dataset_portfolio_artifacts(
    config: Mapping[str, Any],
    *,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create the frozen portfolio atomically; an existing output is immutable."""

    validated = validate_dataset_portfolio_config(config)
    tables, contracts = build_dataset_portfolio_artifacts(validated)
    config_file = Path(config_path).resolve()
    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Dataset-portfolio output already exists: {output}")

    lock_path = output.parent / f".{output.name}.create.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Dataset-portfolio output creation is locked: {lock_path}"
        ) from exc

    temporary_output: Path | None = None
    try:
        os.close(lock_fd)
        temporary_output = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
        )
        for filename, table in tables.items():
            _write_table(table, temporary_output / filename)
        for filename, value in contracts.items():
            _write_json(value, temporary_output / filename)

        artifacts: dict[str, Any] = {}
        for filename, table in tables.items():
            artifacts[filename] = {
                "sha256": _sha256_file(temporary_output / filename),
                "rows": int(len(table)),
                "columns": list(table.columns),
            }
        for filename, value in contracts.items():
            artifacts[filename] = {
                "sha256": _sha256_file(temporary_output / filename),
                "schema_name": value["schema_name"],
            }

        build_record = _build_record(
            validated,
            config_file=config_file,
            artifacts=artifacts,
        )
        _write_json(build_record, temporary_output / BUILD_RECORD_FILE)
        if output.exists():
            raise FileExistsError(f"Dataset-portfolio output already exists: {output}")
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


def build_and_write_dataset_portfolio(
    *, config_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    config = load_dataset_portfolio_config(config_path)
    return write_dataset_portfolio_artifacts(
        config, config_path=config_path, output_dir=output_dir
    )


def _normalized_table(table: pd.DataFrame) -> pd.DataFrame:
    return table.fillna("").astype(str).reset_index(drop=True)


def validate_dataset_portfolio_output(
    *, config_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Rebuild expected values in memory and validate every frozen artifact."""

    config_file = Path(config_path).resolve()
    config = load_dataset_portfolio_config(config_file)
    output = Path(output_dir).resolve()
    expected_names = {
        ROLE_MATRIX_FILE,
        SOURCE_MANIFEST_FILE,
        AUTHORIZATION_DECISION_FILE,
        ENDODERM_CONTRACT_FILE,
        SOUND_LIFE_PREREG_FILE,
        PAIRED_DEFERRAL_FILE,
        BUILD_RECORD_FILE,
    }
    if not output.is_dir():
        raise FileNotFoundError(f"Dataset-portfolio output is missing: {output}")
    observed_names = {path.name for path in output.iterdir() if path.is_file()}
    _require_exact(observed_names, expected_names, "output_file_set")

    expected_tables, expected_contracts = build_dataset_portfolio_artifacts(config)
    expected_artifacts: dict[str, Any] = {}
    for filename, expected_table in expected_tables.items():
        path = output / filename
        observed = pd.read_csv(path, sep="\t", keep_default_na=False)
        if not _normalized_table(observed).equals(_normalized_table(expected_table)):
            raise ValueError(f"Frozen dataset-portfolio table mismatch: {filename}")
        expected_artifacts[filename] = {
            "sha256": _sha256_file(path),
            "rows": int(len(expected_table)),
            "columns": list(expected_table.columns),
        }
    for filename, expected_value in expected_contracts.items():
        path = output / filename
        observed_value = json.loads(path.read_text(encoding="utf-8"))
        _require_exact(observed_value, expected_value, filename)
        expected_artifacts[filename] = {
            "sha256": _sha256_file(path),
            "schema_name": expected_value["schema_name"],
        }

    record = json.loads((output / BUILD_RECORD_FILE).read_text(encoding="utf-8"))
    expected_record = _build_record(
        config,
        config_file=config_file,
        artifacts=expected_artifacts,
    )
    _require_exact(record, expected_record, "build_record")

    result = dict(record)
    result["output_dir"] = str(output)
    result["build_record_sha256"] = _sha256_file(output / BUILD_RECORD_FILE)
    result["validation_status"] = "pass_frozen_outcome_blind_portfolio_v2"
    return result


__all__ = [
    "AUTHORIZATION_DECISION_FILE",
    "BUILD_RECORD_FILE",
    "ENDODERM_CONTRACT_FILE",
    "EXPECTED_CANDIDATE_IDS",
    "EXPECTED_ROLES",
    "EXPECTED_STAGE_IDS",
    "FROZEN_CONFIG_PAYLOAD_SHA256",
    "PAIRED_DEFERRAL_FILE",
    "PORTFOLIO_ID",
    "ROLE_MATRIX_FILE",
    "SOUND_LIFE_PREREG_FILE",
    "SOURCE_MANIFEST_FILE",
    "build_and_write_dataset_portfolio",
    "build_dataset_portfolio_artifacts",
    "load_dataset_portfolio_config",
    "validate_dataset_portfolio_config",
    "validate_dataset_portfolio_output",
    "write_dataset_portfolio_artifacts",
]
