from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import pandas as pd
import yaml


SCHEMA_NAME = "trajpathmix_endoderm_metadata_audit_contract"
SCHEMA_VERSION = "1.0.0"
AUDIT_ID = "trajpathmix_endoderm_metadata_audit_v1"
FROZEN_CONFIG_PAYLOAD_SHA256 = (
    "70fc17fcd42926f4d51a344cf8718302439f933d47f108b38088f18ea201d49b"
)

DAY_SUMMARY_FILE = "endoderm_day_support_summary_v1.tsv"
LINE_DAY_SUPPORT_FILE = "endoderm_line_day_support_v1.tsv"
HIDDEN_FIELDS_FILE = "endoderm_trajectory_hidden_fields_v1.tsv"
AUDIT_FILE = "endoderm_metadata_structural_audit_v1.json"
EVIDENCE_REVISION_FILE = "portfolio_evidence_revision_v1.json"
BUILD_RECORD_FILE = "endoderm_metadata_audit_build_record_v1.json"

DAY_SUMMARY_COLUMNS = (
    "day",
    "n_cells",
    "n_donors",
    "n_lines",
    "day_used_for_trajectory",
    "allowed_use",
)
LINE_DAY_COLUMNS = (
    "line_id",
    "donor_id",
    "n_experiments",
    "day0_cells",
    "day1_cells",
    "day2_cells",
    "day3_cells",
    "minimum_day_cells",
    "all_days_present",
    "all_days_over_10",
    "day3_over_10",
)
HIDDEN_FIELD_COLUMNS = (
    "column_name",
    "present_in_metadata",
    "used_for_trajectory",
    "allowed_use",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _hash_file(path: Path, algorithm: str, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.new(algorithm)
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
            f"Frozen endoderm metadata-audit mismatch for {label}: "
            f"expected {expected!r}, observed {value!r}"
        )


def validate_endoderm_metadata_audit_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    _require_exact(config.get("schema_name"), SCHEMA_NAME, "schema_name")
    _require_exact(config.get("schema_version"), SCHEMA_VERSION, "schema_version")
    _require_exact(config.get("audit_id"), AUDIT_ID, "audit_id")
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
    binding = config.get("portfolio_binding", {})
    _require_exact(
        binding.get("portfolio_id"),
        "trajpathmix_dataset_portfolio_v2",
        "portfolio_id",
    )
    _require_exact(
        binding.get("candidate_id"), "hipsci_endoderm_125_v2", "candidate_id"
    )
    _require_exact(
        binding.get("evidence_revision_mode"), "append_only", "revision_mode"
    )
    scope = config.get("scope", {})
    for key, expected in {
        "metadata_only": True,
        "expression_matrix_read": False,
        "expression_values_read": False,
        "pathway_outcomes_read": False,
        "pathway_scoring_performed": False,
        "day_used_for_trajectory": False,
        "deposited_trajectory_fields_used_for_trajectory": False,
    }.items():
        _require_exact(scope.get(key), expected, f"scope.{key}")
    identifiers = config.get("identifiers", {})
    _require_exact(
        tuple(identifiers.get("expected_day_order", [])),
        ("day0", "day1", "day2", "day3"),
        "expected_day_order",
    )
    revision = config.get("evidence_revision", {})
    _require_exact(revision.get("superseded_value"), 98, "superseded_value")
    _require_exact(revision.get("corrected_value"), 75, "corrected_value")
    _require_exact(
        revision.get("portfolio_role_or_authorization_changed"),
        False,
        "portfolio_role_or_authorization_changed",
    )
    _require_exact(
        revision.get("pathway_scoring_authorized"),
        False,
        "pathway_scoring_authorized",
    )
    result = deepcopy(dict(config))
    result["_config_payload_sha256"] = FROZEN_CONFIG_PAYLOAD_SHA256
    return result


def load_endoderm_metadata_audit_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Endoderm metadata-audit config must be a YAML mapping")
    return validate_endoderm_metadata_audit_config(value)


def _verify_input(path: Path, config: Mapping[str, Any]) -> None:
    expected = config["input"]
    if not path.is_file():
        raise FileNotFoundError(f"Endoderm metadata input is missing: {path}")
    _require_exact(path.stat().st_size, int(expected["size_bytes"]), "input.size")
    _require_exact(
        _hash_file(path, "md5"), expected["publisher_md5"], "input.publisher_md5"
    )
    _require_exact(
        _hash_file(path, "sha256"), expected["local_sha256"], "input.local_sha256"
    )


def _build_tables_and_audit(
    config: Mapping[str, Any], metadata_path: Path
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    _verify_input(metadata_path, config)
    ids = config["identifiers"]
    days = list(ids["expected_day_order"])
    required = [
        ids["cell_id_column"],
        ids["donor_id_column"],
        ids["line_id_column"],
        ids["line_short_id_column"],
        ids["experiment_id_column"],
        ids["day_column"],
        "cell_filter",
        "used_in_expt",
        "well_type",
    ]
    header = pd.read_csv(metadata_path, sep="\t", nrows=0).columns.tolist()
    missing = sorted(set(required) - set(header))
    if missing:
        raise ValueError(f"Required endoderm metadata columns are missing: {missing}")
    frame = pd.read_csv(metadata_path, sep="\t", usecols=required)
    if frame[required].isna().any().any():
        raise ValueError("Required endoderm metadata fields contain missing values")
    _require_exact(
        sorted(frame[ids["day_column"]].unique().tolist()), sorted(days), "day_values"
    )
    if not frame[ids["cell_id_column"]].is_unique:
        raise ValueError("Endoderm metadata cell IDs are not unique")
    _require_exact(frame["cell_filter"].eq(True).all(), True, "cell_filter")  # noqa: E712
    _require_exact(frame["used_in_expt"].eq(True).all(), True, "used_in_expt")  # noqa: E712
    _require_exact(frame["well_type"].eq("single cell").all(), True, "well_type")

    day_group = frame.groupby(ids["day_column"], observed=True)
    day_summary = pd.DataFrame(
        [
            {
                "day": day,
                "n_cells": int((frame[ids["day_column"]] == day).sum()),
                "n_donors": int(
                    day_group.get_group(day)[ids["donor_id_column"]].nunique()
                ),
                "n_lines": int(
                    day_group.get_group(day)[ids["line_id_column"]].nunique()
                ),
                "day_used_for_trajectory": False,
                "allowed_use": "locked_validation_and_structural_support_audit_only",
            }
            for day in days
        ],
        columns=list(DAY_SUMMARY_COLUMNS),
    )

    counts = (
        frame.groupby([ids["line_id_column"], ids["day_column"]], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=days, fill_value=0)
        .sort_index()
    )
    donor_by_line = frame.groupby(ids["line_id_column"])[
        ids["donor_id_column"]
    ].nunique()
    _require_exact(donor_by_line.eq(1).all(), True, "line_to_donor_mapping")
    donor_map = frame.groupby(ids["line_id_column"])[ids["donor_id_column"]].first()
    experiments = frame.groupby(ids["line_id_column"])[
        ids["experiment_id_column"]
    ].nunique()
    line_day = pd.DataFrame(
        {
            "line_id": counts.index,
            "donor_id": donor_map.reindex(counts.index).values,
            "n_experiments": experiments.reindex(counts.index).astype(int).values,
            "day0_cells": counts["day0"].astype(int).values,
            "day1_cells": counts["day1"].astype(int).values,
            "day2_cells": counts["day2"].astype(int).values,
            "day3_cells": counts["day3"].astype(int).values,
            "minimum_day_cells": counts.min(axis=1).astype(int).values,
            "all_days_present": counts.gt(0).all(axis=1).values,
            "all_days_over_10": counts.gt(10).all(axis=1).values,
            "day3_over_10": counts["day3"].gt(10).values,
        },
        columns=list(LINE_DAY_COLUMNS),
    )

    hidden = config["trajectory_blinding"]["exact_hidden_columns"]
    hidden_fields = pd.DataFrame(
        [
            {
                "column_name": column,
                "present_in_metadata": column in header,
                "used_for_trajectory": False,
                "allowed_use": "locked_validation_only",
            }
            for column in hidden
        ],
        columns=list(HIDDEN_FIELD_COLUMNS),
    )
    _require_exact(
        hidden_fields["present_in_metadata"].all(), True, "hidden_field_presence"
    )

    donor_experiments = frame.groupby(ids["donor_id_column"])[
        ids["experiment_id_column"]
    ].nunique()
    observed = {
        "n_rows": int(len(frame)),
        "n_unique_cells": int(frame[ids["cell_id_column"]].nunique()),
        "n_donors": int(frame[ids["donor_id_column"]].nunique()),
        "n_lines": int(frame[ids["line_id_column"]].nunique()),
        "n_experiments": int(frame[ids["experiment_id_column"]].nunique()),
        "n_donors_in_multiple_experiments": int(donor_experiments.gt(1).sum()),
        "day_cell_counts": dict(zip(day_summary["day"], day_summary["n_cells"])),
        "day_donor_counts": dict(zip(day_summary["day"], day_summary["n_donors"])),
        "day_line_counts": dict(zip(day_summary["day"], day_summary["n_lines"])),
        "n_lines_all_days_present": int(line_day["all_days_present"].sum()),
        "n_lines_all_days_over_10": int(line_day["all_days_over_10"].sum()),
        "n_lines_day3_over_10": int(line_day["day3_over_10"].sum()),
    }
    _require_exact(
        observed, config["expected_observed_structure"], "observed_structure"
    )
    audit = {
        "schema_name": "trajpathmix_endoderm_metadata_structural_audit",
        "schema_version": "1.0.0",
        "audit_id": AUDIT_ID,
        "config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
        "candidate_id": config["portfolio_binding"]["candidate_id"],
        "input_relative_path": config["input"]["relative_path"],
        "input_size_bytes": config["input"]["size_bytes"],
        "input_publisher_md5": config["input"]["publisher_md5"],
        "input_local_sha256": config["input"]["local_sha256"],
        "n_metadata_columns": int(len(header)),
        "observed_structure": observed,
        "all_required_fields_complete": True,
        "all_hidden_fields_present": True,
        "metadata_only": True,
        "expression_matrix_read": False,
        "expression_values_read": False,
        "pathway_outcomes_read": False,
        "pathway_scoring_performed": False,
        "day_used_for_trajectory": False,
        "deposited_trajectory_fields_used_for_trajectory": False,
        "structural_audit_status": "pass_with_append_only_evidence_revision",
        "pathway_scoring_authorized": False,
        "next_gate": "raw_count_acquisition_then_fractional_count_schema_and_blinded_trajectory_preflight",
    }
    revision = {
        "schema_name": "trajpathmix_portfolio_evidence_revision",
        "schema_version": "1.0.0",
        "revision_id": "trajpathmix_dataset_portfolio_v2_evidence_revision_1",
        "portfolio_id": config["portfolio_binding"]["portfolio_id"],
        "portfolio_config_payload_sha256": config["portfolio_binding"][
            "portfolio_config_payload_sha256"
        ],
        "candidate_id": config["portfolio_binding"]["candidate_id"],
        "revision_mode": "append_only",
        **deepcopy(config["evidence_revision"]),
        "evidence_source": config["input"]["relative_path"],
        "evidence_source_sha256": config["input"]["local_sha256"],
        "pathway_outcomes_read": False,
    }
    return {
        DAY_SUMMARY_FILE: day_summary,
        LINE_DAY_SUPPORT_FILE: line_day,
        HIDDEN_FIELDS_FILE: hidden_fields,
    }, {
        AUDIT_FILE: audit,
        EVIDENCE_REVISION_FILE: revision,
    }


def _write_table(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, sep="\t", index=False, lineterminator="\n")


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def build_and_write_endoderm_metadata_audit(
    *,
    config_path: str | Path,
    repository_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config = load_endoderm_metadata_audit_config(config_file)
    root = Path(repository_root).resolve()
    metadata_path = root / config["input"]["relative_path"]
    tables, json_artifacts = _build_tables_and_audit(config, metadata_path)
    output = Path(output_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(
            f"Endoderm metadata-audit output already exists: {output}"
        )
    lock_path = output.parent / f".{output.name}.create.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Endoderm metadata audit is locked: {lock_path}"
        ) from exc
    temporary: Path | None = None
    try:
        os.close(lock_fd)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
        )
        for filename, table in tables.items():
            _write_table(table, temporary / filename)
        for filename, value in json_artifacts.items():
            _write_json(value, temporary / filename)
        artifacts: dict[str, Any] = {}
        for filename, table in tables.items():
            artifacts[filename] = {
                "sha256": _hash_file(temporary / filename, "sha256"),
                "rows": int(len(table)),
                "columns": list(table.columns),
            }
        for filename, value in json_artifacts.items():
            artifacts[filename] = {
                "sha256": _hash_file(temporary / filename, "sha256"),
                "schema_name": value["schema_name"],
            }
        record = {
            "schema_name": "trajpathmix_endoderm_metadata_audit_build_record",
            "schema_version": "1.0.0",
            "audit_id": AUDIT_ID,
            "audit_frozen_at_utc": config["frozen_at_utc"],
            "config_file": "config/trajpathmix_endoderm_metadata_audit_v1.yaml",
            "config_file_sha256": _hash_file(config_file, "sha256"),
            "config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
            "implementation_file": "pyfgsea/trajpathmix_endoderm_metadata_audit.py",
            "implementation_sha256": _hash_file(Path(__file__).resolve(), "sha256"),
            "artifacts": artifacts,
            "metadata_only": True,
            "expression_matrix_read": False,
            "pathway_outcomes_read": False,
            "pathway_scoring_authorized": False,
            "evidence_revision_mode": "append_only",
        }
        _write_json(record, temporary / BUILD_RECORD_FILE)
        os.rename(temporary, output)
        temporary = None
        result = dict(record)
        result["output_dir"] = str(output)
        result["build_record_sha256"] = _hash_file(output / BUILD_RECORD_FILE, "sha256")
        return result
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
        lock_path.unlink(missing_ok=True)


def validate_endoderm_metadata_audit_output(
    *,
    config_path: str | Path,
    repository_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config = load_endoderm_metadata_audit_config(config_file)
    root = Path(repository_root).resolve()
    metadata_path = root / config["input"]["relative_path"]
    expected_tables, expected_json = _build_tables_and_audit(config, metadata_path)
    output = Path(output_dir).resolve()
    expected_names = {
        *expected_tables,
        *expected_json,
        BUILD_RECORD_FILE,
    }
    if not output.is_dir():
        raise FileNotFoundError(f"Endoderm metadata-audit output is missing: {output}")
    _require_exact(
        {path.name for path in output.iterdir() if path.is_file()},
        expected_names,
        "output_file_set",
    )
    artifacts: dict[str, Any] = {}
    for filename, expected in expected_tables.items():
        path = output / filename
        observed = pd.read_csv(path, sep="\t", keep_default_na=False)
        if not observed.fillna("").astype(str).equals(expected.fillna("").astype(str)):
            raise ValueError(f"Endoderm metadata-audit table mismatch: {filename}")
        artifacts[filename] = {
            "sha256": _hash_file(path, "sha256"),
            "rows": int(len(expected)),
            "columns": list(expected.columns),
        }
    for filename, expected in expected_json.items():
        path = output / filename
        _require_exact(json.loads(path.read_text(encoding="utf-8")), expected, filename)
        artifacts[filename] = {
            "sha256": _hash_file(path, "sha256"),
            "schema_name": expected["schema_name"],
        }
    expected_record = {
        "schema_name": "trajpathmix_endoderm_metadata_audit_build_record",
        "schema_version": "1.0.0",
        "audit_id": AUDIT_ID,
        "audit_frozen_at_utc": config["frozen_at_utc"],
        "config_file": "config/trajpathmix_endoderm_metadata_audit_v1.yaml",
        "config_file_sha256": _hash_file(config_file, "sha256"),
        "config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
        "implementation_file": "pyfgsea/trajpathmix_endoderm_metadata_audit.py",
        "implementation_sha256": _hash_file(Path(__file__).resolve(), "sha256"),
        "artifacts": artifacts,
        "metadata_only": True,
        "expression_matrix_read": False,
        "pathway_outcomes_read": False,
        "pathway_scoring_authorized": False,
        "evidence_revision_mode": "append_only",
    }
    record = json.loads((output / BUILD_RECORD_FILE).read_text(encoding="utf-8"))
    _require_exact(record, expected_record, "build_record")
    result = dict(record)
    result["output_dir"] = str(output)
    result["build_record_sha256"] = _hash_file(output / BUILD_RECORD_FILE, "sha256")
    result["validation_status"] = "pass_metadata_only_append_only_revision"
    return result


__all__ = [
    "AUDIT_FILE",
    "AUDIT_ID",
    "BUILD_RECORD_FILE",
    "DAY_SUMMARY_FILE",
    "EVIDENCE_REVISION_FILE",
    "FROZEN_CONFIG_PAYLOAD_SHA256",
    "HIDDEN_FIELDS_FILE",
    "LINE_DAY_SUPPORT_FILE",
    "build_and_write_endoderm_metadata_audit",
    "load_endoderm_metadata_audit_config",
    "validate_endoderm_metadata_audit_config",
    "validate_endoderm_metadata_audit_output",
]
