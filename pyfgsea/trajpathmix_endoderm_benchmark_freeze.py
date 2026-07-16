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


SCHEMA_NAME = "trajpathmix_endoderm_benchmark_freeze_contract"
SCHEMA_VERSION = "1.0.0"
FREEZE_ID = "trajpathmix_endoderm_benchmark_freeze_v1"
FROZEN_CONFIG_PAYLOAD_SHA256 = (
    "0754b2d41b5b6d532f5cce7f21d91fc87a98cf66427fb734c02eb03b3c72acd2"
)

LINE_EXPERIMENT_DAY_FILE = (
    "endoderm_line_donor_experiment_day_cell_counts_v1.tsv"
)
DONOR_DAY_FILE = "endoderm_donor_day_cell_counts_v1.tsv"
DONOR_COHORT_FILE = "endoderm_donor_cohort_membership_v1.tsv"
EXPERIMENT_DAY_FILE = "endoderm_experiment_day_support_v1.tsv"
STATISTICAL_UNIT_AUDIT_FILE = "endoderm_statistical_unit_audit_v1.json"
BENCHMARK_CONTRACT_FILE = "endoderm_benchmark_contract_freeze_v1.json"
BUILD_RECORD_FILE = "endoderm_benchmark_freeze_build_record_v1.json"

LINE_EXPERIMENT_DAY_COLUMNS = (
    "line_id",
    "donor_id",
    "experiment_id",
    "day",
    "cell_count",
)
DONOR_DAY_COLUMNS = (
    "donor_id",
    "day",
    "cell_count",
    "n_lines",
    "n_experiments",
)
DONOR_COHORT_COLUMNS = (
    "donor_id",
    "n_lines",
    "n_experiments",
    "day0_cells",
    "day1_cells",
    "day2_cells",
    "day3_cells",
    "minimum_day_cells",
    "primary_complete_support",
    "missingness_sensitivity",
    "all_donors",
)
EXPERIMENT_DAY_COLUMNS = (
    "experiment_id",
    "day",
    "cell_count",
    "n_donors",
    "n_lines",
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
            f"Frozen endoderm benchmark mismatch for {label}: "
            f"expected {expected!r}, observed {value!r}"
        )


def validate_endoderm_benchmark_freeze_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    _require_exact(config.get("schema_name"), SCHEMA_NAME, "schema_name")
    _require_exact(config.get("schema_version"), SCHEMA_VERSION, "schema_version")
    _require_exact(config.get("freeze_id"), FREEZE_ID, "freeze_id")
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
    bindings = config.get("bindings", {})
    _require_exact(
        bindings.get("portfolio_id"),
        "trajpathmix_dataset_portfolio_v2",
        "bindings.portfolio_id",
    )
    _require_exact(
        bindings.get("candidate_id"),
        "hipsci_endoderm_125_v2",
        "bindings.candidate_id",
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
        "biological_discovery_claims": False,
    }.items():
        _require_exact(scope.get(key), expected, f"scope.{key}")
    identifiers = config.get("identifiers", {})
    _require_exact(
        tuple(identifiers.get("expected_day_order", [])),
        ("day0", "day1", "day2", "day3"),
        "identifiers.expected_day_order",
    )
    _require_exact(
        identifiers.get("primary_statistical_unit"),
        "donor_id",
        "identifiers.primary_statistical_unit",
    )
    _require_exact(
        identifiers.get("line_role"),
        "nested_within_donor",
        "identifiers.line_role",
    )
    _require_exact(
        identifiers.get("sex_role", {}).get("status"),
        "unavailable_in_authorized_cell_metadata",
        "identifiers.sex_role.status",
    )
    _require_exact(
        identifiers.get("sex_role", {}).get("action"),
        "do_not_impute_or_infer",
        "identifiers.sex_role.action",
    )
    trajectory = config.get("trajectory_preflight", {})
    _require_exact(
        trajectory.get("true_day_labels_hidden_during_construction"),
        True,
        "trajectory.true_day_labels_hidden_during_construction",
    )
    _require_exact(
        trajectory.get("deposited_trajectory_columns_forbidden"),
        True,
        "trajectory.deposited_trajectory_columns_forbidden",
    )
    _require_exact(
        trajectory.get("fixed_common_grid_bins"),
        20,
        "trajectory.fixed_common_grid_bins",
    )
    fractional = config.get("fractional_count_contract", {})
    _require_exact(
        fractional.get("integer_count_likelihood_allowed"),
        False,
        "fractional.integer_count_likelihood_allowed",
    )
    _require_exact(
        fractional.get("integer_coercion_allowed"),
        False,
        "fractional.integer_coercion_allowed",
    )
    null = config.get("randomized_empirical_null", {})
    _require_exact(null.get("assignment_unit"), "donor_id", "null.assignment_unit")
    _require_exact(
        [stage.get("replicates") for stage in null.get("stages", [])],
        [500, 2000, 10000],
        "null.stages",
    )
    _require_exact(
        config.get("claim_boundary", {}).get("pathway_scoring_authorized_at_freeze"),
        False,
        "claim_boundary.pathway_scoring_authorized_at_freeze",
    )
    result = deepcopy(dict(config))
    result["_config_payload_sha256"] = FROZEN_CONFIG_PAYLOAD_SHA256
    return result


def load_endoderm_benchmark_freeze_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Endoderm benchmark-freeze config must be a YAML mapping")
    return validate_endoderm_benchmark_freeze_config(value)


def _repository_file(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Frozen benchmark input must be repository-local") from exc
    return path


def _verify_inputs(root: Path, config: Mapping[str, Any]) -> tuple[Path, Path]:
    expected = config["input"]
    metadata_path = _repository_file(root, str(expected["relative_path"]))
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    _require_exact(
        metadata_path.stat().st_size, int(expected["size_bytes"]), "input.size"
    )
    _require_exact(
        _hash_file(metadata_path, "md5"),
        expected["publisher_md5"],
        "input.publisher_md5",
    )
    _require_exact(
        _hash_file(metadata_path, "sha256"),
        expected["local_sha256"],
        "input.local_sha256",
    )
    universe = config["pathway_universe"]
    universe_path = _repository_file(root, str(universe["relative_path"]))
    if not universe_path.is_file():
        raise FileNotFoundError(universe_path)
    _require_exact(
        _hash_file(universe_path, "sha256"),
        universe["file_sha256"],
        "pathway_universe.file_sha256",
    )
    universe_frame = pd.read_csv(universe_path, sep="\t", dtype="string")
    _require_exact(len(universe_frame), universe["n_memberships"], "universe.rows")
    _require_exact(
        universe_frame["pathway_id"].nunique(),
        universe["n_pathways"],
        "universe.pathways",
    )
    _require_exact(
        universe_frame["level_1_family_id"].nunique(),
        universe["n_level_1_families"],
        "universe.families",
    )
    _require_exact(
        sorted(universe_frame["pathway_universe_logical_sha256"].unique()),
        [universe["logical_sha256"]],
        "universe.logical_sha256",
    )
    return metadata_path, universe_path


def _cross_with_days(frame: pd.DataFrame, days: list[str]) -> pd.DataFrame:
    return frame.merge(pd.DataFrame({"day": days}), how="cross")


def _sort_day(frame: pd.DataFrame, days: list[str], keys: list[str]) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["day"] = pd.Categorical(ordered["day"], categories=days, ordered=True)
    ordered = ordered.sort_values([*keys, "day"], kind="stable").reset_index(drop=True)
    ordered["day"] = ordered["day"].astype("string")
    return ordered


def _build_artifacts(
    config: Mapping[str, Any], root: Path
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    metadata_path, universe_path = _verify_inputs(root, config)
    ids = config["identifiers"]
    days = list(ids["expected_day_order"])
    source_columns = [
        ids["donor_id_column"],
        ids["line_id_column"],
        ids["experiment_id_column"],
        ids["day_column"],
    ]
    frame = pd.read_csv(
        metadata_path, sep="\t", usecols=source_columns, dtype="string"
    ).rename(
        columns={
            ids["donor_id_column"]: "donor_id",
            ids["line_id_column"]: "line_id",
            ids["experiment_id_column"]: "experiment_id",
            ids["day_column"]: "day",
        }
    )
    if frame.isna().any().any():
        raise ValueError("Required endoderm design fields contain missing values")
    _require_exact(sorted(frame["day"].unique()), sorted(days), "day_values")
    line_to_donor = frame.groupby("line_id", observed=True)["donor_id"].nunique()
    _require_exact(line_to_donor.eq(1).all(), True, "line_nested_within_donor")

    line_experiment = frame[
        ["line_id", "donor_id", "experiment_id"]
    ].drop_duplicates()
    long_grid = _cross_with_days(line_experiment, days)
    observed_long = (
        frame.groupby(
            ["line_id", "donor_id", "experiment_id", "day"], observed=True
        )
        .size()
        .rename("cell_count")
        .reset_index()
    )
    line_experiment_day = long_grid.merge(
        observed_long,
        how="left",
        on=["line_id", "donor_id", "experiment_id", "day"],
        validate="one_to_one",
    )
    line_experiment_day["cell_count"] = (
        line_experiment_day["cell_count"].fillna(0).astype(int)
    )
    line_experiment_day = _sort_day(
        line_experiment_day, days, ["line_id", "experiment_id"]
    ).loc[:, list(LINE_EXPERIMENT_DAY_COLUMNS)]

    donor_ids = pd.DataFrame({"donor_id": sorted(frame["donor_id"].unique())})
    donor_grid = _cross_with_days(donor_ids, days)
    donor_observed = (
        frame.groupby(["donor_id", "day"], observed=True)
        .size()
        .rename("cell_count")
        .reset_index()
    )
    donor_day = donor_grid.merge(
        donor_observed,
        how="left",
        on=["donor_id", "day"],
        validate="one_to_one",
    )
    donor_day["cell_count"] = donor_day["cell_count"].fillna(0).astype(int)
    donor_line_counts = frame.groupby("donor_id", observed=True)["line_id"].nunique()
    donor_experiment_counts = frame.groupby("donor_id", observed=True)[
        "experiment_id"
    ].nunique()
    donor_day["n_lines"] = donor_day["donor_id"].map(donor_line_counts).astype(int)
    donor_day["n_experiments"] = (
        donor_day["donor_id"].map(donor_experiment_counts).astype(int)
    )
    donor_day = _sort_day(donor_day, days, ["donor_id"]).loc[
        :, list(DONOR_DAY_COLUMNS)
    ]

    donor_pivot = (
        donor_day.pivot(index="donor_id", columns="day", values="cell_count")
        .reindex(columns=days)
        .astype(int)
    )
    primary = donor_pivot.gt(10).all(axis=1)
    missingness = donor_pivot.gt(0).all(axis=1)
    donor_cohort = pd.DataFrame(
        {
            "donor_id": donor_pivot.index,
            "n_lines": donor_line_counts.reindex(donor_pivot.index).astype(int).values,
            "n_experiments": donor_experiment_counts.reindex(donor_pivot.index)
            .astype(int)
            .values,
            "day0_cells": donor_pivot["day0"].values,
            "day1_cells": donor_pivot["day1"].values,
            "day2_cells": donor_pivot["day2"].values,
            "day3_cells": donor_pivot["day3"].values,
            "minimum_day_cells": donor_pivot.min(axis=1).values,
            "primary_complete_support": primary.values,
            "missingness_sensitivity": missingness.values,
            "all_donors": True,
        },
        columns=list(DONOR_COHORT_COLUMNS),
    ).sort_values("donor_id", kind="stable", ignore_index=True)

    experiment_ids = pd.DataFrame(
        {"experiment_id": sorted(frame["experiment_id"].unique())}
    )
    experiment_grid = _cross_with_days(experiment_ids, days)
    experiment_observed = (
        frame.groupby(["experiment_id", "day"], observed=True)
        .agg(
            cell_count=("donor_id", "size"),
            n_donors=("donor_id", "nunique"),
            n_lines=("line_id", "nunique"),
        )
        .reset_index()
    )
    experiment_day = experiment_grid.merge(
        experiment_observed,
        how="left",
        on=["experiment_id", "day"],
        validate="one_to_one",
    )
    for column in ("cell_count", "n_donors", "n_lines"):
        experiment_day[column] = experiment_day[column].fillna(0).astype(int)
    experiment_day = _sort_day(experiment_day, days, ["experiment_id"]).loc[
        :, list(EXPERIMENT_DAY_COLUMNS)
    ]

    primary_donors = set(
        donor_cohort.loc[donor_cohort["primary_complete_support"], "donor_id"]
    )
    missingness_donors = set(
        donor_cohort.loc[donor_cohort["missingness_sensitivity"], "donor_id"]
    )
    multiple_line_donors = sorted(
        donor_line_counts[donor_line_counts.gt(1)].index.astype(str).tolist()
    )
    experiments_with_all_days = int(
        experiment_day.pivot(
            index="experiment_id", columns="day", values="cell_count"
        )
        .reindex(columns=days)
        .gt(0)
        .all(axis=1)
        .sum()
    )
    observed = {
        "n_cells": int(len(frame)),
        "n_donors": int(frame["donor_id"].nunique()),
        "n_lines": int(frame["line_id"].nunique()),
        "n_experiments": int(frame["experiment_id"].nunique()),
        "n_line_experiment_pairs": int(len(line_experiment)),
        "n_line_experiment_day_rows": int(len(line_experiment_day)),
        "n_zero_line_experiment_day_rows": int(
            line_experiment_day["cell_count"].eq(0).sum()
        ),
        "n_donor_day_rows": int(len(donor_day)),
        "n_zero_donor_day_rows": int(donor_day["cell_count"].eq(0).sum()),
        "n_donors_primary_complete_support": int(len(primary_donors)),
        "n_lines_nested_in_primary_complete_support_donors": int(
            frame.loc[frame["donor_id"].isin(primary_donors), "line_id"].nunique()
        ),
        "n_donors_missingness_sensitivity": int(len(missingness_donors)),
        "n_lines_nested_in_missingness_sensitivity_donors": int(
            frame.loc[frame["donor_id"].isin(missingness_donors), "line_id"].nunique()
        ),
        "n_donors_with_multiple_lines": int(len(multiple_line_donors)),
        "multiple_line_donor_ids": multiple_line_donors,
        "n_donors_in_multiple_experiments": int(
            donor_experiment_counts.gt(1).sum()
        ),
        "n_experiments_with_all_four_days": experiments_with_all_days,
    }
    _require_exact(
        observed, config["expected_observed_structure"], "observed_structure"
    )
    _require_exact(
        int(line_experiment_day["cell_count"].sum()), len(frame), "long_count_sum"
    )
    _require_exact(int(donor_day["cell_count"].sum()), len(frame), "donor_count_sum")

    audit = {
        "schema_name": "trajpathmix_endoderm_statistical_unit_audit",
        "schema_version": "1.0.0",
        "freeze_id": FREEZE_ID,
        "config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
        "candidate_id": config["bindings"]["candidate_id"],
        "observed_structure": observed,
        "primary_statistical_unit": "donor_id",
        "line_role": "nested_within_donor",
        "experiment_role": deepcopy(ids["experiment_role"]),
        "sex_role": deepcopy(ids["sex_role"]),
        "primary_complete_support_definition": deepcopy(
            config["cell_count_support"]["primary_complete_support"]
        ),
        "missingness_sensitivity_definition": deepcopy(
            config["cell_count_support"]["missingness_sensitivity"]
        ),
        "primary_complete_support_independent_donors": int(len(primary_donors)),
        "missingness_sensitivity_independent_donors": int(len(missingness_donors)),
        "missingness_sensitivity_nested_lines": int(
            observed["n_lines_nested_in_missingness_sensitivity_donors"]
        ),
        "line_and_donor_denominators_not_interchangeable": True,
        "metadata_only": True,
        "expression_matrix_read": False,
        "pathway_outcomes_read": False,
        "pathway_scoring_authorized": False,
        "audit_status": "pass_donor_is_primary_repeat",
    }
    contract = {
        "schema_name": "trajpathmix_endoderm_benchmark_contract_freeze",
        "schema_version": "1.0.0",
        "freeze_id": FREEZE_ID,
        "frozen_at_utc": config["frozen_at_utc"],
        "config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
        "bindings": deepcopy(config["bindings"]),
        "statistical_units": {
            "primary": "donor_id",
            "line": "nested_within_donor",
            "experiment": deepcopy(ids["experiment_role"]),
            "sex": deepcopy(ids["sex_role"]),
        },
        "cohorts": {
            "primary_complete_support_independent_donors": int(
                len(primary_donors)
            ),
            "missingness_sensitivity_independent_donors": int(
                len(missingness_donors)
            ),
            "all_independent_donors": int(frame["donor_id"].nunique()),
            "cell_count_support": deepcopy(config["cell_count_support"]),
        },
        "trajectory_preflight": deepcopy(config["trajectory_preflight"]),
        "fractional_count_contract": deepcopy(config["fractional_count_contract"]),
        "pathway_universe": deepcopy(config["pathway_universe"]),
        "randomized_empirical_null": deepcopy(
            config["randomized_empirical_null"]
        ),
        "semi_synthetic_events": deepcopy(config["semi_synthetic_events"]),
        "baselines": deepcopy(config["baselines"]),
        "downsampling": deepcopy(config["downsampling"]),
        "acceptance_policy": deepcopy(config["acceptance_policy"]),
        "claim_boundary": deepcopy(config["claim_boundary"]),
        "metadata_input_sha256": config["input"]["local_sha256"],
        "pathway_universe_file_sha256": _hash_file(universe_path, "sha256"),
        "pathway_outcomes_read": False,
        "pathway_scoring_authorized": False,
        "next_gate": "raw_archive_byte_verification_then_fractional_schema_and_blinded_trajectory_preflight",
    }
    return {
        LINE_EXPERIMENT_DAY_FILE: line_experiment_day,
        DONOR_DAY_FILE: donor_day,
        DONOR_COHORT_FILE: donor_cohort,
        EXPERIMENT_DAY_FILE: experiment_day,
    }, {
        STATISTICAL_UNIT_AUDIT_FILE: audit,
        BENCHMARK_CONTRACT_FILE: contract,
    }


def _write_table(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, sep="\t", index=False, lineterminator="\n")


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _build_record(
    *,
    config_file: Path,
    tables: Mapping[str, pd.DataFrame],
    json_artifacts: Mapping[str, Mapping[str, Any]],
    artifact_dir: Path,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for filename, table in tables.items():
        artifacts[filename] = {
            "sha256": _hash_file(artifact_dir / filename, "sha256"),
            "rows": int(len(table)),
            "columns": list(table.columns),
        }
    for filename, value in json_artifacts.items():
        artifacts[filename] = {
            "sha256": _hash_file(artifact_dir / filename, "sha256"),
            "schema_name": value["schema_name"],
        }
    return {
        "schema_name": "trajpathmix_endoderm_benchmark_freeze_build_record",
        "schema_version": "1.0.0",
        "freeze_id": FREEZE_ID,
        "config_file": "config/trajpathmix_endoderm_benchmark_freeze_v1.yaml",
        "config_file_sha256": _hash_file(config_file, "sha256"),
        "config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
        "implementation_file": "pyfgsea/trajpathmix_endoderm_benchmark_freeze.py",
        "implementation_sha256": _hash_file(Path(__file__).resolve(), "sha256"),
        "artifacts": artifacts,
        "metadata_only": True,
        "expression_matrix_read": False,
        "pathway_outcomes_read": False,
        "pathway_scoring_authorized": False,
        "evidence_revision_mode": "append_only",
    }


def build_and_write_endoderm_benchmark_freeze(
    *,
    config_path: str | Path,
    repository_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config = load_endoderm_benchmark_freeze_config(config_file)
    root = Path(repository_root).resolve()
    tables, json_artifacts = _build_artifacts(config, root)
    output = Path(output_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Endoderm benchmark-freeze output exists: {output}")
    lock_path = output.parent / f".{output.name}.create.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FileExistsError(f"Endoderm benchmark freeze is locked: {lock_path}") from exc
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
        record = _build_record(
            config_file=config_file,
            tables=tables,
            json_artifacts=json_artifacts,
            artifact_dir=temporary,
        )
        _write_json(record, temporary / BUILD_RECORD_FILE)
        os.rename(temporary, output)
        temporary = None
        result = dict(record)
        result["output_dir"] = str(output)
        result["build_record_sha256"] = _hash_file(
            output / BUILD_RECORD_FILE, "sha256"
        )
        return result
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
        lock_path.unlink(missing_ok=True)


def validate_endoderm_benchmark_freeze_output(
    *,
    config_path: str | Path,
    repository_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config = load_endoderm_benchmark_freeze_config(config_file)
    root = Path(repository_root).resolve()
    expected_tables, expected_json = _build_artifacts(config, root)
    output = Path(output_dir).resolve()
    if not output.is_dir():
        raise FileNotFoundError(output)
    expected_names = {*expected_tables, *expected_json, BUILD_RECORD_FILE}
    _require_exact(
        {path.name for path in output.iterdir() if path.is_file()},
        expected_names,
        "output_file_set",
    )
    for filename, expected in expected_tables.items():
        observed = pd.read_csv(output / filename, sep="\t", keep_default_na=False)
        if not observed.fillna("").astype(str).equals(
            expected.fillna("").astype(str)
        ):
            raise ValueError(f"Endoderm benchmark table mismatch: {filename}")
    for filename, expected in expected_json.items():
        observed = json.loads((output / filename).read_text(encoding="utf-8"))
        _require_exact(observed, expected, filename)
    expected_record = _build_record(
        config_file=config_file,
        tables=expected_tables,
        json_artifacts=expected_json,
        artifact_dir=output,
    )
    record = json.loads((output / BUILD_RECORD_FILE).read_text(encoding="utf-8"))
    _require_exact(record, expected_record, "build_record")
    result = dict(record)
    result["output_dir"] = str(output)
    result["build_record_sha256"] = _hash_file(
        output / BUILD_RECORD_FILE, "sha256"
    )
    result["validation_status"] = "pass_donor_level_benchmark_freeze"
    return result


__all__ = [
    "BENCHMARK_CONTRACT_FILE",
    "BUILD_RECORD_FILE",
    "DONOR_COHORT_FILE",
    "DONOR_DAY_FILE",
    "EXPERIMENT_DAY_FILE",
    "FREEZE_ID",
    "FROZEN_CONFIG_PAYLOAD_SHA256",
    "LINE_EXPERIMENT_DAY_FILE",
    "STATISTICAL_UNIT_AUDIT_FILE",
    "build_and_write_endoderm_benchmark_freeze",
    "load_endoderm_benchmark_freeze_config",
    "validate_endoderm_benchmark_freeze_config",
    "validate_endoderm_benchmark_freeze_output",
]
