from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .trajpathmix_corebench_cb2_implementation_freeze_v1 import (
    BUILD_RECORD_FILE as IMPLEMENTATION_BUILD_RECORD_FILE,
    FREEZE_FILE as IMPLEMENTATION_FREEZE_FILE,
    FROZEN_CONFIG_PAYLOAD_SHA256,
    MATERIAL_PASSPORT_FILE as IMPLEMENTATION_PASSPORT_FILE,
    load_cb2_implementation_config,
    validate_cb2_implementation_freeze_output,
    validate_frozen_assignment_manifest,
    verify_cb2_implementation_bindings,
)


SCHEMA_VERSION = "2.0.0"
READINESS_ID = "trajpathmix_corebench_cb2a_implementation_readiness_v2"
CONFIG_FILE = "config/trajpathmix_corebench_cb2_implementation_freeze_v1.yaml"
IMPLEMENTATION_FILE = "pyfgsea/trajpathmix_corebench_cb2a_readiness_v2.py"

DECISION_FILE = "CB2A_IMPLEMENTATION_READINESS_DECISION_v2.json"
ASSIGNMENT_AUDIT_FILE = "cb2a_v2_assignment_integrity_audit.tsv"
BUILD_RECORD_FILE = "cb2a_implementation_readiness_build_record_v2.json"

RANK_RELATIVE_TOLERANCE = 1.0e-10
ASSIGNMENT_COLUMNS = (
    "assignment_id",
    "assignment_sha256",
    "donor_id",
    "pseudo_condition",
    "pseudo_case",
)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _assignment_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.uint8))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise ValueError(
            f"CB2a-v2 mismatch for {label}: expected {expected!r}, "
            f"observed {observed!r}"
        )


def _repo_file(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("CB2a-v2 bindings must be repository-local") from exc
    return path


def _strict_json_load(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"Non-finite JSON constant {value!r} in {path}")
        ),
    )


def _assert_json_finite(value: Any, label: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_json_finite(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_json_finite(item, f"{label}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(f"Non-finite JSON value at {label}")


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    _assert_json_finite(value)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _table_text(frame: pd.DataFrame) -> str:
    output = io.StringIO()
    frame.to_csv(
        output,
        sep="\t",
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    return output.getvalue()


def _read_bool(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    _require(
        bool(normalized.isin({"true", "false"}).all()),
        f"Column {series.name!r} contains non-boolean values",
    )
    return normalized.eq("true")


def _factorial_orbit(groups: Sequence[Sequence[int]]) -> int:
    result = 1
    for group in groups:
        result *= math.factorial(len(group))
    return int(result)


def _svd_rank(values: np.ndarray) -> int:
    singular = np.linalg.svd(np.asarray(values, dtype=float), compute_uv=False)
    if not len(singular) or singular[0] == 0:
        return 0
    return int(np.sum(singular > singular[0] * RANK_RELATIVE_TOLERANCE))


def _partition(keys: Sequence[str]) -> list[np.ndarray]:
    lookup: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        lookup[str(key)].append(index)
    return [np.asarray(lookup[key], dtype=int) for key in sorted(lookup)]


def _load_assignment_matrix(
    root: Path, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, tuple[str, ...], np.ndarray, tuple[str, ...]]:
    binding = config["bindings"]["frozen_assignment_manifest_v1"]
    path = _repo_file(root, str(binding["relative_path"]))
    validate_frozen_assignment_manifest(path, config)
    frame = pd.read_csv(path, sep="\t", dtype="string")
    _require_equal(tuple(frame.columns), ASSIGNMENT_COLUMNS, "assignment columns")
    frame["pseudo_case_bool"] = _read_bool(frame["pseudo_case"])
    assignment_ids = tuple(frame["assignment_id"].drop_duplicates().astype(str))
    first = frame.loc[frame["assignment_id"].eq(assignment_ids[0])]
    donors = tuple(first["donor_id"].astype(str))
    matrix = np.empty((len(assignment_ids), len(donors)), dtype=np.uint8)
    for row_index, assignment_id in enumerate(assignment_ids):
        group = frame.loc[frame["assignment_id"].eq(assignment_id)]
        _require_equal(tuple(group["donor_id"].astype(str)), donors, f"{assignment_id} donors")
        matrix[row_index] = group["pseudo_case_bool"].to_numpy(dtype=np.uint8)
    hashes = tuple(
        frame.groupby("assignment_id", sort=False)["assignment_sha256"].first().astype(str)
    )
    _require_equal(
        tuple(_assignment_hash(row) for row in matrix), hashes, "assignment hashes"
    )
    return frame, donors, matrix, assignment_ids


def _load_structural_inputs(
    root: Path,
    config: Mapping[str, Any],
    donors: tuple[str, ...],
) -> dict[str, Any]:
    bindings = config["bindings"]
    counts = pd.read_csv(
        _repo_file(root, bindings["donor_experiment_counts_v1"]["relative_path"]),
        sep="\t",
        dtype={"donor_id": "string", "experiment_id": "string"},
    )
    experiments = tuple(sorted(counts["experiment_id"].astype(str).unique()))
    _require_equal(len(experiments), 28, "frozen experiment universe")
    donor_index = {donor: index for index, donor in enumerate(donors)}
    experiment_index = {
        experiment: index for index, experiment in enumerate(experiments)
    }
    incidence = np.zeros((len(donors), len(experiments)), dtype=np.uint8)
    pairs = counts.loc[counts["donor_id"].astype(str).isin(donors), ["donor_id", "experiment_id"]].drop_duplicates()
    for row in pairs.itertuples(index=False):
        incidence[donor_index[str(row.donor_id)], experiment_index[str(row.experiment_id)]] = 1

    cohort = pd.read_csv(
        _repo_file(root, bindings["donor_cohort_v1"]["relative_path"]),
        sep="\t",
        dtype="string",
    )
    cohort["primary_complete_support"] = _read_bool(cohort["primary_complete_support"])
    primary = tuple(
        sorted(cohort.loc[cohort["primary_complete_support"], "donor_id"].astype(str))
    )
    _require_equal(primary, donors, "frozen primary donor cohort")

    availability_long = pd.read_csv(
        _repo_file(root, bindings["donor_bin_availability_v2"]["relative_path"]),
        sep="\t",
        dtype={"donor_id": "string"},
    )
    availability_long["available_bool"] = _read_bool(availability_long["available"])
    local = availability_long.loc[
        availability_long["donor_id"].astype(str).isin(donors)
    ].copy()
    _require_equal(len(local), len(donors) * 20, "donor-bin availability rows")
    cell_count = (
        local.pivot(index="donor_id", columns="bin_id", values="cell_count")
        .reindex(index=donors, columns=range(20))
        .to_numpy(dtype=int)
    )
    availability = (
        local.pivot(index="donor_id", columns="bin_id", values="available_bool")
        .reindex(index=donors, columns=range(20))
        .to_numpy(dtype=bool)
    )
    _require(
        bool(np.array_equal(availability, cell_count >= 5)),
        "Availability differs from the frozen >=5-cell rule",
    )

    coordinate = pd.read_csv(
        _repo_file(root, bindings["fixed_cb1_coordinate"]["relative_path"]),
        sep="\t",
        usecols=["donor_id", "experiment_id", "corebench_coordinate"],
        dtype={"donor_id": "string", "experiment_id": "string"},
    )
    coordinate = coordinate.loc[
        coordinate["donor_id"].astype(str).isin(donors)
    ].copy()
    values = coordinate["corebench_coordinate"].to_numpy(dtype=float)
    _require(
        bool(np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()),
        "Frozen coordinate is invalid",
    )
    coordinate["bin_id"] = np.minimum(19, np.floor(values * 20.0).astype(int))
    grouped = (
        coordinate.groupby(["donor_id", "bin_id", "experiment_id"], sort=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    cube = np.zeros((len(donors), 20, len(experiments)), dtype=float)
    for row in grouped.itertuples(index=False):
        cube[
            donor_index[str(row.donor_id)],
            int(row.bin_id),
            experiment_index[str(row.experiment_id)],
        ] = int(row.n_cells)
    _require(
        bool(np.array_equal(cube.sum(axis=2).astype(int), cell_count)),
        "Coordinate-derived counts differ from frozen availability counts",
    )
    experiment_fraction = np.divide(
        cube,
        cube.sum(axis=2, keepdims=True),
        out=np.zeros_like(cube),
        where=cube.sum(axis=2, keepdims=True) > 0,
    )
    signatures = tuple(
        "".join("1" if value else "0" for value in row) for row in availability
    )
    return {
        "experiments": experiments,
        "incidence": incidence,
        "availability": availability,
        "cell_count": cell_count,
        "experiment_fraction": experiment_fraction,
        "signatures": signatures,
    }


def _audit_structure(
    root: Path, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _, donors, assignments, assignment_ids = _load_assignment_matrix(root, config)
    inputs = _load_structural_inputs(root, config, donors)
    incidence = inputs["incidence"]
    experiment_totals = incidence.sum(axis=0).astype(int)
    case_by_experiment = assignments @ incidence
    control_by_experiment = experiment_totals[None, :] - case_by_experiment
    experiment_imbalance = np.abs(case_by_experiment - control_by_experiment)
    lower = np.floor(experiment_totals / 2).astype(int)
    upper = np.ceil(experiment_totals / 2).astype(int)
    experiment_balance = np.all(
        (case_by_experiment >= lower[None, :])
        & (case_by_experiment <= upper[None, :]),
        axis=1,
    )

    availability = inputs["availability"]
    fractions = inputs["experiment_fraction"]
    groups = _partition(inputs["signatures"])
    n_assignments = len(assignments)
    minimum_df = np.full(n_assignments, np.iinfo(np.int32).max, dtype=int)
    maximum_vif = np.zeros(n_assignments, dtype=float)
    minimum_information = np.full(n_assignments, np.inf, dtype=float)
    minimum_information_fraction = np.full(n_assignments, np.inf, dtype=float)
    minimum_case = np.full(n_assignments, np.iinfo(np.int32).max, dtype=int)
    minimum_control = np.full(n_assignments, np.iinfo(np.int32).max, dtype=int)
    all_estimable = np.ones(n_assignments, dtype=bool)

    for bin_id in range(20):
        indices = np.flatnonzero(availability[:, bin_id])
        local_fractions = fractions[indices, bin_id, :]
        active_columns = np.any(np.abs(local_fractions) > 1.0e-12, axis=0)
        z = local_fractions[:, active_columns]
        rank_reduced = _svd_rank(z)
        _require_equal(
            rank_reduced,
            z.shape[1],
            f"bin {bin_id} reduced nuisance design rank",
        )
        c = assignments[:, indices].T.astype(float)
        q, _ = np.linalg.qr(z, mode="reduced")
        u = c - q @ (q.T @ c)
        information = np.sum(u**2, axis=0)
        cases = np.sum(c, axis=0).astype(int)
        controls = len(indices) - cases
        unadjusted = cases * controls / len(indices)
        vif = np.divide(
            unadjusted,
            information,
            out=np.full(n_assignments, np.inf),
            where=information > 1e-14 * np.maximum(1.0, unadjusted),
        )
        information_fraction = np.divide(
            information,
            unadjusted,
            out=np.zeros(n_assignments),
            where=unadjusted > 0,
        )
        label_information = np.zeros(n_assignments, dtype=float)
        for group in groups:
            if not availability[int(group[0]), bin_id] or len(group) <= 1:
                continue
            local_case = assignments[:, group].sum(axis=1).astype(float)
            label_information += local_case * (len(group) - local_case) / len(group)
        residual_df = len(indices) - rank_reduced - 1
        local_pass = (
            (cases >= 10)
            & (controls >= 10)
            & (information > 1e-14 * np.maximum(1.0, unadjusted))
            & (residual_df >= 3)
            & np.isfinite(vif)
            & (vif <= 10.0)
            & (label_information > 1e-14)
        )
        minimum_df = np.minimum(minimum_df, residual_df)
        maximum_vif = np.maximum(maximum_vif, vif)
        minimum_information = np.minimum(minimum_information, information)
        minimum_information_fraction = np.minimum(
            minimum_information_fraction, information_fraction
        )
        minimum_case = np.minimum(minimum_case, cases)
        minimum_control = np.minimum(minimum_control, controls)
        all_estimable &= local_pass

    hashes = tuple(_assignment_hash(row) for row in assignments)
    donor_case_fraction = assignments.mean(axis=0)
    audit = pd.DataFrame(
        {
            "assignment_id": assignment_ids,
            "assignment_sha256": hashes,
            "n_pseudo_case": assignments.sum(axis=1).astype(int),
            "n_pseudo_control": (len(donors) - assignments.sum(axis=1)).astype(int),
            "maximum_absolute_experiment_donor_count_imbalance": experiment_imbalance.max(axis=1).astype(int),
            "all_experiment_floor_ceil_constraints_pass": experiment_balance,
            "minimum_case_donors_in_any_bin": minimum_case,
            "minimum_control_donors_in_any_bin": minimum_control,
            "minimum_full_residual_df": minimum_df,
            "maximum_condition_vif": maximum_vif,
            "minimum_condition_information": minimum_information,
            "minimum_condition_information_fraction": minimum_information_fraction,
            "all_20_bins_estimable": all_estimable,
        }
    )
    signature_sizes = [len(group) for group in groups]
    mobile = sum(size for size in signature_sizes if size > 1)
    orbit = _factorial_orbit(groups)
    active_experiments = int(np.sum(experiment_totals > 0))
    nuisance_mobile_blocks_evaluated = 0
    nuisance_mobile_blocks_varying = 0
    for group in groups:
        if len(group) <= 1:
            continue
        for bin_id in range(20):
            if not availability[int(group[0]), bin_id]:
                continue
            nuisance_mobile_blocks_evaluated += 1
            local = fractions[group, bin_id, :]
            nuisance_mobile_blocks_varying += int(
                not bool(np.allclose(local, local[0], rtol=0.0, atol=1.0e-12))
            )
    summary = {
        "assignment_source": "read_and_hash_validate_bound_v1_tsv_only",
        "assignment_generator_imported_or_called": False,
        "n_assignments": int(len(assignments)),
        "n_unique_assignment_hashes": int(len(set(hashes))),
        "n_donors": int(len(donors)),
        "n_label_immobile_donors": int(
            np.sum((donor_case_fraction == 0) | (donor_case_fraction == 1))
        ),
        "minimum_donor_pseudo_case_fraction": float(donor_case_fraction.min()),
        "maximum_donor_pseudo_case_fraction": float(donor_case_fraction.max()),
        "n_frozen_experiments": int(len(inputs["experiments"])),
        "n_experiments_represented_in_primary": active_experiments,
        "absent_primary_experiments": [
            inputs["experiments"][index]
            for index in np.flatnonzero(experiment_totals == 0)
        ],
        "maximum_absolute_experiment_donor_count_imbalance": int(
            experiment_imbalance.max()
        ),
        "all_experiment_floor_ceil_constraints_pass": bool(experiment_balance.all()),
        "n_unique_availability_signatures": int(len(groups)),
        "n_mobile_donors_under_full_signature_mapping": int(mobile),
        "n_immobile_donors_under_full_signature_mapping": int(len(donors) - mobile),
        "residual_mapping_orbit_size": int(orbit),
        "n_unique_nonidentity_mappings_possible": int(orbit - 1),
        "nuisance_mobile_block_bins_with_varying_design": int(
            nuisance_mobile_blocks_varying
        ),
        "nuisance_mobile_block_bins_evaluated": int(
            nuisance_mobile_blocks_evaluated
        ),
        "all_500_assignments_all_20_bins_estimable": bool(all_estimable.all()),
        "minimum_case_donors_in_any_bin": int(minimum_case.min()),
        "minimum_control_donors_in_any_bin": int(minimum_control.min()),
        "minimum_full_residual_df": int(minimum_df.min()),
        "maximum_condition_vif": float(maximum_vif.max()),
        "minimum_condition_information": float(minimum_information.min()),
        "minimum_condition_information_fraction": float(
            minimum_information_fraction.min()
        ),
        "residual_mappings_per_replicate": 999,
        "sampled_reference_resolution": 0.001,
    }
    return audit, summary


def _compare_expected_metrics(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> tuple[bool, list[str]]:
    tolerance = float(expected["float_comparison_absolute_tolerance"])
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        if key == "float_comparison_absolute_tolerance":
            continue
        if key not in observed:
            mismatches.append(f"missing:{key}")
            continue
        observed_value = observed[key]
        if isinstance(expected_value, float):
            if not math.isclose(
                float(observed_value),
                expected_value,
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                mismatches.append(
                    f"{key}:expected={expected_value!r}:observed={observed_value!r}"
                )
        elif observed_value != expected_value:
            mismatches.append(
                f"{key}:expected={expected_value!r}:observed={observed_value!r}"
            )
    return not mismatches, mismatches


def _build_artifacts(
    *, root: Path, config: Mapping[str, Any], implementation_freeze_dir: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    freeze_validation = validate_cb2_implementation_freeze_output(
        config_path=root / CONFIG_FILE,
        repository_root=root,
        output_dir=implementation_freeze_dir,
    )
    verify_cb2_implementation_bindings(root, config)
    v1_decision_path = _repo_file(
        root, config["bindings"]["cb2a_v1_decision"]["relative_path"]
    )
    v1_decision = _strict_json_load(v1_decision_path)
    _require_equal(v1_decision["design_precheck_pass"], True, "v1 design pass")
    _require_equal(
        v1_decision["implementation_readiness_pass"], False, "v1 implementation fail"
    )
    _require_equal(v1_decision["cb2_500_start_allowed"], False, "v1 CB2 stop")
    assignment_audit, structural = _audit_structure(root, config)
    structural_pass, structural_mismatches = _compare_expected_metrics(
        structural, config["expected_cb2a_v2_structural_metrics"]
    )
    freeze_payload = _strict_json_load(
        implementation_freeze_dir / IMPLEMENTATION_FREEZE_FILE
    )
    synthetic = freeze_payload["synthetic_capability_evidence"]
    required_checks = set(
        config["synthetic_capability_test_contract"]["required_check_ids"]
    )
    observed_checks = {str(row["check_id"]): bool(row["pass"]) for row in synthetic["checks"]}
    missing_or_failed = sorted(
        check_id for check_id in required_checks if not observed_checks.get(check_id, False)
    )
    synthetic_pass = bool(
        synthetic["all_required_checks_pass"] and not missing_or_failed
    )
    implementation_pass = bool(
        freeze_validation["implementation_capability_pass"]
        and freeze_payload["implementation_capability_pass"]
        and synthetic_pass
    )
    all_pass = bool(structural_pass and implementation_pass)
    blocking = []
    if not structural_pass:
        blocking.append("frozen_assignment_or_design_metrics_mismatch")
    blocking.extend(f"synthetic_capability_failed:{item}" for item in missing_or_failed)
    decision = {
        "schema_name": "trajpathmix_corebench_cb2a_implementation_readiness_decision",
        "schema_version": SCHEMA_VERSION,
        "readiness_id": READINESS_ID,
        "implementation_config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
        "implementation_freeze_build_record_sha256": _hash_file(
            implementation_freeze_dir / IMPLEMENTATION_BUILD_RECORD_FILE
        ),
        "implementation_freeze_sha256": _hash_file(
            implementation_freeze_dir / IMPLEMENTATION_FREEZE_FILE
        ),
        "implementation_material_passport_sha256": _hash_file(
            implementation_freeze_dir / IMPLEMENTATION_PASSPORT_FILE
        ),
        "preserved_cb2a_v1_decision_sha256": _hash_file(v1_decision_path),
        "frozen_assignment_manifest_sha256": config["bindings"][
            "frozen_assignment_manifest_v1"
        ]["sha256"],
        "permutation_reference": config["permutation_reference"],
        "material_passport": {
            **config["material_passport"],
            "verification_status": "VERIFIED" if all_pass else "FAIL_CLOSED",
            "validation_stage": "CB2a_v2_implementation_and_design_readiness",
            "empirical_null_replicates_run": 0,
        },
        "fallacy_scan": {
            **config["material_passport"]["fallacy_scan"],
            "verification_coverage": "11_of_11",
            "status": "pass_design_claim_ceiling_scan",
        },
        "claim_ceiling": {
            "complete_null_weak_fwer_only": True,
            "strong_fwer_under_partial_null_claimed": False,
            "partial_null_fdr_claimed": False,
            "alternative_simultaneous_coverage_claimed": False,
            "pathway_recovery_claimed": False,
            "timing_recovery_claimed": False,
            "empirical_null_calibration_performed": False,
        },
        "v1_evidence_preservation": {
            "v1_design_precheck_pass": True,
            "v1_implementation_readiness_pass": False,
            "v1_cb2_500_start_allowed": False,
            "v1_artifacts_modified": False,
        },
        "assignment_and_design_structural_recheck": structural,
        "expected_structural_metrics_match": structural_pass,
        "structural_metric_mismatches": structural_mismatches,
        "synthetic_capability_evidence": synthetic,
        "missing_or_failed_synthetic_checks": missing_or_failed,
        "implementation_readiness_pass": implementation_pass,
        "design_readiness_pass": structural_pass,
        "cb2a_pass": all_pass,
        "cb2a_v2_pass": all_pass,
        "cb2_500_technical_gate_pass": all_pass,
        "cb2_500_start_allowed": all_pass,
        "cb2_500_start_allowed_scope": (
            "technical_readiness_only_requires_separate_explicit_execution_authorization"
        ),
        "separate_user_execution_authorization_required": True,
        "pathway_scoring_authorized_by_this_artifact": False,
        "decision": (
            "pass_cb2a_v2_technical_readiness_pending_separate_cb2_500_authorization"
            if all_pass
            else "fail_closed_before_cb2_500"
        ),
        "blocking_reason_codes": blocking,
        "next_gate": (
            "separate_explicit_cb2_500_execution_authorization"
            if all_pass
            else "theory_driven_append_only_implementation_revision"
        ),
        "assignment_generator_imported_or_called": False,
        "expression_values_read": False,
        "pathway_outcomes_read": False,
        "pathway_scoring_performed": False,
        "real_condition_contrast_read_or_generated": False,
        "timing_computed": False,
        "timing_fields_present": False,
        "timing_fields_output": False,
        "cb2_500_run": False,
    }
    return assignment_audit, decision


def _build_record(
    *, config_path: Path, implementation_freeze_dir: Path, artifact_dir: Path
) -> dict[str, Any]:
    names = (ASSIGNMENT_AUDIT_FILE, DECISION_FILE)
    return {
        "schema_name": "trajpathmix_corebench_cb2a_implementation_readiness_build_record",
        "schema_version": SCHEMA_VERSION,
        "readiness_id": READINESS_ID,
        "config_file": CONFIG_FILE,
        "config_file_sha256": _hash_file(config_path),
        "config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
        "implementation_file": IMPLEMENTATION_FILE,
        "implementation_sha256": _hash_file(Path(__file__).resolve()),
        "implementation_freeze_build_record_sha256": _hash_file(
            implementation_freeze_dir / IMPLEMENTATION_BUILD_RECORD_FILE
        ),
        "artifacts": {
            name: {
                "sha256": _hash_file(artifact_dir / name),
                "bytes": int((artifact_dir / name).stat().st_size),
            }
            for name in names
        },
        "assignment_source_mode": "read_and_hash_validate_original_tsv_no_generator",
        "assignment_generator_imported_or_called": False,
        "expression_values_read": False,
        "pathway_scoring_performed": False,
        "real_condition_contrast_read_or_generated": False,
        "timing_computed": False,
        "timing_fields_present": False,
        "cb2_500_run": False,
        "evidence_revision_mode": "create_only_append_only",
    }


def build_and_write_cb2a_readiness_v2(
    *,
    config_path: str | Path,
    repository_root: str | Path,
    implementation_freeze_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    root = Path(repository_root).resolve()
    freeze_dir = Path(implementation_freeze_dir).resolve()
    output = Path(output_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"CB2a-v2 output exists: {output}")
    lock = output.parent / f".{output.name}.create.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FileExistsError(f"CB2a-v2 output is locked: {lock}") from exc
    temporary: Path | None = None
    try:
        os.close(descriptor)
        config = load_cb2_implementation_config(config_file)
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
        audit, decision = _build_artifacts(
            root=root,
            config=config,
            implementation_freeze_dir=freeze_dir,
        )
        (temporary / ASSIGNMENT_AUDIT_FILE).write_text(
            _table_text(audit), encoding="utf-8"
        )
        _write_json(decision, temporary / DECISION_FILE)
        record = _build_record(
            config_path=config_file,
            implementation_freeze_dir=freeze_dir,
            artifact_dir=temporary,
        )
        _write_json(record, temporary / BUILD_RECORD_FILE)
        os.rename(temporary, output)
        temporary = None
        return {
            **record,
            "output_dir": str(output),
            "build_record_sha256": _hash_file(output / BUILD_RECORD_FILE),
            "cb2a_v2_pass": bool(decision["cb2a_v2_pass"]),
            "decision": decision["decision"],
        }
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
        lock.unlink(missing_ok=True)


def validate_cb2a_readiness_v2_output(
    *,
    config_path: str | Path,
    repository_root: str | Path,
    implementation_freeze_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    root = Path(repository_root).resolve()
    freeze_dir = Path(implementation_freeze_dir).resolve()
    output = Path(output_dir).resolve()
    if not output.is_dir():
        raise FileNotFoundError(output)
    expected_names = {ASSIGNMENT_AUDIT_FILE, DECISION_FILE, BUILD_RECORD_FILE}
    _require_equal(
        {path.name for path in output.iterdir()},
        expected_names,
        "CB2a-v2 output file set",
    )
    config = load_cb2_implementation_config(config_file)
    audit, decision = _build_artifacts(
        root=root,
        config=config,
        implementation_freeze_dir=freeze_dir,
    )
    _require_equal(
        (output / ASSIGNMENT_AUDIT_FILE).read_text(encoding="utf-8"),
        _table_text(audit),
        ASSIGNMENT_AUDIT_FILE,
    )
    _require_equal(_strict_json_load(output / DECISION_FILE), decision, DECISION_FILE)
    record = _build_record(
        config_path=config_file,
        implementation_freeze_dir=freeze_dir,
        artifact_dir=output,
    )
    _require_equal(_strict_json_load(output / BUILD_RECORD_FILE), record, BUILD_RECORD_FILE)
    return {
        **record,
        "output_dir": str(output),
        "build_record_sha256": _hash_file(output / BUILD_RECORD_FILE),
        "cb2a_v2_pass": bool(decision["cb2a_v2_pass"]),
        "decision": decision["decision"],
        "validation_status": "pass_cb2a_v2_readiness_artifact_integrity",
    }


__all__ = [
    "ASSIGNMENT_AUDIT_FILE",
    "BUILD_RECORD_FILE",
    "DECISION_FILE",
    "READINESS_ID",
    "build_and_write_cb2a_readiness_v2",
    "validate_cb2a_readiness_v2_output",
]
