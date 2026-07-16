from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from jsonschema import Draft202012Validator


SCHEMA_NAME = "trajpathmix_corebench_cb2_implementation_freeze"
SCHEMA_VERSION = "1.0.0"
FREEZE_ID = "trajpathmix_corebench_cb2_implementation_freeze_v1"
FROZEN_CONFIG_PAYLOAD_SHA256 = "4699bba0893d5bf727a34ae90d14f46b4dd3c4f03f7a89e944536bc7f8af2f51"

CONFIG_FILE = "config/trajpathmix_corebench_cb2_implementation_freeze_v1.yaml"
IMPLEMENTATION_FILE = "pyfgsea/trajpathmix_corebench_cb2_implementation_freeze_v1.py"
KERNEL_MODULE = "pyfgsea.trajpathmix_functional_core_v1"
STRICT_SCHEMA_FILE = "schemas/trajpathmix_functional_core_result_v1.schema.json"

FREEZE_FILE = "corebench_cb2_implementation_freeze_v1.json"
MATERIAL_PASSPORT_FILE = "corebench_cb2_implementation_material_passport_v1.json"
BUILD_RECORD_FILE = "corebench_cb2_implementation_freeze_build_record_v1.json"

ASSIGNMENT_COLUMNS = (
    "assignment_id",
    "assignment_sha256",
    "donor_id",
    "pseudo_condition",
    "pseudo_case",
)


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


def _hash_array(values: np.ndarray, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise ValueError(
            f"Frozen CB2 implementation mismatch for {label}: expected "
            f"{expected!r}, observed {observed!r}"
        )


def _repo_file(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("CB2 implementation bindings must be repository-local") from exc
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


def _read_bool(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    _require(
        bool(normalized.isin({"true", "false"}).all()),
        f"Column {series.name!r} contains non-boolean values",
    )
    return normalized.eq("true")


def _assignment_hash(values: np.ndarray) -> str:
    return _hash_array(values, "uint8")


def validate_cb2_implementation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    _require_equal(config.get("schema_name"), SCHEMA_NAME, "schema_name")
    _require_equal(config.get("schema_version"), SCHEMA_VERSION, "schema_version")
    _require_equal(config.get("freeze_id"), FREEZE_ID, "freeze_id")
    _require_equal(
        config.get("frozen_payload_sha256"),
        FROZEN_CONFIG_PAYLOAD_SHA256,
        "frozen_payload_sha256",
    )
    _require_equal(_payload_hash(config), FROZEN_CONFIG_PAYLOAD_SHA256, "payload_sha256")

    semantics = config["append_only_semantics"]
    for key, expected in {
        "append_only": True,
        "modifies_parent_corebench_contract": False,
        "modifies_cb2_functional_null_amendment_v1": False,
        "modifies_cb2a_design_preflight_v1": False,
        "overwrites_cb1_or_cb2a_v1_artifacts": False,
    }.items():
        _require_equal(semantics.get(key), expected, f"append_only_semantics.{key}")

    assignment = config["frozen_assignment_bank"]
    for key, expected in {
        "generator_may_be_called_by_cb2a_v2": False,
        "generator_may_replace_or_reorder_assignments": False,
        "n_assignments": 500,
        "n_donors": 75,
        "pseudo_case_donors": 37,
        "pseudo_control_donors": 38,
    }.items():
        _require_equal(assignment.get(key), expected, f"frozen_assignment_bank.{key}")

    grid = config["grid_contract"]
    _require_equal(grid.get("n_bins"), 20, "grid_contract.n_bins")
    edges = np.asarray(grid.get("edges"), dtype="<f8")
    _require_equal(edges.shape, (21,), "grid_contract.edges shape")
    _require(
        bool(np.array_equal(edges, np.linspace(0.0, 1.0, 21, dtype="<f8"))),
        "grid_contract.edges are not the frozen 20-bin grid",
    )
    _require_equal(
        _hash_array(edges, "<f8"),
        grid["edges_float64_le_sha256"],
        "grid_contract.edges_float64_le_sha256",
    )

    mapping = config["availability_mapping_contract"]
    for key, expected in {
        "selected_bin_signature_substitution_allowed": False,
        "experiment_as_hard_mapping_block": False,
        "one_global_donor_mapping_applied_to_all_valid_bins_and_pathways": True,
        "singleton_blocks_map_to_identity": True,
        "identity_global_mapping_excluded_from_sampled_null": True,
        "mapping_stream_shared_by_testing_and_simultaneous_bands": True,
    }.items():
        _require_equal(mapping.get(key), expected, f"availability_mapping_contract.{key}")

    nuisance = config["nuisance_design_contract"]
    _require_equal(nuisance.get("bin_specific_design_required"), True, "bin-specific design")
    _require_equal(nuisance.get("intercept"), "omitted_because_available_experiment_fractions_sum_to_one", "intercept")
    _require_equal(nuisance.get("rank_relative_tolerance"), 1e-10, "rank tolerance")
    _require_equal(nuisance.get("solve_method"), "full_rank_QR_without_pseudoinverse", "solve method")

    inference = config["functional_inference_contract"]
    for key, expected in {
        "functional_only": True,
        "timing_computed": False,
        "timing_fields_present": False,
        "timing_output": False,
        "timing_affects_acceptance": False,
        "finite_sample_exact": False,
        "reference_type": "freedman_lane_monte_carlo_approximation",
        "nuisance_block_invariant": False,
        "acceptance_thresholds_modified": False,
    }.items():
        _require_equal(inference.get(key), expected, f"functional_inference_contract.{key}")
    band = inference["simultaneous_band"]
    for key, expected in {
        "scope": "global_frozen_50_pathway_universe_by_all_valid_bins",
        "separate_13_family_bands": False,
        "statistic": "two_sided_absolute_studentized_max",
        "residual_basis": "full_model_residuals",
        "identity_excluded": True,
        "mapping_stream_shared_with_testing": True,
    }.items():
        _require_equal(band.get(key), expected, f"simultaneous_band.{key}")

    output = config["strict_output_contract"]
    _require_equal(output.get("additional_properties_allowed"), False, "strict schema")
    _require_equal(output.get("json_nan_or_infinity_allowed"), False, "strict JSON")
    _require_equal(output.get("timing_computed_must_equal"), False, "timing marker")
    _require_equal(
        output.get("timing_fields_present_must_equal"),
        False,
        "timing-fields-present marker",
    )
    _require_equal(
        output.get("forbidden_key_tokens"),
        [
            "onset",
            "duration",
            "phase",
            "delay",
            "phase_shift",
            "peak_location",
            "peak_time",
            "heterochrony",
            "transient",
            "sustained",
            "event_support",
            "event_time",
        ],
        "strict_output_contract.forbidden_key_tokens",
    )

    reference = config["permutation_reference"]
    for key, expected in {
        "method": "restricted_whole_donor_freedman_lane",
        "exact_finite_sample": False,
        "label": "monte_carlo_approximation",
        "mappings_per_replicate": 999,
    }.items():
        _require_equal(reference.get(key), expected, f"permutation_reference.{key}")

    state = config["execution_state"]
    for key in (
        "cb2_500_currently_authorized",
        "pathway_scoring_currently_authorized",
        "real_endoderm_contrast_authorized",
    ):
        _require_equal(state.get(key), False, f"execution_state.{key}")
    result = deepcopy(dict(config))
    result["_config_payload_sha256"] = FROZEN_CONFIG_PAYLOAD_SHA256
    return result


def load_cb2_implementation_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("CB2 implementation-freeze config must be a YAML mapping")
    return validate_cb2_implementation_config(value)


def verify_cb2_implementation_bindings(
    repository_root: str | Path, config: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    root = Path(repository_root).resolve()
    verified: dict[str, dict[str, Any]] = {}
    for name, binding in config["bindings"].items():
        path = _repo_file(root, str(binding["relative_path"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = _hash_file(path)
        _require_equal(observed, str(binding["sha256"]), f"bindings.{name}.sha256")
        verified[name] = {
            "relative_path": str(binding["relative_path"]),
            "sha256": observed,
            "bytes": int(path.stat().st_size),
        }
    return verified


def validate_frozen_assignment_manifest(
    path: str | Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    assignment_config = config["frozen_assignment_bank"]
    frame = pd.read_csv(Path(path), sep="\t", dtype="string")
    _require_equal(tuple(frame.columns), ASSIGNMENT_COLUMNS, "assignment columns")
    frame["pseudo_case_bool"] = _read_bool(frame["pseudo_case"])
    expected_ids = [f"CB2P_{index:04d}" for index in range(1, 501)]
    observed_ids = frame["assignment_id"].drop_duplicates().astype(str).tolist()
    _require_equal(observed_ids, expected_ids, "assignment id order")

    donor_order: tuple[str, ...] | None = None
    hashes: list[str] = []
    donor_case_counts: dict[str, int] = {}
    for assignment_id, group in frame.groupby("assignment_id", sort=False):
        donors = tuple(group["donor_id"].astype(str))
        _require_equal(len(donors), int(assignment_config["n_donors"]), f"{assignment_id} donor count")
        _require_equal(len(set(donors)), len(donors), f"{assignment_id} unique donors")
        _require_equal(donors, tuple(sorted(donors)), f"{assignment_id} donor order")
        if donor_order is None:
            donor_order = donors
            donor_case_counts = {donor: 0 for donor in donors}
        else:
            _require_equal(donors, donor_order, f"{assignment_id} donor identity")
        values = group["pseudo_case_bool"].to_numpy(dtype=np.uint8)
        _require_equal(int(values.sum()), 37, f"{assignment_id} pseudo-case count")
        labels = np.where(values.astype(bool), "pseudo_case", "pseudo_control")
        _require(
            bool(np.array_equal(labels, group["pseudo_condition"].astype(str).to_numpy())),
            f"{assignment_id} pseudo-condition labels disagree",
        )
        digest = _assignment_hash(values)
        recorded = group["assignment_sha256"].astype(str).unique().tolist()
        _require_equal(recorded, [digest], f"{assignment_id} assignment hash")
        hashes.append(digest)
        for donor, value in zip(donors, values, strict=True):
            donor_case_counts[donor] += int(value)

    _require_equal(len(set(hashes)), 500, "unique assignment hashes")
    _require(donor_order is not None, "Assignment manifest is empty")
    immobile = sum(count in {0, 500} for count in donor_case_counts.values())
    _require_equal(immobile, 0, "label-immobile donors")
    return {
        "relative_authority": "bound_frozen_assignment_manifest_v1",
        "sha256": _hash_file(Path(path)),
        "n_rows": int(len(frame)),
        "n_assignments": 500,
        "n_unique_assignment_hashes": 500,
        "n_donors": 75,
        "pseudo_case_donors_per_assignment": 37,
        "pseudo_control_donors_per_assignment": 38,
        "n_label_immobile_donors": 0,
        "minimum_donor_pseudo_case_fraction": float(min(donor_case_counts.values()) / 500),
        "maximum_donor_pseudo_case_fraction": float(max(donor_case_counts.values()) / 500),
        "generator_called": False,
    }


def _schema_validator(root: Path, config: Mapping[str, Any]) -> Draft202012Validator:
    binding = config["bindings"]["strict_functional_result_schema_v1"]
    schema = _strict_json_load(_repo_file(root, str(binding["relative_path"])))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _forbidden_result_keys(value: Any, tokens: Sequence[str], prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            normalized = str(key).lower()
            if any(token.lower() in normalized for token in tokens):
                found.append(path)
            found.extend(_forbidden_result_keys(item, tokens, path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_forbidden_result_keys(item, tokens, f"{prefix}[{index}]"))
    return found


def _kernel_interface_evidence(config: Mapping[str, Any]) -> dict[str, Any]:
    module = importlib.import_module(KERNEL_MODULE)
    expected = [
        *config["implementation_api"]["public_functions"],
        *config["implementation_api"]["public_dataclasses"],
        config["implementation_api"]["public_exception"],
    ]
    missing = [name for name in expected if not hasattr(module, name)]
    signatures = {
        name: str(inspect.signature(getattr(module, name)))
        for name in config["implementation_api"]["public_functions"]
        if hasattr(module, name)
    }
    return {
        "module": KERNEL_MODULE,
        "required_public_names": expected,
        "missing_public_names": missing,
        "function_signatures": signatures,
        "pass": not missing,
    }


def run_synthetic_capability_checks(
    *, repository_root: str | Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Run deterministic pure-array checks against the frozen functional API.

    This function intentionally contains no expression loader, pathway scorer, real
    condition reader, or assignment generator. The kernel-facing portion is kept in
    one function so the final implementation API can be frozen and audited directly.
    """

    root = Path(repository_root).resolve()
    required_ids = list(
        config["synthetic_capability_test_contract"]["required_check_ids"]
    )
    interface = _kernel_interface_evidence(config)
    checks: list[dict[str, Any]] = []

    def record(check_id: str, passed: bool, evidence: Mapping[str, Any]) -> None:
        checks.append(
            {
                "check_id": check_id,
                "pass": bool(passed),
                "evidence": deepcopy(dict(evidence)),
            }
        )

    if not interface["pass"]:
        for check_id in required_ids:
            record(
                check_id,
                False,
                {"reason_code": "functional_core_public_api_incomplete"},
            )
    else:
        try:
            module = importlib.import_module(KERNEL_MODULE)
            n_donors = 28
            n_bins = 20
            n_pathways = 2
            n_mappings = 999
            seed = 2026071403
            donor_ids = tuple(f"D{index:02d}" for index in range(n_donors))
            condition = np.arange(n_donors, dtype=np.uint8) % 2

            availability = np.ones((n_donors, n_bins), dtype=bool)
            availability[24:26, 0] = False
            availability[26, 1] = False
            availability[27, 2:4] = False

            experiment_ids = (
                "expt_a",
                "expt_b",
                "expt_c",
                "expt_z_global_zero",
            )
            fractions = np.zeros(
                (n_donors, n_bins, len(experiment_ids)), dtype=float
            )
            for donor_index in range(n_donors):
                for bin_index in range(n_bins):
                    if not availability[donor_index, bin_index]:
                        continue
                    if bin_index == 0:
                        active = (0, 1)
                    elif bin_index == 1:
                        active = (1, 2)
                    else:
                        active = (0, 1, 2)
                    raw = np.asarray(
                        [
                            1.0
                            + (
                                donor_index * (experiment_index + 2)
                                + bin_index * (experiment_index + 1)
                                + experiment_index * experiment_index
                                + 3
                            )
                            % (5 + experiment_index)
                            for experiment_index in active
                        ],
                        dtype=float,
                    )
                    fractions[donor_index, bin_index, list(active)] = (
                        raw / raw.sum()
                    )

            outcomes = np.full(
                (n_donors, n_bins, n_pathways), np.nan, dtype=float
            )
            for donor_index in range(n_donors):
                for bin_index in range(n_bins):
                    if not availability[donor_index, bin_index]:
                        continue
                    for pathway_index in range(n_pathways):
                        outcomes[donor_index, bin_index, pathway_index] = (
                            0.2
                            * condition[donor_index]
                            * (pathway_index + 1)
                            + 0.03 * bin_index * (pathway_index + 1)
                            + 0.15
                            * np.sin(
                                (donor_index + 1) * (pathway_index + 2)
                                + 0.7 * bin_index
                            )
                            + 0.07
                            * np.cos(
                                (donor_index + 3) * (bin_index + 1) * 0.11
                            )
                        )

            mapping_plan = module.build_full_availability_mapping_plan(
                donor_ids,
                availability,
                n_mappings=n_mappings,
                seed=seed,
            )
            identity = np.arange(n_donors, dtype=np.int32)
            signature_preserved = all(
                bool(np.array_equal(availability, availability[mapping]))
                for mapping in mapping_plan.mappings
            )
            record(
                "full_20_bin_availability_signature_mapping",
                bool(
                    mapping_plan.availability.shape[1] == 20
                    and mapping_plan.n_unique_signatures == 4
                    and signature_preserved
                ),
                {
                    "availability_signature_width": int(
                        mapping_plan.availability.shape[1]
                    ),
                    "n_unique_signatures": int(
                        mapping_plan.n_unique_signatures
                    ),
                    "all_mappings_preserve_full_signature": bool(
                        signature_preserved
                    ),
                    "selected_bin_signature_substitution_used": False,
                },
            )

            singleton_indices = [
                int(group[0]) for group in mapping_plan.groups if len(group) == 1
            ]
            singleton_identity = all(
                bool(
                    np.all(
                        mapping_plan.mappings[:, donor_index] == donor_index
                    )
                )
                for donor_index in singleton_indices
            )
            global_identity_present = bool(
                np.any(np.all(mapping_plan.mappings == identity[None, :], axis=1))
            )
            record(
                "singleton_blocks_identity_and_global_mapping",
                bool(
                    len(singleton_indices) == 2
                    and singleton_identity
                    and not global_identity_present
                    and len(set(mapping_plan.mapping_hashes)) == n_mappings
                ),
                {
                    "n_singleton_blocks": int(len(singleton_indices)),
                    "singleton_blocks_identity": bool(singleton_identity),
                    "global_identity_mapping_present": bool(
                        global_identity_present
                    ),
                    "n_unique_mapping_hashes": int(
                        len(set(mapping_plan.mapping_hashes))
                    ),
                    "mapping_stream_sha256": mapping_plan.mapping_stream_sha256,
                },
            )

            design_plan = module.build_bin_specific_designs(
                donor_ids,
                condition,
                availability,
                fractions,
                experiment_ids,
            )
            bin_specific = bool(
                design_plan.bins[0].active_experiment_ids
                != design_plan.bins[1].active_experiment_ids
                and design_plan.bins[0].dropped_bin_all_zero_experiment_ids
                == ("expt_c",)
                and design_plan.bins[1].dropped_bin_all_zero_experiment_ids
                == ("expt_a",)
            )
            record(
                "bin_specific_experiment_fraction_design",
                bool(design_plan.all_estimable and bin_specific),
                {
                    "n_bins": int(len(design_plan.bins)),
                    "all_bins_estimable": bool(design_plan.all_estimable),
                    "bin_specific_active_columns_observed": bool(bin_specific),
                    "bin_0_active_column_count": int(
                        len(design_plan.bins[0].active_experiment_ids)
                    ),
                    "bin_1_active_column_count": int(
                        len(design_plan.bins[1].active_experiment_ids)
                    ),
                },
            )

            qr_rank_contract = all(
                item.reduced_rank == len(item.active_experiment_ids)
                and item.full_rank == item.reduced_rank + 1
                and item.condition_column_index == item.reduced_rank
                for item in design_plan.bins
            )
            record(
                "global_zero_column_drop_no_intercept_rank_tolerance",
                bool(
                    design_plan.dropped_global_all_zero_experiment_ids
                    == ("expt_z_global_zero",)
                    and design_plan.rank_relative_tolerance == 1.0e-10
                    and design_plan.to_dict()["no_intercept"] is True
                    and qr_rank_contract
                ),
                {
                    "global_zero_column_dropped": bool(
                        design_plan.dropped_global_all_zero_experiment_ids
                        == ("expt_z_global_zero",)
                    ),
                    "no_intercept": bool(design_plan.to_dict()["no_intercept"]),
                    "rank_relative_tolerance": float(
                        design_plan.rank_relative_tolerance
                    ),
                    "all_reduced_and_full_designs_full_rank": bool(
                        qr_rank_contract
                    ),
                    "solve_method": "reduced_QR_without_pseudoinverse",
                },
            )

            result = module.run_functional_core(
                outcomes=outcomes,
                donor_ids=donor_ids,
                condition=condition,
                availability=availability,
                experiment_fractions=fractions,
                experiment_ids=experiment_ids,
                pathway_ids=("synthetic_curve_1", "synthetic_curve_2"),
                family_ids=("synthetic_family", "synthetic_family"),
                n_mappings=n_mappings,
                seed=seed,
            )
            shared_stream = bool(
                result.mapping_plan.mapping_stream_sha256
                == mapping_plan.mapping_stream_sha256
                and result.null_effect.shape[0] == n_mappings
                and result.bootstrap_effect.shape[0] == n_mappings
                and result.inference_metadata["test_reference"]
                == "reduced_residual_freedman_lane"
                and result.inference_metadata["band_reference"]
                == "full_residual_bootstrap"
                and result.inference_metadata["mapping_scope"]
                == "whole_donor_same_mapping_all_bins_and_pathways"
            )
            record(
                "whole_donor_freedman_lane_shared_mapping_stream",
                shared_stream,
                {
                    "test_reference": result.inference_metadata[
                        "test_reference"
                    ],
                    "band_reference": result.inference_metadata[
                        "band_reference"
                    ],
                    "mapping_scope": result.inference_metadata["mapping_scope"],
                    "n_mappings": int(result.mapping_plan.n_mappings),
                    "mapping_stream_sha256": (
                        result.mapping_plan.mapping_stream_sha256
                    ),
                    "test_and_band_mapping_axes_match_single_plan": bool(
                        shared_stream
                    ),
                },
            )

            expected_order = min(
                n_mappings,
                int(math.ceil((n_mappings + 1) * (1.0 - 0.05))),
            )
            bootstrap_maximum = np.max(
                np.abs(result.bootstrap_studentized_deviation)[
                    :, result.support_mask
                ],
                axis=1,
            )
            recomputed_critical = float(
                np.sort(bootstrap_maximum)[expected_order - 1]
            )
            bands_match = bool(
                result.band_order_index_1based == expected_order
                and math.isfinite(result.simultaneous_critical)
                and math.isclose(
                    result.simultaneous_critical,
                    recomputed_critical,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                and np.allclose(
                    result.simultaneous_lower,
                    result.effect
                    - result.simultaneous_critical * result.standard_error,
                    rtol=0.0,
                    atol=1.0e-12,
                )
                and np.allclose(
                    result.simultaneous_upper,
                    result.effect
                    + result.simultaneous_critical * result.standard_error,
                    rtol=0.0,
                    atol=1.0e-12,
                )
            )
            record(
                "global_simultaneous_band_full_residual_calibration",
                bands_match,
                {
                    "band_scope": result.inference_metadata["band_scope"],
                    "band_reference": result.inference_metadata[
                        "band_reference"
                    ],
                    "two_sided_absolute_studentized_max": True,
                    "band_order_index_1based": int(
                        result.band_order_index_1based
                    ),
                    "expected_band_order_index_1based": int(expected_order),
                    "simultaneous_critical_finite": bool(
                        math.isfinite(result.simultaneous_critical)
                    ),
                    "critical_recomputed_from_full_residual_bootstrap": bool(
                        bands_match
                    ),
                },
            )

            payload = result.to_dict()
            schema_binding = config["bindings"][
                "strict_functional_result_schema_v1"
            ]
            external_schema = _strict_json_load(
                _repo_file(root, str(schema_binding["relative_path"]))
            )
            schema_equal = bool(
                external_schema == module.FUNCTIONAL_CORE_RESULT_SCHEMA
            )
            validator = _schema_validator(root, config)
            validator.validate(payload)
            injected_timing_payload = deepcopy(payload)
            injected_timing_payload["onset"] = [0.5]
            schema_rejects_injected_timing_field = bool(
                list(validator.iter_errors(injected_timing_payload))
            )
            expected_top_level = set(external_schema["required"])
            exact_top_level = set(payload) == expected_top_level
            raw_array_keys = {
                "null_effect",
                "null_studentized_effect",
                "bootstrap_effect",
                "bootstrap_studentized_deviation",
                "mappings",
                "mapping_hashes",
            }
            forbidden = _forbidden_result_keys(
                payload, config["strict_output_contract"]["forbidden_key_tokens"]
            )
            kernel_path = _repo_file(
                root,
                str(
                    config["bindings"]["functional_core_kernel_v1"][
                        "relative_path"
                    ]
                ),
            )
            parsed = ast.parse(kernel_path.read_text(encoding="utf-8"))
            imported_modules: list[str] = []
            for node in ast.walk(parsed):
                if isinstance(node, ast.Import):
                    imported_modules.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_modules.append(node.module)
            trajectory_events_imported = any(
                "trajectory_events" in name for name in imported_modules
            )
            with tempfile.TemporaryDirectory(
                prefix="trajpathmix-cb2-synthetic-writer-"
            ) as temporary_directory:
                target = Path(temporary_directory) / "functional_result.json"
                write_record = module.write_functional_core_result_json(
                    result, target, create_only=True
                )
                written_payload = _strict_json_load(target)
                validator.validate(written_payload)
                writer_roundtrip = bool(written_payload == payload)
                writer_sha256 = str(write_record["sha256"])
            claim_scope = payload["claim_scope"]
            strict_pass = bool(
                schema_equal
                and external_schema.get("additionalProperties") is False
                and exact_top_level
                and not (set(payload) & raw_array_keys)
                and not forbidden
                and claim_scope["functional_core_only"] is True
                and claim_scope["timing_computed"] is False
                and claim_scope["timing_fields_present"] is False
                and claim_scope["finite_sample_exact"] is False
                and claim_scope["nuisance_block_invariant"] is False
                and claim_scope["strong_fwer_claimed"] is False
                and claim_scope["fwer_scope"]
                == "complete_null_weak_fwer_only"
                and claim_scope["nonzero_curve_coverage_claimed"] is False
                and not trajectory_events_imported
                and writer_roundtrip
                and schema_rejects_injected_timing_field
            )
            record(
                "strict_functional_only_timing_free_schema",
                strict_pass,
                {
                    "external_schema_equals_kernel_schema": bool(schema_equal),
                    "additional_properties_allowed": False,
                    "exact_top_level_fields": bool(exact_top_level),
                    "raw_null_bootstrap_or_mapping_arrays_present": bool(
                        set(payload) & raw_array_keys
                    ),
                    "forbidden_timing_key_count": int(len(forbidden)),
                    "functional_core_only": bool(
                        claim_scope["functional_core_only"]
                    ),
                    "timing_computed": bool(claim_scope["timing_computed"]),
                    "timing_fields_present": bool(
                        claim_scope["timing_fields_present"]
                    ),
                    "finite_sample_exact": bool(
                        claim_scope["finite_sample_exact"]
                    ),
                    "reference_type": claim_scope["reference_type"],
                    "nuisance_block_invariant": bool(
                        claim_scope["nuisance_block_invariant"]
                    ),
                    "strong_fwer_claimed": bool(
                        claim_scope["strong_fwer_claimed"]
                    ),
                    "fwer_scope": claim_scope["fwer_scope"],
                    "nonzero_curve_coverage_claimed": bool(
                        claim_scope["nonzero_curve_coverage_claimed"]
                    ),
                    "trajectory_events_imported": bool(
                        trajectory_events_imported
                    ),
                    "strict_writer_roundtrip": bool(writer_roundtrip),
                    "strict_writer_payload_sha256": writer_sha256,
                    "schema_rejects_injected_timing_field": bool(
                        schema_rejects_injected_timing_field
                    ),
                },
            )
        except Exception as exc:
            observed_ids = {str(row["check_id"]) for row in checks}
            for check_id in required_ids:
                if check_id not in observed_ids:
                    record(
                        check_id,
                        False,
                        {
                            "reason_code": "synthetic_capability_exception",
                            "exception_type": type(exc).__name__,
                            "exception_message": str(exc),
                        },
                    )

    check_by_id = {str(row["check_id"]): bool(row["pass"]) for row in checks}
    exact_check_set = len(check_by_id) == len(checks) and set(check_by_id) == set(
        required_ids
    )
    all_pass = bool(
        interface["pass"]
        and exact_check_set
        and all(check_by_id.get(check_id, False) for check_id in required_ids)
    )
    return {
        "schema_name": "trajpathmix_corebench_cb2_synthetic_capability_evidence",
        "schema_version": "1.0.0",
        "interface": interface,
        "checks": checks,
        "required_check_ids": required_ids,
        "exact_required_check_set": bool(exact_check_set),
        "all_required_checks_pass": all_pass,
        "blocking_reason_codes": [
            f"synthetic_capability_failed:{check_id}"
            for check_id in required_ids
            if not check_by_id.get(check_id, False)
        ],
        "pure_synthetic_arrays_only": True,
        "synthetic_shape": {
            "n_donors": 28,
            "n_bins": 20,
            "n_pathways": 2,
            "n_mappings": 999,
        },
        "expression_values_read": False,
        "pathway_scoring_performed": False,
        "real_condition_labels_read": False,
        "timing_computed": False,
        "timing_fields_present": False,
        "cb2_500_run": False,
    }


def _freeze_payload(
    *, root: Path, config: Mapping[str, Any], verified: Mapping[str, Any]
) -> dict[str, Any]:
    assignment_path = _repo_file(
        root,
        str(config["bindings"]["frozen_assignment_manifest_v1"]["relative_path"]),
    )
    assignments = validate_frozen_assignment_manifest(assignment_path, config)
    evidence = run_synthetic_capability_checks(repository_root=root, config=config)
    return {
        "schema_name": "trajpathmix_corebench_cb2_implementation_freeze_artifact",
        "schema_version": SCHEMA_VERSION,
        "freeze_id": FREEZE_ID,
        "config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
        "append_only_semantics": config["append_only_semantics"],
        "verified_input_bindings": verified,
        "frozen_assignment_bank_validation": assignments,
        "grid_contract": config["grid_contract"],
        "implementation_api": config["implementation_api"],
        "availability_mapping_contract": config["availability_mapping_contract"],
        "nuisance_design_contract": config["nuisance_design_contract"],
        "functional_inference_contract": config["functional_inference_contract"],
        "permutation_reference": config["permutation_reference"],
        "strict_output_contract": config["strict_output_contract"],
        "synthetic_capability_evidence": evidence,
        "implementation_capability_pass": bool(evidence["all_required_checks_pass"]),
        "preserves_cb2a_v1_fail_closed_evidence": True,
        "expression_values_read": False,
        "pathway_outcomes_read": False,
        "pathway_scoring_performed": False,
        "real_condition_contrast_read_or_generated": False,
        "timing_computed": False,
        "timing_fields_present": False,
        "cb2_500_run": False,
    }


def _passport_payload(config: Mapping[str, Any], freeze: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_name": "trajpathmix_corebench_cb2_implementation_material_passport",
        "schema_version": SCHEMA_VERSION,
        "freeze_id": FREEZE_ID,
        **config["material_passport"],
        "input_binding_sha256": {
            name: binding["sha256"] for name, binding in config["bindings"].items()
        },
        "frozen_assignment_manifest_sha256": freeze["frozen_assignment_bank_validation"]["sha256"],
        "synthetic_capability_pass": freeze["implementation_capability_pass"],
        "verification_status": "VERIFIED_PURE_SYNTHETIC_CAPABILITIES" if freeze["implementation_capability_pass"] else "FAIL_CLOSED_INCOMPLETE_IMPLEMENTATION",
        "next_gate": "cb2a_v2_implementation_readiness_without_pathway_scoring" if freeze["implementation_capability_pass"] else "complete_and_refreeze_theory_driven_implementation",
    }


def _build_record(
    *, config_path: Path, root: Path, artifact_dir: Path, verified: Mapping[str, Any]
) -> dict[str, Any]:
    names = (FREEZE_FILE, MATERIAL_PASSPORT_FILE)
    return {
        "schema_name": "trajpathmix_corebench_cb2_implementation_freeze_build_record",
        "schema_version": SCHEMA_VERSION,
        "freeze_id": FREEZE_ID,
        "config_file": CONFIG_FILE,
        "config_file_sha256": _hash_file(config_path),
        "config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
        "implementation_file": IMPLEMENTATION_FILE,
        "implementation_sha256": _hash_file(Path(__file__).resolve()),
        "verified_bindings": verified,
        "artifacts": {
            name: {
                "sha256": _hash_file(artifact_dir / name),
                "bytes": int((artifact_dir / name).stat().st_size),
            }
            for name in names
        },
        "runtime": {
            "python": os.sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "expression_values_read": False,
        "pathway_scoring_performed": False,
        "real_condition_contrast_read_or_generated": False,
        "timing_computed": False,
        "timing_fields_present": False,
        "cb2_500_run": False,
        "evidence_revision_mode": "create_only_append_only",
    }


def build_and_write_cb2_implementation_freeze(
    *, config_path: str | Path, repository_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    root = Path(repository_root).resolve()
    output = Path(output_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"CB2 implementation-freeze output exists: {output}")
    lock = output.parent / f".{output.name}.create.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FileExistsError(f"CB2 implementation-freeze output is locked: {lock}") from exc
    temporary: Path | None = None
    try:
        os.close(descriptor)
        config = load_cb2_implementation_config(config_file)
        verified = verify_cb2_implementation_bindings(root, config)
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
        freeze = _freeze_payload(root=root, config=config, verified=verified)
        if not freeze["implementation_capability_pass"]:
            reasons = freeze["synthetic_capability_evidence"].get(
                "blocking_reason_codes", ["synthetic_capability_failure"]
            )
            raise RuntimeError(
                "Refusing to materialize a failed CB2 implementation freeze: "
                + "|".join(map(str, reasons))
            )
        passport = _passport_payload(config, freeze)
        _write_json(freeze, temporary / FREEZE_FILE)
        _write_json(passport, temporary / MATERIAL_PASSPORT_FILE)
        record = _build_record(
            config_path=config_file,
            root=root,
            artifact_dir=temporary,
            verified=verified,
        )
        _write_json(record, temporary / BUILD_RECORD_FILE)
        os.rename(temporary, output)
        temporary = None
        return {
            **record,
            "output_dir": str(output),
            "build_record_sha256": _hash_file(output / BUILD_RECORD_FILE),
            "implementation_capability_pass": freeze["implementation_capability_pass"],
        }
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
        lock.unlink(missing_ok=True)


def validate_cb2_implementation_freeze_output(
    *, config_path: str | Path, repository_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    root = Path(repository_root).resolve()
    output = Path(output_dir).resolve()
    if not output.is_dir():
        raise FileNotFoundError(output)
    expected_names = {FREEZE_FILE, MATERIAL_PASSPORT_FILE, BUILD_RECORD_FILE}
    _require_equal(
        {path.name for path in output.iterdir()},
        expected_names,
        "implementation-freeze output file set",
    )
    config = load_cb2_implementation_config(config_file)
    verified = verify_cb2_implementation_bindings(root, config)
    expected_freeze = _freeze_payload(root=root, config=config, verified=verified)
    observed_freeze = _strict_json_load(output / FREEZE_FILE)
    _require_equal(observed_freeze, expected_freeze, FREEZE_FILE)
    expected_passport = _passport_payload(config, expected_freeze)
    _require_equal(_strict_json_load(output / MATERIAL_PASSPORT_FILE), expected_passport, MATERIAL_PASSPORT_FILE)
    expected_record = _build_record(
        config_path=config_file,
        root=root,
        artifact_dir=output,
        verified=verified,
    )
    _require_equal(_strict_json_load(output / BUILD_RECORD_FILE), expected_record, BUILD_RECORD_FILE)
    return {
        **expected_record,
        "output_dir": str(output),
        "build_record_sha256": _hash_file(output / BUILD_RECORD_FILE),
        "implementation_capability_pass": expected_freeze["implementation_capability_pass"],
        "validation_status": "pass_append_only_cb2_implementation_freeze_integrity",
    }


__all__ = [
    "BUILD_RECORD_FILE",
    "CONFIG_FILE",
    "FREEZE_FILE",
    "FREEZE_ID",
    "FROZEN_CONFIG_PAYLOAD_SHA256",
    "MATERIAL_PASSPORT_FILE",
    "build_and_write_cb2_implementation_freeze",
    "load_cb2_implementation_config",
    "run_synthetic_capability_checks",
    "validate_cb2_implementation_config",
    "validate_cb2_implementation_freeze_output",
    "validate_frozen_assignment_manifest",
    "verify_cb2_implementation_bindings",
]
