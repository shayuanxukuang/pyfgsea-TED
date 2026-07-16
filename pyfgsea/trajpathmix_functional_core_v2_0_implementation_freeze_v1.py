from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .trajpathmix_functional_core_v2 import (
    FunctionalCoreV2DesignError,
    METHOD_ID,
    fit_lodo_donor_influence,
    generate_multiplier_stream,
    run_functional_core_v2,
)


SCHEMA_NAME = "trajpathmix_functional_core_v2_0_implementation_freeze"
SCHEMA_VERSION = "1.0.0"
PROJECT_ID = "TRAJPATHMIX_FUNCTIONAL_CORE_V2_0_IMPLEMENTATION_FREEZE_v1"
FROZEN_PAYLOAD_SHA256 = "41f5402f96a9be0bff4f7534436af82c5c32840e530ef7daf7c4e71fee90ca76"

CONFIG_FILE = "config/trajpathmix_functional_core_v2_0_implementation_freeze_v1.yaml"
CORE_MODULE_FILE = "pyfgsea/trajpathmix_functional_core_v2.py"
MODULE_FILE = "pyfgsea/trajpathmix_functional_core_v2_0_implementation_freeze_v1.py"
RUNNER_FILE = "scripts/freeze_trajpathmix_functional_core_v2_0_implementation_v1.py"
TEST_FILE = "tests/test_trajpathmix_functional_core_v2_0_implementation_freeze_v1.py"
CORE_TEST_FILE = "tests/test_trajpathmix_functional_core_v2.py"
OUTPUT_DIR = "data_external/trajpathmix_functional_core_v2_0_implementation_freeze_v1"

DECISION_FILE = "TRAJPATHMIX_FUNCTIONAL_CORE_V2_0_IMPLEMENTATION_DECISION_v1.json"
CHECKS_FILE = "trajpathmix_functional_core_v2_0_check_results_v1.tsv"
PASSPORT_FILE = "trajpathmix_functional_core_v2_0_material_passport_v1.json"
BUILD_RECORD_FILE = "trajpathmix_functional_core_v2_0_build_record_v1.json"
EXACT_OUTPUT_FILES = (DECISION_FILE, CHECKS_FILE, PASSPORT_FILE, BUILD_RECORD_FILE)

CHECK_IDS = (
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

FALLACY_IDS = (
    "simpsons_paradox",
    "ecological_fallacy",
    "berksons_paradox",
    "collider_bias",
    "base_rate_neglect",
    "regression_to_the_mean",
    "survivorship_bias",
    "look_elsewhere_effect",
    "garden_of_forking_paths",
    "correlation_not_causation",
    "reverse_causality",
)


class V20ImplementationFreezeError(ValueError):
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


def _hash_normalized_freeze_harness(path: Path) -> str:
    source = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    lines = source.splitlines(keepends=True)
    matched = 0
    normalized: list[str] = []
    for line in lines:
        if line.startswith("FROZEN_PAYLOAD_SHA256 = "):
            normalized.append('FROZEN_PAYLOAD_SHA256 = "<NORMALIZED>"\n')
            matched += 1
        else:
            normalized.append(line)
    _require_equal(matched, 1, "normalized harness payload-constant count")
    return hashlib.sha256("".join(normalized).encode("utf-8")).hexdigest()


def _hash_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    payload = (
        str(array.dtype).encode("ascii")
        + b"|"
        + _canonical_json(list(array.shape)).encode("ascii")
        + b"|"
        + array.tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def _repo_file(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise V20ImplementationFreezeError(
            "V2-0 implementation-freeze paths must be repository-local"
        ) from exc
    return path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V20ImplementationFreezeError(message)


def _require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise V20ImplementationFreezeError(
            f"V2-0 implementation-freeze mismatch for {label}: "
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
        raise V20ImplementationFreezeError(f"Non-finite JSON value at {label}")


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
            V20ImplementationFreezeError(
                f"Non-finite JSON constant {value!r} in {path}"
            )
        ),
    )


def load_v2_0_implementation_freeze_config(path: str | Path) -> dict[str, Any]:
    observed = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(observed, Mapping):
        raise V20ImplementationFreezeError("V2-0 freeze config must be a mapping")
    config = deepcopy(dict(observed))
    validate_v2_0_implementation_freeze_config(config)
    return config


def validate_v2_0_implementation_freeze_config(
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
        "modifies_v2_rebuild_authorization": False,
        "modifies_any_cb1_or_cb2_v1_artifact": False,
        "overwrites_prior_evidence": False,
    }.items():
        _require_equal(semantics.get(key), expected, f"append_only_semantics.{key}")
    implementation = config["implementation_contract"]
    for key, expected in {
        "method_id": METHOD_ID,
        "primary_independence_unit": "donor",
        "primary_multiplier": "rademacher",
        "mandatory_sensitivity_partition": "derived_donor_experiment_bipartite_connected_components",
        "mandatory_sensitivity_multiplier": "webb_six_point",
        "arbitrary_caller_component_partition_allowed": False,
        "one_component_primary_fit_allowed_with_sensitivity_unavailable": True,
        "shared_multiplier_across_all_coordinates": True,
        "signed_auc_uses_shared_cross_bin_influence": True,
        "global_band_scope": "50_pathways_x_20_bins",
        "monte_carlo_draws_for_999_draw_formal_fixture": 999,
        "q95_order_1based": 950,
        "unattainable_finite_draw_quantile_action": "fail_closed",
        "nonfinite_or_nonpositive_supported_se_action": "fail_closed",
    }.items():
        _require_equal(implementation.get(key), expected, f"implementation_contract.{key}")
    _require_equal(
        tuple(config["implementation_bindings"]),
        (
            "v2_core_module",
            "retained_v1_design_dependency",
            "v2_0_freeze_harness",
            "v2_0_runner",
            "v2_0_freeze_test",
            "v2_core_test",
        ),
        "implementation binding ids",
    )
    firewall = config["permanent_retirement_firewall"]
    _require_equal(firewall.get("retired_formal_engine_may_be_called"), False, "retired engine")
    _require_equal(firewall.get("old_cb2_outputs_may_be_read"), False, "old outputs")
    _require_equal(firewall.get("old_assignment_bank_may_be_read"), False, "old bank")
    checks = config["required_checks"]
    _require_equal(tuple(checks["check_ids"]), CHECK_IDS, "required check ids")
    _require_equal(checks.get("all_checks_required"), True, "all checks required")
    _require_equal(
        checks.get("suite_exception_action"),
        "atomically_publish_fail_for_every_affected_check_without_retry",
        "suite exception action",
    )
    suites = config["synthetic_suites"]
    algebra = suites["algebraic_and_invariance"]
    _require_equal(
        algebra.get("pathway_chunk_partitions_in_execution_order"),
        [[2, 3], [0, 2], [3, 5]],
        "pathway chunks",
    )
    _require_equal(algebra.get("worker_count_parameter_present"), False, "worker API")
    _require_equal(
        algebra.get("axis_invariance_orders"),
        {
            "donor": "reverse",
            "pathway": [4, 1, 3, 0, 2],
            "bin": [2, 0, 3, 1],
            "experiment": [2, 0, 7, 3, 1, 6, 4, 5],
        },
        "axis invariance orders",
    )
    _require_equal(
        algebra.get("rank_compression_fixture"),
        {
            "data_seed": 2026071704,
            "donors": 24,
            "experiments": 3,
            "first_component_donors": 18,
            "expected_full_maximum_leverage": 1.0,
            "expected_maximum_lodo_nuisance_rank_loss": 1,
        },
        "rank compression fixture",
    )
    heteroskedastic = suites["heteroskedastic_high_leverage_null"]
    _require_equal(heteroskedastic.get("data_seed"), 2026071705, "heteroskedastic seed")
    _require_equal(heteroskedastic.get("true_condition_effect"), 0.0, "null effect")
    oracle = suites["balanced_label_oracle"]
    _require_equal(oracle.get("nuisance_design"), "one_constant_column", "oracle nuisance")
    _require_equal(oracle.get("minimum_donors_per_condition"), 5, "oracle minimum group")
    _require_equal(oracle.get("exact_balanced_label_count"), 924, "oracle labels")
    global_suite = suites["global_50_x_20"]
    _require_equal(global_suite.get("donors"), 75, "global donors")
    _require_equal(global_suite.get("bins"), 20, "global bins")
    _require_equal(global_suite.get("pathways"), 50, "global pathways")
    _require_equal(global_suite.get("expected_component_count"), 7, "global components")
    _require_equal(
        global_suite.get("expected_component_sensitivity_informative"),
        False,
        "global sensitivity information",
    )
    oracle_seed = int(oracle["multiplier_seed"])
    formal_seeds = {
        int(algebra["data_seed"]),
        int(algebra["donor_multiplier_seed"]),
        int(algebra["component_multiplier_seed"]),
        int(algebra["rank_compression_fixture"]["data_seed"]),
        int(heteroskedastic["data_seed"]),
        oracle_seed,
        oracle_seed + 1,
        int(global_suite["data_seed"]),
        int(global_suite["donor_multiplier_seed"]),
        int(global_suite["component_multiplier_seed"]),
    }
    _require_equal(len(formal_seeds), 10, "formal synthetic stream seed uniqueness")
    fallacies = config["fallacy_scan_contract"]
    _require_equal(tuple(fallacies["required_items"]), FALLACY_IDS, "fallacy ids")
    _require_equal(fallacies.get("unresolved_red_flag_max"), 0, "red flag maximum")
    ceiling = config["claim_ceiling"]
    false_keys = (
        "functional_core_v2_calibrated",
        "curve_inference_allowed",
        "signed_auc_inference_allowed",
        "simultaneous_band_inference_allowed",
        "family_inference_allowed",
        "timing_claim_allowed",
        "real_condition_unblinding_allowed",
        "injection_recovery_allowed",
        "biological_discovery_allowed",
        "manuscript_method_claim_allowed",
        "release_allowed",
        "new_holdout_assignment_bank_materialization_authorized",
        "new_holdout_500_execution_authorized",
    )
    for key in false_keys:
        _require_equal(ceiling.get(key), False, f"claim_ceiling.{key}")
    _require_equal(ceiling.get("next_stage_authorized"), "none", "next stage")
    execution = config["execution_firewall"]
    for key in (
        "real_expression_read",
        "real_pathway_scores_read",
        "real_condition_labels_read",
        "current_cb2_500_assignment_bank_read",
        "current_cb2_500_outputs_read",
        "current_cb2_500_cache_read",
        "new_holdout_assignment_bank_generated",
        "future_holdout_multiplier_stream_generated",
        "pure_synthetic_labels_persisted",
        "biological_interpretation_performed",
        "timing_computed",
        "automatic_retry_allowed",
        "post_execution_threshold_change_allowed",
        "formal_fixture_result_inspection_before_materialization_allowed",
        "validate_only_may_rerun_synthetic_suites",
    ):
        _require_equal(execution.get(key), False, f"execution_firewall.{key}")
    _require_equal(
        execution.get("pure_synthetic_balanced_labels_enumerated_in_memory"),
        True,
        "synthetic oracle labels",
    )
    output = config["output_contract"]
    _require_equal(output.get("default_output_dir"), OUTPUT_DIR, "output dir")
    _require_equal(output.get("create_only"), True, "create-only")
    _require_equal(output.get("alternate_output_dir_allowed"), False, "alternate output")
    _require_equal(tuple(output["exact_output_files"]), EXACT_OUTPUT_FILES, "output files")
    return {
        "valid": True,
        "project_id": PROJECT_ID,
        "frozen_payload_sha256": FROZEN_PAYLOAD_SHA256,
        "n_required_checks": len(CHECK_IDS),
        "fallacy_scan_denominator": len(FALLACY_IDS),
    }


def verify_v2_0_authorization_bindings(
    root: str | Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    repository = Path(root).resolve()
    observed: dict[str, Any] = {}
    for binding_id, record in config["authorization_bindings"].items():
        path = _repo_file(repository, str(record["relative_path"]))
        _require(path.is_file(), f"Missing V2 authorization binding: {path}")
        digest = _hash_file(path)
        _require_equal(digest, record["sha256"], f"binding {binding_id}")
        observed[binding_id] = {
            "relative_path": str(record["relative_path"]),
            "sha256": digest,
            "size_bytes": int(path.stat().st_size),
        }
    authorization = _strict_json_load(
        _repo_file(
            repository,
            config["authorization_bindings"]["v2_rebuild_authorization"][
                "relative_path"
            ],
        )
    )
    _require_equal(
        authorization["rebuild_authorization"][
            "v2_0_analytic_and_pure_synthetic_implementation_authorized"
        ],
        True,
        "bound V2-0 authorization",
    )
    _require_equal(
        authorization["rebuild_authorization"][
            "v2_1_new_holdout_assignment_bank_materialization_authorized"
        ],
        False,
        "bound holdout-bank closure",
    )
    _require_equal(
        authorization["rebuild_authorization"]["v2_1_holdout_500_execution_authorized"],
        False,
        "bound holdout execution closure",
    )
    return observed


def verify_v2_0_implementation_bindings(
    root: str | Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    repository = Path(root).resolve()
    observed: dict[str, Any] = {}
    for binding_id, record in config["implementation_bindings"].items():
        path = _repo_file(repository, str(record["relative_path"]))
        _require(path.is_file(), f"Missing frozen implementation input: {path}")
        hash_mode = str(record["hash_mode"])
        if hash_mode == "file_sha256":
            digest = _hash_file(path)
        elif hash_mode == "normalized_frozen_payload_constant_sha256":
            digest = _hash_normalized_freeze_harness(path)
        else:
            raise V20ImplementationFreezeError(
                f"Unknown implementation binding hash mode: {hash_mode}"
            )
        _require_equal(digest, record["sha256"], f"implementation {binding_id}")
        observed[binding_id] = {
            "relative_path": str(record["relative_path"]),
            "hash_mode": hash_mode,
            "sha256": digest,
            "size_bytes": int(path.stat().st_size),
        }
    return observed


def _max_abs(value: Any) -> float:
    array = np.asarray(value, dtype=float)
    return float(np.max(np.abs(array))) if array.size else 0.0


def _direct_lodo_condition_weights(
    nuisance: np.ndarray, condition: np.ndarray
) -> np.ndarray:
    u, singular, _ = np.linalg.svd(np.asarray(nuisance, dtype=float), full_matrices=False)
    rank = (
        int(np.sum(singular > singular[0] * 1.0e-10))
        if len(singular) and singular[0] > 0
        else 0
    )
    basis = u[:, :rank]
    design = np.column_stack([basis, np.asarray(condition, dtype=float)])
    coefficient_map = np.linalg.lstsq(design, np.eye(len(condition)), rcond=None)[0]
    return coefficient_map[-1]


def _algebraic_fixture(spec: Mapping[str, Any]) -> dict[str, Any]:
    n_donors = int(spec["donors"])
    n_bins = int(spec["bins"])
    n_pathways = int(spec["pathways"])
    n_experiments = int(spec["experiments"])
    rng = np.random.default_rng(int(spec["data_seed"]))
    donors = tuple(f"SYN_D{index:03d}" for index in range(n_donors))
    bins = tuple(f"SYN_B{index:02d}" for index in range(n_bins))
    pathways = tuple(f"SYN_P{index:02d}" for index in range(n_pathways))
    experiments = tuple(f"SYN_E{index:02d}" for index in range(n_experiments))
    condition = np.asarray([index % 2 for index in range(n_donors)], dtype=np.uint8)
    availability = np.ones((n_donors, n_bins), dtype=bool)
    availability[-8:, -1] = False
    group_size = n_donors // n_experiments
    experiment_index = np.repeat(np.arange(n_experiments), group_size)
    fractions = np.zeros((n_donors, n_bins, n_experiments), dtype=float)
    fractions[
        np.arange(n_donors)[:, None],
        np.arange(n_bins)[None, :],
        experiment_index[:, None],
    ] = 1.0
    fractions[~availability] = 0.0
    nuisance_beta = rng.normal(scale=0.7, size=(n_experiments, n_pathways))
    true_effect = rng.normal(scale=0.2, size=(n_bins, n_pathways))
    donor_scale = np.linspace(0.35, 1.25, n_donors)
    outcomes = (
        np.einsum("dbe,ep->dbp", fractions, nuisance_beta, optimize=True)
        + condition[:, None, None] * true_effect[None, :, :]
        + rng.normal(size=(n_donors, n_bins, n_pathways))
        * donor_scale[:, None, None]
    )
    outcomes[~availability] = np.nan
    component_ids = tuple(f"provided_C{index:02d}" for index in experiment_index)
    family_ids = tuple(f"F{index % 3:02d}" for index in range(n_pathways))
    bin_weights = np.linspace(1.0, 2.0, n_bins)
    return {
        "outcomes": outcomes,
        "donor_ids": donors,
        "bin_ids": bins,
        "pathway_ids": pathways,
        "condition": condition,
        "availability": availability,
        "experiment_fractions": fractions,
        "experiment_ids": experiments,
        "experiment_component_ids": component_ids,
        "family_ids": family_ids,
        "bin_weights": bin_weights,
    }


def _run_algebraic_suite(spec: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, bool], Any]:
    inputs = _algebraic_fixture(spec)
    result = run_functional_core_v2(
        **inputs,
        n_multiplier_draws=int(spec["multiplier_draws"]),
        donor_multiplier_seed=int(spec["donor_multiplier_seed"]),
        component_multiplier_seed=int(spec["component_multiplier_seed"]),
        alpha=0.05,
    )
    fit = result.fit
    tolerance = float(spec["numeric_tolerance"])
    full_ols_error = 0.0
    lodo_error = 0.0
    variance_error = 0.0
    influence_sum_error = 0.0
    for item in fit.design_plan.bins:
        b = int(item.bin_index)
        indices = np.asarray(item.available_donor_indices, dtype=int)
        local_y = np.asarray(inputs["outcomes"])[indices, b]
        direct_full = np.linalg.lstsq(item.full_design, local_y, rcond=None)[0][-1]
        full_ols_error = max(full_ols_error, _max_abs(direct_full - fit.effect[b]))
        local_lodo = fit.leave_one_out_effect[indices, b]
        for local_index in range(len(indices)):
            keep = np.arange(len(indices)) != local_index
            weights = _direct_lodo_condition_weights(
                item.reduced_design[keep],
                np.asarray(inputs["condition"])[indices][keep],
            )
            direct = weights @ local_y[keep]
            lodo_error = max(
                lodo_error,
                _max_abs(direct - local_lodo[local_index]),
            )
        local_influence = fit.donor_influence[indices, b]
        influence_sum_error = max(
            influence_sum_error, _max_abs(local_influence.sum(axis=0))
        )
        direct_variance = (len(indices) - 1.0) / len(indices) * np.sum(
            (local_lodo - local_lodo.mean(axis=0)) ** 2, axis=0
        )
        variance_error = max(
            variance_error,
            _max_abs(direct_variance - fit.standard_error[b] ** 2),
        )

    identity_donor = np.arange(len(inputs["donor_ids"]))
    identity_bin = np.arange(len(inputs["bin_ids"]))
    identity_pathway = np.arange(len(inputs["pathway_ids"]))
    identity_experiment = np.arange(len(inputs["experiment_ids"]))

    def run_reordered(
        *,
        donor_order: np.ndarray = identity_donor,
        bin_order: np.ndarray = identity_bin,
        pathway_order: np.ndarray = identity_pathway,
        experiment_order: np.ndarray = identity_experiment,
    ):
        moved = dict(inputs)
        moved["donor_ids"] = tuple(np.asarray(inputs["donor_ids"])[donor_order])
        moved["bin_ids"] = tuple(np.asarray(inputs["bin_ids"])[bin_order])
        moved["pathway_ids"] = tuple(
            np.asarray(inputs["pathway_ids"])[pathway_order]
        )
        moved["experiment_ids"] = tuple(
            np.asarray(inputs["experiment_ids"])[experiment_order]
        )
        moved["condition"] = np.asarray(inputs["condition"])[donor_order]
        moved["availability"] = np.asarray(inputs["availability"])[donor_order][
            :, bin_order
        ]
        moved["outcomes"] = np.asarray(inputs["outcomes"])[donor_order][
            :, bin_order
        ][:, :, pathway_order]
        moved["experiment_fractions"] = np.asarray(
            inputs["experiment_fractions"]
        )[donor_order][:, bin_order][:, :, experiment_order]
        moved["experiment_component_ids"] = tuple(
            np.asarray(inputs["experiment_component_ids"])[donor_order]
        )
        moved["family_ids"] = tuple(
            np.asarray(inputs["family_ids"])[pathway_order]
        )
        moved["bin_weights"] = np.asarray(inputs["bin_weights"])[bin_order]
        return run_functional_core_v2(
            **moved,
            n_multiplier_draws=int(spec["multiplier_draws"]),
            donor_multiplier_seed=int(spec["donor_multiplier_seed"]),
            component_multiplier_seed=int(spec["component_multiplier_seed"]),
            alpha=0.05,
        )

    def reorder_errors(reordered: Any) -> dict[str, Any]:
        return {
            "effect": _max_abs(reordered.fit.effect - fit.effect),
            "influence": _max_abs(
                np.nan_to_num(reordered.fit.donor_influence - fit.donor_influence)
            ),
            "draw": _max_abs(
                np.nan_to_num(
                    reordered.donor_primary.studentized_draws
                    - result.donor_primary.studentized_draws
                )
            ),
            "stream_hash_equal": (
                reordered.donor_primary.multiplier_stream_sha256
                == result.donor_primary.multiplier_stream_sha256
            ),
        }

    axis_orders = spec["axis_invariance_orders"]
    _require_equal(axis_orders["donor"], "reverse", "donor reorder rule")
    donor_reorder = reorder_errors(run_reordered(donor_order=identity_donor[::-1]))
    pathway_reorder = reorder_errors(
        run_reordered(pathway_order=np.asarray(axis_orders["pathway"], dtype=int))
    )
    bin_reorder = reorder_errors(
        run_reordered(bin_order=np.asarray(axis_orders["bin"], dtype=int))
    )
    experiment_reorder = reorder_errors(
        run_reordered(
            experiment_order=np.asarray(axis_orders["experiment"], dtype=int)
        )
    )

    safe_influence = np.where(np.isfinite(fit.donor_influence), fit.donor_influence, 0.0)
    primary = result.donor_primary
    manual_numerator = np.einsum(
        "rd,dbp->rbp", primary.multipliers, safe_influence, optimize=True
    )
    shared_draw_error = _max_abs(
        np.nan_to_num(
            manual_numerator
            - primary.studentized_draws
            * primary.coordinate_standard_error[None, :, :]
        )
    )
    signed_from_coordinates = np.einsum(
        "rbp,bp->rp", manual_numerator, fit.pathway_bin_weights, optimize=True
    )
    signed_draw_error = _max_abs(
        signed_from_coordinates
        - primary.signed_auc_studentized_draws
        * primary.signed_auc_standard_error[None, :]
    )

    chunked = np.empty_like(manual_numerator)
    chunks = tuple(
        (int(start), int(stop))
        for start, stop in spec["pathway_chunk_partitions_in_execution_order"]
    )
    for start, stop in chunks:
        chunked[:, :, start:stop] = np.einsum(
            "rd,dbp->rbp",
            primary.multipliers,
            safe_influence[:, :, start:stop],
            optimize=True,
        )
    chunk_error = _max_abs(chunked - manual_numerator)
    regenerated = generate_multiplier_stream(
        n_draws=int(spec["multiplier_draws"]),
        n_units=len(fit.donor_ids),
        seed=int(spec["donor_multiplier_seed"]),
        distribution="rademacher",
    )
    bank_exact = bool(np.array_equal(regenerated, primary.multipliers))
    public_parameters = set(inspect.signature(run_functional_core_v2).parameters)
    worker_or_chunk_parameters = sorted(
        public_parameters
        & {"worker", "workers", "n_workers", "n_jobs", "chunk_size", "pathway_chunk_size"}
    )

    unavailable = ~fit.availability
    missing_na_exact = bool(
        np.isnan(fit.donor_influence[unavailable]).all()
        and np.isnan(fit.leverage[unavailable]).all()
    )
    corrupted = dict(inputs)
    corrupted["outcomes"] = np.asarray(inputs["outcomes"]).copy()
    corrupted["outcomes"][~np.asarray(inputs["availability"])] = 0.0
    missing_zero_rejected = False
    try:
        fit_lodo_donor_influence(**corrupted)
    except ValueError as exc:
        missing_zero_rejected = "remain NA" in str(exc)

    rank_spec = spec["rank_compression_fixture"]
    n = int(rank_spec["donors"])
    n_rank_experiments = int(rank_spec["experiments"])
    first_component_donors = int(rank_spec["first_component_donors"])
    _require_equal(n_rank_experiments, 3, "rank fixture experiment count")
    rank_rng = np.random.default_rng(
        int(rank_spec["data_seed"])
    )
    rank_fractions = np.zeros((n, 1, n_rank_experiments), dtype=float)
    rank_fractions[:first_component_donors, 0, :2] = 0.5
    rank_fractions[0, 0, :2] = (1.0, 0.0)
    rank_fractions[first_component_donors:, 0, 2] = 1.0
    rank_condition = np.asarray([index % 2 for index in range(n)], dtype=np.uint8)
    rank_outcomes = (
        rank_fractions[:, :, :1] * 0.6
        + rank_condition[:, None, None] * np.asarray([[[0.2, -0.1]]])
        + rank_rng.normal(scale=0.4, size=(n, 1, 2))
    )
    rank_fit = fit_lodo_donor_influence(
        outcomes=rank_outcomes,
        donor_ids=tuple(f"R{index:02d}" for index in range(n)),
        bin_ids=("RB0",),
        pathway_ids=("RP0", "RP1"),
        condition=rank_condition,
        availability=np.ones((n, 1), dtype=bool),
        experiment_fractions=rank_fractions,
        experiment_ids=("RE0", "RE1", "RE2"),
        experiment_component_ids=tuple(
            "RC0" if index < first_component_donors else "RC1"
            for index in range(n)
        ),
    )
    fake_partition_rejected = False
    try:
        fit_lodo_donor_influence(
            outcomes=rank_outcomes,
            donor_ids=tuple(f"R{index:02d}" for index in range(n)),
            bin_ids=("RB0",),
            pathway_ids=("RP0", "RP1"),
            condition=rank_condition,
            availability=np.ones((n, 1), dtype=bool),
            experiment_fractions=rank_fractions,
            experiment_ids=("RE0", "RE1", "RE2"),
            experiment_component_ids=tuple(f"fake_{index % 2}" for index in range(n)),
        )
    except FunctionalCoreV2DesignError as exc:
        fake_partition_rejected = "do not equal" in str(exc)

    sensitivity = result.experiment_overlap_sensitivity
    webb_present = bool(
        sensitivity is not None
        and sensitivity.distribution == "webb_six_point"
        and len(sensitivity.unit_ids) == int(spec["connected_components"])
        and sensitivity.finite_sample_scale
        == math.sqrt(int(spec["connected_components"]) / (int(spec["connected_components"]) - 1.0))
        and sensitivity.sensitivity_informative
        and fit.design_audit["experiment_component_source"]
        == "derived_donor_experiment_bipartite_connected_components"
        and fit.design_audit["supplied_component_partition_validated"]
        and fake_partition_rejected
    )

    metrics = {
        "fixture_outcome_sha256": _hash_array(inputs["outcomes"]),
        "full_ols_max_abs_error": full_ols_error,
        "all_deletion_lodo_max_abs_error": lodo_error,
        "influence_sum_max_abs_error": influence_sum_error,
        "jackknife_variance_max_abs_error": variance_error,
        "donor_reorder_effect_max_abs_error": donor_reorder["effect"],
        "donor_reorder_influence_max_abs_error": donor_reorder["influence"],
        "donor_reorder_draw_max_abs_error": donor_reorder["draw"],
        "donor_reorder_stream_hash_equal": donor_reorder["stream_hash_equal"],
        "pathway_reorder_effect_max_abs_error": pathway_reorder["effect"],
        "pathway_reorder_influence_max_abs_error": pathway_reorder["influence"],
        "pathway_reorder_draw_max_abs_error": pathway_reorder["draw"],
        "pathway_reorder_stream_hash_equal": pathway_reorder["stream_hash_equal"],
        "bin_reorder_effect_max_abs_error": bin_reorder["effect"],
        "bin_reorder_influence_max_abs_error": bin_reorder["influence"],
        "bin_reorder_draw_max_abs_error": bin_reorder["draw"],
        "bin_reorder_stream_hash_equal": bin_reorder["stream_hash_equal"],
        "experiment_reorder_effect_max_abs_error": experiment_reorder["effect"],
        "experiment_reorder_influence_max_abs_error": experiment_reorder[
            "influence"
        ],
        "experiment_reorder_draw_max_abs_error": experiment_reorder["draw"],
        "experiment_reorder_stream_hash_equal": experiment_reorder[
            "stream_hash_equal"
        ],
        "donor_multiplier_stream_sha256": primary.multiplier_stream_sha256,
        "component_multiplier_stream_sha256": (
            result.experiment_overlap_sensitivity.multiplier_stream_sha256
            if result.experiment_overlap_sensitivity is not None
            else ""
        ),
        "shared_coordinate_draw_max_abs_error": shared_draw_error,
        "signed_auc_draw_max_abs_error": signed_draw_error,
        "pathway_chunk_max_abs_error": chunk_error,
        "worker_count_parameter_present": bool(worker_or_chunk_parameters),
        "worker_or_chunk_parameters_observed": worker_or_chunk_parameters,
        "serial_kernel_has_no_worker_dependent_branch": not bool(
            worker_or_chunk_parameters
        ),
        "multiplier_bank_regeneration_exact": bank_exact,
        "missing_na_exact": missing_na_exact,
        "missing_zero_payload_rejected": missing_zero_rejected,
        "rank_compression_maximum_leverage": float(np.nanmax(rank_fit.leverage)),
        "rank_compression_maximum_nuisance_rank_loss": int(
            rank_fit.design_audit["maximum_lodo_nuisance_rank_loss"]
        ),
        "fake_component_partition_rejected": fake_partition_rejected,
        "derived_component_count": int(fit.design_audit["experiment_component_count"]),
        "webb_sensitivity_present": webb_present,
    }
    checks = {
        CHECK_IDS[0]: lodo_error <= tolerance,
        CHECK_IDS[1]: full_ols_error <= tolerance,
        CHECK_IDS[2]: (
            influence_sum_error <= float(spec["influence_sum_tolerance"])
            and variance_error <= tolerance
        ),
        CHECK_IDS[3]: all(
            donor_reorder[key] <= tolerance
            for key in ("effect", "influence", "draw")
        )
        and donor_reorder["stream_hash_equal"],
        CHECK_IDS[4]: all(
            pathway_reorder[key] <= tolerance
            for key in ("effect", "influence", "draw")
        )
        and pathway_reorder["stream_hash_equal"],
        CHECK_IDS[5]: all(
            bin_reorder[key] <= tolerance
            for key in ("effect", "influence", "draw")
        )
        and bin_reorder["stream_hash_equal"],
        CHECK_IDS[6]: shared_draw_error <= tolerance,
        CHECK_IDS[7]: chunk_error <= tolerance and bank_exact,
        CHECK_IDS[9]: missing_na_exact and missing_zero_rejected,
        CHECK_IDS[10]: signed_draw_error <= tolerance,
        CHECK_IDS[13]: (
            float(np.nanmax(rank_fit.leverage))
            >= float(rank_spec["expected_full_maximum_leverage"]) - tolerance
            and int(rank_fit.design_audit["maximum_lodo_nuisance_rank_loss"])
            == int(rank_spec["expected_maximum_lodo_nuisance_rank_loss"])
            and all(
                experiment_reorder[key] <= tolerance
                for key in ("effect", "influence", "draw")
            )
            and experiment_reorder["stream_hash_equal"]
        ),
        CHECK_IDS[14]: webb_present,
    }
    return metrics, checks, result


def _run_heteroskedastic_suite(
    spec: Mapping[str, Any]
) -> tuple[dict[str, Any], bool]:
    n_donors = int(spec["donors"])
    n_pathways = int(spec["null_replicates_as_synthetic_pathways"])
    group_sizes = tuple(int(value) for value in spec["experiment_group_sizes"])
    _require_equal(sum(group_sizes), n_donors, "heteroskedastic donor count")
    experiment_index = np.concatenate(
        [np.full(size, index, dtype=int) for index, size in enumerate(group_sizes)]
    )
    fractions = np.zeros((n_donors, 1, len(group_sizes)), dtype=float)
    fractions[np.arange(n_donors), 0, experiment_index] = 1.0
    condition = np.asarray([index % 2 for index in range(n_donors)], dtype=np.uint8)
    rng = np.random.default_rng(int(spec["data_seed"]))
    nuisance = rng.normal(scale=0.6, size=(len(group_sizes), n_pathways))
    donor_sd = np.linspace(
        float(spec["donor_sd_min"]),
        float(spec["donor_sd_max"]),
        n_donors,
    )
    outcomes = (
        nuisance[experiment_index, None, :]
        + rng.normal(size=(n_donors, 1, n_pathways)) * donor_sd[:, None, None]
    )
    fit = fit_lodo_donor_influence(
        outcomes=outcomes,
        donor_ids=tuple(f"H_D{index:03d}" for index in range(n_donors)),
        bin_ids=("H_B00",),
        pathway_ids=tuple(f"NULL_REP{index:04d}" for index in range(n_pathways)),
        condition=condition,
        availability=np.ones((n_donors, 1), dtype=bool),
        experiment_fractions=fractions,
        experiment_ids=tuple(f"H_E{index:02d}" for index in range(len(group_sizes))),
        experiment_component_ids=None,
    )
    effect = fit.effect[0]
    standard_error = fit.standard_error[0]
    empirical_sd = float(np.std(effect, ddof=1))
    rms_standard_error = float(np.sqrt(np.mean(standard_error**2)))
    ratio = rms_standard_error / empirical_sd
    coverage = float(np.mean(np.abs(effect) <= 1.96 * standard_error))
    maximum_leverage = float(np.nanmax(fit.leverage))
    passed = bool(
        float(spec["rms_se_to_empirical_sd_min"])
        <= ratio
        <= float(spec["rms_se_to_empirical_sd_max"])
        and float(spec["normal_95_coverage_min"])
        <= coverage
        <= float(spec["normal_95_coverage_max"])
        and maximum_leverage >= float(spec["maximum_leverage_min"])
        and int(fit.design_audit["maximum_lodo_nuisance_rank_loss"]) >= 1
        and int(fit.design_audit["n_lodo_refits"]) == n_donors
    )
    metrics = {
        "fixture_outcome_sha256": _hash_array(outcomes),
        "null_replicate_count": n_pathways,
        "empirical_effect_sd": empirical_sd,
        "rms_lodo_standard_error": rms_standard_error,
        "rms_se_to_empirical_sd": ratio,
        "normal_95_interval_coverage": coverage,
        "maximum_observed_leverage": maximum_leverage,
        "maximum_lodo_nuisance_rank_loss": int(
            fit.design_audit["maximum_lodo_nuisance_rank_loss"]
        ),
        "hc3_used": False,
        "lodo_refit_count": int(fit.design_audit["n_lodo_refits"]),
    }
    return metrics, passed


def _balanced_mean_difference_lodo(
    outcomes: np.ndarray, treated_indices: Sequence[int]
) -> tuple[float, float, np.ndarray, float]:
    y = np.asarray(outcomes, dtype=float)
    n = len(y)
    treated = np.zeros(n, dtype=bool)
    treated[np.asarray(tuple(treated_indices), dtype=int)] = True
    effect = float(y[treated].mean() - y[~treated].mean())
    leave_one_out = np.empty(n, dtype=float)
    for donor in range(n):
        keep = np.arange(n) != donor
        leave_one_out[donor] = float(
            y[keep & treated].mean() - y[keep & ~treated].mean()
        )
    influence = math.sqrt((n - 1.0) / n) * (
        leave_one_out.mean() - leave_one_out
    )
    standard_error = float(np.sqrt(np.sum(influence**2)))
    return effect, standard_error, influence, effect / standard_error


def _run_oracle_suite(spec: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    n_donors = int(spec["donors"])
    treated_count = int(spec["treated"])
    outcomes = np.asarray(spec["fixed_outcomes"], dtype=float)
    observed_treated = tuple(int(value) for value in spec["observed_treated_indices"])
    combinations = tuple(itertools.combinations(range(n_donors), treated_count))
    _require_equal(
        len(combinations),
        int(spec["exact_balanced_label_count"]),
        "oracle label count",
    )
    observed_condition = np.zeros(n_donors, dtype=np.uint8)
    observed_condition[np.asarray(observed_treated, dtype=int)] = 1
    result = run_functional_core_v2(
        outcomes=outcomes[:, None, None],
        donor_ids=tuple(f"ORACLE_D{index:02d}" for index in range(n_donors)),
        bin_ids=("ORACLE_B0",),
        pathway_ids=("ORACLE_P0",),
        condition=observed_condition,
        availability=np.ones((n_donors, 1), dtype=bool),
        experiment_fractions=np.ones((n_donors, 1, 1), dtype=float),
        experiment_ids=("ORACLE_CONSTANT",),
        experiment_component_ids=tuple("ONLY_COMPONENT" for _ in range(n_donors)),
        n_multiplier_draws=int(spec["multiplier_draws"]),
        donor_multiplier_seed=int(spec["multiplier_seed"]),
        component_multiplier_seed=int(spec["multiplier_seed"]) + 1,
        alpha=0.05,
        min_donors_per_condition=int(spec["minimum_donors_per_condition"]),
    )
    effects = np.empty(len(combinations), dtype=float)
    standard_errors = np.empty(len(combinations), dtype=float)
    influences = np.empty((len(combinations), n_donors), dtype=float)
    statistics = np.empty(len(combinations), dtype=float)
    for index, treated in enumerate(combinations):
        effect, standard_error, influence, statistic = _balanced_mean_difference_lodo(
            outcomes, treated
        )
        effects[index] = effect
        standard_errors[index] = standard_error
        influences[index] = influence
        statistics[index] = statistic
    observed_index = combinations.index(observed_treated)
    core_effect_error = abs(float(result.fit.effect[0, 0]) - effects[observed_index])
    core_se_error = abs(
        float(result.fit.standard_error[0, 0]) - standard_errors[observed_index]
    )
    core_influence_error = _max_abs(
        result.fit.donor_influence[:, 0, 0] - influences[observed_index]
    )

    absolute_statistics = np.abs(statistics)
    exact_p = np.mean(
        absolute_statistics[:, None]
        >= absolute_statistics[None, :] - 1.0e-12,
        axis=0,
    )
    multiplier_bank = result.donor_primary.multipliers
    multiplier_statistics = (
        multiplier_bank @ influences.T / standard_errors[None, :]
    )
    multiplier_p = (
        1.0
        + np.sum(
            np.abs(multiplier_statistics)
            >= absolute_statistics[None, :] - 1.0e-12,
            axis=0,
        )
    ) / (len(multiplier_bank) + 1.0)
    exact_order = int(math.ceil(0.95 * len(combinations)))
    multiplier_order = int(
        math.ceil((len(multiplier_bank) + 1) * (1.0 - 0.05))
    )
    exact_q95 = float(np.sort(absolute_statistics)[exact_order - 1])
    observed_null = np.abs(multiplier_statistics[:, observed_index])
    multiplier_q95 = float(np.sort(observed_null)[multiplier_order - 1])
    q95_ratio = multiplier_q95 / exact_q95
    exact_reject = exact_p <= 0.05
    multiplier_reject = multiplier_p <= 0.05
    disagreement = float(np.mean(exact_reject != multiplier_reject))
    rejection_rate_difference = abs(
        float(np.mean(exact_reject)) - float(np.mean(multiplier_reject))
    )
    median_absolute_p_difference = float(np.median(np.abs(exact_p - multiplier_p)))
    observed_core_p_error = abs(
        float(result.donor_primary.curve_p_value[0])
        - float(multiplier_p[observed_index])
    )
    passed = bool(
        core_effect_error <= 1.0e-12
        and core_se_error <= 1.0e-12
        and core_influence_error <= 1.0e-12
        and observed_core_p_error <= 1.0e-12
        and float(spec["q95_ratio_min"])
        <= q95_ratio
        <= float(spec["q95_ratio_max"])
        and disagreement <= float(spec["rejection_disagreement_max"])
        and rejection_rate_difference
        <= float(spec["rejection_rate_absolute_difference_max"])
        and median_absolute_p_difference
        <= float(spec["median_absolute_p_difference_max"])
        and result.experiment_overlap_sensitivity is None
    )
    metrics = {
        "fixed_outcome_sha256": _hash_array(outcomes),
        "exact_balanced_label_count": len(combinations),
        "synthetic_oracle_labels_persisted": False,
        "core_effect_max_abs_error": core_effect_error,
        "core_standard_error_max_abs_error": core_se_error,
        "core_influence_max_abs_error": core_influence_error,
        "observed_core_p_max_abs_error": observed_core_p_error,
        "exact_randomization_q95": exact_q95,
        "multiplier_q95": multiplier_q95,
        "multiplier_to_exact_q95_ratio": q95_ratio,
        "exact_rejection_rate": float(np.mean(exact_reject)),
        "multiplier_rejection_rate": float(np.mean(multiplier_reject)),
        "rejection_decision_disagreement": disagreement,
        "rejection_rate_absolute_difference": rejection_rate_difference,
        "median_absolute_p_difference": median_absolute_p_difference,
        "one_component_sensitivity_unavailable": result.experiment_overlap_sensitivity is None,
        "donor_multiplier_stream_sha256": result.donor_primary.multiplier_stream_sha256,
    }
    return metrics, passed


def _plus_one(reference: np.ndarray, observed: np.ndarray) -> np.ndarray:
    null = np.asarray(reference, dtype=float)
    target = np.asarray(observed, dtype=float)
    return (
        1.0
        + np.sum(null >= target[None, ...] - 1.0e-12, axis=0)
    ) / (null.shape[0] + 1.0)


def _run_global_suite(spec: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    n_donors = int(spec["donors"])
    n_bins = int(spec["bins"])
    n_pathways = int(spec["pathways"])
    n_experiments = int(spec["experiments"])
    rng = np.random.default_rng(int(spec["data_seed"]))
    donor_ids = tuple(f"G_D{index:03d}" for index in range(n_donors))
    bin_ids = tuple(f"G_B{index:02d}" for index in range(n_bins))
    pathway_ids = tuple(f"G_P{index:02d}" for index in range(n_pathways))
    experiment_ids = tuple(f"G_E{index:02d}" for index in range(n_experiments))
    condition = np.asarray([index % 2 for index in range(n_donors)], dtype=np.uint8)
    component_sizes = tuple(
        int(value) for value in spec["experiment_component_donor_sizes"]
    )
    _require_equal(sum(component_sizes), n_donors, "global component donor count")
    _require_equal(len(component_sizes), n_experiments, "global experiment count")
    experiment_index = np.concatenate(
        [
            np.full(size, index, dtype=int)
            for index, size in enumerate(component_sizes)
        ]
    )
    fractions = np.zeros((n_donors, n_bins, n_experiments), dtype=float)
    fractions[
        np.arange(n_donors)[:, None],
        np.arange(n_bins)[None, :],
        experiment_index[:, None],
    ] = 1.0
    nuisance = rng.normal(scale=0.55, size=(n_experiments, n_pathways))
    donor_latent = rng.normal(size=(n_donors, 3))
    bin_loading = rng.normal(scale=0.45, size=(n_bins, 3))
    pathway_loading = rng.normal(scale=0.45, size=(n_pathways, 3))
    correlated = np.einsum(
        "dk,bk,pk->dbp",
        donor_latent,
        bin_loading,
        pathway_loading,
        optimize=True,
    )
    donor_scale = np.linspace(0.4, 1.4, n_donors)
    noise = rng.normal(size=(n_donors, n_bins, n_pathways)) * donor_scale[:, None, None]
    curve_effect = 0.12 * np.sin(
        np.linspace(0.0, 2.0 * math.pi, n_bins, endpoint=False)
    )[:, None] * np.where(np.arange(n_pathways) % 2 == 0, 1.0, -1.0)[None, :]
    outcomes = (
        nuisance[experiment_index, None, :]
        + 0.35 * correlated
        + noise
        + condition[:, None, None] * curve_effect[None, :, :]
    )
    family_ids = tuple(
        f"G_F{index % int(spec['families']):02d}" for index in range(n_pathways)
    )
    result = run_functional_core_v2(
        outcomes=outcomes,
        donor_ids=donor_ids,
        bin_ids=bin_ids,
        pathway_ids=pathway_ids,
        condition=condition,
        availability=np.ones((n_donors, n_bins), dtype=bool),
        experiment_fractions=fractions,
        experiment_ids=experiment_ids,
        experiment_component_ids=None,
        family_ids=family_ids,
        bin_weights=np.full(n_bins, 1.0 / n_bins),
        n_multiplier_draws=int(spec["multiplier_draws"]),
        donor_multiplier_seed=int(spec["donor_multiplier_seed"]),
        component_multiplier_seed=int(spec["component_multiplier_seed"]),
        alpha=0.05,
    )
    fit = result.fit
    primary = result.donor_primary
    tolerance = float(spec["numeric_tolerance"])
    safe_influence = np.where(np.isfinite(fit.donor_influence), fit.donor_influence, 0.0)
    numerator = np.einsum(
        "rd,dbp->rbp", primary.multipliers, safe_influence, optimize=True
    )
    manual_draws = numerator / fit.standard_error[None, :, :]
    coordinate_draw_error = _max_abs(manual_draws - primary.studentized_draws)
    global_null = np.max(np.abs(manual_draws), axis=(1, 2))
    global_null_error = _max_abs(
        global_null - np.nanmax(np.abs(primary.studentized_draws), axis=(1, 2))
    )
    order = int(math.ceil((len(global_null) + 1) * 0.95))
    critical = float(np.sort(global_null)[order - 1])
    critical_error = abs(critical - float(primary.simultaneous_critical))
    curve_observed = np.max(np.abs(fit.studentized_effect), axis=0)
    curve_null = np.max(np.abs(manual_draws), axis=1)
    curve_p_error = _max_abs(
        _plus_one(curve_null, curve_observed) - primary.curve_p_value
    )
    global_p_error = _max_abs(
        _plus_one(global_null[:, None], curve_observed)
        - primary.global_curve_maxT_p_value
    )
    family_order = tuple(dict.fromkeys(fit.family_ids))
    family_observed = []
    family_null = []
    for family in family_order:
        members = np.asarray([value == family for value in fit.family_ids])
        family_observed.append(float(np.max(curve_observed[members])))
        family_null.append(np.max(curve_null[:, members], axis=1))
    family_null_array = np.asarray(family_null).T
    family_p_error = _max_abs(
        _plus_one(family_null_array, np.asarray(family_observed))
        - primary.family_maxT_p_value
    )
    signed_numerator = np.einsum(
        "rbp,bp->rp", numerator, fit.pathway_bin_weights, optimize=True
    )
    signed_draw_error = _max_abs(
        signed_numerator
        - primary.signed_auc_studentized_draws
        * primary.signed_auc_standard_error[None, :]
    )
    signed_null = np.abs(primary.signed_auc_studentized_draws)
    signed_observed = np.abs(primary.signed_auc_studentized)
    signed_p_error = _max_abs(
        _plus_one(signed_null, signed_observed) - primary.signed_auc_p_value
    )
    signed_global_null = np.max(signed_null, axis=1)
    signed_global_p_error = _max_abs(
        _plus_one(signed_global_null[:, None], signed_observed)
        - primary.signed_auc_global_maxT_p_value
    )
    pathway_criticals = np.sort(curve_null, axis=0)[order - 1]
    family_criticals = np.sort(family_null_array, axis=0)[order - 1]
    critical_dominates_subsets = bool(
        critical + tolerance >= float(np.max(pathway_criticals))
        and critical + tolerance >= float(np.max(family_criticals))
    )
    lower_error = _max_abs(
        primary.simultaneous_lower
        - (fit.effect - critical * fit.standard_error)
    )
    upper_error = _max_abs(
        primary.simultaneous_upper
        - (fit.effect + critical * fit.standard_error)
    )
    support_coordinates = int(fit.support_mask.sum())
    sensitivity = result.experiment_overlap_sensitivity
    passed = bool(
        support_coordinates == int(spec["supported_coordinates"])
        and primary.simultaneous_order_index_1based == 950
        and coordinate_draw_error <= tolerance
        and global_null_error <= tolerance
        and critical_error <= tolerance
        and curve_p_error <= tolerance
        and global_p_error <= tolerance
        and family_p_error <= tolerance
        and signed_draw_error <= tolerance
        and signed_p_error <= tolerance
        and signed_global_p_error <= tolerance
        and lower_error <= tolerance
        and upper_error <= tolerance
        and critical_dominates_subsets
        and sensitivity is not None
        and sensitivity.distribution == "webb_six_point"
        and len(sensitivity.unit_ids) == int(spec["expected_component_count"])
        and fit.design_audit["experiment_component_donor_sizes_descending"]
        == sorted(
            [int(value) for value in spec["experiment_component_donor_sizes"]],
            reverse=True,
        )
        and sensitivity.finite_sample_scale
        == math.sqrt(
            int(spec["expected_component_count"])
            / (int(spec["expected_component_count"]) - 1.0)
        )
        and sensitivity.sensitivity_informative
        is bool(spec["expected_component_sensitivity_informative"])
    )
    metrics = {
        "fixture_outcome_sha256": _hash_array(outcomes),
        "supported_coordinate_count": support_coordinates,
        "global_null_manual_max_abs_error": global_null_error,
        "coordinate_draw_manual_max_abs_error": coordinate_draw_error,
        "signed_auc_draw_manual_max_abs_error": signed_draw_error,
        "curve_p_manual_max_abs_error": curve_p_error,
        "global_curve_p_manual_max_abs_error": global_p_error,
        "family_p_manual_max_abs_error": family_p_error,
        "signed_auc_p_manual_max_abs_error": signed_p_error,
        "signed_auc_global_p_manual_max_abs_error": signed_global_p_error,
        "simultaneous_order_index_1based": int(primary.simultaneous_order_index_1based),
        "simultaneous_critical": float(primary.simultaneous_critical),
        "simultaneous_critical_manual_abs_error": critical_error,
        "simultaneous_lower_manual_max_abs_error": lower_error,
        "simultaneous_upper_manual_max_abs_error": upper_error,
        "global_critical_dominates_all_pathway_and_family_criticals": critical_dominates_subsets,
        "family_count": len(family_order),
        "component_sensitivity_present": sensitivity is not None,
        "donor_multiplier_stream_sha256": primary.multiplier_stream_sha256,
        "component_multiplier_stream_sha256": (
            sensitivity.multiplier_stream_sha256 if sensitivity is not None else ""
        ),
        "component_count": len(sensitivity.unit_ids) if sensitivity is not None else 0,
        "component_donor_sizes_descending": fit.design_audit[
            "experiment_component_donor_sizes_descending"
        ],
        "component_finite_sample_scale": (
            float(sensitivity.finite_sample_scale) if sensitivity is not None else 0.0
        ),
        "component_sensitivity_informative": (
            bool(sensitivity.sensitivity_informative) if sensitivity is not None else False
        ),
    }
    return metrics, passed


def _walk_keys(value: Any):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key).lower()
            yield from _walk_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_keys(item)


def _call_name(node: ast.Call) -> str:
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return ""


def _runtime_input_read_whitelist(config: Mapping[str, Any]) -> list[str]:
    paths = [CONFIG_FILE]
    paths.extend(
        str(record["relative_path"])
        for record in config["authorization_bindings"].values()
    )
    paths.extend(
        str(record["relative_path"])
        for record in config["implementation_bindings"].values()
    )
    return sorted(set(paths))


def _run_firewall_suite(
    *, root: Path, config: Mapping[str, Any], algebraic_result: Any
) -> tuple[dict[str, Any], bool]:
    core_path = _repo_file(root, config["implementation_contract"]["core_module"])
    source = core_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    v1_imports: list[str] = []
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported_modules.append(str(node.module))
            if node.module == "trajpathmix_functional_core_v1":
                v1_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
    allowed_v1 = set(config["permanent_retirement_firewall"]["only_allowed_v1_imports"])
    observed_v1 = set(v1_imports)
    forbidden_calls = set(
        config["permanent_retirement_firewall"]["forbidden_core_calls"]
    )
    observed_calls = {
        _call_name(node) for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    forbidden_modules = tuple(
        module for module in imported_modules if "trajpathmix_corebench_cb2" in module
    )
    result_keys = (
        set(_walk_keys(algebraic_result.to_dict()))
        if algebraic_result is not None
        else set()
    )
    algebraic_result_present = algebraic_result is not None
    result_claim_scope_checked = bool(
        algebraic_result_present
        and algebraic_result.claim_scope["formal_inference_authorized"] is False
        and algebraic_result.claim_scope["holdout_500_authorized"] is False
    )
    forbidden_result_fields = set(
        config["permanent_retirement_firewall"]["forbidden_result_fields"]
    )
    authorization = _strict_json_load(
        _repo_file(
            root,
            config["authorization_bindings"]["v2_rebuild_authorization"][
                "relative_path"
            ],
        )
    )
    holdout_closed = bool(
        not authorization["rebuild_authorization"][
            "v2_1_new_holdout_assignment_bank_materialization_authorized"
        ]
        and not authorization["rebuild_authorization"][
            "v2_1_holdout_500_execution_authorized"
        ]
        and not authorization["claim_state"]["functional_core_v2_calibrated"]
    )
    read_whitelist = _runtime_input_read_whitelist(config)
    read_whitelist_clean = not any(
        "trajpathmix_corebench_cb2_500" in path.lower()
        or "assignment_axis" in path.lower()
        or "pseudobulk" in path.lower()
        or "pathway_score" in path.lower()
        for path in read_whitelist
    )
    passed = bool(
        observed_v1 == allowed_v1
        and not (forbidden_calls & observed_calls)
        and not forbidden_modules
        and not ({"__import__", "import_module"} & observed_calls)
        and not (forbidden_result_fields & result_keys)
        and holdout_closed
        and read_whitelist_clean
        and algebraic_result_present
        and result_claim_scope_checked
    )
    metrics = {
        "core_source_sha256": _hash_file(core_path),
        "observed_v1_imports": sorted(observed_v1),
        "allowed_v1_imports_exact": observed_v1 == allowed_v1,
        "forbidden_core_calls_observed": sorted(forbidden_calls & observed_calls),
        "forbidden_cb2_modules_imported": list(forbidden_modules),
        "dynamic_import_calls_observed": sorted(
            {"__import__", "import_module"} & observed_calls
        ),
        "forbidden_result_fields_observed": sorted(
            forbidden_result_fields & result_keys
        ),
        "bound_holdout_bank_and_execution_closed": holdout_closed,
        "runtime_input_read_whitelist": read_whitelist,
        "runtime_input_read_whitelist_clean": read_whitelist_clean,
        "algebraic_result_present": algebraic_result_present,
        "result_claim_scope_checked": result_claim_scope_checked,
        "formal_inference_authorized": False,
        "timing_computed": False,
        "by_used_for_acceptance": False,
        "nonstudentized_l1_used": False,
    }
    return metrics, passed


def _fallacy_scan() -> list[dict[str, Any]]:
    return [
        {
            "fallacy_id": "simpsons_paradox",
            "status": "not_detected",
            "rationale": "synthetic suites are reported separately and no pooled result rescues a failed suite",
            "affected_claim": "suite-specific V2-0 correctness",
        },
        {
            "fallacy_id": "ecological_fallacy",
            "status": "caution",
            "rationale": "the inferential unit is donor and no cell-level or gene-level individual effect is claimed",
            "affected_claim": "donor-level interpretation only",
        },
        {
            "fallacy_id": "berksons_paradox",
            "status": "caution",
            "rationale": "synthetic validation cannot remove cohort-selection limitations from future real-data use",
            "affected_claim": "future real-cohort generalization remains closed",
        },
        {
            "fallacy_id": "collider_bias",
            "status": "caution",
            "rationale": "availability and experiment composition may be selection structures so causal interpretation remains closed",
            "affected_claim": "causal interpretation remains closed",
        },
        {
            "fallacy_id": "base_rate_neglect",
            "status": "not_detected",
            "rationale": "all oracle labels and null-replicate denominators are fixed and reported",
            "affected_claim": "reported synthetic denominators",
        },
        {
            "fallacy_id": "regression_to_the_mean",
            "status": "not_detected",
            "rationale": "fixtures are generated once from frozen seeds without selecting extreme replicates",
            "affected_claim": "synthetic scale diagnostics",
        },
        {
            "fallacy_id": "survivorship_bias",
            "status": "not_detected",
            "rationale": "no failed synthetic coordinate or label is removed from a denominator",
            "affected_claim": "all frozen checks",
        },
        {
            "fallacy_id": "look_elsewhere_effect",
            "status": "not_detected",
            "rationale": "the global fixture contains the predeclared 50 by 20 scope and frozen families",
            "affected_claim": "50 by 20 multiplicity implementation",
        },
        {
            "fallacy_id": "garden_of_forking_paths",
            "status": "not_detected",
            "rationale": "seeds thresholds code and exact outputs are frozen before materialization and retries are forbidden",
            "affected_claim": "one-shot V2-0 validation",
        },
        {
            "fallacy_id": "correlation_not_causation",
            "status": "not_detected",
            "rationale": "V2-0 addresses implementation correctness only and makes no biological or causal claim",
            "affected_claim": "no biological or causal inference",
        },
        {
            "fallacy_id": "reverse_causality",
            "status": "not_detected",
            "rationale": "no real condition contrast timing endpoint or directional biology enters V2-0",
            "affected_claim": "no directional biological interpretation",
        },
    ]


def _reduce_suite_metrics_to_checks(
    suite_metrics: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, bool]:
    checks = {check_id: False for check_id in CHECK_IDS}
    suites = config["synthetic_suites"]
    algebra = suite_metrics["algebraic_and_invariance"]
    heteroskedastic = suite_metrics["heteroskedastic_high_leverage_null"]
    oracle = suite_metrics["balanced_label_oracle"]
    global_metrics = suite_metrics["global_50_x_20"]
    firewall = suite_metrics["source_and_claim_firewall"]
    algebra_ok = not bool(algebra.get("execution_exception", False))
    if algebra_ok:
        tolerance = float(suites["algebraic_and_invariance"]["numeric_tolerance"])
        influence_tolerance = float(
            suites["algebraic_and_invariance"]["influence_sum_tolerance"]
        )
        checks[CHECK_IDS[0]] = algebra["all_deletion_lodo_max_abs_error"] <= tolerance
        checks[CHECK_IDS[1]] = algebra["full_ols_max_abs_error"] <= tolerance
        checks[CHECK_IDS[2]] = bool(
            algebra["influence_sum_max_abs_error"] <= influence_tolerance
            and algebra["jackknife_variance_max_abs_error"] <= tolerance
        )
        checks[CHECK_IDS[3]] = bool(
            algebra["donor_reorder_effect_max_abs_error"] <= tolerance
            and algebra["donor_reorder_influence_max_abs_error"] <= tolerance
            and algebra["donor_reorder_draw_max_abs_error"] <= tolerance
            and algebra["donor_reorder_stream_hash_equal"]
        )
        checks[CHECK_IDS[4]] = bool(
            algebra["pathway_reorder_effect_max_abs_error"] <= tolerance
            and algebra["pathway_reorder_influence_max_abs_error"] <= tolerance
            and algebra["pathway_reorder_draw_max_abs_error"] <= tolerance
            and algebra["pathway_reorder_stream_hash_equal"]
        )
        checks[CHECK_IDS[5]] = bool(
            algebra["bin_reorder_effect_max_abs_error"] <= tolerance
            and algebra["bin_reorder_influence_max_abs_error"] <= tolerance
            and algebra["bin_reorder_draw_max_abs_error"] <= tolerance
            and algebra["bin_reorder_stream_hash_equal"]
        )
        checks[CHECK_IDS[6]] = algebra["shared_coordinate_draw_max_abs_error"] <= tolerance
        checks[CHECK_IDS[7]] = bool(
            algebra["pathway_chunk_max_abs_error"] <= tolerance
            and algebra["multiplier_bank_regeneration_exact"]
            and algebra["worker_count_parameter_present"] is False
            and algebra["serial_kernel_has_no_worker_dependent_branch"] is True
        )
        checks[CHECK_IDS[9]] = bool(
            algebra["missing_na_exact"] and algebra["missing_zero_payload_rejected"]
        )
        checks[CHECK_IDS[10]] = algebra["signed_auc_draw_max_abs_error"] <= tolerance
        checks[CHECK_IDS[13]] = bool(
            algebra["rank_compression_maximum_leverage"]
            >= float(
                suites["algebraic_and_invariance"]["rank_compression_fixture"][
                    "expected_full_maximum_leverage"
                ]
            )
            - tolerance
            and algebra["rank_compression_maximum_nuisance_rank_loss"]
            == int(
                suites["algebraic_and_invariance"]["rank_compression_fixture"][
                    "expected_maximum_lodo_nuisance_rank_loss"
                ]
            )
            and algebra["experiment_reorder_effect_max_abs_error"] <= tolerance
            and algebra["experiment_reorder_influence_max_abs_error"] <= tolerance
            and algebra["experiment_reorder_draw_max_abs_error"] <= tolerance
            and algebra["experiment_reorder_stream_hash_equal"]
        )
        checks[CHECK_IDS[14]] = bool(
            algebra["webb_sensitivity_present"]
            and algebra["fake_component_partition_rejected"]
        )
    if not bool(heteroskedastic.get("execution_exception", False)):
        spec = suites["heteroskedastic_high_leverage_null"]
        checks[CHECK_IDS[8]] = bool(
            float(spec["rms_se_to_empirical_sd_min"])
            <= heteroskedastic["rms_se_to_empirical_sd"]
            <= float(spec["rms_se_to_empirical_sd_max"])
            and float(spec["normal_95_coverage_min"])
            <= heteroskedastic["normal_95_interval_coverage"]
            <= float(spec["normal_95_coverage_max"])
            and heteroskedastic["maximum_observed_leverage"]
            >= float(spec["maximum_leverage_min"])
            and heteroskedastic["maximum_lodo_nuisance_rank_loss"] >= 1
            and heteroskedastic["lodo_refit_count"] == int(spec["donors"])
            and heteroskedastic["hc3_used"] is False
        )
    if not bool(oracle.get("execution_exception", False)):
        spec = suites["balanced_label_oracle"]
        checks[CHECK_IDS[12]] = bool(
            oracle["exact_balanced_label_count"]
            == int(spec["exact_balanced_label_count"])
            and oracle["synthetic_oracle_labels_persisted"] is False
            and oracle["core_effect_max_abs_error"] <= 1.0e-12
            and oracle["core_standard_error_max_abs_error"] <= 1.0e-12
            and oracle["core_influence_max_abs_error"] <= 1.0e-12
            and oracle["observed_core_p_max_abs_error"] <= 1.0e-12
            and float(spec["q95_ratio_min"])
            <= oracle["multiplier_to_exact_q95_ratio"]
            <= float(spec["q95_ratio_max"])
            and oracle["rejection_decision_disagreement"]
            <= float(spec["rejection_disagreement_max"])
            and oracle["rejection_rate_absolute_difference"]
            <= float(spec["rejection_rate_absolute_difference_max"])
            and oracle["median_absolute_p_difference"]
            <= float(spec["median_absolute_p_difference_max"])
            and oracle["one_component_sensitivity_unavailable"] is True
        )
    if not bool(global_metrics.get("execution_exception", False)):
        spec = suites["global_50_x_20"]
        tolerance = float(spec["numeric_tolerance"])
        checks[CHECK_IDS[11]] = bool(
            global_metrics["supported_coordinate_count"]
            == int(spec["supported_coordinates"])
            and global_metrics["simultaneous_order_index_1based"] == 950
            and global_metrics["global_null_manual_max_abs_error"] <= tolerance
            and global_metrics["coordinate_draw_manual_max_abs_error"] <= tolerance
            and global_metrics["signed_auc_draw_manual_max_abs_error"] <= tolerance
            and global_metrics["curve_p_manual_max_abs_error"] <= tolerance
            and global_metrics["global_curve_p_manual_max_abs_error"] <= tolerance
            and global_metrics["family_p_manual_max_abs_error"] <= tolerance
            and global_metrics["signed_auc_p_manual_max_abs_error"] <= tolerance
            and global_metrics["signed_auc_global_p_manual_max_abs_error"] <= tolerance
            and global_metrics["simultaneous_critical_manual_abs_error"] <= tolerance
            and global_metrics["simultaneous_lower_manual_max_abs_error"] <= tolerance
            and global_metrics["simultaneous_upper_manual_max_abs_error"] <= tolerance
            and global_metrics[
                "global_critical_dominates_all_pathway_and_family_criticals"
            ]
            and global_metrics["family_count"] == int(spec["families"])
        )
        checks[CHECK_IDS[10]] = bool(
            checks[CHECK_IDS[10]]
            and global_metrics["signed_auc_draw_manual_max_abs_error"] <= tolerance
        )
        checks[CHECK_IDS[14]] = bool(
            checks[CHECK_IDS[14]]
            and global_metrics["component_sensitivity_present"]
            and global_metrics["component_count"] == 7
            and global_metrics["component_donor_sizes_descending"]
            == [39, 12, 6, 6, 6, 4, 2]
            and global_metrics["component_finite_sample_scale"]
            == math.sqrt(7.0 / 6.0)
            and global_metrics["component_sensitivity_informative"] is False
        )
    else:
        checks[CHECK_IDS[10]] = False
        checks[CHECK_IDS[14]] = False
    if not bool(firewall.get("execution_exception", False)):
        checks[CHECK_IDS[15]] = bool(
            firewall["allowed_v1_imports_exact"]
            and not firewall["forbidden_core_calls_observed"]
            and not firewall["forbidden_cb2_modules_imported"]
            and not firewall["dynamic_import_calls_observed"]
            and not firewall["forbidden_result_fields_observed"]
            and firewall["bound_holdout_bank_and_execution_closed"]
            and firewall["runtime_input_read_whitelist_clean"]
            and firewall["algebraic_result_present"]
            and firewall["result_claim_scope_checked"]
            and firewall["formal_inference_authorized"] is False
            and firewall["timing_computed"] is False
            and firewall["by_used_for_acceptance"] is False
            and firewall["nonstudentized_l1_used"] is False
        )
    return checks


def run_v2_0_implementation_checks(
    *, repository_root: str | Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    binding_audit = verify_v2_0_authorization_bindings(root, config)
    implementation_binding_audit = verify_v2_0_implementation_bindings(root, config)
    suites = config["synthetic_suites"]
    check_pass = {check_id: False for check_id in CHECK_IDS}
    suite_errors: dict[str, dict[str, str]] = {}
    algebra_result = None
    try:
        algebra_metrics, algebra_checks, algebra_result = _run_algebraic_suite(
            suites["algebraic_and_invariance"]
        )
        check_pass.update(algebra_checks)
    except Exception as exc:  # formal suite failures must remain evidence, not retries
        algebra_metrics = {
            "execution_exception": True,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        suite_errors["algebraic_and_invariance"] = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    try:
        heteroskedastic_metrics, heteroskedastic_pass = _run_heteroskedastic_suite(
            suites["heteroskedastic_high_leverage_null"]
        )
        check_pass[CHECK_IDS[8]] = heteroskedastic_pass
    except Exception as exc:
        heteroskedastic_metrics = {
            "execution_exception": True,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        suite_errors["heteroskedastic_high_leverage_null"] = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    try:
        oracle_metrics, oracle_pass = _run_oracle_suite(
            suites["balanced_label_oracle"]
        )
        check_pass[CHECK_IDS[12]] = oracle_pass
    except Exception as exc:
        oracle_metrics = {
            "execution_exception": True,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        suite_errors["balanced_label_oracle"] = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    try:
        global_metrics, global_pass = _run_global_suite(suites["global_50_x_20"])
        check_pass[CHECK_IDS[11]] = global_pass
        global_component_pass = bool(
            global_metrics["component_count"] == 7
            and global_metrics["component_donor_sizes_descending"]
            == [39, 12, 6, 6, 6, 4, 2]
            and global_metrics["component_finite_sample_scale"]
            == math.sqrt(7.0 / 6.0)
            and global_metrics["component_sensitivity_informative"] is False
        )
        check_pass[CHECK_IDS[14]] = bool(
            check_pass[CHECK_IDS[14]] and global_component_pass
        )
    except Exception as exc:
        global_metrics = {
            "execution_exception": True,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        suite_errors["global_50_x_20"] = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        check_pass[CHECK_IDS[14]] = False
    try:
        firewall_metrics, firewall_pass = _run_firewall_suite(
            root=root,
            config=config,
            algebraic_result=algebra_result,
        )
        check_pass[CHECK_IDS[15]] = firewall_pass
    except Exception as exc:
        firewall_metrics = {
            "execution_exception": True,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        suite_errors["source_and_claim_firewall"] = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    suite_metrics = {
        "algebraic_and_invariance": algebra_metrics,
        "heteroskedastic_high_leverage_null": heteroskedastic_metrics,
        "balanced_label_oracle": oracle_metrics,
        "global_50_x_20": global_metrics,
        "source_and_claim_firewall": firewall_metrics,
    }
    check_pass = _reduce_suite_metrics_to_checks(suite_metrics, config)
    _require_equal(tuple(check_pass), CHECK_IDS, "executed check ids")

    suite_for_check = {
        CHECK_IDS[0]: "algebraic_and_invariance",
        CHECK_IDS[1]: "algebraic_and_invariance",
        CHECK_IDS[2]: "algebraic_and_invariance",
        CHECK_IDS[3]: "algebraic_and_invariance",
        CHECK_IDS[4]: "algebraic_and_invariance",
        CHECK_IDS[5]: "algebraic_and_invariance",
        CHECK_IDS[6]: "algebraic_and_invariance",
        CHECK_IDS[7]: "algebraic_and_invariance",
        CHECK_IDS[8]: "heteroskedastic_high_leverage_null",
        CHECK_IDS[9]: "algebraic_and_invariance",
        CHECK_IDS[10]: "algebraic_and_invariance_and_global_50_x_20",
        CHECK_IDS[11]: "global_50_x_20",
        CHECK_IDS[12]: "balanced_label_oracle",
        CHECK_IDS[13]: "algebraic_rank_compression",
        CHECK_IDS[14]: "algebraic_and_global_component_sensitivity",
        CHECK_IDS[15]: "source_and_claim_firewall",
    }
    suite_keys_for_check = {
        CHECK_IDS[0]: ("algebraic_and_invariance",),
        CHECK_IDS[1]: ("algebraic_and_invariance",),
        CHECK_IDS[2]: ("algebraic_and_invariance",),
        CHECK_IDS[3]: ("algebraic_and_invariance",),
        CHECK_IDS[4]: ("algebraic_and_invariance",),
        CHECK_IDS[5]: ("algebraic_and_invariance",),
        CHECK_IDS[6]: ("algebraic_and_invariance",),
        CHECK_IDS[7]: ("algebraic_and_invariance",),
        CHECK_IDS[8]: ("heteroskedastic_high_leverage_null",),
        CHECK_IDS[9]: ("algebraic_and_invariance",),
        CHECK_IDS[10]: ("algebraic_and_invariance", "global_50_x_20"),
        CHECK_IDS[11]: ("global_50_x_20",),
        CHECK_IDS[12]: ("balanced_label_oracle",),
        CHECK_IDS[13]: ("algebraic_and_invariance",),
        CHECK_IDS[14]: ("algebraic_and_invariance", "global_50_x_20"),
        CHECK_IDS[15]: ("algebraic_and_invariance", "source_and_claim_firewall"),
    }
    acceptance_rule_for_check = {
        CHECK_IDS[0]: "all direct deletion coefficients differ by at most 1e-10",
        CHECK_IDS[1]: "all full OLS coefficients differ by at most 1e-10",
        CHECK_IDS[2]: "influence sums at most 1e-12 and variance error at most 1e-10",
        CHECK_IDS[3]: "canonical donor reorder errors at most 1e-10 and stream hash exact",
        CHECK_IDS[4]: "canonical pathway reorder errors at most 1e-10",
        CHECK_IDS[5]: "canonical bin reorder errors at most 1e-10",
        CHECK_IDS[6]: "manual shared-donor matrix product error at most 1e-10",
        CHECK_IDS[7]: "frozen chunks reproduce full result and multiplier bank exactly",
        CHECK_IDS[8]: "predeclared SE ratio coverage leverage rank-loss and denominator gates all hold",
        CHECK_IDS[9]: "unavailable values remain NA and zero payload is rejected",
        CHECK_IDS[10]: "signed-AUC shared cross-bin influence identity error at most 1e-10",
        CHECK_IDS[11]: "exactly 1000 coordinates order 950 and all manual maxT/band identities pass",
        CHECK_IDS[12]: "all predeclared exact-label q95 p-value and rejection-difference gates pass",
        CHECK_IDS[13]: "maximum leverage equals one and deletion nuisance rank loss equals one",
        CHECK_IDS[14]: "derived fake-partition rejection plus G=7 Webb sqrt(7/6) noninformative sensitivity",
        CHECK_IDS[15]: "AST runtime-key authorization and exact-read firewalls all pass",
    }

    def failure_detail(check_id: str) -> str:
        if check_pass[check_id]:
            return ""
        messages = [
            f"{suite}: {record['error_type']}: {record['error_message']}"
            for suite, record in suite_errors.items()
            if suite in suite_keys_for_check[check_id]
        ]
        return " | ".join(messages) if messages else "predeclared acceptance rule not met"

    check_records = [
        {
            "check_id": check_id,
            "status": "PASS" if check_pass[check_id] else "FAIL",
            "hard_gate": True,
            "evidence_kind": suite_for_check[check_id],
            "observed": {
                "suite": suite_for_check[check_id],
                "suite_keys": list(suite_keys_for_check[check_id]),
                "suite_exception": any(
                    suite in suite_errors for suite in suite_keys_for_check[check_id]
                ),
            },
            "acceptance_rule": acceptance_rule_for_check[check_id],
            "failure_detail": failure_detail(check_id),
        }
        for check_id in CHECK_IDS
    ]
    scan = _fallacy_scan()
    _require_equal(tuple(item["fallacy_id"] for item in scan), FALLACY_IDS, "fallacy scan")
    allowed_fallacy_statuses = set(
        config["fallacy_scan_contract"]["allowed_statuses"]
    )
    _require(
        all(
            item["status"] in allowed_fallacy_statuses
            and bool(item.get("rationale"))
            and bool(item.get("affected_claim"))
            for item in scan
        ),
        "Fallacy scan records must have an allowed status, rationale, and affected claim",
    )
    unresolved_red_flags = sum(item["status"] == "red_flag" for item in scan)
    all_checks_pass = all(check_pass.values())
    overall_pass = bool(all_checks_pass and unresolved_red_flags == 0)
    decision = {
        "schema_name": "trajpathmix_functional_core_v2_0_implementation_decision",
        "schema_version": SCHEMA_VERSION,
        "project_id": PROJECT_ID,
        "frozen_at_utc": config["frozen_at_utc"],
        "frozen_payload_sha256": FROZEN_PAYLOAD_SHA256,
        "decision": "CONFORMS" if overall_pass else "STOP",
        "v2_0_conforms_to_all_16_frozen_checks": overall_pass,
        "implementation_source_bindings_frozen": True,
        "all_16_required_checks_pass": all_checks_pass,
        "required_check_count": len(CHECK_IDS),
        "passed_check_count": int(sum(check_pass.values())),
        "check_results": check_records,
        "suite_metrics": suite_metrics,
        "fallacy_scan": scan,
        "fallacy_scan_coverage": "11/11",
        "unresolved_red_flag_count": unresolved_red_flags,
        "caution_count": int(sum(item["status"] == "caution" for item in scan)),
        "authorization_binding_audit": binding_audit,
        "implementation_binding_audit": implementation_binding_audit,
        "suite_exception_count": len(suite_errors),
        "suite_exceptions": suite_errors,
        "execution_firewall": deepcopy(config["execution_firewall"]),
        "runtime_input_read_whitelist": _runtime_input_read_whitelist(config),
        "claim_ceiling": deepcopy(config["claim_ceiling"]),
        "functional_core_v2_calibrated": False,
        "new_holdout_assignment_bank_materialization_authorized": False,
        "new_holdout_500_execution_authorized": False,
        "next_stage_authorized": "none",
        "failure_action_if_any_check_failed": config["required_checks"][
            "hard_failure_action"
        ],
    }
    return decision


def _sanitize_tsv_cell(value: Any) -> str:
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _checks_tsv(decision: Mapping[str, Any]) -> str:
    columns = (
        "check_id",
        "status",
        "evidence_kind",
        "observed_json",
        "acceptance_rule",
        "failure_detail",
    )
    rows = ["\t".join(columns)]
    for record in decision["check_results"]:
        rows.append(
            "\t".join(
                (
                    _sanitize_tsv_cell(record["check_id"]),
                    _sanitize_tsv_cell(record["status"]),
                    _sanitize_tsv_cell(record["evidence_kind"]),
                    _sanitize_tsv_cell(_canonical_json(record["observed"])),
                    _sanitize_tsv_cell(record["acceptance_rule"]),
                    _sanitize_tsv_cell(record["failure_detail"]),
                )
            )
        )
    return "\n".join(rows) + "\n"


def _write_tsv(value: str, path: Path) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _passport_payload(
    config: Mapping[str, Any], decision: Mapping[str, Any]
) -> dict[str, Any]:
    passport = deepcopy(config["material_passport"])
    passport.update(
        {
            "schema_name": "trajpathmix_functional_core_v2_0_material_passport",
            "schema_version": SCHEMA_VERSION,
            "project_id": PROJECT_ID,
            "frozen_payload_sha256": FROZEN_PAYLOAD_SHA256,
            "verification_status": "ANALYZED",
            "decision": decision["decision"],
            "v2_0_conforms_to_all_16_frozen_checks": decision[
                "v2_0_conforms_to_all_16_frozen_checks"
            ],
            "required_check_count": 16,
            "passed_check_count": decision["passed_check_count"],
            "fallacy_scan_coverage": "11/11",
            "unresolved_red_flag_count": decision["unresolved_red_flag_count"],
            "pure_synthetic_only": True,
            "real_expression_read": False,
            "real_pathway_scores_read": False,
            "real_condition_labels_read": False,
            "current_cb2_500_assignment_bank_read": False,
            "current_cb2_500_outputs_or_cache_read": False,
            "new_holdout_assignment_bank_generated": False,
            "future_holdout_multiplier_stream_generated": False,
            "timing_computed": False,
            "biological_interpretation_performed": False,
            "functional_core_v2_calibrated": False,
            "formal_inference_authorized": False,
            "new_holdout_500_execution_authorized": False,
            "next_stage_authorized": "none",
        }
    )
    return passport


def _source_hashes(root: Path) -> dict[str, dict[str, str]]:
    source_paths = {
        "config": CONFIG_FILE,
        "v2_core": CORE_MODULE_FILE,
        "v1_design_dependency": "pyfgsea/trajpathmix_functional_core_v1.py",
        "freeze_module": MODULE_FILE,
        "runner": RUNNER_FILE,
        "freeze_test": TEST_FILE,
        "core_test": CORE_TEST_FILE,
    }
    return {
        key: {
            "relative_path": relative_path,
            "sha256": _hash_file(_repo_file(root, relative_path)),
        }
        for key, relative_path in source_paths.items()
    }


def _collect_synthetic_hashes(decision: Mapping[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for suite, metrics in decision["suite_metrics"].items():
        if not isinstance(metrics, Mapping):
            continue
        for key, value in metrics.items():
            if str(key).endswith("_sha256") and isinstance(value, str) and value:
                observed[f"{suite}.{key}"] = value
    return dict(sorted(observed.items()))


def _build_record_payload(
    *,
    root: Path,
    config: Mapping[str, Any],
    decision: Mapping[str, Any],
    authorization_audit: Mapping[str, Any],
    implementation_audit: Mapping[str, Any],
    artifact_hashes: Mapping[str, str],
) -> dict[str, Any]:
    source_hashes = _source_hashes(root)
    full_read_whitelist = sorted(
        set(_runtime_input_read_whitelist(config))
        | {record["relative_path"] for record in source_hashes.values()}
    )
    return {
        "schema_name": "trajpathmix_functional_core_v2_0_build_record",
        "schema_version": SCHEMA_VERSION,
        "project_id": PROJECT_ID,
        "frozen_at_utc": config["frozen_at_utc"],
        "frozen_payload_sha256": FROZEN_PAYLOAD_SHA256,
        "decision": decision["decision"],
        "artifact_sha256": deepcopy(dict(artifact_hashes)),
        "authorization_binding_sha256": {
            key: value["sha256"] for key, value in authorization_audit.items()
        },
        "implementation_binding_sha256": {
            key: value["sha256"] for key, value in implementation_audit.items()
        },
        "source_sha256": source_hashes,
        "synthetic_fixture_and_stream_sha256": _collect_synthetic_hashes(decision),
        "exact_files_read_during_materialization": full_read_whitelist,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyyaml": yaml.__version__,
            "platform": platform.platform(),
        },
        "attempt_number": 1,
        "automatic_retry_used": False,
        "automatic_resume_used": False,
        "thresholds_changed_after_execution": False,
        "formal_fixture_results_inspected_before_materialization": False,
        "validate_only_reran_synthetic_suites": False,
        "post_write_structural_validation_performed": True,
        "real_expression_read": False,
        "real_condition_labels_read": False,
        "current_cb2_500_assignment_bank_read": False,
        "new_holdout_assignment_bank_generated": False,
        "timing_computed": False,
        "biological_interpretation_performed": False,
        "verification_status": "ANALYZED",
    }


def materialize_v2_0_implementation_freeze(
    *,
    repository_root: str | Path,
    config_path: str | Path,
    explicit_execution_authorization: bool,
) -> dict[str, Any]:
    _require(
        explicit_execution_authorization,
        "Explicit V2-0 implementation-freeze execution authorization is required",
    )
    root = Path(repository_root).resolve()
    expected_config = _repo_file(root, CONFIG_FILE)
    observed_config = Path(config_path).resolve()
    _require_equal(observed_config, expected_config, "config path")
    config = load_v2_0_implementation_freeze_config(observed_config)
    authorization_audit = verify_v2_0_authorization_bindings(root, config)
    implementation_audit = verify_v2_0_implementation_bindings(root, config)
    target = _repo_file(root, OUTPUT_DIR)
    staging = target.with_name(target.name + ".incomplete")
    _require(not target.exists(), f"V2-0 freeze output already exists: {target}")
    _require(not staging.exists(), f"V2-0 freeze stop marker already exists: {staging}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        decision = run_v2_0_implementation_checks(
            repository_root=root,
            config=config,
        )
        passport = _passport_payload(config, decision)
        _write_json(decision, staging / DECISION_FILE)
        _write_tsv(_checks_tsv(decision), staging / CHECKS_FILE)
        _write_json(passport, staging / PASSPORT_FILE)
        artifact_hashes = {
            name: _hash_file(staging / name)
            for name in (DECISION_FILE, CHECKS_FILE, PASSPORT_FILE)
        }
        build_record = _build_record_payload(
            root=root,
            config=config,
            decision=decision,
            authorization_audit=authorization_audit,
            implementation_audit=implementation_audit,
            artifact_hashes=artifact_hashes,
        )
        _write_json(build_record, staging / BUILD_RECORD_FILE)
        os.replace(staging, target)
    except BaseException:
        # An incomplete directory is an explicit one-shot stop marker.  It is
        # intentionally preserved so an execution failure cannot become a
        # silent retry with a newly inspected formal fixture.
        raise
    return validate_v2_0_implementation_freeze_output(
        repository_root=root,
        config_path=observed_config,
    )


def _validate_decision_reduction(
    *,
    decision: Mapping[str, Any],
    config: Mapping[str, Any],
    authorization_audit: Mapping[str, Any],
    implementation_audit: Mapping[str, Any],
) -> None:
    expected_keys = {
        "schema_name",
        "schema_version",
        "project_id",
        "frozen_at_utc",
        "frozen_payload_sha256",
        "decision",
        "v2_0_conforms_to_all_16_frozen_checks",
        "implementation_source_bindings_frozen",
        "all_16_required_checks_pass",
        "required_check_count",
        "passed_check_count",
        "check_results",
        "suite_metrics",
        "fallacy_scan",
        "fallacy_scan_coverage",
        "unresolved_red_flag_count",
        "caution_count",
        "authorization_binding_audit",
        "implementation_binding_audit",
        "suite_exception_count",
        "suite_exceptions",
        "execution_firewall",
        "runtime_input_read_whitelist",
        "claim_ceiling",
        "functional_core_v2_calibrated",
        "new_holdout_assignment_bank_materialization_authorized",
        "new_holdout_500_execution_authorized",
        "next_stage_authorized",
        "failure_action_if_any_check_failed",
    }
    _require_equal(set(decision), expected_keys, "decision field set")
    _require_equal(
        decision.get("schema_name"),
        "trajpathmix_functional_core_v2_0_implementation_decision",
        "decision schema",
    )
    _require_equal(decision.get("schema_version"), SCHEMA_VERSION, "decision version")
    _require_equal(decision.get("project_id"), PROJECT_ID, "decision project")
    _require_equal(
        decision.get("frozen_payload_sha256"),
        FROZEN_PAYLOAD_SHA256,
        "decision payload hash",
    )
    _assert_finite_json(decision)
    _require_equal(
        set(decision["suite_metrics"]),
        {
            "algebraic_and_invariance",
            "heteroskedastic_high_leverage_null",
            "balanced_label_oracle",
            "global_50_x_20",
            "source_and_claim_firewall",
        },
        "suite metric ids",
    )
    reduced = _reduce_suite_metrics_to_checks(decision["suite_metrics"], config)
    records = decision["check_results"]
    _require_equal(len(records), len(CHECK_IDS), "check row count")
    _require_equal(tuple(record["check_id"] for record in records), CHECK_IDS, "check order")
    for record in records:
        _require_equal(set(record), {
            "check_id",
            "status",
            "hard_gate",
            "evidence_kind",
            "observed",
            "acceptance_rule",
            "failure_detail",
        }, f"check fields {record['check_id']}")
        expected_status = "PASS" if reduced[record["check_id"]] else "FAIL"
        _require_equal(record["status"], expected_status, f"check status {record['check_id']}")
        _require_equal(record["hard_gate"], True, f"hard gate {record['check_id']}")
        _require(bool(record["evidence_kind"]), "Check evidence kind must be nonblank")
        _require(bool(record["acceptance_rule"]), "Check acceptance rule must be nonblank")
        if expected_status == "PASS":
            _require_equal(record["failure_detail"], "", f"pass detail {record['check_id']}")
        else:
            _require(bool(record["failure_detail"]), "Failed check must retain a detail")
    passed_count = int(sum(reduced.values()))
    all_pass = all(reduced.values())
    _require_equal(decision["required_check_count"], 16, "required count")
    _require_equal(decision["passed_check_count"], passed_count, "passed count")
    _require_equal(decision["all_16_required_checks_pass"], all_pass, "check reduction")

    expected_scan = _fallacy_scan()
    _require_equal(decision["fallacy_scan"], expected_scan, "fallacy scan records")
    _require_equal(decision["fallacy_scan_coverage"], "11/11", "fallacy coverage")
    unresolved = int(sum(item["status"] == "red_flag" for item in expected_scan))
    cautions = int(sum(item["status"] == "caution" for item in expected_scan))
    _require_equal(decision["unresolved_red_flag_count"], unresolved, "red flags")
    _require_equal(decision["caution_count"], cautions, "cautions")
    conforms = bool(all_pass and unresolved == 0)
    _require_equal(
        decision["decision"], "CONFORMS" if conforms else "STOP", "overall decision"
    )
    _require_equal(
        decision["v2_0_conforms_to_all_16_frozen_checks"],
        conforms,
        "V2-0 conformance",
    )
    _require_equal(decision["implementation_source_bindings_frozen"], True, "source freeze")
    _require_equal(
        decision["authorization_binding_audit"],
        authorization_audit,
        "authorization audit",
    )
    _require_equal(
        decision["implementation_binding_audit"],
        implementation_audit,
        "implementation audit",
    )
    _require_equal(
        decision["suite_exception_count"],
        len(decision["suite_exceptions"]),
        "suite exception count",
    )
    _require_equal(decision["execution_firewall"], config["execution_firewall"], "execution firewall")
    _require_equal(
        decision["runtime_input_read_whitelist"],
        _runtime_input_read_whitelist(config),
        "runtime input whitelist",
    )
    _require_equal(decision["claim_ceiling"], config["claim_ceiling"], "claim ceiling")
    for key in (
        "functional_core_v2_calibrated",
        "new_holdout_assignment_bank_materialization_authorized",
        "new_holdout_500_execution_authorized",
    ):
        _require_equal(decision[key], False, key)
    _require_equal(decision["next_stage_authorized"], "none", "next stage")
    _require_equal(
        decision["failure_action_if_any_check_failed"],
        config["required_checks"]["hard_failure_action"],
        "failure action",
    )


def validate_v2_0_implementation_freeze_output(
    *, repository_root: str | Path, config_path: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = load_v2_0_implementation_freeze_config(config_path)
    authorization_audit = verify_v2_0_authorization_bindings(root, config)
    implementation_audit = verify_v2_0_implementation_bindings(root, config)
    target = _repo_file(root, OUTPUT_DIR)
    _require(target.is_dir(), f"Missing V2-0 freeze output: {target}")
    _require(
        not any(path.is_dir() for path in target.iterdir()),
        "V2-0 freeze output must not contain subdirectories",
    )
    observed_files = tuple(sorted(path.name for path in target.iterdir() if path.is_file()))
    _require_equal(observed_files, tuple(sorted(EXACT_OUTPUT_FILES)), "output file set")
    decision = _strict_json_load(target / DECISION_FILE)
    passport = _strict_json_load(target / PASSPORT_FILE)
    build = _strict_json_load(target / BUILD_RECORD_FILE)
    _validate_decision_reduction(
        decision=decision,
        config=config,
        authorization_audit=authorization_audit,
        implementation_audit=implementation_audit,
    )
    expected_tsv = _checks_tsv(decision)
    _require_equal(
        (target / CHECKS_FILE).read_text(encoding="utf-8"),
        expected_tsv,
        "check-results TSV",
    )
    expected_passport = _passport_payload(config, decision)
    _require_equal(passport, expected_passport, "material passport")
    expected_artifact_hashes = {
        name: _hash_file(target / name)
        for name in (DECISION_FILE, CHECKS_FILE, PASSPORT_FILE)
    }
    expected_build = _build_record_payload(
        root=root,
        config=config,
        decision=decision,
        authorization_audit=authorization_audit,
        implementation_audit=implementation_audit,
        artifact_hashes=expected_artifact_hashes,
    )
    _require_equal(build, expected_build, "complete build record")
    return {
        "valid": True,
        "project_id": PROJECT_ID,
        "output_dir": str(target),
        "frozen_payload_sha256": FROZEN_PAYLOAD_SHA256,
        "decision": decision["decision"],
        "passed_check_count": decision["passed_check_count"],
        "required_check_count": decision["required_check_count"],
        "functional_core_v2_calibrated": False,
        "new_holdout_500_execution_authorized": False,
        "decision_sha256": _hash_file(target / DECISION_FILE),
        "build_record_sha256": _hash_file(target / BUILD_RECORD_FILE),
    }


__all__ = [
    "BUILD_RECORD_FILE",
    "CHECKS_FILE",
    "CHECK_IDS",
    "CONFIG_FILE",
    "CORE_MODULE_FILE",
    "DECISION_FILE",
    "EXACT_OUTPUT_FILES",
    "FALLACY_IDS",
    "FROZEN_PAYLOAD_SHA256",
    "MODULE_FILE",
    "OUTPUT_DIR",
    "PASSPORT_FILE",
    "PROJECT_ID",
    "RUNNER_FILE",
    "TEST_FILE",
    "V20ImplementationFreezeError",
    "load_v2_0_implementation_freeze_config",
    "materialize_v2_0_implementation_freeze",
    "run_v2_0_implementation_checks",
    "validate_v2_0_implementation_freeze_config",
    "validate_v2_0_implementation_freeze_output",
    "verify_v2_0_authorization_bindings",
    "verify_v2_0_implementation_bindings",
]
