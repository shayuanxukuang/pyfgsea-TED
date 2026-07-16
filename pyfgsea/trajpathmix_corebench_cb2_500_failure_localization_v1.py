"""Append-only failure localization for the failed TrajPathMix CB2-500 run.

This module is deliberately diagnostic.  It consumes only the published
CB2-500 artifacts and the already-frozen cache, assignments, and mapping
streams.  It cannot generate assignments or mappings, cannot alter an
acceptance threshold, and cannot authorize a later stage.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .trajpathmix_corebench_cb2_500 import (
    _factorized_cache_from_arrays,
    derive_residual_mapping_seed,
    run_cached_functional_core_batch,
    validate_cb2_500_output,
)


PROJECT_ID = "COREBENCH_CB2_500_FAILURE_LOCALIZATION_v1"
SCHEMA_VERSION = "1.0.0"
DEFAULT_CONFIG_FILE = Path("config/trajpathmix_corebench_cb2_500_failure_localization_v1.yaml")
DEFAULT_OUTPUT_DIR = Path("data_external/trajpathmix_corebench_cb2_500_failure_localization_v1")
DEFAULT_CACHE_DIR = Path("data_external/trajpathmix_corebench_cb2_500_cache_v1")
DEFAULT_FORMAL_OUTPUT_DIR = Path("data_external/trajpathmix_corebench_cb2_500_execution_v1")
CB2_EXECUTION_CONFIG = Path("config/trajpathmix_corebench_cb2_500_execution_v1.yaml")

STATE_FILE = "cb2_500_failure_localization_project_state_v1.json"
INPUT_AUDIT_FILE = "cb2_500_failure_localization_input_audit_v1.json"
BAND_SCOPE_FILE = "cb2_500_failure_localization_band_scope_summary_v1.json"
BAND_PATHWAY_FILE = "cb2_500_failure_localization_band_pathway_coverage_v1.tsv"
BAND_FAMILY_FILE = "cb2_500_failure_localization_band_family_coverage_v1.tsv"
BIN_DIAGNOSTIC_FILE = "cb2_500_failure_localization_bin_missingness_diagnostics_v1.tsv"
INTEGRATED_PATHWAY_FILE = "cb2_500_failure_localization_integrated_pathway_diagnostics_v1.tsv"
INTEGRATED_FAMILY_FILE = "cb2_500_failure_localization_integrated_family_diagnostics_v1.tsv"
REFERENCE_REPLICATE_FILE = "cb2_500_failure_localization_reference_replicate_diagnostics_v1.tsv"
REFERENCE_PATHWAY_FILE = "cb2_500_failure_localization_reference_pathway_diagnostics_v1.tsv"
DIRECT_REFERENCE_FILE = "cb2_500_failure_localization_direct_label_vs_fl_summary_v1.json"
BY_FILE = "cb2_500_failure_localization_by_evaluability_v1.json"
DECISION_FILE = "COREBENCH_CB2_500_FAILURE_LOCALIZATION_DECISION_v1.json"
PASSPORT_FILE = "cb2_500_failure_localization_material_passport_v1.json"
BUILD_RECORD_FILE = "cb2_500_failure_localization_build_record_v1.json"

OUTPUT_FILES = (
    STATE_FILE,
    INPUT_AUDIT_FILE,
    BAND_SCOPE_FILE,
    BAND_PATHWAY_FILE,
    BAND_FAMILY_FILE,
    BIN_DIAGNOSTIC_FILE,
    INTEGRATED_PATHWAY_FILE,
    INTEGRATED_FAMILY_FILE,
    REFERENCE_REPLICATE_FILE,
    REFERENCE_PATHWAY_FILE,
    DIRECT_REFERENCE_FILE,
    BY_FILE,
    DECISION_FILE,
    PASSPORT_FILE,
    BUILD_RECORD_FILE,
)

REQUIRED_BINDINGS = {
    "cb2_execution_config_v1",
    "formal_cb2_500_decision_v1",
    "formal_cb2_500_summary_v1",
    "formal_cb2_500_build_record_v1",
    "formal_cb2_500_material_passport_v1",
    "cb2_500_cache_manifest_v1",
    "cb2_500_cache_build_record_v1",
    "cb2_500_module",
    "cb2_500_runner",
    "cb2_500_test",
    "failure_localization_module_v1",
    "failure_localization_runner_v1",
    "failure_localization_test_v1",
}

CURVE_FILE = "cb2_500_curve_bin_metrics_v1.tsv"
PATHWAY_FILE = "cb2_500_pathway_replicate_metrics_v1.tsv"
FAMILY_FILE = "cb2_500_family_replicate_metrics_v1.tsv"
FORMAL_SUMMARY_FILE = "cb2_500_acceptance_summary_v1.json"
FORMAL_DECISION_FILE = "CB2_500_ACCEPTANCE_DECISION_v1.json"
FORMAL_BUILD_FILE = "cb2_500_build_record_v1.json"
FORMAL_PASSPORT_FILE = "cb2_500_material_passport_v1.json"

SCORES_FILE = "donor_bin_pathway_scores_float64_v1.npy"
MAPPINGS_FILE = "residual_mapping_bank_uint8_v1.npy"
SUPPORT_FILE = "simultaneous_band_family_mask_bool_v1.npy"
DESIGN_FILE = "bin_specific_nuisance_design_cache_factorized_v1.npz"
MEMBERSHIP_FILE = "pathway_membership_matrix_bool_v1.npy"
DONOR_AXIS_FILE = "donor_axis_v1.tsv"
PATHWAY_AXIS_FILE = "pathway_axis_v1.tsv"
ASSIGNMENT_AXIS_FILE = "assignment_axis_v1.tsv"
EXPERIMENT_AXIS_FILE = "experiment_axis_v1.tsv"
FAMILY_AXIS_FILE = "family_axis_v1.tsv"
FAMILY_INDEX_FILE = "pathway_family_index_v1.tsv"
CACHE_MANIFEST_FILE = "cb2_500_cache_manifest_v1.json"
CACHE_BUILD_FILE = "cb2_500_cache_build_record_v1.json"

ALPHA = 0.05
N_REPLICATES = 500
N_MAPPINGS = 999
N_PATHWAYS = 50
N_BINS = 20
N_FAMILIES = 13
NUMERICAL_TOLERANCE = 1.0e-12

INFERENCE_STATE = {
    "functional_core_calibrated": False,
    "curve_maxT_inference_allowed": False,
    "integrated_effect_inference_allowed": False,
    "simultaneous_band_inference_allowed": False,
    "pathway_family_inference_allowed": False,
    "timing_claim_allowed": False,
    "biological_discovery_allowed": False,
    "unblinding_allowed": False,
    "release_allowed": False,
    "cb2_2000_allowed": False,
    "cb3_injection_allowed": False,
    "phase_b_allowed": False,
    "sound_life_allowed": False,
    "paired_v2_allowed": False,
}


class FailureLocalizationContractError(ValueError):
    """Raised when an append-only localization contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FailureLocalizationContractError(message)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _payload_hash(config: Mapping[str, Any]) -> str:
    payload = {
        str(key): value
        for key, value in config.items()
        if key != "frozen_payload_sha256" and not str(key).startswith("_")
    }
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_load(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            FailureLocalizationContractError(
                f"Non-finite JSON constant {value!r} in {path}"
            )
        ),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    safe = _json_safe(value)
    text = json.dumps(
        safe,
        sort_keys=True,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    output = frame.copy()
    for column in output.columns:
        if pd.api.types.is_bool_dtype(output[column].dtype):
            output[column] = output[column].map({True: "true", False: "false"})
    text = output.to_csv(
        sep="\t",
        index=False,
        na_rep="NA",
        lineterminator="\n",
    )
    path.write_text(text, encoding="utf-8", newline="\n")


def _read_bool(series: pd.Series, label: str) -> np.ndarray:
    normalized = series.astype(str).str.strip().str.lower()
    _require(normalized.isin(("true", "false")).all(), f"Invalid boolean column {label}")
    return normalized.eq("true").to_numpy(dtype=bool)


def _higher_quantile(values: Any, q: float, axis: int | None = None) -> np.ndarray:
    return np.quantile(np.asarray(values, dtype=float), q, axis=axis, method="higher")


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0.0:
        return None
    return float(numerator / denominator)


def load_failure_localization_contract(
    path: str | Path = DEFAULT_CONFIG_FILE,
    *,
    require_frozen: bool = True,
) -> dict[str, Any]:
    config_path = Path(path)
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise FailureLocalizationContractError("Localization YAML must be a mapping")
    config = dict(value)
    _require(
        config.get("schema_name")
        == "trajpathmix_corebench_cb2_500_failure_localization_contract",
        "Unexpected localization schema",
    )
    _require(config.get("schema_version") == SCHEMA_VERSION, "Unexpected schema version")
    _require(config.get("project_id") == PROJECT_ID, "Unexpected project id")
    observed_payload = _payload_hash(config)
    frozen_payload = str(config.get("frozen_payload_sha256", ""))
    if require_frozen:
        _require(len(frozen_payload) == 64, "Localization payload hash is not frozen")
        _require(observed_payload == frozen_payload, "Localization payload hash mismatch")
    _require(
        config.get("project_state") == INFERENCE_STATE,
        "Project inference state does not match the required closure",
    )
    execution = config.get("execution_authorization", {})
    _require(execution.get("failure_localization_authorized") is True, "Localization not authorized")
    for forbidden in (
        "new_assignments_allowed",
        "new_mappings_allowed",
        "threshold_changes_allowed",
        "real_condition_contrast_allowed",
        "injection_recovery_allowed",
        "biological_interpretation_allowed",
        "timing_computation_allowed",
        "automatic_retry_allowed",
        "automatic_resume_allowed",
    ):
        _require(execution.get(forbidden) is False, f"Forbidden authorization is open: {forbidden}")
    replay = config.get("replay_contract", {})
    _require(replay.get("replicates") == N_REPLICATES, "Replay replicate count changed")
    _require(replay.get("mappings_per_replicate") == N_MAPPINGS, "Replay mapping count changed")
    _require(replay.get("persist_mapping_level_null_arrays") is False, "Null arrays may not be persisted")
    _require(
        set(config.get("bindings", {})) == REQUIRED_BINDINGS,
        "Localization binding set is incomplete or contains extras",
    )
    _require(
        set(config.get("sources", {}))
        == {"cb2_execution_config", "cache_dir", "formal_output_dir"},
        "Localization source set is incomplete or contains extras",
    )
    output_contract = config.get("output_contract", {})
    _require(
        output_contract.get("default_output_dir")
        == "data_external/trajpathmix_corebench_cb2_500_failure_localization_v1",
        "Unexpected localization output directory",
    )
    _require(
        tuple(output_contract.get("exact_output_files", ())) == OUTPUT_FILES,
        "Localization output file contract mismatch",
    )
    _require(
        set(output_contract.get("tables", {}))
        == {
            BAND_PATHWAY_FILE,
            BAND_FAMILY_FILE,
            BIN_DIAGNOSTIC_FILE,
            INTEGRATED_PATHWAY_FILE,
            INTEGRATED_FAMILY_FILE,
            REFERENCE_REPLICATE_FILE,
            REFERENCE_PATHWAY_FILE,
        },
        "Localization table schema set mismatch",
    )
    for name, spec in output_contract["tables"].items():
        _require(
            isinstance(spec, Mapping)
            and isinstance(spec.get("columns"), list)
            and len(spec["columns"]) == len(set(spec["columns"]))
            and len(spec["columns"]) > 0,
            f"Invalid table schema: {name}",
        )
    _require(
        set(output_contract.get("json_exact_keys", {}))
        == {
            STATE_FILE,
            INPUT_AUDIT_FILE,
            BAND_SCOPE_FILE,
            DIRECT_REFERENCE_FILE,
            BY_FILE,
            DECISION_FILE,
            PASSPORT_FILE,
            BUILD_RECORD_FILE,
        },
        "Localization JSON schema set mismatch",
    )
    config["_config_payload_sha256"] = observed_payload
    config["_config_path"] = str(config_path.resolve())
    return config


def _resolve(root: Path, relative: str | Path) -> Path:
    path = Path(relative)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise FailureLocalizationContractError(f"Path escapes repository root: {relative}") from exc
    return resolved


def validate_failure_localization_sources(
    config_path: str | Path = DEFAULT_CONFIG_FILE,
    repository_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    resolved_config = _resolve(root, config_path)
    config = load_failure_localization_contract(resolved_config, require_frozen=True)
    verified_bindings: dict[str, dict[str, Any]] = {}
    for name, binding in config.get("bindings", {}).items():
        path = _resolve(root, str(binding["relative_path"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = _hash_file(path)
        _require(digest == str(binding["sha256"]), f"Binding hash mismatch: {name}")
        verified_bindings[str(name)] = {
            "relative_path": str(binding["relative_path"]),
            "sha256": digest,
            "size_bytes": int(path.stat().st_size),
        }
    _require(
        _resolve(
            root,
            config["bindings"]["failure_localization_module_v1"]["relative_path"],
        )
        == Path(__file__).resolve(),
        "Loaded localization module differs from the frozen module binding",
    )

    sources = config["sources"]
    cache_dir = _resolve(root, sources["cache_dir"])
    formal_dir = _resolve(root, sources["formal_output_dir"])
    cb2_config = _resolve(root, sources["cb2_execution_config"])
    upstream = validate_cb2_500_output(cb2_config, root, cache_dir, formal_dir)
    decision = _strict_json_load(formal_dir / FORMAL_DECISION_FILE)
    summary = _strict_json_load(formal_dir / FORMAL_SUMMARY_FILE)
    _require(decision.get("cb2_500_pass") is False, "Localization requires a failed CB2-500")
    _require(
        decision.get("decision") == "fail_cb2_500_stop_before_cb2_2000",
        "Formal stop decision is not intact",
    )
    _require(decision.get("cb2_2000_start_allowed") is False, "CB2-2000 is unexpectedly allowed")
    _require(summary.get("n_completed_replicates") == N_REPLICATES, "Formal run incomplete")
    _require(summary.get("n_refused_replicates") == 0, "Formal run contains refusals")
    _require(summary.get("n_false_open_replicates") == 0, "Formal run contains false-open results")
    return {
        "config": config,
        "repository_root": root,
        "cache_dir": cache_dir,
        "formal_output_dir": formal_dir,
        "cb2_execution_config": cb2_config,
        "verified_bindings": verified_bindings,
        "upstream_validation": upstream,
    }


def _reference_replay_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from threadpoolctl import threadpool_limits

        limiter = threadpool_limits(limits=1)
    except ImportError:  # pragma: no cover
        limiter = nullcontext()
    with limiter:
        result = run_cached_functional_core_batch(
            outcomes=payload["outcomes"],
            donor_ids=payload["donor_ids"],
            condition=payload["condition"],
            availability=payload["availability"],
            experiment_fractions=payload["experiment_fractions"],
            experiment_ids=payload["experiment_ids"],
            pathway_ids=payload["pathway_ids"],
            mappings=payload["mappings"],
            factorized_cache=payload["factorized_cache"],
            family_index=payload["family_index"],
            support_mask=payload["support_mask"],
            alpha=ALPHA,
            chunk_size=payload["chunk_size"],
            mapping_seed=payload["mapping_seed"],
        )
    support = result.support_mask
    reduced_global = np.max(np.abs(result.null_studentized_effect[:, support]), axis=1)
    full_global = np.max(
        np.abs(result.bootstrap_studentized_deviation[:, support]), axis=1
    )
    observed_global = float(np.max(result.curve_statistic))
    curve_q95 = _higher_quantile(result.null_curve_statistic, 0.95, axis=0)
    integrated_q95 = _higher_quantile(result.null_integrated_effect, 0.95, axis=0)
    integrated_sd = np.std(result.null_integrated_effect, axis=0, ddof=1)
    return {
        "replicate_index_0based": int(payload["replicate_index_0based"]),
        "assignment_id": str(payload["assignment_id"]),
        "mapping_stream_sha256": str(payload["mapping_stream_sha256"]),
        "observed_global_max": observed_global,
        "observed_curve_statistic": result.curve_statistic,
        "observed_integrated_effect": result.integrated_effect,
        "reduced_global_q50": float(_higher_quantile(reduced_global, 0.50)),
        "reduced_global_q90": float(_higher_quantile(reduced_global, 0.90)),
        "reduced_global_q95": float(_higher_quantile(reduced_global, 0.95)),
        "reduced_global_q99": float(_higher_quantile(reduced_global, 0.99)),
        "reduced_global_sd": float(np.std(reduced_global, ddof=1)),
        "reduced_global_exceedance_count": int(
            np.sum(reduced_global >= observed_global - 1.0e-12)
        ),
        "full_global_q50": float(_higher_quantile(full_global, 0.50)),
        "full_global_q90": float(_higher_quantile(full_global, 0.90)),
        "full_global_q95": float(_higher_quantile(full_global, 0.95)),
        "full_global_q99": float(_higher_quantile(full_global, 0.99)),
        "full_global_sd": float(np.std(full_global, ddof=1)),
        "full_global_exceedance_count": int(
            np.sum(full_global >= observed_global - 1.0e-12)
        ),
        "curve_q95": curve_q95,
        "integrated_q95": integrated_q95,
        "integrated_sd": integrated_sd,
        "integrated_mean": np.mean(result.null_integrated_effect, axis=0),
        "integrated_median": _higher_quantile(
            result.null_integrated_effect, 0.50, axis=0
        ),
    }


def _replay_references(
    cache_dir: Path,
    *,
    processes: int,
    chunk_size: int,
) -> list[dict[str, Any]]:
    _require(processes > 0, "processes must be positive")
    _require(chunk_size > 0, "chunk_size must be positive")
    donors = tuple(
        pd.read_csv(cache_dir / DONOR_AXIS_FILE, sep="\t", dtype="string")["donor_id"].astype(str)
    )
    pathways = tuple(
        pd.read_csv(cache_dir / PATHWAY_AXIS_FILE, sep="\t", dtype="string")["pathway_id"].astype(str)
    )
    assignments_axis = pd.read_csv(cache_dir / ASSIGNMENT_AXIS_FILE, sep="\t", dtype="string")
    experiments = tuple(
        pd.read_csv(cache_dir / EXPERIMENT_AXIS_FILE, sep="\t", dtype="string")["experiment_id"].astype(str)
    )
    family_index = pd.read_csv(cache_dir / FAMILY_INDEX_FILE, sep="\t")[
        "level_1_family_index"
    ].to_numpy(dtype=int)
    outcomes = np.load(cache_dir / SCORES_FILE, mmap_mode="r", allow_pickle=False)
    mappings = np.load(cache_dir / MAPPINGS_FILE, mmap_mode="r", allow_pickle=False)
    support = np.load(cache_dir / SUPPORT_FILE, mmap_mode="r", allow_pickle=False)
    with np.load(cache_dir / DESIGN_FILE, allow_pickle=False) as archive:
        design_arrays = {name: archive[name] for name in archive.files}
    assignments = np.asarray(design_arrays["frozen_assignments_uint8"], dtype=np.uint8)
    availability = np.asarray(design_arrays["availability_bool"], dtype=bool)
    fractions = np.asarray(design_arrays["experiment_fractions_float64"], dtype=float)
    mapping_audit = pd.read_csv(
        cache_dir / "residual_mapping_bank_stream_audit_v1.tsv", sep="\t", dtype="string"
    )

    def payload_for(index: int) -> dict[str, Any]:
        condition = assignments[index]
        return {
            "replicate_index_0based": index,
            "assignment_id": str(assignments_axis.iloc[index]["assignment_id"]),
            "mapping_seed": derive_residual_mapping_seed(index),
            "mapping_stream_sha256": str(mapping_audit.iloc[index]["stream_sha256"]),
            "outcomes": np.asarray(outcomes),
            "donor_ids": donors,
            "condition": condition,
            "availability": availability,
            "experiment_fractions": fractions,
            "experiment_ids": experiments,
            "pathway_ids": pathways,
            "mappings": np.asarray(mappings[index], dtype=np.int32),
            "factorized_cache": _factorized_cache_from_arrays(
                design_arrays,
                index,
                donor_ids=donors,
                experiment_ids=experiments,
                condition=condition,
                availability=availability,
                experiment_fractions=fractions,
            ),
            "family_index": family_index,
            "support_mask": np.asarray(support, dtype=bool),
            "chunk_size": chunk_size,
        }

    results: dict[int, dict[str, Any]] = {}
    if processes == 1:
        for index in range(N_REPLICATES):
            result = _reference_replay_worker(payload_for(index))
            results[index] = result
    else:
        with ProcessPoolExecutor(max_workers=processes) as executor:
            next_index = 0
            pending: dict[Any, int] = {}
            while next_index < N_REPLICATES and len(pending) < processes * 2:
                pending[executor.submit(_reference_replay_worker, payload_for(next_index))] = next_index
                next_index += 1
            while pending:
                completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in completed:
                    expected_index = pending.pop(future)
                    result = future.result()
                    _require(
                        int(result["replicate_index_0based"]) == expected_index,
                        "Replay worker index mismatch",
                    )
                    results[expected_index] = result
                    if next_index < N_REPLICATES:
                        pending[
                            executor.submit(_reference_replay_worker, payload_for(next_index))
                        ] = next_index
                        next_index += 1
    _require(set(results) == set(range(N_REPLICATES)), "Replay did not cover all 500 replicates")
    return [results[index] for index in range(N_REPLICATES)]


def _fallacy_scan() -> dict[str, Any]:
    return {
        "coverage": "11_of_11",
        "simpsons_paradox": "checked_no_inference_from_aggregate_without_pathway_family_strata",
        "ecological_fallacy": "checked_donor_is_independence_unit_no_cell_level_claim",
        "berksons_paradox": "checked_frozen_cohort_no_outcome_based_selection",
        "collider_bias": "checked_no_new_covariate_or_outcome_conditioned_filter",
        "base_rate_neglect": "checked_fixed_complete_null_denominators_reported",
        "regression_to_mean": "checked_all_500_frozen_assignments_retained",
        "survivorship_bias": "checked_zero_refusals_and_fixed_denominators",
        "look_elsewhere_effect": "checked_five_diagnostics_frozen_before_replay",
        "garden_of_forking_paths": "checked_no_threshold_pathway_or_mapping_changes",
        "correlation_not_causation": "checked_no_biological_or_causal_claim",
        "reverse_causality": "checked_pseudo_condition_null_has_no_causal_direction_claim",
    }


def _spearman_without_p(x: Sequence[float], y: Sequence[float]) -> float | None:
    left = pd.Series(np.asarray(x, dtype=float)).rank(method="average")
    right = pd.Series(np.asarray(y, dtype=float)).rank(method="average")
    value = left.corr(right, method="pearson")
    return None if pd.isna(value) else float(value)


def _analyze(
    formal_dir: Path,
    cache_dir: Path,
    replay: list[dict[str, Any]],
) -> dict[str, Any]:
    pathways = pd.read_csv(formal_dir / PATHWAY_FILE, sep="\t", dtype={"pathway_id": "string"})
    families = pd.read_csv(formal_dir / FAMILY_FILE, sep="\t", dtype={"family_id": "string"})
    curve = pd.read_csv(
        formal_dir / CURVE_FILE,
        sep="\t",
        dtype={"pathway_id": "string", "level_1_family_id": "string"},
    )
    _require(len(pathways) == 25_000, "Formal pathway row count changed")
    _require(len(families) == 6_500, "Formal family row count changed")
    _require(len(curve) == 500_000, "Formal curve row count changed")
    path_axis = pd.read_csv(cache_dir / PATHWAY_AXIS_FILE, sep="\t", dtype={"pathway_id": "string"})
    family_axis = pd.read_csv(cache_dir / FAMILY_AXIS_FILE, sep="\t", dtype={"family_id": "string"})
    family_index_table = pd.read_csv(
        cache_dir / FAMILY_INDEX_FILE, sep="\t", dtype={"pathway_id": "string"}
    )
    assignment_axis = pd.read_csv(cache_dir / ASSIGNMENT_AXIS_FILE, sep="\t", dtype="string")
    membership = np.load(cache_dir / MEMBERSHIP_FILE, mmap_mode="r", allow_pickle=False)
    with np.load(cache_dir / DESIGN_FILE, allow_pickle=False) as design:
        availability = np.asarray(design["availability_bool"], dtype=bool)

    pointwise_contains = (
        (curve["pointwise_lower"].to_numpy(dtype=float) <= 0.0)
        & (curve["pointwise_upper"].to_numpy(dtype=float) >= 0.0)
    )
    simultaneous_contains = (
        (curve["simultaneous_lower"].to_numpy(dtype=float) <= 0.0)
        & (curve["simultaneous_upper"].to_numpy(dtype=float) >= 0.0)
    )
    saved_simultaneous = _read_bool(curve["simultaneous_contains_zero"], "simultaneous_contains_zero")
    _require(
        np.array_equal(simultaneous_contains, saved_simultaneous),
        "Saved simultaneous coverage differs from numeric bounds",
    )
    se = curve["standard_error"].to_numpy(dtype=float)
    effect = curve["effect"].to_numpy(dtype=float)
    _require(np.all(se > 0.0), "Supported standard errors must be positive")
    abs_t = np.abs(effect / se)
    full_critical = (curve["simultaneous_upper"].to_numpy(dtype=float) - effect) / se
    pointwise_critical = (curve["pointwise_upper"].to_numpy(dtype=float) - effect) / se
    curve = curve.assign(
        _pointwise_contains=pointwise_contains,
        _simultaneous_contains=simultaneous_contains,
        _abs_t=abs_t,
        _full_critical=full_critical,
        _pointwise_critical=pointwise_critical,
    )

    path_units = (
        curve.groupby(["replicate_index_0based", "pathway_id"], sort=False, observed=True)
        .agg(
            global_band_pathway_localization_covered=("_simultaneous_contains", "all"),
            observed_curve_statistic=("_abs_t", "max"),
        )
        .reset_index()
    )
    family_curve = curve[curve["level_1_family_id"].notna()].copy()
    family_units = (
        family_curve.groupby(
            ["replicate_index_0based", "level_1_family_id"], sort=False, observed=True
        )["_simultaneous_contains"]
        .all()
        .rename("global_band_family_localization_covered")
        .reset_index()
    )
    global_units = (
        curve.groupby("replicate_index_0based", sort=False)["_simultaneous_contains"]
        .all()
        .to_numpy(dtype=bool)
    )
    observed_global_from_curve = (
        curve.groupby("replicate_index_0based", sort=False)["_abs_t"].max().to_numpy(dtype=float)
    )
    formal_full_critical = (
        curve.groupby("replicate_index_0based", sort=False)["_full_critical"]
        .agg(["min", "max", "median"])
        .reset_index()
    )
    _require(
        float((formal_full_critical["max"] - formal_full_critical["min"]).max())
        <= NUMERICAL_TOLERANCE,
        "A formal replicate does not use one global band critical",
    )

    path_order = tuple(path_axis["pathway_id"].astype(str))
    family_order = tuple(family_axis["family_id"].astype(str))
    observed_curve_matrix = (
        pathways.pivot(index="replicate_index_0based", columns="pathway_id", values="curve_statistic")
        .reindex(index=range(N_REPLICATES), columns=path_order)
        .to_numpy(dtype=float)
    )
    observed_integrated_matrix = (
        pathways.pivot(index="replicate_index_0based", columns="pathway_id", values="integrated_effect")
        .reindex(index=range(N_REPLICATES), columns=path_order)
        .to_numpy(dtype=float)
    )
    formal_curve_reject_matrix = (
        pathways.assign(
            _reject=_read_bool(pathways["global_50_curve_reject_alpha"], "global curve reject")
        )
        .pivot(index="replicate_index_0based", columns="pathway_id", values="_reject")
        .reindex(index=range(N_REPLICATES), columns=path_order)
        .to_numpy(dtype=bool)
    )
    formal_integrated_reject_matrix = (
        pathways.assign(
            _reject=_read_bool(pathways["integrated_reject_alpha"], "integrated reject")
        )
        .pivot(index="replicate_index_0based", columns="pathway_id", values="_reject")
        .reindex(index=range(N_REPLICATES), columns=path_order)
        .to_numpy(dtype=bool)
    )
    _require(
        np.max(np.abs(observed_curve_matrix - path_units.pivot(
            index="replicate_index_0based", columns="pathway_id", values="observed_curve_statistic"
        ).reindex(index=range(N_REPLICATES), columns=path_order).to_numpy(dtype=float)))
        <= NUMERICAL_TOLERANCE,
        "Formal curve statistics do not equal max-bin |effect/SE|",
    )

    replay_observed_curve = np.vstack([item["observed_curve_statistic"] for item in replay])
    replay_observed_integrated = np.vstack([item["observed_integrated_effect"] for item in replay])
    _require(
        np.max(np.abs(replay_observed_curve - observed_curve_matrix))
        <= NUMERICAL_TOLERANCE,
        "Replayed curve statistic differs from formal output",
    )
    _require(
        np.max(np.abs(replay_observed_integrated - observed_integrated_matrix))
        <= NUMERICAL_TOLERANCE,
        "Replayed integrated effect differs from formal output",
    )
    replay_full_q95 = np.asarray([item["full_global_q95"] for item in replay], dtype=float)
    _require(
        np.max(np.abs(replay_full_q95 - formal_full_critical["median"].to_numpy(dtype=float)))
        <= NUMERICAL_TOLERANCE,
        "Replayed full-residual critical differs from published bands",
    )

    observed_global = np.max(observed_curve_matrix, axis=1)

    family_observed_matrix = (
        families.pivot(
            index="replicate_index_0based",
            columns="family_id",
            values="observed_family_max_statistic",
        )
        .reindex(index=range(N_REPLICATES), columns=family_order)
        .to_numpy(dtype=float)
    )
    empirical_family_q95 = _higher_quantile(family_observed_matrix, 0.95, axis=0)
    formal_family_reject = (
        families.assign(_reject=_read_bool(families["family_reject_alpha"], "family reject"))
        .pivot(index="replicate_index_0based", columns="family_id", values="_reject")
        .reindex(index=range(N_REPLICATES), columns=family_order)
        .to_numpy(dtype=bool)
    )

    membership_array = np.asarray(membership, dtype=bool)
    pathway_size = membership_array.sum(axis=0).astype(int)
    intersections = membership_array.T.astype(np.int32) @ membership_array.astype(np.int32)
    unions = pathway_size[:, None] + pathway_size[None, :] - intersections
    jaccard = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=float),
        where=unions > 0,
    )
    np.fill_diagonal(jaccard, 0.0)
    overlap_degree = ((intersections > 0) & (~np.eye(N_PATHWAYS, dtype=bool))).sum(axis=1)
    max_jaccard = jaccard.max(axis=1)
    mean_jaccard = jaccard.sum(axis=1) / float(N_PATHWAYS - 1)

    effect_cube = (
        curve.pivot_table(
            index="replicate_index_0based",
            columns=["pathway_id", "bin_index"],
            values="effect",
            sort=False,
        )
        .reindex(
            index=range(N_REPLICATES),
            columns=pd.MultiIndex.from_product([path_order, range(N_BINS)]),
        )
        .to_numpy(dtype=float)
        .reshape(N_REPLICATES, N_PATHWAYS, N_BINS)
    )
    weights = np.full(N_BINS, 1.0 / N_BINS, dtype=float)
    full_signed_sd = np.empty(N_PATHWAYS, dtype=float)
    diagonal_signed_sd = np.empty(N_PATHWAYS, dtype=float)
    covariance_contribution = np.empty(N_PATHWAYS, dtype=float)
    for path_index in range(N_PATHWAYS):
        covariance = np.cov(effect_cube[:, path_index, :], rowvar=False, ddof=1)
        full_variance = float(weights @ covariance @ weights)
        diagonal_variance = float(np.sum(weights * weights * np.diag(covariance)))
        full_signed_sd[path_index] = math.sqrt(max(0.0, full_variance))
        diagonal_signed_sd[path_index] = math.sqrt(max(0.0, diagonal_variance))
        covariance_contribution[path_index] = full_variance - diagonal_variance

    replay_curve_q95 = np.vstack([item["curve_q95"] for item in replay])
    replay_integrated_q95 = np.vstack([item["integrated_q95"] for item in replay])
    replay_integrated_sd = np.vstack([item["integrated_sd"] for item in replay])
    replay_integrated_mean = np.vstack([item["integrated_mean"] for item in replay])
    replay_integrated_median = np.vstack([item["integrated_median"] for item in replay])
    integrated_empirical_sd = np.std(observed_integrated_matrix, axis=0, ddof=1)
    integrated_empirical_variance = np.var(observed_integrated_matrix, axis=0, ddof=1)
    median_fl_integrated_sd = np.median(replay_integrated_sd, axis=0)
    empirical_curve_q95 = _higher_quantile(observed_curve_matrix, 0.95, axis=0)
    empirical_integrated_q95 = _higher_quantile(observed_integrated_matrix, 0.95, axis=0)
    median_fl_curve_q95 = np.median(replay_curve_q95, axis=0)
    median_fl_integrated_q95 = np.median(replay_integrated_q95, axis=0)
    missing_fraction = float(1.0 - availability.mean())

    path_coverage = (
        path_units.groupby("pathway_id", sort=False)["global_band_pathway_localization_covered"]
        .agg(["sum", "count"])
        .reindex(path_order)
    )
    pointwise_by_path = (
        curve.groupby("pathway_id", sort=False)["_pointwise_contains"]
        .agg(["sum", "count"])
        .reindex(path_order)
    )
    band_pathway_rows: list[dict[str, Any]] = []
    for index, pathway_id in enumerate(path_order):
        band_pathway_rows.append(
            {
                "pathway_index": index,
                "pathway_id": pathway_id,
                "pointwise_units": int(pointwise_by_path.iloc[index]["count"]),
                "pointwise_zero_coverage_count": int(pointwise_by_path.iloc[index]["sum"]),
                "pointwise_zero_coverage_rate": float(
                    pointwise_by_path.iloc[index]["sum"] / pointwise_by_path.iloc[index]["count"]
                ),
                "replicate_units": int(path_coverage.iloc[index]["count"]),
                "global_band_pathway_localization_coverage_count": int(
                    path_coverage.iloc[index]["sum"]
                ),
                "global_band_pathway_localization_coverage_rate": float(
                    path_coverage.iloc[index]["sum"] / path_coverage.iloc[index]["count"]
                ),
                "formal_global_curve_reject_count": int(formal_curve_reject_matrix[:, index].sum()),
                "formal_global_curve_reject_rate": float(formal_curve_reject_matrix[:, index].mean()),
            }
        )
    band_pathway = pd.DataFrame(band_pathway_rows)

    family_coverage = (
        family_units.groupby("level_1_family_id", sort=False)[
            "global_band_family_localization_covered"
        ]
        .agg(["sum", "count"])
        .reindex(family_order)
    )
    member_counts = (
        family_index_table[family_index_table["level_1_family_index"] >= 0]
        .groupby("level_1_family_index")["pathway_id"]
        .count()
        .reindex(range(N_FAMILIES))
        .to_numpy(dtype=int)
    )
    band_family_rows: list[dict[str, Any]] = []
    for index, family_id in enumerate(family_order):
        band_family_rows.append(
            {
                "family_index": index,
                "family_id": family_id,
                "n_member_pathways": int(member_counts[index]),
                "replicate_units": int(family_coverage.iloc[index]["count"]),
                "global_band_family_localization_coverage_count": int(
                    family_coverage.iloc[index]["sum"]
                ),
                "global_band_family_localization_coverage_rate": float(
                    family_coverage.iloc[index]["sum"] / family_coverage.iloc[index]["count"]
                ),
                "formal_family_reject_count": int(formal_family_reject[:, index].sum()),
                "formal_family_reject_rate": float(formal_family_reject[:, index].mean()),
                "empirical_frozen_assignment_family_max_q95": float(
                    empirical_family_q95[index]
                ),
            }
        )
    band_family = pd.DataFrame(band_family_rows)

    bin_rows: list[dict[str, Any]] = []
    for bin_index in range(N_BINS):
        local = curve[curve["bin_index"] == bin_index]
        available_donors = int(availability[:, bin_index].sum())
        bin_rows.append(
            {
                "bin_index": bin_index,
                "available_donors": available_donors,
                "missing_donors": int(availability.shape[0] - available_donors),
                "missing_donor_fraction": float(1.0 - available_donors / availability.shape[0]),
                "pointwise_units": int(len(local)),
                "pointwise_zero_coverage_count": int(local["_pointwise_contains"].sum()),
                "pointwise_zero_coverage_rate": float(local["_pointwise_contains"].mean()),
                "global_band_point_zero_coverage_count": int(local["_simultaneous_contains"].sum()),
                "global_band_point_zero_coverage_rate": float(local["_simultaneous_contains"].mean()),
                "mean_absolute_studentized_effect": float(local["_abs_t"].mean()),
            }
        )
    bin_diagnostics = pd.DataFrame(bin_rows)

    family_codes = family_index_table["level_1_family_index"].to_numpy(dtype=int)
    integrated_pathway_rows: list[dict[str, Any]] = []
    for index, pathway_id in enumerate(path_order):
        family_code = int(family_codes[index])
        family_id = None if family_code < 0 else family_order[family_code]
        integrated_pathway_rows.append(
            {
                "pathway_index": index,
                "pathway_id": pathway_id,
                "level_1_family_index": family_code,
                "level_1_family_id": family_id,
                "matched_pathway_gene_count": int(pathway_size[index]),
                "overlap_degree_count": int(overlap_degree[index]),
                "mean_jaccard_overlap": float(mean_jaccard[index]),
                "maximum_jaccard_overlap": float(max_jaccard[index]),
                "mean_missing_donor_fraction": missing_fraction,
                "empirical_sd_integrated_effect_across_500": float(integrated_empirical_sd[index]),
                "empirical_variance_integrated_effect_across_500": float(
                    integrated_empirical_variance[index]
                ),
                "reported_integrated_model_se_available": False,
                "mean_reported_integrated_model_se": None,
                "empirical_sd_over_mean_model_se": None,
                "empirical_variance_over_estimated_variance": None,
                "median_within_replicate_fl_null_integrated_sd": float(
                    median_fl_integrated_sd[index]
                ),
                "median_within_replicate_fl_null_integrated_mean": float(
                    np.median(replay_integrated_mean[:, index])
                ),
                "median_within_replicate_fl_null_integrated_median": float(
                    np.median(replay_integrated_median[:, index])
                ),
                "empirical_sd_over_median_fl_null_sd": _safe_ratio(
                    float(integrated_empirical_sd[index]), float(median_fl_integrated_sd[index])
                ),
                "empirical_direct_label_integrated_q95": float(empirical_integrated_q95[index]),
                "median_within_replicate_fl_integrated_q95": float(
                    median_fl_integrated_q95[index]
                ),
                "direct_q95_over_median_fl_q95": _safe_ratio(
                    float(empirical_integrated_q95[index]),
                    float(median_fl_integrated_q95[index]),
                ),
                "formal_fl_integrated_reject_count": int(
                    formal_integrated_reject_matrix[:, index].sum()
                ),
                "formal_fl_integrated_reject_rate": float(
                    formal_integrated_reject_matrix[:, index].mean()
                ),
                "hypothetical_signed_integral_empirical_sd_full_covariance": float(
                    full_signed_sd[index]
                ),
                "hypothetical_signed_integral_empirical_sd_diagonal_only": float(
                    diagonal_signed_sd[index]
                ),
                "hypothetical_signed_integral_full_to_diagonal_sd_ratio": _safe_ratio(
                    float(full_signed_sd[index]), float(diagonal_signed_sd[index])
                ),
                "hypothetical_signed_integral_off_diagonal_covariance_contribution": float(
                    covariance_contribution[index]
                ),
            }
        )
    integrated_pathway = pd.DataFrame(integrated_pathway_rows)

    integrated_family_rows: list[dict[str, Any]] = []
    for family_code in tuple(range(N_FAMILIES)) + (-1,):
        members = family_codes == family_code
        family_id = family_order[family_code] if family_code >= 0 else "UNASSIGNED"
        fl_values = formal_integrated_reject_matrix[:, members]
        observed_values = observed_integrated_matrix[:, members]
        integrated_family_rows.append(
            {
                "level_1_family_index": family_code,
                "level_1_family_id": family_id,
                "n_member_pathways": int(members.sum()),
                "pathway_replicate_units": int(fl_values.size),
                "formal_fl_integrated_reject_count": int(fl_values.sum()),
                "formal_fl_integrated_reject_rate": float(fl_values.mean()),
                "empirical_frozen_assignment_integrated_mean": float(observed_values.mean()),
                "empirical_frozen_assignment_integrated_sd": float(
                    np.std(observed_values, ddof=1)
                ),
                "empirical_frozen_assignment_integrated_q95": float(
                    _higher_quantile(observed_values.reshape(-1), 0.95)
                ),
            }
        )
    integrated_family = pd.DataFrame(integrated_family_rows)

    formal_global_reject = formal_curve_reject_matrix.any(axis=1)
    reference_rows: list[dict[str, Any]] = []
    for index, item in enumerate(replay):
        full_q95 = float(item["full_global_q95"])
        reduced_q95 = float(item["reduced_global_q95"])
        reference_rows.append(
            {
                "replicate_index_0based": index,
                "assignment_id": str(assignment_axis.iloc[index]["assignment_id"]),
                "mapping_stream_sha256": item["mapping_stream_sha256"],
                "observed_global_max": float(observed_global[index]),
                "formal_full_band_critical_recovered": float(
                    formal_full_critical.iloc[index]["median"]
                ),
                "replayed_full_model_residual_global_q50": float(item["full_global_q50"]),
                "replayed_full_model_residual_global_q90": float(item["full_global_q90"]),
                "replayed_full_model_residual_global_q95": full_q95,
                "replayed_full_model_residual_global_q99": float(item["full_global_q99"]),
                "replayed_full_model_residual_global_sd": float(item["full_global_sd"]),
                "replayed_reduced_model_residual_global_q50": float(item["reduced_global_q50"]),
                "replayed_reduced_model_residual_global_q90": float(item["reduced_global_q90"]),
                "replayed_reduced_model_residual_global_q95": reduced_q95,
                "replayed_reduced_model_residual_global_q99": float(item["reduced_global_q99"]),
                "replayed_reduced_model_residual_global_sd": float(item["reduced_global_sd"]),
                "replayed_reduced_model_residual_exceedance_count": int(
                    item["reduced_global_exceedance_count"]
                ),
                "observed_over_full_q95": _safe_ratio(float(observed_global[index]), full_q95),
                "observed_over_reduced_q95": _safe_ratio(float(observed_global[index]), reduced_q95),
                "full_q95_over_reduced_q95": _safe_ratio(full_q95, reduced_q95),
                "observed_exceeds_full_q95": bool(observed_global[index] > full_q95),
                "observed_exceeds_reduced_q95": bool(observed_global[index] > reduced_q95),
                "observed_exceeds_full_but_not_reduced_q95": bool(
                    observed_global[index] > full_q95
                    and observed_global[index] <= reduced_q95
                ),
                "replayed_full_model_residual_exceedance_count": int(
                    item["full_global_exceedance_count"]
                ),
                "formal_global_band_covered": bool(global_units[index]),
                "formal_global_curve_reject": bool(formal_global_reject[index]),
                "replay_matches_formal_observed_statistics": True,
                "replay_matches_formal_full_band_critical": True,
            }
        )
    reference_replicate = pd.DataFrame(reference_rows)

    reference_pathway_rows: list[dict[str, Any]] = []
    for index, pathway_id in enumerate(path_order):
        reference_pathway_rows.append(
            {
                "pathway_index": index,
                "pathway_id": pathway_id,
                "empirical_direct_label_curve_q95": float(empirical_curve_q95[index]),
                "median_within_replicate_fl_curve_q95": float(median_fl_curve_q95[index]),
                "direct_curve_q95_over_median_fl_q95": _safe_ratio(
                    float(empirical_curve_q95[index]), float(median_fl_curve_q95[index])
                ),
                "empirical_direct_label_integrated_q95": float(empirical_integrated_q95[index]),
                "median_within_replicate_fl_integrated_q95": float(
                    median_fl_integrated_q95[index]
                ),
                "direct_integrated_q95_over_median_fl_q95": _safe_ratio(
                    float(empirical_integrated_q95[index]),
                    float(median_fl_integrated_q95[index]),
                ),
                "empirical_sd_integrated_effect_across_500": float(integrated_empirical_sd[index]),
                "median_within_replicate_fl_null_integrated_sd": float(
                    median_fl_integrated_sd[index]
                ),
                "empirical_integrated_sd_over_median_fl_null_sd": _safe_ratio(
                    float(integrated_empirical_sd[index]), float(median_fl_integrated_sd[index])
                ),
            }
        )
    reference_pathway = pd.DataFrame(reference_pathway_rows)

    pointwise_count = int(pointwise_contains.sum())
    per_path_count = int(path_units["global_band_pathway_localization_covered"].sum())
    family_count = int(family_units["global_band_family_localization_covered"].sum())
    global_count = int(global_units.sum())
    band_scope = {
        "schema_name": "trajpathmix_cb2_500_failure_localization_band_scope",
        "schema_version": SCHEMA_VERSION,
        "project_id": PROJECT_ID,
        "band_critical_actual_max_scope": "all_50_pathways_x_all_20_supported_bins",
        "support_mask_shape": [N_BINS, N_PATHWAYS],
        "support_mask_true_count": N_BINS * N_PATHWAYS,
        "max_scope_implementation": "max_abs_full_model_residual_bootstrap_t_over_boolean_support",
        "scope_mismatch_hypothesis_supported": False,
        "coverage": {
            "pointwise_zero_coverage": {
                "numerator": pointwise_count,
                "denominator": int(len(curve)),
                "estimate": float(pointwise_count / len(curve)),
                "unit": "pathway_bin_replicate",
            },
            "global_band_pathway_localization_coverage": {
                "numerator": per_path_count,
                "denominator": int(len(path_units)),
                "estimate": float(per_path_count / len(path_units)),
                "unit": "pathway_replicate_using_global_band",
                "not_a_pathway_specific_critical": True,
            },
            "global_band_family_localization_coverage": {
                "numerator": family_count,
                "denominator": int(len(family_units)),
                "estimate": float(family_count / len(family_units)),
                "unit": "family_replicate_using_global_band",
                "not_a_family_specific_critical": True,
            },
            "global_50x20_zero_curve_coverage": {
                "numerator": global_count,
                "denominator": N_REPLICATES,
                "estimate": float(global_count / N_REPLICATES),
                "unit": "replicate",
            },
        },
        "band_failure_pattern": "global_scope_correct_reference_scale_adjudicated_in_direct_label_vs_fl_output",
        "formal_band_failures": int(N_REPLICATES - global_count),
        "formal_global_curve_rejections": int(formal_global_reject.sum()),
        "band_failure_without_global_curve_rejection_count": int(
            np.sum((~global_units) & (~formal_global_reject))
        ),
    }

    empirical_observed_global_q95 = float(_higher_quantile(observed_global, 0.95))
    median_full_q95 = float(np.median(replay_full_q95))
    reduced_q95_values = np.asarray([item["reduced_global_q95"] for item in replay], dtype=float)
    median_reduced_q95 = float(np.median(reduced_q95_values))
    paired_full_fail_reduced_cover = (
        (observed_global > replay_full_q95) & (observed_global <= reduced_q95_values)
    )
    direct_summary = {
        "schema_name": "trajpathmix_cb2_500_failure_localization_direct_label_vs_fl",
        "schema_version": SCHEMA_VERSION,
        "project_id": PROJECT_ID,
        "direct_reference": {
            "assignment_bank": "same_500_frozen_outcome_blind_experiment_balanced_assignments",
            "empirical_quantile_rule": "numpy_higher_order_statistic",
            "diagnostic_only": True,
            "assignment_bank_uniform_over_legal_assignments": False,
            "assignment_bank_used_as_randomization_p_value_denominator": False,
            "p_values_or_rejection_calls_computed": False,
            "formal_method_replacement": False,
        },
        "global_curve": {
            "formal_fl_reject_count": int(formal_global_reject.sum()),
            "formal_fl_reject_rate": float(formal_global_reject.mean()),
            "empirical_observed_global_max_q95": empirical_observed_global_q95,
            "median_internal_reduced_fl_global_q95": median_reduced_q95,
            "empirical_q95_over_median_internal_q95": _safe_ratio(
                empirical_observed_global_q95, median_reduced_q95
            ),
        },
        "simultaneous_band": {
            "formal_full_residual_coverage_count": global_count,
            "formal_full_residual_coverage_rate": float(global_count / N_REPLICATES),
            "empirical_observed_global_max_q95": empirical_observed_global_q95,
            "median_internal_full_residual_global_q95": median_full_q95,
            "empirical_q95_over_median_internal_q95": _safe_ratio(
                empirical_observed_global_q95, median_full_q95
            ),
            "median_full_q95_over_median_reduced_q95": _safe_ratio(
                median_full_q95, median_reduced_q95
            ),
            "paired_observed_exceeds_full_but_not_reduced_count": int(
                paired_full_fail_reduced_cover.sum()
            ),
            "paired_observed_exceeds_full_but_not_reduced_rate": float(
                paired_full_fail_reduced_cover.mean()
            ),
        },
        "integrated": {
            "formal_fl_reject_count": int(formal_integrated_reject_matrix.sum()),
            "formal_fl_reject_rate": float(formal_integrated_reject_matrix.mean()),
            "median_pathway_empirical_sd_over_median_fl_null_sd": float(
                np.median(integrated_empirical_sd / median_fl_integrated_sd)
            ),
            "median_pathway_direct_q95_over_median_fl_q95": float(
                np.median(empirical_integrated_q95 / median_fl_integrated_q95)
            ),
        },
        "pathway_size_and_overlap_descriptive_associations": {
            "formal_integrated_rejection_rate_spearman_pathway_size": _spearman_without_p(
                pathway_size, formal_integrated_reject_matrix.mean(axis=0)
            ),
            "formal_integrated_rejection_rate_spearman_overlap_degree": _spearman_without_p(
                overlap_degree, formal_integrated_reject_matrix.mean(axis=0)
            ),
            "formal_integrated_rejection_rate_spearman_maximum_jaccard": _spearman_without_p(
                max_jaccard, formal_integrated_reject_matrix.mean(axis=0)
            ),
            "p_values_computed": False,
        },
        "missingness_association": {
            "integrated_pathway_level_identifiable": False,
            "reason": "all_50_pathways_share_the_same_20_bin_availability_and_support",
            "mean_missing_donor_fraction": missing_fraction,
            "bin_level_descriptive_output": BIN_DIAGNOSTIC_FILE,
        },
    }

    by = {
        "schema_name": "trajpathmix_cb2_500_failure_localization_by_evaluability",
        "schema_version": SCHEMA_VERSION,
        "project_id": PROJECT_ID,
        "number_of_pathways": 50,
        "harmonic_number": float(np.sum(1.0 / np.arange(1, 51))),
        "alpha": ALPHA,
        "first_by_threshold_alpha_0_05": float(ALPHA / (50.0 * np.sum(1.0 / np.arange(1, 51)))),
        "mc_mappings": N_MAPPINGS,
        "minimum_attainable_p": 1.0 / (N_MAPPINGS + 1.0),
        "first_rank_by_rejection_numerically_attainable": False,
        "by_step_up_rejection_numerically_attainable": True,
        "smallest_rank_at_which_p_0_001_satisfies_by_threshold": 5,
        "five_p_values_at_0_001_are_sufficient_for_step_up_rejection": True,
        "fifth_by_threshold_alpha_0_05": float(
            5.0 * ALPHA / (50.0 * np.sum(1.0 / np.arange(1, 51)))
        ),
        "formal_result": "pass_but_resolution_limited_and_structurally_noninformative",
        "formal_complete_null_any_rejection_count": int(
            pathways.assign(_reject=_read_bool(pathways["BY_reject_alpha"], "BY reject"))
            .groupby("replicate_index_0based")["_reject"]
            .any()
            .sum()
        ),
        "allowed_interpretation": "numerical_resolution_audit_only",
        "forbidden_interpretation": "evidence_that_BY_p_values_are_well_calibrated_or_discovery_ready",
        "requested_blanket_unattainability_flag_corrected": True,
        "correction_reason": "BY_step_up_can_reject_at_rank_5_if_at_least_five_p_values_equal_0_001",
    }

    return {
        "band_scope": band_scope,
        "band_pathway": band_pathway,
        "band_family": band_family,
        "bin_diagnostics": bin_diagnostics,
        "integrated_pathway": integrated_pathway,
        "integrated_family": integrated_family,
        "reference_replicate": reference_replicate,
        "reference_pathway": reference_pathway,
        "direct_summary": direct_summary,
        "by": by,
    }


def _project_state(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_name": "trajpathmix_cb2_500_failure_localization_project_state",
        "schema_version": SCHEMA_VERSION,
        "project_id": PROJECT_ID,
        **INFERENCE_STATE,
        "historical_cb2_500_artifacts_retained": True,
        "historical_inferential_outputs_status": "failed_engine_experimental_reproduction_only",
        "descriptive_effect_curve_allowed": True,
        "descriptive_integrated_point_estimate_allowed": True,
        "data_engineering_and_estimability_gates_retained": True,
        "current_inferential_engine_closed": True,
        "failure_localization_authorized": True,
        "functional_core_v2_implementation_authorized": False,
        "next_stage_authorized": "none",
        "contract_payload_sha256": config["_config_payload_sha256"],
        "timing_computed": False,
        "timing_fields_present": False,
    }


def _input_audit(sources: Mapping[str, Any]) -> dict[str, Any]:
    config = sources["config"]
    return {
        "schema_name": "trajpathmix_cb2_500_failure_localization_input_audit",
        "schema_version": SCHEMA_VERSION,
        "project_id": PROJECT_ID,
        "contract_payload_sha256": config["_config_payload_sha256"],
        "verified_bindings": sources["verified_bindings"],
        "formal_cb2_500_output_validated": True,
        "formal_cb2_500_failed": True,
        "formal_cb2_500_decision_preserved": True,
        "cache_validated": True,
        "frozen_assignments_reused": N_REPLICATES,
        "frozen_mapping_streams_reused": N_REPLICATES,
        "mappings_per_replicate": N_MAPPINGS,
        "new_assignments_generated": False,
        "new_mappings_generated": False,
        "acceptance_threshold_changed": False,
        "pathway_universe_changed": False,
        "raw_expression_read": False,
        "real_condition_labels_read": False,
        "real_condition_contrast_generated": False,
        "injection_recovery_read": False,
        "biological_interpretation_performed": False,
        "timing_computed": False,
        "timing_fields_present": False,
    }


def _decision(analysis: Mapping[str, Any]) -> dict[str, Any]:
    direct = analysis["direct_summary"]
    band_ratio = direct["simultaneous_band"]["empirical_q95_over_median_internal_q95"]
    integrated_ratio = direct["integrated"][
        "median_pathway_empirical_sd_over_median_fl_null_sd"
    ]
    reduced_ratio = direct["global_curve"]["empirical_q95_over_median_internal_q95"]
    paired_divergence_rate = direct["simultaneous_band"][
        "paired_observed_exceeds_full_but_not_reduced_rate"
    ]
    clear_reference_mismatch = bool(
        band_ratio is not None
        and integrated_ratio is not None
        and reduced_ratio is not None
        and paired_divergence_rate is not None
        and band_ratio >= 1.10
        and integrated_ratio >= 1.10
        and reduced_ratio > 1.0
        and paired_divergence_rate >= 0.50
    )
    supported_findings = (
        [
            "full_and_reduced_residual_reference_scales_are_too_narrow_relative_to_the_frozen_assignment_bank_empirical_distribution",
            "raw_L1_integrated_effect_reference_scale_is_too_narrow_relative_to_across_assignment_variation",
        ]
        if clear_reference_mismatch
        else []
    )
    return {
        "schema_name": "trajpathmix_cb2_500_failure_localization_decision",
        "schema_version": SCHEMA_VERSION,
        "project_id": PROJECT_ID,
        "formal_cb2_500_decision_preserved": True,
        "current_inferential_engine_closed": True,
        "band_scope_mismatch_root_cause_supported": False,
        "integrated_analytic_se_auditable": False,
        "integrated_analytic_se_reason": "no_formal_within_replicate_model_based_SE_or_cross_bin_covariance_estimator_exists_for_the_L1_integrated_endpoint",
        "reference_distribution_mismatch_supported": clear_reference_mismatch,
        "supported_findings": supported_findings,
        "falsifiable_mechanism_candidates_not_yet_causally_adjudicated": [
            "availability_restricted_raw_residual_reference_perturbs_only_27_of_75_donors",
            "raw_residual_replay_has_no_leverage_or_heteroskedasticity_correction_despite_high_rank_to_n_ratios",
            "integrated_endpoint_is_nonstudentized_weighted_L1_not_a_signed_linear_contrast",
        ],
        "reference_mismatch_materiality_rule": {
            "empirical_observed_global_q95_over_median_full_q95_min": 1.10,
            "median_pathway_empirical_integrated_sd_over_median_fl_null_sd_min": 1.10,
            "empirical_observed_global_q95_over_median_reduced_q95_strictly_greater_than": 1.0,
            "paired_observed_exceeds_full_but_not_reduced_rate_min": 0.50,
        },
        "small_number_of_falsifiable_reference_failure_modes_localized": clear_reference_mismatch,
        "causal_mechanism_adjudicated": False,
        "one_theory_driven_core_reconstruction_may_be_proposed": clear_reference_mismatch,
        "functional_core_v2_implementation_authorized": False,
        "allowed_future_proposal": (
            "donor_level_influence_function_multiplier_or_restricted_wild_cluster_core"
            if clear_reference_mismatch
            else "none"
        ),
        "variance_floor_tweak_allowed": False,
        "critical_factor_tweak_allowed": False,
        "mapping_count_only_tweak_allowed": False,
        "acceptance_threshold_change_allowed": False,
        "pathway_deletion_allowed": False,
        "cb2_500_rerun_authorized": False,
        "cb2_2000_allowed": False,
        "cb3_injection_allowed": False,
        "real_condition_contrast_allowed": False,
        "timing_allowed": False,
        "next_stage_authorized": "none",
        "decision": "retain_descriptive_framework_close_current_inferential_engine_pending_separate_v2_authorization",
        "timing_computed": False,
        "timing_fields_present": False,
    }


def build_cb2_500_failure_localization(
    config_path: str | Path = DEFAULT_CONFIG_FILE,
    repository_root: str | Path = ".",
    output_dir: str | Path | None = None,
    *,
    explicit_execution_authorization: bool = False,
    processes: int = 16,
    chunk_size: int = 32,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    _require(explicit_execution_authorization, "Explicit localization execution authorization required")
    _require(int(processes) > 0, "processes must be positive")
    _require(int(chunk_size) > 0, "chunk_size must be positive")
    resolved_config = _resolve(root, config_path)
    config = load_failure_localization_contract(resolved_config, require_frozen=True)
    target = _resolve(root, config["output_contract"]["default_output_dir"])
    if output_dir is not None:
        explicit_target = _resolve(root, output_dir)
        _require(
            explicit_target == target,
            "Localization execution output is fixed; alternate targets are forbidden",
        )
    if target.exists():
        raise FileExistsError(f"Create-only localization output exists: {target}")
    incomplete = target.parent / f"{target.name}.incomplete"
    if incomplete.exists():
        raise FileExistsError(
            f"Prior incomplete localization evidence exists; retry/resume forbidden: {incomplete}"
        )
    sources = validate_failure_localization_sources(resolved_config, root)
    _require(
        sources["config"]["_config_payload_sha256"] == config["_config_payload_sha256"],
        "Contract changed during source validation",
    )
    incomplete.mkdir(parents=True, exist_ok=False)
    _write_json(_project_state(config), incomplete / STATE_FILE)
    _write_json(_input_audit(sources), incomplete / INPUT_AUDIT_FILE)

    stage = "reference_replay"
    try:
        replay = _replay_references(
            sources["cache_dir"], processes=int(processes), chunk_size=int(chunk_size)
        )
        stage = "formal_artifact_analysis"
        analysis = _analyze(sources["formal_output_dir"], sources["cache_dir"], replay)
        stage = "artifact_materialization"
        _write_json(analysis["band_scope"], incomplete / BAND_SCOPE_FILE)
        _write_table(analysis["band_pathway"], incomplete / BAND_PATHWAY_FILE)
        _write_table(analysis["band_family"], incomplete / BAND_FAMILY_FILE)
        _write_table(analysis["bin_diagnostics"], incomplete / BIN_DIAGNOSTIC_FILE)
        _write_table(analysis["integrated_pathway"], incomplete / INTEGRATED_PATHWAY_FILE)
        _write_table(analysis["integrated_family"], incomplete / INTEGRATED_FAMILY_FILE)
        _write_table(analysis["reference_replicate"], incomplete / REFERENCE_REPLICATE_FILE)
        _write_table(analysis["reference_pathway"], incomplete / REFERENCE_PATHWAY_FILE)
        _write_json(analysis["direct_summary"], incomplete / DIRECT_REFERENCE_FILE)
        _write_json(analysis["by"], incomplete / BY_FILE)
        decision = _decision(analysis)
        _write_json(decision, incomplete / DECISION_FILE)
        passport = {
            "schema_name": "trajpathmix_cb2_500_failure_localization_material_passport",
            "schema_version": SCHEMA_VERSION,
            "project_id": PROJECT_ID,
            "origin_skill": "academic_research_suite_experiment_agent",
            "origin_mode": "validate",
            "verification_status": "ANALYZED",
            "version_label": "validation_v1",
            "contract_payload_sha256": config["_config_payload_sha256"],
            "formal_statistical_outputs_consumed": True,
            "functional_kernel_replayed": True,
            "replay_scope": "same_500_assignments_same_999_mapping_streams_compact_diagnostic_projections_only",
            "mapping_level_null_arrays_persisted": False,
            "raw_expression_read": False,
            "new_assignments_generated": False,
            "new_mappings_generated": False,
            "real_condition_contrast_generated": False,
            "pathway_discovery_performed": False,
            "biological_interpretation_performed": False,
            "timing_computed": False,
            "timing_fields_present": False,
            "next_stage_authorized": "none",
            "claim_ceiling": "complete_null_failure_localization_only",
            "fallacy_scan": _fallacy_scan(),
        }
        _write_json(passport, incomplete / PASSPORT_FILE)
        artifact_hashes = {
            name: _hash_file(incomplete / name)
            for name in OUTPUT_FILES
            if name != BUILD_RECORD_FILE
        }
        build_record = {
            "schema_name": "trajpathmix_cb2_500_failure_localization_build_record",
            "schema_version": SCHEMA_VERSION,
            "project_id": PROJECT_ID,
            "contract_payload_sha256": config["_config_payload_sha256"],
            "artifact_sha256": artifact_hashes,
            "formal_cb2_500_decision_sha256": _hash_file(
                sources["formal_output_dir"] / FORMAL_DECISION_FILE
            ),
            "formal_cb2_500_summary_sha256": _hash_file(
                sources["formal_output_dir"] / FORMAL_SUMMARY_FILE
            ),
            "cache_manifest_sha256": _hash_file(sources["cache_dir"] / CACHE_MANIFEST_FILE),
            "replicates_replayed_once": N_REPLICATES,
            "mappings_per_replicate_reused": N_MAPPINGS,
            "worker_processes": int(processes),
            "mapping_chunk_size": int(chunk_size),
            "automatic_retry_used": False,
            "automatic_resume_used": False,
            "new_assignments_generated": False,
            "new_mappings_generated": False,
            "acceptance_threshold_changed": False,
            "mapping_level_null_arrays_persisted": False,
            "real_condition_contrast_generated": False,
            "timing_computed": False,
            "timing_fields_present": False,
        }
        _write_json(build_record, incomplete / BUILD_RECORD_FILE)
        stage = "prepublish_validation"
        validation = validate_cb2_500_failure_localization_output(
            resolved_config, root, incomplete
        )
        stage = "atomic_publish"
        os.replace(incomplete, target)
    except BaseException as exc:
        if incomplete.is_dir():
            try:
                _write_json(
                    {
                        "schema_name": "trajpathmix_cb2_500_failure_localization_execution_failure",
                        "schema_version": SCHEMA_VERSION,
                        "project_id": PROJECT_ID,
                        "failure_stage": stage,
                        "exception_type": type(exc).__name__,
                        "exception_detail": str(exc),
                        "automatic_retry_used": False,
                        "automatic_resume_used": False,
                        "incomplete_evidence_preserved": True,
                        "new_assignments_generated": False,
                        "new_mappings_generated": False,
                        "next_stage_authorized": "none",
                    },
                    incomplete / "CB2_500_FAILURE_LOCALIZATION_EXECUTION_FAILURE_v1.json",
                )
            except BaseException:
                pass
        raise
    validation["output_dir"] = str(target)
    return validation


def validate_cb2_500_failure_localization_output(
    config_path: str | Path = DEFAULT_CONFIG_FILE,
    repository_root: str | Path = ".",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    sources = validate_failure_localization_sources(config_path, root)
    config = sources["config"]
    directory = (
        _resolve(root, config["output_contract"]["default_output_dir"])
        if output_dir is None
        else Path(output_dir).resolve()
    )
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    observed_files = {path.name for path in directory.iterdir() if path.is_file()}
    _require(observed_files == set(OUTPUT_FILES), "Localization output file set mismatch")
    _require(not any(path.is_dir() for path in directory.iterdir()), "Unexpected output subdirectory")

    json_values: dict[str, Any] = {}
    for name, keys in config["output_contract"]["json_exact_keys"].items():
        value = _strict_json_load(directory / name)
        _require(isinstance(value, Mapping), f"JSON output is not an object: {name}")
        _require(set(value) == set(keys), f"Unexpected top-level JSON schema in {name}")
        _require(value.get("schema_version") == SCHEMA_VERSION, f"JSON version mismatch: {name}")
        _require(value.get("project_id") == PROJECT_ID, f"JSON project mismatch: {name}")
        json_values[name] = value

    state = json_values[STATE_FILE]
    for key, expected in INFERENCE_STATE.items():
        _require(state.get(key) is expected, f"Inference closure flag mismatch: {key}")
    _require(state.get("current_inferential_engine_closed") is True, "Engine closure missing")
    decision = json_values[DECISION_FILE]
    _require(decision.get("formal_cb2_500_decision_preserved") is True, "Formal decision changed")
    _require(decision.get("functional_core_v2_implementation_authorized") is False, "v2 opened")
    _require(decision.get("cb2_2000_allowed") is False, "CB2-2000 opened")
    _require(decision.get("cb3_injection_allowed") is False, "CB3 opened")
    _require(decision.get("next_stage_authorized") == "none", "Next stage opened")
    by = json_values[BY_FILE]
    _require(by.get("number_of_pathways") == 50, "BY pathway count changed")
    _require(abs(float(by.get("harmonic_number")) - 4.499205338329425) < 1.0e-12, "BY harmonic number changed")
    _require(abs(float(by.get("minimum_attainable_p")) - 0.001) < 1.0e-15, "BY p resolution changed")
    _require(
        by.get("first_rank_by_rejection_numerically_attainable") is False,
        "BY first-rank attainability mislabeled",
    )
    _require(
        by.get("by_step_up_rejection_numerically_attainable") is True,
        "BY step-up attainability mislabeled",
    )
    _require(
        by.get("smallest_rank_at_which_p_0_001_satisfies_by_threshold") == 5,
        "BY p=0.001 first-satisfying rank changed",
    )
    _require(
        by.get("five_p_values_at_0_001_are_sufficient_for_step_up_rejection") is True,
        "BY sufficient construction changed",
    )
    _require(
        by.get("formal_result")
        == "pass_but_resolution_limited_and_structurally_noninformative",
        "BY status changed",
    )

    table_specs = {
        BAND_PATHWAY_FILE: (N_PATHWAYS, ("pathway_index", "pathway_id")),
        BAND_FAMILY_FILE: (N_FAMILIES, ("family_index", "family_id")),
        BIN_DIAGNOSTIC_FILE: (N_BINS, ("bin_index",)),
        INTEGRATED_PATHWAY_FILE: (N_PATHWAYS, ("pathway_index", "pathway_id")),
        INTEGRATED_FAMILY_FILE: (N_FAMILIES + 1, ("level_1_family_index", "level_1_family_id")),
        REFERENCE_REPLICATE_FILE: (N_REPLICATES, ("replicate_index_0based", "assignment_id")),
        REFERENCE_PATHWAY_FILE: (N_PATHWAYS, ("pathway_index", "pathway_id")),
    }
    frames: dict[str, pd.DataFrame] = {}
    for name, (rows, key) in table_specs.items():
        raw = (directory / name).read_bytes()
        _require(b"\r" not in raw and raw.endswith(b"\n"), f"TSV lexical contract failed: {name}")
        frame = pd.read_csv(directory / name, sep="\t", keep_default_na=False)
        _require(len(frame) == rows, f"Unexpected row count in {name}")
        _require(
            tuple(frame.columns)
            == tuple(config["output_contract"]["tables"][name]["columns"]),
            f"Unexpected column schema in {name}",
        )
        _require(not frame.duplicated(list(key)).any(), f"Duplicate key in {name}")
        frames[name] = frame
    _require(
        frames[REFERENCE_REPLICATE_FILE]["replicate_index_0based"].tolist()
        == list(range(N_REPLICATES)),
        "Reference replicate order changed",
    )
    _require(
        frames[INTEGRATED_PATHWAY_FILE]["reported_integrated_model_se_available"]
        .astype(str)
        .str.lower()
        .eq("false")
        .all(),
        "A nonexistent integrated model SE was claimed",
    )

    band = json_values[BAND_SCOPE_FILE]
    _require(
        band.get("band_critical_actual_max_scope")
        == "all_50_pathways_x_all_20_supported_bins",
        "Band max scope changed",
    )
    _require(band.get("scope_mismatch_hypothesis_supported") is False, "Scope mismatch mislabeled")
    formal_curve = pd.read_csv(sources["formal_output_dir"] / CURVE_FILE, sep="\t")
    numeric_contains = (
        (formal_curve["simultaneous_lower"].to_numpy(dtype=float) <= 0.0)
        & (formal_curve["simultaneous_upper"].to_numpy(dtype=float) >= 0.0)
    )
    global_covered = pd.Series(numeric_contains).groupby(
        formal_curve["replicate_index_0based"]
    ).all()
    global_record = band["coverage"]["global_50x20_zero_curve_coverage"]
    _require(int(global_record["numerator"]) == int(global_covered.sum()), "Global coverage mismatch")
    _require(int(global_record["denominator"]) == N_REPLICATES, "Global coverage denominator")
    pointwise_contains = (
        (formal_curve["pointwise_lower"].to_numpy(dtype=float) <= 0.0)
        & (formal_curve["pointwise_upper"].to_numpy(dtype=float) >= 0.0)
    )
    pointwise_record = band["coverage"]["pointwise_zero_coverage"]
    _require(
        int(pointwise_record["numerator"]) == int(pointwise_contains.sum())
        and int(pointwise_record["denominator"]) == len(formal_curve),
        "Pointwise coverage mismatch",
    )
    pathway_covered = (
        pd.Series(numeric_contains)
        .groupby(
            [
                formal_curve["replicate_index_0based"],
                formal_curve["pathway_id"],
            ]
        )
        .all()
    )
    pathway_record = band["coverage"]["global_band_pathway_localization_coverage"]
    _require(
        int(pathway_record["numerator"]) == int(pathway_covered.sum())
        and int(pathway_record["denominator"]) == len(pathway_covered),
        "Pathway localization coverage mismatch",
    )
    assigned = formal_curve["level_1_family_id"].notna()
    family_covered = (
        pd.Series(numeric_contains[assigned.to_numpy(dtype=bool)])
        .groupby(
            [
                formal_curve.loc[assigned, "replicate_index_0based"].reset_index(drop=True),
                formal_curve.loc[assigned, "level_1_family_id"].reset_index(drop=True),
            ]
        )
        .all()
    )
    family_record = band["coverage"]["global_band_family_localization_coverage"]
    _require(
        int(family_record["numerator"]) == int(family_covered.sum())
        and int(family_record["denominator"]) == len(family_covered),
        "Family localization coverage mismatch",
    )
    reference = frames[REFERENCE_REPLICATE_FILE]
    formal_global_max = (
        formal_curve.assign(
            _abs_t=np.abs(
                formal_curve["effect"].to_numpy(dtype=float)
                / formal_curve["standard_error"].to_numpy(dtype=float)
            )
        )
        .groupby("replicate_index_0based")["_abs_t"]
        .max()
        .to_numpy(dtype=float)
    )
    _require(
        np.max(
            np.abs(reference["observed_global_max"].to_numpy(dtype=float) - formal_global_max)
        )
        <= NUMERICAL_TOLERANCE,
        "Reference observed maxima differ from formal output",
    )
    formal_effect = formal_curve["effect"].to_numpy(dtype=float)
    formal_se = formal_curve["standard_error"].to_numpy(dtype=float)
    formal_full_critical = (
        pd.Series(
            (formal_curve["simultaneous_upper"].to_numpy(dtype=float) - formal_effect)
            / formal_se
        )
        .groupby(formal_curve["replicate_index_0based"])
        .median()
        .to_numpy(dtype=float)
    )
    _require(
        np.max(
            np.abs(
                reference["formal_full_band_critical_recovered"].to_numpy(dtype=float)
                - formal_full_critical
            )
        )
        <= NUMERICAL_TOLERANCE,
        "Recovered full-band critical differs from formal bounds",
    )
    _require(
        np.max(
            np.abs(
                reference["replayed_full_model_residual_global_q95"].to_numpy(dtype=float)
                - formal_full_critical
            )
        )
        <= NUMERICAL_TOLERANCE,
        "Replayed full-band critical differs from formal bounds",
    )
    mapping_audit = pd.read_csv(
        sources["cache_dir"] / "residual_mapping_bank_stream_audit_v1.tsv",
        sep="\t",
        dtype="string",
    )
    _require(
        reference["mapping_stream_sha256"].astype(str).tolist()
        == mapping_audit["stream_sha256"].astype(str).tolist(),
        "Reference mapping stream hashes changed",
    )
    _require(
        reference["replay_matches_formal_observed_statistics"]
        .astype(str)
        .str.lower()
        .eq("true")
        .all(),
        "Replay/formal statistic match flag false",
    )
    passport = json_values[PASSPORT_FILE]
    _require(passport.get("verification_status") == "ANALYZED", "Passport verification overclaimed")
    _require(passport.get("fallacy_scan", {}).get("coverage") == "11_of_11", "Fallacy scan incomplete")
    _require(passport.get("next_stage_authorized") == "none", "Passport opens a next stage")
    direct_summary = json_values[DIRECT_REFERENCE_FILE]
    _require(
        direct_summary.get("direct_reference", {}).get(
            "assignment_bank_used_as_randomization_p_value_denominator"
        )
        is False,
        "Frozen assignment bank was used as a p-value denominator",
    )
    reference_values = frames[REFERENCE_REPLICATE_FILE]
    reference_pathway_values = frames[REFERENCE_PATHWAY_FILE]
    empirical_observed_q95 = float(
        _higher_quantile(reference_values["observed_global_max"].to_numpy(dtype=float), 0.95)
    )
    median_full_q95 = float(
        np.median(
            reference_values["replayed_full_model_residual_global_q95"].to_numpy(
                dtype=float
            )
        )
    )
    median_reduced_q95 = float(
        np.median(
            reference_values["replayed_reduced_model_residual_global_q95"].to_numpy(
                dtype=float
            )
        )
    )
    paired_rate = float(
        reference_values["observed_exceeds_full_but_not_reduced_q95"]
        .astype(str)
        .str.lower()
        .eq("true")
        .mean()
    )
    median_integrated_sd_ratio = float(
        np.median(
            reference_pathway_values[
                "empirical_integrated_sd_over_median_fl_null_sd"
            ].to_numpy(dtype=float)
        )
    )
    _require(
        abs(
            float(direct_summary["global_curve"]["empirical_observed_global_max_q95"])
            - empirical_observed_q95
        )
        <= NUMERICAL_TOLERANCE,
        "Direct-reference observed q95 summary mismatch",
    )
    _require(
        abs(
            float(direct_summary["global_curve"]["median_internal_reduced_fl_global_q95"])
            - median_reduced_q95
        )
        <= NUMERICAL_TOLERANCE,
        "Reduced-reference q95 summary mismatch",
    )
    _require(
        abs(
            float(
                direct_summary["global_curve"][
                    "empirical_q95_over_median_internal_q95"
                ]
            )
            - empirical_observed_q95 / median_reduced_q95
        )
        <= NUMERICAL_TOLERANCE,
        "Reduced-reference q95 ratio mismatch",
    )
    _require(
        abs(
            float(
                direct_summary["simultaneous_band"][
                    "median_internal_full_residual_global_q95"
                ]
            )
            - median_full_q95
        )
        <= NUMERICAL_TOLERANCE,
        "Full-reference q95 summary mismatch",
    )
    _require(
        abs(
            float(
                direct_summary["simultaneous_band"][
                    "empirical_q95_over_median_internal_q95"
                ]
            )
            - empirical_observed_q95 / median_full_q95
        )
        <= NUMERICAL_TOLERANCE,
        "Full-reference q95 ratio mismatch",
    )
    _require(
        abs(
            float(
                direct_summary["simultaneous_band"][
                    "paired_observed_exceeds_full_but_not_reduced_rate"
                ]
            )
            - paired_rate
        )
        <= NUMERICAL_TOLERANCE,
        "Paired reference-divergence summary mismatch",
    )
    _require(
        abs(
            float(
                direct_summary["integrated"][
                    "median_pathway_empirical_sd_over_median_fl_null_sd"
                ]
            )
            - median_integrated_sd_ratio
        )
        <= NUMERICAL_TOLERANCE,
        "Integrated reference-scale summary mismatch",
    )
    recomputed_decision = _decision({"direct_summary": direct_summary})
    _require(decision == recomputed_decision, "Localization decision was not deterministically recomputed")

    input_audit = json_values[INPUT_AUDIT_FILE]
    _require(
        input_audit.get("verified_bindings") == sources["verified_bindings"],
        "Input-audit binding set changed",
    )
    build = json_values[BUILD_RECORD_FILE]
    expected_hashes = {
        name: _hash_file(directory / name)
        for name in OUTPUT_FILES
        if name != BUILD_RECORD_FILE
    }
    _require(build.get("artifact_sha256") == expected_hashes, "Output artifact hash mismatch")
    _require(
        build.get("formal_cb2_500_decision_sha256")
        == _hash_file(sources["formal_output_dir"] / FORMAL_DECISION_FILE),
        "Formal decision binding changed in build record",
    )
    _require(
        build.get("formal_cb2_500_summary_sha256")
        == _hash_file(sources["formal_output_dir"] / FORMAL_SUMMARY_FILE),
        "Formal summary binding changed in build record",
    )
    _require(
        build.get("cache_manifest_sha256")
        == _hash_file(sources["cache_dir"] / CACHE_MANIFEST_FILE),
        "Cache manifest binding changed in build record",
    )
    _require(build.get("replicates_replayed_once") == N_REPLICATES, "Replay count changed")
    _require(build.get("automatic_retry_used") is False, "Automatic retry recorded")
    _require(build.get("automatic_resume_used") is False, "Automatic resume recorded")
    return {
        "valid": True,
        "project_id": PROJECT_ID,
        "output_dir": str(directory),
        "contract_payload_sha256": config["_config_payload_sha256"],
        "decision_sha256": _hash_file(directory / DECISION_FILE),
        "build_record_sha256": _hash_file(directory / BUILD_RECORD_FILE),
        "artifact_sha256": expected_hashes,
    }


__all__ = [
    "PROJECT_ID",
    "INFERENCE_STATE",
    "FailureLocalizationContractError",
    "load_failure_localization_contract",
    "validate_failure_localization_sources",
    "build_cb2_500_failure_localization",
    "validate_cb2_500_failure_localization_output",
]
