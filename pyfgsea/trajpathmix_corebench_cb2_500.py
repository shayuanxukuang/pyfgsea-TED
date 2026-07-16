"""Append-only CB2-500 cache and execution adapter.

The module deliberately separates three operations:

* validation of the frozen execution contract and its SHA-256 bindings;
* create-only construction/validation of the eight frozen statistical-
  benchmark cache products used by CB2-500; and
* a pure-array, QR-only Freedman--Lane engine that consumes explicit designs
  and whole-donor mappings.

Importing this module never reads expression values and never starts CB2-500.
The two state-changing entry points require an explicit authorization argument,
refuse scaffold/unfrozen contracts, use a temporary sibling directory, and
publish with an atomic rename.  No assignment generator, real condition label,
timing detector, or event detector is imported here.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np
import pandas as pd
import yaml


SCHEMA_NAME = "trajpathmix_corebench_cb2_500_execution_contract"
SCHEMA_VERSION = "1.0.0"
EXECUTION_ID = "trajpathmix_corebench_cb2_500_execution_v1"
IMPLEMENTATION_FILE = "pyfgsea/trajpathmix_corebench_cb2_500.py"
DEFAULT_CONFIG_FILE = "config/trajpathmix_corebench_cb2_500_execution_v1.yaml"

RANK_RELATIVE_TOLERANCE = 1.0e-10
NUMERICAL_TOLERANCE = 1.0e-12
DEFAULT_ALPHA = 0.05
DEFAULT_CHUNK_SIZE = 32
DEFAULT_PROCESSES = 16

CACHE_FILES = (
    "donor_bin_gene_pseudobulk_float64_v1.npy",
    "pathway_membership_matrix_bool_v1.npy",
    "donor_bin_pathway_scores_float64_v1.npy",
    "bin_specific_nuisance_design_cache_factorized_v1.npz",
    "availability_signatures_v1.tsv",
    "residual_mapping_bank_uint8_v1.npy",
    "pathway_family_index_v1.tsv",
    "simultaneous_band_family_mask_bool_v1.npy",
)
CACHE_AXIS_FILES = (
    "donor_axis_v1.tsv",
    "bin_axis_v1.tsv",
    "matched_gene_axis_v1.tsv",
    "pathway_axis_v1.tsv",
    "assignment_axis_v1.tsv",
    "experiment_axis_v1.tsv",
    "family_axis_v1.tsv",
)
CACHE_MATCHED_SOURCE_AUDIT_FILE = "pathway_membership_matched_source_rows_audit_v1.tsv"
CACHE_AUDIT_FILE = "residual_mapping_bank_stream_audit_v1.tsv"
CACHE_OVERLAP_AUDIT_FILE = "cb2_500_pre_scoring_overlap_audit_v1.json"
CACHE_MANIFEST_FILE = "cb2_500_cache_manifest_v1.json"
CACHE_BUILD_RECORD_FILE = "cb2_500_cache_build_record_v1.json"

OVERLAP_AUDIT_FILE = "cb2_500_pre_scoring_overlap_audit_v1.json"
REPLICATE_FILE = "cb2_500_replicate_manifest_v1.tsv"
CURVE_FILE = "cb2_500_curve_bin_metrics_v1.tsv"
PATHWAY_FILE = "cb2_500_pathway_replicate_metrics_v1.tsv"
FAMILY_FILE = "cb2_500_family_replicate_metrics_v1.tsv"
MAPPING_OUTPUT_FILE = "cb2_500_mapping_audit_v1.tsv"
REFUSAL_FILE = "cb2_500_refusal_audit_v1.tsv"
PVALUE_DIAGNOSTICS_FILE = "cb2_500_pvalue_diagnostics_v1.tsv"
SUMMARY_FILE = "cb2_500_acceptance_summary_v1.json"
DECISION_FILE = "CB2_500_ACCEPTANCE_DECISION_v1.json"
PASSPORT_FILE = "cb2_500_material_passport_v1.json"
RUN_BUILD_RECORD_FILE = "cb2_500_build_record_v1.json"

OUTPUT_FILES = (
    OVERLAP_AUDIT_FILE,
    REPLICATE_FILE,
    CURVE_FILE,
    PATHWAY_FILE,
    FAMILY_FILE,
    MAPPING_OUTPUT_FILE,
    REFUSAL_FILE,
    PVALUE_DIAGNOSTICS_FILE,
    SUMMARY_FILE,
    DECISION_FILE,
    PASSPORT_FILE,
    RUN_BUILD_RECORD_FILE,
)

_OVERLAP_JSON_KEYS = {
    "schema_name", "schema_version", "execution_id", "coordinate_gene_count",
    "frozen_pathway_unique_gene_count", "coordinate_pathway_gene_overlap_count",
    "coordinate_pathway_gene_overlap", "coordinate_gene_folds_sha256",
    "coordinate_gene_exclusion_audit_sha256", "frozen_pathway_universe_sha256",
    "pass", "pathway_scoring_allowed", "biological_interpretation",
    "timing_computed", "timing_fields_present",
}
_PASSPORT_JSON_KEYS = {
    "schema_name", "schema_version", "execution_id",
    "execution_config_payload_sha256", "cache_manifest_sha256",
    "acceptance_decision_sha256", "statistical_benchmark_only",
    "biological_interpretation", "real_condition_contrast_generated",
    "assignment_generator_imported_or_called", "next_stage_authorized",
    "data_provenance", "code_provenance", "fallacy_scan", "safeguards",
    "verification_status", "timing_computed", "timing_fields_present",
}
_BUILD_RECORD_JSON_KEYS = {
    "schema_name", "schema_version", "execution_id",
    "execution_config_payload_sha256", "cache_manifest_sha256", "artifact_sha256",
    "replicates_attempted_once", "automatic_resume_used", "automatic_retry_used",
    "worker_processes", "mapping_chunk_size", "assignment_generator_imported_or_called",
    "real_condition_contrast_generated", "timing_computed", "timing_fields_present",
}
_ARS_FALLACY_KEYS = {
    "simpsons_paradox", "ecological_fallacy", "berksons_paradox", "collider_bias",
    "base_rate_neglect", "regression_to_mean", "survivorship_bias",
    "look_elsewhere_effect", "garden_of_forking_paths",
    "correlation_not_causation", "reverse_causality",
}

ASSIGNMENT_COLUMNS = (
    "assignment_id",
    "assignment_sha256",
    "donor_id",
    "pseudo_condition",
    "pseudo_case",
)
FORBIDDEN_RESULT_TOKENS = (
    "onset",
    "duration",
    "phase",
    "delay",
    "peak_location",
    "peak_time",
    "heterochrony",
    "transient",
    "sustained",
    "event_support",
    "event_time",
)


class CB2500ContractError(ValueError):
    """Raised when frozen execution/cache provenance is not satisfied."""


class CB2500DesignError(CB2500ContractError):
    """Raised when a required functional design is not estimable."""


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


def _hash_array(values: Any, dtype: str | np.dtype | None = None) -> str:
    array = np.asarray(values, dtype=dtype)
    array = np.ascontiguousarray(array)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        result = value.tolist()
        return _json_safe(result)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _strict_json_load(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            CB2500ContractError(f"Non-finite JSON constant {value!r} in {path}")
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
        raise CB2500ContractError(f"Non-finite JSON value at {label}")


def _assert_formal_output_firewall(value: Any, label: str = "root") -> None:
    """Reject scientific-event/timing fields except two required false flags."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in {"timing_computed", "timing_fields_present"}:
                if child is not False:
                    raise CB2500ContractError(f"{label}.{key} must be false")
                continue
            if "timing" in normalized or any(
                token in normalized for token in FORBIDDEN_RESULT_TOKENS
            ):
                raise CB2500ContractError(
                    f"Forbidden scientific-event/timing key at {label}.{key}"
                )
            _assert_formal_output_firewall(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_formal_output_firewall(child, f"{label}[{index}]")


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    _assert_json_finite(value)
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)


def _table_text(frame: pd.DataFrame) -> str:
    serial = frame.copy()
    for column in serial.columns:
        if pd.api.types.is_bool_dtype(serial[column].dtype):
            serial[column] = serial[column].map({True: "true", False: "false"})
    output = io.StringIO()
    serial.to_csv(
        output,
        sep="\t",
        index=False,
        na_rep="NA",
        lineterminator="\n",
        float_format=lambda value: repr(float(value)),
    )
    return output.getvalue()


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(_table_text(frame))


def _write_npy(path: Path, values: Any) -> None:
    with path.open("xb") as handle:
        np.save(handle, np.asarray(values), allow_pickle=False)


def _write_deterministic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    """Write an NPZ with stable member order and timestamps."""

    with path.open("xb") as raw:
        with zipfile.ZipFile(raw, mode="w", compression=zipfile.ZIP_STORED) as archive:
            for name in sorted(arrays):
                if not name or "/" in name or "\\" in name:
                    raise CB2500ContractError(f"Invalid NPZ member name: {name!r}")
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o600 << 16
                with archive.open(info, mode="w", force_zip64=True) as member:
                    np.lib.format.write_array(
                        member, np.asarray(arrays[name]), allow_pickle=False
                    )


def _repo_file(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise CB2500ContractError("CB2-500 binding is not repository-local") from exc
    return path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CB2500ContractError(message)


def _require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise CB2500ContractError(
            f"CB2-500 mismatch for {label}: expected {expected!r}, observed {observed!r}"
        )


def _is_placeholder(value: Any) -> bool:
    text = str(value).strip().upper()
    return not text or text.startswith("TO_BE_") or text.startswith("<")


def validate_cb2_500_execution_config(
    config: Mapping[str, Any], *, require_frozen: bool = True
) -> dict[str, Any]:
    """Validate the append-only execution contract without touching data files."""

    _require_equal(config.get("schema_name"), SCHEMA_NAME, "schema_name")
    _require_equal(config.get("schema_version"), SCHEMA_VERSION, "schema_version")
    _require_equal(config.get("execution_id"), EXECUTION_ID, "execution_id")
    semantics = config.get("append_only_semantics", {})
    for key, expected in {
        "append_only": True,
        "modifies_parent_corebench_contract": False,
        "modifies_cb2_functional_null_amendment_v1": False,
        "modifies_cb2_implementation_freeze_v1": False,
        "modifies_cb2a_v2_decision": False,
        "overwrites_prior_artifacts": False,
    }.items():
        _require_equal(semantics.get(key), expected, f"append_only_semantics.{key}")
    population = config.get("population_and_assignment_contract", {})
    for key, expected in {
        "n_replicates": 500,
        "assignment_generator_may_be_called": False,
        "n_donors": 75,
        "pseudo_case_donors": 37,
        "pseudo_control_donors": 38,
    }.items():
        _require_equal(population.get(key), expected, f"population.{key}")
    mapping = config.get("seed_and_mapping_contract", {})
    for key, expected in {
        "mappings_per_replicate": 999,
        "mapping_unit": "whole_donor",
        "identity_mapping_excluded": True,
        "mappings_unique_within_replicate": True,
        "generate_exactly_once_per_replicate": True,
    }.items():
        _require_equal(mapping.get(key), expected, f"seed_and_mapping_contract.{key}")
    pathways = config.get("pathway_and_family_contract", {})
    for key, expected in {
        "n_pathways": 50,
        "n_level_1_families": 13,
        "n_level_1_family_member_pathways": 36,
        "unassigned_level_1_pathway_count": 14,
    }.items():
        _require_equal(pathways.get(key), expected, f"pathway_and_family_contract.{key}")
    inference = config.get("functional_inference_contract", {})
    for key, expected in {
        "functional_only": True,
        "timing_computed": False,
        "timing_fields_present": False,
        "solve_method": "full_rank_QR_without_pseudoinverse",
        "finite_sample_exact": False,
        "reference_type": "freedman_lane_monte_carlo_approximation",
    }.items():
        _require_equal(inference.get(key), expected, f"functional_inference_contract.{key}")
    cache_contract = config.get("cache_contract", {})
    _require_equal(cache_contract.get("exact_cache_item_type_count"), 8, "cache item type count")
    configured_cache_files = tuple(
        str(item.get("relative_path")) for item in cache_contract.get("items", [])
    )
    _require_equal(configured_cache_files, CACHE_FILES, "configured cache item files")
    output_contract = config.get("output_contract", {})
    _require_equal(
        tuple(output_contract.get("exact_top_level_files", [])),
        OUTPUT_FILES,
        "configured output files",
    )
    _require_equal(
        output_contract.get("per_replicate_kernel_result_JSON_emitted"),
        False,
        "per-replicate kernel JSON firewall",
    )
    _require_equal(
        config.get("decision_contract", {}).get(
            "cb2_2000_start_allowed_value_for_this_execution"
        ),
        False,
        "CB2-2000 start authorization firewall",
    )
    frozen = config.get("frozen_payload_sha256")
    if require_frozen:
        _require(not _is_placeholder(frozen), "Execution config payload is not frozen")
        _require_equal(_payload_hash(config), str(frozen), "frozen_payload_sha256")
        _require(
            not _is_placeholder(config.get("frozen_at_utc")),
            "Execution config freeze marker is incomplete",
        )
        for name, binding in config.get("bindings", {}).items():
            _require(
                not _is_placeholder(binding.get("sha256")),
                f"Binding {name!r} is not frozen",
            )
    result = deepcopy(dict(config))
    result["_config_payload_sha256"] = (
        str(frozen) if not _is_placeholder(frozen) else _payload_hash(config)
    )
    result["_config_is_frozen"] = not _is_placeholder(frozen)
    return result


def load_cb2_500_execution_config(
    path: str | Path, *, require_frozen: bool = True
) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise CB2500ContractError("CB2-500 execution YAML must be a mapping")
    return validate_cb2_500_execution_config(value, require_frozen=require_frozen)


def verify_cb2_500_bindings(
    repository_root: str | Path,
    config: Mapping[str, Any],
    *,
    allow_placeholders: bool = False,
) -> dict[str, dict[str, Any]]:
    """Hash-verify every frozen binding, including large raw inputs."""

    root = Path(repository_root).resolve()
    verified: dict[str, dict[str, Any]] = {}
    for name, binding in config.get("bindings", {}).items():
        expected = binding.get("sha256")
        if _is_placeholder(expected):
            if allow_placeholders:
                continue
            raise CB2500ContractError(f"Binding {name!r} is not frozen")
        path = _repo_file(root, str(binding["relative_path"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = _hash_file(path)
        _require_equal(observed, str(expected), f"bindings.{name}.sha256")
        if "size_bytes" in binding:
            _require_equal(
                int(path.stat().st_size), int(binding["size_bytes"]), f"bindings.{name}.size_bytes"
            )
        verified[name] = {
            "relative_path": str(binding["relative_path"]),
            "sha256": observed,
            "bytes": int(path.stat().st_size),
        }
    return verified


def verify_cb2_500_source_bindings(
    repository_root: str | Path,
    config: Mapping[str, Any],
    *,
    module_path: str | Path | None = None,
    cli_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify config payload plus any frozen module/CLI runner bindings."""

    root = Path(repository_root).resolve()
    validated = validate_cb2_500_execution_config(config, require_frozen=True)
    observed = {
        "config_payload_sha256": validated["_config_payload_sha256"],
        "module_sha256": _hash_file(
            Path(module_path).resolve() if module_path else Path(__file__).resolve()
        ),
    }
    if cli_path is not None:
        observed["runner_sha256"] = _hash_file(Path(cli_path).resolve())
    source_names = {"cb2_500_module", "cb2_500_script", "cb2_500_test"}
    _require(
        source_names.issubset(set(config.get("bindings", {}))),
        "Execution config is missing one or more module/script/test source bindings",
    )
    for name, binding in config.get("bindings", {}).items():
        if name not in source_names:
            continue
        if _is_placeholder(binding.get("sha256")):
            raise CB2500ContractError(f"Source binding {name!r} is not frozen")
        path = _repo_file(root, str(binding["relative_path"]))
        digest = _hash_file(path)
        _require_equal(digest, str(binding["sha256"]), f"source binding {name}")
        if name == "cb2_500_module":
            _require_equal(observed["module_sha256"], digest, "loaded CB2-500 module hash")
        if name == "cb2_500_script" and cli_path is not None:
            _require_equal(observed["runner_sha256"], digest, "loaded CB2-500 runner hash")
    return observed


def derive_residual_mapping_seed(
    replicate_index: int,
    *,
    namespace: str = "trajpathmix_corebench_cb2_functional_null_v1",
    purpose: str = "residual_mappings",
    scenario: str = "primary_balanced_null",
) -> int:
    """Derive the frozen uint64 mapping seed for one zero-based replicate."""

    if isinstance(replicate_index, bool) or int(replicate_index) != replicate_index:
        raise CB2500ContractError("replicate_index must be an integer")
    index = int(replicate_index)
    if index < 0 or index >= 500:
        raise CB2500ContractError("replicate_index must be in the frozen range 0..499")
    message = f"{namespace}:{purpose}:{scenario}:{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(message).digest()[:8], "little", signed=False)


def _rank(values: np.ndarray, tolerance: float) -> tuple[int, float]:
    singular = np.linalg.svd(np.asarray(values, dtype=float), compute_uv=False)
    if not len(singular) or singular[0] == 0:
        return 0, 0.0
    threshold = float(singular[0] * tolerance)
    return int(np.sum(singular > threshold)), threshold


def _canonical_qr(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q, r = np.linalg.qr(np.asarray(values, dtype=float), mode="reduced")
    if r.size:
        signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
        q = q * signs[None, :]
        r = signs[:, None] * r
    return q, r


@dataclass(frozen=True)
class FactorizedBinDesign:
    bin_index: int
    available_donor_indices: np.ndarray
    active_experiment_indices: np.ndarray
    active_experiment_ids: tuple[str, ...]
    reduced_q: np.ndarray
    reduced_r: np.ndarray
    full_q: np.ndarray
    full_r: np.ndarray
    reduced_rank: int
    full_rank: int
    residual_df: int
    n_case: int
    n_control: int
    condition_information: float
    condition_vif: float
    reduced_rank_threshold: float
    full_rank_threshold: float
    estimable: bool
    reasons: tuple[str, ...]

    def diagnostics(self) -> dict[str, Any]:
        return {
            "bin_index": int(self.bin_index),
            "available_donor_indices": self.available_donor_indices.tolist(),
            "active_experiment_ids": list(self.active_experiment_ids),
            "dropped_bin_all_zero_experiment_ids": [],
            "reduced_rank": int(self.reduced_rank),
            "full_rank": int(self.full_rank),
            "condition_column_index": int(self.full_r.shape[0] - 1),
            "residual_df": int(self.residual_df),
            "n_case": int(self.n_case),
            "n_control": int(self.n_control),
            "condition_information": float(self.condition_information),
            "condition_vif": float(self.condition_vif),
            "reduced_rank_threshold": float(self.reduced_rank_threshold),
            "full_rank_threshold": float(self.full_rank_threshold),
            "estimable": bool(self.estimable),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class FactorizedNuisanceCache:
    donor_ids: tuple[str, ...]
    experiment_ids: tuple[str, ...]
    retained_experiment_ids: tuple[str, ...]
    dropped_global_all_zero_experiment_ids: tuple[str, ...]
    condition: np.ndarray
    availability: np.ndarray
    experiment_fractions: np.ndarray
    bins: tuple[FactorizedBinDesign, ...]
    rank_tolerance: float

    @property
    def all_estimable(self) -> bool:
        return all(item.estimable for item in self.bins)

    def diagnostics(self) -> dict[str, Any]:
        retained = set(self.retained_experiment_ids)
        return {
            "donor_ids": list(self.donor_ids),
            "experiment_ids": list(self.experiment_ids),
            "retained_experiment_ids": list(self.retained_experiment_ids),
            "dropped_global_all_zero_experiment_ids": list(
                self.dropped_global_all_zero_experiment_ids
            ),
            "rank_relative_tolerance": float(self.rank_tolerance),
            "no_intercept": True,
            "all_estimable": bool(self.all_estimable),
            "bins": [
                {
                    **item.diagnostics(),
                    "dropped_bin_all_zero_experiment_ids": [
                        value
                        for value in self.retained_experiment_ids
                        if value not in set(item.active_experiment_ids)
                    ],
                }
                for item in self.bins
            ],
        }


def build_factorized_nuisance_cache(
    donor_ids: Sequence[Any],
    condition: Any,
    availability: Any,
    experiment_fractions: Any,
    experiment_ids: Sequence[Any],
    *,
    rank_tolerance: float = RANK_RELATIVE_TOLERANCE,
    min_donors_per_condition: int = 10,
    min_residual_df: int = 3,
    max_condition_vif: float = 10.0,
) -> FactorizedNuisanceCache:
    """Build QR-factorized, bin-specific reduced/full nuisance designs."""

    _require_equal(
        float(rank_tolerance),
        RANK_RELATIVE_TOLERANCE,
        "rank_relative_tolerance",
    )
    donors = tuple(str(value) for value in donor_ids)
    experiments = tuple(str(value) for value in experiment_ids)
    _require(len(donors) == len(set(donors)), "donor_ids must be unique")
    _require(donors == tuple(sorted(donors)), "donor_ids must be lexicographic")
    _require(len(experiments) == len(set(experiments)), "experiment_ids must be unique")
    _require(experiments == tuple(sorted(experiments)), "experiment_ids must be lexicographic")
    c = np.asarray(condition)
    if c.shape != (len(donors),) or not np.isin(c, [0, 1, False, True]).all():
        raise CB2500ContractError("condition must be a binary donor vector")
    c = c.astype(np.uint8)
    available = np.asarray(availability)
    if available.ndim != 2 or available.shape[0] != len(donors):
        raise CB2500ContractError("availability must be donor x bin")
    if not np.isin(available, [0, 1, False, True]).all():
        raise CB2500ContractError("availability must be binary")
    available = available.astype(bool)
    fractions = np.asarray(experiment_fractions, dtype=float)
    expected = (len(donors), available.shape[1], len(experiments))
    if fractions.shape != expected or not np.isfinite(fractions).all():
        raise CB2500ContractError(f"experiment_fractions must have shape {expected}")
    if np.any(fractions < -NUMERICAL_TOLERANCE):
        raise CB2500ContractError("experiment_fractions must be nonnegative")
    row_sums = fractions.sum(axis=2)
    if not np.allclose(row_sums[available], 1.0, atol=1e-10, rtol=0.0):
        raise CB2500ContractError("Available experiment fractions must sum to one")
    masked_fractions = np.where(available[:, :, None], fractions, 0.0)
    globally_active = np.any(np.abs(masked_fractions) > 1e-12, axis=(0, 1))
    retained_indices = np.flatnonzero(globally_active)
    retained = tuple(experiments[index] for index in retained_indices)
    dropped = tuple(experiments[index] for index in np.flatnonzero(~globally_active))
    bins: list[FactorizedBinDesign] = []
    for bin_index in range(available.shape[1]):
        indices = np.flatnonzero(available[:, bin_index])
        local = fractions[indices, bin_index][:, retained_indices]
        locally_active = np.any(np.abs(local) > 1e-12, axis=0)
        active_retained = np.flatnonzero(locally_active)
        z = local[:, active_retained]
        active_global = retained_indices[active_retained]
        reduced_rank, reduced_threshold = _rank(z, float(rank_tolerance))
        reduced_q, reduced_r = _canonical_qr(z)
        local_c = c[indices].astype(float)
        full = np.column_stack([z, local_c])
        full_rank, full_threshold = _rank(full, float(rank_tolerance))
        full_q, full_r = _canonical_qr(full)
        residual_c = local_c - reduced_q @ (reduced_q.T @ local_c)
        information = float(np.dot(residual_c, residual_c))
        n_case = int(local_c.sum())
        n_control = int(len(local_c) - n_case)
        centered = local_c - float(local_c.mean()) if len(local_c) else local_c
        unadjusted = float(centered @ centered)
        vif = (
            float(unadjusted / information)
            if information > 1.0e-14 * max(1.0, unadjusted)
            else math.inf
        )
        residual_df = int(len(local_c) - full_rank)
        reasons: list[str] = []
        if z.shape[1] == 0 or reduced_rank != z.shape[1]:
            reasons.append("rank_deficient_reduced_experiment_design")
        if full_rank != reduced_rank + 1 or full_rank != full.shape[1]:
            reasons.append("condition_not_identifiable")
        if min(n_case, n_control) < int(min_donors_per_condition):
            reasons.append("insufficient_donors_per_condition")
        if residual_df < int(min_residual_df):
            reasons.append("insufficient_residual_df")
        if not math.isfinite(vif) or vif > float(max_condition_vif):
            reasons.append("condition_vif_exceeds_threshold")
        bins.append(
            FactorizedBinDesign(
                bin_index=bin_index,
                available_donor_indices=indices.astype(np.int16),
                active_experiment_indices=active_global.astype(np.int16),
                active_experiment_ids=tuple(experiments[index] for index in active_global),
                reduced_q=reduced_q,
                reduced_r=reduced_r,
                full_q=full_q,
                full_r=full_r,
                reduced_rank=reduced_rank,
                full_rank=full_rank,
                residual_df=residual_df,
                n_case=n_case,
                n_control=n_control,
                condition_information=information,
                condition_vif=vif,
                reduced_rank_threshold=reduced_threshold,
                full_rank_threshold=full_threshold,
                estimable=not reasons,
                reasons=tuple(dict.fromkeys(reasons)),
            )
        )
    return FactorizedNuisanceCache(
        donor_ids=donors,
        experiment_ids=experiments,
        retained_experiment_ids=retained,
        dropped_global_all_zero_experiment_ids=dropped,
        condition=c,
        availability=available,
        experiment_fractions=fractions,
        bins=tuple(bins),
        rank_tolerance=float(rank_tolerance),
    )


def build_availability_mapping_bank(
    donor_ids: Sequence[Any],
    availability: Any,
    *,
    n_mappings: int,
    seed: int,
) -> np.ndarray:
    """Generate one explicit identity-excluding mapping bank using the frozen kernel."""

    donors = tuple(str(value) for value in donor_ids)
    _require(donors == tuple(sorted(donors)), "donor_ids must be lexicographic")
    from .trajpathmix_functional_core_v1 import build_full_availability_mapping_plan

    plan = build_full_availability_mapping_plan(
        donors, availability, n_mappings=int(n_mappings), seed=int(seed)
    )
    _require_equal(plan.donor_ids, donors, "mapping donor order")
    mappings = np.asarray(plan.mappings, dtype=np.int32)
    _require_equal(mappings.shape, (int(n_mappings), len(donors)), "mapping shape")
    _require_equal(len({_hash_array(row, "<i4") for row in mappings}), int(n_mappings), "unique mappings")
    identity = np.arange(len(donors), dtype=np.int32)
    _require(not np.any(np.all(mappings == identity[None, :], axis=1)), "identity mapping present")
    return mappings


def _plus_one_p(null: np.ndarray, observed: np.ndarray) -> np.ndarray:
    null_values = np.asarray(null, dtype=float)
    observed_values = np.asarray(observed, dtype=float)
    return (
        1.0
        + np.sum(
            null_values >= observed_values[None, ...] - NUMERICAL_TOLERANCE,
            axis=0,
        )
    ) / (null_values.shape[0] + 1.0)


def _by_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all():
        raise CB2500ContractError("BY adjustment requires finite one-dimensional p-values")
    count = len(p)
    harmonic = float(np.sum(1.0 / np.arange(1, count + 1)))
    order = np.argsort(p, kind="mergesort")
    ranked = p[order] * count * harmonic / np.arange(1, count + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    output = np.empty_like(ranked)
    output[order] = np.minimum(1.0, ranked)
    return output


def compute_global_family_endpoints(
    observed_curve_statistics: Any,
    null_curve_statistics: Any,
    *,
    family_index: Sequence[int] | None,
    observed_integrated_statistics: Any | None = None,
    null_integrated_statistics: Any | None = None,
) -> dict[str, np.ndarray]:
    """Compute raw/global-50/family endpoints from explicit null statistics."""

    observed = np.asarray(observed_curve_statistics, dtype=float)
    null = np.asarray(null_curve_statistics, dtype=float)
    if observed.ndim != 1 or null.ndim != 2 or null.shape[1] != len(observed):
        raise CB2500ContractError("Curve statistics must be pathway and mapping-by-pathway")
    if not np.isfinite(observed).all() or not np.isfinite(null).all():
        raise CB2500ContractError("Curve statistics must be finite")
    raw_curve_p = _plus_one_p(null, observed)
    global_null = np.max(null, axis=1)
    global_curve_p = _plus_one_p(global_null[:, None], observed)
    q_by = _by_adjust(raw_curve_p)
    if family_index is None:
        codes = np.full(len(observed), -1, dtype=int)
    else:
        codes = np.asarray(family_index, dtype=int)
    if codes.shape != observed.shape or np.any(codes < -1):
        raise CB2500ContractError("family_index must align to pathways and use -1 or nonnegative codes")
    assigned = np.unique(codes[codes >= 0])
    family_observed = np.asarray(
        [np.max(observed[codes == code]) for code in assigned], dtype=float
    )
    family_null = np.asarray(
        [np.max(null[:, codes == code], axis=1) for code in assigned], dtype=float
    ).T if len(assigned) else np.empty((len(null), 0), dtype=float)
    family_p = (
        _plus_one_p(family_null, family_observed)
        if len(assigned)
        else np.empty(0, dtype=float)
    )
    pathway_family_p = np.ones(len(observed), dtype=float)
    for position, code in enumerate(assigned):
        members = codes == code
        pathway_family_p[members] = _plus_one_p(
            family_null[:, position][:, None], observed[members]
        )
    output: dict[str, np.ndarray] = {
        "raw_curve_p_value": raw_curve_p,
        "global_50_curve_maxT_p_value": global_curve_p,
        "BY_q_value": q_by,
        "family_codes": assigned,
        "observed_family_max_statistic": family_observed,
        "null_family_max_statistic": family_null,
        "family_maxT_p_value": family_p,
        "pathway_family_maxT_p_value": pathway_family_p,
    }
    if observed_integrated_statistics is not None or null_integrated_statistics is not None:
        integrated = np.asarray(observed_integrated_statistics, dtype=float)
        null_integrated = np.asarray(null_integrated_statistics, dtype=float)
        if integrated.shape != observed.shape or null_integrated.shape != null.shape:
            raise CB2500ContractError("Integrated statistics do not align with curve statistics")
        if not np.isfinite(integrated).all() or not np.isfinite(null_integrated).all():
            raise CB2500ContractError("Integrated statistics must be finite")
        output["raw_integrated_p_value"] = _plus_one_p(null_integrated, integrated)
    return output


def _studentize(
    numerator: np.ndarray,
    standard_error: np.ndarray,
    required_mask: np.ndarray,
) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=float)
    standard_error = np.asarray(standard_error, dtype=float)
    required = np.asarray(required_mask, dtype=bool)
    if required.shape != numerator.shape[-required.ndim :]:
        try:
            required = np.broadcast_to(required, numerator.shape)
        except ValueError as exc:
            raise CB2500ContractError("Studentization mask does not align") from exc
    else:
        required = np.broadcast_to(required, numerator.shape)
    output = np.zeros_like(numerator)
    positive = standard_error > NUMERICAL_TOLERANCE
    np.divide(numerator, standard_error, out=output, where=positive)
    if np.any((~positive) & (np.abs(numerator) > NUMERICAL_TOLERANCE) & required):
        raise CB2500DesignError(
            "Studentization is undefined for a nonzero supported coefficient"
        )
    return output


def _fit_with_factorized_design(
    item: FactorizedBinDesign,
    outcomes: np.ndarray,
    *,
    coefficient_offset: np.ndarray | None = None,
    required_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit one full design to ``(..., donor, pathway)`` outcomes."""

    y = np.asarray(outcomes, dtype=float)
    if y.shape[-2] != len(item.available_donor_indices):
        raise CB2500ContractError("Factorized full design and outcome rows do not align")
    if not item.estimable:
        raise CB2500DesignError(
            f"Bin {item.bin_index} is not estimable: {','.join(item.reasons)}"
        )
    q = np.asarray(item.full_q, dtype=float)
    r = np.asarray(item.full_r, dtype=float)
    inverse_r = np.linalg.solve(r, np.eye(r.shape[0], dtype=float))
    condition_row = inverse_r[-1]
    condition_weights = q @ condition_row
    coefficient = np.einsum("n,...np->...p", condition_weights, y, optimize=False)
    projected = np.einsum("nk,...np->...kp", q, y, optimize=False)
    fitted = np.einsum("nk,...kp->...np", q, projected, optimize=False)
    residual = y - fitted
    sigma_squared = np.sum(residual * residual, axis=-2) / float(item.residual_df)
    variance_factor = float(np.sum(condition_row * condition_row))
    standard_error = np.sqrt(np.maximum(0.0, sigma_squared * variance_factor))
    numerator = (
        coefficient
        if coefficient_offset is None
        else coefficient - np.asarray(coefficient_offset, dtype=float)
    )
    studentized = _studentize(numerator, standard_error, required_mask)
    return coefficient, standard_error, studentized, fitted, residual


def _mapping_audit(
    donor_ids: Sequence[str], availability: np.ndarray, mappings: np.ndarray, seed: int | None
) -> dict[str, Any]:
    signatures = tuple(
        "".join("1" if value else "0" for value in row)
        for row in np.asarray(availability, dtype=bool)
    )
    groups_by_signature: dict[str, list[int]] = {}
    for index, signature in enumerate(signatures):
        groups_by_signature.setdefault(signature, []).append(index)
    groups = tuple(sorted(groups_by_signature.values(), key=lambda value: value[0]))
    orbit = int(math.prod(math.factorial(len(group)) for group in groups))
    mobile = int(sum(len(group) for group in groups if len(group) > 1))
    canonical = np.ascontiguousarray(mappings, dtype="<i4")
    hashes = tuple(_hash_array(row, "<i4") for row in canonical)
    identity = np.arange(len(donor_ids), dtype=np.int32)
    return {
        "seed": None if seed is None else int(seed),
        "mapping_stream_sha256": hashlib.sha256(canonical.tobytes()).hexdigest(),
        "n_mappings": int(len(canonical)),
        "n_unique_mapping_hashes": int(len(set(hashes))),
        "identity_mapping_present": bool(
            np.any(np.all(canonical == identity[None, :], axis=1))
        ),
        "n_unique_availability_signatures": int(len(groups)),
        "n_mobile_donors": mobile,
        "n_immobile_donors": int(len(donor_ids) - mobile),
        "orbit_size": orbit,
        "n_unique_nonidentity_mappings_possible": int(orbit - 1),
        "attainable_exact_p_resolution": float(1.0 / orbit),
        "sampled_p_resolution": float(1.0 / (len(canonical) + 1)),
        "same_stream_all_endpoints": True,
    }


@dataclass(frozen=True)
class CachedFunctionalCoreBatchResult:
    donor_ids: tuple[str, ...]
    pathway_ids: tuple[str, ...]
    support_mask: np.ndarray
    effect: np.ndarray
    standard_error: np.ndarray
    studentized_effect: np.ndarray
    null_effect: np.ndarray
    null_studentized_effect: np.ndarray
    bootstrap_effect: np.ndarray
    bootstrap_studentized_deviation: np.ndarray
    pointwise_p: np.ndarray
    pointwise_critical: np.ndarray
    pointwise_lower: np.ndarray
    pointwise_upper: np.ndarray
    simultaneous_critical: float
    simultaneous_lower: np.ndarray
    simultaneous_upper: np.ndarray
    band_order_index_1based: int
    curve_statistic: np.ndarray
    null_curve_statistic: np.ndarray
    raw_curve_p_value: np.ndarray
    global_50_curve_maxT_p_value: np.ndarray
    BY_q_value: np.ndarray
    integrated_effect: np.ndarray
    null_integrated_effect: np.ndarray
    raw_integrated_p_value: np.ndarray
    integrated_p_maxT: np.ndarray
    integrated_q_by: np.ndarray
    family_codes: np.ndarray
    observed_family_max_statistic: np.ndarray
    null_family_max_statistic: np.ndarray
    family_maxT_p_value: np.ndarray
    pathway_family_maxT_p_value: np.ndarray
    mapping_audit: Mapping[str, Any]
    design_diagnostics: Mapping[str, Any]

    @property
    def curve_p_raw(self) -> np.ndarray:
        return self.raw_curve_p_value

    @property
    def curve_p_maxT(self) -> np.ndarray:
        # Compatibility with the frozen kernel, where curve_p_maxT was the
        # within-pathway curve max-over-bins p-value.  The CB2-500 global-50
        # endpoint is exposed separately and is acceptance-bearing.
        return self.raw_curve_p_value

    @property
    def curve_q_by(self) -> np.ndarray:
        return self.BY_q_value

    @property
    def integrated_absolute_effect(self) -> np.ndarray:
        return self.integrated_effect

    @property
    def integrated_studentized_statistic(self) -> np.ndarray:
        return self.integrated_effect

    @property
    def integrated_p_raw(self) -> np.ndarray:
        return self.raw_integrated_p_value

    @property
    def family_p_maxT(self) -> np.ndarray:
        return self.pathway_family_maxT_p_value


def run_cached_functional_core_batch(
    *,
    outcomes: Any,
    donor_ids: Sequence[Any],
    condition: Any,
    availability: Any,
    experiment_fractions: Any,
    experiment_ids: Sequence[Any],
    pathway_ids: Sequence[Any],
    mappings: Any,
    factorized_cache: FactorizedNuisanceCache | None = None,
    family_index: Sequence[int] | None = None,
    support_mask: Any = None,
    bin_weights: Any = None,
    alpha: float = DEFAULT_ALPHA,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    mapping_seed: int | None = None,
    rank_tolerance: float = RANK_RELATIVE_TOLERANCE,
    min_donors_per_condition: int = 10,
    min_residual_df: int = 3,
    max_condition_vif: float = 10.0,
) -> CachedFunctionalCoreBatchResult:
    """Run the frozen functional core using supplied QR factors and mappings.

    This is the directly unit-testable optimized adapter.  It never generates
    assignments or mappings and has no filesystem side effects.
    """

    donors = tuple(str(value).strip() for value in donor_ids)
    pathways = tuple(str(value).strip() for value in pathway_ids)
    experiments = tuple(str(value).strip() for value in experiment_ids)
    _require(donors == tuple(sorted(donors)), "donor_ids must be lexicographic")
    _require(len(donors) == len(set(donors)) and all(donors), "donor_ids must be unique")
    _require(len(pathways) == len(set(pathways)) and all(pathways), "pathway_ids must be unique")
    y = np.asarray(outcomes, dtype=float)
    available = np.asarray(availability)
    if available.ndim != 2 or available.shape[0] != len(donors):
        raise CB2500ContractError("availability must be donor x bin")
    if not np.isin(available, [0, 1, False, True]).all():
        raise CB2500ContractError("availability must be binary")
    available = available.astype(bool)
    expected_shape = (len(donors), available.shape[1], len(pathways))
    if y.shape != expected_shape:
        raise CB2500ContractError(f"outcomes must have shape {expected_shape}")
    if not np.isfinite(y[available]).all() or not np.isnan(y[~available]).all():
        raise CB2500ContractError(
            "Available outcomes must be finite and unavailable outcomes must be NA"
        )
    c = np.asarray(condition)
    if c.shape != (len(donors),) or not np.isin(c, [0, 1, False, True]).all():
        raise CB2500ContractError("condition must be a binary donor vector")
    fractions = np.asarray(experiment_fractions, dtype=float)
    if factorized_cache is None:
        factorized_cache = build_factorized_nuisance_cache(
            donors,
            c,
            available,
            fractions,
            experiments,
            rank_tolerance=rank_tolerance,
            min_donors_per_condition=min_donors_per_condition,
            min_residual_df=min_residual_df,
            max_condition_vif=max_condition_vif,
        )
    else:
        _require_equal(factorized_cache.donor_ids, donors, "factorized donor axis")
        _require_equal(factorized_cache.experiment_ids, experiments, "factorized experiment axis")
        _require(np.array_equal(factorized_cache.condition, c.astype(np.uint8)), "factorized condition")
        _require(np.array_equal(factorized_cache.availability, available), "factorized availability")
        _require(
            np.array_equal(factorized_cache.experiment_fractions, fractions),
            "factorized experiment fractions",
        )
    if not factorized_cache.all_estimable:
        raise CB2500DesignError(
            "One or more bin-specific designs are not estimable",
        )
    n_bins = available.shape[1]
    n_pathways = len(pathways)
    if support_mask is None:
        support = np.ones((n_bins, n_pathways), dtype=bool)
    else:
        support = np.asarray(support_mask)
        if support.shape == (n_bins,):
            support = np.repeat(support[:, None], n_pathways, axis=1)
        if support.shape != (n_bins, n_pathways) or not np.isin(
            support, [0, 1, False, True]
        ).all():
            raise CB2500ContractError("support_mask must be binary bin x pathway")
        support = support.astype(bool)
    if not support.any(axis=0).all():
        raise CB2500DesignError("Every pathway must have supported bins")
    if bin_weights is None:
        weights = np.full(n_bins, 1.0 / n_bins, dtype=float)
    else:
        weights = np.asarray(bin_weights, dtype=float)
        if weights.shape != (n_bins,) or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise CB2500ContractError("bin_weights must be finite and strictly positive")
    alpha_value = float(alpha)
    if not 0.0 < alpha_value < 1.0:
        raise CB2500ContractError("alpha must lie strictly between zero and one")
    chunk = int(chunk_size)
    if chunk <= 0:
        raise CB2500ContractError("chunk_size must be positive")
    mapping_array = np.asarray(mappings)
    if mapping_array.ndim != 2 or mapping_array.shape[1] != len(donors):
        raise CB2500ContractError("mappings must be mapping x donor")
    _require(len(mapping_array) > 0, "At least one residual mapping is required")
    if not np.issubdtype(mapping_array.dtype, np.integer):
        raise CB2500ContractError("mappings must contain integer donor indices")
    mapping_array = mapping_array.astype(np.int32, copy=False)
    if np.any(mapping_array < 0) or np.any(mapping_array >= len(donors)):
        raise CB2500ContractError("mapping donor index is out of range")
    identity = np.arange(len(donors), dtype=np.int32)
    if np.any(np.all(mapping_array == identity[None, :], axis=1)):
        raise CB2500ContractError("identity residual mapping is forbidden")
    mapping_hashes = [_hash_array(row, "<i4") for row in mapping_array]
    if len(set(mapping_hashes)) != len(mapping_array):
        raise CB2500ContractError("Residual mappings must be unique")
    for mapping in mapping_array:
        if not np.array_equal(np.sort(mapping), identity):
            raise CB2500ContractError("Each residual mapping must be a donor permutation")
        if not np.array_equal(available[mapping], available):
            raise CB2500ContractError("Residual mapping crossed an availability signature")

    effect = np.empty((n_bins, n_pathways), dtype=float)
    standard_error = np.empty_like(effect)
    studentized = np.empty_like(effect)
    reduced_fitted: list[np.ndarray] = []
    reduced_residual: list[np.ndarray] = []
    full_fitted: list[np.ndarray] = []
    full_residual: list[np.ndarray] = []
    for item in factorized_cache.bins:
        local_y = y[item.available_donor_indices, item.bin_index]
        coefficient, se, t_value, fitted, residual = _fit_with_factorized_design(
            item, local_y, required_mask=support[item.bin_index]
        )
        q0 = item.reduced_q
        projected0 = q0.T @ local_y
        fitted0 = q0 @ projected0
        effect[item.bin_index] = coefficient
        standard_error[item.bin_index] = se
        studentized[item.bin_index] = t_value
        reduced_fitted.append(fitted0)
        reduced_residual.append(local_y - fitted0)
        full_fitted.append(fitted)
        full_residual.append(residual)

    n_reference = len(mapping_array)
    null_effect = np.empty((n_reference, n_bins, n_pathways), dtype=float)
    null_t = np.empty_like(null_effect)
    bootstrap_effect = np.empty_like(null_effect)
    bootstrap_t = np.empty_like(null_effect)
    for start in range(0, n_reference, chunk):
        stop = min(n_reference, start + chunk)
        current = mapping_array[start:stop]
        for item in factorized_cache.bins:
            indices = item.available_donor_indices.astype(int)
            global_to_local = np.full(len(donors), -1, dtype=int)
            global_to_local[indices] = np.arange(len(indices))
            source_local = global_to_local[current[:, indices]]
            if np.any(source_local < 0):
                raise CB2500ContractError("Mapping crossed a bin availability set")
            null_y = reduced_fitted[item.bin_index][None, :, :] + reduced_residual[
                item.bin_index
            ][source_local]
            null_fit = _fit_with_factorized_design(
                item, null_y, required_mask=support[item.bin_index]
            )
            null_effect[start:stop, item.bin_index] = null_fit[0]
            null_t[start:stop, item.bin_index] = null_fit[2]
            bootstrap_y = full_fitted[item.bin_index][None, :, :] + full_residual[
                item.bin_index
            ][source_local]
            bootstrap_fit = _fit_with_factorized_design(
                item,
                bootstrap_y,
                coefficient_offset=effect[item.bin_index],
                required_mask=support[item.bin_index],
            )
            bootstrap_effect[start:stop, item.bin_index] = bootstrap_fit[0]
            bootstrap_t[start:stop, item.bin_index] = bootstrap_fit[2]

    absolute_observed_t = np.abs(studentized)
    absolute_null_t = np.abs(null_t)
    pointwise_p = _plus_one_p(absolute_null_t, absolute_observed_t)
    curve_statistic = np.asarray(
        [
            np.max(absolute_observed_t[support[:, index], index])
            for index in range(n_pathways)
        ],
        dtype=float,
    )
    null_curve = np.asarray(
        [
            np.max(absolute_null_t[:, support[:, index], index], axis=1)
            for index in range(n_pathways)
        ],
        dtype=float,
    ).T
    integrated = np.asarray(
        [
            np.sum(np.abs(effect[support[:, index], index]) * weights[support[:, index]])
            for index in range(n_pathways)
        ],
        dtype=float,
    )
    null_integrated = np.asarray(
        [
            np.sum(
                np.abs(null_effect[:, support[:, index], index])
                * weights[support[:, index]][None, :],
                axis=1,
            )
            for index in range(n_pathways)
        ],
        dtype=float,
    ).T
    endpoints = compute_global_family_endpoints(
        curve_statistic,
        null_curve,
        family_index=family_index,
        observed_integrated_statistics=integrated,
        null_integrated_statistics=null_integrated,
    )
    order_index = min(
        n_reference,
        int(math.ceil((n_reference + 1) * (1.0 - alpha_value))),
    )
    absolute_bootstrap_t = np.abs(bootstrap_t)
    pointwise_critical = np.sort(absolute_bootstrap_t, axis=0)[order_index - 1]
    simultaneous_reference = np.max(absolute_bootstrap_t[:, support], axis=1)
    simultaneous_critical = float(np.sort(simultaneous_reference)[order_index - 1])
    pointwise_lower = effect - pointwise_critical * standard_error
    pointwise_upper = effect + pointwise_critical * standard_error
    simultaneous_lower = effect - simultaneous_critical * standard_error
    simultaneous_upper = effect + simultaneous_critical * standard_error
    for array in (
        effect,
        standard_error,
        studentized,
        pointwise_p,
        pointwise_critical,
        pointwise_lower,
        pointwise_upper,
        simultaneous_lower,
        simultaneous_upper,
    ):
        array[~support] = np.nan
    null_effect[:, ~support] = np.nan
    null_t[:, ~support] = np.nan
    bootstrap_effect[:, ~support] = np.nan
    bootstrap_t[:, ~support] = np.nan
    return CachedFunctionalCoreBatchResult(
        donor_ids=donors,
        pathway_ids=pathways,
        support_mask=support,
        effect=effect,
        standard_error=standard_error,
        studentized_effect=studentized,
        null_effect=null_effect,
        null_studentized_effect=null_t,
        bootstrap_effect=bootstrap_effect,
        bootstrap_studentized_deviation=bootstrap_t,
        pointwise_p=pointwise_p,
        pointwise_critical=pointwise_critical,
        pointwise_lower=pointwise_lower,
        pointwise_upper=pointwise_upper,
        simultaneous_critical=simultaneous_critical,
        simultaneous_lower=simultaneous_lower,
        simultaneous_upper=simultaneous_upper,
        band_order_index_1based=order_index,
        curve_statistic=curve_statistic,
        null_curve_statistic=null_curve,
        raw_curve_p_value=endpoints["raw_curve_p_value"],
        global_50_curve_maxT_p_value=endpoints["global_50_curve_maxT_p_value"],
        BY_q_value=endpoints["BY_q_value"],
        integrated_effect=integrated,
        null_integrated_effect=null_integrated,
        raw_integrated_p_value=endpoints["raw_integrated_p_value"],
        integrated_p_maxT=_plus_one_p(
            np.max(null_integrated, axis=1)[:, None], integrated
        ),
        integrated_q_by=_by_adjust(endpoints["raw_integrated_p_value"]),
        family_codes=endpoints["family_codes"],
        observed_family_max_statistic=endpoints["observed_family_max_statistic"],
        null_family_max_statistic=endpoints["null_family_max_statistic"],
        family_maxT_p_value=endpoints["family_maxT_p_value"],
        pathway_family_maxT_p_value=endpoints["pathway_family_maxT_p_value"],
        mapping_audit=_mapping_audit(donors, available, mapping_array, mapping_seed),
        design_diagnostics=factorized_cache.diagnostics(),
    )


def _read_bool_column(series: pd.Series, label: str) -> np.ndarray:
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise CB2500ContractError(f"Column {label!r} contains non-boolean values")
    return normalized.eq("true").to_numpy(dtype=bool)


def _load_frozen_assignments(
    root: Path, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, tuple[str, ...], np.ndarray, tuple[str, ...], tuple[str, ...]]:
    binding = config["bindings"]["frozen_assignment_manifest_v1"]
    path = _repo_file(root, str(binding["relative_path"]))
    _require_equal(_hash_file(path), str(binding["sha256"]), "assignment manifest hash")
    frame = pd.read_csv(path, sep="\t", dtype="string", keep_default_na=False)
    _require_equal(tuple(frame.columns), ASSIGNMENT_COLUMNS, "assignment columns")
    expected_ids = tuple(f"CB2P_{index:04d}" for index in range(1, 501))
    assignment_ids = tuple(frame["assignment_id"].drop_duplicates().astype(str))
    _require_equal(assignment_ids, expected_ids, "assignment id axis")
    frame["pseudo_case_bool"] = _read_bool_column(frame["pseudo_case"], "pseudo_case")
    first = frame.loc[frame["assignment_id"].eq(assignment_ids[0])]
    donors = tuple(first["donor_id"].astype(str))
    _require_equal(len(donors), 75, "assignment donor count")
    _require_equal(donors, tuple(sorted(donors)), "assignment donor order")
    assignments = np.empty((500, 75), dtype=np.uint8)
    hashes: list[str] = []
    for replicate_index, assignment_id in enumerate(assignment_ids):
        group = frame.loc[frame["assignment_id"].eq(assignment_id)]
        _require_equal(len(group), 75, f"{assignment_id} row count")
        _require_equal(tuple(group["donor_id"].astype(str)), donors, f"{assignment_id} donors")
        values = group["pseudo_case_bool"].to_numpy(dtype=np.uint8)
        _require_equal(int(values.sum()), 37, f"{assignment_id} pseudo-case count")
        expected_labels = np.where(values.astype(bool), "pseudo_case", "pseudo_control")
        _require(
            np.array_equal(expected_labels, group["pseudo_condition"].astype(str).to_numpy()),
            f"{assignment_id} pseudo-condition labels disagree",
        )
        digest = _hash_array(values, np.uint8)
        recorded = tuple(group["assignment_sha256"].astype(str).unique())
        _require_equal(recorded, (digest,), f"{assignment_id} hash")
        assignments[replicate_index] = values
        hashes.append(digest)
    _require_equal(len(set(hashes)), 500, "unique assignment hashes")
    return frame, donors, assignments, assignment_ids, tuple(hashes)


def _load_structural_inputs(
    root: Path, config: Mapping[str, Any], donors: tuple[str, ...]
) -> dict[str, Any]:
    bindings = config["bindings"]
    cohort = pd.read_csv(
        _repo_file(root, bindings["donor_cohort_v1"]["relative_path"]),
        sep="\t",
        dtype="string",
        keep_default_na=False,
    )
    primary_bool = _read_bool_column(cohort["primary_complete_support"], "primary_complete_support")
    primary = tuple(sorted(cohort.loc[primary_bool, "donor_id"].astype(str)))
    _require_equal(primary, donors, "primary donor cohort")
    availability_long = pd.read_csv(
        _repo_file(root, bindings["donor_bin_availability_v2"]["relative_path"]),
        sep="\t",
        dtype={"donor_id": "string"},
    )
    availability_long["available_bool"] = _read_bool_column(
        availability_long["available"], "available"
    )
    local_availability = availability_long.loc[
        availability_long["donor_id"].astype(str).isin(donors)
    ].copy()
    _require_equal(len(local_availability), 75 * 20, "donor-bin availability rows")
    counts = (
        local_availability.pivot(index="donor_id", columns="bin_id", values="cell_count")
        .reindex(index=donors, columns=range(20))
        .to_numpy(dtype=np.int64)
    )
    availability = (
        local_availability.pivot(index="donor_id", columns="bin_id", values="available_bool")
        .reindex(index=donors, columns=range(20))
        .to_numpy(dtype=bool)
    )
    _require(np.array_equal(availability, counts >= 5), "Frozen >=5-cell availability differs")
    donor_counts = pd.read_csv(
        _repo_file(root, bindings["donor_experiment_counts_v1"]["relative_path"]),
        sep="\t",
        dtype={"donor_id": "string", "experiment_id": "string"},
    )
    experiments = tuple(sorted(donor_counts["experiment_id"].astype(str).unique()))
    _require_equal(len(experiments), 28, "experiment axis length")
    coordinate = pd.read_csv(
        _repo_file(root, bindings["fixed_cb1_coordinate"]["relative_path"]),
        sep="\t",
        dtype={
            "cell_id": "string",
            "donor_id": "string",
            "experiment_id": "string",
        },
    )
    _require_equal(len(coordinate), int(coordinate["cell_id"].nunique()), "coordinate cell ids")
    values = coordinate["corebench_coordinate"].to_numpy(dtype=np.float64)
    _require(
        np.isfinite(values).all() and np.all((values >= 0.0) & (values <= 1.0)),
        "Frozen coordinate is invalid",
    )
    coordinate["bin_id"] = np.minimum(19, np.floor(values * 20.0).astype(np.int64))
    primary_coordinate = coordinate.loc[coordinate["donor_id"].astype(str).isin(donors)].copy()
    donor_index = {donor: index for index, donor in enumerate(donors)}
    experiment_index = {value: index for index, value in enumerate(experiments)}
    cube = np.zeros((75, 20, len(experiments)), dtype=np.float64)
    grouped = (
        primary_coordinate.groupby(["donor_id", "bin_id", "experiment_id"], sort=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    for row in grouped.itertuples(index=False):
        cube[
            donor_index[str(row.donor_id)],
            int(row.bin_id),
            experiment_index[str(row.experiment_id)],
        ] = int(row.n_cells)
    _require(
        np.array_equal(cube.sum(axis=2).astype(np.int64), counts),
        "Coordinate counts differ from frozen donor-bin counts",
    )
    fractions = np.divide(
        cube,
        cube.sum(axis=2, keepdims=True),
        out=np.zeros_like(cube),
        where=cube.sum(axis=2, keepdims=True) > 0,
    )
    signatures = tuple(
        "".join("1" if value else "0" for value in row) for row in availability
    )
    return {
        "coordinate": coordinate,
        "primary_coordinate": primary_coordinate,
        "experiments": experiments,
        "availability": availability,
        "cell_count": counts,
        "experiment_fractions": fractions,
        "availability_signatures": signatures,
    }


def _stack_factorized_caches(
    caches: Sequence[FactorizedNuisanceCache], assignments: np.ndarray
) -> dict[str, np.ndarray]:
    """Serialize all fixed/reusable QR components without a pseudoinverse."""

    if not caches:
        raise CB2500ContractError("No factorized design caches supplied")
    n_replicates = len(caches)
    n_bins = len(caches[0].bins)
    n_donors = len(caches[0].donor_ids)
    n_experiments = len(caches[0].experiment_ids)
    max_reduced = max(item.reduced_r.shape[0] for cache in caches for item in cache.bins)
    max_full = max(item.full_r.shape[0] for cache in caches for item in cache.bins)
    available_indices = np.full((n_bins, n_donors), -1, dtype=np.int16)
    available_count = np.zeros(n_bins, dtype=np.int16)
    active_experiment_indices = np.full((n_bins, n_experiments), -1, dtype=np.int16)
    active_experiment_count = np.zeros(n_bins, dtype=np.int16)
    reduced_design = np.zeros((n_bins, n_donors, max_reduced), dtype="<f8")
    reduced_q = np.zeros_like(reduced_design)
    reduced_r = np.zeros((n_bins, max_reduced, max_reduced), dtype="<f8")
    full_q = np.zeros((n_replicates, n_bins, n_donors, max_full), dtype="<f8")
    full_r = np.zeros((n_replicates, n_bins, max_full, max_full), dtype="<f8")
    condition_columns = np.full((n_replicates, n_bins, n_donors), 255, dtype=np.uint8)
    reduced_rank = np.zeros((n_replicates, n_bins), dtype=np.int16)
    full_rank = np.zeros_like(reduced_rank)
    residual_df = np.zeros_like(reduced_rank)
    n_case = np.zeros_like(reduced_rank)
    n_control = np.zeros_like(reduced_rank)
    condition_information = np.zeros((n_replicates, n_bins), dtype="<f8")
    condition_vif = np.zeros_like(condition_information)
    reduced_rank_threshold = np.zeros_like(condition_information)
    full_rank_threshold = np.zeros_like(condition_information)
    estimable = np.zeros((n_replicates, n_bins), dtype=bool)
    base = caches[0]
    for bin_index, item in enumerate(base.bins):
        n = len(item.available_donor_indices)
        k0 = item.reduced_r.shape[0]
        available_indices[bin_index, :n] = item.available_donor_indices
        available_count[bin_index] = n
        active_experiment_indices[bin_index, :k0] = item.active_experiment_indices
        active_experiment_count[bin_index] = k0
        reduced_design[bin_index, :n, :k0] = base.experiment_fractions[
            item.available_donor_indices, bin_index
        ][:, item.active_experiment_indices]
        reduced_q[bin_index, :n, :k0] = item.reduced_q
        reduced_r[bin_index, :k0, :k0] = item.reduced_r
    for replicate_index, cache in enumerate(caches):
        _require_equal(cache.donor_ids, base.donor_ids, "factorized donor axes")
        _require_equal(cache.experiment_ids, base.experiment_ids, "factorized experiment axes")
        for bin_index, item in enumerate(cache.bins):
            n = len(item.available_donor_indices)
            k = item.full_r.shape[0]
            reference = base.bins[bin_index]
            _require(
                np.array_equal(item.available_donor_indices, reference.available_donor_indices),
                "Factorized availability changed across assignments",
            )
            full_q[replicate_index, bin_index, :n, :k] = item.full_q
            full_r[replicate_index, bin_index, :k, :k] = item.full_r
            condition_columns[replicate_index, bin_index, :n] = assignments[
                replicate_index, item.available_donor_indices
            ]
            reduced_rank[replicate_index, bin_index] = item.reduced_rank
            full_rank[replicate_index, bin_index] = item.full_rank
            residual_df[replicate_index, bin_index] = item.residual_df
            n_case[replicate_index, bin_index] = item.n_case
            n_control[replicate_index, bin_index] = item.n_control
            condition_information[replicate_index, bin_index] = item.condition_information
            condition_vif[replicate_index, bin_index] = item.condition_vif
            reduced_rank_threshold[replicate_index, bin_index] = item.reduced_rank_threshold
            full_rank_threshold[replicate_index, bin_index] = item.full_rank_threshold
            estimable[replicate_index, bin_index] = item.estimable
    return {
        "frozen_assignments_uint8": np.asarray(assignments, dtype=np.uint8),
        "availability_bool": np.asarray(base.availability, dtype=bool),
        "experiment_fractions_float64": np.asarray(
            base.experiment_fractions, dtype="<f8"
        ),
        "available_donor_indices_by_bin": available_indices,
        "available_donor_count_by_bin": available_count,
        "fixed_experiment_fraction_columns_by_bin": active_experiment_indices,
        "active_experiment_count_by_bin": active_experiment_count,
        "fixed_reduced_design_by_bin": reduced_design,
        "reusable_reduced_Q_by_bin": reduced_q,
        "reusable_reduced_R_by_bin": reduced_r,
        "condition_columns_for_all_500_assignments_by_bin": condition_columns,
        "reusable_full_Q_all_assignments_by_bin": full_q,
        "reusable_full_R_all_assignments_by_bin": full_r,
        "reduced_rank": reduced_rank,
        "full_rank": full_rank,
        "residual_df": residual_df,
        "n_case": n_case,
        "n_control": n_control,
        "condition_information": condition_information,
        "condition_vif": condition_vif,
        "reduced_rank_threshold": reduced_rank_threshold,
        "full_rank_threshold": full_rank_threshold,
        "estimable": estimable,
    }


def _factorized_cache_from_arrays(
    arrays: Mapping[str, np.ndarray],
    replicate_index: int,
    *,
    donor_ids: Sequence[str],
    experiment_ids: Sequence[str],
    condition: np.ndarray,
    availability: np.ndarray,
    experiment_fractions: np.ndarray,
) -> FactorizedNuisanceCache:
    bins: list[FactorizedBinDesign] = []
    available_counts = arrays["available_donor_count_by_bin"]
    active_counts = arrays["active_experiment_count_by_bin"]
    for bin_index in range(availability.shape[1]):
        n = int(available_counts[bin_index])
        k0 = int(active_counts[bin_index])
        k = k0 + 1
        indices = arrays["available_donor_indices_by_bin"][bin_index, :n].astype(np.int16)
        active = arrays["fixed_experiment_fraction_columns_by_bin"][
            bin_index, :k0
        ].astype(np.int16)
        reasons: list[str] = []
        if not bool(arrays["estimable"][replicate_index, bin_index]):
            reasons.append("cached_design_not_estimable")
        bins.append(
            FactorizedBinDesign(
                bin_index=bin_index,
                available_donor_indices=indices,
                active_experiment_indices=active,
                active_experiment_ids=tuple(str(experiment_ids[i]) for i in active),
                reduced_q=np.asarray(arrays["reusable_reduced_Q_by_bin"][bin_index, :n, :k0]),
                reduced_r=np.asarray(arrays["reusable_reduced_R_by_bin"][bin_index, :k0, :k0]),
                full_q=np.asarray(
                    arrays["reusable_full_Q_all_assignments_by_bin"][replicate_index, bin_index, :n, :k]
                ),
                full_r=np.asarray(
                    arrays["reusable_full_R_all_assignments_by_bin"][replicate_index, bin_index, :k, :k]
                ),
                reduced_rank=int(arrays["reduced_rank"][replicate_index, bin_index]),
                full_rank=int(arrays["full_rank"][replicate_index, bin_index]),
                residual_df=int(arrays["residual_df"][replicate_index, bin_index]),
                n_case=int(arrays["n_case"][replicate_index, bin_index]),
                n_control=int(arrays["n_control"][replicate_index, bin_index]),
                condition_information=float(
                    arrays["condition_information"][replicate_index, bin_index]
                ),
                condition_vif=float(arrays["condition_vif"][replicate_index, bin_index]),
                reduced_rank_threshold=float(
                    arrays["reduced_rank_threshold"][replicate_index, bin_index]
                ),
                full_rank_threshold=float(
                    arrays["full_rank_threshold"][replicate_index, bin_index]
                ),
                estimable=not reasons,
                reasons=tuple(reasons),
            )
        )
    globally_active = np.any(
        np.abs(np.where(availability[:, :, None], experiment_fractions, 0.0)) > 1e-12,
        axis=(0, 1),
    )
    return FactorizedNuisanceCache(
        donor_ids=tuple(donor_ids),
        experiment_ids=tuple(experiment_ids),
        retained_experiment_ids=tuple(
            str(experiment_ids[index]) for index in np.flatnonzero(globally_active)
        ),
        dropped_global_all_zero_experiment_ids=tuple(
            str(experiment_ids[index]) for index in np.flatnonzero(~globally_active)
        ),
        condition=np.asarray(condition, dtype=np.uint8),
        availability=np.asarray(availability, dtype=bool),
        experiment_fractions=np.asarray(experiment_fractions, dtype=float),
        bins=tuple(bins),
        rank_tolerance=RANK_RELATIVE_TOLERANCE,
    )


def _require_execution_authorization(
    config: Mapping[str, Any], explicit_execution_authorization: bool
) -> None:
    if explicit_execution_authorization is not True:
        raise CB2500ContractError(
            "CB2-500 requires explicit_execution_authorization=True"
        )
    firewall = config.get("authorization_and_firewall", {})
    for key in (
        "cb2_500_execution_authorized",
        "explicit_user_authorization_recorded",
    ):
        _require_equal(firewall.get(key), True, f"authorization_and_firewall.{key}")
    _require_equal(firewall.get("real_condition_labels_read"), False, "real condition firewall")
    _require_equal(firewall.get("timing_computed"), False, "timing firewall")
    _require_equal(firewall.get("timing_fields_present"), False, "timing field firewall")


def _verify_cb2a_gate(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    binding = config["bindings"]["cb2a_v2_decision"]
    path = _repo_file(root, str(binding["relative_path"]))
    _require_equal(_hash_file(path), str(binding["sha256"]), "CB2a-v2 decision hash")
    decision = _strict_json_load(path)
    for key in (
        "cb2a_pass",
        "cb2a_v2_pass",
        "cb2_500_technical_gate_pass",
        "cb2_500_start_allowed",
        "design_readiness_pass",
        "implementation_readiness_pass",
    ):
        _require_equal(decision.get(key), True, f"CB2a-v2.{key}")
    _require_equal(decision.get("assignment_generator_imported_or_called"), False, "CB2a assignment generator")
    _require_equal(decision.get("expression_values_read"), False, "CB2a expression firewall")
    _require_equal(decision.get("timing_computed"), False, "CB2a timing firewall")
    return decision


def _load_pathway_contract(
    root: Path, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, tuple[str, ...], pd.DataFrame, tuple[str, ...], dict[str, int]]:
    bindings = config["bindings"]
    universe = pd.read_csv(
        _repo_file(root, bindings["frozen_pathway_universe_v1"]["relative_path"]),
        sep="\t",
        dtype="string",
        keep_default_na=False,
    )
    _require_equal(len(universe), 7254, "frozen pathway membership rows")
    pathway_ids = tuple(universe["pathway_id"].drop_duplicates().astype(str))
    _require_equal(len(pathway_ids), 50, "frozen pathway axis")
    families = pd.read_csv(
        _repo_file(root, bindings["frozen_pathway_families_v1"]["relative_path"]),
        sep="\t",
        dtype="string",
        keep_default_na=False,
    )
    level_one = families.loc[families["analysis_level"].eq("level_1")].copy()
    level_one = level_one.sort_values("family_id", kind="mergesort").reset_index(drop=True)
    _require_equal(len(level_one), 13, "level-1 family rows")
    family_ids = tuple(level_one["family_id"].astype(str))
    family_index: dict[str, int] = {}
    for index, row in enumerate(level_one.itertuples(index=False)):
        members = json.loads(str(row.pathways_json))
        _require_equal(len(members), int(row.n_pathways), f"family {row.family_id} members")
        for pathway in members:
            _require(str(pathway) in pathway_ids, f"Unknown pathway in family {row.family_id}")
            _require(str(pathway) not in family_index, "Level-1 families must be disjoint")
            family_index[str(pathway)] = index
    _require_equal(len(family_index), 36, "assigned level-1 pathways")
    return universe, pathway_ids, level_one, family_ids, family_index


def _read_bound_raw_counts_float32(
    path: str | Path, frozen_cell_ids: Sequence[Any]
) -> pd.DataFrame:
    """Read the bound raw-count archive without casting its feature-id index."""

    cells = tuple(str(value) for value in frozen_cell_ids)
    _require(cells and len(cells) == len(set(cells)), "Frozen raw-count cell ids must be unique")
    return pd.read_csv(
        Path(path),
        index_col=0,
        dtype={cell_id: np.float32 for cell_id in cells},
        compression="zip",
    )


def _cache_manifest_payload(
    directory: Path,
    config: Mapping[str, Any],
    *,
    verified_bindings: Mapping[str, Any],
    reference_library: float,
) -> dict[str, Any]:
    file_names = list(CACHE_FILES) + list(CACHE_AXIS_FILES) + [
        CACHE_MATCHED_SOURCE_AUDIT_FILE,
        CACHE_AUDIT_FILE,
        CACHE_OVERLAP_AUDIT_FILE,
    ]
    files = {
        name: {
            "sha256": _hash_file(directory / name),
            "size_bytes": int((directory / name).stat().st_size),
        }
        for name in file_names
    }
    return {
        "schema_name": "trajpathmix_corebench_cb2_500_cache_manifest",
        "schema_version": "1.0.0",
        "execution_id": EXECUTION_ID,
        "execution_config_payload_sha256": config["_config_payload_sha256"],
        "bound_assignment_manifest_sha256": config["bindings"]["frozen_assignment_manifest_v1"]["sha256"],
        "bound_input_sha256_set": {
            name: binding["sha256"] for name, binding in config["bindings"].items()
        },
        "exact_cache_item_type_count": 8,
        "cache_items": list(CACHE_FILES),
        "ancillary_files": list(CACHE_AXIS_FILES)
        + [CACHE_MATCHED_SOURCE_AUDIT_FILE, CACHE_AUDIT_FILE, CACHE_OVERLAP_AUDIT_FILE],
        "files": files,
        "reference_library": float(reference_library),
        "verified_binding_count": int(len(verified_bindings)),
        "pathway_score_values_present": True,
        "outcome_role": "statistical_benchmark_only",
        "biological_interpretation": False,
        "real_condition_labels_present": False,
        "timing_computed": False,
        "timing_fields_present": False,
    }


def build_cb2_500_cache(
    config_path: str | Path = DEFAULT_CONFIG_FILE,
    repository_root: str | Path = ".",
    output_dir: str | Path | None = None,
    *,
    explicit_execution_authorization: bool = False,
) -> dict[str, Any]:
    """Build and atomically publish the exact eight create-only CB2-500 caches."""

    root = Path(repository_root).resolve()
    config = load_cb2_500_execution_config(
        _repo_file(root, str(config_path)) if not Path(config_path).is_absolute() else config_path,
        require_frozen=True,
    )
    _require_execution_authorization(config, explicit_execution_authorization)
    target = (
        _repo_file(root, config["cache_contract"]["default_cache_dir"])
        if output_dir is None
        else Path(output_dir).resolve()
    )
    # Fail before the multi-gigabyte raw binding hash when create-only
    # publication is already impossible.
    if target.exists():
        raise FileExistsError(f"Create-only CB2-500 cache already exists: {target}")
    _verify_cb2a_gate(root, config)
    verified_bindings = verify_cb2_500_bindings(root, config)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.parent / f".{target.name}.create.lock"
    with lock.open("xb") as _lock_handle:
        # The zero-byte file is the create-only claim; close its handle so the
        # finally block can remove it on Windows after publish or failure.
        _lock_handle.close()
        if target.exists():
            lock.unlink(missing_ok=True)
            raise FileExistsError(f"Create-only CB2-500 cache already exists: {target}")
        try:
            temporary = Path(
                tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
            )
        except BaseException:
            lock.unlink(missing_ok=True)
            raise
        try:
            _, donors, assignments, assignment_ids, assignment_hashes = _load_frozen_assignments(
                root, config
            )
            structural = _load_structural_inputs(root, config, donors)
            universe, pathway_ids, families, family_ids, family_lookup = _load_pathway_contract(
                root, config
            )
            bindings = config["bindings"]
            coordinate_genes = pd.read_csv(
                _repo_file(root, bindings["coordinate_gene_folds_v1"]["relative_path"]),
                sep="\t",
                dtype="string",
                keep_default_na=False,
            )
            coordinate_gene_bool = _read_bool_column(
                coordinate_genes["coordinate_gene"], "coordinate_gene"
            )
            coordinate_gene_ids = set(
                coordinate_genes.loc[
                    coordinate_gene_bool, "ensembl_gene_id"
                ].astype(str)
            )
            pathway_gene_ids = set(universe["gene_id"].astype(str))
            overlap = sorted(coordinate_gene_ids & pathway_gene_ids)
            _require_equal(overlap, [], "coordinate/pathway gene overlap")
            exclusion_path = _repo_file(
                root,
                bindings["coordinate_gene_exclusion_audit_v1"]["relative_path"],
            )
            exclusion_audit = _strict_json_load(exclusion_path)
            _require_equal(
                exclusion_audit.get("coordinate_injection_gene_overlap"),
                0,
                "bound coordinate exclusion audit overlap",
            )
            _require_equal(
                exclusion_audit.get("coordinate_gene_count"),
                len(coordinate_gene_ids),
                "bound coordinate gene count",
            )
            _write_json(
                {
                    "schema_name": "trajpathmix_corebench_cb2_500_pre_scoring_overlap_audit",
                    "schema_version": "1.0.0",
                    "execution_id": EXECUTION_ID,
                    "coordinate_gene_count": len(coordinate_gene_ids),
                    "frozen_pathway_unique_gene_count": len(pathway_gene_ids),
                    "coordinate_pathway_gene_overlap_count": 0,
                    "coordinate_pathway_gene_overlap": [],
                    "coordinate_gene_folds_sha256": bindings["coordinate_gene_folds_v1"]["sha256"],
                    "coordinate_gene_exclusion_audit_sha256": bindings["coordinate_gene_exclusion_audit_v1"]["sha256"],
                    "frozen_pathway_universe_sha256": bindings["frozen_pathway_universe_v1"]["sha256"],
                    "pass": True,
                    "pathway_scoring_allowed": True,
                    "biological_interpretation": False,
                    "timing_computed": False,
                    "timing_fields_present": False,
                },
                temporary / CACHE_OVERLAP_AUDIT_FILE,
            )
            metadata = pd.read_csv(
                _repo_file(root, bindings["cell_metadata"]["relative_path"]),
                sep="\t",
                usecols=[
                    "cell_name",
                    "donor",
                    "donor_long_id",
                    "experiment",
                    "day",
                    "size_factor",
                    "total_counts_endogenous",
                ],
                dtype={"cell_name": "string"},
            ).rename(
                columns={
                    "cell_name": "cell_id",
                    "donor": "donor_id",
                    "donor_long_id": "line_id",
                    "experiment": "experiment_id",
                }
            )
            _require_equal(len(metadata), 36044, "metadata cell axis length")
            _require_equal(int(metadata["cell_id"].nunique()), 36044, "metadata cell ids")
            frozen_cells = pd.read_csv(
                _repo_file(root, bindings["frozen_cell_order_v2"]["relative_path"]),
                sep="\t",
                dtype={"cell_id": "string"},
            )
            _require_equal(tuple(frozen_cells.columns), ("row_index", "cell_id"), "cell order columns")
            _require(
                np.array_equal(frozen_cells["row_index"].to_numpy(dtype=int), np.arange(36044)),
                "Frozen cell row index is not contiguous",
            )
            frozen_cell_ids = frozen_cells["cell_id"].astype(str).to_numpy()
            _require(
                np.array_equal(metadata["cell_id"].astype(str).to_numpy(), frozen_cell_ids),
                "Renamed metadata axis differs from frozen cell order",
            )
            coordinate = structural["coordinate"].set_index("cell_id", verify_integrity=True)
            _require_equal(set(coordinate.index.astype(str)), set(frozen_cell_ids), "coordinate cell identity")
            coordinate = coordinate.reindex(frozen_cell_ids)
            for metadata_column, coordinate_column in (
                ("donor_id", "donor_id"),
                ("line_id", "line_id"),
                ("experiment_id", "experiment_id"),
                ("day", "day"),
            ):
                _require(
                    np.array_equal(
                        metadata[metadata_column].astype(str).to_numpy(),
                        coordinate[coordinate_column].astype(str).to_numpy(),
                    ),
                    f"Metadata-coordinate mismatch for {metadata_column}",
                )

            # This is the one expression materialization.  It normalizes the
            # complete bound 36,044-cell axis before applying the primary filter.
            raw = _read_bound_raw_counts_float32(
                _repo_file(root, bindings["raw_counts_archive"]["relative_path"]),
                frozen_cell_ids,
            )
            _require_equal(raw.shape[1], 36044, "raw count cell axis length")
            _require(
                np.array_equal(raw.columns.astype(str).to_numpy(), frozen_cell_ids),
                "Raw count cell axis differs from frozen cell order",
            )
            raw_feature_ids = tuple(raw.index.astype(str))
            parsed_gene_ids = tuple(value.split("_", 1)[0] for value in raw_feature_ids)
            _require(all(parsed_gene_ids), "Empty parsed raw Ensembl identifier")
            _require_equal(len(parsed_gene_ids), len(set(parsed_gene_ids)), "parsed raw gene ids")
            pathway_gene_set = set(universe["gene_id"].astype(str))
            matched_raw_indices = np.asarray(
                [index for index, gene in enumerate(parsed_gene_ids) if gene in pathway_gene_set],
                dtype=np.int64,
            )
            matched_gene_ids = tuple(parsed_gene_ids[index] for index in matched_raw_indices)
            _require_equal(len(matched_gene_ids), 2995, "matched pathway genes")
            matched_index = {gene: index for index, gene in enumerate(matched_gene_ids)}
            pathway_index = {pathway: index for index, pathway in enumerate(pathway_ids)}
            membership = np.zeros((2995, 50), dtype=bool)
            audit_rows: list[dict[str, Any]] = []
            for source_row_index, row in enumerate(universe.itertuples(index=False)):
                gene = str(row.gene_id)
                if gene not in matched_index:
                    continue
                gene_index = matched_index[gene]
                p_index = pathway_index[str(row.pathway_id)]
                _require(not membership[gene_index, p_index], "Duplicate matched membership")
                membership[gene_index, p_index] = True
                audit_rows.append(
                    {
                        "source_row_index": source_row_index,
                        "pathway_id": str(row.pathway_id),
                        "source_ensembl_gene_id": gene,
                        "matched_gene_index": gene_index,
                        "matched_pathway_index": p_index,
                    }
                )
            _require_equal(int(membership.sum()), 5148, "matched memberships")
            per_pathway = membership.sum(axis=0)
            _require_equal(int(per_pathway.min()), 14, "minimum matched genes per pathway")
            _require_equal(int(per_pathway.max()), 198, "maximum matched genes per pathway")

            size_factor = metadata["size_factor"].to_numpy(dtype=np.float32)
            total_counts = metadata["total_counts_endogenous"].to_numpy(dtype=np.float64)
            _require(
                np.isfinite(size_factor).all() and np.all(size_factor > 0.0),
                "Deposited size factors must be finite and positive",
            )
            _require(
                np.isfinite(total_counts).all() and np.all(total_counts >= 0.0),
                "Total endogenous counts must be finite and nonnegative",
            )
            reference_library = float(
                np.median(total_counts / size_factor.astype(np.float64))
            )
            _require(
                math.isfinite(reference_library) and reference_library > 0.0,
                "Reference library must be finite and positive",
            )
            multiplier = np.divide(
                np.float32(1_000_000.0 / reference_library),
                size_factor,
                dtype=np.float32,
            )
            expression = raw.to_numpy(dtype=np.float32, copy=False)
            _require(
                np.isfinite(expression).all() and np.all(expression >= 0.0),
                "Raw count matrix must be finite and nonnegative",
            )
            expression *= multiplier[None, :]
            np.add(expression, np.float32(1.0), out=expression)
            np.log2(expression, out=expression)
            matched_expression = np.asarray(expression[matched_raw_indices], dtype=np.float32)
            del raw, expression

            donor_index = {donor: index for index, donor in enumerate(donors)}
            donor_values = coordinate["donor_id"].astype(str).to_numpy()
            bin_values = coordinate["bin_id"].to_numpy(dtype=np.int64)
            pseudobulk = np.full((75, 20, 2995), np.nan, dtype="<f8")
            for donor, donor_position in donor_index.items():
                for bin_index in range(20):
                    cell_indices = np.flatnonzero(
                        (donor_values == donor) & (bin_values == bin_index)
                    )
                    _require_equal(
                        len(cell_indices),
                        int(structural["cell_count"][donor_position, bin_index]),
                        f"{donor} bin {bin_index} cell count",
                    )
                    if not structural["availability"][donor_position, bin_index]:
                        continue
                    _require(len(cell_indices) >= 5, "Available donor-bin has fewer than five cells")
                    pseudobulk[donor_position, bin_index] = np.mean(
                        matched_expression[:, cell_indices],
                        axis=1,
                        dtype=np.float32,
                    ).astype(np.float64)
            del matched_expression
            _require(np.isfinite(pseudobulk[structural["availability"]]).all(), "Pseudobulk finite gate")
            _require(np.isnan(pseudobulk[~structural["availability"]]).all(), "Pseudobulk NA mask")
            flat = pseudobulk.reshape(-1, 2995)
            center = np.nanmean(flat, axis=0, dtype=np.float64)
            scale = np.nanstd(flat, axis=0, ddof=1, dtype=np.float64)
            floor = 1.0e-6 * np.maximum(np.abs(center), 1.0)
            near_constant = ~np.isfinite(scale) | (scale <= floor)
            effective_scale = scale.copy()
            effective_scale[near_constant] = 1.0
            standardized = (flat - center[None, :]) / effective_scale[None, :]
            standardized[:, near_constant] = np.where(
                np.isfinite(flat[:, near_constant]), 0.0, np.nan
            )
            scores_flat = np.empty((flat.shape[0], 50), dtype="<f8")
            for p_index in range(50):
                scores_flat[:, p_index] = np.mean(
                    standardized[:, membership[:, p_index]], axis=1, dtype=np.float64
                )
            scores = scores_flat.reshape(75, 20, 50)
            _require(np.isfinite(scores[structural["availability"]]).all(), "Pathway score finite gate")
            _require(np.isnan(scores[~structural["availability"]]).all(), "Pathway score NA mask")

            _write_npy(temporary / CACHE_FILES[0], pseudobulk.astype("<f8", copy=False))
            _write_npy(temporary / CACHE_FILES[1], membership)
            _write_npy(temporary / CACHE_FILES[2], scores.astype("<f8", copy=False))
            factorized = [
                build_factorized_nuisance_cache(
                    donors,
                    assignments[index],
                    structural["availability"],
                    structural["experiment_fractions"],
                    structural["experiments"],
                )
                for index in range(500)
            ]
            _require(all(cache.all_estimable for cache in factorized), "All 500 designs must be estimable")
            stacked_design = _stack_factorized_caches(factorized, assignments)
            del factorized
            _write_deterministic_npz(
                temporary / CACHE_FILES[3], stacked_design
            )
            del stacked_design
            availability_frame = pd.DataFrame(
                {
                    "donor_index": np.arange(75, dtype=int),
                    "donor_id": donors,
                    "full_availability_signature": structural["availability_signatures"],
                }
            )
            _write_table(availability_frame, temporary / CACHE_FILES[4])
            mapping_bank = np.empty((500, 999, 75), dtype=np.uint8)
            mapping_audits: list[dict[str, Any]] = []
            for replicate_index in range(500):
                seed = derive_residual_mapping_seed(replicate_index)
                mapping = build_availability_mapping_bank(
                    donors, structural["availability"], n_mappings=999, seed=seed
                )
                mapping_bank[replicate_index] = mapping.astype(np.uint8)
                audit = _mapping_audit(donors, structural["availability"], mapping, seed)
                mapping_audits.append(
                    {
                        "replicate_index_0based": replicate_index,
                        "assignment_id": assignment_ids[replicate_index],
                        "seed_uint64": seed,
                        "stream_sha256": audit["mapping_stream_sha256"],
                        "n_unique_mappings": audit["n_unique_mapping_hashes"],
                        "identity_mapping_present": audit["identity_mapping_present"],
                    }
                )
            _write_npy(temporary / CACHE_FILES[5], mapping_bank)
            family_indices = np.asarray(
                [family_lookup.get(pathway, -1) for pathway in pathway_ids], dtype=int
            )
            _require_equal(int(np.sum(family_indices >= 0)), 36, "assigned family pathways")
            _write_table(
                pd.DataFrame(
                    {
                        "pathway_index": np.arange(50, dtype=int),
                        "pathway_id": pathway_ids,
                        "level_1_family_index": family_indices,
                    }
                ),
                temporary / CACHE_FILES[6],
            )
            _write_npy(temporary / CACHE_FILES[7], np.ones((20, 50), dtype=bool))

            axis_frames = (
                pd.DataFrame({"donor_index": range(75), "donor_id": donors}),
                pd.DataFrame({"bin_index": range(20)}),
                pd.DataFrame(
                    {
                        "matched_gene_index": range(2995),
                        "ensembl_gene_id": matched_gene_ids,
                        "raw_feature_id": [raw_feature_ids[index] for index in matched_raw_indices],
                    }
                ),
                pd.DataFrame({"pathway_index": range(50), "pathway_id": pathway_ids}),
                pd.DataFrame(
                    {
                        "replicate_index_0based": range(500),
                        "assignment_id": assignment_ids,
                        "assignment_sha256": assignment_hashes,
                    }
                ),
                pd.DataFrame(
                    {"experiment_index": range(28), "experiment_id": structural["experiments"]}
                ),
                pd.DataFrame({"family_index": range(13), "family_id": family_ids}),
            )
            for name, frame in zip(CACHE_AXIS_FILES, axis_frames, strict=True):
                _write_table(frame, temporary / name)
            _write_table(pd.DataFrame(audit_rows), temporary / CACHE_MATCHED_SOURCE_AUDIT_FILE)
            _write_table(pd.DataFrame(mapping_audits), temporary / CACHE_AUDIT_FILE)
            manifest = _cache_manifest_payload(
                temporary,
                config,
                verified_bindings=verified_bindings,
                reference_library=reference_library,
            )
            _write_json(manifest, temporary / CACHE_MANIFEST_FILE)
            build_record = {
                "schema_name": "trajpathmix_corebench_cb2_500_cache_build_record",
                "schema_version": "1.0.0",
                "execution_id": EXECUTION_ID,
                "execution_config_payload_sha256": config["_config_payload_sha256"],
                "cache_manifest_sha256": _hash_file(temporary / CACHE_MANIFEST_FILE),
                "cache_item_sha256": {
                    name: manifest["files"][name]["sha256"] for name in CACHE_FILES
                },
                "raw_archive_materializations": 1,
                "normalization_full_cell_axis_count": 36044,
                "primary_filter_applied_after_normalization": True,
                "assignment_generator_imported_or_called": False,
                "real_condition_contrast_generated": False,
                "timing_computed": False,
                "timing_fields_present": False,
            }
            _write_json(build_record, temporary / CACHE_BUILD_RECORD_FILE)
            os.replace(temporary, target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        finally:
            try:
                lock.unlink()
            except FileNotFoundError:
                pass
    return validate_cb2_500_cache(config_path, root, target)


def validate_cb2_500_cache(
    config_path: str | Path = DEFAULT_CONFIG_FILE,
    repository_root: str | Path = ".",
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Fail-closed validation of a published cache without reading expression data."""

    root = Path(repository_root).resolve()
    resolved_config = (
        _repo_file(root, str(config_path)) if not Path(config_path).is_absolute() else Path(config_path)
    )
    config = load_cb2_500_execution_config(resolved_config, require_frozen=True)
    verify_cb2_500_source_bindings(root, config)
    directory = (
        _repo_file(root, config["cache_contract"]["default_cache_dir"])
        if cache_dir is None
        else Path(cache_dir).resolve()
    )
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    expected_files = set(CACHE_FILES) | set(CACHE_AXIS_FILES) | {
        CACHE_MATCHED_SOURCE_AUDIT_FILE,
        CACHE_AUDIT_FILE,
        CACHE_OVERLAP_AUDIT_FILE,
        CACHE_MANIFEST_FILE,
        CACHE_BUILD_RECORD_FILE,
    }
    observed_files = {path.name for path in directory.iterdir() if path.is_file()}
    _require_equal(observed_files, expected_files, "cache directory file set")
    _require(not any(path.is_dir() for path in directory.iterdir()), "Cache directory contains subdirectories")
    manifest = _strict_json_load(directory / CACHE_MANIFEST_FILE)
    _require_equal(manifest.get("schema_name"), "trajpathmix_corebench_cb2_500_cache_manifest", "cache manifest schema")
    _require_equal(manifest.get("schema_version"), "1.0.0", "cache manifest version")
    _require_equal(manifest.get("execution_id"), EXECUTION_ID, "cache execution id")
    _require_equal(
        manifest.get("execution_config_payload_sha256"),
        config["_config_payload_sha256"],
        "cache config payload hash",
    )
    _require_equal(manifest.get("exact_cache_item_type_count"), 8, "cache item count")
    _require_equal(tuple(manifest.get("cache_items", [])), CACHE_FILES, "cache items")
    _require_equal(
        manifest.get("bound_assignment_manifest_sha256"),
        config["bindings"]["frozen_assignment_manifest_v1"]["sha256"],
        "cache assignment binding",
    )
    expected_bound_hashes = {
        name: value["sha256"] for name, value in config["bindings"].items()
    }
    _require_equal(manifest.get("bound_input_sha256_set"), expected_bound_hashes, "cache binding hash set")
    _require_equal(manifest.get("pathway_score_values_present"), True, "cache pathway-score declaration")
    _require_equal(manifest.get("outcome_role"), "statistical_benchmark_only", "cache outcome role")
    _require_equal(manifest.get("biological_interpretation"), False, "cache interpretation firewall")
    _require_equal(manifest.get("real_condition_labels_present"), False, "cache condition firewall")
    _require_equal(manifest.get("timing_computed"), False, "cache timing firewall")
    _require_equal(manifest.get("timing_fields_present"), False, "cache timing-field firewall")
    file_records = manifest.get("files", {})
    hashed_names = set(CACHE_FILES) | set(CACHE_AXIS_FILES) | {
        CACHE_MATCHED_SOURCE_AUDIT_FILE,
        CACHE_AUDIT_FILE,
        CACHE_OVERLAP_AUDIT_FILE,
    }
    _require_equal(set(file_records), hashed_names, "cache manifest hashed files")
    for name in sorted(hashed_names):
        path = directory / name
        record = file_records[name]
        _require_equal(_hash_file(path), record.get("sha256"), f"cache file hash {name}")
        _require_equal(int(path.stat().st_size), int(record.get("size_bytes", -1)), f"cache file size {name}")

    pseudobulk = np.load(directory / CACHE_FILES[0], mmap_mode="r", allow_pickle=False)
    _require_equal(pseudobulk.shape, (75, 20, 2995), "pseudobulk shape")
    _require_equal(pseudobulk.dtype, np.dtype("<f8"), "pseudobulk dtype")
    membership = np.load(directory / CACHE_FILES[1], mmap_mode="r", allow_pickle=False)
    _require_equal(membership.shape, (2995, 50), "membership shape")
    _require_equal(membership.dtype, np.dtype(bool), "membership dtype")
    _require_equal(int(membership.sum()), 5148, "membership true count")
    scores = np.load(directory / CACHE_FILES[2], mmap_mode="r", allow_pickle=False)
    _require_equal(scores.shape, (75, 20, 50), "score shape")
    _require_equal(scores.dtype, np.dtype("<f8"), "score dtype")
    mappings = np.load(directory / CACHE_FILES[5], mmap_mode="r", allow_pickle=False)
    _require_equal(mappings.shape, (500, 999, 75), "mapping bank shape")
    _require_equal(mappings.dtype, np.dtype(np.uint8), "mapping bank dtype")
    family = pd.read_csv(directory / CACHE_FILES[6], sep="\t", dtype={"pathway_id": "string"})
    _require_equal(
        tuple(family.columns),
        ("pathway_index", "pathway_id", "level_1_family_index"),
        "family index columns",
    )
    _require_equal(len(family), 50, "family index rows")
    family_values = family["level_1_family_index"].to_numpy(dtype=int)
    _require_equal(int(np.sum(family_values >= 0)), 36, "family assigned count")
    _require_equal(int(np.sum(family_values == -1)), 14, "family unassigned count")
    _require_equal(tuple(sorted(set(family_values[family_values >= 0]))), tuple(range(13)), "family codes")
    mask = np.load(directory / CACHE_FILES[7], mmap_mode="r", allow_pickle=False)
    _require_equal(mask.shape, (20, 50), "band mask shape")
    _require_equal(mask.dtype, np.dtype(bool), "band mask dtype")
    _require(bool(mask.all()), "Every simultaneous-band mask entry must be true")
    availability = pd.read_csv(directory / CACHE_FILES[4], sep="\t", dtype="string")
    _require_equal(
        tuple(availability.columns),
        ("donor_index", "donor_id", "full_availability_signature"),
        "availability signature columns",
    )
    _require_equal(len(availability), 75, "availability signature rows")
    _require(
        availability["full_availability_signature"].str.fullmatch("[01]{20}").all(),
        "Availability signature encoding is invalid",
    )
    signature_matrix = np.asarray(
        [
            [character == "1" for character in signature]
            for signature in availability["full_availability_signature"].astype(str)
        ],
        dtype=bool,
    )
    _require_equal(len(set(availability["full_availability_signature"].astype(str))), 58, "unique availability signatures")
    matched_audit = pd.read_csv(
        directory / CACHE_MATCHED_SOURCE_AUDIT_FILE,
        sep="\t",
        dtype={"pathway_id": "string", "source_ensembl_gene_id": "string"},
    )
    _require_equal(
        tuple(matched_audit.columns),
        (
            "source_row_index",
            "pathway_id",
            "source_ensembl_gene_id",
            "matched_gene_index",
            "matched_pathway_index",
        ),
        "matched membership audit columns",
    )
    _require_equal(len(matched_audit), 5148, "matched membership audit rows")
    overlap_audit = _strict_json_load(directory / CACHE_OVERLAP_AUDIT_FILE)
    _require_equal(overlap_audit.get("pass"), True, "pre-scoring overlap audit")
    _require_equal(overlap_audit.get("coordinate_pathway_gene_overlap_count"), 0, "pre-scoring overlap count")
    mapping_audit = pd.read_csv(directory / CACHE_AUDIT_FILE, sep="\t", dtype="string")
    required_audit = (
        "replicate_index_0based",
        "assignment_id",
        "seed_uint64",
        "stream_sha256",
        "n_unique_mappings",
        "identity_mapping_present",
    )
    _require_equal(tuple(mapping_audit.columns), required_audit, "mapping audit columns")
    _require_equal(len(mapping_audit), 500, "mapping audit rows")
    for index in range(500):
        stream = np.asarray(mappings[index], dtype=np.int32)
        record = mapping_audit.iloc[index]
        _require_equal(int(record["replicate_index_0based"]), index, "mapping audit index")
        _require_equal(str(record["assignment_id"]), f"CB2P_{index + 1:04d}", "mapping audit assignment")
        _require_equal(int(record["seed_uint64"]), derive_residual_mapping_seed(index), "mapping seed")
        _require_equal(
            hashlib.sha256(np.ascontiguousarray(stream, dtype="<i4").tobytes()).hexdigest(),
            str(record["stream_sha256"]),
            "mapping stream hash",
        )
        _require_equal(int(record["n_unique_mappings"]), 999, "unique mappings")
        _require_equal(str(record["identity_mapping_present"]).lower(), "false", "identity mapping flag")
        identity = np.arange(75, dtype=np.int32)
        _require(
            np.all(np.sort(stream, axis=1) == identity[None, :]),
            f"Mapping stream {index} contains a non-permutation",
        )
        _require_equal(len(np.unique(stream, axis=0)), 999, f"mapping stream {index} uniqueness")
        _require(
            not np.any(np.all(stream == identity[None, :], axis=1)),
            f"Mapping stream {index} contains identity",
        )
        _require(
            np.array_equal(signature_matrix[stream], np.broadcast_to(signature_matrix, (999, 75, 20))),
            f"Mapping stream {index} crossed an availability signature",
        )
    with np.load(directory / CACHE_FILES[3], allow_pickle=False) as design:
        required_design_names = {
            "frozen_assignments_uint8",
            "availability_bool",
            "experiment_fractions_float64",
            "available_donor_indices_by_bin",
            "available_donor_count_by_bin",
            "fixed_experiment_fraction_columns_by_bin",
            "active_experiment_count_by_bin",
            "fixed_reduced_design_by_bin",
            "reusable_reduced_Q_by_bin",
            "reusable_reduced_R_by_bin",
            "condition_columns_for_all_500_assignments_by_bin",
            "reusable_full_Q_all_assignments_by_bin",
            "reusable_full_R_all_assignments_by_bin",
            "reduced_rank",
            "full_rank",
            "residual_df",
            "n_case",
            "n_control",
            "condition_information",
            "condition_vif",
            "reduced_rank_threshold",
            "full_rank_threshold",
            "estimable",
        }
        _require_equal(set(design.files), required_design_names, "design cache NPZ members")
        _require_equal(design["frozen_assignments_uint8"].shape, (500, 75), "assignment cache shape")
        _require_equal(design["frozen_assignments_uint8"].dtype, np.dtype(np.uint8), "assignment cache dtype")
        _require(np.all(design["frozen_assignments_uint8"].sum(axis=1) == 37), "assignment cache case counts")
        _require_equal(design["availability_bool"].shape, (75, 20), "availability cache shape")
        _require_equal(design["availability_bool"].dtype, np.dtype(bool), "availability cache dtype")
        cached_availability = np.asarray(design["availability_bool"], dtype=bool)
        _require(np.array_equal(cached_availability, signature_matrix), "Availability TSV/design mismatch")
        _require(np.isfinite(pseudobulk[cached_availability]).all(), "Available pseudobulk must be finite")
        _require(np.isnan(pseudobulk[~cached_availability]).all(), "Unavailable pseudobulk must be NA")
        _require(np.isfinite(scores[cached_availability]).all(), "Available pathway scores must be finite")
        _require(np.isnan(scores[~cached_availability]).all(), "Unavailable pathway scores must be NA")
        _require_equal(design["experiment_fractions_float64"].shape[:2], (75, 20), "fraction cache axes")
        assignment_axis = pd.read_csv(
            directory / "assignment_axis_v1.tsv", sep="\t", dtype="string"
        )
        _require_equal(len(assignment_axis), 500, "assignment axis rows")
        for index, values in enumerate(design["frozen_assignments_uint8"]):
            _require_equal(
                _hash_array(values, np.uint8),
                str(assignment_axis.iloc[index]["assignment_sha256"]),
                f"cached assignment hash {index}",
            )
        _require_equal(design["available_donor_indices_by_bin"].shape[0], 20, "design bin axis")
        _require_equal(design["condition_columns_for_all_500_assignments_by_bin"].shape[:2], (500, 20), "condition cache axes")
        _require_equal(design["reusable_full_Q_all_assignments_by_bin"].shape[:2], (500, 20), "full Q axes")
        _require_equal(design["reusable_full_R_all_assignments_by_bin"].shape[:2], (500, 20), "full R axes")
        _require(bool(design["estimable"].all()), "Cached design estimability failure")
        _require_equal(int(design["n_case"].min()) >= 10, True, "cached minimum cases")
        _require_equal(int(design["n_control"].min()) >= 10, True, "cached minimum controls")
        _require_equal(int(design["residual_df"].min()) >= 3, True, "cached residual df")
        _require(float(design["condition_vif"].max()) <= 10.0, "Cached condition VIF exceeds gate")
        available_indices = np.asarray(design["available_donor_indices_by_bin"])
        available_counts = np.asarray(design["available_donor_count_by_bin"])
        active_counts = np.asarray(design["active_experiment_count_by_bin"])
        reduced_design = np.asarray(design["fixed_reduced_design_by_bin"])
        reduced_q = np.asarray(design["reusable_reduced_Q_by_bin"])
        reduced_r = np.asarray(design["reusable_reduced_R_by_bin"])
        full_q = np.asarray(design["reusable_full_Q_all_assignments_by_bin"])
        full_r = np.asarray(design["reusable_full_R_all_assignments_by_bin"])
        local_conditions = np.asarray(
            design["condition_columns_for_all_500_assignments_by_bin"]
        )
        assignment_values = np.asarray(design["frozen_assignments_uint8"])
        for bin_index in range(20):
            n = int(available_counts[bin_index])
            k0 = int(active_counts[bin_index])
            donors_local = available_indices[bin_index, :n].astype(int)
            _require(
                np.array_equal(np.sort(donors_local), np.flatnonzero(cached_availability[:, bin_index])),
                f"Available donor indices differ in bin {bin_index}",
            )
            reconstructed_reduced = (
                reduced_q[bin_index, :n, :k0]
                @ reduced_r[bin_index, :k0, :k0]
            )
            _require(
                float(
                    np.max(
                        np.abs(
                            reconstructed_reduced
                            - reduced_design[bin_index, :n, :k0]
                        ),
                        initial=0.0,
                    )
                )
                <= NUMERICAL_TOLERANCE,
                f"Reduced Q/R reconstruction failed in bin {bin_index}",
            )
            for replicate_index in range(500):
                condition_local = local_conditions[replicate_index, bin_index, :n]
                _require(
                    np.array_equal(
                        condition_local,
                        assignment_values[replicate_index, donors_local],
                    ),
                    f"Condition cache mismatch replicate {replicate_index} bin {bin_index}",
                )
                expected_full = np.column_stack(
                    [reduced_design[bin_index, :n, :k0], condition_local.astype(float)]
                )
                reconstructed_full = (
                    full_q[replicate_index, bin_index, :n, : k0 + 1]
                    @ full_r[
                        replicate_index,
                        bin_index,
                        : k0 + 1,
                        : k0 + 1,
                    ]
                )
                _require(
                    float(np.max(np.abs(reconstructed_full - expected_full), initial=0.0))
                    <= NUMERICAL_TOLERANCE,
                    f"Full Q/R reconstruction failed replicate {replicate_index} bin {bin_index}",
                )
    build_record = _strict_json_load(directory / CACHE_BUILD_RECORD_FILE)
    _require_equal(build_record.get("cache_manifest_sha256"), _hash_file(directory / CACHE_MANIFEST_FILE), "cache build manifest hash")
    _require_equal(build_record.get("raw_archive_materializations"), 1, "raw archive materialization count")
    _require_equal(build_record.get("assignment_generator_imported_or_called"), False, "cache assignment generator firewall")
    _require_equal(build_record.get("real_condition_contrast_generated"), False, "cache real-condition firewall")
    _require_equal(build_record.get("timing_computed"), False, "cache build timing firewall")
    _require_equal(
        build_record.get("cache_item_sha256"),
        {name: manifest["files"][name]["sha256"] for name in CACHE_FILES},
        "cache build item hashes",
    )
    return {
        "valid": True,
        "cache_dir": str(directory),
        "execution_config_payload_sha256": config["_config_payload_sha256"],
        "cache_manifest_sha256": _hash_file(directory / CACHE_MANIFEST_FILE),
        "cache_build_record_sha256": _hash_file(directory / CACHE_BUILD_RECORD_FILE),
        "cache_item_sha256": {name: _hash_file(directory / name) for name in CACHE_FILES},
    }


def _column(frame: pd.DataFrame, candidates: Sequence[str], label: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise CB2500ContractError(
        f"Missing {label}; expected one of {', '.join(candidates)}"
    )


def _wilson_interval(numerator: int, denominator: int) -> dict[str, Any]:
    if denominator <= 0 or numerator < 0 or numerator > denominator:
        raise CB2500ContractError("Invalid Wilson interval numerator/denominator")
    z = 1.959963984540054
    estimate = numerator / denominator
    z2 = z * z
    denominator_adjusted = 1.0 + z2 / denominator
    center = (estimate + z2 / (2.0 * denominator)) / denominator_adjusted
    half = (
        z
        * math.sqrt(
            estimate * (1.0 - estimate) / denominator
            + z2 / (4.0 * denominator * denominator)
        )
        / denominator_adjusted
    )
    return {
        "confidence_level": 0.95,
        "z_value": z,
        "lower": max(0.0, center - half),
        "upper": min(1.0, center + half),
        "descriptive_only": True,
    }


def _rate_gate(
    numerator: int,
    denominator: int,
    *,
    threshold: float,
    operator: str,
) -> dict[str, Any]:
    estimate = numerator / denominator
    if operator == "less_than_or_equal":
        passed = estimate <= threshold
    elif operator == "greater_than_or_equal":
        passed = estimate >= threshold
    else:
        raise CB2500ContractError(f"Unsupported threshold operator: {operator}")
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "estimate": float(estimate),
        "threshold_operator": operator,
        "threshold": float(threshold),
        "pass": bool(passed),
        "wilson_95": _wilson_interval(int(numerator), int(denominator)),
    }


def _pvalue_distribution(values: np.ndarray) -> dict[str, Any]:
    observed = np.asarray(values, dtype=float)
    observed = observed[np.isfinite(observed)]
    if not len(observed):
        return {"count": 0, "minimum": None, "quartiles": [], "median": None, "mean": None, "maximum": None, "decile_histogram": [0] * 10}
    histogram, _ = np.histogram(observed, bins=np.linspace(0.0, 1.0, 11))
    quantiles = np.quantile(observed, [0.25, 0.5, 0.75])
    return {
        "count": int(len(observed)),
        "minimum": float(np.min(observed)),
        "quartiles": [float(value) for value in quantiles],
        "median": float(np.median(observed)),
        "mean": float(np.mean(observed)),
        "maximum": float(np.max(observed)),
        "decile_histogram": histogram.astype(int).tolist(),
    }


def aggregate_cb2_500_acceptance(
    *,
    pathway_metrics: pd.DataFrame,
    family_metrics: pd.DataFrame,
    replicate_metrics: pd.DataFrame,
    refusal_audit: pd.DataFrame,
    curve_metrics: pd.DataFrame | None = None,
    alpha: float = DEFAULT_ALPHA,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate all acceptance endpoints with immutable planned denominators."""

    pathways = pathway_metrics.copy()
    families = family_metrics.copy()
    replicates = replicate_metrics.copy()
    refusals = refusal_audit.copy()
    replicate_column = _column(
        replicates, ("replicate_index_0based", "replicate_index"), "replicate index"
    )
    pathway_replicate = _column(
        pathways, ("replicate_index_0based", "replicate_index"), "pathway replicate index"
    )
    family_replicate = _column(
        families, ("replicate_index_0based", "replicate_index"), "family replicate index"
    )
    refusal_replicate = _column(
        refusals, ("replicate_index_0based", "replicate_index"), "refusal replicate index"
    )
    _require_equal(len(replicates), 500, "replicate metric rows")
    _require_equal(len(pathways), 25_000, "pathway metric rows")
    _require_equal(len(families), 6_500, "family metric rows")
    _require_equal(len(refusals), 500, "refusal audit rows")
    expected_replicates = set(range(500))
    for frame, column_name, label in (
        (replicates, replicate_column, "replicate metrics"),
        (refusals, refusal_replicate, "refusal audit"),
    ):
        values = frame[column_name].to_numpy(dtype=int)
        _require_equal(set(values), expected_replicates, f"{label} replicate coverage")
        _require_equal(len(set(values)), 500, f"{label} primary key")
    status_column = _column(
        refusals, ("replicate_status", "status"), "refusal status"
    )
    normalized_status = refusals[status_column].astype(str).str.strip().str.lower()
    completed_mask = normalized_status.isin({"complete", "completed", "success", "ok"})
    completed_replicates = set(
        refusals.loc[completed_mask, refusal_replicate].to_numpy(dtype=int)
    )
    n_completed = int(len(completed_replicates))
    n_refused = int(500 - n_completed)
    pathway_id = _column(pathways, ("pathway_id", "pathway"), "pathway id")
    family_id = _column(families, ("family_id", "level_1_family_id"), "family id")
    _require_equal(
        int(pathways.groupby(pathway_replicate, sort=False)[pathway_id].nunique().min()),
        50,
        "pathways per replicate",
    )
    _require_equal(
        int(pathways.groupby(pathway_replicate, sort=False)[pathway_id].nunique().max()),
        50,
        "pathways per replicate",
    )
    _require(not pathways.duplicated([pathway_replicate, pathway_id]).any(), "Duplicate pathway primary key")
    _require_equal(
        int(families.groupby(family_replicate, sort=False)[family_id].nunique().min()),
        13,
        "families per replicate",
    )
    _require_equal(
        int(families.groupby(family_replicate, sort=False)[family_id].nunique().max()),
        13,
        "families per replicate",
    )
    _require(not families.duplicated([family_replicate, family_id]).any(), "Duplicate family primary key")
    for values, label in (
        (set(pathways[pathway_replicate].astype(int)), "pathway"),
        (set(families[family_replicate].astype(int)), "family"),
    ):
        _require_equal(values, expected_replicates, f"{label} replicate coverage")
    alpha_value = float(alpha)
    _require(0.0 < alpha_value < 1.0, "alpha must lie strictly between zero and one")
    global_column = _column(
        pathways,
        ("global_50_curve_maxT_p_value", "global_curve_p_value", "global_maxT_p_value"),
        "global-50 curve p-value",
    )
    integrated_column = _column(
        pathways,
        ("raw_integrated_p_value", "integrated_p_raw", "p_integrated"),
        "raw integrated p-value",
    )
    by_column = _column(pathways, ("BY_q_value", "q_by", "curve_q_by"), "BY q-value")
    family_p_column = _column(
        families, ("family_maxT_p_value", "p_family", "family_p_value"), "family p-value"
    )
    raw_curve_column = next(
        (name for name in ("raw_curve_p_value", "curve_p_raw") if name in pathways.columns),
        global_column,
    )
    band_column = _column(
        replicates,
        (
            "global_band_coverage_indicator",
            "global_zero_curve_simultaneous_band_coverage_indicator",
            "global_band_covered",
            "simultaneous_band_covered",
        ),
        "global band coverage indicator",
    )
    for frame, column_name, replicate_name, label in (
        (pathways, global_column, pathway_replicate, "global curve p-values"),
        (pathways, raw_curve_column, pathway_replicate, "raw curve p-values"),
        (pathways, integrated_column, pathway_replicate, "integrated p-values"),
        (pathways, by_column, pathway_replicate, "BY q-values"),
        (families, family_p_column, family_replicate, "family p-values"),
    ):
        values = frame[column_name].to_numpy(dtype=float)
        completed_rows = frame[replicate_name].astype(int).isin(completed_replicates).to_numpy()
        _require(
            np.isfinite(values[completed_rows]).all()
            and np.all((values[completed_rows] >= 0.0) & (values[completed_rows] <= 1.0)),
            f"Completed {label} must be finite probabilities",
        )
        _require(
            np.isnan(values[~completed_rows]).all(),
            f"Refused {label} must be NA",
        )
    for frame, replicate_name, candidate in (
        (pathways, pathway_replicate, "global_50_curve_reject_alpha"),
        (pathways, pathway_replicate, "integrated_reject_alpha"),
        (pathways, pathway_replicate, "BY_reject_alpha"),
        (families, family_replicate, "family_reject_alpha"),
    ):
        if candidate in frame.columns:
            values = frame[candidate].to_numpy()
            _require(np.isin(values, [0, 1, False, True]).all(), f"{candidate} must be binary")
            refused_rows = ~frame[replicate_name].astype(int).isin(completed_replicates).to_numpy()
            _require(not np.asarray(values[refused_rows], dtype=bool).any(), f"Refused {candidate} must be false")
    band_values = replicates[band_column].to_numpy()
    _require(np.isin(band_values, [0, 1, False, True]).all(), "Band coverage must be binary")
    refused_replicate_rows = ~replicates[replicate_column].astype(int).isin(completed_replicates).to_numpy()
    _require(
        not np.asarray(band_values[refused_replicate_rows], dtype=bool).any(),
        "Refused global-band coverage indicators must be false",
    )
    global_any = (
        pathways.assign(_reject=pathways[global_column].astype(float) <= alpha_value)
        .groupby(pathway_replicate, sort=False)["_reject"]
        .any()
    )
    by_any = (
        pathways.assign(_reject=pathways[by_column].astype(float) <= alpha_value)
        .groupby(pathway_replicate, sort=False)["_reject"]
        .any()
    )
    thresholds = {
        "global": 0.06,
        "family": 0.06,
        "integrated": 0.06,
        "by": 0.06,
        "band": 0.93,
    }
    if config is not None:
        endpoint = config.get("acceptance_endpoints", {})
        thresholds = {
            "global": float(endpoint.get("global_50_curve_maxT_fwer", {}).get("threshold", 0.06)),
            "family": float(endpoint.get("family_13_macro_fwer", {}).get("threshold", 0.06)),
            "integrated": float(endpoint.get("pooled_integrated_effect_type_i_error", {}).get("threshold", 0.06)),
            "by": float(endpoint.get("complete_null_BY_fdr", {}).get("threshold", 0.06)),
            "band": float(endpoint.get("global_zero_curve_simultaneous_band_coverage", {}).get("threshold", 0.93)),
        }
    global_gate = _rate_gate(
        int(global_any.sum()), 500, threshold=thresholds["global"], operator="less_than_or_equal"
    )
    family_reject = families[family_p_column].astype(float) <= alpha_value
    family_gate = _rate_gate(
        int(family_reject.sum()), 6500, threshold=thresholds["family"], operator="less_than_or_equal"
    )
    integrated_reject = pathways[integrated_column].astype(float) <= alpha_value
    integrated_gate = _rate_gate(
        int(integrated_reject.sum()), 25_000, threshold=thresholds["integrated"], operator="less_than_or_equal"
    )
    by_gate = _rate_gate(
        int(by_any.sum()), 500, threshold=thresholds["by"], operator="less_than_or_equal"
    )
    band_gate = _rate_gate(
        int(np.asarray(band_values, dtype=bool).sum()),
        500,
        threshold=thresholds["band"],
        operator="greater_than_or_equal",
    )
    family_work = families[[family_id]].copy()
    family_work["reject"] = family_reject.to_numpy(dtype=bool)
    per_family: dict[str, Any] = {}
    for value, group in family_work.groupby(family_id, sort=True, observed=True):
        per_family[str(value)] = _rate_gate(
            int(group["reject"].sum()), 500, threshold=thresholds["family"], operator="less_than_or_equal"
        )
    _require_equal(len(per_family), 13, "per-family rate count")
    maximum_family = max(float(item["estimate"]) for item in per_family.values())
    false_open_column = _column(
        refusals, ("false_open_nonestimable", "false_open"), "false-open indicator"
    )
    false_open = refusals[false_open_column].to_numpy()
    _require(np.isin(false_open, [0, 1, False, True]).all(), "False-open indicator must be binary")
    n_false_open = int(np.asarray(false_open, dtype=bool).sum())
    all_numeric = all(
        item["pass"]
        for item in (global_gate, family_gate, integrated_gate, by_gate, band_gate)
    )
    refusal_override = n_refused == 0 and n_false_open == 0
    overall = bool(all_numeric and refusal_override)
    p_distributions = {
        "raw_pathway_curve_p_values": _pvalue_distribution(pathways[raw_curve_column].to_numpy(dtype=float)),
        "global_50_curve_maxT_p_values": _pvalue_distribution(pathways[global_column].to_numpy(dtype=float)),
        "raw_integrated_p_values": _pvalue_distribution(pathways[integrated_column].to_numpy(dtype=float)),
        "BY_q_values": _pvalue_distribution(pathways[by_column].to_numpy(dtype=float)),
        "family_p_values": _pvalue_distribution(families[family_p_column].to_numpy(dtype=float)),
    }
    finite_effect = np.empty(0, dtype=float)
    maximum_absolute_mean = None
    if curve_metrics is not None:
        curves = curve_metrics.copy()
        curve_replicate = _column(
            curves, ("replicate_index_0based", "replicate_index"), "curve replicate index"
        )
        curve_pathway = _column(curves, ("pathway_id", "pathway"), "curve pathway id")
        curve_bin = _column(curves, ("bin_index", "bin_id"), "curve bin index")
        curve_effect = _column(curves, ("effect", "signed_effect"), "signed curve effect")
        curve_supported = _column(curves, ("supported", "support"), "curve support flag")
        supported_values = curves[curve_supported].to_numpy()
        _require(
            np.isin(supported_values, [0, 1, False, True]).all(),
            "Curve support flag must be binary",
        )
        supported_rows = np.asarray(supported_values, dtype=bool)
        completed_curve_rows = curves[curve_replicate].astype(int).isin(completed_replicates).to_numpy()
        effect_values = curves[curve_effect].to_numpy(dtype=float)
        required_effect = supported_rows & completed_curve_rows
        _require(np.isfinite(effect_values[required_effect]).all(), "Supported completed curve effects must be finite")
        _require(
            np.isnan(effect_values[~completed_curve_rows]).all(),
            "Refused replicate curve effects must be NA",
        )
        finite_effect = effect_values[required_effect]
        signed = curves.loc[required_effect, [curve_pathway, curve_bin]].copy()
        signed["_effect"] = finite_effect
        group_means = signed.groupby(
            [curve_pathway, curve_bin], sort=True, observed=True
        )["_effect"].mean()
        maximum_absolute_mean = (
            float(np.max(np.abs(group_means.to_numpy(dtype=float))))
            if len(group_means)
            else None
        )
    point_bias = {
        "n_available_estimates": int(len(finite_effect)),
        "mean_signed_effect": float(np.mean(finite_effect)) if len(finite_effect) else None,
        "median_signed_effect": float(np.median(finite_effect)) if len(finite_effect) else None,
        "mean_absolute_effect": float(np.mean(np.abs(finite_effect))) if len(finite_effect) else None,
        "maximum_absolute_mean_signed_effect": maximum_absolute_mean,
        "numeric_gate": False,
    }
    summary = {
        "schema_name": "trajpathmix_corebench_cb2_500_acceptance_summary",
        "schema_version": "1.0.0",
        "execution_id": EXECUTION_ID,
        "config_payload_sha256": None if config is None else config.get("_config_payload_sha256", config.get("frozen_payload_sha256")),
        "n_planned_replicates": 500,
        "n_completed_replicates": n_completed,
        "n_refused_replicates": n_refused,
        "n_false_open_replicates": n_false_open,
        "global_50_curve_maxT_fwer": global_gate,
        "family_13_macro_fwer": family_gate,
        "per_family_rejection_rates": per_family,
        "maximum_per_family_rejection_rate": maximum_family,
        "pooled_integrated_effect_type_i_error": integrated_gate,
        "complete_null_BY_fdr": by_gate,
        "global_zero_curve_simultaneous_band_coverage": band_gate,
        "point_estimate_bias": point_bias,
        "p_value_distribution": p_distributions,
        "refusal_summary": {
            "n_completed": n_completed,
            "n_refused": n_refused,
            "n_false_open": n_false_open,
            "refusal_rate": _rate_gate(n_refused, 500, threshold=0.0, operator="less_than_or_equal"),
        },
        "mapping_summary": {
            "mappings_per_replicate": 999,
            "same_stream_all_endpoints": True,
        },
        "all_numeric_gates_pass": bool(all_numeric),
        "refusal_override_pass": bool(refusal_override),
        "overall_pass": overall,
        "cb2_500_pass": overall,
        "cb2_2000_technical_gate_satisfied": overall,
        "cb2_2000_start_allowed": False,
        "claim_scope": {
            "statistical_benchmark_only": True,
            "complete_null_weak_fwer_calibration_only": True,
            "biological_discovery": False,
            "real_endoderm_condition_contrast": False,
            "pathway_recovery_claimed": False,
            "strong_fwer_under_partial_null_claimed": False,
            "partial_null_fdr_claimed": False,
            "nonzero_or_alternative_curve_coverage_claimed": False,
            "next_stage_authorized": "none",
            "cb2_2000_requires_separate_authorization": True,
            "cb3_requires_separate_authorization": True,
            "cb2_2000_separate_explicit_user_authorization_required": True,
            "cb3_separate_explicit_user_authorization_required": True,
        },
    }
    _assert_json_finite(summary)
    return summary


def _execute_cached_replicate(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Worker entry point; consumes only explicit cache-derived arrays."""

    try:
        from threadpoolctl import threadpool_limits

        limiter = threadpool_limits(limits=1)
    except ImportError:  # pragma: no cover - dependency is present in formal env
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
            alpha=payload["alpha"],
            chunk_size=payload["chunk_size"],
            mapping_seed=payload["mapping_seed"],
        )
    replicate_index = int(payload["replicate_index_0based"])
    assignment_id = str(payload["assignment_id"])
    family_index = np.asarray(payload["family_index"], dtype=int)
    family_ids = tuple(payload["family_ids"])
    level_one_ids = [
        None if code < 0 else family_ids[int(code)] for code in family_index
    ]
    band_covered = bool(
        np.all(result.simultaneous_lower[result.support_mask] <= 0.0)
        and np.all(result.simultaneous_upper[result.support_mask] >= 0.0)
    )
    curve_rows: list[dict[str, Any]] = []
    pathway_rows: list[dict[str, Any]] = []
    for pathway_position, pathway_id in enumerate(result.pathway_ids):
        family_id = level_one_ids[pathway_position]
        for bin_index in range(result.effect.shape[0]):
            supported = bool(result.support_mask[bin_index, pathway_position])
            lower = result.simultaneous_lower[bin_index, pathway_position]
            upper = result.simultaneous_upper[bin_index, pathway_position]
            curve_rows.append(
                {
                    "replicate_index_0based": replicate_index,
                    "assignment_id": assignment_id,
                    "pathway_id": pathway_id,
                    "level_1_family_id": family_id,
                    "bin_index": bin_index,
                    "supported": supported,
                    "effect": result.effect[bin_index, pathway_position],
                    "standard_error": result.standard_error[bin_index, pathway_position],
                    "pointwise_lower": result.pointwise_lower[bin_index, pathway_position],
                    "pointwise_upper": result.pointwise_upper[bin_index, pathway_position],
                    "simultaneous_lower": lower,
                    "simultaneous_upper": upper,
                    "simultaneous_contains_zero": bool(
                        supported and lower <= 0.0 and upper >= 0.0
                    ),
                }
            )
        pathway_rows.append(
            {
                "replicate_index_0based": replicate_index,
                "assignment_id": assignment_id,
                "pathway_id": pathway_id,
                "level_1_family_id": family_id,
                "functional_estimable": True,
                "curve_statistic": result.curve_statistic[pathway_position],
                "raw_curve_p_value": result.raw_curve_p_value[pathway_position],
                "global_50_curve_maxT_p_value": result.global_50_curve_maxT_p_value[pathway_position],
                "global_50_curve_reject_alpha": bool(
                    result.global_50_curve_maxT_p_value[pathway_position]
                    <= payload["alpha"]
                ),
                "integrated_effect": result.integrated_effect[pathway_position],
                "raw_integrated_p_value": result.raw_integrated_p_value[pathway_position],
                "integrated_reject_alpha": bool(
                    result.raw_integrated_p_value[pathway_position]
                    <= payload["alpha"]
                ),
                "BY_q_value": result.BY_q_value[pathway_position],
                "BY_reject_alpha": bool(
                    result.BY_q_value[pathway_position] <= payload["alpha"]
                ),
            }
        )
    family_rows = [
        {
            "replicate_index_0based": replicate_index,
            "assignment_id": assignment_id,
            "family_id": family_ids[position],
            "n_member_pathways": int(np.sum(family_index == code)),
            "observed_family_max_statistic": result.observed_family_max_statistic[position],
            "family_maxT_p_value": result.family_maxT_p_value[position],
            "family_reject_alpha": bool(
                result.family_maxT_p_value[position] <= payload["alpha"]
            ),
        }
        for position, code in enumerate(result.family_codes)
    ]
    audit = dict(result.mapping_audit)
    return {
        "replicate_index_0based": replicate_index,
        "manifest": {
            "replicate_index_0based": replicate_index,
            "assignment_id": assignment_id,
            "assignment_sha256": payload["assignment_sha256"],
            "canonical_replicate_index_text": str(replicate_index),
            "residual_mapping_seed_uint64": payload["mapping_seed"],
            "mapping_stream_sha256": audit["mapping_stream_sha256"],
            "estimability_status": "estimable",
            "replicate_status": "completed",
            "refusal_reason_code": None,
            "false_open_nonestimable": False,
            "all_50_pathways_complete": True,
        },
        "curve": curve_rows,
        "pathway": pathway_rows,
        "family": family_rows,
        "mapping": {
            "replicate_index_0based": replicate_index,
            "assignment_id": assignment_id,
            "residual_mapping_seed_uint64": payload["mapping_seed"],
            "mapping_stream_sha256": audit["mapping_stream_sha256"],
            "n_mappings_requested": 999,
            "n_unique_mapping_hashes": audit["n_unique_mapping_hashes"],
            "identity_mapping_present": audit["identity_mapping_present"],
            "n_unique_availability_signatures": audit["n_unique_availability_signatures"],
            "n_mobile_donors": audit["n_mobile_donors"],
            "n_immobile_donors": audit["n_immobile_donors"],
            "orbit_size": audit["orbit_size"],
            "n_unique_nonidentity_mappings_possible": audit[
                "n_unique_nonidentity_mappings_possible"
            ],
            "attainable_exact_p_resolution": audit["attainable_exact_p_resolution"],
            "sampled_p_resolution": audit["sampled_p_resolution"],
            "same_stream_all_endpoints": True,
        },
        "refusal": {
            "replicate_index_0based": replicate_index,
            "assignment_id": assignment_id,
            "attempted": True,
            "replicate_status": "completed",
            "refusal_stage": None,
            "refusal_reason_code": None,
            "refusal_detail": None,
            "any_design_gate_false": False,
            "formal_result_returned": True,
            "false_open_nonestimable": False,
            "included_in_fixed_denominators": True,
        },
        "band_covered": band_covered,
    }


def _refused_replicate_rows(
    payload: Mapping[str, Any], error: BaseException
) -> dict[str, Any]:
    replicate_index = int(payload["replicate_index_0based"])
    assignment_id = str(payload["assignment_id"])
    reason = "design_not_estimable" if isinstance(error, CB2500DesignError) else "replicate_execution_error"
    family_index = np.asarray(payload["family_index"], dtype=int)
    family_ids = tuple(payload["family_ids"])
    pathways = tuple(payload["pathway_ids"])
    curve_rows = []
    pathway_rows = []
    for pathway_position, pathway_id in enumerate(pathways):
        code = int(family_index[pathway_position])
        family_id = None if code < 0 else family_ids[code]
        for bin_index in range(20):
            curve_rows.append(
                {
                    "replicate_index_0based": replicate_index,
                    "assignment_id": assignment_id,
                    "pathway_id": pathway_id,
                    "level_1_family_id": family_id,
                    "bin_index": bin_index,
                    "supported": True,
                    "effect": None,
                    "standard_error": None,
                    "pointwise_lower": None,
                    "pointwise_upper": None,
                    "simultaneous_lower": None,
                    "simultaneous_upper": None,
                    "simultaneous_contains_zero": False,
                }
            )
        pathway_rows.append(
            {
                "replicate_index_0based": replicate_index,
                "assignment_id": assignment_id,
                "pathway_id": pathway_id,
                "level_1_family_id": family_id,
                "functional_estimable": False,
                "curve_statistic": None,
                "raw_curve_p_value": None,
                "global_50_curve_maxT_p_value": None,
                "global_50_curve_reject_alpha": False,
                "integrated_effect": None,
                "raw_integrated_p_value": None,
                "integrated_reject_alpha": False,
                "BY_q_value": None,
                "BY_reject_alpha": False,
            }
        )
    family_rows = [
        {
            "replicate_index_0based": replicate_index,
            "assignment_id": assignment_id,
            "family_id": family_id,
            "n_member_pathways": int(np.sum(family_index == index)),
            "observed_family_max_statistic": None,
            "family_maxT_p_value": None,
            "family_reject_alpha": False,
        }
        for index, family_id in enumerate(family_ids)
    ]
    audit = _mapping_audit(
        payload["donor_ids"], payload["availability"], payload["mappings"], payload["mapping_seed"]
    )
    mapping_row = {
        "replicate_index_0based": replicate_index,
        "assignment_id": assignment_id,
        "residual_mapping_seed_uint64": payload["mapping_seed"],
        "mapping_stream_sha256": audit["mapping_stream_sha256"],
        "n_mappings_requested": 999,
        "n_unique_mapping_hashes": audit["n_unique_mapping_hashes"],
        "identity_mapping_present": audit["identity_mapping_present"],
        "n_unique_availability_signatures": audit["n_unique_availability_signatures"],
        "n_mobile_donors": audit["n_mobile_donors"],
        "n_immobile_donors": audit["n_immobile_donors"],
        "orbit_size": audit["orbit_size"],
        "n_unique_nonidentity_mappings_possible": audit["n_unique_nonidentity_mappings_possible"],
        "attainable_exact_p_resolution": audit["attainable_exact_p_resolution"],
        "sampled_p_resolution": audit["sampled_p_resolution"],
        "same_stream_all_endpoints": True,
    }
    return {
        "replicate_index_0based": replicate_index,
        "manifest": {
            "replicate_index_0based": replicate_index,
            "assignment_id": assignment_id,
            "assignment_sha256": payload["assignment_sha256"],
            "canonical_replicate_index_text": str(replicate_index),
            "residual_mapping_seed_uint64": payload["mapping_seed"],
            "mapping_stream_sha256": audit["mapping_stream_sha256"],
            "estimability_status": "not_estimable" if isinstance(error, CB2500DesignError) else "unknown",
            "replicate_status": "refused",
            "refusal_reason_code": reason,
            "false_open_nonestimable": False,
            "all_50_pathways_complete": False,
        },
        "curve": curve_rows,
        "pathway": pathway_rows,
        "family": family_rows,
        "mapping": mapping_row,
        "refusal": {
            "replicate_index_0based": replicate_index,
            "assignment_id": assignment_id,
            "attempted": True,
            "replicate_status": "refused",
            "refusal_stage": "functional_inference",
            "refusal_reason_code": reason,
            "refusal_detail": f"{type(error).__name__}: {error}",
            "any_design_gate_false": isinstance(error, CB2500DesignError),
            "formal_result_returned": False,
            "false_open_nonestimable": False,
            "included_in_fixed_denominators": True,
        },
        "band_covered": False,
    }


_REPLICATE_COLUMNS = (
    "replicate_index_0based", "assignment_id", "assignment_sha256",
    "canonical_replicate_index_text", "residual_mapping_seed_uint64",
    "mapping_stream_sha256", "estimability_status", "replicate_status",
    "refusal_reason_code", "false_open_nonestimable", "all_50_pathways_complete",
)
_CURVE_COLUMNS = (
    "replicate_index_0based", "assignment_id", "pathway_id", "level_1_family_id",
    "bin_index", "supported", "effect", "standard_error", "pointwise_lower",
    "pointwise_upper", "simultaneous_lower", "simultaneous_upper",
    "simultaneous_contains_zero",
)
_PATHWAY_COLUMNS = (
    "replicate_index_0based", "assignment_id", "pathway_id", "level_1_family_id",
    "functional_estimable", "curve_statistic", "raw_curve_p_value",
    "global_50_curve_maxT_p_value", "global_50_curve_reject_alpha",
    "integrated_effect", "raw_integrated_p_value", "integrated_reject_alpha",
    "BY_q_value", "BY_reject_alpha",
)
_FAMILY_COLUMNS = (
    "replicate_index_0based", "assignment_id", "family_id", "n_member_pathways",
    "observed_family_max_statistic", "family_maxT_p_value", "family_reject_alpha",
)
_MAPPING_COLUMNS = (
    "replicate_index_0based", "assignment_id", "residual_mapping_seed_uint64",
    "mapping_stream_sha256", "n_mappings_requested", "n_unique_mapping_hashes",
    "identity_mapping_present", "n_unique_availability_signatures", "n_mobile_donors",
    "n_immobile_donors", "orbit_size", "n_unique_nonidentity_mappings_possible",
    "attainable_exact_p_resolution", "sampled_p_resolution", "same_stream_all_endpoints",
)
_REFUSAL_COLUMNS = (
    "replicate_index_0based", "assignment_id", "attempted", "replicate_status",
    "refusal_stage", "refusal_reason_code", "refusal_detail", "any_design_gate_false",
    "formal_result_returned", "false_open_nonestimable", "included_in_fixed_denominators",
)


def _merge_tsv_parts(parts: Sequence[Path], destination: Path) -> None:
    with destination.open("xb") as output:
        for index, part in enumerate(parts):
            with part.open("rb") as source:
                if index:
                    source.readline()
                shutil.copyfileobj(source, output, length=1024 * 1024)


def _pvalue_diagnostic_table(
    pathway_metrics: pd.DataFrame, family_metrics: pd.DataFrame
) -> pd.DataFrame:
    sources = {
        "raw_pathway_curve_p_values": pathway_metrics["raw_curve_p_value"],
        "global_50_curve_maxT_p_values": pathway_metrics["global_50_curve_maxT_p_value"],
        "raw_integrated_p_values": pathway_metrics["raw_integrated_p_value"],
        "BY_q_values": pathway_metrics["BY_q_value"],
        "family_p_values": family_metrics["family_maxT_p_value"],
    }
    rows: list[dict[str, Any]] = []
    for family, series in sources.items():
        values = series.to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        diagnostics = _pvalue_distribution(values)
        scalar_diagnostics = {
            "minimum": diagnostics["minimum"],
            "quartile_25": diagnostics["quartiles"][0] if diagnostics["quartiles"] else None,
            "median": diagnostics["median"],
            "quartile_75": diagnostics["quartiles"][2] if diagnostics["quartiles"] else None,
            "mean": diagnostics["mean"],
            "maximum": diagnostics["maximum"],
        }
        for name, diagnostic_value in scalar_diagnostics.items():
            rows.append(
                {
                    "p_value_family": family,
                    "diagnostic": name,
                    "bin_lower_inclusive": None,
                    "bin_upper_exclusive": None,
                    "count": len(values),
                    "denominator": len(values),
                    "value": diagnostic_value,
                }
            )
        histogram, edges = np.histogram(values, bins=np.linspace(0.0, 1.0, 11))
        for bin_index, count in enumerate(histogram):
            rows.append(
                {
                    "p_value_family": family,
                    "diagnostic": f"decile_{bin_index}",
                    "bin_lower_inclusive": edges[bin_index],
                    "bin_upper_exclusive": edges[bin_index + 1],
                    "count": int(count),
                    "denominator": len(values),
                    "value": float(count / len(values)) if len(values) else None,
                }
            )
    return pd.DataFrame(
        rows,
        columns=(
            "p_value_family", "diagnostic", "bin_lower_inclusive",
            "bin_upper_exclusive", "count", "denominator", "value",
        ),
    )


def _validate_tsv_lexical_contract(
    path: Path, boolean_columns: Sequence[str] = ()
) -> None:
    raw_bytes = path.read_bytes()
    _require(b"\r" not in raw_bytes, f"TSV must use LF-only line endings: {path.name}")
    raw = pd.read_csv(
        io.BytesIO(raw_bytes),
        sep="\t",
        dtype="string",
        keep_default_na=False,
        na_filter=False,
    )
    _require(not raw.eq("").any(axis=None), f"TSV missing values must be literal NA: {path.name}")
    missing_like = {"na", "nan", "n/a", "null", "none"}
    for token in raw.to_numpy(dtype=str).ravel():
        if token.strip().lower() in missing_like:
            _require(token == "NA", f"TSV missing values must be literal NA: {path.name}")
    for column in boolean_columns:
        _require(column in raw.columns, f"Missing boolean column {column} in {path.name}")
        tokens = set(raw[column].astype(str))
        _require(
            tokens.issubset({"true", "false"}),
            f"Boolean column {column} must use only lowercase true/false",
        )


def run_cb2_500(
    config_path: str | Path = DEFAULT_CONFIG_FILE,
    repository_root: str | Path = ".",
    cache_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    explicit_execution_authorization: bool = False,
    processes: int = DEFAULT_PROCESSES,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, Any]:
    """Execute the frozen 500-assignment benchmark once from cache only."""

    root = Path(repository_root).resolve()
    # An explicit create-only collision is rejected before config/cache reads.
    explicit_target = Path(output_dir).resolve() if output_dir is not None else None
    if explicit_target is not None and explicit_target.exists():
        raise FileExistsError(f"Create-only CB2-500 output already exists: {explicit_target}")
    resolved_config = (
        _repo_file(root, str(config_path)) if not Path(config_path).is_absolute() else Path(config_path)
    )
    config = load_cb2_500_execution_config(resolved_config, require_frozen=True)
    _require_execution_authorization(config, explicit_execution_authorization)
    target = (
        _repo_file(root, config["output_contract"]["default_output_dir"])
        if explicit_target is None
        else explicit_target
    )
    if target.exists():
        raise FileExistsError(f"Create-only CB2-500 output already exists: {target}")
    incomplete = target.parent / f"{target.name}.incomplete"
    if incomplete.exists():
        raise FileExistsError(
            f"Prior incomplete evidence exists; automatic resume/retry is forbidden: {incomplete}"
        )
    worker_count = int(processes)
    mapping_chunk = int(chunk_size)
    _require(worker_count > 0, "processes must be positive")
    _require(mapping_chunk > 0, "chunk_size must be positive")
    cache_validation = validate_cb2_500_cache(config_path, root, cache_dir)
    cache = Path(cache_validation["cache_dir"])
    incomplete.mkdir(parents=True, exist_ok=False)
    scratch = incomplete / ".replicate_parts"
    scratch.mkdir()
    # From this point onward exceptions intentionally preserve .incomplete.
    donor_axis = pd.read_csv(cache / "donor_axis_v1.tsv", sep="\t", dtype="string")
    pathway_axis = pd.read_csv(cache / "pathway_axis_v1.tsv", sep="\t", dtype="string")
    assignment_axis = pd.read_csv(cache / "assignment_axis_v1.tsv", sep="\t", dtype="string")
    experiment_axis = pd.read_csv(cache / "experiment_axis_v1.tsv", sep="\t", dtype="string")
    family_axis = pd.read_csv(cache / "family_axis_v1.tsv", sep="\t", dtype="string")
    family_table = pd.read_csv(cache / CACHE_FILES[6], sep="\t", dtype={"pathway_id": "string"})
    donors = tuple(donor_axis["donor_id"].astype(str))
    pathways = tuple(pathway_axis["pathway_id"].astype(str))
    experiments = tuple(experiment_axis["experiment_id"].astype(str))
    family_ids = tuple(family_axis["family_id"].astype(str))
    family_index = family_table["level_1_family_index"].to_numpy(dtype=int)
    outcomes = np.load(cache / CACHE_FILES[2], mmap_mode="r", allow_pickle=False)
    mapping_bank = np.load(cache / CACHE_FILES[5], mmap_mode="r", allow_pickle=False)
    support_mask = np.load(cache / CACHE_FILES[7], mmap_mode="r", allow_pickle=False)
    with np.load(cache / CACHE_FILES[3], allow_pickle=False) as archive:
        design_arrays = {name: archive[name] for name in archive.files}
    assignments = np.asarray(design_arrays["frozen_assignments_uint8"], dtype=np.uint8)
    availability = np.asarray(design_arrays["availability_bool"], dtype=bool)
    fractions = np.asarray(design_arrays["experiment_fractions_float64"], dtype=float)
    alpha = float(config["functional_inference_contract"]["alpha"])

    def payload_for(index: int) -> dict[str, Any]:
        condition = assignments[index]
        return {
            "replicate_index_0based": index,
            "assignment_id": str(assignment_axis.iloc[index]["assignment_id"]),
            "assignment_sha256": str(assignment_axis.iloc[index]["assignment_sha256"]),
            "mapping_seed": derive_residual_mapping_seed(index),
            "outcomes": np.asarray(outcomes),
            "donor_ids": donors,
            "condition": condition,
            "availability": availability,
            "experiment_fractions": fractions,
            "experiment_ids": experiments,
            "pathway_ids": pathways,
            "mappings": np.asarray(mapping_bank[index], dtype=np.int32),
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
            "family_ids": family_ids,
            "support_mask": np.asarray(support_mask, dtype=bool),
            "alpha": alpha,
            "chunk_size": mapping_chunk,
        }

    manifests: dict[int, Mapping[str, Any]] = {}
    mappings_out: dict[int, Mapping[str, Any]] = {}
    refusals: dict[int, Mapping[str, Any]] = {}
    band_coverage: dict[int, bool] = {}

    def record(result: Mapping[str, Any]) -> None:
        index = int(result["replicate_index_0based"])
        manifests[index] = result["manifest"]
        mappings_out[index] = result["mapping"]
        refusals[index] = result["refusal"]
        band_coverage[index] = bool(result["band_covered"])
        for key, columns in (
            ("curve", _CURVE_COLUMNS),
            ("pathway", _PATHWAY_COLUMNS),
            ("family", _FAMILY_COLUMNS),
        ):
            _write_table(
                pd.DataFrame(result[key], columns=columns),
                scratch / f"{key}_{index:03d}.tsv",
            )

    if worker_count == 1:
        for index in range(500):
            payload = payload_for(index)
            try:
                record(_execute_cached_replicate(payload))
            except Exception as exc:
                record(_refused_replicate_rows(payload, exc))
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            next_index = 0
            pending: dict[Any, tuple[int, Mapping[str, Any]]] = {}
            while next_index < 500 and len(pending) < worker_count * 2:
                payload = payload_for(next_index)
                pending[executor.submit(_execute_cached_replicate, payload)] = (next_index, payload)
                next_index += 1
            while pending:
                completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in completed:
                    index, payload = pending.pop(future)
                    try:
                        record(future.result())
                    except Exception as exc:
                        record(_refused_replicate_rows(payload, exc))
                    if next_index < 500:
                        next_payload = payload_for(next_index)
                        pending[executor.submit(_execute_cached_replicate, next_payload)] = (
                            next_index,
                            next_payload,
                        )
                        next_index += 1
    _require_equal(set(manifests), set(range(500)), "executed replicate coverage")
    for key, destination in (
        ("curve", CURVE_FILE),
        ("pathway", PATHWAY_FILE),
        ("family", FAMILY_FILE),
    ):
        _merge_tsv_parts(
            [scratch / f"{key}_{index:03d}.tsv" for index in range(500)],
            incomplete / destination,
        )
    shutil.rmtree(scratch)
    _write_table(
        pd.DataFrame([manifests[index] for index in range(500)], columns=_REPLICATE_COLUMNS),
        incomplete / REPLICATE_FILE,
    )
    _write_table(
        pd.DataFrame([mappings_out[index] for index in range(500)], columns=_MAPPING_COLUMNS),
        incomplete / MAPPING_OUTPUT_FILE,
    )
    refusal_frame = pd.DataFrame(
        [refusals[index] for index in range(500)], columns=_REFUSAL_COLUMNS
    )
    _write_table(refusal_frame, incomplete / REFUSAL_FILE)
    overlap = _strict_json_load(cache / CACHE_OVERLAP_AUDIT_FILE)
    _write_json(overlap, incomplete / OVERLAP_AUDIT_FILE)
    design_arrays.clear()
    try:
        del payload
    except UnboundLocalError:
        pass
    try:
        del next_payload
    except UnboundLocalError:
        pass
    del assignments, outcomes, mapping_bank, support_mask
    gc.collect()
    pathway_frame = pd.read_csv(incomplete / PATHWAY_FILE, sep="\t")
    family_frame = pd.read_csv(incomplete / FAMILY_FILE, sep="\t")
    curve_frame = pd.read_csv(incomplete / CURVE_FILE, sep="\t")
    replicate_internal = pd.DataFrame(
        {
            "replicate_index_0based": range(500),
            "global_band_coverage_indicator": [band_coverage[index] for index in range(500)],
        }
    )
    summary = aggregate_cb2_500_acceptance(
        pathway_metrics=pathway_frame,
        family_metrics=family_frame,
        replicate_metrics=replicate_internal,
        refusal_audit=refusal_frame,
        curve_metrics=curve_frame,
        alpha=alpha,
        config=config,
    )
    _write_table(
        _pvalue_diagnostic_table(pathway_frame, family_frame),
        incomplete / PVALUE_DIAGNOSTICS_FILE,
    )
    _write_json(summary, incomplete / SUMMARY_FILE)
    blocking = []
    for name in (
        "global_50_curve_maxT_fwer", "family_13_macro_fwer",
        "pooled_integrated_effect_type_i_error", "complete_null_BY_fdr",
        "global_zero_curve_simultaneous_band_coverage",
    ):
        if not summary[name]["pass"]:
            blocking.append(f"numeric_gate_failed:{name}")
    if summary["n_false_open_replicates"]:
        blocking.append("false_open_nonestimable")
    if summary["n_refused_replicates"]:
        blocking.append("refused_or_incomplete_replicate")
    decision = {
        "schema_name": "trajpathmix_corebench_cb2_500_acceptance_decision",
        "schema_version": "1.0.0",
        "execution_id": EXECUTION_ID,
        "acceptance_summary_sha256": _hash_file(incomplete / SUMMARY_FILE),
        "all_required_artifact_hashes": True,
        "all_five_numeric_gates_pass": summary["all_numeric_gates_pass"],
        "zero_false_open_replicates": summary["n_false_open_replicates"] == 0,
        "zero_other_refused_or_incomplete_replicates": summary["n_refused_replicates"] == 0,
        "cb2_500_pass": summary["cb2_500_pass"],
        "cb2_2000_technical_gate_satisfied": summary["cb2_500_pass"],
        "cb2_2000_start_allowed": False,
        "decision": (
            "pass_cb2_500_technical_gate_pending_separate_cb2_2000_authorization"
            if summary["cb2_500_pass"]
            else "fail_cb2_500_stop_before_cb2_2000"
        ),
        "blocking_reason_codes": blocking,
        "claim_ceiling": summary["claim_scope"],
        "timing_computed": False,
        "timing_fields_present": False,
        "real_condition_contrast_generated": False,
    }
    _write_json(decision, incomplete / DECISION_FILE)
    passport = {
        "schema_name": "trajpathmix_corebench_cb2_500_material_passport",
        "schema_version": "1.0.0",
        "execution_id": EXECUTION_ID,
        "execution_config_payload_sha256": config["_config_payload_sha256"],
        "cache_manifest_sha256": cache_validation["cache_manifest_sha256"],
        "acceptance_decision_sha256": _hash_file(incomplete / DECISION_FILE),
        "statistical_benchmark_only": True,
        "biological_interpretation": False,
        "real_condition_contrast_generated": False,
        "assignment_generator_imported_or_called": False,
        "next_stage_authorized": "none",
        "data_provenance": {
            "cache_manifest_sha256": cache_validation["cache_manifest_sha256"],
            "all_cache_items_hash_validated": True,
            "frozen_assignment_bank_only": True,
        },
        "code_provenance": {
            "execution_config_payload_sha256": config["_config_payload_sha256"],
            "module_binding_sha256": config["bindings"]["cb2_500_module"]["sha256"],
            "script_binding_sha256": config["bindings"]["cb2_500_script"]["sha256"],
            "test_binding_sha256": config["bindings"]["cb2_500_test"]["sha256"],
        },
        "fallacy_scan": {
            "coverage": "11_of_11",
            "simpsons_paradox": {
                "status": "pass",
                "detail": "experiment fractions are included bin-specifically and the donor remains the independence unit",
            },
            "ecological_fallacy": {
                "status": "pass",
                "detail": "claims are limited to complete-null statistical calibration, not individual biology",
            },
            "berksons_paradox": {
                "status": "pass",
                "detail": "the primary donor cohort and all assignment denominators were frozen before scoring",
            },
            "collider_bias": {
                "status": "pass",
                "detail": "no outcome-dependent donor, bin, pathway, or family filtering is permitted",
            },
            "base_rate_neglect": {
                "status": "pass",
                "detail": "complete-null rates use the planned 500, 25000, and 6500 denominators",
            },
            "regression_to_mean": {
                "status": "pass",
                "detail": "all frozen assignments are evaluated once without selection or rerun",
            },
            "survivorship_bias": {
                "status": "pass",
                "detail": "refused replicates remain in fixed denominators and override an overall pass",
            },
            "look_elsewhere_effect": {
                "status": "pass",
                "detail": "global-50 maxT, family maxT, and BY multiplicity endpoints are predeclared",
            },
            "garden_of_forking_paths": {
                "status": "pass",
                "detail": "assignments, pathways, families, seeds, mappings, endpoints, and thresholds are frozen",
            },
            "correlation_not_causation": {
                "status": "pass",
                "detail": "the claim ceiling excludes causal and biological interpretation",
            },
            "reverse_causality": {
                "status": "pass",
                "detail": "the benchmark uses pseudo-condition assignments and makes no direction-of-cause claim",
            },
        },
        "safeguards": {
            "outcome_blind_frozen_assignments": True,
            "fixed_denominators": True,
            "no_result_based_selection": True,
            "donor_independence_unit_preserved": True,
            "frozen_pathway_universe": True,
            "complete_null_claim_ceiling": True,
            "no_biological_or_causal_inference": True,
            "no_real_condition_contrast": True,
            "no_automatic_next_stage_authorization": True,
            "refusal_override_fail_closed": True,
            "source_and_artifact_hashes_required": True,
        },
        "verification_status": "self_validated_before_atomic_publish",
        "timing_computed": False,
        "timing_fields_present": False,
    }
    _write_json(passport, incomplete / PASSPORT_FILE)
    artifact_hashes = {
        name: _hash_file(incomplete / name)
        for name in OUTPUT_FILES
        if name != RUN_BUILD_RECORD_FILE
    }
    build_record = {
        "schema_name": "trajpathmix_corebench_cb2_500_build_record",
        "schema_version": "1.0.0",
        "execution_id": EXECUTION_ID,
        "execution_config_payload_sha256": config["_config_payload_sha256"],
        "cache_manifest_sha256": cache_validation["cache_manifest_sha256"],
        "artifact_sha256": artifact_hashes,
        "replicates_attempted_once": 500,
        "automatic_resume_used": False,
        "automatic_retry_used": False,
        "worker_processes": worker_count,
        "mapping_chunk_size": mapping_chunk,
        "assignment_generator_imported_or_called": False,
        "real_condition_contrast_generated": False,
        "timing_computed": False,
        "timing_fields_present": False,
    }
    _write_json(build_record, incomplete / RUN_BUILD_RECORD_FILE)
    validate_cb2_500_output(config_path, root, cache, incomplete)
    os.replace(incomplete, target)
    return validate_cb2_500_output(config_path, root, cache, target)


def validate_cb2_500_output(
    config_path: str | Path = DEFAULT_CONFIG_FILE,
    repository_root: str | Path = ".",
    cache_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate fixed schemas, denominators, decisions, provenance, and hashes."""

    root = Path(repository_root).resolve()
    resolved_config = (
        _repo_file(root, str(config_path)) if not Path(config_path).is_absolute() else Path(config_path)
    )
    config = load_cb2_500_execution_config(resolved_config, require_frozen=True)
    verify_cb2_500_source_bindings(root, config)
    cache_validation = validate_cb2_500_cache(config_path, root, cache_dir)
    directory = (
        _repo_file(root, config["output_contract"]["default_output_dir"])
        if output_dir is None
        else Path(output_dir).resolve()
    )
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    observed_files = {path.name for path in directory.iterdir() if path.is_file()}
    _require_equal(observed_files, set(OUTPUT_FILES), "CB2-500 output file set")
    _require(not any(path.is_dir() for path in directory.iterdir()), "CB2-500 output contains subdirectories")
    frames: dict[str, pd.DataFrame] = {}
    table_specs = {
        REPLICATE_FILE: (_REPLICATE_COLUMNS, 500),
        CURVE_FILE: (_CURVE_COLUMNS, 500_000),
        PATHWAY_FILE: (_PATHWAY_COLUMNS, 25_000),
        FAMILY_FILE: (_FAMILY_COLUMNS, 6_500),
        MAPPING_OUTPUT_FILE: (_MAPPING_COLUMNS, 500),
        REFUSAL_FILE: (_REFUSAL_COLUMNS, 500),
    }
    boolean_columns = {
        REPLICATE_FILE: ("false_open_nonestimable", "all_50_pathways_complete"),
        CURVE_FILE: ("supported", "simultaneous_contains_zero"),
        PATHWAY_FILE: (
            "functional_estimable", "global_50_curve_reject_alpha",
            "integrated_reject_alpha", "BY_reject_alpha",
        ),
        FAMILY_FILE: ("family_reject_alpha",),
        MAPPING_OUTPUT_FILE: ("identity_mapping_present", "same_stream_all_endpoints"),
        REFUSAL_FILE: (
            "attempted", "any_design_gate_false", "formal_result_returned",
            "false_open_nonestimable", "included_in_fixed_denominators",
        ),
    }
    for name, (columns, rows) in table_specs.items():
        _assert_formal_output_firewall({column: None for column in columns}, name)
        _validate_tsv_lexical_contract(directory / name, boolean_columns[name])
        frame = pd.read_csv(directory / name, sep="\t")
        _require_equal(tuple(frame.columns), tuple(columns), f"output columns {name}")
        _require_equal(len(frame), rows, f"output row count {name}")
        frames[name] = frame
    _validate_tsv_lexical_contract(directory / PVALUE_DIAGNOSTICS_FILE)
    pvalue_diagnostics = pd.read_csv(directory / PVALUE_DIAGNOSTICS_FILE, sep="\t")
    _assert_formal_output_firewall(
        {column: None for column in pvalue_diagnostics.columns},
        PVALUE_DIAGNOSTICS_FILE,
    )
    _require_equal(
        tuple(pvalue_diagnostics.columns),
        (
            "p_value_family", "diagnostic", "bin_lower_inclusive",
            "bin_upper_exclusive", "count", "denominator", "value",
        ),
        "p-value diagnostic columns",
    )
    _require(
        not pvalue_diagnostics.duplicated(["p_value_family", "diagnostic"]).any(),
        "Duplicate p-value diagnostics",
    )
    replicate = frames[REPLICATE_FILE]
    curve = frames[CURVE_FILE]
    pathway = frames[PATHWAY_FILE]
    family = frames[FAMILY_FILE]
    mapping = frames[MAPPING_OUTPUT_FILE]
    refusal = frames[REFUSAL_FILE]
    expected_pvalue_diagnostics = _pvalue_diagnostic_table(pathway, family)
    try:
        pd.testing.assert_frame_equal(
            pvalue_diagnostics.reset_index(drop=True),
            expected_pvalue_diagnostics.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1.0e-14,
            atol=1.0e-15,
        )
    except AssertionError as exc:
        raise CB2500ContractError("P-value diagnostics do not match metric tables") from exc
    expected_indices = np.arange(500, dtype=int)
    for frame, label in (
        (replicate, "replicate manifest"),
        (mapping, "mapping audit"),
        (refusal, "refusal audit"),
    ):
        _require(
            np.array_equal(frame["replicate_index_0based"].to_numpy(dtype=int), expected_indices),
            f"{label} order/coverage differs",
        )
    _require(not curve.duplicated(["replicate_index_0based", "pathway_id", "bin_index"]).any(), "Duplicate curve key")
    _require(not pathway.duplicated(["replicate_index_0based", "pathway_id"]).any(), "Duplicate pathway key")
    _require(not family.duplicated(["replicate_index_0based", "family_id"]).any(), "Duplicate family key")
    _require_equal(
        set(curve["replicate_index_0based"].astype(int)), set(range(500)), "curve replicate coverage"
    )
    _require_equal(
        set(pathway["replicate_index_0based"].astype(int)), set(range(500)), "pathway replicate coverage"
    )
    _require_equal(
        set(family["replicate_index_0based"].astype(int)), set(range(500)), "family replicate coverage"
    )
    _require(
        np.all(pathway.groupby("replicate_index_0based")["pathway_id"].nunique() == 50),
        "Each replicate must contain all 50 pathways",
    )
    _require(
        np.all(family.groupby("replicate_index_0based")["family_id"].nunique() == 13),
        "Each replicate must contain all 13 families",
    )
    _require(
        np.all(curve.groupby(["replicate_index_0based", "pathway_id"]).size() == 20),
        "Each replicate/pathway must contain 20 curve bins",
    )
    _require(
        np.all(mapping["n_mappings_requested"].to_numpy(dtype=int) == 999),
        "Mapping request count differs",
    )
    _require(
        np.all(mapping["n_unique_mapping_hashes"].to_numpy(dtype=int) == 999),
        "Mapping uniqueness differs",
    )
    _require(
        not mapping["identity_mapping_present"].astype(str).str.lower().eq("true").any(),
        "Identity mapping reported",
    )
    cache = Path(cache_validation["cache_dir"])
    cache_assignment_axis = pd.read_csv(
        cache / "assignment_axis_v1.tsv", sep="\t", dtype="string"
    )
    cache_mapping_audit = pd.read_csv(
        cache / CACHE_AUDIT_FILE, sep="\t", dtype="string"
    )
    cache_signatures = pd.read_csv(
        cache / CACHE_FILES[4], sep="\t", dtype="string"
    )["full_availability_signature"].astype(str)
    signature_counts = cache_signatures.value_counts(sort=False).to_numpy(dtype=int)
    expected_orbit = int(math.prod(math.factorial(int(value)) for value in signature_counts))
    expected_mobile = int(np.sum(signature_counts[signature_counts > 1]))
    _require(
        mapping["same_stream_all_endpoints"].astype(str).str.lower().eq("true").all(),
        "Every mapping row must certify the same stream for all endpoints",
    )
    _require(np.all(mapping["n_unique_availability_signatures"].to_numpy(dtype=int) == len(signature_counts)), "Mapping signature count differs")
    _require(np.all(mapping["n_mobile_donors"].to_numpy(dtype=int) == expected_mobile), "Mapping mobile-donor count differs")
    _require(np.all(mapping["n_immobile_donors"].to_numpy(dtype=int) == 75 - expected_mobile), "Mapping immobile-donor count differs")
    _require(np.all(mapping["orbit_size"].to_numpy(dtype=object) == expected_orbit), "Mapping orbit size differs")
    _require(np.all(mapping["n_unique_nonidentity_mappings_possible"].to_numpy(dtype=object) == expected_orbit - 1), "Mapping nonidentity orbit differs")
    _require(
        np.allclose(
            mapping["attainable_exact_p_resolution"].to_numpy(dtype=float),
            1.0 / expected_orbit,
            rtol=0.0,
            atol=np.finfo(float).eps,
        ),
        "Mapping exact resolution differs",
    )
    _require(
        np.array_equal(
            mapping["sampled_p_resolution"].to_numpy(dtype=float),
            np.full(500, 0.001, dtype=float),
        ),
        "Mapping sampled resolution differs",
    )
    for index in range(500):
        assignment_id = str(cache_assignment_axis.iloc[index]["assignment_id"])
        assignment_hash = str(cache_assignment_axis.iloc[index]["assignment_sha256"])
        seed = int(cache_mapping_audit.iloc[index]["seed_uint64"])
        stream_hash = str(cache_mapping_audit.iloc[index]["stream_sha256"])
        _require_equal(str(replicate.iloc[index]["assignment_id"]), assignment_id, "manifest assignment id")
        _require_equal(str(replicate.iloc[index]["assignment_sha256"]), assignment_hash, "manifest assignment hash")
        _require_equal(str(replicate.iloc[index]["canonical_replicate_index_text"]), str(index), "manifest canonical index")
        _require_equal(int(replicate.iloc[index]["residual_mapping_seed_uint64"]), seed, "manifest mapping seed")
        _require_equal(str(replicate.iloc[index]["mapping_stream_sha256"]), stream_hash, "manifest mapping stream")
        _require_equal(str(mapping.iloc[index]["assignment_id"]), assignment_id, "mapping assignment id")
        _require_equal(int(mapping.iloc[index]["residual_mapping_seed_uint64"]), seed, "mapping output seed")
        _require_equal(str(mapping.iloc[index]["mapping_stream_sha256"]), stream_hash, "mapping output stream")
        _require_equal(str(refusal.iloc[index]["assignment_id"]), assignment_id, "refusal assignment id")
    manifest_false_open = replicate["false_open_nonestimable"].astype(str).str.lower().eq("true").to_numpy()
    refusal_false_open = refusal["false_open_nonestimable"].astype(str).str.lower().eq("true").to_numpy()
    any_design_false = refusal["any_design_gate_false"].astype(str).str.lower().eq("true").to_numpy()
    formal_returned = refusal["formal_result_returned"].astype(str).str.lower().eq("true").to_numpy()
    defined_false_open = any_design_false & formal_returned
    _require(np.array_equal(refusal_false_open, defined_false_open), "False-open definition mismatch")
    _require(np.array_equal(manifest_false_open, refusal_false_open), "Manifest/refusal false-open mismatch")
    _require(
        np.array_equal(
            replicate["replicate_status"].astype(str).str.lower().to_numpy(),
            refusal["replicate_status"].astype(str).str.lower().to_numpy(),
        ),
        "Manifest/refusal replicate status mismatch",
    )
    cache_support = np.load(cache / CACHE_FILES[7], allow_pickle=False)
    cache_pathway_axis = pd.read_csv(
        cache / "pathway_axis_v1.tsv", sep="\t", dtype="string"
    )
    cache_family_axis = pd.read_csv(
        cache / "family_axis_v1.tsv", sep="\t", dtype="string"
    )
    cache_family_index = pd.read_csv(
        cache / CACHE_FILES[6], sep="\t", dtype={"pathway_id": "string"}
    )
    frozen_pathways = tuple(cache_pathway_axis["pathway_id"].astype(str))
    frozen_families = tuple(cache_family_axis["family_id"].astype(str))
    _require_equal(len(frozen_pathways), 50, "cached pathway axis length")
    _require_equal(len(frozen_families), 13, "cached family axis length")
    expected_family_text = {
        str(row.pathway_id): (
            None
            if int(row.level_1_family_index) < 0
            else frozen_families[int(row.level_1_family_index)]
        )
        for row in cache_family_index.itertuples(index=False)
    }
    for replicate_index in range(500):
        pathway_group = pathway.loc[
            pathway["replicate_index_0based"].eq(replicate_index)
        ]
        family_group = family.loc[family["replicate_index_0based"].eq(replicate_index)]
        curve_group = curve.loc[curve["replicate_index_0based"].eq(replicate_index)]
        _require_equal(
            tuple(pathway_group["pathway_id"].astype(str)),
            frozen_pathways,
            f"pathway axis replicate {replicate_index}",
        )
        _require_equal(
            tuple(family_group["family_id"].astype(str)),
            frozen_families,
            f"family axis replicate {replicate_index}",
        )
        expected_curve_pathways = tuple(
            pathway_id for pathway_id in frozen_pathways for _ in range(20)
        )
        _require_equal(
            tuple(curve_group["pathway_id"].astype(str)),
            expected_curve_pathways,
            f"curve pathway axis replicate {replicate_index}",
        )
        _require(
            np.array_equal(
                curve_group["bin_index"].to_numpy(dtype=int),
                np.tile(np.arange(20, dtype=int), 50),
            ),
            f"curve bin axis replicate {replicate_index}",
        )
    expected_pathway_family = pathway["pathway_id"].astype(str).map(expected_family_text)
    observed_pathway_family = pathway["level_1_family_id"].where(
        pathway["level_1_family_id"].notna(), None
    )
    _require(
        all(
            (expected is None and (observed is None or pd.isna(observed)))
            or str(expected) == str(observed)
            for expected, observed in zip(
                expected_pathway_family, observed_pathway_family, strict=True
            )
        ),
        "Pathway level-1 family mapping differs from cache",
    )
    expected_curve_family = curve["pathway_id"].astype(str).map(expected_family_text)
    observed_curve_family = curve["level_1_family_id"].where(
        curve["level_1_family_id"].notna(), None
    )
    _require(
        all(
            (expected is None and (observed is None or pd.isna(observed)))
            or str(expected) == str(observed)
            for expected, observed in zip(
                expected_curve_family, observed_curve_family, strict=True
            )
        ),
        "Curve level-1 family mapping differs from cache",
    )
    pathway_positions = {
        str(value): index
        for index, value in enumerate(cache_pathway_axis["pathway_id"].astype(str))
    }
    curve_pathway_positions = curve["pathway_id"].astype(str).map(pathway_positions)
    _require(not curve_pathway_positions.isna().any(), "Curve table contains an unknown pathway")
    curve_bins = curve["bin_index"].to_numpy(dtype=int)
    _require(np.all((curve_bins >= 0) & (curve_bins < 20)), "Curve bin is out of range")
    expected_support = cache_support[
        curve_bins, curve_pathway_positions.to_numpy(dtype=int)
    ]
    observed_support = curve["supported"].astype(str).str.lower().eq("true").to_numpy()
    _require(np.array_equal(observed_support, expected_support), "Curve support differs from cached mask")
    completed_ids = set(
        refusal.loc[
            refusal["replicate_status"].astype(str).str.lower().isin(
                {"complete", "completed", "success", "ok"}
            ),
            "replicate_index_0based",
        ].to_numpy(dtype=int)
    )
    completed_curve_rows = curve["replicate_index_0based"].astype(int).isin(completed_ids).to_numpy()
    lower_values = curve["simultaneous_lower"].to_numpy(dtype=float)
    upper_values = curve["simultaneous_upper"].to_numpy(dtype=float)
    required_bounds = completed_curve_rows & expected_support
    _require(
        np.isfinite(lower_values[required_bounds]).all()
        and np.isfinite(upper_values[required_bounds]).all(),
        "Completed supported simultaneous bounds must be finite",
    )
    _require(
        np.isnan(lower_values[~completed_curve_rows]).all()
        and np.isnan(upper_values[~completed_curve_rows]).all(),
        "Refused simultaneous bounds must be NA",
    )
    derived_contains = (
        expected_support
        & completed_curve_rows
        & (lower_values <= 0.0)
        & (upper_values >= 0.0)
    )
    observed_contains = (
        curve["simultaneous_contains_zero"].astype(str).str.lower().eq("true").to_numpy()
    )
    _require(
        np.array_equal(observed_contains, derived_contains),
        "simultaneous_contains_zero does not match the numeric bounds",
    )
    overlap = _strict_json_load(directory / OVERLAP_AUDIT_FILE)
    _require_equal(overlap.get("pass"), True, "output overlap audit")
    _require_equal(overlap.get("coordinate_pathway_gene_overlap_count"), 0, "output overlap count")
    summary = _strict_json_load(directory / SUMMARY_FILE)
    decision = _strict_json_load(directory / DECISION_FILE)
    passport = _strict_json_load(directory / PASSPORT_FILE)
    build_record = _strict_json_load(directory / RUN_BUILD_RECORD_FILE)
    for label, payload in (
        ("overlap_audit", overlap),
        ("acceptance_summary", summary),
        ("acceptance_decision", decision),
        ("material_passport", passport),
        ("build_record", build_record),
    ):
        _assert_formal_output_firewall(payload, label)
    summary_required = tuple(
        config["output_contract"]["json_schemas"][SUMMARY_FILE]["required_keys"]
    )
    decision_required = tuple(
        config["output_contract"]["json_schemas"][DECISION_FILE]["required_keys"]
    )
    _require_equal(set(summary), set(summary_required), "acceptance summary exact keys")
    _require_equal(set(decision), set(decision_required), "acceptance decision exact keys")
    _require_equal(set(overlap), _OVERLAP_JSON_KEYS, "overlap audit exact keys")
    _require_equal(overlap.get("execution_id"), EXECUTION_ID, "overlap execution id")
    _require_equal(overlap.get("coordinate_pathway_gene_overlap"), [], "overlap gene list")
    _require_equal(overlap.get("pathway_scoring_allowed"), True, "overlap scoring gate")
    _require_equal(overlap.get("biological_interpretation"), False, "overlap interpretation firewall")
    _require_equal(
        overlap.get("coordinate_gene_folds_sha256"),
        config["bindings"]["coordinate_gene_folds_v1"]["sha256"],
        "overlap coordinate-gene hash",
    )
    _require_equal(
        overlap.get("coordinate_gene_exclusion_audit_sha256"),
        config["bindings"]["coordinate_gene_exclusion_audit_v1"]["sha256"],
        "overlap exclusion-audit hash",
    )
    _require_equal(
        overlap.get("frozen_pathway_universe_sha256"),
        config["bindings"]["frozen_pathway_universe_v1"]["sha256"],
        "overlap pathway-universe hash",
    )
    _require_equal(set(passport), _PASSPORT_JSON_KEYS, "material passport exact keys")
    _require_equal(set(build_record), _BUILD_RECORD_JSON_KEYS, "build record exact keys")
    _require_equal(summary["config_payload_sha256"], config["_config_payload_sha256"], "summary config hash")
    _require_equal(decision["acceptance_summary_sha256"], _hash_file(directory / SUMMARY_FILE), "decision summary hash")
    _require_equal(decision["all_required_artifact_hashes"], True, "decision artifact hash gate")
    _require_equal(decision["cb2_2000_start_allowed"], False, "CB2-2000 authorization firewall")
    _require_equal(decision["timing_computed"], False, "decision timing firewall")
    _require_equal(decision["timing_fields_present"], False, "decision timing fields")
    _require_equal(decision["real_condition_contrast_generated"], False, "decision condition firewall")
    _require_equal(passport.get("next_stage_authorized"), "none", "passport next-stage authorization")
    _require_equal(passport.get("timing_computed"), False, "passport timing firewall")
    _require_equal(
        passport.get("execution_config_payload_sha256"),
        config["_config_payload_sha256"],
        "passport config hash",
    )
    _require_equal(
        passport.get("cache_manifest_sha256"),
        cache_validation["cache_manifest_sha256"],
        "passport cache hash",
    )
    _require_equal(
        passport.get("acceptance_decision_sha256"),
        _hash_file(directory / DECISION_FILE),
        "passport decision hash",
    )
    _require_equal(passport.get("verification_status"), "self_validated_before_atomic_publish", "passport verification status")
    _require_equal(
        set(passport.get("fallacy_scan", {})),
        {"coverage"} | _ARS_FALLACY_KEYS,
        "passport fallacy scan keys",
    )
    _require_equal(passport["fallacy_scan"].get("coverage"), "11_of_11", "passport fallacy coverage")
    for name in sorted(_ARS_FALLACY_KEYS):
        entry = passport["fallacy_scan"][name]
        _require_equal(set(entry), {"status", "detail"}, f"fallacy entry {name} keys")
        _require_equal(entry["status"], "pass", f"fallacy entry {name} status")
        _require(bool(str(entry["detail"]).strip()), f"fallacy entry {name} detail is blank")
    _require_equal(
        set(passport.get("safeguards", {})),
        {
            "outcome_blind_frozen_assignments", "fixed_denominators",
            "no_result_based_selection", "donor_independence_unit_preserved",
            "frozen_pathway_universe", "complete_null_claim_ceiling",
            "no_biological_or_causal_inference", "no_real_condition_contrast",
            "no_automatic_next_stage_authorization", "refusal_override_fail_closed",
            "source_and_artifact_hashes_required",
        },
        "passport safeguard keys",
    )
    _require(all(value is True for value in passport["safeguards"].values()), "Every passport safeguard must pass")
    _require_equal(
        passport.get("data_provenance"),
        {
            "cache_manifest_sha256": cache_validation["cache_manifest_sha256"],
            "all_cache_items_hash_validated": True,
            "frozen_assignment_bank_only": True,
        },
        "passport data provenance",
    )
    _require_equal(
        passport.get("code_provenance"),
        {
            "execution_config_payload_sha256": config["_config_payload_sha256"],
            "module_binding_sha256": config["bindings"]["cb2_500_module"]["sha256"],
            "script_binding_sha256": config["bindings"]["cb2_500_script"]["sha256"],
            "test_binding_sha256": config["bindings"]["cb2_500_test"]["sha256"],
        },
        "passport code provenance",
    )
    _require_equal(build_record.get("cache_manifest_sha256"), cache_validation["cache_manifest_sha256"], "run cache hash")
    _require_equal(build_record.get("execution_config_payload_sha256"), config["_config_payload_sha256"], "run config hash")
    _require_equal(build_record.get("replicates_attempted_once"), 500, "run attempt count")
    _require_equal(build_record.get("automatic_resume_used"), False, "automatic resume firewall")
    _require_equal(build_record.get("automatic_retry_used"), False, "automatic retry firewall")
    _require(int(build_record.get("worker_processes", 0)) > 0, "worker_processes must be positive")
    _require(int(build_record.get("mapping_chunk_size", 0)) > 0, "mapping_chunk_size must be positive")
    _require_equal(build_record.get("assignment_generator_imported_or_called"), False, "run assignment generator firewall")
    _require_equal(build_record.get("real_condition_contrast_generated"), False, "run condition firewall")
    _require_equal(build_record.get("timing_computed"), False, "run timing firewall")
    _require_equal(build_record.get("timing_fields_present"), False, "run timing-field firewall")
    expected_artifacts = {
        name: _hash_file(directory / name)
        for name in OUTPUT_FILES
        if name != RUN_BUILD_RECORD_FILE
    }
    _require_equal(build_record.get("artifact_sha256"), expected_artifacts, "run artifact hashes")
    completed_status = refusal["replicate_status"].astype(str).str.lower().isin(
        {"complete", "completed", "success", "ok"}
    )
    band_by_replicate = (
        curve.assign(
            _contains=derived_contains,
            _supported=expected_support,
        )
        .groupby("replicate_index_0based", sort=True)
        .apply(lambda frame: bool(frame.loc[frame["_supported"], "_contains"].all()), include_groups=False)
    )
    band_values = band_by_replicate.reindex(range(500)).to_numpy(dtype=bool)
    band_values[~completed_status.to_numpy(dtype=bool)] = False
    recomputed = aggregate_cb2_500_acceptance(
        pathway_metrics=pathway,
        family_metrics=family,
        replicate_metrics=pd.DataFrame(
            {
                "replicate_index_0based": range(500),
                "global_band_coverage_indicator": band_values,
            }
        ),
        refusal_audit=refusal,
        curve_metrics=curve,
        alpha=float(config["functional_inference_contract"]["alpha"]),
        config=config,
    )
    _require_equal(_canonical_json(summary), _canonical_json(recomputed), "recomputed acceptance summary")
    _require_equal(decision["cb2_500_pass"], summary["cb2_500_pass"], "decision pass")
    _require_equal(decision["cb2_2000_technical_gate_satisfied"], summary["cb2_500_pass"], "decision technical gate")
    _require_equal(
        decision["all_five_numeric_gates_pass"],
        summary["all_numeric_gates_pass"],
        "decision numeric gates",
    )
    _require_equal(
        decision["zero_false_open_replicates"],
        summary["n_false_open_replicates"] == 0,
        "decision false-open gate",
    )
    _require_equal(
        decision["zero_other_refused_or_incomplete_replicates"],
        summary["n_refused_replicates"] == 0,
        "decision refusal gate",
    )
    expected_blocking: list[str] = []
    for name in (
        "global_50_curve_maxT_fwer",
        "family_13_macro_fwer",
        "pooled_integrated_effect_type_i_error",
        "complete_null_BY_fdr",
        "global_zero_curve_simultaneous_band_coverage",
    ):
        if not summary[name]["pass"]:
            expected_blocking.append(f"numeric_gate_failed:{name}")
    if summary["n_false_open_replicates"]:
        expected_blocking.append("false_open_nonestimable")
    if summary["n_refused_replicates"]:
        expected_blocking.append("refused_or_incomplete_replicate")
    _require_equal(decision["blocking_reason_codes"], expected_blocking, "decision blocking reasons")
    _require_equal(
        decision["decision"],
        (
            "pass_cb2_500_technical_gate_pending_separate_cb2_2000_authorization"
            if summary["cb2_500_pass"]
            else "fail_cb2_500_stop_before_cb2_2000"
        ),
        "decision text",
    )
    _require_equal(
        _canonical_json(decision["claim_ceiling"]),
        _canonical_json(summary["claim_scope"]),
        "decision claim ceiling",
    )
    return {
        "valid": True,
        "output_dir": str(directory),
        "cb2_500_pass": bool(summary["cb2_500_pass"]),
        "cb2_2000_technical_gate_satisfied": bool(summary["cb2_500_pass"]),
        "cb2_2000_start_allowed": False,
        "acceptance_summary_sha256": _hash_file(directory / SUMMARY_FILE),
        "acceptance_decision_sha256": _hash_file(directory / DECISION_FILE),
        "build_record_sha256": _hash_file(directory / RUN_BUILD_RECORD_FILE),
    }
