"""Timing-free TrajPathMix functional-core inference on donor arrays.

This module deliberately has no dependency on event, onset, peak, duration, or
other low-level trajectory feature code.  Its unit of independence and residual
mapping is the whole donor.  The same donor mapping is used for every bin and
pathway, and mappings are restricted by the complete availability signature.

The implementation is intentionally narrow:

* arrays are ``donor x bin x pathway``;
* donors are canonicalized lexicographically before any stochastic operation;
* experiment fractions are a bin-specific, no-intercept nuisance design;
* active nuisance columns must be full rank at the frozen relative tolerance;
* all fitted models use reduced QR plus triangular solves (never a pseudoinverse);
* tests use reduced-model Freedman--Lane residual mappings;
* bands use full-model residual bootstrap mappings; and
* the two references share the exact same identity-excluding mapping plan.

The public result and its JSON representation expose functional curves only.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_NAME = "trajpathmix_functional_core_result"
SCHEMA_VERSION = "1.0.0"
RANK_RELATIVE_TOLERANCE = 1.0e-10
DEFAULT_MAPPING_SEED = 2026071403
_TOLERANCE = 1.0e-12
_FORBIDDEN_KEY_TOKENS = (
    "onset",
    "duration",
    "peak_location",
    "peak_time",
    "phase",
    "delay",
    "heterochron",
    "transient",
    "sustained",
    "event_support",
    "event_time",
)
_ALLOWED_NEGATIVE_SCOPE_KEYS = {"timing_computed", "timing_fields_present"}


class FunctionalCoreDesignError(ValueError):
    """Fail-closed error carrying design diagnostics."""

    def __init__(self, message: str, diagnostics: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


def _freeze_array(values: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _nonblank_unique(values: Sequence[Any], label: str) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values)
    if not normalized or any(not value for value in normalized):
        raise ValueError(f"{label} must contain nonblank values")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} must be unique")
    return normalized


def _canonical_donors(
    donor_ids: Sequence[Any],
) -> tuple[tuple[str, ...], np.ndarray]:
    observed = _nonblank_unique(donor_ids, "donor_ids")
    order = np.asarray(sorted(range(len(observed)), key=lambda index: observed[index]))
    canonical = tuple(observed[int(index)] for index in order)
    return canonical, order


def _canonical_experiments(
    experiment_ids: Sequence[Any],
) -> tuple[tuple[str, ...], np.ndarray]:
    observed = _nonblank_unique(experiment_ids, "experiment_ids")
    order = np.asarray(sorted(range(len(observed)), key=lambda index: observed[index]))
    canonical = tuple(observed[int(index)] for index in order)
    return canonical, order


def _as_binary(values: Any, *, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}; observed {array.shape}")
    if not bool(np.isin(array, [0, 1, False, True]).all()):
        raise ValueError(f"{label} must contain only binary values")
    return np.asarray(array, dtype=bool)


def _mapping_hash(mapping: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(mapping, dtype="<i4"))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable")


def _assert_functional_only_keys(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in _ALLOWED_NEGATIVE_SCOPE_KEYS:
                if child is not False:
                    raise ValueError(f"{path}.{key} must be false")
                continue
            if any(token in normalized for token in _FORBIDDEN_KEY_TOKENS):
                raise ValueError(f"Forbidden non-functional key at {path}.{key}")
            _assert_functional_only_keys(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_functional_only_keys(child, f"{path}[{index}]")


def _array_schema(depth: int) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "array"}
    cursor = schema
    for _ in range(depth - 1):
        child = {"type": "array"}
        cursor["items"] = child
        cursor = child
    cursor["items"] = {"type": ["number", "null"]}
    return schema


_MAPPING_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "donor_ids",
        "availability_signatures",
        "groups",
        "mappings",
        "mapping_hashes",
        "n_mappings",
        "seed",
        "n_unique_signatures",
        "n_mobile_donors",
        "n_immobile_donors",
        "orbit_size",
        "n_unique_nonidentity_mappings",
        "attainable_exact_p_resolution",
        "sampled_p_resolution",
        "identity_included",
    ],
    "properties": {
        "donor_ids": {"type": "array", "items": {"type": "string"}},
        "availability_signatures": {
            "type": "array",
            "items": {"type": "string"},
        },
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["signature", "donor_indices", "donor_ids"],
                "properties": {
                    "signature": {"type": "string"},
                    "donor_indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "donor_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
        "mappings": _array_schema(2),
        "mapping_hashes": {"type": "array", "items": {"type": "string"}},
        "n_mappings": {"type": "integer"},
        "seed": {"type": "integer"},
        "n_unique_signatures": {"type": "integer"},
        "n_mobile_donors": {"type": "integer"},
        "n_immobile_donors": {"type": "integer"},
        "orbit_size": {"type": "integer"},
        "n_unique_nonidentity_mappings": {"type": "integer"},
        "attainable_exact_p_resolution": {"type": "number"},
        "sampled_p_resolution": {"type": "number"},
        "identity_included": {"const": False},
    },
}


_BIN_DIAGNOSTIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "bin_index",
        "available_donor_indices",
        "active_experiment_ids",
        "dropped_bin_all_zero_experiment_ids",
        "reduced_rank",
        "full_rank",
        "condition_column_index",
        "residual_df",
        "n_case",
        "n_control",
        "condition_information",
        "condition_vif",
        "reduced_rank_threshold",
        "full_rank_threshold",
        "estimable",
        "reasons",
    ],
    "properties": {
        "bin_index": {"type": "integer"},
        "available_donor_indices": {
            "type": "array",
            "items": {"type": "integer"},
        },
        "active_experiment_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "dropped_bin_all_zero_experiment_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "reduced_rank": {"type": "integer"},
        "full_rank": {"type": "integer"},
        "condition_column_index": {"type": "integer"},
        "residual_df": {"type": "integer"},
        "n_case": {"type": "integer"},
        "n_control": {"type": "integer"},
        "condition_information": {"type": ["number", "null"]},
        "condition_vif": {"type": ["number", "null"]},
        "reduced_rank_threshold": {"type": "number"},
        "full_rank_threshold": {"type": "number"},
        "estimable": {"type": "boolean"},
        "reasons": {"type": "array", "items": {"type": "string"}},
    },
}


_DESIGN_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "donor_ids",
        "experiment_ids",
        "retained_experiment_ids",
        "dropped_global_all_zero_experiment_ids",
        "rank_relative_tolerance",
        "no_intercept",
        "all_estimable",
        "bins",
    ],
    "properties": {
        "donor_ids": {"type": "array", "items": {"type": "string"}},
        "experiment_ids": {"type": "array", "items": {"type": "string"}},
        "retained_experiment_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "dropped_global_all_zero_experiment_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "rank_relative_tolerance": {"const": RANK_RELATIVE_TOLERANCE},
        "no_intercept": {"const": True},
        "all_estimable": {"type": "boolean"},
        "bins": {"type": "array", "items": _BIN_DIAGNOSTIC_SCHEMA},
    },
}


_FAMILY_TEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "family_id",
        "n_pathways",
        "observed_statistic",
        "p_raw",
        "p_maxT",
        "simultaneous_critical",
    ],
    "properties": {
        "family_id": {"type": "string"},
        "n_pathways": {"type": "integer"},
        "observed_statistic": {"type": "number"},
        "p_raw": {"type": "number"},
        "p_maxT": {"type": "number"},
        "simultaneous_critical": {"type": "number"},
    },
}


_INFERENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "method",
        "test_reference",
        "band_reference",
        "mapping_scope",
        "band_scope",
        "identity_in_null",
        "plus_one_p_values",
        "rank_relative_tolerance",
        "no_intercept",
        "functional_fields_only",
    ],
    "properties": {
        "method": {"type": "string"},
        "test_reference": {"type": "string"},
        "band_reference": {"type": "string"},
        "mapping_scope": {"type": "string"},
        "band_scope": {"type": "string"},
        "identity_in_null": {"const": False},
        "plus_one_p_values": {"const": True},
        "rank_relative_tolerance": {"const": RANK_RELATIVE_TOLERANCE},
        "no_intercept": {"const": True},
        "functional_fields_only": {"const": True},
    },
}


_MAPPING_AUDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "n_mappings",
        "seed",
        "n_unique_signatures",
        "n_mobile_donors",
        "n_immobile_donors",
        "orbit_size",
        "n_unique_nonidentity_mappings",
        "attainable_exact_p_resolution",
        "sampled_p_resolution",
        "identity_included",
        "same_mapping_all_bins_and_pathways",
        "availability_signature_width",
        "n_unique_mapping_hashes",
        "mapping_stream_sha256",
    ],
    "properties": {
        "n_mappings": {"type": "integer"},
        "seed": {"type": "integer"},
        "n_unique_signatures": {"type": "integer"},
        "n_mobile_donors": {"type": "integer"},
        "n_immobile_donors": {"type": "integer"},
        "orbit_size": {"type": "integer"},
        "n_unique_nonidentity_mappings": {"type": "integer"},
        "attainable_exact_p_resolution": {"type": "number"},
        "sampled_p_resolution": {"type": "number"},
        "identity_included": {"const": False},
        "same_mapping_all_bins_and_pathways": {"const": True},
        "availability_signature_width": {"type": "integer"},
        "n_unique_mapping_hashes": {"type": "integer"},
        "mapping_stream_sha256": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
    },
}


_CLAIM_SCOPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "functional_core_only",
        "timing_computed",
        "timing_fields_present",
        "finite_sample_exact",
        "reference_type",
        "nuisance_block_invariant",
        "strong_fwer_claimed",
        "fwer_scope",
        "nonzero_curve_coverage_claimed",
        "band_coverage_scope",
    ],
    "properties": {
        "functional_core_only": {"const": True},
        "timing_computed": {"const": False},
        "timing_fields_present": {"const": False},
        "finite_sample_exact": {"const": False},
        "reference_type": {"type": "string"},
        "nuisance_block_invariant": {"const": False},
        "strong_fwer_claimed": {"const": False},
        "fwer_scope": {"const": "complete_null_weak_fwer_only"},
        "nonzero_curve_coverage_claimed": {"const": False},
        "band_coverage_scope": {"const": "complete_null_zero_curve_only"},
    },
}


FUNCTIONAL_CORE_RESULT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "TrajPathMix functional-core result v1",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_name",
        "schema_version",
        "donor_ids",
        "bin_indices",
        "pathway_ids",
        "effect_curve",
        "standard_error_curve",
        "pointwise_lower",
        "pointwise_upper",
        "simultaneous_lower",
        "simultaneous_upper",
        "simultaneous_critical",
        "integrated_effect",
        "p_curve_maxT",
        "p_integrated",
        "p_family_maxT",
        "q_by",
        "support_mask",
        "design_diagnostics",
        "mapping_audit",
        "claim_scope",
    ],
    "properties": {
        "schema_name": {"const": SCHEMA_NAME},
        "schema_version": {"const": SCHEMA_VERSION},
        "donor_ids": {"type": "array", "items": {"type": "string"}},
        "bin_indices": {"type": "array", "items": {"type": "integer"}},
        "pathway_ids": {"type": "array", "items": {"type": "string"}},
        "effect_curve": _array_schema(2),
        "standard_error_curve": _array_schema(2),
        "pointwise_lower": _array_schema(2),
        "pointwise_upper": _array_schema(2),
        "simultaneous_lower": _array_schema(2),
        "simultaneous_upper": _array_schema(2),
        "simultaneous_critical": {"type": "number"},
        "integrated_effect": _array_schema(1),
        "p_curve_maxT": _array_schema(1),
        "p_integrated": _array_schema(1),
        "p_family_maxT": _array_schema(1),
        "q_by": _array_schema(1),
        "support_mask": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "boolean"}},
        },
        "design_diagnostics": _DESIGN_SUMMARY_SCHEMA,
        "mapping_audit": _MAPPING_AUDIT_SCHEMA,
        "claim_scope": _CLAIM_SCOPE_SCHEMA,
    },
}


def functional_core_result_json_schema() -> dict[str, Any]:
    """Return an independent strict JSON Schema for :class:`FunctionalCoreResult`."""

    return deepcopy(FUNCTIONAL_CORE_RESULT_SCHEMA)


@dataclass(frozen=True)
class AvailabilityMappingPlan:
    donor_ids: tuple[str, ...]
    canonical_order: np.ndarray
    availability: np.ndarray
    availability_signatures: tuple[str, ...]
    groups: tuple[tuple[int, ...], ...]
    mappings: np.ndarray
    mapping_hashes: tuple[str, ...]
    mapping_stream_sha256: str
    n_unique_signatures: int
    n_mobile_donors: int
    n_immobile_donors: int
    orbit_size: int
    n_unique_nonidentity_mappings: int
    attainable_exact_p_resolution: float
    sampled_p_resolution: float
    seed: int

    @property
    def n_mappings(self) -> int:
        return int(self.mappings.shape[0])

    def to_dict(self, *, include_mappings: bool = True) -> dict[str, Any]:
        groups = []
        for group in self.groups:
            signature = self.availability_signatures[group[0]]
            groups.append(
                {
                    "signature": signature,
                    "donor_indices": list(group),
                    "donor_ids": [self.donor_ids[index] for index in group],
                }
            )
        return _json_safe(
            {
                "donor_ids": self.donor_ids,
                "availability_signatures": self.availability_signatures,
                "groups": groups,
                "mappings": self.mappings if include_mappings else [],
                "mapping_hashes": self.mapping_hashes,
                "n_mappings": self.n_mappings,
                "seed": int(self.seed),
                "n_unique_signatures": int(self.n_unique_signatures),
                "n_mobile_donors": int(self.n_mobile_donors),
                "n_immobile_donors": int(self.n_immobile_donors),
                "orbit_size": int(self.orbit_size),
                "n_unique_nonidentity_mappings": int(
                    self.n_unique_nonidentity_mappings
                ),
                "attainable_exact_p_resolution": float(
                    self.attainable_exact_p_resolution
                ),
                "sampled_p_resolution": float(self.sampled_p_resolution),
                "identity_included": False,
            }
        )


@dataclass(frozen=True)
class BinSpecificDesign:
    bin_index: int
    available_donor_indices: np.ndarray
    active_experiment_ids: tuple[str, ...]
    dropped_bin_all_zero_experiment_ids: tuple[str, ...]
    reduced_design: np.ndarray
    full_design: np.ndarray
    reduced_rank: int
    full_rank: int
    condition_column_index: int
    residual_df: int
    n_case: int
    n_control: int
    condition_information: float
    condition_vif: float
    reduced_rank_threshold: float
    full_rank_threshold: float
    estimable: bool
    reasons: tuple[str, ...]

    @property
    def available_indices(self) -> np.ndarray:
        return self.available_donor_indices

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "bin_index": int(self.bin_index),
                "available_donor_indices": self.available_donor_indices,
                "active_experiment_ids": self.active_experiment_ids,
                "dropped_bin_all_zero_experiment_ids": (
                    self.dropped_bin_all_zero_experiment_ids
                ),
                "reduced_rank": int(self.reduced_rank),
                "full_rank": int(self.full_rank),
                "condition_column_index": int(self.condition_column_index),
                "residual_df": int(self.residual_df),
                "n_case": int(self.n_case),
                "n_control": int(self.n_control),
                "condition_information": float(self.condition_information),
                "condition_vif": float(self.condition_vif),
                "reduced_rank_threshold": float(self.reduced_rank_threshold),
                "full_rank_threshold": float(self.full_rank_threshold),
                "estimable": bool(self.estimable),
                "reasons": self.reasons,
            }
        )


@dataclass(frozen=True)
class BinSpecificDesignPlan:
    donor_ids: tuple[str, ...]
    canonical_donor_order: np.ndarray
    experiment_ids: tuple[str, ...]
    canonical_experiment_order: np.ndarray
    retained_experiment_ids: tuple[str, ...]
    dropped_global_all_zero_experiment_ids: tuple[str, ...]
    condition: np.ndarray
    availability: np.ndarray
    experiment_fractions: np.ndarray
    bins: tuple[BinSpecificDesign, ...]
    rank_relative_tolerance: float

    @property
    def all_estimable(self) -> bool:
        return all(item.estimable for item in self.bins)

    @property
    def bin_designs(self) -> tuple[BinSpecificDesign, ...]:
        return self.bins

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "donor_ids": self.donor_ids,
                "experiment_ids": self.experiment_ids,
                "retained_experiment_ids": self.retained_experiment_ids,
                "dropped_global_all_zero_experiment_ids": (
                    self.dropped_global_all_zero_experiment_ids
                ),
                "rank_relative_tolerance": float(self.rank_relative_tolerance),
                "no_intercept": True,
                "all_estimable": bool(self.all_estimable),
                "bins": [item.to_dict() for item in self.bins],
            }
        )


@dataclass(frozen=True)
class FunctionalCoreResult:
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
    curve_p_raw: np.ndarray
    curve_p_maxT: np.ndarray
    curve_q_by: np.ndarray
    integrated_absolute_effect: np.ndarray
    integrated_studentized_statistic: np.ndarray
    integrated_p_raw: np.ndarray
    integrated_p_maxT: np.ndarray
    integrated_q_by: np.ndarray
    family_p_maxT: np.ndarray
    family_tests: tuple[Mapping[str, Any], ...]
    mapping_plan: AvailabilityMappingPlan
    design_plan: BinSpecificDesignPlan
    inference_metadata: Mapping[str, Any]

    @property
    def bin_indices(self) -> tuple[int, ...]:
        return tuple(range(self.effect.shape[0]))

    @property
    def effect_curve(self) -> np.ndarray:
        return self.effect

    @property
    def standard_error_curve(self) -> np.ndarray:
        return self.standard_error

    @property
    def simultaneous_critical_value(self) -> float:
        return self.simultaneous_critical

    @property
    def integrated_effect(self) -> np.ndarray:
        return self.integrated_absolute_effect

    def to_dict(self) -> dict[str, Any]:
        mapping_audit = {
            "n_mappings": self.mapping_plan.n_mappings,
            "seed": int(self.mapping_plan.seed),
            "n_unique_signatures": int(self.mapping_plan.n_unique_signatures),
            "n_mobile_donors": int(self.mapping_plan.n_mobile_donors),
            "n_immobile_donors": int(self.mapping_plan.n_immobile_donors),
            "orbit_size": int(self.mapping_plan.orbit_size),
            "n_unique_nonidentity_mappings": int(
                self.mapping_plan.n_unique_nonidentity_mappings
            ),
            "attainable_exact_p_resolution": float(
                self.mapping_plan.attainable_exact_p_resolution
            ),
            "sampled_p_resolution": float(
                self.mapping_plan.sampled_p_resolution
            ),
            "identity_included": False,
            "same_mapping_all_bins_and_pathways": True,
            "availability_signature_width": int(
                self.mapping_plan.availability.shape[1]
            ),
            "n_unique_mapping_hashes": int(
                len(set(self.mapping_plan.mapping_hashes))
            ),
            "mapping_stream_sha256": self.mapping_plan.mapping_stream_sha256,
        }
        claim_scope = {
            "functional_core_only": True,
            "timing_computed": False,
            "timing_fields_present": False,
            "finite_sample_exact": False,
            "reference_type": "freedman_lane_monte_carlo_approximation",
            "nuisance_block_invariant": False,
            "strong_fwer_claimed": False,
            "fwer_scope": "complete_null_weak_fwer_only",
            "nonzero_curve_coverage_claimed": False,
            "band_coverage_scope": "complete_null_zero_curve_only",
        }
        payload = _json_safe(
            {
                "schema_name": SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                "donor_ids": self.donor_ids,
                "bin_indices": self.bin_indices,
                "pathway_ids": self.pathway_ids,
                "effect_curve": self.effect,
                "standard_error_curve": self.standard_error,
                "pointwise_lower": self.pointwise_lower,
                "pointwise_upper": self.pointwise_upper,
                "simultaneous_lower": self.simultaneous_lower,
                "simultaneous_upper": self.simultaneous_upper,
                "simultaneous_critical": float(self.simultaneous_critical),
                "integrated_effect": self.integrated_absolute_effect,
                "p_curve_maxT": self.curve_p_maxT,
                "p_integrated": self.integrated_p_raw,
                "p_family_maxT": self.family_p_maxT,
                "q_by": self.curve_q_by,
                "support_mask": self.support_mask,
                "design_diagnostics": self.design_plan.to_dict(),
                "mapping_audit": mapping_audit,
                "claim_scope": claim_scope,
            }
        )
        _assert_functional_only_keys(payload)
        return payload

    def to_tables(self) -> dict[str, Any]:
        """Return functional-only pandas tables suitable for external writers."""

        import pandas as pd

        curves: list[dict[str, Any]] = []
        for bin_index in range(self.effect.shape[0]):
            for pathway_index, pathway in enumerate(self.pathway_ids):
                curves.append(
                    _json_safe(
                        {
                            "bin_index": bin_index,
                            "pathway_id": pathway,
                            "supported": bool(
                                self.support_mask[bin_index, pathway_index]
                            ),
                            "effect": self.effect[bin_index, pathway_index],
                            "standard_error": self.standard_error[
                                bin_index, pathway_index
                            ],
                            "studentized_effect": self.studentized_effect[
                                bin_index, pathway_index
                            ],
                            "pointwise_p": self.pointwise_p[
                                bin_index, pathway_index
                            ],
                            "pointwise_lower": self.pointwise_lower[
                                bin_index, pathway_index
                            ],
                            "pointwise_upper": self.pointwise_upper[
                                bin_index, pathway_index
                            ],
                            "simultaneous_lower": self.simultaneous_lower[
                                bin_index, pathway_index
                            ],
                            "simultaneous_upper": self.simultaneous_upper[
                                bin_index, pathway_index
                            ],
                        }
                    )
                )
        tests: list[dict[str, Any]] = []
        for pathway_index, pathway in enumerate(self.pathway_ids):
            tests.append(
                _json_safe(
                    {
                        "pathway_id": pathway,
                        "p_curve_maxT": self.curve_p_maxT[pathway_index],
                        "q_by": self.curve_q_by[pathway_index],
                        "integrated_effect": (
                            self.integrated_absolute_effect[pathway_index]
                        ),
                        "p_integrated": self.integrated_p_raw[pathway_index],
                        "p_family_maxT": self.family_p_maxT[pathway_index],
                    }
                )
            )
        payload = {
            "functional_curves": pd.DataFrame(curves),
            "pathway_tests": pd.DataFrame(tests),
            "family_tests": pd.DataFrame(
                [_json_safe(item) for item in self.family_tests]
            ),
        }
        for frame in payload.values():
            _assert_functional_only_keys({column: None for column in frame.columns})
        return payload


def _signature_groups(signatures: tuple[str, ...]) -> tuple[tuple[int, ...], ...]:
    lookup: dict[str, list[int]] = {}
    for index, signature in enumerate(signatures):
        lookup.setdefault(signature, []).append(index)
    # The random stream must not depend on lexicographic signature text.  Group
    # order is the order of the smallest canonical donor index in each group.
    groups = [tuple(indices) for indices in lookup.values()]
    return tuple(sorted(groups, key=lambda group: group[0]))


def _enumerate_mappings(
    n_donors: int, groups: tuple[tuple[int, ...], ...]
) -> list[np.ndarray]:
    group_permutations = [list(itertools.permutations(group)) for group in groups]
    mappings: list[np.ndarray] = []
    identity = np.arange(n_donors, dtype=np.int32)
    for choices in itertools.product(*group_permutations):
        mapping = identity.copy()
        for group, permuted in zip(groups, choices):
            mapping[np.asarray(group, dtype=int)] = np.asarray(permuted, dtype=int)
        mappings.append(mapping)
    return mappings


def build_full_availability_mapping_plan(
    donor_ids: Sequence[Any],
    availability: Any,
    *,
    n_mappings: int = 999,
    seed: int = DEFAULT_MAPPING_SEED,
) -> AvailabilityMappingPlan:
    """Build identity-excluding whole-donor mappings within full signatures."""

    canonical_donors, order = _canonical_donors(donor_ids)
    observed = np.asarray(availability)
    if observed.ndim != 2 or observed.shape[0] != len(canonical_donors):
        raise ValueError("availability must be donor x bin")
    n_bins = int(observed.shape[1])
    if n_bins <= 0:
        raise ValueError("availability must contain at least one bin")
    canonical_availability = _as_binary(
        observed[order],
        shape=(len(canonical_donors), n_bins),
        label="availability",
    )
    requested = int(n_mappings)
    if requested <= 0:
        raise ValueError("n_mappings must be positive")
    signatures = tuple(
        "".join("1" if value else "0" for value in row)
        for row in canonical_availability
    )
    groups = _signature_groups(signatures)
    orbit = int(math.prod(math.factorial(len(group)) for group in groups))
    available_nonidentity = int(orbit - 1)
    if requested > available_nonidentity:
        raise FunctionalCoreDesignError(
            "Requested mappings exceed the non-identity full-signature orbit",
            {
                "requested_mappings": requested,
                "orbit_size": orbit,
                "n_unique_nonidentity_mappings": available_nonidentity,
            },
        )
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    identity = np.arange(len(canonical_donors), dtype=np.int32)
    selected: list[np.ndarray] = []
    if orbit <= 20_000:
        candidates = [
            mapping
            for mapping in _enumerate_mappings(len(canonical_donors), groups)
            if not np.array_equal(mapping, identity)
        ]
        chosen = generator.choice(len(candidates), size=requested, replace=False)
        selected = [candidates[int(index)] for index in chosen]
    else:
        seen: set[str] = set()
        maximum_attempts = max(10_000, requested * 200)
        for _ in range(maximum_attempts):
            mapping = identity.copy()
            for group in groups:
                if len(group) > 1:
                    indices = np.asarray(group, dtype=int)
                    mapping[indices] = generator.permutation(indices)
            if np.array_equal(mapping, identity):
                continue
            digest = _mapping_hash(mapping)
            if digest in seen:
                continue
            seen.add(digest)
            selected.append(mapping)
            if len(selected) == requested:
                break
        if len(selected) != requested:
            raise RuntimeError(
                f"Generated only {len(selected)} of {requested} unique mappings"
            )
    mappings = np.stack(selected, axis=0).astype(np.int32, copy=False)
    hashes = tuple(_mapping_hash(row) for row in mappings)
    if len(set(hashes)) != requested:
        raise RuntimeError("Internal duplicate residual mapping")
    for group in groups:
        group_set = set(group)
        for mapping in mappings:
            if set(map(int, mapping[list(group)])) != group_set:
                raise RuntimeError("Residual mapping crossed an availability signature")
    mobile = int(sum(len(group) for group in groups if len(group) > 1))
    return AvailabilityMappingPlan(
        donor_ids=canonical_donors,
        canonical_order=_freeze_array(order, dtype=int),
        availability=_freeze_array(canonical_availability, dtype=bool),
        availability_signatures=signatures,
        groups=groups,
        mappings=_freeze_array(mappings, dtype=np.int32),
        mapping_hashes=hashes,
        mapping_stream_sha256=hashlib.sha256(
            np.ascontiguousarray(mappings, dtype="<i4").tobytes()
        ).hexdigest(),
        n_unique_signatures=len(groups),
        n_mobile_donors=mobile,
        n_immobile_donors=len(canonical_donors) - mobile,
        orbit_size=orbit,
        n_unique_nonidentity_mappings=available_nonidentity,
        attainable_exact_p_resolution=float(1.0 / orbit),
        sampled_p_resolution=float(1.0 / (requested + 1)),
        seed=int(seed),
    )


def _svd_rank(values: np.ndarray) -> tuple[int, float]:
    singular = np.linalg.svd(np.asarray(values, dtype=float), compute_uv=False)
    if not len(singular) or singular[0] == 0:
        return 0, 0.0
    threshold = float(singular[0] * RANK_RELATIVE_TOLERANCE)
    return int(np.sum(singular > threshold)), threshold


def _qr_projection_residual(design: np.ndarray, values: np.ndarray) -> np.ndarray:
    if design.shape[1] == 0:
        return np.asarray(values, dtype=float).copy()
    q, r = np.linalg.qr(design, mode="reduced")
    if np.any(np.abs(np.diag(r)) <= RANK_RELATIVE_TOLERANCE * np.abs(r).max()):
        raise FunctionalCoreDesignError("Rank-deficient nuisance design")
    return np.asarray(values, dtype=float) - q @ (q.T @ np.asarray(values, dtype=float))


def build_bin_specific_designs(
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
) -> BinSpecificDesignPlan:
    """Construct and audit no-intercept experiment-fraction designs by bin."""

    if float(rank_tolerance) != RANK_RELATIVE_TOLERANCE:
        raise ValueError(
            f"rank_tolerance is frozen at {RANK_RELATIVE_TOLERANCE:.1e}"
        )
    canonical_donors, donor_order = _canonical_donors(donor_ids)
    n_donors = len(canonical_donors)
    observed_availability = np.asarray(availability)
    if observed_availability.ndim != 2 or observed_availability.shape[0] != n_donors:
        raise ValueError("availability must be donor x bin")
    n_bins = int(observed_availability.shape[1])
    if n_bins <= 0:
        raise ValueError("availability must contain at least one bin")
    canonical_availability = _as_binary(
        observed_availability[donor_order],
        shape=(n_donors, n_bins),
        label="availability",
    )
    observed_condition = np.asarray(condition)
    if observed_condition.shape != (n_donors,):
        raise ValueError("condition must have shape (n_donors,)")
    canonical_condition = _as_binary(
        observed_condition[donor_order],
        shape=(n_donors,),
        label="condition",
    ).astype(float)
    canonical_experiments, experiment_order = _canonical_experiments(experiment_ids)
    fractions = np.asarray(experiment_fractions, dtype=float)
    expected_shape = (n_donors, n_bins, len(canonical_experiments))
    if fractions.shape != expected_shape:
        raise ValueError(
            f"experiment_fractions must have shape {expected_shape}; "
            f"observed {fractions.shape}"
        )
    fractions = fractions[donor_order][:, :, experiment_order]
    available_values = fractions[canonical_availability]
    if not bool(np.isfinite(available_values).all()):
        raise ValueError("Available experiment fractions must be finite")
    if bool((available_values < -_TOLERANCE).any()):
        raise ValueError("Experiment fractions must be nonnegative")
    row_sums = fractions.sum(axis=2)[canonical_availability]
    if not bool(np.allclose(row_sums, 1.0, rtol=0.0, atol=1.0e-10)):
        raise ValueError("Available experiment-fraction rows must sum to one")
    masked = np.where(canonical_availability[:, :, None], fractions, 0.0)
    global_nonzero = np.any(np.abs(masked) > _TOLERANCE, axis=(0, 1))
    retained_indices = np.flatnonzero(global_nonzero)
    dropped_global_indices = np.flatnonzero(~global_nonzero)
    retained_ids = tuple(canonical_experiments[index] for index in retained_indices)
    dropped_global_ids = tuple(
        canonical_experiments[index] for index in dropped_global_indices
    )
    bins: list[BinSpecificDesign] = []
    for bin_index in range(n_bins):
        donor_indices = np.flatnonzero(canonical_availability[:, bin_index])
        local_fractions = fractions[donor_indices, bin_index][:, retained_indices]
        bin_nonzero = np.any(np.abs(local_fractions) > _TOLERANCE, axis=0)
        active_local = np.flatnonzero(bin_nonzero)
        inactive_local = np.flatnonzero(~bin_nonzero)
        active_ids = tuple(retained_ids[index] for index in active_local)
        dropped_bin_ids = tuple(retained_ids[index] for index in inactive_local)
        reduced = local_fractions[:, active_local]
        local_condition = canonical_condition[donor_indices]
        full = np.column_stack([reduced, local_condition])
        reduced_rank, reduced_threshold = _svd_rank(reduced)
        full_rank, full_threshold = _svd_rank(full)
        residual_df = int(len(donor_indices) - full_rank)
        n_case = int(local_condition.sum())
        n_control = int(len(local_condition) - n_case)
        reasons: list[str] = []
        if reduced.shape[1] == 0 or reduced_rank != reduced.shape[1]:
            reasons.append("rank_deficient_reduced_experiment_design")
        if full_rank != reduced_rank + 1 or full_rank != full.shape[1]:
            reasons.append("condition_not_identifiable")
        if min(n_case, n_control) < int(min_donors_per_condition):
            reasons.append("insufficient_donors_per_condition")
        if residual_df < int(min_residual_df):
            reasons.append("insufficient_residual_df")
        condition_information = 0.0
        condition_vif = math.inf
        if reduced_rank == reduced.shape[1] and reduced.shape[1] > 0:
            residualized = _qr_projection_residual(reduced, local_condition)
            condition_information = float(residualized @ residualized)
            centered = local_condition - float(local_condition.mean())
            unadjusted = float(centered @ centered)
            if condition_information > 1.0e-14 * max(1.0, unadjusted):
                condition_vif = float(unadjusted / condition_information)
        if not math.isfinite(condition_vif) or condition_vif > float(
            max_condition_vif
        ):
            reasons.append("condition_vif_exceeds_threshold")
        bins.append(
            BinSpecificDesign(
                bin_index=bin_index,
                available_donor_indices=_freeze_array(donor_indices, dtype=int),
                active_experiment_ids=active_ids,
                dropped_bin_all_zero_experiment_ids=dropped_bin_ids,
                reduced_design=_freeze_array(reduced, dtype=float),
                full_design=_freeze_array(full, dtype=float),
                reduced_rank=int(reduced_rank),
                full_rank=int(full_rank),
                condition_column_index=int(reduced.shape[1]),
                residual_df=residual_df,
                n_case=n_case,
                n_control=n_control,
                condition_information=condition_information,
                condition_vif=condition_vif,
                reduced_rank_threshold=reduced_threshold,
                full_rank_threshold=full_threshold,
                estimable=not reasons,
                reasons=tuple(dict.fromkeys(reasons)),
            )
        )
    plan = BinSpecificDesignPlan(
        donor_ids=canonical_donors,
        canonical_donor_order=_freeze_array(donor_order, dtype=int),
        experiment_ids=canonical_experiments,
        canonical_experiment_order=_freeze_array(experiment_order, dtype=int),
        retained_experiment_ids=retained_ids,
        dropped_global_all_zero_experiment_ids=dropped_global_ids,
        condition=_freeze_array(canonical_condition, dtype=float),
        availability=_freeze_array(canonical_availability, dtype=bool),
        experiment_fractions=_freeze_array(fractions, dtype=float),
        bins=tuple(bins),
        rank_relative_tolerance=RANK_RELATIVE_TOLERANCE,
    )
    if not plan.all_estimable:
        raise FunctionalCoreDesignError(
            "One or more bin-specific designs are not estimable",
            plan.to_dict(),
        )
    return plan


@dataclass(frozen=True)
class _Fit:
    coefficient: np.ndarray
    standard_error: np.ndarray
    studentized: np.ndarray
    fitted: np.ndarray
    residual: np.ndarray


def _studentize(
    numerator: np.ndarray,
    standard_error: np.ndarray,
    required_mask: np.ndarray | None = None,
) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=float)
    standard_error = np.asarray(standard_error, dtype=float)
    output = np.zeros_like(numerator)
    positive = standard_error > _TOLERANCE
    output[positive] = numerator[positive] / standard_error[positive]
    invalid = ~positive & (np.abs(numerator) > _TOLERANCE)
    required = (
        np.ones_like(invalid, dtype=bool)
        if required_mask is None
        else np.asarray(required_mask, dtype=bool)
    )
    if required.shape != invalid.shape:
        raise ValueError("required_mask must align with the fitted outcomes")
    if bool((invalid & required).any()):
        raise FunctionalCoreDesignError(
            "Studentization is undefined for a nonzero coefficient"
        )
    return output


def _fit_qr(
    design: np.ndarray,
    outcomes: np.ndarray,
    condition_column: int,
    required_mask: np.ndarray | None = None,
) -> _Fit:
    design = np.asarray(design, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    n_rows, n_columns = design.shape
    if n_rows <= n_columns:
        raise FunctionalCoreDesignError("Full model has no residual degrees of freedom")
    q, r = np.linalg.qr(design, mode="reduced")
    diagonal = np.abs(np.diag(r))
    threshold = (
        float(diagonal.max()) * RANK_RELATIVE_TOLERANCE if len(diagonal) else 0.0
    )
    if len(diagonal) != n_columns or bool((diagonal <= threshold).any()):
        raise FunctionalCoreDesignError("QR detected a rank-deficient fitted design")
    beta = np.linalg.solve(r, q.T @ outcomes)
    fitted = design @ beta
    residual = outcomes - fitted
    residual_df = n_rows - n_columns
    sigma_squared = np.sum(residual**2, axis=0) / residual_df
    inverse_r = np.linalg.solve(r, np.eye(n_columns))
    variance_factor = float(np.sum(inverse_r[condition_column] ** 2))
    standard_error = np.sqrt(np.maximum(0.0, sigma_squared * variance_factor))
    coefficient = beta[condition_column]
    studentized = _studentize(coefficient, standard_error, required_mask)
    return _Fit(
        coefficient=np.asarray(coefficient, dtype=float),
        standard_error=np.asarray(standard_error, dtype=float),
        studentized=np.asarray(studentized, dtype=float),
        fitted=np.asarray(fitted, dtype=float),
        residual=np.asarray(residual, dtype=float),
    )


def _fit_reduced(design: np.ndarray, outcomes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    design = np.asarray(design, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    if design.shape[1] == 0:
        return np.zeros_like(outcomes), outcomes.copy()
    q, r = np.linalg.qr(design, mode="reduced")
    diagonal = np.abs(np.diag(r))
    threshold = float(diagonal.max()) * RANK_RELATIVE_TOLERANCE
    if bool((diagonal <= threshold).any()):
        raise FunctionalCoreDesignError("Rank-deficient reduced fitted design")
    beta = np.linalg.solve(r, q.T @ outcomes)
    fitted = design @ beta
    return fitted, outcomes - fitted


def _plus_one_p(null: np.ndarray, observed: np.ndarray) -> np.ndarray:
    null = np.asarray(null, dtype=float)
    observed = np.asarray(observed, dtype=float)
    return (1.0 + np.sum(null >= observed[None, ...] - _TOLERANCE, axis=0)) / (
        null.shape[0] + 1.0
    )


def _by_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not bool(np.isfinite(p).all()):
        raise ValueError("BY adjustment requires finite one-dimensional p-values")
    count = len(p)
    harmonic = float(np.sum(1.0 / np.arange(1, count + 1)))
    order = np.argsort(p, kind="mergesort")
    ranked = p[order] * count * harmonic / np.arange(1, count + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum(1.0, ranked)
    return adjusted


def _order_critical(values: np.ndarray, order_index_1based: int) -> np.ndarray:
    sorted_values = np.sort(np.asarray(values, dtype=float), axis=0)
    return sorted_values[int(order_index_1based) - 1]


def _normalize_support_mask(mask: Any, n_bins: int, n_pathways: int) -> np.ndarray:
    if mask is None:
        result = np.ones((n_bins, n_pathways), dtype=bool)
    else:
        observed = np.asarray(mask)
        if observed.shape == (n_bins,):
            observed = np.repeat(observed[:, None], n_pathways, axis=1)
        result = _as_binary(
            observed,
            shape=(n_bins, n_pathways),
            label="support_mask",
        )
    if not bool(result.any(axis=0).all()):
        raise FunctionalCoreDesignError(
            "Every pathway must have at least one supported bin"
        )
    return result


def run_functional_core(
    *,
    outcomes: Any,
    donor_ids: Sequence[Any],
    condition: Any,
    availability: Any,
    experiment_fractions: Any,
    experiment_ids: Sequence[Any],
    pathway_ids: Sequence[Any],
    family_ids: Sequence[Any] | None = None,
    support_mask: Any = None,
    bin_weights: Any = None,
    n_mappings: int = 999,
    seed: int = DEFAULT_MAPPING_SEED,
    alpha: float = 0.05,
    rank_tolerance: float = RANK_RELATIVE_TOLERANCE,
    min_donors_per_condition: int = 10,
    min_residual_df: int = 3,
    max_condition_vif: float = 10.0,
) -> FunctionalCoreResult:
    """Run functional-only donor-curve inference with a shared mapping plan."""

    alpha_value = float(alpha)
    if not (0.0 < alpha_value < 1.0):
        raise ValueError("alpha must lie strictly between zero and one")
    pathways = _nonblank_unique(pathway_ids, "pathway_ids")
    observed = np.asarray(outcomes, dtype=float)
    if observed.ndim != 3:
        raise ValueError("outcomes must be donor x bin x pathway")
    n_bins = int(observed.shape[1])
    expected = (len(donor_ids), n_bins, len(pathways))
    if observed.shape != expected:
        raise ValueError(f"outcomes must have shape {expected}; observed {observed.shape}")
    design_plan = build_bin_specific_designs(
        donor_ids,
        condition,
        availability,
        experiment_fractions,
        experiment_ids,
        rank_tolerance=rank_tolerance,
        min_donors_per_condition=min_donors_per_condition,
        min_residual_df=min_residual_df,
        max_condition_vif=max_condition_vif,
    )
    mapping_plan = build_full_availability_mapping_plan(
        donor_ids,
        availability,
        n_mappings=n_mappings,
        seed=seed,
    )
    if mapping_plan.donor_ids != design_plan.donor_ids:
        raise RuntimeError("Canonical donor orders differ between design and mapping")
    if not design_plan.all_estimable:
        raise FunctionalCoreDesignError(
            "One or more bin-specific designs are not estimable",
            design_plan.to_dict(),
        )
    canonical_outcomes = observed[design_plan.canonical_donor_order]
    available = design_plan.availability
    if not bool(np.isfinite(canonical_outcomes[available]).all()):
        raise ValueError("All available donor-bin pathway outcomes must be finite")
    if not bool(np.isnan(canonical_outcomes[~available]).all()):
        raise ValueError("Unavailable donor-bin pathway outcomes must remain NA")
    support = _normalize_support_mask(support_mask, n_bins, len(pathways))
    if bin_weights is None:
        weights = np.full(n_bins, 1.0 / n_bins, dtype=float)
    else:
        weights = np.asarray(bin_weights, dtype=float)
        if weights.shape != (n_bins,) or not bool(np.isfinite(weights).all()):
            raise ValueError("bin_weights must be a finite length-n_bins vector")
        if bool((weights <= 0).any()):
            raise ValueError("bin_weights must be strictly positive")
    if family_ids is None:
        families: tuple[str, ...] | None = None
    else:
        if len(family_ids) != len(pathways):
            raise ValueError("family_ids must align one-to-one with pathway_ids")
        families = tuple(str(item).strip() for item in family_ids)
        if any(not item for item in families):
            raise ValueError("family_ids must be nonblank")

    n_pathways = len(pathways)
    n_reference = mapping_plan.n_mappings
    effect = np.empty((n_bins, n_pathways), dtype=float)
    standard_error = np.empty_like(effect)
    studentized = np.empty_like(effect)
    reduced_fitted: list[np.ndarray] = []
    reduced_residual: list[np.ndarray] = []
    full_fitted: list[np.ndarray] = []
    full_residual: list[np.ndarray] = []
    for item in design_plan.bins:
        local_y = canonical_outcomes[item.available_donor_indices, item.bin_index]
        observed_fit = _fit_qr(
            item.full_design,
            local_y,
            item.full_design.shape[1] - 1,
            support[item.bin_index],
        )
        fitted_zero, residual_zero = _fit_reduced(item.reduced_design, local_y)
        effect[item.bin_index] = observed_fit.coefficient
        standard_error[item.bin_index] = observed_fit.standard_error
        studentized[item.bin_index] = observed_fit.studentized
        reduced_fitted.append(fitted_zero)
        reduced_residual.append(residual_zero)
        full_fitted.append(observed_fit.fitted)
        full_residual.append(observed_fit.residual)

    null_effect = np.empty((n_reference, n_bins, n_pathways), dtype=float)
    null_t = np.empty_like(null_effect)
    bootstrap_effect = np.empty_like(null_effect)
    bootstrap_t = np.empty_like(null_effect)
    for mapping_index, mapping in enumerate(mapping_plan.mappings):
        for item in design_plan.bins:
            indices = item.available_donor_indices
            global_to_local = np.full(len(design_plan.donor_ids), -1, dtype=int)
            global_to_local[indices] = np.arange(len(indices))
            source_local = global_to_local[mapping[indices]]
            if bool((source_local < 0).any()):
                raise RuntimeError("Mapping violated full availability signatures")
            null_y = reduced_fitted[item.bin_index] + reduced_residual[
                item.bin_index
            ][source_local]
            null_fit = _fit_qr(
                item.full_design,
                null_y,
                item.full_design.shape[1] - 1,
                support[item.bin_index],
            )
            null_effect[mapping_index, item.bin_index] = null_fit.coefficient
            null_t[mapping_index, item.bin_index] = null_fit.studentized
            bootstrap_y = full_fitted[item.bin_index] + full_residual[
                item.bin_index
            ][source_local]
            bootstrap_fit = _fit_qr(
                item.full_design,
                bootstrap_y,
                item.full_design.shape[1] - 1,
                support[item.bin_index],
            )
            bootstrap_effect[mapping_index, item.bin_index] = (
                bootstrap_fit.coefficient
            )
            bootstrap_t[mapping_index, item.bin_index] = _studentize(
                bootstrap_fit.coefficient - effect[item.bin_index],
                bootstrap_fit.standard_error,
                support[item.bin_index],
            )

    absolute_observed_t = np.abs(studentized)
    absolute_null_t = np.abs(null_t)
    pointwise_p = _plus_one_p(absolute_null_t, absolute_observed_t)
    curve_statistic = np.asarray(
        [
            np.max(absolute_observed_t[support[:, index], index])
            for index in range(n_pathways)
        ]
    )
    null_curve = np.asarray(
        [
            np.max(absolute_null_t[:, support[:, index], index], axis=1)
            for index in range(n_pathways)
        ]
    ).T
    curve_p_raw = _plus_one_p(null_curve, curve_statistic)
    # Each pathway statistic is already maxT over its supported bins.  The
    # all-pathway maximum is reserved for the simultaneous band reference.
    curve_p_max_t = curve_p_raw.copy()
    curve_q_by = _by_adjust(curve_p_raw)
    integrated_effect = np.asarray(
        [
            np.sum(np.abs(effect[support[:, index], index]) * weights[support[:, index]])
            for index in range(n_pathways)
        ]
    )
    integrated_statistic = integrated_effect.copy()
    null_integrated = np.asarray(
        [
            np.sum(
                np.abs(null_effect[:, support[:, index], index])
                * weights[support[:, index]][None, :],
                axis=1,
            )
            for index in range(n_pathways)
        ]
    ).T
    integrated_p_raw = _plus_one_p(null_integrated, integrated_statistic)
    integrated_global = np.max(null_integrated, axis=1)
    integrated_p_max_t = _plus_one_p(
        integrated_global[:, None], integrated_statistic
    )
    integrated_q_by = _by_adjust(integrated_p_raw)

    order_index = min(
        n_reference,
        int(math.ceil((n_reference + 1) * (1.0 - alpha_value))),
    )
    absolute_bootstrap_t = np.abs(bootstrap_t)
    pointwise_critical = _order_critical(absolute_bootstrap_t, order_index)
    global_bootstrap_max = np.max(absolute_bootstrap_t[:, support], axis=1)
    simultaneous_critical = float(
        _order_critical(global_bootstrap_max, order_index)
    )
    pointwise_lower = effect - pointwise_critical * standard_error
    pointwise_upper = effect + pointwise_critical * standard_error
    simultaneous_lower = effect - simultaneous_critical * standard_error
    simultaneous_upper = effect + simultaneous_critical * standard_error

    # Unsupported pathway/bin points remain visible as explicit missing values and
    # cannot leak into any statistic or band family.
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

    family_rows: list[Mapping[str, Any]] = []
    family_p_max_t = curve_p_max_t.copy()
    if families is not None:
        family_order = tuple(dict.fromkeys(families))
        for family in family_order:
            members = np.asarray([item == family for item in families])
            family_null_max = np.max(null_curve[:, members], axis=1)
            family_p_max_t[members] = _plus_one_p(
                family_null_max[:, None], curve_statistic[members]
            )
            observed_family = float(np.max(curve_statistic[members]))
            family_rows.append(
                {
                    "family_id": family,
                    "n_pathways": int(members.sum()),
                    "observed_statistic": observed_family,
                    "p_family": float(
                        _plus_one_p(
                            family_null_max[:, None], np.asarray([observed_family])
                        )[0]
                    ),
                }
            )

    metadata = {
        "method": "trajpathmix_functional_core_v1",
        "test_reference": "reduced_residual_freedman_lane",
        "band_reference": "full_residual_bootstrap",
        "mapping_scope": "whole_donor_same_mapping_all_bins_and_pathways",
        "band_scope": "global_all_pathways_by_supported_bins",
        "identity_in_null": False,
        "plus_one_p_values": True,
        "rank_relative_tolerance": RANK_RELATIVE_TOLERANCE,
        "no_intercept": True,
        "functional_fields_only": True,
    }
    result = FunctionalCoreResult(
        donor_ids=design_plan.donor_ids,
        pathway_ids=pathways,
        support_mask=_freeze_array(support, dtype=bool),
        effect=_freeze_array(effect, dtype=float),
        standard_error=_freeze_array(standard_error, dtype=float),
        studentized_effect=_freeze_array(studentized, dtype=float),
        null_effect=_freeze_array(null_effect, dtype=float),
        null_studentized_effect=_freeze_array(null_t, dtype=float),
        bootstrap_effect=_freeze_array(bootstrap_effect, dtype=float),
        bootstrap_studentized_deviation=_freeze_array(bootstrap_t, dtype=float),
        pointwise_p=_freeze_array(pointwise_p, dtype=float),
        pointwise_critical=_freeze_array(pointwise_critical, dtype=float),
        pointwise_lower=_freeze_array(pointwise_lower, dtype=float),
        pointwise_upper=_freeze_array(pointwise_upper, dtype=float),
        simultaneous_critical=simultaneous_critical,
        simultaneous_lower=_freeze_array(simultaneous_lower, dtype=float),
        simultaneous_upper=_freeze_array(simultaneous_upper, dtype=float),
        band_order_index_1based=order_index,
        curve_statistic=_freeze_array(curve_statistic, dtype=float),
        curve_p_raw=_freeze_array(curve_p_raw, dtype=float),
        curve_p_maxT=_freeze_array(curve_p_max_t, dtype=float),
        curve_q_by=_freeze_array(curve_q_by, dtype=float),
        integrated_absolute_effect=_freeze_array(integrated_effect, dtype=float),
        integrated_studentized_statistic=_freeze_array(
            integrated_statistic, dtype=float
        ),
        integrated_p_raw=_freeze_array(integrated_p_raw, dtype=float),
        integrated_p_maxT=_freeze_array(integrated_p_max_t, dtype=float),
        integrated_q_by=_freeze_array(integrated_q_by, dtype=float),
        family_p_maxT=_freeze_array(family_p_max_t, dtype=float),
        family_tests=tuple(_json_safe(item) for item in family_rows),
        mapping_plan=mapping_plan,
        design_plan=design_plan,
        inference_metadata=metadata,
    )
    # Exercise the strict key firewall before returning a result that could be
    # persisted by a caller.
    _assert_functional_only_keys(result.to_dict())
    return result


def write_functional_core_result_json(
    result: FunctionalCoreResult,
    path: str | Path,
    *,
    create_only: bool = True,
) -> dict[str, Any]:
    """Atomically write the strict functional-only result JSON."""

    if not isinstance(result, FunctionalCoreResult):
        raise TypeError("result must be a FunctionalCoreResult")
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if create_only and target.exists():
        raise FileExistsError(f"Functional-core output exists: {target}")
    payload = result.to_dict()
    _assert_functional_only_keys(payload)
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if create_only and target.exists():
            raise FileExistsError(f"Functional-core output exists: {target}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return {
        "path": str(target),
        "sha256": digest,
        "bytes": len(encoded.encode("utf-8")),
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "functional_fields_only": True,
    }


__all__ = [
    "AvailabilityMappingPlan",
    "BinSpecificDesign",
    "BinSpecificDesignPlan",
    "DEFAULT_MAPPING_SEED",
    "FUNCTIONAL_CORE_RESULT_SCHEMA",
    "FunctionalCoreDesignError",
    "FunctionalCoreResult",
    "RANK_RELATIVE_TOLERANCE",
    "build_bin_specific_designs",
    "build_full_availability_mapping_plan",
    "functional_core_result_json_schema",
    "run_functional_core",
    "write_functional_core_result_json",
]
