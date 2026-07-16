"""Outcome-blind trajectory and fate construction for the T21 data product.

The implementation is intentionally a reconstruction of the public analysis
*structure*, not a claim that the author's MIRA/FLE coordinates or CellRank
GPCCA result have been reproduced.  A donor-balanced reference sketch is used
to keep graph construction bounded, while every cell in the input H5AD remains
on the output axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Callable, Iterable, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import (
    connected_components,
    dijkstra,
    minimum_spanning_tree,
)
from scipy.spatial.distance import cdist
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import yaml

from .t21_data_product import (
    FINAL_PRODUCT_NAMES,
    FATE_PROBABILITY_COLUMNS,
    REQUIRED_DONOR_DESIGN_COLUMNS,
    cell_id_set_hash,
    sha256_file,
    stable_json,
    tree_digest,
    utc_now,
    validate_donor_design,
    validate_fate_probabilities,
    validate_trajectory_scrna_alignment,
    validate_trajectory_zarr,
    write_fate_probabilities,
    write_trajectory_zarr,
)


PLAN_SCHEMA_NAME = "t21_outcome_blind_trajectory_and_fate_plan"
PLAN_SCHEMA_VERSION = "1.0.0"
IMPLEMENTATION_ID = "t21_outcome_blind_reference_dpt_and_dag_absorption_v1"
APPROXIMATE_NEIGHBOR_THRESHOLD = 4096
INFERENCE_OBS_COLUMNS = (
    "donor_id",
    "analysis_cell_type",
    "lineage_inclusion",
    "lineage_inclusion_reason",
)
FORBIDDEN_INFERENCE_OBS_COLUMNS = frozenset(
    {"condition", "condition_original", "sample", "disease", "genotype"}
)
FATE_ORDER = ("erythroid", "megakaryocyte", "myeloid", "other")
BUILD_RECORD_NAMES = (
    "t21_trajectory_fate_build_record_v1.json",
    "t21_trajectory_build_record_v1.json",
    "t21_fate_build_record_v1.json",
    "t21_donor_design_build_record_v1.json",
)


@dataclass(frozen=True)
class TrajectoryFateInference:
    """In-memory products and diagnostics prior to artifact publication."""

    cell_ids: tuple[str, ...]
    draw_ids: tuple[str, ...]
    donor_ids: tuple[str, ...]
    pseudotime: np.ndarray
    mapped: np.ndarray
    bin_left: np.ndarray
    bin_center: np.ndarray
    bin_right: np.ndarray
    donor_bin_cell_count: np.ndarray
    donor_bin_available: np.ndarray
    draw_metadata: tuple[dict[str, Any], ...]
    fate_probabilities: pd.DataFrame
    primary_draw_id: str
    fate_draw_diagnostics: tuple[dict[str, Any], ...]
    representation_diagnostics: dict[str, Any]


def _sha256_bytes(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return sha256(array.view(np.uint8)).hexdigest()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _require_sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    return value


def _nonempty_unique_strings(values: Any, label: str) -> list[str]:
    sequence = _require_sequence(values, label)
    normalized = [str(value).strip() for value in sequence]
    if any(not value for value in normalized) or len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} must contain unique non-empty strings")
    return normalized


def _validate_grid(grid: Mapping[str, Any]) -> None:
    left = np.asarray(grid.get("bin_left"), dtype=float)
    center = np.asarray(grid.get("bin_center"), dtype=float)
    right = np.asarray(grid.get("bin_right"), dtype=float)
    if not (
        left.ndim == center.ndim == right.ndim == 1
        and len(left) == len(center) == len(right)
        and len(left) > 0
        and np.all(np.isfinite(left))
        and np.all(np.isfinite(center))
        and np.all(np.isfinite(right))
        and np.all((0 <= left) & (left < center))
        and np.all((center < right) & (right <= 1))
        and np.isclose(left[0], 0.0)
        and np.isclose(right[-1], 1.0)
        and np.allclose(left[1:], right[:-1], atol=1e-12, rtol=0)
    ):
        raise ValueError("The frozen common pseudotime grid is invalid")


def validate_trajectory_fate_plan(
    plan: Mapping[str, Any], *, repository_root: str | Path | None = None
) -> dict[str, Any]:
    """Validate the frozen plan and, when requested, its public-code binding."""
    if plan.get("schema_name") != PLAN_SCHEMA_NAME:
        raise ValueError("Unexpected trajectory/fate plan schema_name")
    if str(plan.get("schema_version")) != PLAN_SCHEMA_VERSION:
        raise ValueError("Unexpected trajectory/fate plan schema_version")
    if plan.get("outcome_blinded_at_freeze") is not True:
        raise ValueError("Trajectory/fate plan was not frozen outcome-blind")
    if plan.get("real_pathway_outcomes_inspected") is not False:
        raise ValueError("Trajectory/fate plan indicates prior pathway outcome inspection")

    scope = _require_mapping(plan.get("scope"), "scope")
    if scope.get("condition_used_for_inference") is not False:
        raise ValueError("Condition information is forbidden for trajectory inference")
    if scope.get("candidate_pathway_genes_used_for_inference") is not False:
        raise ValueError("Candidate pathway genes are forbidden for trajectory inference")
    if scope.get("retain_every_h5ad_cell") is not True:
        raise ValueError("The plan must retain every H5AD cell")

    representation = _require_mapping(
        plan.get("feature_representation"), "feature_representation"
    )
    feature_obs = set(
        _nonempty_unique_strings(
            representation.get("obs_columns_used_for_feature_values", []),
            "feature_representation.obs_columns_used_for_feature_values",
        )
    )
    if feature_obs.intersection(FORBIDDEN_INFERENCE_OBS_COLUMNS):
        raise ValueError("Condition-like obs columns may not be expression features")
    if feature_obs:
        raise ValueError("This plan permits no obs-derived expression features")
    stratification = set(
        _nonempty_unique_strings(
            representation.get("obs_columns_used_for_stratification"),
            "feature_representation.obs_columns_used_for_stratification",
        )
    )
    allowed_stratification = {"donor_id", "analysis_cell_type", "lineage_inclusion"}
    if not stratification.issubset(allowed_stratification):
        raise ValueError("The plan contains a forbidden stratification column")
    if str(representation.get("matrix")) != "layers/counts":
        raise ValueError("Trajectory inference must use the pinned counts layer")
    if representation.get("neighbor_backend") != (
        "deterministic_pynndescent_above_4096_reference_occurrences_"
        "else_exact_ball_tree"
    ):
        raise ValueError("The scalable nearest-neighbor backend is not frozen")
    for field in ("n_top_variable_features", "n_components", "mapping_chunk_size"):
        if int(representation.get(field, 0)) <= 0:
            raise ValueError(f"feature_representation.{field} must be positive")

    roots = _require_mapping(plan.get("root_definitions"), "root_definitions")
    terminals = _require_mapping(
        plan.get("terminal_definitions"), "terminal_definitions"
    )
    for root_id, definition in roots.items():
        root = _require_mapping(definition, f"root_definitions.{root_id}")
        _nonempty_unique_strings(
            root.get("analysis_cell_types"),
            f"root_definitions.{root_id}.analysis_cell_types",
        )
        if root.get("anchor_rule") not in {
            "farthest_from_reference_global_median_within_root_labels",
            "nearest_to_root_label_centroid",
        }:
            raise ValueError(f"Unknown root anchor rule for {root_id}")
    for terminal_id, definition in terminals.items():
        terminal = _require_mapping(definition, f"terminal_definitions.{terminal_id}")
        quantile = float(terminal.get("within_fate_pseudotime_quantile", np.nan))
        if not np.isfinite(quantile) or not 0 <= quantile < 1:
            raise ValueError(f"Invalid terminal boundary quantile for {terminal_id}")
        fate_types = _require_mapping(
            terminal.get("fate_analysis_cell_types"),
            f"terminal_definitions.{terminal_id}.fate_analysis_cell_types",
        )
        if tuple(fate_types) != FATE_ORDER:
            raise ValueError("Terminal fate mapping must use the frozen four-fate order")
        seen: set[str] = set()
        for fate in FATE_ORDER:
            labels = _nonempty_unique_strings(
                fate_types[fate],
                f"terminal_definitions.{terminal_id}.{fate}",
            )
            overlap = seen.intersection(labels)
            if overlap:
                raise ValueError(f"Terminal cell types overlap across fates: {overlap}")
            seen.update(labels)

    fate_model = _require_mapping(plan.get("fate_model"), "fate_model")
    if tuple(fate_model.get("fate_order", [])) != FATE_ORDER:
        raise ValueError("fate_model.fate_order differs from the fixed four-fate order")
    if str(fate_model.get("transition_rule")) != "increasing_pseudotime_knn_edges_only":
        raise ValueError("The fate model must use increasing-pseudotime transitions")
    teleport = float(fate_model.get("terminal_teleport_weight", np.nan))
    if not np.isfinite(teleport) or not 0 < teleport <= 1:
        raise ValueError("terminal_teleport_weight must be in (0, 1]")

    grid = _require_mapping(
        plan.get("fixed_common_pseudotime_grid"),
        "fixed_common_pseudotime_grid",
    )
    _validate_grid(grid)

    required_sources = set(
        _nonempty_unique_strings(
            plan.get("required_uncertainty_sources"),
            "required_uncertainty_sources",
        )
    )
    draws = _require_sequence(plan.get("draws"), "draws")
    if not draws:
        raise ValueError("At least one trajectory draw is required")
    draw_ids: list[str] = []
    observed_sources: set[str] = set()
    primary_ids: list[str] = []
    for index, value in enumerate(draws):
        draw = _require_mapping(value, f"draws[{index}]")
        draw_id = str(draw.get("trajectory_draw_id", "")).strip()
        if not draw_id or draw_id in draw_ids:
            raise ValueError("Trajectory draw IDs must be unique and non-empty")
        draw_ids.append(draw_id)
        if draw.get("is_primary") is True:
            primary_ids.append(draw_id)
        observed_sources.add(str(draw.get("uncertainty_source", "")))
        if draw.get("method") not in {
            "scanpy_dpt_reference_mapping",
            "pca_knn_geodesic_reference_mapping",
        }:
            raise ValueError(f"Unknown trajectory method for draw {draw_id}")
        if (
            draw.get("uncertainty_source") == "second_method"
            and draw.get("method") != "pca_knn_geodesic_reference_mapping"
        ):
            raise ValueError("The second-method draw must use the frozen geodesic method")
        if str(draw.get("root_definition_id")) not in roots:
            raise ValueError(f"Unknown root definition for draw {draw_id}")
        if str(draw.get("terminal_definition_id")) not in terminals:
            raise ValueError(f"Unknown terminal definition for draw {draw_id}")
        if str(draw.get("reference_scheme")) not in {
            "primary",
            "resampled",
            "donor_bootstrap",
        }:
            raise ValueError(f"Unknown reference scheme for draw {draw_id}")
        if int(draw.get("n_neighbors", 0)) < 2:
            raise ValueError(f"n_neighbors must be at least two for draw {draw_id}")
        if int(draw.get("mapping_neighbors", 0)) < 1:
            raise ValueError(f"mapping_neighbors must be positive for draw {draw_id}")
    if len(primary_ids) != 1:
        raise ValueError("Exactly one trajectory draw must be primary")
    if str(fate_model.get("output_draw")) != primary_ids[0]:
        raise ValueError("The fate output draw must be the primary trajectory draw")
    missing_sources = required_sources.difference(observed_sources)
    if missing_sources:
        raise ValueError(f"Trajectory draws omit uncertainty sources: {sorted(missing_sources)}")

    public_binding = _require_mapping(
        plan.get("public_method_binding"), "public_method_binding"
    )
    if public_binding.get("role") != "structural_precedent_not_exact_reproduction":
        raise ValueError("Public-method role must forbid an exact-reproduction claim")
    if repository_root is not None and public_binding.get("required_local_evidence") is True:
        source = Path(repository_root) / str(public_binding["repository_relative_path"])
        if not source.is_file():
            raise FileNotFoundError(f"Pinned public method source is missing: {source}")
        expected_hash = str(public_binding.get("sha256", "")).lower()
        if sha256_file(source) != expected_hash:
            raise ValueError("Pinned public method source SHA256 differs from the plan")

    return {
        "plan_id": str(plan.get("plan_id")),
        "primary_draw_id": primary_ids[0],
        "draw_ids": draw_ids,
        "required_uncertainty_sources": sorted(required_sources),
    }


def load_trajectory_fate_plan(
    path: str | Path, *, repository_root: str | Path | None = None
) -> dict[str, Any]:
    """Load and validate a frozen trajectory/fate YAML plan."""
    path = Path(path)
    plan = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("Trajectory/fate plan must contain a YAML mapping")
    validate_trajectory_fate_plan(plan, repository_root=repository_root)
    return plan


def _strict_boolean(values: pd.Series, label: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values.dtype):
        if values.isna().any():
            raise ValueError(f"{label} may not contain missing values")
        return values.to_numpy(dtype=bool)
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError(f"{label} must contain only true/false")
    return normalized.eq("true").to_numpy(dtype=bool)


def _inference_obs(adata: ad.AnnData) -> pd.DataFrame:
    """Copy only the predeclared outcome-blind obs columns.

    In particular, this function never selects or materializes ``condition``.
    """
    missing = sorted(set(INFERENCE_OBS_COLUMNS).difference(adata.obs.columns))
    if missing:
        raise ValueError(f"H5AD lacks trajectory inference obs columns: {missing}")
    obs = adata.obs.loc[:, list(INFERENCE_OBS_COLUMNS)].copy()
    if not adata.obs_names.is_unique:
        raise ValueError("H5AD cell IDs must be unique")
    obs.index = pd.Index(adata.obs_names.astype(str), name="cell_id")
    for column in ("donor_id", "analysis_cell_type", "lineage_inclusion_reason"):
        if obs[column].isna().any() or obs[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Trajectory inference field {column!r} must be non-empty")
        obs[column] = obs[column].astype(str)
    obs["lineage_inclusion"] = _strict_boolean(
        obs["lineage_inclusion"], "obs.lineage_inclusion"
    )
    if not obs["lineage_inclusion"].any():
        raise ValueError("No cells pass the frozen lineage rule")
    return obs


def _counts_rows(
    adata: ad.AnnData,
    row_indices: np.ndarray,
    feature_indices: np.ndarray | None = None,
) -> sparse.csr_matrix:
    """Read selected count rows while bounding backed-H5AD memory use."""
    if "counts" not in adata.layers:
        raise ValueError('Trajectory inference requires layers["counts"]')
    rows = np.asarray(row_indices, dtype=np.int64)
    if rows.ndim != 1 or not len(rows):
        raise ValueError("Count-row selection may not be empty")
    unique_rows, inverse = np.unique(rows, return_inverse=True)
    columns: slice | np.ndarray = (
        slice(None)
        if feature_indices is None
        else np.asarray(feature_indices, dtype=np.int64)
    )
    if getattr(adata, "isbacked", False):
        view = adata[unique_rows, columns].to_memory()
        counts = view.layers["counts"]
    else:
        counts = adata.layers["counts"][unique_rows]
        if feature_indices is not None:
            counts = counts[:, np.asarray(feature_indices, dtype=np.int64)]
    if not sparse.issparse(counts):
        counts = sparse.csr_matrix(np.asarray(counts))
    counts = sparse.csr_matrix(counts)[inverse]
    if counts.data.size:
        if np.any(~np.isfinite(counts.data)) or np.any(counts.data < 0):
            raise ValueError("Counts contain negative or non-finite values")
        if not np.array_equal(counts.data, np.rint(counts.data)):
            raise ValueError("Counts are not integer-valued")
    return counts


def _normalize_log1p(counts: sparse.csr_matrix, target_sum: float) -> sparse.csr_matrix:
    result = sparse.csr_matrix(counts, dtype=np.float64, copy=True)
    totals = np.asarray(result.sum(axis=1)).ravel()
    scaling = np.divide(
        target_sum,
        totals,
        out=np.zeros_like(totals, dtype=float),
        where=totals > 0,
    )
    result = sparse.diags(scaling).dot(result).tocsr()
    result.data = np.log1p(result.data)
    return result


def _balanced_reference_indices(
    obs: pd.DataFrame,
    eligible_indices: np.ndarray,
    *,
    seed: int,
    max_per_donor_type: int,
    max_per_donor: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eligible = obs.iloc[eligible_indices].copy()
    eligible["_global_index"] = eligible_indices
    selected: list[int] = []
    for _donor_id, donor_group in eligible.groupby("donor_id", sort=True):
        donor_selected: list[int] = []
        for _label, label_group in donor_group.groupby("analysis_cell_type", sort=True):
            candidates = label_group["_global_index"].to_numpy(dtype=np.int64)
            if len(candidates) > max_per_donor_type:
                candidates = rng.choice(
                    candidates, size=max_per_donor_type, replace=False
                )
            donor_selected.extend(np.sort(candidates).tolist())
        donor_array = np.asarray(donor_selected, dtype=np.int64)
        if len(donor_array) > max_per_donor:
            donor_array = rng.choice(donor_array, size=max_per_donor, replace=False)
        selected.extend(np.sort(donor_array).tolist())
    result = np.asarray(sorted(set(selected)), dtype=np.int64)
    if not len(result):
        raise ValueError("The donor-balanced reference sketch is empty")
    return result


def _required_analysis_labels(plan: Mapping[str, Any]) -> tuple[str, ...]:
    labels: set[str] = set()
    for root in plan["root_definitions"].values():
        labels.update(map(str, root["analysis_cell_types"]))
    for terminal in plan["terminal_definitions"].values():
        for fate_labels in terminal["fate_analysis_cell_types"].values():
            labels.update(map(str, fate_labels))
    return tuple(sorted(labels))


def _add_required_label_anchors(
    reference_indices: np.ndarray,
    primary_reference_indices: np.ndarray,
    labels: np.ndarray,
    required_labels: Iterable[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    present = set(labels[reference_indices])
    additions: list[int] = []
    anchored: list[str] = []
    for label in sorted(set(map(str, required_labels)).difference(present)):
        candidates = primary_reference_indices[labels[primary_reference_indices] == label]
        if not len(candidates):
            raise ValueError(
                f"The primary reference lacks required analysis_cell_type {label!r}"
            )
        additions.append(int(candidates[0]))
        anchored.append(label)
    if additions:
        reference_indices = np.concatenate(
            [reference_indices, np.asarray(additions, dtype=np.int64)]
        )
    return reference_indices, tuple(anchored)


def _select_variable_features(
    normalized_reference: sparse.csr_matrix,
    var_names: Sequence[str],
    n_top: int,
) -> tuple[np.ndarray, np.ndarray]:
    means = np.asarray(normalized_reference.mean(axis=0)).ravel()
    squares = np.asarray(normalized_reference.power(2).mean(axis=0)).ravel()
    variances = np.maximum(squares - means**2, 0.0)
    valid = np.flatnonzero(np.isfinite(variances) & (variances > 0))
    if len(valid) < 2:
        raise ValueError("Fewer than two variable expression features are available")
    names = np.asarray(list(map(str, var_names)), dtype=object)
    name_order = np.argsort(names[valid], kind="stable")
    valid = valid[name_order]
    variance_order = np.argsort(-variances[valid], kind="stable")
    selected = valid[variance_order[: min(n_top, len(valid))]]
    return selected.astype(np.int64), variances[selected]


def _fit_representation(
    adata: ad.AnnData,
    obs: pd.DataFrame,
    eligible_indices: np.ndarray,
    primary_reference_indices: np.ndarray,
    plan: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    representation = plan["feature_representation"]
    target_sum = float(representation["normalize_total"])
    reference_counts = _counts_rows(adata, primary_reference_indices)
    reference_normalized = _normalize_log1p(reference_counts, target_sum)
    feature_indices, feature_variances = _select_variable_features(
        reference_normalized,
        adata.var_names.astype(str),
        int(representation["n_top_variable_features"]),
    )
    reference_selected = reference_normalized[:, feature_indices]
    n_components = min(
        int(representation["n_components"]),
        reference_selected.shape[0] - 1,
        reference_selected.shape[1] - 1,
    )
    if n_components < 2:
        raise ValueError("Reference sketch is too small to fit a trajectory representation")
    svd = TruncatedSVD(
        n_components=n_components,
        random_state=int(representation["svd_random_state"]),
    )
    reference_embedding = svd.fit_transform(reference_selected)
    center = reference_embedding.mean(axis=0)
    scale = reference_embedding.std(axis=0)
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0

    chunk_size = int(representation["mapping_chunk_size"])
    eligible_embedding = np.empty((len(eligible_indices), n_components), dtype=np.float32)
    for start in range(0, len(eligible_indices), chunk_size):
        stop = min(start + chunk_size, len(eligible_indices))
        counts = _counts_rows(
            adata, eligible_indices[start:stop], feature_indices=feature_indices
        )
        normalized = _normalize_log1p(counts, target_sum)
        transformed = (svd.transform(normalized) - center) / scale
        if np.any(~np.isfinite(transformed)):
            raise ValueError("Non-finite values arose in the reference mapping embedding")
        eligible_embedding[start:stop] = transformed.astype(np.float32)

    selected_names = adata.var_names.astype(str).to_numpy()[feature_indices]
    feature_model_hash = sha256(
        stable_json(
            {
                "feature_names": selected_names.tolist(),
                "target_sum": target_sum,
                "n_components": n_components,
                "svd_random_state": int(representation["svd_random_state"]),
                "component_scaling": representation["component_scaling"],
            }
        ).encode("utf-8")
        + np.ascontiguousarray(svd.components_, dtype=np.float64).tobytes()
        + np.ascontiguousarray(center, dtype=np.float64).tobytes()
        + np.ascontiguousarray(scale, dtype=np.float64).tobytes()
    ).hexdigest()
    diagnostics = {
        "n_h5ad_cells": int(adata.n_obs),
        "n_lineage_cells": int(len(eligible_indices)),
        "n_primary_reference_cells": int(len(primary_reference_indices)),
        "n_selected_features": int(len(feature_indices)),
        "selected_feature_order_hash": sha256(
            "\n".join(selected_names).encode("utf-8")
        ).hexdigest(),
        "selected_feature_variance_sha256": _sha256_bytes(
            np.asarray(feature_variances, dtype=np.float64)
        ),
        "n_components": int(n_components),
        "feature_model_hash": feature_model_hash,
        "condition_column_read_for_inference": False,
        "candidate_pathway_artifact_read_for_inference": False,
        "obs_columns_read_for_inference": list(INFERENCE_OBS_COLUMNS),
        "donor_set_hash": cell_id_set_hash(sorted(set(obs["donor_id"].astype(str)))),
    }
    return eligible_embedding, diagnostics


def _draw_reference_indices(
    draw: Mapping[str, Any],
    obs: pd.DataFrame,
    eligible_indices: np.ndarray,
    primary_reference_indices: np.ndarray,
    plan: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    scheme = str(draw["reference_scheme"])
    reference_plan = plan["reference_sketch"]
    seed = int(draw["seed"])
    if scheme == "primary":
        indices = primary_reference_indices.copy()
        details: dict[str, Any] = {"reference_scheme": "primary"}
    elif scheme == "resampled":
        indices = _balanced_reference_indices(
            obs,
            eligible_indices,
            seed=seed,
            max_per_donor_type=int(
                reference_plan["max_cells_per_donor_analysis_cell_type"]
            ),
            max_per_donor=int(reference_plan["max_cells_per_donor"]),
        )
        details = {"reference_scheme": "resampled", "reference_seed": seed}
    elif scheme == "donor_bootstrap":
        donor_by_index = obs["donor_id"].astype(str).to_numpy()
        donors = np.asarray(
            sorted(set(donor_by_index[primary_reference_indices])), dtype=object
        )
        rng = np.random.default_rng(seed)
        sampled = rng.choice(donors, size=len(donors), replace=True)
        multiplicity = {
            str(donor): int(np.sum(sampled == donor)) for donor in donors
        }
        occurrences = [
            primary_reference_indices[
                donor_by_index[primary_reference_indices] == str(donor)
            ]
            for donor in sampled
        ]
        indices = np.concatenate(occurrences).astype(np.int64)
        details = {
            "reference_scheme": "donor_bootstrap",
            "reference_seed": seed,
            "donor_bootstrap_multiplicity": multiplicity,
        }
    else:  # pragma: no cover - plan validation closes this branch
        raise ValueError(f"Unknown reference scheme {scheme!r}")

    labels = obs["analysis_cell_type"].astype(str).to_numpy()
    indices, anchored_labels = _add_required_label_anchors(
        indices,
        primary_reference_indices,
        labels,
        _required_analysis_labels(plan),
    )
    details["anchored_missing_analysis_cell_types"] = list(anchored_labels)
    details["reference_occurrence_count"] = int(len(indices))
    details["reference_unique_cell_count"] = int(len(np.unique(indices)))
    details["reference_unique_cell_set_hash"] = cell_id_set_hash(
        obs.index.to_numpy()[np.unique(indices)]
    )
    details["reference_donor_ids"] = sorted(
        set(obs["donor_id"].astype(str).to_numpy()[indices])
    )
    return indices, details


def _reference_embedding(
    eligible_indices: np.ndarray,
    eligible_embedding: np.ndarray,
    reference_indices: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    positions = np.searchsorted(eligible_indices, reference_indices)
    if np.any(positions >= len(eligible_indices)) or not np.array_equal(
        eligible_indices[positions], reference_indices
    ):
        raise ValueError("Reference sketch contains a non-lineage cell")
    result = np.asarray(eligible_embedding[positions], dtype=np.float64).copy()
    if len(np.unique(reference_indices)) != len(reference_indices):
        rng = np.random.default_rng(seed)
        result += rng.normal(0.0, 1e-8, size=result.shape)
    return result


def _neighbor_query_factory(
    reference_embedding: np.ndarray,
    *,
    n_neighbors: int,
    seed: int,
):
    """Fit a deterministic bounded-reference nearest-neighbor query backend."""
    effective = min(max(1, n_neighbors), len(reference_embedding))
    if len(reference_embedding) >= APPROXIMATE_NEIGHBOR_THRESHOLD:
        from pynndescent import NNDescent

        index_neighbors = min(
            len(reference_embedding) - 1,
            max(30, effective * 2),
        )
        index = NNDescent(
            np.asarray(reference_embedding, dtype=np.float32),
            n_neighbors=index_neighbors,
            metric="euclidean",
            random_state=seed,
            low_memory=True,
            n_jobs=1,
        )
        index.prepare()

        def query(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            indices, distances = index.query(
                np.asarray(values, dtype=np.float32), k=effective
            )
            return np.asarray(distances, dtype=float), np.asarray(indices, dtype=np.int64)

        details = {
            "neighbor_backend": "pynndescent",
            "neighbor_backend_version": importlib_metadata.version("pynndescent"),
            "neighbor_backend_n_jobs": 1,
            "neighbor_backend_index_neighbors": int(index_neighbors),
        }
    else:
        model = NearestNeighbors(
            n_neighbors=effective,
            metric="euclidean",
            algorithm="ball_tree",
            leaf_size=40,
            n_jobs=1,
        )
        model.fit(reference_embedding)

        def query(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            distances, indices = model.kneighbors(values)
            return np.asarray(distances, dtype=float), np.asarray(indices, dtype=np.int64)

        details = {
            "neighbor_backend": "sklearn_exact_ball_tree",
            "neighbor_backend_version": importlib_metadata.version("scikit-learn"),
            "neighbor_backend_n_jobs": 1,
            "neighbor_backend_leaf_size": 40,
        }
    return query, effective, details


def _reference_neighbors_without_self(
    reference_embedding: np.ndarray,
    *,
    n_neighbors: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    effective = min(max(1, n_neighbors), len(reference_embedding) - 1)
    query, _query_neighbors, details = _neighbor_query_factory(
        reference_embedding,
        n_neighbors=effective + 1,
        seed=seed,
    )
    distances_with_self, neighbors_with_self = query(reference_embedding)
    neighbors = np.empty((len(reference_embedding), effective), dtype=np.int64)
    distances = np.empty((len(reference_embedding), effective), dtype=float)
    for row_index in range(len(reference_embedding)):
        keep = neighbors_with_self[row_index] != row_index
        row_neighbors = neighbors_with_self[row_index, keep][:effective]
        row_distances = distances_with_self[row_index, keep][:effective]
        if len(row_neighbors) < effective:
            raise ValueError("Nearest-neighbor backend returned too few non-self neighbors")
        neighbors[row_index] = row_neighbors
        distances[row_index] = row_distances
    return distances, neighbors, details


def _select_root_position(
    reference_embedding: np.ndarray,
    reference_labels: np.ndarray,
    reference_cell_ids: np.ndarray,
    definition: Mapping[str, Any],
) -> tuple[int, dict[str, Any]]:
    root_labels = set(map(str, definition["analysis_cell_types"]))
    candidates = np.flatnonzero(np.isin(reference_labels, sorted(root_labels)))
    if not len(candidates):
        raise ValueError("Reference sketch contains no frozen root-label cells")
    rule = str(definition["anchor_rule"])
    if rule == "farthest_from_reference_global_median_within_root_labels":
        target = np.median(reference_embedding, axis=0)
        distances = np.linalg.norm(reference_embedding[candidates] - target, axis=1)
        root_position = int(candidates[np.argmax(distances)])
    elif rule == "nearest_to_root_label_centroid":
        target = reference_embedding[candidates].mean(axis=0)
        distances = np.linalg.norm(reference_embedding[candidates] - target, axis=1)
        root_position = int(candidates[np.argmin(distances)])
    else:  # pragma: no cover - plan validation closes this branch
        raise ValueError(f"Unknown root rule {rule!r}")
    selected_id = str(reference_cell_ids[root_position])
    details = {
        "root_anchor_rule": rule,
        "root_selected_cell_id": selected_id,
        "root_cell_set_hash": cell_id_set_hash([selected_id]),
        "root_candidate_cell_set_hash": cell_id_set_hash(
            sorted(set(map(str, reference_cell_ids[candidates])))
        ),
        "n_root_candidates": int(len(set(reference_cell_ids[candidates]))),
    }
    return root_position, details


def _scanpy_dpt(
    reference_embedding: np.ndarray,
    *,
    root_position: int,
    n_neighbors: int,
    n_diffusion_components: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    import scanpy as sc

    n_reference = len(reference_embedding)
    effective_neighbors = min(max(2, n_neighbors), n_reference - 1)
    if effective_neighbors < 2:
        raise ValueError("Reference sketch is too small for a neighborhood graph")
    reference = ad.AnnData(X=np.asarray(reference_embedding, dtype=np.float32))
    reference.obsm["X_reference"] = np.asarray(reference_embedding, dtype=np.float32)
    sc.pp.neighbors(
        reference,
        n_neighbors=effective_neighbors,
        use_rep="X_reference",
        metric="euclidean",
        random_state=seed,
    )
    connectivities = sparse.csr_matrix(reference.obsp["connectivities"])
    n_components_before, component_labels = connected_components(
        connectivities, directed=False
    )
    bridge_records: list[dict[str, Any]] = []
    if n_components_before > 1:
        component_centers = np.vstack(
            [
                reference_embedding[component_labels == component].mean(axis=0)
                for component in range(n_components_before)
            ]
        )
        center_distances = cdist(component_centers, component_centers)
        tree = minimum_spanning_tree(center_distances).tocoo()
        connectivity_lil = connectivities.tolil(copy=True)
        distances_lil = sparse.csr_matrix(reference.obsp["distances"]).tolil(copy=True)
        existing_distances = np.asarray(reference.obsp["distances"].data, dtype=float)
        positive = existing_distances[existing_distances > 0]
        distance_scale = float(np.median(positive)) if positive.size else 1.0
        if not np.isfinite(distance_scale) or distance_scale <= 0:
            distance_scale = 1.0
        for left_component, right_component in zip(tree.row, tree.col):
            left_indices = np.flatnonzero(component_labels == int(left_component))
            right_indices = np.flatnonzero(component_labels == int(right_component))
            neighbor = NearestNeighbors(n_neighbors=1, metric="euclidean")
            neighbor.fit(reference_embedding[right_indices])
            pair_distances, pair_positions = neighbor.kneighbors(
                reference_embedding[left_indices]
            )
            left_local = int(np.argmin(pair_distances[:, 0]))
            left_index = int(left_indices[left_local])
            right_index = int(right_indices[int(pair_positions[left_local, 0])])
            distance = float(pair_distances[left_local, 0])
            weight = float(np.exp(-distance / distance_scale))
            weight = max(weight, np.finfo(float).eps)
            connectivity_lil[left_index, right_index] = weight
            connectivity_lil[right_index, left_index] = weight
            distances_lil[left_index, right_index] = distance
            distances_lil[right_index, left_index] = distance
            bridge_records.append(
                {
                    "left_reference_position": left_index,
                    "right_reference_position": right_index,
                    "distance": distance,
                    "connectivity_weight": weight,
                }
            )
        reference.obsp["connectivities"] = connectivity_lil.tocsr()
        reference.obsp["distances"] = distances_lil.tocsr()
    n_components_after = connected_components(
        sparse.csr_matrix(reference.obsp["connectivities"]), directed=False
    )[0]
    if n_components_after != 1:
        raise ValueError("Deterministic component bridging did not connect the DPT graph")
    n_components = min(
        max(3, int(n_diffusion_components) + 1),
        n_reference - 1,
    )
    reference.uns["iroot"] = int(root_position)
    sc.tl.diffmap(reference, n_comps=n_components, random_state=seed)
    n_dcs = min(max(2, int(n_diffusion_components)), n_components - 1)
    sc.tl.dpt(reference, n_dcs=n_dcs)
    pseudotime = pd.to_numeric(
        reference.obs["dpt_pseudotime"], errors="coerce"
    ).to_numpy(dtype=float)
    if np.any(~np.isfinite(pseudotime)):
        raise ValueError(
            "Scanpy DPT produced non-finite reference pseudotime; the frozen "
            "reference graph is disconnected"
        )
    minimum, maximum = float(pseudotime.min()), float(pseudotime.max())
    if maximum <= minimum:
        raise ValueError("Scanpy DPT produced a constant pseudotime")
    pseudotime = (pseudotime - minimum) / (maximum - minimum)
    return pseudotime, {
        "n_neighbors_effective": effective_neighbors,
        "n_diffusion_components_effective": n_dcs,
        "scanpy_version": importlib_metadata.version("scanpy"),
        "n_graph_components_before_bridge": int(n_components_before),
        "n_graph_components_after_bridge": int(n_components_after),
        "component_bridge_policy": (
            "deterministic_component_centroid_mst_nearest_cell_bridges"
            if bridge_records
            else "not_needed"
        ),
        "component_bridges": bridge_records,
    }


def _knn_geodesic_pseudotime(
    reference_embedding: np.ndarray,
    *,
    root_position: int,
    n_neighbors: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    n_reference = len(reference_embedding)
    effective_neighbors = min(max(2, n_neighbors), n_reference - 1)
    neighbor_distances, neighbor_indices, backend_details = (
        _reference_neighbors_without_self(
            reference_embedding,
            n_neighbors=effective_neighbors,
            seed=seed,
        )
    )
    rows = np.repeat(np.arange(n_reference, dtype=np.int64), effective_neighbors)
    graph = sparse.csr_matrix(
        (
            neighbor_distances.ravel(),
            (rows, neighbor_indices.ravel()),
        ),
        shape=(n_reference, n_reference),
    )
    graph = graph.maximum(graph.transpose()).tocsr()
    distances = np.asarray(
        dijkstra(graph, directed=False, indices=int(root_position)), dtype=float
    )
    finite = np.isfinite(distances)
    disconnected = int((~finite).sum())
    if not finite.any():
        raise ValueError("The geodesic reference graph has no finite root distances")
    if disconnected:
        euclidean = np.linalg.norm(
            reference_embedding - reference_embedding[root_position], axis=1
        )
        finite_max = float(distances[finite].max())
        euclidean_scale = float(np.max(euclidean[~finite]))
        if euclidean_scale <= 0:
            euclidean_scale = 1.0
        distances[~finite] = finite_max + euclidean[~finite] / euclidean_scale
    scale = float(np.quantile(distances, 0.99))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(distances.max())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("The geodesic method produced a constant pseudotime")
    pseudotime = np.clip(distances / scale, 0.0, 1.0)
    return pseudotime, {
        "n_neighbors_effective": effective_neighbors,
        "n_disconnected_reference_occurrences": disconnected,
        "geodesic_disconnected_policy": (
            "root_euclidean_tail" if disconnected else "not_needed"
        ),
        "scipy_version": importlib_metadata.version("scipy"),
        **backend_details,
    }


def _orient_reference_pseudotime(
    pseudotime: np.ndarray,
    labels: np.ndarray,
    root_definition: Mapping[str, Any],
    terminal_definition: Mapping[str, Any],
) -> tuple[np.ndarray, bool]:
    root_labels = set(map(str, root_definition["analysis_cell_types"]))
    terminal_labels: set[str] = set()
    for values in terminal_definition["fate_analysis_cell_types"].values():
        terminal_labels.update(map(str, values))
    root_values = pseudotime[np.isin(labels, sorted(root_labels))]
    terminal_values = pseudotime[np.isin(labels, sorted(terminal_labels))]
    if not len(root_values) or not len(terminal_values):
        raise ValueError("Root or terminal labels are absent during orientation")
    reverse = float(np.median(terminal_values)) < float(np.median(root_values))
    oriented = 1.0 - pseudotime if reverse else pseudotime.copy()
    minimum, maximum = float(oriented.min()), float(oriented.max())
    if maximum <= minimum:
        raise ValueError("Trajectory orientation produced a constant pseudotime")
    return np.clip((oriented - minimum) / (maximum - minimum), 0.0, 1.0), reverse


def _terminal_masks(
    pseudotime: np.ndarray,
    labels: np.ndarray,
    terminal_definition: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    quantile = float(terminal_definition["within_fate_pseudotime_quantile"])
    masks = np.zeros((len(labels), len(FATE_ORDER)), dtype=bool)
    details: dict[str, Any] = {
        "within_fate_pseudotime_quantile": quantile,
        "terminal_occurrence_counts": {},
    }
    for fate_index, fate in enumerate(FATE_ORDER):
        fate_labels = list(
            map(str, terminal_definition["fate_analysis_cell_types"][fate])
        )
        candidates = np.flatnonzero(np.isin(labels, fate_labels))
        if not len(candidates):
            raise ValueError(f"Reference sketch lacks terminal candidates for {fate}")
        threshold = float(np.quantile(pseudotime[candidates], quantile))
        selected = candidates[pseudotime[candidates] >= threshold - 1e-12]
        if not len(selected):  # pragma: no cover - quantile selection guarantees one
            selected = np.asarray([candidates[np.argmax(pseudotime[candidates])]])
        masks[selected, fate_index] = True
        details["terminal_occurrence_counts"][fate] = int(len(selected))
    if np.any(masks.sum(axis=1) > 1):
        raise ValueError("A reference occurrence belongs to multiple terminal fates")
    details["n_terminal_occurrences"] = int(masks.any(axis=1).sum())
    return masks, details


def _terminal_distance_prior(
    embedding: np.ndarray, terminal_masks: np.ndarray
) -> np.ndarray:
    distances = np.empty((len(embedding), len(FATE_ORDER)), dtype=float)
    for fate_index in range(len(FATE_ORDER)):
        terminal_embedding = embedding[terminal_masks[:, fate_index]]
        if not len(terminal_embedding):
            raise ValueError(f"No terminal reference cells exist for {FATE_ORDER[fate_index]}")
        model = NearestNeighbors(
            n_neighbors=1,
            metric="euclidean",
            algorithm="ball_tree",
            leaf_size=40,
            n_jobs=1,
        )
        model.fit(terminal_embedding)
        distances[:, fate_index] = model.kneighbors(
            embedding, return_distance=True
        )[0][:, 0]
    positive = distances[distances > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    logits = -distances / scale
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def _directed_absorption_probabilities(
    reference_embedding: np.ndarray,
    reference_pseudotime: np.ndarray,
    reference_labels: np.ndarray,
    terminal_definition: Mapping[str, Any],
    *,
    n_neighbors: int,
    minimum_forward_delta: float,
    soft_threshold_nu: float,
    terminal_teleport_weight: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Approximate absorbing probabilities on an acyclic forward-time graph.

    Processing reference occurrences from late to early makes every retained
    transition point to an already-solved state.  A small terminal-distance
    teleport term makes disconnected local branches auditable and normalized.
    """
    terminal_masks, terminal_details = _terminal_masks(
        reference_pseudotime, reference_labels, terminal_definition
    )
    prior = _terminal_distance_prior(reference_embedding, terminal_masks)
    effective_neighbors = min(max(2, n_neighbors), len(reference_embedding) - 1)
    distances, neighbors, backend_details = _reference_neighbors_without_self(
        reference_embedding,
        n_neighbors=effective_neighbors,
        seed=seed,
    )
    probabilities = np.zeros((len(reference_embedding), len(FATE_ORDER)), dtype=float)
    terminal_any = terminal_masks.any(axis=1)
    probabilities[terminal_any] = terminal_masks[terminal_any].astype(float)
    fallback_count = 0
    order = np.argsort(-reference_pseudotime, kind="stable")
    for index in order:
        if terminal_any[index]:
            continue
        candidate_neighbors = neighbors[index]
        candidate_distances = distances[index]
        forward = (
            reference_pseudotime[candidate_neighbors]
            > reference_pseudotime[index] + minimum_forward_delta
        )
        solved = probabilities[candidate_neighbors].sum(axis=1) > 0
        selected = forward & solved
        if selected.any():
            targets = candidate_neighbors[selected]
            target_distances = candidate_distances[selected]
            positive = target_distances[target_distances > 0]
            distance_scale = float(np.median(positive)) if positive.size else 1.0
            if not np.isfinite(distance_scale) or distance_scale <= 0:
                distance_scale = 1.0
            time_delta = (
                reference_pseudotime[targets] - reference_pseudotime[index]
            )
            weights = np.exp(
                -(target_distances / distance_scale) ** 2
                + soft_threshold_nu * time_delta
            )
            local = np.average(probabilities[targets], axis=0, weights=weights)
            probabilities[index] = (
                (1.0 - terminal_teleport_weight) * local
                + terminal_teleport_weight * prior[index]
            )
        else:
            probabilities[index] = prior[index]
            fallback_count += 1
    probabilities = np.clip(probabilities, 0.0, None)
    row_sums = probabilities.sum(axis=1)
    if np.any(~np.isfinite(row_sums)) or np.any(row_sums <= 0):
        raise ValueError("Directed absorption produced invalid probability rows")
    probabilities /= row_sums[:, None]
    return probabilities, {
        **terminal_details,
        "transition_neighbors_effective": effective_neighbors,
        "n_no_forward_neighbor_terminal_prior_fallback": int(fallback_count),
        "minimum_forward_delta": float(minimum_forward_delta),
        "soft_threshold_nu": float(soft_threshold_nu),
        "terminal_teleport_weight": float(terminal_teleport_weight),
        "algorithm": "directed_acyclic_knn_absorption_with_terminal_distance_teleport",
        "transition_neighbor_backend": backend_details,
    }


def _map_reference_values(
    eligible_embedding: np.ndarray,
    reference_embedding: np.ndarray,
    reference_values: np.ndarray,
    *,
    n_neighbors: int,
    chunk_size: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    query, effective_neighbors, backend_details = _neighbor_query_factory(
        reference_embedding,
        n_neighbors=n_neighbors,
        seed=seed,
    )
    values = np.asarray(reference_values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    result = np.empty((len(eligible_embedding), values.shape[1]), dtype=np.float64)
    for start in range(0, len(eligible_embedding), chunk_size):
        stop = min(start + chunk_size, len(eligible_embedding))
        distances, neighbors = query(eligible_embedding[start:stop])
        row_scale = distances[:, -1].copy()
        positive = distances[distances > 0]
        global_scale = float(np.median(positive)) if positive.size else 1.0
        row_scale[~np.isfinite(row_scale) | (row_scale <= 1e-12)] = global_scale
        weights = np.exp(-((distances / row_scale[:, None]) ** 2))
        exact = distances[:, 0] <= 1e-12
        if exact.any():
            weights[exact] = 0.0
            weights[exact, 0] = 1.0
        weights /= weights.sum(axis=1, keepdims=True)
        result[start:stop] = np.einsum(
            "ij,ijk->ik", weights, values[neighbors], optimize=True
        )
    return result, {
        "mapping_neighbors_effective": effective_neighbors,
        "mapping_metric": "euclidean",
        "mapping_weight": "row_adaptive_gaussian_exact_match_preserved",
        **backend_details,
    }


def _safe_correlation(primary: np.ndarray, other: np.ndarray) -> float:
    selected = np.isfinite(primary) & np.isfinite(other)
    if selected.sum() < 2:
        raise ValueError("Fewer than two shared mapped cells exist for draw correlation")
    left, right = primary[selected], other[selected]
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 1.0 if np.allclose(left, right, atol=1e-12, rtol=0) else 0.0
    return float(np.clip(np.corrcoef(left, right)[0, 1], -1.0, 1.0))


def _donor_bin_counts(
    pseudotime: np.ndarray,
    mapped: np.ndarray,
    cell_donors: np.ndarray,
    donor_ids: Sequence[str],
    bin_left: np.ndarray,
    bin_right: np.ndarray,
) -> np.ndarray:
    donor_lookup = {str(donor): index for index, donor in enumerate(donor_ids)}
    donor_index = np.asarray([donor_lookup[str(donor)] for donor in cell_donors])
    edges = np.concatenate(([bin_left[0]], bin_right))
    counts = np.zeros(
        (len(donor_ids), len(bin_left), pseudotime.shape[1]), dtype=np.int64
    )
    for draw_index in range(pseudotime.shape[1]):
        selected = mapped[:, draw_index]
        bin_index = np.searchsorted(
            edges, pseudotime[selected, draw_index], side="right"
        ) - 1
        bin_index = np.clip(bin_index, 0, len(bin_left) - 1)
        np.add.at(
            counts[:, :, draw_index],
            (donor_index[selected], bin_index),
            1,
        )
    return counts


def infer_t21_trajectory_fates(
    adata: ad.AnnData, plan: Mapping[str, Any]
) -> TrajectoryFateInference:
    """Infer outcome-blind trajectory draws and primary soft fates.

    Only :data:`INFERENCE_OBS_COLUMNS` are selected from ``adata.obs``.  The
    condition columns may be present for later formal inference, but are not
    read by this function.
    """
    plan_summary = validate_trajectory_fate_plan(plan)
    obs = _inference_obs(adata)
    cell_ids = obs.index.astype(str).to_numpy()
    donors = obs["donor_id"].astype(str).to_numpy()
    labels = obs["analysis_cell_type"].astype(str).to_numpy()
    lineage = obs["lineage_inclusion"].to_numpy(dtype=bool)
    eligible_indices = np.flatnonzero(lineage).astype(np.int64)

    reference_plan = plan["reference_sketch"]
    primary_reference_indices = _balanced_reference_indices(
        obs,
        eligible_indices,
        seed=int(reference_plan["primary_seed"]),
        max_per_donor_type=int(
            reference_plan["max_cells_per_donor_analysis_cell_type"]
        ),
        max_per_donor=int(reference_plan["max_cells_per_donor"]),
    )
    primary_reference_indices, _ = _add_required_label_anchors(
        primary_reference_indices,
        eligible_indices,
        labels,
        _required_analysis_labels(plan),
    )
    eligible_embedding, representation_diagnostics = _fit_representation(
        adata,
        obs,
        eligible_indices,
        primary_reference_indices,
        plan,
    )

    draws = list(plan["draws"])
    draw_ids = tuple(str(draw["trajectory_draw_id"]) for draw in draws)
    primary_draw_id = str(plan_summary["primary_draw_id"])
    primary_draw_index = draw_ids.index(primary_draw_id)
    pseudotime = np.full((adata.n_obs, len(draws)), np.nan, dtype=np.float32)
    mapped = np.zeros((adata.n_obs, len(draws)), dtype=bool)
    draw_metadata: list[dict[str, Any]] = []
    fate_diagnostics: list[dict[str, Any]] = []
    primary_fates: np.ndarray | None = None
    representation = plan["feature_representation"]
    fate_plan = plan["fate_model"]

    for draw_index, draw in enumerate(draws):
        reference_indices, reference_details = _draw_reference_indices(
            draw,
            obs,
            eligible_indices,
            primary_reference_indices,
            plan,
        )
        reference_cell_ids = cell_ids[reference_indices]
        reference_labels = labels[reference_indices]
        reference_embedding = _reference_embedding(
            eligible_indices,
            eligible_embedding,
            reference_indices,
            seed=int(draw["seed"]),
        )
        root_id = str(draw["root_definition_id"])
        terminal_id = str(draw["terminal_definition_id"])
        root_definition = plan["root_definitions"][root_id]
        terminal_definition = plan["terminal_definitions"][terminal_id]
        root_position, root_details = _select_root_position(
            reference_embedding,
            reference_labels,
            reference_cell_ids,
            root_definition,
        )
        if draw["method"] == "scanpy_dpt_reference_mapping":
            reference_pseudotime, method_details = _scanpy_dpt(
                reference_embedding,
                root_position=root_position,
                n_neighbors=int(draw["n_neighbors"]),
                n_diffusion_components=int(draw["n_diffusion_components"]),
                seed=int(draw["seed"]),
            )
            method_version = (
                f"scanpy-{method_details['scanpy_version']}+"
                "pyfgsea-reference-mapping-v1"
            )
        else:
            reference_pseudotime, method_details = _knn_geodesic_pseudotime(
                reference_embedding,
                root_position=root_position,
                n_neighbors=int(draw["n_neighbors"]),
                seed=int(draw["seed"]),
            )
            method_version = (
                f"scipy-{method_details['scipy_version']}+"
                "pyfgsea-reference-mapping-v1"
            )
        reference_pseudotime, orientation_reversed = _orient_reference_pseudotime(
            reference_pseudotime,
            reference_labels,
            root_definition,
            terminal_definition,
        )
        reference_fates, absorption_details = _directed_absorption_probabilities(
            reference_embedding,
            reference_pseudotime,
            reference_labels,
            terminal_definition,
            n_neighbors=int(draw["n_neighbors"]),
            minimum_forward_delta=float(fate_plan["minimum_forward_delta"]),
            soft_threshold_nu=float(fate_plan["soft_threshold_nu"]),
            terminal_teleport_weight=float(fate_plan["terminal_teleport_weight"]),
            seed=int(draw["seed"]),
        )
        mapped_values, mapping_details = _map_reference_values(
            eligible_embedding,
            reference_embedding,
            np.column_stack([reference_pseudotime, reference_fates]),
            n_neighbors=int(draw["mapping_neighbors"]),
            chunk_size=int(representation["mapping_chunk_size"]),
            seed=int(draw["seed"]),
        )
        mapped_pseudotime = np.clip(mapped_values[:, 0], 0.0, 1.0)
        mapped_fates = np.clip(mapped_values[:, 1:], 0.0, None)
        mapped_fate_sums = mapped_fates.sum(axis=1)
        if np.any(~np.isfinite(mapped_fate_sums)) or np.any(mapped_fate_sums <= 0):
            raise ValueError("Reference-mapped fate probabilities are invalid")
        mapped_fates /= mapped_fate_sums[:, None]
        pseudotime[eligible_indices, draw_index] = mapped_pseudotime.astype(np.float32)
        mapped[eligible_indices, draw_index] = True
        if draw_index == primary_draw_index:
            primary_fates = mapped_fates.copy()

        terminal_definition_hash = sha256(
            stable_json(terminal_definition).encode("utf-8")
        ).hexdigest()
        parameters = {
            "draw_role": "primary" if draw.get("is_primary") else "uncertainty",
            "uncertainty_source": str(draw["uncertainty_source"]),
            "n_neighbors_requested": int(draw["n_neighbors"]),
            "n_diffusion_components_requested": int(
                draw["n_diffusion_components"]
            ),
            "mapping_neighbors_requested": int(draw["mapping_neighbors"]),
            "feature_model_hash": representation_diagnostics["feature_model_hash"],
            **reference_details,
            **method_details,
            **mapping_details,
            **absorption_details,
            "orientation_reversed_after_root_terminal_check": orientation_reversed,
        }
        parameters_json = stable_json(parameters)
        metadata = {
            "trajectory_draw_id": str(draw["trajectory_draw_id"]),
            "method": str(draw["method"]),
            "method_version": method_version,
            "parameters_json": parameters_json,
            "parameters_hash": sha256(parameters_json.encode("utf-8")).hexdigest(),
            "root_definition_id": root_id,
            "root_cell_set_hash": root_details["root_cell_set_hash"],
            "terminal_definition_id": terminal_id,
            "terminal_definition_hash": terminal_definition_hash,
            "seed": int(draw["seed"]),
            "rng": "numpy.random.Generator.PCG64",
            "used_condition_information": False,
            "used_candidate_pathway_genes": False,
            "orientation": "root_to_terminal_increasing",
            "status": "pass_primary" if draw.get("is_primary") else "pass_uncertainty",
            "correlation_with_primary": 1.0,
            "is_primary": bool(draw.get("is_primary")),
            "uncertainty_source": str(draw["uncertainty_source"]),
            **root_details,
        }
        draw_metadata.append(metadata)
        fate_diagnostics.append(
            {
                "trajectory_draw_id": str(draw["trajectory_draw_id"]),
                "terminal_definition_id": terminal_id,
                "terminal_definition_hash": terminal_definition_hash,
                "eligible_probability_sha256": _sha256_bytes(
                    np.asarray(mapped_fates, dtype=np.float32)
                ),
                "mean_probabilities": {
                    fate: float(mapped_fates[:, fate_index].mean())
                    for fate_index, fate in enumerate(FATE_ORDER)
                },
                "n_eligible": int(len(mapped_fates)),
                **absorption_details,
            }
        )

    if primary_fates is None:  # pragma: no cover - plan validation guarantees primary
        raise RuntimeError("Primary fate probabilities were not computed")
    if not np.all(mapped[lineage]):
        raise ValueError("At least one frozen-lineage cell was not mapped in every draw")
    if np.any(mapped[~lineage]) or np.any(~np.isnan(pseudotime[~lineage])):
        raise ValueError("A non-lineage cell was mapped into the trajectory")
    primary_values = pseudotime[:, primary_draw_index].astype(float)
    for draw_index, metadata in enumerate(draw_metadata):
        metadata["correlation_with_primary"] = _safe_correlation(
            primary_values, pseudotime[:, draw_index].astype(float)
        )

    grid = plan["fixed_common_pseudotime_grid"]
    bin_left = np.asarray(grid["bin_left"], dtype=float)
    bin_center = np.asarray(grid["bin_center"], dtype=float)
    bin_right = np.asarray(grid["bin_right"], dtype=float)
    donor_ids = tuple(sorted(set(map(str, donors))))
    donor_bin_cell_count = _donor_bin_counts(
        pseudotime,
        mapped,
        donors,
        donor_ids,
        bin_left,
        bin_right,
    )

    fate_frame = pd.DataFrame({"cell_id": cell_ids})
    fate_frame["erythroid_probability"] = np.nan
    fate_frame["megakaryocyte_probability"] = np.nan
    fate_frame["myeloid_probability"] = np.nan
    fate_frame["other_probability"] = np.nan
    for fate_index, column in enumerate(FATE_PROBABILITY_COLUMNS):
        fate_frame.loc[lineage, column] = primary_fates[:, fate_index]
    fate_frame["fate_eligible"] = lineage
    fate_frame["fate_ineligibility_reason"] = ""
    fate_frame.loc[~lineage, "fate_ineligibility_reason"] = (
        "outside_frozen_lineage:"
        + obs.loc[~lineage, "lineage_inclusion_reason"].astype(str).to_numpy()
    )
    fate_frame["fate_model_id"] = str(fate_plan["model_id"])
    fate_frame["trajectory_draw_id"] = primary_draw_id
    fate_frame["terminal_definition_hash"] = draw_metadata[primary_draw_index][
        "terminal_definition_hash"
    ]
    validate_fate_probabilities(fate_frame, expected_cell_ids=cell_ids)

    return TrajectoryFateInference(
        cell_ids=tuple(cell_ids),
        draw_ids=draw_ids,
        donor_ids=donor_ids,
        pseudotime=pseudotime,
        mapped=mapped,
        bin_left=bin_left,
        bin_center=bin_center,
        bin_right=bin_right,
        donor_bin_cell_count=donor_bin_cell_count,
        donor_bin_available=donor_bin_cell_count > 0,
        draw_metadata=tuple(draw_metadata),
        fate_probabilities=fate_frame,
        primary_draw_id=primary_draw_id,
        fate_draw_diagnostics=tuple(fate_diagnostics),
        representation_diagnostics=representation_diagnostics,
    )


def finalize_donor_design_with_trajectory(
    base_design: pd.DataFrame,
    scrna_obs: pd.DataFrame,
    inference: TrajectoryFateInference,
    *,
    sampling_frame_id: str,
) -> pd.DataFrame:
    """Update the frozen donor table without using condition in inference.

    Canonical condition/PCW/sex values are copied from the pre-existing design
    evidence.  Only donor, gate, lineage, and trajectory coverage are computed
    from the H5AD.
    """
    missing_base = sorted(set(REQUIRED_DONOR_DESIGN_COLUMNS).difference(base_design.columns))
    if missing_base:
        raise ValueError(f"Base donor design is missing columns: {missing_base}")
    if base_design["donor_id"].astype(str).duplicated().any():
        raise ValueError("Base donor design contains duplicate donor IDs")
    required_obs = {"donor_id", "sort_gate", "lineage_inclusion"}
    missing_obs = sorted(required_obs.difference(scrna_obs.columns))
    if missing_obs:
        raise ValueError(f"H5AD lacks donor-design coverage fields: {missing_obs}")
    observed_donors = set(scrna_obs["donor_id"].astype(str))
    if observed_donors != set(inference.donor_ids):
        raise ValueError("Trajectory and H5AD donor sets differ during design finalization")
    base = base_design.assign(
        donor_id=base_design["donor_id"].astype(str)
    ).set_index("donor_id", drop=False)
    missing_donors = observed_donors.difference(base.index)
    if missing_donors:
        raise ValueError(f"Base donor design omits H5AD donors: {sorted(missing_donors)}")
    result = base.sort_index().copy().reset_index(drop=True)
    result["number_of_cells_in_primary_lineage"] = 0
    result["trajectory_bin_coverage_fraction"] = 0.0
    result["trajectory_coverage_status"] = "trajectory_not_in_primary_sampling_frame"
    for column, default in (
        ("number_of_cells_in_primary_sampling_frame", 0),
        ("primary_sampling_frame_id", "not_in_primary_sampling_frame"),
        ("primary_trajectory_draw_id", "not_applicable"),
        ("n_trajectory_bins_planned", 0),
        ("n_trajectory_bins_observed", 0),
        ("design_stage", "metadata_only_not_in_primary_sampling_frame"),
    ):
        result[column] = default

    lineage = _strict_boolean(
        scrna_obs["lineage_inclusion"], "obs.lineage_inclusion"
    )
    coverage_obs = pd.DataFrame(
        {
            "donor_id": scrna_obs["donor_id"].astype(str).to_numpy(),
            "sort_gate": scrna_obs["sort_gate"].astype(str).to_numpy(),
            "lineage_inclusion": lineage,
        },
        index=scrna_obs.index,
    )
    donor_lookup = {donor: index for index, donor in enumerate(inference.donor_ids)}
    primary_index = inference.draw_ids.index(inference.primary_draw_id)
    for row_index, donor_id in enumerate(result["donor_id"].astype(str)):
        if donor_id not in observed_donors:
            continue
        donor_obs = coverage_obs.loc[coverage_obs["donor_id"].eq(donor_id)]
        gate_counts = {
            str(gate): int(count)
            for gate, count in donor_obs["sort_gate"].value_counts(sort=False).items()
        }
        trajectory_counts = inference.donor_bin_cell_count[
            donor_lookup[donor_id], :, primary_index
        ]
        n_observed = int((trajectory_counts > 0).sum())
        result.at[row_index, "number_of_cells_by_gate"] = stable_json(gate_counts)
        result.at[row_index, "number_of_cells_in_primary_lineage"] = int(
            donor_obs["lineage_inclusion"].sum()
        )
        result.at[row_index, "number_of_cells_in_primary_sampling_frame"] = int(
            len(donor_obs)
        )
        result.at[row_index, "primary_sampling_frame_id"] = sampling_frame_id
        result.at[row_index, "primary_trajectory_draw_id"] = inference.primary_draw_id
        result.at[row_index, "n_trajectory_bins_planned"] = int(
            len(inference.bin_center)
        )
        result.at[row_index, "n_trajectory_bins_observed"] = n_observed
        result.at[row_index, "trajectory_bin_coverage_fraction"] = (
            n_observed / len(inference.bin_center)
        )
        result.at[row_index, "trajectory_coverage_status"] = (
            "primary_sampling_frame_mapped"
        )
        result.at[row_index, "design_stage"] = (
            "analysis_ready_counts_lineage_and_trajectory_coverage"
        )
    validate_donor_design(result, scrna_obs=scrna_obs)
    return result


def _atomic_write_json(value: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_write_tsv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False, lineterminator="\n")
    os.replace(temporary, path)


def _tree_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _artifact(path: Path, role: str) -> dict[str, Any]:
    if path.is_dir():
        return {
            "role": role,
            "file_name": path.name,
            "format": "zarr",
            "bytes": _tree_bytes(path),
            "tree_digest_sha256": tree_digest(path),
        }
    return {
        "role": role,
        "file_name": path.name,
        "format": path.suffix.lstrip("."),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _git_state(repository_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        process = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return process.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        status = run("status", "--porcelain")
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "not_available", "dirty": True, "status": "git_unavailable"}
    return {
        "commit": commit,
        "dirty": bool(status),
        "status": "dirty" if status else "clean",
    }


def _capture_trajectory_code_bindings(
    repository_root: Path, *, cli_path: str | Path | None
) -> list[dict[str, str]]:
    """Hash executable module/CLI bytes before any input parsing or inference."""

    paths = [("trajectory_fate_implementation", Path(__file__).resolve())]
    if cli_path is not None:
        paths.append(("trajectory_fate_cli", Path(cli_path).resolve()))
    bindings: list[dict[str, str]] = []
    for role, path in paths:
        try:
            relative = path.relative_to(repository_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Trajectory code binding lies outside repository: {path}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Trajectory code binding is missing: {path}")
        bindings.append({"role": role, "path": relative, "sha256": sha256_file(path)})
    return bindings


def _assert_trajectory_code_bindings_unchanged(
    code_bindings: Sequence[Mapping[str, Any]], repository_root: Path
) -> None:
    """Fail publication if executable bytes changed during the long build."""

    for binding in code_bindings:
        path = (repository_root / str(binding.get("path", ""))).resolve()
        try:
            path.relative_to(repository_root)
        except ValueError as exc:
            raise ValueError("Trajectory code binding escaped the repository") from exc
        if not path.is_file() or sha256_file(path) != str(binding.get("sha256", "")):
            raise RuntimeError(
                f"Trajectory executable changed during construction: {path}"
            )


def _normalize_prebound_trajectory_code_bindings(
    code_bindings: Sequence[Mapping[str, Any]],
    repository_root: Path,
    *,
    cli_path: str | Path | None,
) -> list[dict[str, str]]:
    """Validate caller-captured executable bindings without recapturing them."""

    expected = [("trajectory_fate_implementation", Path(__file__).resolve())]
    if cli_path is not None:
        expected.append(("trajectory_fate_cli", Path(cli_path).resolve()))
    if len(code_bindings) != len(expected):
        raise ValueError("Prebound trajectory code-binding set differs")
    normalized: list[dict[str, str]] = []
    for binding, (expected_role, expected_path) in zip(code_bindings, expected):
        if not isinstance(binding, Mapping):
            raise ValueError("Prebound trajectory code bindings must be mappings")
        try:
            expected_relative = expected_path.relative_to(repository_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"Trajectory code binding lies outside repository: {expected_path}"
            ) from exc
        role = str(binding.get("role", ""))
        relative = str(binding.get("path", ""))
        digest = str(binding.get("sha256", "")).lower()
        if role != expected_role or relative != expected_relative:
            raise ValueError("Prebound trajectory code-binding identity differs")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("Prebound trajectory code-binding SHA256 is malformed")
        normalized.append({"role": role, "path": relative, "sha256": digest})
    _assert_trajectory_code_bindings_unchanged(normalized, repository_root)
    return normalized


def _role_record(
    comprehensive_record: Mapping[str, Any],
    *,
    role: str,
    comprehensive_record_name: str,
) -> dict[str, Any]:
    output = next(
        item for item in comprehensive_record["outputs"] if item["role"] == role
    )
    return {
        "schema_name": f"t21_{role}_build_record",
        "schema_version": "1.0.0",
        "build_id": comprehensive_record["build_id"],
        "created_at_utc": comprehensive_record["created_at_utc"],
        "plan_id": comprehensive_record["plan_id"],
        "plan_sha256": comprehensive_record["plan_sha256"],
        "implementation_id": comprehensive_record["implementation_id"],
        "implementation_sha256": comprehensive_record["implementation_sha256"],
        "used_condition_information_for_inference": False,
        "used_candidate_pathway_genes": False,
        "comprehensive_record": comprehensive_record_name,
        "output": output,
    }


def build_t21_trajectory_fate_products(
    *,
    h5ad_path: str | Path,
    donor_design_base_path: str | Path,
    plan_path: str | Path,
    analysis_plan_path: str | Path | None = None,
    output_dir: str | Path,
    repository_root: str | Path,
    prebound_code_bindings: Sequence[Mapping[str, Any]],
    command: Sequence[str] | None = None,
    cli_path: str | Path | None = None,
    expected_h5ad_sha256: str | None = None,
    h5ad_validator: Callable[[ad.AnnData], None] | None = None,
    logical_input_names: Mapping[str, str] | None = None,
    private_snapshot_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build all trajectory-dependent candidate artifacts in one directory."""
    repository_root = Path(repository_root).resolve()
    code_bindings = _normalize_prebound_trajectory_code_bindings(
        prebound_code_bindings,
        repository_root,
        cli_path=cli_path,
    )
    h5ad_path = Path(h5ad_path).resolve()
    donor_design_base_path = Path(donor_design_base_path).resolve()
    plan_path = Path(plan_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed_existing: set[Path] = set()
    if private_snapshot_dir is not None:
        snapshot_dir = Path(private_snapshot_dir).resolve()
        if (
            snapshot_dir.parent != output_dir
            or not snapshot_dir.name.startswith(".")
            or not snapshot_dir.is_dir()
        ):
            raise ValueError(
                "Private trajectory input snapshots must occupy one hidden output child"
            )
        allowed_existing.add(snapshot_dir)
    unexpected_existing = [
        path for path in output_dir.iterdir() if path.resolve() not in allowed_existing
    ]
    if unexpected_existing:
        raise FileExistsError(
            "Trajectory build directory contains unexpected inputs: "
            + ", ".join(sorted(path.name for path in unexpected_existing))
        )
    if analysis_plan_path is None:
        analysis_plan_path = repository_root / "config" / "t21_data_product_v1.yaml"
    analysis_plan_path = Path(analysis_plan_path).resolve()
    for path, label in (
        (h5ad_path, "assembled H5AD"),
        (donor_design_base_path, "base donor design"),
        (plan_path, "trajectory/fate plan"),
        (analysis_plan_path, "master T21 analysis plan"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")

    required_input_roles = {
        "assembled_scrna_h5ad",
        "base_donor_design",
        "frozen_trajectory_fate_plan",
        "master_t21_analysis_plan",
    }
    if logical_input_names is None:
        logical_names = {
            "assembled_scrna_h5ad": h5ad_path.name,
            "base_donor_design": donor_design_base_path.name,
            "frozen_trajectory_fate_plan": plan_path.name,
            "master_t21_analysis_plan": analysis_plan_path.name,
        }
    else:
        logical_names = {str(role): str(name) for role, name in logical_input_names.items()}
        if set(logical_names) != required_input_roles:
            raise ValueError("Logical trajectory input filename roles are incomplete")
        if any(
            not name or Path(name).name != name for name in logical_names.values()
        ):
            raise ValueError("Logical trajectory input names must be plain filenames")

    implementation_hash = sha256(
        stable_json(code_bindings).encode("utf-8")
    ).hexdigest()
    input_hashes = {
        "h5ad": sha256_file(h5ad_path),
        "donor_design_base": sha256_file(donor_design_base_path),
        "plan": sha256_file(plan_path),
        "analysis_plan": sha256_file(analysis_plan_path),
    }
    if expected_h5ad_sha256 is not None:
        expected = str(expected_h5ad_sha256).lower()
        if len(expected) != 64 or any(
            value not in "0123456789abcdef" for value in expected
        ):
            raise ValueError("Expected assembly H5AD SHA256 is malformed")
        if input_hashes["h5ad"] != expected:
            raise ValueError(
                "Assembled H5AD snapshot SHA256 differs from assembly evidence"
            )

    plan = load_trajectory_fate_plan(plan_path, repository_root=repository_root)
    analysis_plan = yaml.safe_load(analysis_plan_path.read_text(encoding="utf-8"))
    if not isinstance(analysis_plan, Mapping):
        raise ValueError("Master T21 analysis plan must contain a YAML mapping")
    trajectory_binding = _require_mapping(
        analysis_plan.get("trajectory"), "analysis_plan.trajectory"
    )
    observed_plan_hash = input_hashes["plan"]
    if trajectory_binding.get("frozen_implementation_plan_id") != plan["plan_id"]:
        raise ValueError("Master analysis plan binds a different trajectory plan ID")
    if (
        str(trajectory_binding.get("frozen_implementation_plan_sha256", "")).lower()
        != observed_plan_hash
    ):
        raise ValueError("Master analysis plan binds a different trajectory plan SHA256")
    if trajectory_binding.get("exact_author_reproduction_claim_allowed") is not False:
        raise ValueError("Master analysis plan must forbid exact-author-reproduction claims")
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        if h5ad_validator is not None:
            h5ad_validator(adata)
        inference = infer_t21_trajectory_fates(adata, plan)
        base_design = pd.read_csv(
            donor_design_base_path, sep="\t", dtype=str, keep_default_na=False
        )
        donor_design = finalize_donor_design_with_trajectory(
            base_design,
            adata.obs,
            inference,
            sampling_frame_id=str(plan["scope"]["sampling_frame_id"]),
        )

        trajectory_path = output_dir / FINAL_PRODUCT_NAMES["trajectory"]
        fate_path = output_dir / FINAL_PRODUCT_NAMES["fates"]
        donor_path = output_dir / FINAL_PRODUCT_NAMES["donor_design"]
        write_trajectory_zarr(
            trajectory_path,
            cell_ids=inference.cell_ids,
            draw_ids=inference.draw_ids,
            pseudotime=inference.pseudotime,
            mapped=inference.mapped,
            donor_ids=inference.donor_ids,
            bin_left=inference.bin_left,
            bin_center=inference.bin_center,
            bin_right=inference.bin_right,
            donor_bin_cell_count=inference.donor_bin_cell_count,
            donor_bin_available=inference.donor_bin_available,
            draw_metadata=inference.draw_metadata,
        )
        import zarr

        trajectory_group = zarr.open_group(trajectory_path, mode="r+")
        trajectory_group.attrs.update(
            {
                "plan_id": str(plan["plan_id"]),
                "plan_sha256": input_hashes["plan"],
                "master_analysis_plan_sha256": input_hashes["analysis_plan"],
                "implementation_id": IMPLEMENTATION_ID,
                "primary_trajectory_draw_id": inference.primary_draw_id,
                "public_method_claim": "structural_precedent_not_exact_reproduction",
            }
        )
        fate_mean = np.asarray(
            [
                [record["mean_probabilities"][fate] for fate in FATE_ORDER]
                for record in inference.fate_draw_diagnostics
            ],
            dtype=np.float32,
        )
        trajectory_group.create_array(
            "axes/fate",
            data=np.asarray(FATE_ORDER, dtype="U16"),
            overwrite=True,
        )
        trajectory_group.create_array(
            "draw_diagnostics/fate_mean_probability",
            data=fate_mean,
            overwrite=True,
        )
        trajectory_group.create_array(
            "draw_diagnostics/fate_probability_sha256",
            data=np.asarray(
                [
                    record["eligible_probability_sha256"]
                    for record in inference.fate_draw_diagnostics
                ],
                dtype="U64",
            ),
            overwrite=True,
        )
        write_fate_probabilities(
            inference.fate_probabilities,
            fate_path,
            expected_cell_ids=inference.cell_ids,
        )
        _atomic_write_tsv(donor_design, donor_path)

        trajectory_summary = validate_trajectory_scrna_alignment(
            trajectory_path, adata.obs
        )
        fate_summary = validate_fate_probabilities(
            pd.read_parquet(fate_path), expected_cell_ids=adata.obs_names.astype(str)
        )
        donor_summary = validate_donor_design(donor_design, scrna_obs=adata.obs)
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    outputs = [
        _artifact(trajectory_path, "trajectory"),
        _artifact(fate_path, "fates"),
        _artifact(donor_path, "donor_design"),
    ]
    plan_hash = input_hashes["plan"]
    build_id = "t21-trajectory-fate-" + sha256(
        (
            input_hashes["h5ad"]
            + input_hashes["donor_design_base"]
            + plan_hash
            + input_hashes["analysis_plan"]
            + implementation_hash
        ).encode("ascii")
    ).hexdigest()[:16]
    record = {
        "schema_name": "t21_trajectory_fate_build_record",
        "schema_version": "1.0.0",
        "build_id": build_id,
        "created_at_utc": utc_now(),
        "command": list(command or []),
        "plan_id": str(plan["plan_id"]),
        "plan_sha256": plan_hash,
        "implementation_id": IMPLEMENTATION_ID,
        "implementation_sha256": implementation_hash,
        "code_bindings": code_bindings,
        "git": _git_state(repository_root),
        "inputs": [
            {
                "role": "assembled_scrna_h5ad",
                "file_name": logical_names["assembled_scrna_h5ad"],
                "bytes": h5ad_path.stat().st_size,
                "sha256": input_hashes["h5ad"],
            },
            {
                "role": "base_donor_design",
                "file_name": logical_names["base_donor_design"],
                "bytes": donor_design_base_path.stat().st_size,
                "sha256": input_hashes["donor_design_base"],
            },
            {
                "role": "frozen_trajectory_fate_plan",
                "file_name": logical_names["frozen_trajectory_fate_plan"],
                "bytes": plan_path.stat().st_size,
                "sha256": plan_hash,
            },
            {
                "role": "master_t21_analysis_plan",
                "file_name": logical_names["master_t21_analysis_plan"],
                "bytes": analysis_plan_path.stat().st_size,
                "sha256": input_hashes["analysis_plan"],
            },
        ],
        "outputs": outputs,
        "contracts": {
            "trajectory": trajectory_summary,
            "fates": fate_summary,
            "donor_design": donor_summary,
        },
        "primary_trajectory_draw_id": inference.primary_draw_id,
        "draw_metadata": list(inference.draw_metadata),
        "fate_draw_diagnostics": list(inference.fate_draw_diagnostics),
        "representation_diagnostics": inference.representation_diagnostics,
        "used_condition_information_for_inference": False,
        "used_candidate_pathway_genes": False,
        "read_pathway_result_artifacts": False,
        "public_method_claim": "structural_precedent_not_exact_reproduction",
    }
    _assert_trajectory_code_bindings_unchanged(code_bindings, repository_root)
    comprehensive_name = BUILD_RECORD_NAMES[0]
    _atomic_write_json(record, output_dir / comprehensive_name)
    for role, name in zip(
        ("trajectory", "fates", "donor_design"), BUILD_RECORD_NAMES[1:]
    ):
        _atomic_write_json(
            _role_record(
                record,
                role=role,
                comprehensive_record_name=comprehensive_name,
            ),
            output_dir / name,
        )
    return record


def validate_built_trajectory_fate_directory(
    output_dir: str | Path, *, scrna_obs: pd.DataFrame
) -> dict[str, Any]:
    """Validate a completed build directory before candidate publication."""
    output_dir = Path(output_dir)
    trajectory = output_dir / FINAL_PRODUCT_NAMES["trajectory"]
    fates = output_dir / FINAL_PRODUCT_NAMES["fates"]
    donor_design = output_dir / FINAL_PRODUCT_NAMES["donor_design"]
    for path in (trajectory, fates, donor_design):
        if not path.exists():
            raise FileNotFoundError(f"Trajectory build output is missing: {path}")
    trajectory_summary = validate_trajectory_scrna_alignment(trajectory, scrna_obs)
    fate_summary = validate_fate_probabilities(
        pd.read_parquet(fates), expected_cell_ids=scrna_obs.index.astype(str)
    )
    donor_summary = validate_donor_design(
        pd.read_csv(donor_design, sep="\t", dtype=str, keep_default_na=False),
        scrna_obs=scrna_obs,
    )
    return {
        "trajectory": trajectory_summary,
        "fates": fate_summary,
        "donor_design": donor_summary,
        "trajectory_tree_digest_sha256": validate_trajectory_zarr(trajectory)[
            "tree_digest_sha256"
        ],
    }
