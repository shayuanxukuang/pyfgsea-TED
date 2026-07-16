from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
import yaml

from .t21_covariate_design import (
    build_t21_canonical_donor_design,
    canonical_t21_donor_design_spec_sha256,
    validate_canonical_t21_donor_design_spec,
)
from .t21_data_product import (
    FATE_PROBABILITY_COLUMNS,
    cell_id_set_hash,
    formal_t21_analysis_view,
    ordered_id_hash,
    sha256_file,
    stable_json,
    tree_digest,
    validate_donor_design,
    validate_fate_probabilities,
    validate_scrna_contract,
    validate_trajectory_scrna_alignment,
    validate_trajectory_zarr,
)
from .t21_expression_preprocessing import (
    FORMAL_EXPRESSION_CONTRACT_VERSION,
    FORMAL_EXPRESSION_TARGET_SUM,
    compute_pooled_gene_support_chunked,
    formal_expression_preprocessing_contract,
    formal_expression_preprocessing_contract_sha256,
    formal_expression_preprocessing_source_sha256,
    validate_t21_formal_expression,
)


PROFILE_SCHEMA_NAME = "t21_outcome_blind_calibration_design_profile"
PROFILE_SCHEMA_VERSION = "1.0.0"
BINDINGS_SCHEMA_VERSION = "2.0.0"
FORBIDDEN_PROFILE_KEYS = frozenset(
    {
        "cell_id",
        "condition",
        "donor_id",
        "gene_id",
        "gene_symbol",
        "library_id",
        "pathway_id",
        "pathway_label",
        "source_donor_id",
        "technical_batch",
        "trajectory_draw_id",
    }
)

_CORRELATION_ROUNDOFF_TOLERANCE = 1e-12


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {key: 0.0 for key in ("q00", "q10", "q25", "q50", "q75", "q90", "q100")}
    probabilities = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    return {
        key: float(value)
        for key, value in zip(
            ("q00", "q10", "q25", "q50", "q75", "q90", "q100"),
            np.quantile(array, probabilities),
        )
    }


def _bounded_correlation_value(value: float, *, label: str) -> float:
    """Clamp correlation roundoff while rejecting materially invalid values."""

    numeric = float(value)
    tolerance = _CORRELATION_ROUNDOFF_TOLERANCE
    if not np.isfinite(numeric) or numeric < -1.0 - tolerance or numeric > 1.0 + tolerance:
        raise ValueError(f"{label} is not a finite correlation in [-1, 1]")
    return float(np.clip(numeric, -1.0, 1.0))


def _bounded_correlation_matrix(
    values: np.ndarray, *, label: str
) -> np.ndarray:
    """Canonicalize a numerical correlation matrix without hiding real errors."""

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.size:
        raise ValueError(f"{label} must be a non-empty square matrix")
    if np.any(~np.isfinite(matrix)):
        raise ValueError(f"{label} contains a non-finite value")
    tolerance = _CORRELATION_ROUNDOFF_TOLERANCE
    if np.max(np.abs(matrix - matrix.T)) > tolerance:
        raise ValueError(f"{label} is not symmetric within numerical tolerance")
    if np.any(matrix < -1.0 - tolerance) or np.any(matrix > 1.0 + tolerance):
        raise ValueError(f"{label} contains a value outside [-1, 1]")
    if np.any(np.abs(np.diag(matrix) - 1.0) > tolerance):
        raise ValueError(f"{label} diagonal differs materially from one")
    bounded = np.clip((matrix + matrix.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(bounded, 1.0)
    return bounded


def _normalized_category_vectors(values: pd.Series) -> tuple[list[str], np.ndarray]:
    normalized = values.astype(str).str.strip().str.lower()
    categories = sorted(set(normalized), key=_sha256_text)
    lookup = {value: index for index, value in enumerate(categories)}
    matrix = np.zeros((len(normalized), len(categories)), dtype=float)
    matrix[np.arange(len(normalized)), [lookup[value] for value in normalized]] = 1.0
    return [f"C{index + 1:03d}" for index in range(len(categories))], matrix


def _anonymous_donor_order(donor_ids: Sequence[str]) -> tuple[list[str], dict[str, str]]:
    ordered = sorted((str(value) for value in donor_ids), key=_sha256_text)
    slots = {donor_id: f"D{index + 1:03d}" for index, donor_id in enumerate(ordered)}
    return ordered, slots


def _strict_bool(values: pd.Series, *, label: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values.dtype):
        if values.isna().any():
            raise ValueError(f"{label} contains missing values")
        return values.to_numpy(dtype=bool)
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError(f"{label} must contain only true/false")
    return normalized.eq("true").to_numpy(dtype=bool)


def _matrix_column_moments(
    matrix: Any,
    selected_rows: np.ndarray,
    *,
    n_rows: int,
    n_columns: int,
    row_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if row_chunk_size < 1:
        raise ValueError("row_chunk_size must be positive")
    sums = np.zeros(n_columns, dtype=np.float64)
    sum_squares = np.zeros(n_columns, dtype=np.float64)
    selected_count = 0
    for start in range(0, n_rows, row_chunk_size):
        stop = min(start + row_chunk_size, n_rows)
        local_mask = selected_rows[start:stop]
        if not np.any(local_mask):
            continue
        block = matrix[start:stop, :]
        if hasattr(block, "to_memory"):
            block = block.to_memory()
        block = block[local_mask]
        selected_count += int(np.sum(local_mask))
        if sparse.issparse(block):
            block = block.tocsr()
            sums += np.asarray(block.sum(axis=0)).ravel()
            squared = block.copy()
            squared.data = np.square(squared.data.astype(np.float64, copy=False))
            sum_squares += np.asarray(squared.sum(axis=0)).ravel()
        else:
            dense = np.asarray(block, dtype=np.float64)
            sums += dense.sum(axis=0)
            sum_squares += np.square(dense).sum(axis=0)
    if selected_count < 2:
        raise ValueError("At least two analysis cells are required for mean-variance profiling")
    means = sums / selected_count
    variances = np.maximum(sum_squares / selected_count - np.square(means), 0.0)
    return means, variances


def _anonymous_log_expression_dispersion_bins(
    means: np.ndarray, variances: np.ndarray, *, n_bins: int
) -> list[dict[str, Any]]:
    if n_bins < 2:
        raise ValueError("mean_variance_bins must be at least two")
    if means.shape != variances.shape or means.ndim != 1:
        raise ValueError("Mean and variance vectors must have matching one-dimensional shapes")
    order = np.argsort(means, kind="stable")
    groups = np.array_split(order, min(n_bins, len(order)))
    rows: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        local_means = means[group]
        local_variances = variances[group]
        positive = local_means > 0
        fano = np.divide(
            local_variances,
            local_means,
            out=np.zeros_like(local_variances),
            where=positive,
        )
        rows.append(
            {
                "anonymous_bin_index": index,
                "n_features": int(len(group)),
                "mean_log_expression_mean": float(np.mean(local_means)),
                "mean_log_expression_median": float(np.median(local_means)),
                "log_expression_variance_mean": float(np.mean(local_variances)),
                "log_expression_variance_median": float(np.median(local_variances)),
                "log_expression_variance_to_mean_median": (
                    float(np.median(fano[positive])) if np.any(positive) else 0.0
                ),
            }
        )
    return rows


def _pathway_structure(
    pathway_universe: Path,
    *,
    supported_gene_ids: set[str],
    min_size: int = 5,
    max_size: int = 500,
) -> tuple[dict[str, Any], str]:
    frame = pd.read_csv(pathway_universe, sep="\t", dtype=str, keep_default_na=False)
    required = {
        "pathway_id",
        "gene_id",
        "is_chr21",
        "level_1_family_id",
        "pathway_universe_logical_sha256",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Pathway universe lacks columns: {sorted(missing)}")
    logical = sorted(set(frame["pathway_universe_logical_sha256"].astype(str)))
    if len(logical) != 1 or not re.fullmatch(r"[0-9a-f]{64}", logical[0]):
        raise ValueError("Pathway universe must carry one valid logical SHA256")
    n_input_pathways = int(frame["pathway_id"].nunique())
    # This is the only formal gene filter: pooled non-zero support in the
    # outcome-blind primary analysis view.  It precedes both pathway-size gates
    # and all pathway-weight denominators.
    frame = frame.loc[frame["gene_id"].astype(str).isin(supported_gene_ids)].copy()
    retained_sizes = frame.groupby("pathway_id", observed=True)["gene_id"].nunique()
    retained_ids = set(
        retained_sizes.loc[
            retained_sizes.ge(int(min_size)) & retained_sizes.le(int(max_size))
        ].index.astype(str)
    )
    frame = frame.loc[frame["pathway_id"].astype(str).isin(retained_ids)].copy()
    if frame.empty or len(retained_ids) < 2:
        raise ValueError("Fewer than two pathways pass frozen expression support")
    supported_logical_sha256 = _sha256_text(
        stable_json(
            [
                [str(pathway_id), sorted(set(group["gene_id"].astype(str)))]
                for pathway_id, group in frame.groupby(
                    "pathway_id", sort=True, observed=True
                )
            ]
        )
    )
    membership: list[set[str]] = []
    chr21_counts: list[int] = []
    pathway_ids: list[str] = []
    family_ids: list[str] = []
    for pathway_id, group in frame.groupby("pathway_id", sort=True, observed=True):
        pathway_ids.append(str(pathway_id))
        genes = set(group["gene_id"].astype(str))
        membership.append(genes)
        families = set(group["level_1_family_id"].astype(str).str.strip())
        if len(families) != 1:
            raise ValueError("Each pathway must map to at most one Level-1 family")
        family_ids.append(next(iter(families)))
        chr21 = group.loc[
            group["is_chr21"].astype(str).str.lower().eq("true"), "gene_id"
        ]
        chr21_counts.append(int(chr21.nunique()))
    feature_chr21_rows = (
        frame.assign(
            _is_chr21=frame["is_chr21"].astype(str).str.lower().map(
                {"true": True, "false": False}
            )
        )
        .groupby("gene_id", sort=False, observed=True)["_is_chr21"]
        .agg(lambda values: set(values.tolist()))
    )
    if feature_chr21_rows.map(len).ne(1).any() or feature_chr21_rows.map(
        lambda values: next(iter(values)) if values else None
    ).isna().any():
        raise ValueError("Pathway member chr21 annotations are missing or inconsistent")
    feature_chr21 = {
        str(feature): bool(next(iter(values)))
        for feature, values in feature_chr21_rows.items()
    }
    anonymous_features = sorted(feature_chr21, key=_sha256_text)
    anonymous_feature_index = {
        feature: index for index, feature in enumerate(anonymous_features)
    }
    jaccard = np.eye(len(membership), dtype=float)
    pairwise_jaccard = []
    for left in range(len(membership)):
        for right in range(left + 1, len(membership)):
            union = membership[left] | membership[right]
            value = len(membership[left] & membership[right]) / len(union) if union else 0.0
            pairwise_jaccard.append(value)
            jaccard[left, right] = value
            jaccard[right, left] = value
    dependence = 0.05 * np.ones_like(jaccard) + 0.95 * jaccard
    eigenvalues, eigenvectors = np.linalg.eigh(dependence)
    dependence = (eigenvectors * np.maximum(eigenvalues, 1e-10)) @ eigenvectors.T
    scale = np.sqrt(np.diag(dependence))
    dependence = dependence / np.outer(scale, scale)
    dependence = _bounded_correlation_matrix(
        dependence, label="Pathway dependence correlation"
    )
    anonymous_families = {
        value: index
        for index, value in enumerate(
            sorted({value for value in family_ids if value}, key=_sha256_text)
        )
    }
    sizes = [len(genes) for genes in membership]
    return (
        {
            "n_input_pathways": n_input_pathways,
            "n_pathways": int(len(membership)),
            "n_unique_member_features": int(len(set().union(*membership))),
            "n_memberships": int(sum(sizes)),
            "n_chr21_memberships": int(sum(chr21_counts)),
            "n_pathways_with_chr21": int(sum(value > 0 for value in chr21_counts)),
            "pathway_size_quantiles": _quantiles(sizes),
            "chr21_membership_quantiles": _quantiles(chr21_counts),
            "pairwise_jaccard_quantiles": _quantiles(pairwise_jaccard),
            "pairwise_jaccard_mean": float(np.mean(pairwise_jaccard))
            if pairwise_jaccard
            else 0.0,
            "anonymous_pathway_order_sha256": ordered_id_hash(pathway_ids),
            "anonymous_member_feature_order_sha256": ordered_id_hash(
                anonymous_features
            ),
            "pathway_member_feature_indices": [
                sorted(anonymous_feature_index[feature] for feature in members)
                for members in membership
            ],
            "chr21_member_feature_mask": [
                feature_chr21[feature] for feature in anonymous_features
            ],
            "n_level_1_families": int(len(anonymous_families)),
            "level_1_family_index_by_pathway": [
                int(anonymous_families[value]) if value else -1 for value in family_ids
            ],
            "chr21_pathway_mask": [value > 0 for value in chr21_counts],
            "pathway_dependence_correlation": dependence.tolist(),
            "primary_view_support_filter_applied_before_size_and_weights": True,
            "pathway_min_size_after_support": int(min_size),
            "pathway_max_size_after_support": int(max_size),
            "supported_pathway_universe_logical_sha256": supported_logical_sha256,
        },
        logical[0],
    )


def _trajectory_profile(
    trajectory_path: Path,
    obs: pd.DataFrame,
    donor_order: Sequence[str],
    donor_slots: Mapping[str, str],
    *,
    primary_draw_id: str,
    row_chunk_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import zarr

    summary = validate_trajectory_zarr(trajectory_path)
    group = zarr.open_group(trajectory_path, mode="r")
    trajectory_cells = np.asarray(group["axes/cell_id"][:], dtype=str)
    indexed_obs = obs.copy()
    indexed_obs.index = indexed_obs.index.astype(str)
    if set(trajectory_cells) != set(indexed_obs.index):
        raise ValueError("Trajectory and H5AD cell sets differ")
    cell_donors = indexed_obs.loc[trajectory_cells, "donor_id"].astype(str).to_numpy()
    trajectory_donors = [str(value) for value in group["axes/donor_id"][:]]
    if set(trajectory_donors) != set(donor_order):
        raise ValueError("Trajectory donor axis differs from the donor design")
    donor_axis = {value: index for index, value in enumerate(trajectory_donors)}
    counts = np.asarray(group["donor_bin/cell_count"][:], dtype=np.int64)
    available = np.asarray(group["donor_bin/available"][:], dtype=bool)
    centers = np.asarray(group["axes/bin_center"][:], dtype=float)
    left = np.asarray(group["axes/bin_left"][:], dtype=float)
    right = np.asarray(group["axes/bin_right"][:], dtype=float)
    fixed_rows = []
    for donor_id in donor_order:
        donor_index = donor_axis[donor_id]
        for draw_index in range(counts.shape[2]):
            for bin_index in range(counts.shape[1]):
                fixed_rows.append(
                    {
                        "donor_slot": donor_slots[donor_id],
                        "draw_index": draw_index,
                        "bin_index": bin_index,
                        "cell_count": int(counts[donor_index, bin_index, draw_index]),
                        "missing": not bool(available[donor_index, bin_index, draw_index]),
                    }
                )
    draw_ids = [str(value) for value in group["axes/trajectory_draw_id"][:]]
    if primary_draw_id not in draw_ids:
        raise ValueError("Fate primary draw is absent from the trajectory draw axis")
    primary_draw_index = draw_ids.index(primary_draw_id)
    primary_mask = []
    primary_counts = []
    for donor_id in donor_order:
        donor_index = donor_axis[donor_id]
        primary_mask.append(
            available[donor_index, :, primary_draw_index].astype(bool).tolist()
        )
        primary_counts.append(
            counts[donor_index, :, primary_draw_index].astype(int).tolist()
        )
    adjacent_left = np.log1p(counts[:, :-1, :].ravel())
    adjacent_right = np.log1p(counts[:, 1:, :].ravel())
    if np.std(adjacent_left) > 0 and np.std(adjacent_right) > 0:
        adjacent_correlation = _bounded_correlation_value(
            float(np.corrcoef(adjacent_left, adjacent_right)[0, 1]),
            label="Adjacent donor-bin log-count correlation",
        )
    else:
        adjacent_correlation = 0.0

    pseudotime = group["pseudotime"]
    mapped = group["mapped"]
    dispersion_parts: list[np.ndarray] = []
    donor_sum = dict.fromkeys(donor_order, 0.0)
    donor_n = dict.fromkeys(donor_order, 0)
    for start in range(0, len(trajectory_cells), row_chunk_size):
        stop = min(start + row_chunk_size, len(trajectory_cells))
        local_time = np.asarray(pseudotime[start:stop, :], dtype=float)
        local_mapped = np.asarray(mapped[start:stop, :], dtype=bool)
        local_n = local_mapped.sum(axis=1)
        masked = np.where(local_mapped, local_time, 0.0)
        local_mean = np.divide(
            masked.sum(axis=1), local_n, out=np.zeros(len(local_n)), where=local_n > 0
        )
        squared = np.where(local_mapped, np.square(local_time - local_mean[:, None]), 0.0)
        local_dispersion = np.sqrt(
            np.divide(
                squared.sum(axis=1),
                local_n,
                out=np.zeros(len(local_n)),
                where=local_n > 0,
            )
        )
        dispersion_parts.append(local_dispersion)
        for donor_id in set(cell_donors[start:stop]):
            selection = cell_donors[start:stop] == donor_id
            donor_sum[str(donor_id)] += float(local_dispersion[selection].sum())
            donor_n[str(donor_id)] += int(selection.sum())
    dispersion = np.concatenate(dispersion_parts) if dispersion_parts else np.asarray([])
    donor_dispersion = [
        {
            "donor_slot": donor_slots[donor_id],
            "mean_draw_dispersion": donor_sum[donor_id] / max(donor_n[donor_id], 1),
        }
        for donor_id in donor_order
    ]
    grid = {
        "n_bins": int(len(centers)),
        "n_draws": int(counts.shape[2]),
        "primary_draw_index": primary_draw_index,
        "primary_draw_id_sha256": _sha256_text(primary_draw_id),
        "primary_draw_available_mask_sha256": _sha256_text(stable_json(primary_mask)),
        "primary_draw_cell_count_sha256": _sha256_text(stable_json(primary_counts)),
        "bin_left": left.tolist(),
        "bin_center": centers.tolist(),
        "bin_right": right.tolist(),
        "adjacent_log_count_correlation": adjacent_correlation,
        "fixed_donor_bin_rows": fixed_rows,
    }
    dispersion_summary = {
        "cell_level_draw_sd_quantiles": _quantiles(dispersion),
        "donor_level": donor_dispersion,
    }
    return grid, {**dispersion_summary, **summary}


def _fate_profile(
    frame: pd.DataFrame,
    obs: pd.DataFrame,
    donor_order: Sequence[str],
    donor_slots: Mapping[str, str],
) -> dict[str, Any]:
    validate_fate_probabilities(frame, expected_cell_ids=obs.index.astype(str))
    indexed_obs = obs.copy()
    indexed_obs.index = indexed_obs.index.astype(str)
    working = frame.copy()
    working["_donor"] = indexed_obs.loc[
        working["cell_id"].astype(str), "donor_id"
    ].astype(str).to_numpy()
    eligible = _strict_bool(working["fate_eligible"], label="fate_eligible")
    probabilities = working.loc[:, FATE_PROBABILITY_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    components = []
    for component_index, column in enumerate(FATE_PROBABILITY_COLUMNS):
        components.append(
            {
                "component_index": component_index,
                "probability_quantiles": _quantiles(probabilities.loc[eligible, column]),
            }
        )
    donor_rows = []
    for donor_id in donor_order:
        selection = working["_donor"].eq(donor_id).to_numpy()
        local_eligible = selection & eligible
        local = probabilities.loc[local_eligible].to_numpy(dtype=float)
        if len(local):
            entropy = -np.sum(local * np.log(np.clip(local, 1e-12, 1.0)), axis=1)
            component_variance = float(np.mean(np.var(local, axis=0)))
        else:
            entropy = np.asarray([], dtype=float)
            component_variance = 0.0
        donor_rows.append(
            {
                "donor_slot": donor_slots[donor_id],
                "eligible_fraction": float(np.mean(eligible[selection])) if np.any(selection) else 0.0,
                "mean_entropy": float(np.mean(entropy)) if len(entropy) else 0.0,
                "mean_component_variance": component_variance,
            }
        )
    eligible_probabilities = probabilities.loc[eligible].to_numpy(dtype=float)
    entropy = -np.sum(
        eligible_probabilities * np.log(np.clip(eligible_probabilities, 1e-12, 1.0)),
        axis=1,
    )
    return {
        "n_rows": int(len(frame)),
        "n_eligible": int(eligible.sum()),
        "eligible_fraction": float(np.mean(eligible)),
        "component_distributions": components,
        "entropy_quantiles": _quantiles(entropy),
        "donor_level": donor_rows,
    }


def _assert_no_identifier_keys(value: Any, *, location: str = "profile") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_PROFILE_KEYS or normalized.endswith("_path"):
                raise ValueError(
                    f"Outcome-blind design profile contains forbidden identifier key "
                    f"{key!r} at {location}"
                )
            _assert_no_identifier_keys(item, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_identifier_keys(item, location=f"{location}[{index}]")


def profile_payload_sha256(profile: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in profile.items() if key != "integrity"}
    return _sha256_text(stable_json(payload))


def _code_bindings(repository_root: Path) -> dict[str, str]:
    role_paths = {
        "profile_builder_module_sha256": repository_root
        / "pyfgsea"
        / "t21_calibration_profile.py",
        "calibration_runner_module_sha256": repository_root
        / "pyfgsea"
        / "t21_preunblinding_calibration.py",
        "shared_freedman_lane_kernel_sha256": repository_root
        / "pyfgsea"
        / "trajectory_covariate_simulation.py",
        "covariate_pseudobulk_core_sha256": repository_root
        / "pyfgsea"
        / "trajectory_covariate_pseudobulk.py",
        "pathway_family_inference_core_sha256": repository_root
        / "pyfgsea"
        / "trajectory_pathway_families.py",
        "trajectory_decomposition_core_sha256": repository_root
        / "pyfgsea"
        / "trajectory_decomposition.py",
        "trajectory_event_timing_core_sha256": repository_root
        / "pyfgsea"
        / "trajectory_events.py",
        "t21_covariate_design_core_sha256": repository_root
        / "pyfgsea"
        / "t21_covariate_design.py",
        "t21_expression_preprocessing_core_sha256": repository_root
        / "pyfgsea"
        / "t21_expression_preprocessing.py",
        "t21_data_product_contract_core_sha256": repository_root
        / "pyfgsea"
        / "t21_data_product.py",
        "profile_builder_cli_sha256": repository_root
        / "scripts"
        / "build_t21_calibration_design_profile.py",
        "profile_schema_sha256": repository_root
        / "schemas"
        / "t21_calibration_design_profile_v1.schema.json",
    }
    missing = [str(path) for path in role_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Profile code/schema binding inputs are missing: {missing}")
    return {role: sha256_file(path) for role, path in role_paths.items()}


def build_calibration_design_profile(
    *,
    scrna_path: str | Path,
    trajectory_path: str | Path,
    fates_path: str | Path,
    donor_design_path: str | Path,
    pathway_universe_path: str | Path,
    analysis_plan_path: str | Path,
    repository_root: str | Path,
    row_chunk_size: int = 4096,
    mean_variance_bins: int = 20,
) -> dict[str, Any]:
    """Build an anonymous, pathway-outcome-free simulation design profile."""
    import anndata as ad

    root = Path(repository_root).resolve()
    scrna = Path(scrna_path).resolve()
    trajectory = Path(trajectory_path).resolve()
    fates = Path(fates_path).resolve()
    donor_design_file = Path(donor_design_path).resolve()
    pathway_universe = Path(pathway_universe_path).resolve()
    analysis_plan_file = Path(analysis_plan_path).resolve()
    for source in (
        scrna,
        trajectory,
        fates,
        donor_design_file,
        pathway_universe,
        analysis_plan_file,
    ):
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise ValueError("Every profile input must be repository-local") from exc
    for source in (scrna, fates, donor_design_file, pathway_universe, analysis_plan_file):
        if not source.is_file():
            raise FileNotFoundError(source)
    if not trajectory.is_dir():
        raise FileNotFoundError(trajectory)

    adata = ad.read_h5ad(scrna, backed="r")
    try:
        analysis_plan = yaml.safe_load(analysis_plan_file.read_text(encoding="utf-8"))
        if not isinstance(analysis_plan, Mapping) or analysis_plan.get("schema_name") != (
            "t21_data_product_analysis_plan"
        ):
            raise ValueError("Unexpected T21 analysis-plan schema")
        formal_inference = analysis_plan.get("formal_inference")
        if not isinstance(formal_inference, Mapping):
            raise ValueError("Analysis plan lacks formal inference settings")
        canonical_design_spec = formal_inference.get("canonical_donor_design")
        canonical_design_spec_sha256 = validate_canonical_t21_donor_design_spec(
            canonical_design_spec
        )
        plan_id = str(analysis_plan.get("plan_id", ""))
        primary_frame = analysis_plan.get("primary_sampling_frame")
        if not plan_id or not isinstance(primary_frame, Mapping):
            raise ValueError("Analysis plan lacks the frozen primary sampling frame")
        required_obs = {
            "cell_id",
            "donor_id",
            "condition",
            "pcw",
            "sex",
            "technical_batch",
            "lineage_inclusion",
            "tissue",
            "sort_gate",
        }
        missing_obs = required_obs - set(adata.obs)
        if missing_obs or "counts" not in adata.layers:
            raise ValueError(
                f"H5AD cannot support design profiling; missing obs={sorted(missing_obs)} "
                f"or counts_layer={'counts' not in adata.layers}"
            )
        validate_scrna_contract(
            adata,
            strict_analysis_labels=True,
            require_formal_expression=False,
        )
        if not adata.obs_names.is_unique or not np.array_equal(
            adata.obs_names.astype(str).to_numpy(), adata.obs["cell_id"].astype(str).to_numpy()
        ):
            raise ValueError("H5AD cell identifiers are not unique and aligned")
        obs = adata.obs.copy()
        obs.index = obs.index.astype(str)
        for column, expected in (
            ("tissue", primary_frame.get("tissue")),
            ("sort_gate", primary_frame.get("sort_gate")),
        ):
            observed_scope = set(obs[column].astype(str).str.strip().str.lower())
            if observed_scope != {str(expected).strip().lower()}:
                raise ValueError(
                    f"H5AD {column} scope differs from the frozen primary frame"
                )
        lineage = _strict_bool(obs["lineage_inclusion"], label="lineage_inclusion")
        expression_plan = formal_inference.get("expression_matrix_contract")
        if not isinstance(expression_plan, Mapping):
            raise ValueError("Analysis plan lacks the formal expression contract")
        expected_expression_plan = {
            "contract_version": FORMAL_EXPRESSION_CONTRACT_VERSION,
            "target_sum": FORMAL_EXPRESSION_TARGET_SUM,
            "contract_sha256": (
                formal_expression_preprocessing_contract_sha256()
            ),
            "implementation_source_sha256": (
                formal_expression_preprocessing_source_sha256()
            ),
        }
        for key, expected_value in expected_expression_plan.items():
            if expression_plan.get(key) != expected_value:
                raise ValueError(f"Analysis-plan expression contract changed at {key}")
        product_metadata = adata.uns.get("t21_data_product")
        expression_metadata = (
            product_metadata.get("expression_contract")
            if isinstance(product_metadata, Mapping)
            else None
        )
        if not isinstance(expression_metadata, Mapping):
            raise ValueError("Formal H5AD lacks its expression contract")
        if stable_json(expression_metadata.get("contract")) != stable_json(
            formal_expression_preprocessing_contract()
        ):
            raise ValueError("Formal H5AD expression contract changed")
        expression_validation = expression_metadata.get("validation")
        if not isinstance(expression_validation, Mapping):
            raise ValueError("Formal H5AD lacks expression validation metadata")
        if expression_validation.get(
            "contract_sha256"
        ) != formal_expression_preprocessing_contract_sha256() or expression_validation.get(
            "implementation_source_sha256"
        ) != formal_expression_preprocessing_source_sha256():
            raise ValueError("Formal H5AD expression code binding changed")
        x_semantic_sha256 = str(
            expression_validation.get("expression_csr_semantic_sha256", "")
        )
        if not re.fullmatch(r"[0-9a-f]{64}", x_semantic_sha256):
            raise ValueError("Formal H5AD lacks a valid X semantic SHA256")

        # Backed profile construction must still validate every X value against
        # raw counts; trusting an uns hash would allow a tampered matrix to tune
        # calibration dispersion.  Materialize only bounded contiguous blocks.
        for start in range(0, int(adata.n_obs), row_chunk_size):
            stop = min(start + row_chunk_size, int(adata.n_obs))
            count_block = adata.layers["counts"][start:stop, :]
            expression_block = adata.X[start:stop, :]
            if hasattr(count_block, "to_memory"):
                count_block = count_block.to_memory()
            if hasattr(expression_block, "to_memory"):
                expression_block = expression_block.to_memory()
            validate_t21_formal_expression(count_block, expression_block)

        formal_view = formal_t21_analysis_view(
            adata,
            trajectory_path=trajectory,
            fates_path=fates,
        )
        analysis_view = np.asarray(formal_view["analysis_mask"], dtype=bool)
        if not np.array_equal(lineage, analysis_view):
            raise ValueError("Formal analysis view differs from frozen lineage")
        def count_row_reader(start: int, stop: int):
            block = adata.layers["counts"][start:stop, :]
            return block.to_memory() if hasattr(block, "to_memory") else block

        formal_support = compute_pooled_gene_support_chunked(
            count_row_reader,
            n_cells=int(adata.n_obs),
            n_genes=int(adata.n_vars),
            analysis_cell_mask=analysis_view,
            gene_ids=adata.var_names.astype(str).tolist(),
            chunk_size=row_chunk_size,
        )
        formal_gene_support = formal_support.as_mask()
        formal_gene_support_sha256 = formal_support.gene_order_bound_support_sha256
        supported_gene_ids = set(
            adata.var_names.astype(str)[formal_gene_support].tolist()
        )

        donor_design = pd.read_csv(
            donor_design_file, sep="\t", dtype=str, keep_default_na=False
        )
        donor_summary = validate_donor_design(donor_design, scrna_obs=obs)
        primary_columns = {"primary_sampling_frame_id", "trajectory_coverage_status"}
        missing_primary = primary_columns - set(donor_design)
        if missing_primary:
            raise ValueError(
                "Donor design lacks frozen primary-frame fields: "
                f"{sorted(missing_primary)}"
            )
        primary_mask = donor_design["trajectory_coverage_status"].astype(str).eq(
            "primary_sampling_frame_mapped"
        )
        primary_design = donor_design.loc[primary_mask].copy()
        excluded_design = donor_design.loc[~primary_mask].copy()
        frame_ids = set(primary_design["primary_sampling_frame_id"].astype(str))
        if frame_ids != {plan_id}:
            raise ValueError("Primary donor rows must share one frozen sampling frame")
        donor_ids = primary_design["donor_id"].astype(str).tolist()
        observed_donor_ids = sorted(set(obs["donor_id"].astype(str)))
        if set(donor_ids) != set(observed_donor_ids):
            raise ValueError(
                "H5AD donor set must equal donor-design rows mapped to the primary frame"
            )
        primary_by_donor = primary_design.assign(
            donor_id=primary_design["donor_id"].astype(str)
        ).set_index("donor_id")
        for donor_id, donor_obs in obs.groupby(
            obs["donor_id"].astype(str), observed=True, sort=False
        ):
            design_row = primary_by_donor.loc[str(donor_id)]
            for column in ("condition", "sex"):
                observed_values = {
                    str(value).strip().lower() for value in donor_obs[column]
                }
                expected_value = str(design_row[column]).strip().lower()
                if observed_values != {expected_value}:
                    raise ValueError(
                        f"H5AD donor-level {column} differs from the donor design"
                    )
            observed_pcw = pd.to_numeric(donor_obs["pcw"], errors="raise").to_numpy(
                dtype=float
            )
            expected_pcw = float(design_row["pcw"])
            if not np.isfinite(observed_pcw).all() or not np.allclose(
                observed_pcw, expected_pcw, rtol=0.0, atol=1e-12
            ):
                raise ValueError("H5AD donor-level pcw differs from the donor design")
        donor_order, donor_slots = _anonymous_donor_order(donor_ids)
        design_index = primary_design.set_index("donor_id").loc[donor_order]
        condition = design_index["condition"].astype(str)
        if not condition.isin({"T21", "disomy"}).all():
            raise ValueError("Donor design group labels are invalid")
        assignment = condition.eq("T21").to_numpy(dtype=int)
        if int(np.sum(assignment == 0)) != 3 or int(np.sum(assignment == 1)) != 14:
            raise ValueError("Publication design profile requires exactly 3 + 14 donors")
        pcw = pd.to_numeric(design_index["pcw"], errors="raise").to_numpy(dtype=float)
        pcw_sd = float(np.std(pcw))
        if not math.isfinite(pcw_sd) or pcw_sd <= 0:
            raise ValueError("PCW signature must have finite variation")
        pcw_z = (pcw - np.mean(pcw)) / pcw_sd
        _, sex_vectors = _normalized_category_vectors(design_index["sex"])

        formal_batch_values: list[str] = []
        for donor_id in donor_order:
            donor_batch = pd.unique(
                obs.loc[
                    obs["donor_id"].astype(str).eq(donor_id), "technical_batch"
                ].astype(str)
            )
            if len(donor_batch) != 1:
                raise ValueError(
                    "Canonical technical batch must be donor-constant in the primary frame"
                )
            formal_batch_values.append(str(donor_batch[0]))
        canonical_raw_design = build_t21_canonical_donor_design(
            donor_ids=donor_order,
            conditions=condition.tolist(),
            pcw=pcw,
            technical_batch=formal_batch_values,
            control="disomy",
            case="T21",
            expected_primary_batch_status=str(
                canonical_design_spec["primary_technical_batch_status"]
            ),
        )
        if canonical_raw_design.donor_frame["donor"].tolist() != donor_order:
            raise ValueError("Canonical donor ordering differs from profile anonymization")
        if canonical_raw_design.technical_batch_status == "included_identifiable":
            batch_alias = {
                value: f"B{index + 1:03d}"
                for index, value in enumerate(sorted(set(formal_batch_values)))
            }
            formal_batch_codes = [batch_alias[value] for value in formal_batch_values]
        else:
            formal_batch_codes = [
                canonical_raw_design.technical_batch_status for _ in donor_order
            ]
        canonical_anonymous_design = build_t21_canonical_donor_design(
            donor_ids=[donor_slots[value] for value in donor_order],
            conditions=condition.tolist(),
            pcw=pcw,
            technical_batch=formal_batch_codes,
            control="disomy",
            case="T21",
            expected_primary_batch_status=canonical_raw_design.technical_batch_status,
            donor_order_mode="provided_frozen_slots",
        )
        if not np.array_equal(
            canonical_raw_design.reduced_design,
            canonical_anonymous_design.reduced_design,
        ):
            raise ValueError("Anonymous canonical design changed the production matrix")

        analysis_obs = obs.loc[lineage].copy()
        batch_categories = sorted(
            set(analysis_obs["technical_batch"].astype(str)), key=_sha256_text
        )
        batch_lookup = {value: index for index, value in enumerate(batch_categories)}
        donor_rows = []
        for index, donor_id in enumerate(donor_order):
            donor_all = obs["donor_id"].astype(str).eq(donor_id)
            donor_analysis = analysis_obs["donor_id"].astype(str).eq(donor_id)
            batch_counts = np.zeros(len(batch_categories), dtype=float)
            for value, count in (
                analysis_obs.loc[donor_analysis, "technical_batch"]
                .astype(str)
                .value_counts()
                .items()
            ):
                batch_counts[batch_lookup[value]] = int(count)
            if batch_counts.sum() > 0:
                batch_counts /= batch_counts.sum()
            donor_rows.append(
                {
                    "donor_slot": donor_slots[donor_id],
                    "assignment_code": int(assignment[index]),
                    "all_cell_count": int(donor_all.sum()),
                    "analysis_cell_count": int(donor_analysis.sum()),
                    "pcw": float(pcw[index]),
                    "pcw_z": float(pcw_z[index]),
                    "formal_batch_code": formal_batch_codes[index],
                    "sex_signature": sex_vectors[index].tolist(),
                    "batch_signature": batch_counts.tolist(),
                }
            )

        means, variances = _matrix_column_moments(
            adata.X,
            analysis_view,
            n_rows=int(adata.n_obs),
            n_columns=int(adata.n_vars),
            row_chunk_size=row_chunk_size,
        )
        means = means[formal_gene_support]
        variances = variances[formal_gene_support]
        log_expression_dispersion = _anonymous_log_expression_dispersion_bins(
            means, variances, n_bins=mean_variance_bins
        )

        trajectory_alignment = validate_trajectory_scrna_alignment(trajectory, obs)
        fates_frame = pd.read_parquet(fates)
        fate_draw_ids = sorted(set(fates_frame["trajectory_draw_id"].astype(str)))
        if len(fate_draw_ids) != 1:
            raise ValueError("Fate artifact must bind exactly one primary trajectory draw")
        grid, trajectory_dispersion = _trajectory_profile(
            trajectory,
            obs,
            donor_order,
            donor_slots,
            primary_draw_id=fate_draw_ids[0],
            row_chunk_size=row_chunk_size,
        )
        fate_distribution = _fate_profile(
            fates_frame, obs, donor_order, donor_slots
        )
        pathway_structure, pathway_logical_sha = _pathway_structure(
            pathway_universe,
            supported_gene_ids=supported_gene_ids,
            min_size=5,
            max_size=500,
        )
        fates_summary = validate_fate_probabilities(
            fates_frame, expected_cell_ids=obs.index.astype(str)
        )

        trajectory_bytes = int(
            sum(path.stat().st_size for path in trajectory.rglob("*") if path.is_file())
        )
        profile: dict[str, Any] = {
            "schema_name": PROFILE_SCHEMA_NAME,
            "schema_version": PROFILE_SCHEMA_VERSION,
            "outcome_blinded": True,
            "real_pathway_results_read": False,
            "identifier_anonymization": {
                "donor_slots_sequential_after_digest_sort": True,
                "raw_category_labels_omitted": True,
                "cell_gene_and_pathway_identifiers_omitted": True,
                "condition_specific_pathway_scores_omitted": True,
            },
            "input_bindings": {
                "scrna": {
                    "file_sha256": sha256_file(scrna),
                    "bytes": int(scrna.stat().st_size),
                    "cell_set_hash": cell_id_set_hash(obs.index.astype(str)),
                    "gene_order_hash": ordered_id_hash(adata.var_names.astype(str)),
                    "donor_set_hash": cell_id_set_hash(obs["donor_id"].astype(str).unique()),
                    "expression_contract_version": FORMAL_EXPRESSION_CONTRACT_VERSION,
                    "expression_target_sum": int(FORMAL_EXPRESSION_TARGET_SUM),
                    "expression_contract_sha256": str(
                        expression_validation["contract_sha256"]
                    ),
                    "expression_implementation_sha256": (
                        formal_expression_preprocessing_source_sha256()
                    ),
                    "x_semantic_sha256": x_semantic_sha256,
                    "formal_analysis_cell_set_hash": str(
                        formal_view["analysis_cell_set_hash"]
                    ),
                    "formal_analysis_cell_order_hash": str(
                        formal_view["analysis_cell_order_hash"]
                    ),
                    "formal_analysis_cell_count": int(
                        formal_view["n_analysis_cells"]
                    ),
                    "formal_gene_order_bound_support_sha256": formal_gene_support_sha256,
                    "formal_support_contract_sha256": (
                        formal_support.support_contract_sha256
                    ),
                    "formal_support_mask_sha256_uint8": (
                        formal_support.support_mask_sha256_uint8
                    ),
                    "formal_analysis_cell_mask_sha256_uint8": (
                        formal_support.analysis_cell_mask_sha256_uint8
                    ),
                    "formal_supported_gene_count": int(formal_gene_support.sum()),
                },
                "analysis_plan": {
                    "file_sha256": sha256_file(analysis_plan_file),
                    "bytes": int(analysis_plan_file.stat().st_size),
                    "primary_sampling_frame_id_sha256": _sha256_text(plan_id),
                    "primary_tissue_sha256": _sha256_text(
                        str(primary_frame["tissue"]).strip().lower()
                    ),
                    "primary_sort_gate_sha256": _sha256_text(
                        str(primary_frame["sort_gate"]).strip().lower()
                    ),
                },
                "donor_design": {
                    "file_sha256": sha256_file(donor_design_file),
                    "bytes": int(donor_design_file.stat().st_size),
                    "donor_set_hash": cell_id_set_hash(donor_ids),
                    "full_donor_set_hash": str(donor_summary["donor_set_hash"]),
                    "excluded_donor_set_hash": cell_id_set_hash(
                        excluded_design["donor_id"].astype(str)
                    ),
                    "n_full_donors": int(len(donor_design)),
                    "n_primary_donors": int(len(primary_design)),
                    "n_excluded_donors": int(len(excluded_design)),
                    "primary_sampling_frame_id_sha256": _sha256_text(next(iter(frame_ids))),
                },
                "trajectory": {
                    "tree_digest_sha256": tree_digest(trajectory),
                    "bytes": trajectory_bytes,
                    "grid_hash": str(trajectory_alignment["grid_hash"]),
                    "cell_set_hash": str(trajectory_alignment["cell_id_set_hash"]),
                    "donor_set_hash": str(trajectory_alignment["donor_set_hash"]),
                    "primary_draw_id_sha256": str(
                        formal_view["primary_trajectory_draw_id_sha256"]
                    ),
                    "formal_analysis_cell_set_hash": str(
                        formal_view["analysis_cell_set_hash"]
                    ),
                    "formal_analysis_cell_order_hash": str(
                        formal_view["analysis_cell_order_hash"]
                    ),
                    "formal_analysis_cell_count": int(
                        formal_view["n_analysis_cells"]
                    ),
                },
                "fates": {
                    "file_sha256": sha256_file(fates),
                    "bytes": int(fates.stat().st_size),
                    "cell_set_hash": str(fates_summary["cell_id_set_hash"]),
                    "formal_analysis_cell_set_hash": str(
                        formal_view["analysis_cell_set_hash"]
                    ),
                    "formal_analysis_cell_order_hash": str(
                        formal_view["analysis_cell_order_hash"]
                    ),
                    "formal_analysis_cell_count": int(
                        formal_view["n_analysis_cells"]
                    ),
                },
                "pathway_universe": {
                    "file_sha256": sha256_file(pathway_universe),
                    "bytes": int(pathway_universe.stat().st_size),
                    "logical_sha256": pathway_logical_sha,
                    "supported_logical_sha256": pathway_structure[
                        "supported_pathway_universe_logical_sha256"
                    ],
                    "formal_gene_order_bound_support_sha256": formal_gene_support_sha256,
                },
            },
            "code_bindings": _code_bindings(root),
            "design": {
                "n_donors": int(len(donor_rows)),
                "n_assignment_code_0": int(np.sum(assignment == 0)),
                "n_assignment_code_1": int(np.sum(assignment == 1)),
                "assignment_space_size": int(math.comb(len(assignment), int(np.sum(assignment == 0)))),
                "n_sex_signature_columns": int(sex_vectors.shape[1]),
                "n_batch_signature_columns": int(len(batch_categories)),
                "canonical_formal_design": canonical_anonymous_design.audit_manifest(),
                "canonical_formal_design_spec_sha256": (
                    canonical_design_spec_sha256
                ),
                "donor_rows": donor_rows,
            },
            "fixed_grid": grid,
            "trajectory_draw_dispersion": {
                "cell_level_draw_sd_quantiles": trajectory_dispersion[
                    "cell_level_draw_sd_quantiles"
                ],
                "donor_level": trajectory_dispersion["donor_level"],
            },
            "fate_probability_distribution": fate_distribution,
            "pathway_structure": pathway_structure,
            "formal_analysis_view": {
                "definition": "lineage_inclusion_and_primary_mapped_and_fate_eligible",
                "lineage_primary_trajectory_fate_masks_identical": True,
                "n_analysis_cells": int(formal_view["n_analysis_cells"]),
                "cell_set_hash": str(formal_view["analysis_cell_set_hash"]),
                "cell_order_hash": str(formal_view["analysis_cell_order_hash"]),
                "gene_order_bound_support_sha256": formal_gene_support_sha256,
                "support_contract_sha256": formal_support.support_contract_sha256,
                "support_mask_sha256_uint8": formal_support.support_mask_sha256_uint8,
                "analysis_cell_mask_sha256_uint8": (
                    formal_support.analysis_cell_mask_sha256_uint8
                ),
                "gene_order_sha256": formal_support.gene_order_sha256,
                "n_supported_genes": int(formal_gene_support.sum()),
                "support_rule": "pooled_all_condition_blind_formal_view_count_gt_zero",
            },
            "pooled_anonymous_log_expression_dispersion": {
                "n_analysis_cells": int(analysis_view.sum()),
                "n_features": int(formal_gene_support.sum()),
                "expression_contract_version": FORMAL_EXPRESSION_CONTRACT_VERSION,
                "bins": log_expression_dispersion,
            },
        }
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()
    _assert_no_identifier_keys(profile)
    profile["integrity"] = {"profile_payload_sha256": profile_payload_sha256(profile)}
    return profile


def _profile_schema_path(repository_root: str | Path | None = None) -> Path:
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    return root / "schemas" / "t21_calibration_design_profile_v1.schema.json"


def validate_calibration_design_profile(
    profile: Mapping[str, Any], *, repository_root: str | Path | None = None
) -> dict[str, Any]:
    if not isinstance(profile, Mapping):
        raise TypeError("Calibration design profile must be a JSON object")
    if profile.get("schema_name") != PROFILE_SCHEMA_NAME:
        raise ValueError("Unexpected calibration design profile schema")
    if profile.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError("Unsupported calibration design profile version")
    if profile.get("outcome_blinded") is not True or profile.get(
        "real_pathway_results_read"
    ) is not False:
        raise ValueError("Calibration design profile violates the outcome blind")
    _assert_no_identifier_keys(profile)
    integrity = profile.get("integrity")
    if not isinstance(integrity, Mapping):
        raise ValueError("Calibration design profile lacks integrity metadata")
    expected = profile_payload_sha256(profile)
    if integrity.get("profile_payload_sha256") != expected:
        raise ValueError("Calibration design profile payload SHA256 changed")
    schema_path = _profile_schema_path(repository_root)
    if schema_path.is_file():
        import jsonschema

        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.validate(dict(profile), schema)
    root = schema_path.parents[1]
    if profile.get("code_bindings") != _code_bindings(root):
        raise ValueError("Calibration design profile code/schema bindings changed")
    inputs = profile.get("input_bindings")
    analysis_view = profile.get("formal_analysis_view")
    dispersion = profile.get("pooled_anonymous_log_expression_dispersion")
    if not all(
        isinstance(value, Mapping)
        for value in (inputs, analysis_view, dispersion)
    ):
        raise ValueError("Calibration profile lacks expression/view bindings")
    scrna = inputs.get("scrna")
    trajectory = inputs.get("trajectory")
    fates = inputs.get("fates")
    pathway_input = inputs.get("pathway_universe")
    if not all(
        isinstance(value, Mapping)
        for value in (scrna, trajectory, fates, pathway_input)
    ):
        raise ValueError("Calibration profile input bindings are incomplete")
    view_triplet = {
        (
            binding.get("formal_analysis_cell_set_hash"),
            binding.get("formal_analysis_cell_order_hash"),
            int(binding.get("formal_analysis_cell_count", -1)),
        )
        for binding in (scrna, trajectory, fates)
    }
    expected_view = (
        analysis_view.get("cell_set_hash"),
        analysis_view.get("cell_order_hash"),
        int(analysis_view.get("n_analysis_cells", -1)),
    )
    if view_triplet != {expected_view}:
        raise ValueError("Calibration profile analysis-view bindings disagree")
    if not (
        scrna.get("formal_gene_order_bound_support_sha256")
        == pathway_input.get("formal_gene_order_bound_support_sha256")
        == analysis_view.get("gene_order_bound_support_sha256")
        and scrna.get("formal_support_contract_sha256")
        == analysis_view.get("support_contract_sha256")
        and scrna.get("formal_support_mask_sha256_uint8")
        == analysis_view.get("support_mask_sha256_uint8")
        and scrna.get("formal_analysis_cell_mask_sha256_uint8")
        == analysis_view.get("analysis_cell_mask_sha256_uint8")
        and int(scrna.get("formal_supported_gene_count", -1))
        == int(analysis_view.get("n_supported_genes", -2))
        == int(dispersion.get("n_features", -3))
        and int(dispersion.get("n_analysis_cells", -1)) == expected_view[2]
    ):
        raise ValueError("Calibration profile pooled-gene support bindings disagree")
    if not (
        scrna.get("expression_contract_version")
        == dispersion.get("expression_contract_version")
        == FORMAL_EXPRESSION_CONTRACT_VERSION
        and float(scrna.get("expression_target_sum", math.nan))
        == FORMAL_EXPRESSION_TARGET_SUM
        and scrna.get("expression_contract_sha256")
        == formal_expression_preprocessing_contract_sha256()
        and scrna.get("expression_implementation_sha256")
        == formal_expression_preprocessing_source_sha256()
    ):
        raise ValueError("Calibration profile expression bindings disagree")
    if not (
        scrna.get("cell_set_hash")
        == trajectory.get("cell_set_hash")
        == fates.get("cell_set_hash")
        and scrna.get("donor_set_hash") == trajectory.get("donor_set_hash")
    ):
        raise ValueError("Calibration profile full product axes disagree")
    design = profile.get("design")
    fixed_grid = profile.get("fixed_grid")
    if not isinstance(design, Mapping) or not isinstance(fixed_grid, Mapping):
        raise ValueError("Calibration design profile lacks design/grid summaries")
    rows = design.get("donor_rows")
    if not isinstance(rows, list) or len(rows) != int(design.get("n_donors", -1)):
        raise ValueError("Calibration design donor rows are incomplete")
    donor_slots = [str(row.get("donor_slot", "")) for row in rows]
    if len(donor_slots) != len(set(donor_slots)):
        raise ValueError("Calibration design donor slots must be unique")
    donor_binding = inputs.get("donor_design")
    if not isinstance(donor_binding, Mapping) or not (
        donor_binding.get("donor_set_hash")
        == scrna.get("donor_set_hash")
        == trajectory.get("donor_set_hash")
        and int(donor_binding.get("n_primary_donors", -1))
        == int(design.get("n_donors", -2))
    ):
        raise ValueError("Calibration profile donor bindings disagree")
    fate_distribution = profile.get("fate_probability_distribution")
    trajectory_dispersion = profile.get("trajectory_draw_dispersion")
    if not isinstance(fate_distribution, Mapping) or not isinstance(
        trajectory_dispersion, Mapping
    ):
        raise ValueError("Calibration profile donor distributions are missing")
    if (
        sum(int(row.get("all_cell_count", -1)) for row in rows)
        != int(fate_distribution.get("n_rows", -2))
        or sum(int(row.get("analysis_cell_count", -1)) for row in rows)
        != int(analysis_view.get("n_analysis_cells", -2))
        or int(fate_distribution.get("n_eligible", -1))
        != int(analysis_view.get("n_analysis_cells", -2))
    ):
        raise ValueError("Calibration profile donor cell-count totals disagree")
    expected_slot_set = set(donor_slots)
    for label, donor_level in (
        ("fate", fate_distribution.get("donor_level")),
        ("trajectory", trajectory_dispersion.get("donor_level")),
    ):
        if not isinstance(donor_level, list):
            raise ValueError(f"Calibration {label} donor distribution is missing")
        observed_slots = [str(row.get("donor_slot", "")) for row in donor_level]
        if len(observed_slots) != len(set(observed_slots)) or set(
            observed_slots
        ) != expected_slot_set:
            raise ValueError(f"Calibration {label} donor slots disagree")
    fate_by_slot = {
        str(row["donor_slot"]): row
        for row in fate_distribution["donor_level"]
    }
    for row in rows:
        all_count = int(row["all_cell_count"])
        expected_eligible = int(row["analysis_cell_count"])
        observed_eligible = int(
            round(all_count * float(fate_by_slot[str(row["donor_slot"])]["eligible_fraction"]))
        )
        if observed_eligible != expected_eligible:
            raise ValueError("Calibration fate-eligible donor counts disagree")
    if int(design.get("assignment_space_size", -1)) != 680:
        raise ValueError("Primary calibration assignment space must contain 680 labels")
    canonical_profile_design = build_t21_canonical_donor_design(
        donor_ids=[row["donor_slot"] for row in rows],
        conditions=[
            "T21" if int(row["assignment_code"]) == 1 else "disomy"
            for row in rows
        ],
        pcw=[float(row["pcw"]) for row in rows],
        technical_batch=[row["formal_batch_code"] for row in rows],
        control="disomy",
        case="T21",
        expected_primary_batch_status=str(
            design.get("canonical_formal_design", {}).get(
                "technical_batch_status", ""
            )
        ),
        donor_order_mode="provided_frozen_slots",
    )
    if (
        design.get("canonical_formal_design")
        != canonical_profile_design.audit_manifest()
        or design.get("canonical_formal_design_spec_sha256")
        != canonical_t21_donor_design_spec_sha256()
    ):
        raise ValueError("Calibration profile canonical donor design changed")
    pathway = profile.get("pathway_structure")
    if not isinstance(pathway, Mapping):
        raise ValueError("Calibration design profile lacks pathway topology")
    if (
        pathway_input.get("supported_logical_sha256")
        != pathway.get("supported_pathway_universe_logical_sha256")
        or pathway_input.get("logical_sha256") is None
        or trajectory.get("primary_draw_id_sha256")
        != fixed_grid.get("primary_draw_id_sha256")
    ):
        raise ValueError("Calibration profile pathway/primary-draw bindings disagree")
    n_pathways = int(pathway.get("n_pathways", -1))
    family_index = np.asarray(
        pathway.get("level_1_family_index_by_pathway", []), dtype=int
    )
    chr21_mask = np.asarray(pathway.get("chr21_pathway_mask", []), dtype=bool)
    dependence = np.asarray(
        pathway.get("pathway_dependence_correlation", []), dtype=float
    )
    membership_indices = pathway.get("pathway_member_feature_indices", [])
    chr21_feature_mask = np.asarray(
        pathway.get("chr21_member_feature_mask", []), dtype=bool
    )
    n_member_features = int(pathway.get("n_unique_member_features", -1))
    membership_valid = bool(
        isinstance(membership_indices, list)
        and len(membership_indices) == n_pathways
        and chr21_feature_mask.shape == (n_member_features,)
    )
    if membership_valid:
        for members in membership_indices:
            member_array = np.asarray(members, dtype=int)
            if (
                member_array.ndim != 1
                or not len(member_array)
                or len(np.unique(member_array)) != len(member_array)
                or np.any(member_array < 0)
                or np.any(member_array >= n_member_features)
            ):
                membership_valid = False
                break
        if membership_valid:
            derived_chr21_mask = np.asarray(
                [
                    bool(np.any(chr21_feature_mask[np.asarray(members, dtype=int)]))
                    for members in membership_indices
                ],
                dtype=bool,
            )
            membership_valid = np.array_equal(derived_chr21_mask, chr21_mask)
    if (
        family_index.shape != (n_pathways,)
        or chr21_mask.shape != (n_pathways,)
        or dependence.shape != (n_pathways, n_pathways)
        or np.any(family_index < -1)
        or not np.any(family_index >= 0)
        or not np.any(chr21_mask)
        or not np.allclose(dependence, dependence.T, rtol=0.0, atol=1e-12)
        or not np.allclose(np.diag(dependence), 1.0, rtol=0.0, atol=1e-10)
        or np.min(np.linalg.eigvalsh(dependence)) < -1e-8
        or not membership_valid
    ):
        raise ValueError("Calibration pathway topology is incomplete or invalid")
    if int(fixed_grid.get("n_bins", -1)) != 20:
        raise ValueError("Calibration source grid must contain exactly 20 fixed bins")
    expected_grid_rows = int(design["n_donors"]) * int(fixed_grid["n_bins"]) * int(
        fixed_grid["n_draws"]
    )
    fixed_rows = fixed_grid.get("fixed_donor_bin_rows", [])
    if len(fixed_rows) != expected_grid_rows:
        raise ValueError("Fixed donor-by-bin profile rows are incomplete")
    observed_grid_keys = [
        (
            str(row.get("donor_slot", "")),
            int(row.get("draw_index", -1)),
            int(row.get("bin_index", -1)),
        )
        for row in fixed_rows
    ]
    expected_grid_keys = {
        (slot, draw_index, bin_index)
        for slot in donor_slots
        for draw_index in range(int(fixed_grid["n_draws"]))
        for bin_index in range(int(fixed_grid["n_bins"]))
    }
    if len(observed_grid_keys) != len(set(observed_grid_keys)) or set(
        observed_grid_keys
    ) != expected_grid_keys:
        raise ValueError("Fixed donor-by-bin profile axes are not complete and unique")
    primary_draw_index = int(fixed_grid.get("primary_draw_index", -1))
    analysis_count_by_slot = {
        str(row["donor_slot"]): int(row["analysis_cell_count"]) for row in rows
    }
    for slot in donor_slots:
        primary_sum = sum(
            int(row.get("cell_count", -1))
            for row in fixed_rows
            if str(row.get("donor_slot", "")) == slot
            and int(row.get("draw_index", -1)) == primary_draw_index
        )
        if primary_sum != analysis_count_by_slot[slot]:
            raise ValueError("Primary trajectory bin counts disagree with donor cells")
    dispersion_bins = dispersion.get("bins")
    if not isinstance(dispersion_bins, list):
        raise ValueError("Calibration log-expression dispersion bins are missing")
    dispersion_indices = [
        int(row.get("anonymous_bin_index", -1)) for row in dispersion_bins
    ]
    if dispersion_indices != list(range(len(dispersion_bins))) or sum(
        int(row.get("n_features", -1)) for row in dispersion_bins
    ) != int(dispersion.get("n_features", -2)):
        raise ValueError("Calibration log-expression dispersion bins disagree")
    return {
        "status": "pass",
        "profile_payload_sha256": expected,
        "n_donors": int(design["n_donors"]),
        "n_bins": int(fixed_grid["n_bins"]),
        "n_draws": int(fixed_grid["n_draws"]),
        "assignment_space_size": int(design["assignment_space_size"]),
    }


def write_calibration_design_profile(
    profile: Mapping[str, Any],
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    normalized = dict(profile)
    normalized["integrity"] = {
        "profile_payload_sha256": profile_payload_sha256(normalized)
    }
    validate_calibration_design_profile(normalized, repository_root=repository_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(normalized, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, target)
    digest = sha256_file(target)
    sidecar = target.with_suffix(target.suffix + ".sha256")
    sidecar_temp = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
    sidecar_temp.write_text(f"{digest}\n", encoding="ascii", newline="\n")
    os.replace(sidecar_temp, sidecar)
    return target


def load_calibration_design_profile(
    path: str | Path, *, repository_root: str | Path | None = None
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    profile = json.loads(source.read_text(encoding="utf-8"))
    validate_calibration_design_profile(profile, repository_root=repository_root)
    sidecar = source.with_suffix(source.suffix + ".sha256")
    if not sidecar.is_file():
        raise ValueError("Calibration design profile checksum sidecar is missing")
    declared = sidecar.read_text(encoding="ascii").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", declared) or declared != sha256_file(source):
        raise ValueError("Calibration design profile file checksum changed")
    return profile


def _git_code_binding(
    root: Path,
    *,
    allow_dirty_development: bool,
) -> dict[str, Any]:
    def git(*args: str) -> bytes:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
        ).stdout

    commit = git("rev-parse", "HEAD").decode("ascii").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("A full Git commit is required for calibration bindings")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if not status:
        return {"code_commit": commit, "code_dirty": False}
    if not allow_dirty_development:
        raise RuntimeError("Formal calibration bindings require a clean Git worktree")
    digest = hashlib.sha256()
    digest.update(status)
    digest.update(git("diff", "--binary", "HEAD", "--"))
    for line in status.decode("utf-8", errors="surrogateescape").splitlines():
        if line.startswith("?? "):
            local = root / line[3:]
            if local.is_file():
                digest.update(line[3:].encode("utf-8", errors="surrogateescape"))
                digest.update(sha256_file(local).encode("ascii"))
    return {
        "code_commit": commit,
        "code_dirty": True,
        "code_patch_sha256": digest.hexdigest(),
    }


def build_strict_blind_bindings(
    *,
    design_profile_path: str | Path,
    analysis_plan_path: str | Path,
    calibration_policy_path: str | Path,
    runner_spec_path: str | Path,
    repository_root: str | Path,
    allow_dirty_development: bool = False,
    code_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate v2 scalar-only bindings without candidate paths or labels."""
    root = Path(repository_root).resolve()
    paths = [
        Path(design_profile_path).resolve(),
        Path(analysis_plan_path).resolve(),
        Path(calibration_policy_path).resolve(),
        Path(runner_spec_path).resolve(),
        (root / "schemas" / "t21_calibration_report_v2.schema.json").resolve(),
    ]
    for source in paths:
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise ValueError("Blind-binding inputs must be repository-local") from exc
    profile = load_calibration_design_profile(paths[0], repository_root=root)
    inputs = profile["input_bindings"]
    if inputs.get("analysis_plan", {}).get("file_sha256") != sha256_file(paths[1]):
        raise ValueError("Design profile was built against a different analysis plan")
    binding = {
        "bindings_schema_version": BINDINGS_SCHEMA_VERSION,
        "analysis_plan_sha256": sha256_file(paths[1]),
        "calibration_policy_sha256": sha256_file(paths[2]),
        "runner_spec_sha256": sha256_file(paths[3]),
        "calibration_report_schema_sha256": sha256_file(paths[4]),
        "design_profile_sha256": sha256_file(paths[0]),
        "design_profile_payload_sha256": profile["integrity"][
            "profile_payload_sha256"
        ],
        "scrna_sha256": inputs["scrna"]["file_sha256"],
        "donor_design_sha256": inputs["donor_design"]["file_sha256"],
        "fates_sha256": inputs["fates"]["file_sha256"],
        "scrna_cell_id_set_hash": inputs["scrna"]["cell_set_hash"],
        "scrna_gene_order_hash": inputs["scrna"]["gene_order_hash"],
        "donor_set_hash": inputs["donor_design"]["donor_set_hash"],
        "scrna_donor_set_hash": inputs["scrna"]["donor_set_hash"],
        "expression_contract_sha256": inputs["scrna"][
            "expression_contract_sha256"
        ],
        "expression_implementation_source_sha256": inputs["scrna"][
            "expression_implementation_sha256"
        ],
        "expression_csr_semantic_sha256": inputs["scrna"][
            "x_semantic_sha256"
        ],
        "formal_analysis_cell_set_hash": inputs["scrna"][
            "formal_analysis_cell_set_hash"
        ],
        "formal_analysis_cell_order_hash": inputs["scrna"][
            "formal_analysis_cell_order_hash"
        ],
        "formal_analysis_cell_count": inputs["scrna"][
            "formal_analysis_cell_count"
        ],
        "formal_gene_order_bound_support_sha256": inputs["scrna"][
            "formal_gene_order_bound_support_sha256"
        ],
        "formal_support_contract_sha256": inputs["scrna"][
            "formal_support_contract_sha256"
        ],
        "formal_support_mask_sha256_uint8": inputs["scrna"][
            "formal_support_mask_sha256_uint8"
        ],
        "formal_analysis_cell_mask_sha256_uint8": inputs["scrna"][
            "formal_analysis_cell_mask_sha256_uint8"
        ],
        "trajectory_tree_digest_sha256": inputs["trajectory"][
            "tree_digest_sha256"
        ],
        "trajectory_grid_hash": inputs["trajectory"]["grid_hash"],
        "trajectory_primary_draw_id_sha256": inputs["trajectory"][
            "primary_draw_id_sha256"
        ],
        "pathway_universe_sha256": inputs["pathway_universe"]["file_sha256"],
        "pathway_universe_logical_sha256": inputs["pathway_universe"][
            "logical_sha256"
        ],
        "supported_pathway_universe_logical_sha256": inputs[
            "pathway_universe"
        ]["supported_logical_sha256"],
    }
    binding.update(
        dict(code_binding)
        if code_binding is not None
        else _git_code_binding(root, allow_dirty_development=allow_dirty_development)
    )
    for key, value in binding.items():
        if isinstance(value, str) and ("/" in value or "\\" in value):
            raise ValueError(f"Blind binding {key!r} leaks a filesystem path")
    return binding
