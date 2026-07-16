from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
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
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp

from .trajpathmix_corebench_cb2_contract import (
    AMENDMENT_FILE,
    BUILD_RECORD_FILE as AMENDMENT_BUILD_RECORD_FILE,
    FROZEN_CONFIG_PAYLOAD_SHA256,
    load_cb2_amendment_config,
    validate_cb2_amendment_output,
    verify_cb2_amendment_bindings,
)


PSEUDO_DESIGN_FILE = "cb2_pseudo_condition_design_audit.tsv"
ASSIGNMENT_FILE = "cb2_pseudo_condition_assignments_v1.tsv"
ASSIGNMENT_MOBILITY_FILE = "cb2_assignment_mobility_audit.tsv"
AVAILABILITY_FILE = "cb2_availability_mapping_audit.tsv"
EXPERIMENT_FILE = "cb2_experiment_balance_audit.tsv"
BIN_FILE = "cb2_bin_estimability_audit.tsv"
DECISION_FILE = "cb2_estimability_decision.json"
BUILD_RECORD_FILE = "cb2a_design_preflight_build_record_v1.json"

SCHEMA_VERSION = "1.0.0"
PREFLIGHT_ID = "trajpathmix_corebench_cb2a_design_preflight_v1"
IMPLEMENTATION_FILE = "pyfgsea/trajpathmix_corebench_cb2a_preflight.py"
CONFIG_FILE = "config/trajpathmix_corebench_cb2_functional_null_v1.yaml"
RANK_RELATIVE_TOLERANCE = 1e-10


@dataclass(frozen=True)
class _DesignInputs:
    donors: tuple[str, ...]
    experiments: tuple[str, ...]
    incidence: np.ndarray
    availability: np.ndarray
    cell_count: np.ndarray
    experiment_fraction: np.ndarray
    signatures: tuple[str, ...]
    dominant_experiment: tuple[str, ...]
    exact_incidence_pattern: tuple[str, ...]
    all_experiment_donor_counts: np.ndarray
    line_count_by_donor: np.ndarray


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise ValueError(
            f"CB2a mismatch for {label}: expected {expected!r}, observed {observed!r}"
        )


def _read_bool(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    _require(
        bool(normalized.isin({"true", "false"}).all()),
        f"Boolean column {series.name!r} contains non-boolean values",
    )
    return normalized.eq("true")


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


def _write_json(value: Mapping[str, Any], path: Path) -> None:
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


def _factorial_orbit(groups: Sequence[Sequence[int]]) -> int:
    result = 1
    for group in groups:
        result *= math.factorial(len(group))
    return int(result)


def _longest_true_run(values: Sequence[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return int(longest)


def _svd_rank(
    values: np.ndarray, *, relative_tolerance: float = RANK_RELATIVE_TOLERANCE
) -> tuple[int, float, float, float]:
    singular = np.linalg.svd(np.asarray(values, dtype=float), compute_uv=False)
    if not len(singular) or singular[0] == 0:
        return 0, 0.0, 0.0, 0.0
    threshold = float(singular[0] * relative_tolerance)
    rank = int(np.sum(singular > threshold))
    minimum_retained = float(singular[rank - 1]) if rank else 0.0
    maximum_discarded = float(singular[rank]) if rank < len(singular) else 0.0
    return rank, threshold, minimum_retained, maximum_discarded


def _assignment_hash(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(values, dtype=np.uint8))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _objective_hash(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _partition_groups(keys: Sequence[Any]) -> list[np.ndarray]:
    lookup: dict[Any, list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        lookup[key].append(index)
    return [
        np.asarray(lookup[key], dtype=int)
        for key in sorted(lookup, key=lambda value: repr(value))
    ]


def generate_constrained_assignments(
    incidence: np.ndarray,
    *,
    n_case: int,
    n_assignments: int,
    seed: int,
    max_attempts: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Construct unique, exactly experiment-balanced donor assignments.

    Each experiment receives floor/ceil half of its primary donors in the case
    group. Random linear objectives select distinct feasible vertices; the final
    materialized bit vectors, rather than solver behavior, are authoritative.
    """

    matrix = np.asarray(incidence, dtype=float)
    _require(matrix.ndim == 2 and matrix.shape[0] > 0, "Incidence must be 2D")
    _require(
        bool(np.isin(matrix, [0.0, 1.0]).all()), "Incidence must be binary"
    )
    n_donors = int(matrix.shape[0])
    totals = matrix.sum(axis=0)
    active = totals > 0
    constraint_matrix = np.vstack([np.ones(n_donors), matrix[:, active].T])
    lower = np.concatenate([[n_case], np.floor(totals[active] / 2.0)])
    upper = np.concatenate([[n_case], np.ceil(totals[active] / 2.0)])
    constraints = LinearConstraint(constraint_matrix, lower, upper)
    bounds = Bounds(np.zeros(n_donors), np.ones(n_donors))
    integrality = np.ones(n_donors, dtype=int)
    rng = np.random.default_rng(int(seed))
    assignments: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    failed = duplicates = 0
    for attempt in range(int(max_attempts)):
        objective = rng.standard_normal(n_donors)
        result = milp(
            objective,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options={"disp": False},
        )
        if not result.success or result.x is None:
            failed += 1
            continue
        assignment = np.rint(result.x).astype(np.uint8)
        if not np.allclose(result.x, assignment, rtol=0.0, atol=1e-7):
            raise RuntimeError("MILP returned a non-integral donor assignment")
        if int(assignment.sum()) != int(n_case):
            raise RuntimeError("MILP assignment violates the frozen group size")
        per_experiment = assignment @ matrix
        if np.any(per_experiment[active] < lower[1:] - 1e-8) or np.any(
            per_experiment[active] > upper[1:] + 1e-8
        ):
            raise RuntimeError("MILP assignment violates experiment balance")
        digest = _assignment_hash(assignment)
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        assignments.append(assignment)
        records.append(
            {
                "construction_attempt": int(attempt),
                "assignment_sha256": digest,
                "objective_sha256_float64_le": _objective_hash(objective),
                "solver_objective_value": float(result.fun),
                "solver_status": int(result.status),
                "solver_message": str(result.message),
            }
        )
        if len(assignments) == int(n_assignments):
            break
    if len(assignments) != int(n_assignments):
        raise RuntimeError(
            f"Only generated {len(assignments)} unique assignments after "
            f"{max_attempts} attempts ({failed} failed, {duplicates} duplicate)"
        )
    metadata = pd.DataFrame(records)
    metadata.attrs["failed_attempts"] = failed
    metadata.attrs["duplicate_attempts"] = duplicates
    metadata.attrs["attempts_used"] = int(metadata["construction_attempt"].max()) + 1
    return np.stack(assignments, axis=0), metadata


def _load_design_inputs(root: Path, config: Mapping[str, Any]) -> _DesignInputs:
    bindings = config["bindings"]

    def path(name: str) -> Path:
        return root / str(bindings[name]["relative_path"])

    cohort = pd.read_csv(path("donor_cohort"), sep="\t", dtype="string")
    cohort["primary_complete_support"] = _read_bool(cohort["primary_complete_support"])
    donors = tuple(
        sorted(
            cohort.loc[cohort["primary_complete_support"], "donor_id"].astype(str)
        )
    )
    _require_equal(
        len(donors),
        int(config["population_and_units"]["expected_donors"]),
        "primary donor count",
    )

    line_counts = pd.read_csv(
        path("donor_experiment_counts"),
        sep="\t",
        dtype={
            "line_id": "string",
            "donor_id": "string",
            "experiment_id": "string",
            "day": "string",
            "cell_count": "int64",
        },
    )
    experiments = tuple(sorted(line_counts["experiment_id"].astype(str).unique()))
    _require_equal(
        len(experiments),
        int(config["population_and_units"]["frozen_experiment_universe"]),
        "frozen experiment universe",
    )
    donor_index = {donor: index for index, donor in enumerate(donors)}
    experiment_index = {
        experiment: index for index, experiment in enumerate(experiments)
    }
    incidence = np.zeros((len(donors), len(experiments)), dtype=np.uint8)
    all_experiment_donor_counts = np.zeros(len(experiments), dtype=int)
    for experiment, group in line_counts.groupby("experiment_id", sort=False):
        all_experiment_donor_counts[experiment_index[str(experiment)]] = int(
            group["donor_id"].nunique()
        )
    primary_pairs = line_counts.loc[
        line_counts["donor_id"].astype(str).isin(donors),
        ["donor_id", "experiment_id"],
    ].drop_duplicates()
    for row in primary_pairs.itertuples(index=False):
        incidence[donor_index[str(row.donor_id)], experiment_index[str(row.experiment_id)]] = 1
    primary_line_counts = (
        line_counts.loc[line_counts["donor_id"].astype(str).isin(donors)]
        [["donor_id", "line_id"]]
        .drop_duplicates()
        .groupby("donor_id")["line_id"]
        .nunique()
        .reindex(donors, fill_value=0)
        .to_numpy(dtype=int)
    )

    availability_long = pd.read_csv(
        path("donor_bin_availability"), sep="\t", dtype={"donor_id": "string"}
    )
    availability_long["available"] = _read_bool(availability_long["available"])
    primary_availability = availability_long.loc[
        availability_long["donor_id"].astype(str).isin(donors)
    ].copy()
    _require_equal(
        len(primary_availability), len(donors) * 20, "primary donor-bin rows"
    )
    cell_count = (
        primary_availability.pivot(
            index="donor_id", columns="bin_id", values="cell_count"
        )
        .reindex(index=donors, columns=range(20))
        .to_numpy(dtype=int)
    )
    availability = (
        primary_availability.pivot(
            index="donor_id", columns="bin_id", values="available"
        )
        .reindex(index=donors, columns=range(20))
        .to_numpy(dtype=bool)
    )
    _require(
        bool(np.array_equal(availability, cell_count >= 5)),
        "Frozen availability differs from the >=5-cell rule",
    )

    coordinate = pd.read_csv(
        path("fixed_coordinate"),
        sep="\t",
        usecols=["donor_id", "experiment_id", "corebench_coordinate"],
        dtype={"donor_id": "string", "experiment_id": "string"},
    )
    coordinate = coordinate.loc[coordinate["donor_id"].astype(str).isin(donors)].copy()
    values = coordinate["corebench_coordinate"].to_numpy(dtype=float)
    _require(
        bool(np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()),
        "Frozen coordinate values are invalid",
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
        "Coordinate-derived donor-bin counts differ from frozen CB1 availability",
    )
    experiment_fraction = np.divide(
        cube,
        cube.sum(axis=2, keepdims=True),
        out=np.zeros_like(cube),
        where=cube.sum(axis=2, keepdims=True) > 0,
    )
    available_row_sums = experiment_fraction.sum(axis=2)[availability]
    _require(
        bool(np.allclose(available_row_sums, 1.0, rtol=0.0, atol=1e-12)),
        "Available experiment-fraction rows do not sum to one",
    )
    signatures = tuple(
        "".join("1" if value else "0" for value in row) for row in availability
    )
    exact_patterns = tuple(
        "".join("1" if value else "0" for value in row) for row in incidence
    )
    donor_experiment_totals = cube.sum(axis=1)
    dominant: list[str] = []
    for row in donor_experiment_totals:
        maximum = float(row.max())
        choices = [
            experiments[index]
            for index in np.flatnonzero(np.isclose(row, maximum, rtol=0.0, atol=0.0))
        ]
        dominant.append(sorted(choices)[0])
    return _DesignInputs(
        donors=donors,
        experiments=experiments,
        incidence=incidence,
        availability=availability,
        cell_count=cell_count,
        experiment_fraction=experiment_fraction,
        signatures=signatures,
        dominant_experiment=tuple(dominant),
        exact_incidence_pattern=exact_patterns,
        all_experiment_donor_counts=all_experiment_donor_counts,
        line_count_by_donor=primary_line_counts,
    )


def _zero_overlap_audit(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    bindings = config["bindings"]
    folds = pd.read_csv(
        root / bindings["coordinate_gene_folds"]["relative_path"],
        sep="\t",
        dtype="string",
    )
    coordinate_gene = _read_bool(folds["coordinate_gene"])
    selected = folds.loc[coordinate_gene].copy()
    universe = pd.read_csv(
        root / bindings["pathway_universe"]["relative_path"],
        sep="\t",
        dtype="string",
        usecols=[
            "pathway_id",
            "gene_id",
            "source_symbol",
            "pathway_universe_logical_sha256",
        ],
    )
    logical = sorted(
        universe["pathway_universe_logical_sha256"].dropna().astype(str).unique()
    )
    _require_equal(
        logical,
        [str(bindings["pathway_universe"]["logical_sha256"])],
        "pathway universe logical hash",
    )
    universe_gene = set(universe["gene_id"].dropna().astype(str))
    universe_symbol = set(
        universe["source_symbol"].dropna().astype(str).str.upper()
    )
    coordinate_gene_ids = set(selected["ensembl_gene_id"].dropna().astype(str))
    coordinate_symbols = set(
        selected["gene_symbol"].dropna().astype(str).str.upper()
    )
    gene_overlap = sorted(coordinate_gene_ids & universe_gene)
    symbol_overlap = sorted(coordinate_symbols & universe_symbol)
    return {
        "coordinate_gene_count": int(len(selected)),
        "pathway_count": int(universe["pathway_id"].nunique()),
        "membership_count": int(len(universe)),
        "ensembl_overlap_count": int(len(gene_overlap)),
        "symbol_overlap_count": int(len(symbol_overlap)),
        "overlap_gene_ids": gene_overlap,
        "overlap_symbols": symbol_overlap,
        "zero_overlap_pass": not gene_overlap and not symbol_overlap,
        "coordinate_pathway_zero_overlap_pass": not gene_overlap and not symbol_overlap,
        "coordinate_injection_zero_overlap_pass": not gene_overlap and not symbol_overlap,
        "pathway_universe_is_bound_parent_injection_universe": True,
        "pathway_scoring_performed": False,
    }


def _availability_audit(inputs: _DesignInputs) -> tuple[pd.DataFrame, dict[str, Any]]:
    signature_lookup: dict[str, list[int]] = defaultdict(list)
    for index, signature in enumerate(inputs.signatures):
        signature_lookup[signature].append(index)
    rows: list[dict[str, Any]] = []
    signature_groups: list[np.ndarray] = []
    exact_groups: list[np.ndarray] = []
    dominant_groups: list[np.ndarray] = []
    for signature in sorted(signature_lookup):
        indices = np.asarray(signature_lookup[signature], dtype=int)
        signature_groups.append(indices)
        exact_counts = Counter(inputs.exact_incidence_pattern[index] for index in indices)
        dominant_counts = Counter(inputs.dominant_experiment[index] for index in indices)
        exact_local_groups = _partition_groups(
            [inputs.exact_incidence_pattern[index] for index in indices]
        )
        dominant_local_groups = _partition_groups(
            [inputs.dominant_experiment[index] for index in indices]
        )
        exact_sizes = sorted(exact_counts.values(), reverse=True)
        dominant_sizes = sorted(dominant_counts.values(), reverse=True)
        for local_group in exact_local_groups:
            exact_groups.append(indices[local_group])
        for local_group in dominant_local_groups:
            dominant_groups.append(indices[local_group])
        rows.append(
            {
                "restriction_block": "primary_complete_support_v1",
                "availability_signature": signature,
                "n_available_bins": int(signature.count("1")),
                "n_donors": int(len(indices)),
                "donor_ids": "|".join(inputs.donors[index] for index in indices),
                "mobile_under_frozen_mapping": bool(len(indices) > 1),
                "signature_orbit_contribution": int(math.factorial(len(indices))),
                "log10_signature_orbit_contribution": float(
                    math.lgamma(len(indices) + 1) / math.log(10)
                ),
                "n_exact_experiment_incidence_patterns": int(len(exact_counts)),
                "exact_experiment_signature_subblock_sizes": "|".join(
                    map(str, exact_sizes)
                ),
                "n_donors_in_singleton_exact_experiment_subblocks": int(
                    sum(size for size in exact_sizes if size == 1)
                ),
                "exact_experiment_signature_orbit_contribution": int(
                    math.prod(math.factorial(size) for size in exact_sizes)
                ),
                "n_dominant_experiment_patterns": int(len(dominant_counts)),
                "dominant_experiment_signature_subblock_sizes": "|".join(
                    map(str, dominant_sizes)
                ),
                "dominant_experiment_signature_orbit_contribution": int(
                    math.prod(math.factorial(size) for size in dominant_sizes)
                ),
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["n_donors", "availability_signature"], ascending=[False, True]
    ).reset_index(drop=True)
    orbit = _factorial_orbit(signature_groups)
    exact_orbit = _factorial_orbit(exact_groups)
    dominant_orbit = _factorial_orbit(dominant_groups)
    mobile = int(sum(len(group) for group in signature_groups if len(group) > 1))
    exact_mobile = int(sum(len(group) for group in exact_groups if len(group) > 1))
    dominant_mobile = int(
        sum(len(group) for group in dominant_groups if len(group) > 1)
    )
    varying_reduced_design_blocks = 0
    evaluated_mobile_block_bins = 0
    for group in signature_groups:
        if len(group) <= 1:
            continue
        for bin_id in range(20):
            if not inputs.availability[int(group[0]), bin_id]:
                continue
            evaluated_mobile_block_bins += 1
            rows = inputs.experiment_fraction[group, bin_id, :]
            if not np.allclose(rows, rows[0], rtol=0.0, atol=1e-12):
                varying_reduced_design_blocks += 1
    summary = {
        "n_unique_availability_signatures": int(len(signature_groups)),
        "signature_size_distribution": {
            str(size): int(count)
            for size, count in sorted(
                Counter(len(group) for group in signature_groups).items()
            )
        },
        "n_mobile_donors": mobile,
        "mobile_donor_fraction": float(mobile / len(inputs.donors)),
        "n_immobile_donors": int(len(inputs.donors) - mobile),
        "residual_mapping_orbit_size": orbit,
        "n_unique_null_mappings_possible": int(orbit - 1),
        "exhaustive_reference_resolution": float(1.0 / orbit),
        "reduced_design_block_constant": bool(varying_reduced_design_blocks == 0),
        "evaluated_mobile_block_bins": int(evaluated_mobile_block_bins),
        "mobile_block_bins_with_varying_experiment_fraction_design": int(
            varying_reduced_design_blocks
        ),
        "freedman_lane_exactness_status": (
            "monte_carlo_freedman_lane_not_finite_sample_exact_because_nuisance_rows_vary_within_signature_blocks"
        ),
        "finite_sample_exact_residual_group_claim_allowed": False,
        "exact_experiment_incidence_by_signature": {
            "residual_mapping_orbit_size": exact_orbit,
            "n_mobile_donors": exact_mobile,
            "n_immobile_donors": int(len(inputs.donors) - exact_mobile),
            "exhaustive_reference_resolution": float(1.0 / exact_orbit),
            "selected_for_primary": False,
            "reason": "degenerate_counterfactual_hard_block",
        },
        "dominant_experiment_by_signature": {
            "residual_mapping_orbit_size": dominant_orbit,
            "n_mobile_donors": dominant_mobile,
            "n_immobile_donors": int(len(inputs.donors) - dominant_mobile),
            "exhaustive_reference_resolution": float(1.0 / dominant_orbit),
            "selected_for_primary": False,
            "reason": "dominant_experiment_substitution_forbidden_and_orbit_degenerate",
        },
    }
    return frame, summary


def _single_bin_diagnostics(
    reduced: np.ndarray,
    condition: np.ndarray,
    available: np.ndarray,
    signature_groups: Sequence[np.ndarray],
    *,
    min_donors_per_condition: int,
    min_residual_df: int,
    max_condition_vif: float,
) -> dict[str, Any]:
    indices = np.flatnonzero(available)
    z = np.asarray(reduced[indices], dtype=float)
    c = np.asarray(condition[indices], dtype=float)
    full = np.column_stack([z, c])
    (
        rank_reduced,
        reduced_rank_threshold,
        reduced_minimum_retained,
        reduced_maximum_discarded,
    ) = _svd_rank(z)
    rank_full, full_rank_threshold, full_minimum_retained, full_maximum_discarded = (
        _svd_rank(full)
    )
    residual_df = int(len(indices) - rank_full)
    u = c - z @ (
        np.linalg.pinv(z, rcond=RANK_RELATIVE_TOLERANCE, hermitian=False) @ c
    )
    condition_information = float(u @ u)
    centered = c - float(c.mean())
    unadjusted = float(centered @ centered)
    information_fraction = (
        condition_information / unadjusted if unadjusted > 0 else 0.0
    )
    condition_vif = (
        unadjusted / condition_information
        if condition_information > 1e-14 * max(1.0, unadjusted)
        else math.inf
    )
    global_to_local = {int(value): local for local, value in enumerate(indices)}
    permutation_information = 0.0
    label_information = 0.0
    for group in signature_groups:
        local = [
            global_to_local[int(value)]
            for value in group
            if int(value) in global_to_local
        ]
        if len(local) <= 1:
            continue
        local_index = np.asarray(local, dtype=int)
        residual_values = u[local_index]
        labels = c[local_index]
        permutation_information += float(
            np.sum((residual_values - residual_values.mean()) ** 2)
        )
        label_information += float(np.sum((labels - labels.mean()) ** 2))
    n_case = int(c.sum())
    n_control = int(len(c) - n_case)
    reasons: list[str] = []
    if min(n_case, n_control) < int(min_donors_per_condition):
        reasons.append("insufficient_donors_per_condition")
    if rank_full != rank_reduced + 1 or condition_information <= 1e-14 * max(
        1.0, unadjusted
    ):
        reasons.append("condition_not_identifiable_from_experiment_fractions")
    if residual_df < int(min_residual_df):
        reasons.append("insufficient_residual_df")
    if not np.isfinite(condition_vif) or condition_vif > float(max_condition_vif):
        reasons.append("condition_vif_exceeds_threshold")
    if permutation_information <= 1e-14 * max(1.0, condition_information):
        reasons.append("degenerate_residual_permutation_information")
    if label_information <= 1e-14:
        reasons.append("no_within_signature_condition_label_information")
    return {
        "n_donors_available": int(len(indices)),
        "n_case_available": n_case,
        "n_control_available": n_control,
        "n_experiments_represented": int(np.sum(z.sum(axis=0) > 0)),
        "reduced_model_rank": rank_reduced,
        "full_model_rank": rank_full,
        "rank_relative_tolerance": float(RANK_RELATIVE_TOLERANCE),
        "reduced_rank_absolute_threshold": reduced_rank_threshold,
        "reduced_minimum_retained_singular_value": reduced_minimum_retained,
        "reduced_maximum_discarded_singular_value": reduced_maximum_discarded,
        "full_rank_absolute_threshold": full_rank_threshold,
        "full_minimum_retained_singular_value": full_minimum_retained,
        "full_maximum_discarded_singular_value": full_maximum_discarded,
        "residual_df_full": residual_df,
        "condition_information": condition_information,
        "unadjusted_condition_information": unadjusted,
        "condition_information_fraction": information_fraction,
        "condition_vif": float(condition_vif),
        "permutation_information": permutation_information,
        "condition_label_permutation_information": label_information,
        "design_gate_pass": not reasons,
        "rejection_reason": "|".join(reasons),
    }


def _audit_assignments(
    inputs: _DesignInputs,
    assignments: np.ndarray,
    construction: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    signature_groups = _partition_groups(inputs.signatures)
    thresholds = config["availability_handling"]
    design_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    label_spaces: list[int] = []
    experiment_totals = inputs.incidence.sum(axis=0).astype(int)
    for assignment_index, condition in enumerate(assignments):
        assignment_id = f"CB2P_{assignment_index + 1:04d}"
        construction_row = construction.iloc[assignment_index]
        per_experiment_case = condition @ inputs.incidence
        per_experiment_control = experiment_totals - per_experiment_case
        experiment_imbalance = np.abs(per_experiment_case - per_experiment_control)
        label_space = 1
        for group in signature_groups:
            label_space *= math.comb(len(group), int(condition[group].sum()))
        label_spaces.append(int(label_space))
        local_bin_rows: list[dict[str, Any]] = []
        for bin_id in range(20):
            diagnostics = _single_bin_diagnostics(
                inputs.experiment_fraction[:, bin_id, :],
                condition,
                inputs.availability[:, bin_id],
                signature_groups,
                min_donors_per_condition=int(
                    thresholds["minimum_donors_per_pseudo_condition"]
                ),
                min_residual_df=int(thresholds["minimum_residual_df"]),
                max_condition_vif=float(thresholds["maximum_condition_vif"]),
            )
            row = {
                "assignment_id": assignment_id,
                "bin_id": int(bin_id),
                **diagnostics,
            }
            bin_rows.append(row)
            local_bin_rows.append(row)
        pass_flags = [bool(row["design_gate_pass"]) for row in local_bin_rows]
        reasons = sorted(
            {
                reason
                for row in local_bin_rows
                for reason in str(row["rejection_reason"]).split("|")
                if reason
            }
        )
        design_rows.append(
            {
                "assignment_id": assignment_id,
                "assignment_sha256": str(construction_row["assignment_sha256"]),
                "construction_attempt": int(construction_row["construction_attempt"]),
                "objective_sha256_float64_le": str(
                    construction_row["objective_sha256_float64_le"]
                ),
                "n_pseudo_case": int(condition.sum()),
                "n_pseudo_control": int(len(condition) - condition.sum()),
                "max_absolute_experiment_donor_count_imbalance": int(
                    experiment_imbalance.max(initial=0)
                ),
                "sum_absolute_experiment_donor_count_imbalance": int(
                    experiment_imbalance.sum()
                ),
                "n_active_experiments_with_both_groups": int(
                    np.sum((per_experiment_case > 0) & (per_experiment_control > 0))
                ),
                "minimum_case_donors_in_any_bin": int(
                    min(row["n_case_available"] for row in local_bin_rows)
                ),
                "minimum_control_donors_in_any_bin": int(
                    min(row["n_control_available"] for row in local_bin_rows)
                ),
                "minimum_full_residual_df": int(
                    min(row["residual_df_full"] for row in local_bin_rows)
                ),
                "maximum_condition_vif": float(
                    max(row["condition_vif"] for row in local_bin_rows)
                ),
                "minimum_condition_information": float(
                    min(row["condition_information"] for row in local_bin_rows)
                ),
                "minimum_condition_information_fraction": float(
                    min(row["condition_information_fraction"] for row in local_bin_rows)
                ),
                "minimum_permutation_information": float(
                    min(row["permutation_information"] for row in local_bin_rows)
                ),
                "minimum_condition_label_permutation_information": float(
                    min(
                        row["condition_label_permutation_information"]
                        for row in local_bin_rows
                    )
                ),
                "availability_block_label_orbit_size": int(label_space),
                "availability_block_label_orbit_resolution": float(
                    1.0 / label_space
                ),
                "n_bins_passing_design_gate": int(sum(pass_flags)),
                "longest_contiguous_passing_bins": _longest_true_run(pass_flags),
                "all_20_bins_design_estimable": bool(all(pass_flags)),
                "design_rejection_reasons": "|".join(reasons),
            }
        )
        for donor_index, donor in enumerate(inputs.donors):
            assignment_rows.append(
                {
                    "assignment_id": assignment_id,
                    "assignment_sha256": str(construction_row["assignment_sha256"]),
                    "donor_id": donor,
                    "pseudo_condition": (
                        "pseudo_case" if bool(condition[donor_index]) else "pseudo_control"
                    ),
                    "pseudo_case": bool(condition[donor_index]),
                }
            )

    design = pd.DataFrame(design_rows)
    assignment_manifest = pd.DataFrame(assignment_rows)
    mobility_rows: list[dict[str, Any]] = []
    donor_case_counts = assignments.sum(axis=0).astype(int)
    for donor_index, donor in enumerate(inputs.donors):
        n_case_assignments = int(donor_case_counts[donor_index])
        mobility_rows.append(
            {
                "donor_id": donor,
                "n_lines": int(inputs.line_count_by_donor[donor_index]),
                "n_experiments": int(inputs.incidence[donor_index].sum()),
                "experiment_ids": "|".join(
                    inputs.experiments[index]
                    for index in np.flatnonzero(inputs.incidence[donor_index])
                ),
                "n_pseudo_case_assignments": n_case_assignments,
                "n_pseudo_control_assignments": int(
                    len(assignments) - n_case_assignments
                ),
                "pseudo_case_fraction": float(n_case_assignments / len(assignments)),
                "label_mobile_across_materialized_bank": bool(
                    0 < n_case_assignments < len(assignments)
                ),
            }
        )
    mobility = pd.DataFrame(mobility_rows)
    bin_long = pd.DataFrame(bin_rows)
    bin_summary_rows: list[dict[str, Any]] = []
    for bin_id, group in bin_long.groupby("bin_id", sort=True):
        bin_summary_rows.append(
            {
                "bin_id": int(bin_id),
                "n_donors_available": int(group["n_donors_available"].iloc[0]),
                "n_experiments_represented": int(
                    group["n_experiments_represented"].iloc[0]
                ),
                "reduced_model_rank": int(group["reduced_model_rank"].iloc[0]),
                "minimum_reduced_retained_to_rank_threshold_ratio": float(
                    (
                        group["reduced_minimum_retained_singular_value"]
                        / group["reduced_rank_absolute_threshold"]
                    ).min()
                ),
                "maximum_reduced_discarded_to_rank_threshold_ratio": float(
                    (
                        group["reduced_maximum_discarded_singular_value"]
                        / group["reduced_rank_absolute_threshold"]
                    ).max()
                ),
                "minimum_case_donors": int(group["n_case_available"].min()),
                "minimum_control_donors": int(group["n_control_available"].min()),
                "minimum_full_residual_df": int(group["residual_df_full"].min()),
                "maximum_condition_vif": float(group["condition_vif"].max()),
                "minimum_condition_information": float(
                    group["condition_information"].min()
                ),
                "minimum_condition_information_fraction": float(
                    group["condition_information_fraction"].min()
                ),
                "minimum_permutation_information": float(
                    group["permutation_information"].min()
                ),
                "minimum_condition_label_permutation_information": float(
                    group["condition_label_permutation_information"].min()
                ),
                "n_assignments_passing": int(group["design_gate_pass"].sum()),
                "n_assignments_total": int(len(group)),
            }
        )
    bin_summary = pd.DataFrame(bin_summary_rows)

    experiment_rows: list[dict[str, Any]] = []
    all_case = assignments @ inputs.incidence
    for experiment_index, experiment in enumerate(inputs.experiments):
        primary_count = int(experiment_totals[experiment_index])
        case_values = all_case[:, experiment_index].astype(int)
        control_values = primary_count - case_values
        experiment_rows.append(
            {
                "experiment_id": experiment,
                "all_cohort_donors": int(
                    inputs.all_experiment_donor_counts[experiment_index]
                ),
                "primary_cohort_donors": primary_count,
                "absent_from_primary_cohort": bool(primary_count == 0),
                "attainable_pseudo_case_lower": int(math.floor(primary_count / 2)),
                "attainable_pseudo_case_upper": int(math.ceil(primary_count / 2)),
                "theoretical_minimum_group_count_imbalance": int(primary_count % 2),
                "observed_minimum_pseudo_case": int(case_values.min()),
                "observed_maximum_pseudo_case": int(case_values.max()),
                "observed_mean_pseudo_case": float(case_values.mean()),
                "observed_minimum_pseudo_control": int(control_values.min()),
                "observed_maximum_pseudo_control": int(control_values.max()),
                "maximum_absolute_group_count_imbalance": int(
                    np.abs(case_values - control_values).max(initial=0)
                ),
                "assignments_with_both_groups": int(
                    np.sum((case_values > 0) & (control_values > 0))
                ),
                "n_assignments": int(len(assignments)),
            }
        )
    experiment = pd.DataFrame(experiment_rows)
    summary = {
        "n_assignments": int(len(assignments)),
        "n_unique_assignment_hashes": int(design["assignment_sha256"].nunique()),
        "construction_attempts_used": int(construction.attrs["attempts_used"]),
        "construction_failed_attempts": int(construction.attrs["failed_attempts"]),
        "construction_duplicate_attempts": int(
            construction.attrs["duplicate_attempts"]
        ),
        "all_group_sizes_exact": bool(
            design["n_pseudo_case"].eq(37).all()
            and design["n_pseudo_control"].eq(38).all()
        ),
        "assignment_bank_sampling_claim": (
            "deterministic_nonuniform_balanced_benchmark_bank_not_uniform_over_all_legal_assignments"
        ),
        "assignment_bank_used_for_randomization_p_values": False,
        "n_label_mobile_donors": int(
            mobility["label_mobile_across_materialized_bank"].sum()
        ),
        "n_label_immobile_donors": int(
            (~mobility["label_mobile_across_materialized_bank"]).sum()
        ),
        "minimum_donor_pseudo_case_fraction": float(
            mobility["pseudo_case_fraction"].min()
        ),
        "maximum_donor_pseudo_case_fraction": float(
            mobility["pseudo_case_fraction"].max()
        ),
        "all_experiment_balance_theoretically_optimal": bool(
            design["max_absolute_experiment_donor_count_imbalance"].le(1).all()
        ),
        "all_assignments_all_20_bins_estimable": bool(
            design["all_20_bins_design_estimable"].all()
        ),
        "minimum_full_residual_df": int(design["minimum_full_residual_df"].min()),
        "maximum_condition_vif": float(design["maximum_condition_vif"].max()),
        "minimum_condition_information": float(
            design["minimum_condition_information"].min()
        ),
        "minimum_condition_information_fraction": float(
            design["minimum_condition_information_fraction"].min()
        ),
        "minimum_case_donors_in_any_bin": int(
            design["minimum_case_donors_in_any_bin"].min()
        ),
        "minimum_control_donors_in_any_bin": int(
            design["minimum_control_donors_in_any_bin"].min()
        ),
        "rank_relative_tolerance": float(RANK_RELATIVE_TOLERANCE),
        "minimum_reduced_retained_to_rank_threshold_ratio": float(
            (
                bin_long["reduced_minimum_retained_singular_value"]
                / bin_long["reduced_rank_absolute_threshold"]
            ).min()
        ),
        "maximum_reduced_discarded_to_rank_threshold_ratio": float(
            (
                bin_long["reduced_maximum_discarded_singular_value"]
                / bin_long["reduced_rank_absolute_threshold"]
            ).max()
        ),
        "availability_block_label_orbit_size_minimum": int(min(label_spaces)),
        "availability_block_label_orbit_size_median": float(np.median(label_spaces)),
        "availability_block_label_orbit_size_maximum": int(max(label_spaces)),
        "worst_availability_block_label_orbit_resolution": float(
            1.0 / min(label_spaces)
        ),
        "availability_block_label_orbit_scope": (
            "diagnostic_only_not_the_count_of_globally_experiment_constrained_assignments"
        ),
    }
    return design, assignment_manifest, mobility, experiment, bin_summary, summary


def _implementation_readiness(config: Mapping[str, Any]) -> dict[str, Any]:
    """Audit capabilities of the exact source hashes bound by the amendment.

    These booleans describe the bound implementation, not a generic theoretical
    possibility. Any code change requires a new append-only implementation freeze.
    """

    capabilities = {
        "formal_kernel_accepts_bin_specific_reduced_design": False,
        "formal_kernel_current_design_shape": "single_donor_by_covariate_matrix_reused_across_bins",
        "formal_kernel_uses_full_20_bin_availability_signature": False,
        "simultaneous_confidence_band_constructor_available": False,
        "supported_region_integrated_absolute_statistic_available": True,
        "pathway_family_single_step_maxT_available": True,
        "BY_adjustment_available": True,
        "timing_free_functional_only_output_contract_available": False,
    }
    required = {
        "formal_kernel_accepts_bin_specific_reduced_design": True,
        "formal_kernel_uses_full_20_bin_availability_signature": True,
        "simultaneous_confidence_band_constructor_available": True,
        "supported_region_integrated_absolute_statistic_available": True,
        "pathway_family_single_step_maxT_available": True,
        "BY_adjustment_available": True,
        "timing_free_functional_only_output_contract_available": True,
    }
    missing = [key for key, expected in required.items() if capabilities[key] != expected]
    return {
        "bound_source_hashes": {
            name: config["bindings"][name]["sha256"]
            for name in (
                "formal_kernel",
                "array_precision_kernel",
                "family_kernel",
                "decomposition_kernel",
            )
        },
        "capabilities": capabilities,
        "required_capabilities": required,
        "missing_required_capabilities": missing,
        "implementation_readiness_pass": not missing,
        "audit_basis": "read_only_source_and_public_api_audit_of_exact_bound_hashes",
    }


def _build_artifacts(
    *, repository_root: Path, config: Mapping[str, Any], amendment_dir: Path
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    validate_cb2_amendment_output(
        config_path=repository_root / CONFIG_FILE,
        repository_root=repository_root,
        output_dir=amendment_dir,
    )
    verify_cb2_amendment_bindings(repository_root, config)
    _require_equal(
        float(config["nuisance_model"]["linear_algebra"]["svd_rank_relative_tolerance"]),
        RANK_RELATIVE_TOLERANCE,
        "nuisance_model.linear_algebra.svd_rank_relative_tolerance",
    )
    _require_equal(
        float(config["nuisance_model"]["linear_algebra"]["pseudoinverse_rcond"]),
        RANK_RELATIVE_TOLERANCE,
        "nuisance_model.linear_algebra.pseudoinverse_rcond",
    )
    inputs = _load_design_inputs(repository_root, config)
    zero_overlap = _zero_overlap_audit(repository_root, config)
    assignment_config = config["pseudo_condition_assignment"]["construction"]
    assignments, construction = generate_constrained_assignments(
        inputs.incidence,
        n_case=int(config["population_and_units"]["pseudo_case_donors"]),
        n_assignments=int(config["stages"]["primary_balanced_null"]["replicates"]),
        seed=int(assignment_config["primary_seed"]),
        max_attempts=int(assignment_config["maximum_attempts_for_500_unique"]),
    )
    availability, availability_summary = _availability_audit(inputs)
    (
        design,
        assignment_manifest,
        assignment_mobility,
        experiment,
        bin_summary,
        design_summary,
    ) = (
        _audit_assignments(inputs, assignments, construction, config)
    )
    formal = config["formal_inference"]
    required_mappings = int(formal["residual_mappings_per_replicate"])
    availability_summary["requested_monte_carlo_null_mappings"] = required_mappings
    availability_summary["sampled_reference_resolution"] = float(
        1.0 / (required_mappings + 1)
    )
    availability_summary["enough_unique_residual_mappings"] = bool(
        availability_summary["n_unique_null_mappings_possible"] >= required_mappings
    )
    availability_summary[
        "diagnostic_availability_block_label_orbit_resolution_below_alpha"
    ] = bool(
        design_summary["worst_availability_block_label_orbit_resolution"]
        <= float(config["acceptance_endpoints"]["alpha"])
    )
    implementation = _implementation_readiness(config)
    experiment_counts = inputs.incidence.sum(axis=0).astype(int)
    n_active_experiments = int(np.sum(experiment_counts > 0))
    donor_experiment_counts = inputs.incidence.sum(axis=1).astype(int)
    design_pass = bool(
        design_summary["n_unique_assignment_hashes"] >= 500
        and design_summary["all_group_sizes_exact"]
        and design_summary["all_experiment_balance_theoretically_optimal"]
        and design_summary["n_label_immobile_donors"] == 0
        and design_summary["all_assignments_all_20_bins_estimable"]
        and availability_summary["enough_unique_residual_mappings"]
        and zero_overlap["zero_overlap_pass"]
    )
    implementation_pass = bool(implementation["implementation_readiness_pass"])
    cb2a_pass = bool(design_pass and implementation_pass)
    blocking = list(implementation["missing_required_capabilities"])
    if not design_pass:
        blocking.append("one_or_more_design_integrity_gates_failed")
    decision = {
        "schema_name": "trajpathmix_corebench_cb2a_estimability_decision",
        "schema_version": SCHEMA_VERSION,
        "preflight_id": PREFLIGHT_ID,
        "amendment_config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
        "amendment_file_sha256": _hash_file(amendment_dir / AMENDMENT_FILE),
        "amendment_build_record_sha256": _hash_file(
            amendment_dir / AMENDMENT_BUILD_RECORD_FILE
        ),
        "material_passport": {
            "research_question": config["material_passport"]["research_question"],
            "design": config["material_passport"]["design"],
            "population": "75_frozen_primary_complete_support_donors",
            "experiment_universe": "28_frozen_experiments_with_27_represented_in_primary",
            "nuisance_encoding": config["nuisance_model"]["primary_encoding"],
            "availability": "frozen_20_bin_full_signature_missing_bins_remain_NA",
            "randomization": "37_38_exact_experiment_incidence_balance_milp_materialized",
            "outcomes_read": False,
            "pathway_scoring_performed": False,
            "timing_computed": False,
            "interpretation": "design_and_implementation_readiness_only",
        },
        "population_audit": {
            "n_primary_donors": int(len(inputs.donors)),
            "n_frozen_experiments": int(len(inputs.experiments)),
            "n_experiments_represented_in_primary": n_active_experiments,
            "experiments_absent_from_primary": [
                inputs.experiments[index]
                for index in np.flatnonzero(experiment_counts == 0)
            ],
            "donors_by_n_experiments": {
                str(count): int(n)
                for count, n in sorted(Counter(donor_experiment_counts).items())
            },
            "n_primary_lines": int(inputs.line_count_by_donor.sum()),
            "maximum_lines_per_primary_donor": int(
                inputs.line_count_by_donor.max(initial=0)
            ),
            "primary_donors_with_multiple_lines": int(
                np.sum(inputs.line_count_by_donor > 1)
            ),
            "line_role": "nested_within_donor_not_independent",
        },
        "assignment_design": design_summary,
        "availability_mapping": availability_summary,
        "experiment_fraction_estimability": {
            "encoding": "donor_bin_experiment_cell_fraction_fixed_effects",
            "bin_specific_reduced_design": True,
            "all_500_assignments_all_20_bins_estimable": design_summary[
                "all_assignments_all_20_bins_estimable"
            ],
            "minimum_full_residual_df": design_summary["minimum_full_residual_df"],
            "maximum_condition_vif": design_summary["maximum_condition_vif"],
            "minimum_condition_information": design_summary[
                "minimum_condition_information"
            ],
            "minimum_condition_information_fraction": design_summary[
                "minimum_condition_information_fraction"
            ],
        },
        "counterfactual_hard_block_audit": {
            "experiment_by_signature_is_selected": False,
            "exact_incidence_pattern_orbit": availability_summary[
                "exact_experiment_incidence_by_signature"
            ]["residual_mapping_orbit_size"],
            "dominant_experiment_orbit": availability_summary[
                "dominant_experiment_by_signature"
            ]["residual_mapping_orbit_size"],
            "decision": "fail_closed_if_experiment_is_added_as_a_hard_mapping_block",
            "primary_rule": "experiment_adjusted_as_nuisance_only_full_signature_mapping",
        },
        "reference_resolution_interpretation": {
            "inferential_sampled_monte_carlo_resolution": availability_summary[
                "sampled_reference_resolution"
            ],
            "residual_mapping_orbit_size": availability_summary[
                "residual_mapping_orbit_size"
            ],
            "availability_block_label_orbit_resolution": (
                "diagnostic_only_not_an_exact_p_value_denominator_under_experiment_constraints_or_nonconstant_nuisance"
            ),
            "finite_sample_exact_claim_allowed": False,
        },
        "complete_null_claim_ceiling": config["acceptance_endpoints"][
            "complete_null_claim_ceiling"
        ],
        "zero_overlap_revalidation": zero_overlap,
        "implementation_readiness": implementation,
        "component_decisions": {
            "assignment_design": "pass" if design_pass else "fail",
            "full_availability_signature_mapping": (
                "pass_with_64_percent_immobile_donors_reported"
                if availability_summary["enough_unique_residual_mappings"]
                else "fail_degenerate"
            ),
            "experiment_fraction_structural_estimability": (
                "pass"
                if design_summary["all_assignments_all_20_bins_estimable"]
                else "fail"
            ),
            "formal_kernel_implementation_readiness": (
                "pass" if implementation_pass else "fail_closed"
            ),
        },
        "design_precheck_pass": design_pass,
        "implementation_readiness_pass": implementation_pass,
        "cb2a_pass": cb2a_pass,
        "cb2_500_start_allowed": cb2a_pass,
        "pathway_scoring_authorized_by_this_run": False,
        "decision": (
            "pass_cb2a_ready_for_separate_cb2_500_execution_gate"
            if cb2a_pass
            else "fail_closed_before_cb2_500"
        ),
        "blocking_reason_codes": blocking,
        "required_next_action": (
            "separate_cb2_500_execution_gate"
            if cb2a_pass
            else "append_only_theory_driven_implementation_freeze_for_bin_specific_nuisance_full_20_bin_signatures_simultaneous_bands_and_timing_free_outputs_then_rerun_cb2a"
        ),
        "prohibited_repairs": [
            "delete_difficult_experiments",
            "choose_a_prettier_donor_subset",
            "merge_availability_signatures_after_null_results",
            "substitute_dominant_experiment",
            "run_pathway_scoring_before_cb2a_pass",
        ],
        "expression_values_read": False,
        "pathway_outcomes_read": False,
        "pathway_scoring_performed": False,
        "real_condition_contrast_read_or_generated": False,
        "timing_fields_computed_or_output": False,
    }
    return (
        design,
        assignment_manifest,
        assignment_mobility,
        availability,
        experiment,
        bin_summary,
        decision,
    )


def _write_artifacts(
    directory: Path,
    artifacts: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        dict[str, Any],
    ],
) -> None:
    (
        design,
        assignments,
        assignment_mobility,
        availability,
        experiment,
        bins,
        decision,
    ) = artifacts
    tables = {
        PSEUDO_DESIGN_FILE: design,
        ASSIGNMENT_FILE: assignments,
        ASSIGNMENT_MOBILITY_FILE: assignment_mobility,
        AVAILABILITY_FILE: availability,
        EXPERIMENT_FILE: experiment,
        BIN_FILE: bins,
    }
    for name, frame in tables.items():
        (directory / name).write_text(_table_text(frame), encoding="utf-8")
    _write_json(decision, directory / DECISION_FILE)


def _build_record(
    *,
    config_path: Path,
    amendment_dir: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    artifact_names = (
        PSEUDO_DESIGN_FILE,
        ASSIGNMENT_FILE,
        ASSIGNMENT_MOBILITY_FILE,
        AVAILABILITY_FILE,
        EXPERIMENT_FILE,
        BIN_FILE,
        DECISION_FILE,
    )
    return {
        "schema_name": "trajpathmix_corebench_cb2a_design_preflight_build_record",
        "schema_version": SCHEMA_VERSION,
        "preflight_id": PREFLIGHT_ID,
        "config_file": CONFIG_FILE,
        "config_file_sha256": _hash_file(config_path),
        "config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
        "amendment_file_sha256": _hash_file(amendment_dir / AMENDMENT_FILE),
        "amendment_build_record_sha256": _hash_file(
            amendment_dir / AMENDMENT_BUILD_RECORD_FILE
        ),
        "implementation_file": IMPLEMENTATION_FILE,
        "implementation_sha256": _hash_file(Path(__file__).resolve()),
        "runtime": {
            "python": os.sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "milp_solver": "scipy.optimize.milp_HiGHS",
        },
        "artifacts": {
            name: {
                "sha256": _hash_file(artifact_dir / name),
                "bytes": int((artifact_dir / name).stat().st_size),
            }
            for name in artifact_names
        },
        "decision_sha256": _hash_file(artifact_dir / DECISION_FILE),
        "expression_values_read": False,
        "pathway_outcomes_read": False,
        "pathway_scoring_performed": False,
        "pseudo_conditions_generated_for_design_only": True,
        "real_condition_contrast_read_or_generated": False,
        "timing_computed": False,
        "evidence_revision_mode": "create_only_append_only",
        "materialized_assignment_manifest_is_authority": True,
        "solver_regeneration_required_for_integrity_validation": False,
    }


def build_and_write_cb2a_preflight(
    *,
    config_path: str | Path,
    repository_root: str | Path,
    amendment_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    root = Path(repository_root).resolve()
    amendment = Path(amendment_dir).resolve()
    config = load_cb2_amendment_config(config_file)
    output = Path(output_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"CB2a output exists: {output}")
    lock = output.parent / f".{output.name}.create.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FileExistsError(f"CB2a output is locked: {lock}") from exc
    temporary: Path | None = None
    try:
        os.close(descriptor)
        artifacts = _build_artifacts(
            repository_root=root, config=config, amendment_dir=amendment
        )
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
        )
        _write_artifacts(temporary, artifacts)
        record = _build_record(
            config_path=config_file,
            amendment_dir=amendment,
            artifact_dir=temporary,
        )
        _write_json(record, temporary / BUILD_RECORD_FILE)
        os.rename(temporary, output)
        temporary = None
        result = dict(record)
        result["output_dir"] = str(output)
        result["build_record_sha256"] = _hash_file(output / BUILD_RECORD_FILE)
        result["decision"] = artifacts[-1]["decision"]
        result["cb2a_pass"] = artifacts[-1]["cb2a_pass"]
        return result
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
        lock.unlink(missing_ok=True)


def _assert_json_finite(value: Any, label: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite JSON value at {label}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_json_finite(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_json_finite(item, f"{label}[{index}]")


def _validate_materialized_assignment_bank(
    *,
    output: Path,
    inputs: _DesignInputs,
    config: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    assignments = pd.read_csv(
        output / ASSIGNMENT_FILE,
        sep="\t",
        dtype="string",
        keep_default_na=False,
    )
    required_assignment_columns = {
        "assignment_id",
        "assignment_sha256",
        "donor_id",
        "pseudo_condition",
        "pseudo_case",
    }
    _require_equal(
        set(assignments.columns),
        required_assignment_columns,
        "materialized assignment columns",
    )
    design = pd.read_csv(
        output / PSEUDO_DESIGN_FILE,
        sep="\t",
        dtype="string",
        keep_default_na=False,
    )
    assignment_ids = design["assignment_id"].astype(str).tolist()
    expected_assignments = int(config["stages"]["primary_balanced_null"]["replicates"])
    _require_equal(len(assignment_ids), expected_assignments, "assignment audit rows")
    _require_equal(len(set(assignment_ids)), expected_assignments, "assignment ids")
    donor_set = set(inputs.donors)
    vectors: list[np.ndarray] = []
    hashes: list[str] = []
    for assignment_id in assignment_ids:
        group = assignments.loc[assignments["assignment_id"].eq(assignment_id)].copy()
        _require_equal(len(group), len(inputs.donors), f"{assignment_id} rows")
        _require_equal(
            set(group["donor_id"].astype(str)), donor_set, f"{assignment_id} donors"
        )
        _require_equal(
            int(group["donor_id"].nunique()),
            len(inputs.donors),
            f"{assignment_id} unique donors",
        )
        group["__case"] = _read_bool(group["pseudo_case"])
        _require(
            bool(
                (
                    group["pseudo_condition"].astype(str)
                    == np.where(group["__case"], "pseudo_case", "pseudo_control")
                ).all()
            ),
            f"{assignment_id} condition text differs from pseudo_case",
        )
        by_donor = group.set_index("donor_id").reindex(inputs.donors)
        vector = by_donor["__case"].to_numpy(dtype=np.uint8)
        digest = _assignment_hash(vector)
        observed_hashes = sorted(group["assignment_sha256"].astype(str).unique())
        _require_equal(observed_hashes, [digest], f"{assignment_id} assignment hash")
        design_hash = str(
            design.loc[design["assignment_id"].eq(assignment_id), "assignment_sha256"].iloc[0]
        )
        _require_equal(design_hash, digest, f"{assignment_id} design hash")
        vectors.append(vector)
        hashes.append(digest)
    _require_equal(len(set(hashes)), expected_assignments, "unique assignment hashes")
    matrix = np.stack(vectors, axis=0)
    n_case = int(config["population_and_units"]["pseudo_case_donors"])
    _require(
        bool(np.all(matrix.sum(axis=1) == n_case)),
        "Materialized assignments violate the 37-donor case-group size",
    )
    experiment_totals = inputs.incidence.sum(axis=0).astype(int)
    case_by_experiment = matrix @ inputs.incidence
    active = experiment_totals > 0
    lower = np.floor(experiment_totals / 2).astype(int)
    upper = np.ceil(experiment_totals / 2).astype(int)
    _require(
        bool(
            np.all(case_by_experiment[:, active] >= lower[active])
            and np.all(case_by_experiment[:, active] <= upper[active])
        ),
        "Materialized assignments violate frozen experiment balance",
    )

    mobility = pd.read_csv(
        output / ASSIGNMENT_MOBILITY_FILE,
        sep="\t",
        dtype="string",
        keep_default_na=False,
    ).set_index("donor_id")
    _require_equal(set(mobility.index.astype(str)), donor_set, "mobility donors")
    case_counts = matrix.sum(axis=0).astype(int)
    for donor_index, donor in enumerate(inputs.donors):
        row = mobility.loc[donor]
        observed_case = int(row["n_pseudo_case_assignments"])
        _require_equal(observed_case, int(case_counts[donor_index]), f"{donor} mobility")
        _require_equal(
            int(row["n_pseudo_control_assignments"]),
            expected_assignments - observed_case,
            f"{donor} control mobility",
        )
        expected_mobile = 0 < observed_case < expected_assignments
        observed_mobile = str(row["label_mobile_across_materialized_bank"]).lower()
        _require_equal(
            observed_mobile,
            str(expected_mobile).lower(),
            f"{donor} label mobility",
        )
    _require(
        bool(np.all((case_counts > 0) & (case_counts < expected_assignments))),
        "At least one donor is label-immobile in the materialized assignment bank",
    )
    _require_equal(
        int(decision["assignment_design"]["n_unique_assignment_hashes"]),
        expected_assignments,
        "decision unique assignments",
    )
    _require_equal(
        int(decision["assignment_design"]["n_label_immobile_donors"]),
        0,
        "decision label-immobile donors",
    )
    return {
        "n_assignments": expected_assignments,
        "n_donors": len(inputs.donors),
        "n_unique_hashes": len(set(hashes)),
        "n_label_immobile_donors": int(
            np.sum((case_counts == 0) | (case_counts == expected_assignments))
        ),
    }


def validate_cb2a_preflight_output(
    *,
    config_path: str | Path,
    repository_root: str | Path,
    amendment_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    root = Path(repository_root).resolve()
    amendment = Path(amendment_dir).resolve()
    output = Path(output_dir).resolve()
    if not output.is_dir():
        raise FileNotFoundError(output)
    expected_names = {
        PSEUDO_DESIGN_FILE,
        ASSIGNMENT_FILE,
        ASSIGNMENT_MOBILITY_FILE,
        AVAILABILITY_FILE,
        EXPERIMENT_FILE,
        BIN_FILE,
        DECISION_FILE,
        BUILD_RECORD_FILE,
    }
    _require_equal(
        {path.name for path in output.iterdir() if path.is_file()},
        expected_names,
        "output file set",
    )
    config = load_cb2_amendment_config(config_file)
    validate_cb2_amendment_output(
        config_path=config_file,
        repository_root=root,
        output_dir=amendment,
    )
    verify_cb2_amendment_bindings(root, config)
    artifact_names = {
        PSEUDO_DESIGN_FILE,
        ASSIGNMENT_FILE,
        ASSIGNMENT_MOBILITY_FILE,
        AVAILABILITY_FILE,
        EXPERIMENT_FILE,
        BIN_FILE,
        DECISION_FILE,
    }
    observed_record = json.loads(
        (output / BUILD_RECORD_FILE).read_text(encoding="utf-8")
    )
    _assert_json_finite(observed_record, BUILD_RECORD_FILE)
    for key, expected in {
        "schema_name": "trajpathmix_corebench_cb2a_design_preflight_build_record",
        "schema_version": SCHEMA_VERSION,
        "preflight_id": PREFLIGHT_ID,
        "config_file": CONFIG_FILE,
        "config_file_sha256": _hash_file(config_file),
        "config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
        "amendment_file_sha256": _hash_file(amendment / AMENDMENT_FILE),
        "amendment_build_record_sha256": _hash_file(
            amendment / AMENDMENT_BUILD_RECORD_FILE
        ),
        "implementation_file": IMPLEMENTATION_FILE,
        "implementation_sha256": _hash_file(Path(__file__).resolve()),
        "expression_values_read": False,
        "pathway_outcomes_read": False,
        "pathway_scoring_performed": False,
        "pseudo_conditions_generated_for_design_only": True,
        "real_condition_contrast_read_or_generated": False,
        "timing_computed": False,
        "evidence_revision_mode": "create_only_append_only",
        "materialized_assignment_manifest_is_authority": True,
        "solver_regeneration_required_for_integrity_validation": False,
    }.items():
        _require_equal(observed_record.get(key), expected, f"build_record.{key}")
    _require_equal(
        set(observed_record.get("artifacts", {})),
        artifact_names,
        "build_record artifact set",
    )
    for name in artifact_names:
        metadata = observed_record["artifacts"][name]
        _require_equal(
            metadata.get("sha256"), _hash_file(output / name), f"{name} sha256"
        )
        _require_equal(
            int(metadata.get("bytes")),
            int((output / name).stat().st_size),
            f"{name} bytes",
        )
    observed_decision = json.loads((output / DECISION_FILE).read_text(encoding="utf-8"))
    _assert_json_finite(observed_decision, DECISION_FILE)
    _require_equal(
        observed_record.get("decision_sha256"),
        _hash_file(output / DECISION_FILE),
        "decision_sha256",
    )
    for key, expected in {
        "amendment_config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
        "amendment_file_sha256": _hash_file(amendment / AMENDMENT_FILE),
        "amendment_build_record_sha256": _hash_file(
            amendment / AMENDMENT_BUILD_RECORD_FILE
        ),
        "design_precheck_pass": True,
        "implementation_readiness_pass": False,
        "cb2a_pass": False,
        "cb2_500_start_allowed": False,
        "pathway_scoring_authorized_by_this_run": False,
        "pathway_scoring_performed": False,
        "real_condition_contrast_read_or_generated": False,
        "timing_fields_computed_or_output": False,
        "decision": "fail_closed_before_cb2_500",
    }.items():
        _require_equal(observed_decision.get(key), expected, f"decision.{key}")
    forbidden_event_keys = {
        "onset",
        "duration",
        "phase_shift",
        "peak_location",
        "heterochrony",
    }

    def collect_keys(value: Any) -> set[str]:
        if isinstance(value, Mapping):
            return set(map(str, value)) | set().union(
                *(collect_keys(item) for item in value.values()), set()
            )
        if isinstance(value, list):
            return set().union(*(collect_keys(item) for item in value), set())
        return set()

    _require(
        not (collect_keys(observed_decision) & forbidden_event_keys),
        "CB2a decision contains forbidden timing-event output fields",
    )
    inputs = _load_design_inputs(root, config)
    manifest_validation = _validate_materialized_assignment_bank(
        output=output,
        inputs=inputs,
        config=config,
        decision=observed_decision,
    )
    availability = pd.read_csv(output / AVAILABILITY_FILE, sep="\t", dtype="string")
    signature_counts = Counter(inputs.signatures)
    _require_equal(len(availability), len(signature_counts), "availability audit rows")
    observed_signature_counts = {
        str(row.availability_signature): int(row.n_donors)
        for row in availability.itertuples(index=False)
    }
    _require_equal(
        observed_signature_counts,
        dict(sorted(signature_counts.items())),
        "availability signature donor counts",
    )
    signature_orbit = math.prod(math.factorial(value) for value in signature_counts.values())
    _require_equal(
        int(observed_decision["availability_mapping"]["residual_mapping_orbit_size"]),
        int(signature_orbit),
        "decision residual mapping orbit",
    )
    experiment = pd.read_csv(output / EXPERIMENT_FILE, sep="\t", dtype="string")
    _require_equal(
        experiment["experiment_id"].astype(str).tolist(),
        list(inputs.experiments),
        "experiment audit order",
    )
    _require_equal(
        experiment["primary_cohort_donors"].astype(int).tolist(),
        inputs.incidence.sum(axis=0).astype(int).tolist(),
        "experiment audit primary donor counts",
    )
    bins = pd.read_csv(output / BIN_FILE, sep="\t", dtype="string")
    _require_equal(len(bins), 20, "bin audit rows")
    _require(
        bool(bins["n_assignments_passing"].astype(int).eq(500).all()),
        "A materialized bin has fewer than 500 passing assignments",
    )
    result = dict(observed_record)
    result["output_dir"] = str(output)
    result["build_record_sha256"] = _hash_file(output / BUILD_RECORD_FILE)
    result["validation_status"] = (
        "pass_cb2a_materialized_manifest_constraints_and_artifact_integrity"
    )
    result["manifest_validation"] = manifest_validation
    result["decision"] = observed_decision["decision"]
    result["cb2a_pass"] = bool(observed_decision["cb2a_pass"])
    return result


__all__ = [
    "ASSIGNMENT_FILE",
    "ASSIGNMENT_MOBILITY_FILE",
    "AVAILABILITY_FILE",
    "BIN_FILE",
    "BUILD_RECORD_FILE",
    "DECISION_FILE",
    "EXPERIMENT_FILE",
    "PSEUDO_DESIGN_FILE",
    "build_and_write_cb2a_preflight",
    "generate_constrained_assignments",
    "validate_cb2a_preflight_output",
]
