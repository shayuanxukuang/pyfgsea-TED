from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse.csgraph import connected_components
from scipy.stats import norm, spearmanr
from sklearn.decomposition import PCA

from pyfgsea.trajpathmix_endoderm_benchmark_freeze import (
    FROZEN_CONFIG_PAYLOAD_SHA256,
    load_endoderm_benchmark_freeze_config,
)


COORDINATES_FILE = "endoderm_blinded_trajectory_coordinates_v1.tsv.gz"
HVG_FILE = "endoderm_experiment_adjusted_hvg_rank_v1.tsv"
LOADINGS_FILE = "endoderm_primary_pca_loadings_v1.tsv"
DAY_VALIDATION_FILE = "endoderm_locked_day_validation_v1.tsv"
EXPERIMENT_VALIDATION_FILE = "endoderm_experiment_orientation_v1.tsv"
DONOR_BIN_FILE = "endoderm_primary_donor_bin_support_v1.tsv"
GRID_FILE = "endoderm_grid_estimability_v1.tsv"
DECISION_FILE = "ENDODERM_A1_STRUCTURAL_DECISION_2026-07-14.json"
BUILD_RECORD_FILE = "endoderm_a1_structural_preflight_build_record_v1.json"


def _hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(value: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def longest_true_run(values: Iterable[bool]) -> tuple[int, int]:
    """Return the half-open indices of the longest contiguous true run."""

    best_start = best_end = current_start = 0
    in_run = False
    for index, value in enumerate(values):
        if value and not in_run:
            current_start = index
            in_run = True
        if in_run and (not value):
            if index - current_start > best_end - best_start:
                best_start, best_end = current_start, index
            in_run = False
    length = index + 1 if "index" in locals() else 0
    if in_run and length - current_start > best_end - best_start:
        best_start, best_end = current_start, length
    return best_start, best_end


def _rescale01(values: np.ndarray) -> np.ndarray:
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    if not math.isfinite(minimum) or not math.isfinite(maximum) or maximum <= minimum:
        raise ValueError("Trajectory coordinate cannot be rescaled to [0, 1]")
    return ((values - minimum) / (maximum - minimum)).astype(np.float64)


def _marker_rows(feature_ids: pd.Index, markers: list[str]) -> list[int]:
    symbols = [str(value).split("_", 1)[1] if "_" in str(value) else str(value) for value in feature_ids]
    symbol_to_rows: dict[str, list[int]] = {}
    for index, symbol in enumerate(symbols):
        symbol_to_rows.setdefault(symbol, []).append(index)
    missing = [marker for marker in markers if marker not in symbol_to_rows]
    if missing:
        raise ValueError(f"Frozen direction markers are missing: {missing}")
    return [row for marker in markers for row in symbol_to_rows[marker]]


def _experiment_adjusted_variance(
    expression: np.ndarray, experiment_codes: np.ndarray, n_experiments: int
) -> np.ndarray:
    indicator = np.zeros((expression.shape[1], n_experiments), dtype=np.float32)
    indicator[np.arange(expression.shape[1]), experiment_codes] = 1.0
    group_sizes = indicator.sum(axis=0, dtype=np.float64)
    group_sums = expression @ indicator
    total_squares = np.einsum(
        "ij,ij->i", expression, expression, dtype=np.float64, optimize=True
    )
    explained = np.sum(
        np.square(group_sums, dtype=np.float64) / group_sizes[None, :], axis=1
    )
    residual_df = expression.shape[1] - n_experiments
    return np.maximum(total_squares - explained, 0.0) / residual_df


def _method_day_summary(
    coordinates: pd.DataFrame, days: list[str]
) -> pd.DataFrame:
    grouped = (
        coordinates.groupby(["donor_id", "day"], observed=True)
        .agg(
            n_cells=("cell_id", "size"),
            primary_median=("pseudotime_primary", "median"),
            dpt_median=("pseudotime_dpt", "median"),
        )
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for day in days:
        subset = grouped.loc[grouped["day"] == day]
        rows.append(
            {
                "day": day,
                "n_cells": int(coordinates["day"].eq(day).sum()),
                "n_donors": int(subset["donor_id"].nunique()),
                "median_of_donor_primary_medians": float(
                    subset["primary_median"].median()
                ),
                "median_of_donor_dpt_medians": float(subset["dpt_median"].median()),
                "day_used_for_trajectory": False,
                "allowed_use": "locked_validation_only",
            }
        )
    return pd.DataFrame(rows)


def _experiment_orientation(
    coordinates: pd.DataFrame, days: list[str]
) -> pd.DataFrame:
    day_code = {day: index for index, day in enumerate(days)}
    rows: list[dict[str, Any]] = []
    for experiment_id, subset in coordinates.groupby("experiment_id", observed=True):
        medians = (
            subset.groupby("day", observed=True)
            .agg(
                primary=("pseudotime_primary", "median"),
                dpt=("pseudotime_dpt", "median"),
            )
            .reset_index()
        )
        medians["day_code"] = medians["day"].map(day_code)
        informative = len(medians) >= 2
        primary_rho = (
            float(spearmanr(medians["day_code"], medians["primary"]).statistic)
            if informative
            else float("nan")
        )
        dpt_rho = (
            float(spearmanr(medians["day_code"], medians["dpt"]).statistic)
            if informative
            else float("nan")
        )
        rows.append(
            {
                "experiment_id": experiment_id,
                "n_cells": int(len(subset)),
                "n_days": int(len(medians)),
                "primary_day_spearman": primary_rho,
                "dpt_day_spearman": dpt_rho,
                "primary_orientation_positive": bool(informative and primary_rho > 0),
                "dpt_orientation_positive": bool(informative and dpt_rho > 0),
                "primary_secondary_orientation_agree": bool(
                    informative and primary_rho > 0 and dpt_rho > 0
                ),
                "informative": bool(informative),
            }
        )
    return pd.DataFrame(rows).sort_values("experiment_id", ignore_index=True)


def _grid_support(
    coordinates: pd.DataFrame,
    primary_donors: list[str],
    *,
    n_bins: int,
    min_cells: int,
    min_group_donors: int,
    alpha: float,
    power: float,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[int, int]]:
    subset = coordinates.loc[
        coordinates["donor_id"].isin(primary_donors)
        & coordinates["in_primary_component"]
    ].copy()
    subset["bin_id"] = np.minimum(
        (subset["pseudotime_primary"].to_numpy() * n_bins).astype(int), n_bins - 1
    )
    counts = (
        subset.groupby(["donor_id", "bin_id"], observed=True)
        .size()
        .rename("cell_count")
        .reset_index()
    )
    grid = pd.MultiIndex.from_product(
        [primary_donors, range(n_bins)], names=["donor_id", "bin_id"]
    ).to_frame(index=False)
    donor_bin = grid.merge(
        counts, on=["donor_id", "bin_id"], how="left", validate="one_to_one"
    )
    donor_bin["cell_count"] = donor_bin["cell_count"].fillna(0).astype(int)
    donor_bin["available"] = donor_bin["cell_count"].ge(min_cells)

    n_donors = len(primary_donors)
    n_a = n_donors // 2
    n_b = n_donors - n_a
    z_alpha = norm.ppf(1.0 - alpha / 2.0)
    z_power = norm.ppf(power)
    rows: list[dict[str, Any]] = []
    for bin_id, bin_frame in donor_bin.groupby("bin_id", sort=True):
        available = int(bin_frame["available"].sum())
        missing = n_donors - available
        worst_a = max(0, n_a - missing)
        worst_b = max(0, n_b - missing)
        passes = worst_a >= min_group_donors and worst_b >= min_group_donors
        mde = (
            float((z_alpha + z_power) * math.sqrt(1.0 / worst_a + 1.0 / worst_b))
            if worst_a > 0 and worst_b > 0
            else float("inf")
        )
        rows.append(
            {
                "bin_id": int(bin_id),
                "bin_left": float(bin_id / n_bins),
                "bin_right": float((bin_id + 1) / n_bins),
                "n_primary_donors": n_donors,
                "n_available_donors": available,
                "n_missing_donors": missing,
                "worst_case_group_a_donors": worst_a,
                "worst_case_group_b_donors": worst_b,
                "minimum_cells_per_donor_bin": min_cells,
                "minimum_donors_per_pseudo_condition": min_group_donors,
                "common_support_pass": bool(passes),
                "donor_equal_weight_ratio": 1.0,
                "worst_case_standardized_mde": mde,
            }
        )
    grid_frame = pd.DataFrame(rows)
    segment = longest_true_run(grid_frame["common_support_pass"].tolist())
    grid_frame["selected_longest_segment"] = False
    if segment[1] > segment[0]:
        grid_frame.loc[
            grid_frame["bin_id"].between(segment[0], segment[1] - 1),
            "selected_longest_segment",
        ] = True
    return donor_bin, grid_frame, segment


def run_endoderm_structural_preflight(
    *,
    benchmark_config_path: str | Path,
    repository_root: str | Path,
    output_dir: str | Path,
    seed: int = 20260714,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config_file = Path(benchmark_config_path).resolve()
    config = load_endoderm_benchmark_freeze_config(config_file)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Endoderm A1 output exists: {output}")

    acquisition = root / "data_external/trajpathmix_acquisitions/hipsci_endoderm_125_v2"
    raw_archive = acquisition / "source/raw_counts.csv.zip"
    metadata_path = acquisition / "source/cell_metadata_cols.tsv"
    raw_schema_path = (
        root
        / "data_external/trajpathmix_endoderm_raw_schema_audit_v1"
        / "endoderm_raw_schema_audit_v1.json"
    )
    schema = json.loads(raw_schema_path.read_text(encoding="utf-8"))
    required_schema = {
        "raw_archive_local_sha256": "f85c2e6b768e3096bd3a9150545ce92e9acf5050e2a8df9f3df52cea18e7f90b",
        "raw_archive_publisher_md5": "86044a350cc08ec7c0ad4068206bda0e",
        "n_features": 11231,
        "n_matrix_cell_columns": 36044,
        "cell_axis_exact_set": True,
        "cell_axis_exact_order": True,
        "fractional_count_contract_confirmed": True,
        "integer_coercion_allowed": False,
        "pathway_outcomes_read": False,
    }
    for key, expected in required_schema.items():
        if schema.get(key) != expected:
            raise ValueError(f"Raw-schema gate mismatch for {key}")

    ids = config["identifiers"]
    construction_columns = [
        "cell_name",
        ids["donor_id_column"],
        ids["line_id_column"],
        ids["experiment_id_column"],
        "size_factor",
        "total_counts_endogenous",
    ]
    metadata = pd.read_csv(
        metadata_path, sep="\t", usecols=construction_columns, dtype={"cell_name": "string"}
    ).rename(
        columns={
            ids["donor_id_column"]: "donor_id",
            ids["line_id_column"]: "line_id",
            ids["experiment_id_column"]: "experiment_id",
        }
    )
    if "day" in metadata.columns:
        raise AssertionError("Day leaked into trajectory construction metadata")

    counts_frame = pd.read_csv(
        raw_archive,
        compression="zip",
        index_col=0,
        dtype={str(cell_id): np.float32 for cell_id in metadata["cell_name"]},
    )
    if counts_frame.shape != (schema["n_features"], schema["n_matrix_cell_columns"]):
        raise ValueError("Raw-count matrix shape changed after schema freeze")
    if counts_frame.columns.astype(str).tolist() != metadata["cell_name"].astype(str).tolist():
        raise ValueError("Raw-count cell order changed after schema freeze")
    expression = counts_frame.to_numpy(copy=False)
    if np.any(expression < 0) or not np.isfinite(expression).all():
        raise ValueError("Raw fractional matrix contains negative or nonfinite values")
    size_factor = metadata["size_factor"].to_numpy(dtype=np.float32)
    reference_library = float(
        np.median(
            metadata["total_counts_endogenous"].to_numpy(dtype=np.float64)
            / size_factor.astype(np.float64)
        )
    )
    expression *= (1_000_000.0 / reference_library / size_factor)[None, :]
    expression += 1.0
    np.log2(expression, out=expression)

    experiment_codes, experiment_levels = pd.factorize(
        metadata["experiment_id"], sort=True
    )
    adjusted_variance = _experiment_adjusted_variance(
        expression, experiment_codes, len(experiment_levels)
    )
    hvg_order = np.argsort(-adjusted_variance, kind="stable")
    primary_hvg_count = int(config["trajectory_preflight"]["primary_hvg_count"])
    selected = hvg_order[:primary_hvg_count]
    pca = PCA(n_components=20, svd_solver="randomized", random_state=seed)
    pcs = pca.fit_transform(expression[selected, :].T).astype(np.float32)

    direction = config["trajectory_preflight"]["direction_rule"]
    start_rows = _marker_rows(counts_frame.index, list(direction["start_markers"]))
    end_rows = _marker_rows(counts_frame.index, list(direction["end_markers"]))
    start_score = expression[start_rows, :].mean(axis=0)
    end_score = expression[end_rows, :].mean(axis=0)
    differentiation_score = end_score - start_score
    pc1 = pcs[:, 0].astype(np.float64)
    pc1_marker_rho = float(spearmanr(pc1, differentiation_score).statistic)
    primary_flipped = pc1_marker_rho < 0
    if primary_flipped:
        pc1 = -pc1
        pcs[:, 0] *= -1
        pca.components_[0, :] *= -1
        pc1_marker_rho = -pc1_marker_rho
    primary_pseudotime = _rescale01(pc1)

    neighbor_adata = ad.AnnData(
        X=np.zeros((len(metadata), 1), dtype=np.float32),
        obs=metadata.set_index("cell_name", drop=False),
    )
    neighbor_adata.obsm["X_pca"] = pcs
    sc.pp.neighbors(
        neighbor_adata,
        n_neighbors=int(config["trajectory_preflight"]["secondary_method_n_neighbors"]),
        use_rep="X_pca",
        random_state=seed,
    )
    n_components, component_labels = connected_components(
        neighbor_adata.obsp["connectivities"], directed=False
    )
    component_sizes = np.bincount(component_labels)
    largest_component = int(np.argmax(component_sizes))
    in_primary_component = component_labels == largest_component
    component_fraction = float(in_primary_component.mean())

    component_indices = np.flatnonzero(in_primary_component)
    dpt_adata = ad.AnnData(
        X=np.zeros((len(component_indices), 1), dtype=np.float32),
        obs=metadata.iloc[component_indices].set_index("cell_name", drop=False),
    )
    dpt_adata.obsm["X_pca"] = pcs[component_indices]
    sc.pp.neighbors(
        dpt_adata,
        n_neighbors=int(config["trajectory_preflight"]["secondary_method_n_neighbors"]),
        use_rep="X_pca",
        random_state=seed,
    )
    sc.tl.diffmap(dpt_adata, n_comps=15, random_state=seed)
    dpt_adata.uns["iroot"] = int(
        np.argmax((start_score - end_score)[component_indices])
    )
    sc.tl.dpt(dpt_adata, n_dcs=10)
    dpt = np.full(len(metadata), np.nan, dtype=np.float64)
    dpt[component_indices] = dpt_adata.obs["dpt_pseudotime"].to_numpy(dtype=float)
    dpt_marker_rho = float(
        spearmanr(dpt[component_indices], differentiation_score[component_indices]).statistic
    )
    dpt_flipped = dpt_marker_rho < 0
    if dpt_flipped:
        dpt[component_indices] = 1.0 - dpt[component_indices]
        dpt_marker_rho = -dpt_marker_rho
    method_rho = float(
        spearmanr(primary_pseudotime[component_indices], dpt[component_indices]).statistic
    )

    coordinates = metadata[
        ["cell_name", "donor_id", "line_id", "experiment_id"]
    ].rename(columns={"cell_name": "cell_id"})
    coordinates["in_primary_component"] = in_primary_component
    coordinates["pseudotime_primary"] = primary_pseudotime
    coordinates["pseudotime_dpt"] = dpt
    coordinates["day_used_for_trajectory"] = False

    # Day is read only after both trajectories above are fixed.
    day_frame = pd.read_csv(
        metadata_path, sep="\t", usecols=["cell_name", ids["day_column"]], dtype="string"
    ).rename(columns={ids["day_column"]: "day"})
    if day_frame["cell_name"].astype(str).tolist() != coordinates["cell_id"].astype(str).tolist():
        raise ValueError("Locked day-validation cell order mismatch")
    coordinates["day"] = day_frame["day"].astype(str).values
    days = list(ids["expected_day_order"])
    day_validation = _method_day_summary(coordinates, days)
    experiment_validation = _experiment_orientation(coordinates, days)

    donor_cohort = pd.read_csv(
        root
        / "data_external/trajpathmix_endoderm_benchmark_freeze_v1"
        / "endoderm_donor_cohort_membership_v1.tsv",
        sep="\t",
    )
    primary_donors = sorted(
        donor_cohort.loc[donor_cohort["primary_complete_support"], "donor_id"]
        .astype(str)
        .tolist()
    )
    preflight = config["trajectory_preflight"]
    acceptance = config["acceptance_policy"]
    donor_bin, grid, segment = _grid_support(
        coordinates,
        primary_donors,
        n_bins=int(preflight["fixed_common_grid_bins"]),
        min_cells=int(preflight["minimum_cells_per_donor_bin"]),
        min_group_donors=int(preflight["minimum_donors_per_pseudo_condition"]),
        alpha=float(acceptance["alpha"]),
        power=float(acceptance["minimum_power_at_target_effect"]),
    )
    segment_length = segment[1] - segment[0]
    segment_span = segment_length / int(preflight["fixed_common_grid_bins"])
    selected_grid = grid.loc[grid["selected_longest_segment"]]
    max_mde = (
        float(selected_grid["worst_case_standardized_mde"].max())
        if len(selected_grid)
        else float("inf")
    )
    primary_day_values = day_validation["median_of_donor_primary_medians"].to_numpy()
    dpt_day_values = day_validation["median_of_donor_dpt_medians"].to_numpy()
    primary_day_monotone = bool(np.all(np.diff(primary_day_values) > 0))
    dpt_day_monotone = bool(np.all(np.diff(dpt_day_values) > 0))
    informative_experiments = experiment_validation.loc[
        experiment_validation["informative"]
    ]
    primary_experiment_fraction = float(
        informative_experiments["primary_orientation_positive"].mean()
    )
    dpt_experiment_fraction = float(
        informative_experiments["dpt_orientation_positive"].mean()
    )
    method_orientation_fraction = float(
        informative_experiments["primary_secondary_orientation_agree"].mean()
    )
    gates = {
        "largest_component_fraction": {
            "observed": component_fraction,
            "threshold": float(
                preflight["eligible_linear_lineage"][
                    "minimum_cell_fraction_in_largest_component"
                ]
            ),
            "pass": component_fraction
            >= float(
                preflight["eligible_linear_lineage"][
                    "minimum_cell_fraction_in_largest_component"
                ]
            ),
        },
        "primary_donor_median_day_order_strict": {
            "observed": primary_day_monotone,
            "threshold": True,
            "pass": primary_day_monotone,
        },
        "dpt_donor_median_day_order_strict": {
            "observed": dpt_day_monotone,
            "threshold": True,
            "pass": dpt_day_monotone,
        },
        "primary_experiment_orientation_agreement": {
            "observed": primary_experiment_fraction,
            "threshold": float(
                preflight["locked_validation_gates"][
                    "minimum_experiment_orientation_agreement_fraction"
                ]
            ),
            "pass": primary_experiment_fraction
            >= float(
                preflight["locked_validation_gates"][
                    "minimum_experiment_orientation_agreement_fraction"
                ]
            ),
        },
        "primary_secondary_rank_correlation": {
            "observed": method_rho,
            "threshold": float(
                preflight["locked_validation_gates"][
                    "minimum_primary_secondary_rank_correlation"
                ]
            ),
            "pass": method_rho
            >= float(
                preflight["locked_validation_gates"][
                    "minimum_primary_secondary_rank_correlation"
                ]
            ),
        },
        "primary_secondary_experiment_orientation_agreement": {
            "observed": method_orientation_fraction,
            "threshold": float(
                preflight["locked_validation_gates"][
                    "minimum_primary_secondary_orientation_agreement_fraction"
                ]
            ),
            "pass": method_orientation_fraction
            >= float(
                preflight["locked_validation_gates"][
                    "minimum_primary_secondary_orientation_agreement_fraction"
                ]
            ),
        },
        "minimum_contiguous_supported_bins": {
            "observed": segment_length,
            "threshold": int(preflight["minimum_contiguous_supported_bins"]),
            "pass": segment_length
            >= int(preflight["minimum_contiguous_supported_bins"]),
        },
        "minimum_supported_pseudotime_span": {
            "observed": segment_span,
            "threshold": float(preflight["minimum_supported_pseudotime_span"]),
            "pass": segment_span
            >= float(preflight["minimum_supported_pseudotime_span"]),
        },
        "maximum_standardized_mde": {
            "observed": max_mde,
            "threshold": float(acceptance["maximum_standardized_mde"]),
            "pass": max_mde <= float(acceptance["maximum_standardized_mde"]),
        },
    }
    a1_pass = all(bool(item["pass"]) for item in gates.values())

    hvg = pd.DataFrame(
        {
            "feature_id": counts_frame.index.astype(str),
            "gene_symbol": [
                str(value).split("_", 1)[1] if "_" in str(value) else str(value)
                for value in counts_frame.index
            ],
            "experiment_adjusted_residual_variance": adjusted_variance,
        }
    )
    rank = np.empty(len(hvg), dtype=int)
    rank[hvg_order] = np.arange(1, len(hvg) + 1)
    hvg["hvg_rank"] = rank
    hvg["selected_primary_500"] = hvg["hvg_rank"].le(primary_hvg_count)
    hvg = hvg.sort_values("hvg_rank", ignore_index=True)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{index}" for index in range(1, pca.n_components_ + 1)],
    )
    loadings.insert(0, "feature_id", counts_frame.index[selected].astype(str))

    decision = {
        "schema_name": "trajpathmix_endoderm_a1_structural_decision",
        "schema_version": "1.0.0",
        "decision_id": "endoderm_a1_structural_decision_2026-07-14",
        "benchmark_config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
        "raw_schema_audit_sha256": _hash_file(raw_schema_path),
        "seed": seed,
        "normalization": config["fractional_count_contract"][
            "primary_cell_normalization"
        ],
        "normalization_reference_library": reference_library,
        "primary_hvg_count": primary_hvg_count,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "primary_pc1_marker_spearman_after_orientation": pc1_marker_rho,
        "primary_pc1_flipped": primary_flipped,
        "dpt_marker_spearman_after_orientation": dpt_marker_rho,
        "dpt_flipped": dpt_flipped,
        "n_graph_components": int(n_components),
        "largest_component_cells": int(in_primary_component.sum()),
        "largest_component_fraction": component_fraction,
        "primary_secondary_spearman": method_rho,
        "primary_experiment_orientation_agreement_fraction": primary_experiment_fraction,
        "dpt_experiment_orientation_agreement_fraction": dpt_experiment_fraction,
        "primary_secondary_experiment_orientation_agreement_fraction": method_orientation_fraction,
        "selected_segment_start_bin": int(segment[0]),
        "selected_segment_end_bin_exclusive": int(segment[1]),
        "selected_segment_bins": int(segment_length),
        "selected_segment_span": float(segment_span),
        "selected_segment_max_standardized_mde": max_mde,
        "gates": gates,
        "a1_structural_status": "pass" if a1_pass else "fail_closed",
        "pathway_scoring_authorized_for_a2_a3": bool(a1_pass),
        "timing_estimand_status": "eligible_for_calibration" if a1_pass else "conditional_only",
        "phase_b_authorized": False,
        "day_used_for_trajectory": False,
        "deposited_trajectory_fields_used_for_trajectory": False,
        "pathway_outcomes_read": False,
        "pathway_scoring_performed": False,
        "next_gate": "a2_randomized_empirical_null_smoke500" if a1_pass else "stop_timing_calibration",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.create.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FileExistsError(f"Endoderm A1 output is locked: {lock_path}") from exc
    temporary: Path | None = None
    try:
        os.close(lock_fd)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
        )
        coordinates.drop(columns=["day"]).to_csv(
            temporary / COORDINATES_FILE,
            sep="\t",
            index=False,
            compression="gzip",
            lineterminator="\n",
        )
        hvg.to_csv(temporary / HVG_FILE, sep="\t", index=False, lineterminator="\n")
        loadings.to_csv(
            temporary / LOADINGS_FILE, sep="\t", index=False, lineterminator="\n"
        )
        day_validation.to_csv(
            temporary / DAY_VALIDATION_FILE,
            sep="\t",
            index=False,
            lineterminator="\n",
        )
        experiment_validation.to_csv(
            temporary / EXPERIMENT_VALIDATION_FILE,
            sep="\t",
            index=False,
            lineterminator="\n",
        )
        donor_bin.to_csv(
            temporary / DONOR_BIN_FILE, sep="\t", index=False, lineterminator="\n"
        )
        grid.to_csv(temporary / GRID_FILE, sep="\t", index=False, lineterminator="\n")
        _write_json(decision, temporary / DECISION_FILE)
        artifacts = {}
        for path in sorted(temporary.iterdir()):
            artifacts[path.name] = {"sha256": _hash_file(path), "size_bytes": path.stat().st_size}
        record = {
            "schema_name": "trajpathmix_endoderm_a1_structural_preflight_build_record",
            "schema_version": "1.0.0",
            "benchmark_config_file_sha256": _hash_file(config_file),
            "benchmark_config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
            "implementation_file": "pyfgsea/trajpathmix_endoderm_structural_preflight.py",
            "implementation_sha256": _hash_file(Path(__file__).resolve()),
            "raw_archive_sha256": schema["raw_archive_local_sha256"],
            "metadata_sha256": schema["metadata_local_sha256"],
            "artifacts": artifacts,
            "a1_structural_status": decision["a1_structural_status"],
            "pathway_outcomes_read": False,
            "pathway_scoring_performed": False,
            "day_used_for_trajectory": False,
        }
        _write_json(record, temporary / BUILD_RECORD_FILE)
        os.rename(temporary, output)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
        lock_path.unlink(missing_ok=True)
    result = dict(decision)
    result["output_dir"] = str(output)
    result["decision_sha256"] = _hash_file(output / DECISION_FILE)
    result["build_record_sha256"] = _hash_file(output / BUILD_RECORD_FILE)
    return result


__all__ = ["longest_true_run", "run_endoderm_structural_preflight"]
