from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .trajpathmix_corebench_coordinate import (
    BUILD_RECORD_FILE as MATERIALIZATION_BUILD_RECORD_FILE,
    COORDINATE_FILE,
    FOLD_AUDIT_FILE,
    RECEIPT_FILE as MATERIALIZATION_RECEIPT_FILE,
)
from .trajpathmix_corebench_freeze import (
    BUILD_RECORD_FILE as FREEZE_BUILD_RECORD_FILE,
    EXCLUSION_AUDIT_FILE,
    FROZEN_CONFIG_PAYLOAD_SHA256,
    GENE_FOLD_FILE,
    load_corebench_config,
    validate_corebench_freeze_output,
)


CELL_ORDER_FILE = "corebench_cell_order_v1.tsv"
DONOR_BIN_FILE = "corebench_donor_bin_availability_v1.tsv"
BIN_SUMMARY_FILE = "corebench_coordinate_bin_summary_v1.tsv"
EXPERIMENT_AUDIT_FILE = "corebench_coordinate_experiment_audit_v1.tsv"
VALIDATION_RECEIPT_FILE = "corebench_coordinate_validation_receipt_v1.json"
VALIDATION_BUILD_RECORD_FILE = "corebench_coordinate_validation_build_record_v1.json"

EXPECTED_CELLS = 36_044
EXPECTED_DONORS = 125
EXPECTED_LINES = 126
EXPECTED_EXPERIMENTS = 28
EXPECTED_COORDINATE_GENES = 335


def _hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise ValueError(
            f"CoreBench coordinate validation mismatch for {label}: "
            f"expected {expected!r}, observed {observed!r}"
        )


def _verify_artifact(path: Path, descriptor: Mapping[str, Any], label: str) -> None:
    _require(path.is_file(), f"Missing CoreBench coordinate artifact: {path}")
    _require_equal(path.stat().st_size, int(descriptor["bytes"]), f"{label}.bytes")
    _require_equal(_hash_file(path), descriptor["sha256"], f"{label}.sha256")


def _read_bool(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.lower()
    _require(
        bool(values.isin(["true", "false"]).all()),
        f"Boolean column {series.name!r} contains a non-boolean value",
    )
    return values.eq("true")


def _validate_materialization_artifacts(
    *,
    repository_root: Path,
    coordinate_dir: Path,
    freeze_record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_names = {
        COORDINATE_FILE,
        FOLD_AUDIT_FILE,
        MATERIALIZATION_RECEIPT_FILE,
        MATERIALIZATION_BUILD_RECORD_FILE,
    }
    _require_equal(
        {path.name for path in coordinate_dir.iterdir() if path.is_file()},
        expected_names,
        "coordinate_output_file_set",
    )
    receipt = json.loads(
        (coordinate_dir / MATERIALIZATION_RECEIPT_FILE).read_text(encoding="utf-8")
    )
    build_record = json.loads(
        (coordinate_dir / MATERIALIZATION_BUILD_RECORD_FILE).read_text(
            encoding="utf-8"
        )
    )
    _require_equal(
        receipt.get("schema_name"),
        "trajpathmix_corebench_coordinate_materialization_receipt",
        "materialization_receipt.schema_name",
    )
    _require_equal(
        build_record.get("schema_name"),
        "trajpathmix_corebench_coordinate_materialization_build_record",
        "materialization_build_record.schema_name",
    )
    _require_equal(
        receipt.get("config_payload_sha256"),
        FROZEN_CONFIG_PAYLOAD_SHA256,
        "materialization_receipt.config_payload_sha256",
    )
    _require_equal(
        build_record.get("config_payload_sha256"),
        FROZEN_CONFIG_PAYLOAD_SHA256,
        "materialization_build_record.config_payload_sha256",
    )
    _require_equal(
        receipt.get("corebench_freeze_build_record_sha256"),
        freeze_record["build_record_sha256"],
        "materialization_receipt.corebench_freeze_build_record_sha256",
    )
    _require_equal(
        set(build_record.get("artifacts", {})),
        {COORDINATE_FILE, FOLD_AUDIT_FILE, MATERIALIZATION_RECEIPT_FILE},
        "materialization_build_record.artifacts",
    )
    for name, descriptor in build_record["artifacts"].items():
        _verify_artifact(coordinate_dir / name, descriptor, name)
    _require_equal(
        receipt.get("coordinate_file_sha256"),
        build_record["artifacts"][COORDINATE_FILE]["sha256"],
        "coordinate_file_sha256",
    )
    implementation_path = repository_root / str(build_record["implementation_file"])
    _require_equal(
        _hash_file(implementation_path),
        build_record["implementation_sha256"],
        "materialization_implementation_sha256",
    )
    for key in (
        "pathway_outcomes_read",
        "pathway_scoring_performed",
    ):
        _require_equal(build_record.get(key), False, f"materialization_build_record.{key}")
    for key in (
        "real_pathway_outcomes_read",
        "pathway_scoring_performed",
        "pseudo_conditions_generated",
        "injection_results_generated",
    ):
        _require_equal(receipt.get(key), False, f"materialization_receipt.{key}")
    return receipt, build_record


def _validate_source_files(
    *,
    repository_root: Path,
    config: Mapping[str, Any],
    verify_raw_archive_hash: bool,
) -> tuple[Path, dict[str, Any]]:
    receipt_path = repository_root / config["bindings"]["acquisition_receipt"][
        "relative_path"
    ]
    _require_equal(
        _hash_file(receipt_path),
        config["bindings"]["acquisition_receipt"]["sha256"],
        "acquisition_receipt.sha256",
    )
    acquisition = json.loads(receipt_path.read_text(encoding="utf-8"))
    by_id = {str(item["file_id"]): item for item in acquisition["files"]}
    _require(
        {"endoderm_raw_counts", "endoderm_cell_metadata"}.issubset(by_id),
        "Acquisition receipt is missing an authorized CoreBench source file",
    )
    raw_entry = by_id["endoderm_raw_counts"]
    metadata_entry = by_id["endoderm_cell_metadata"]
    raw_path = receipt_path.parent / raw_entry["local_relative_path"]
    metadata_path = receipt_path.parent / metadata_entry["local_relative_path"]
    for entry, path, label in (
        (raw_entry, raw_path, "raw_archive"),
        (metadata_entry, metadata_path, "metadata"),
    ):
        _require(path.is_file(), f"Missing CoreBench {label}: {path}")
        _require_equal(path.stat().st_size, int(entry["size_bytes"]), f"{label}.bytes")
        _require_equal(
            entry.get("publisher_checksum_status"),
            "verified",
            f"{label}.publisher_checksum_status",
        )
    metadata_sha256 = _hash_file(metadata_path)
    _require_equal(
        metadata_sha256, metadata_entry["local_sha256"], "metadata.local_sha256"
    )
    raw_sha256 = str(raw_entry["local_sha256"])
    if verify_raw_archive_hash:
        raw_sha256 = _hash_file(raw_path)
        _require_equal(raw_sha256, raw_entry["local_sha256"], "raw_archive.local_sha256")

    raw_schema_path = repository_root / config["bindings"]["raw_schema_audit"][
        "relative_path"
    ]
    raw_schema = json.loads(raw_schema_path.read_text(encoding="utf-8"))
    _require_equal(
        raw_schema.get("raw_archive_local_sha256"),
        raw_entry["local_sha256"],
        "raw_schema.raw_archive_local_sha256",
    )
    _require_equal(
        raw_schema.get("metadata_local_sha256"),
        metadata_entry["local_sha256"],
        "raw_schema.metadata_local_sha256",
    )
    _require_equal(
        raw_schema.get("n_matrix_cell_columns"),
        EXPECTED_CELLS,
        "raw_schema.n_matrix_cell_columns",
    )
    _require_equal(
        raw_schema.get("cell_axis_exact_order"), True, "raw_schema.cell_axis_exact_order"
    )
    _require_equal(
        raw_schema.get("fractional_count_contract_confirmed"),
        True,
        "raw_schema.fractional_count_contract_confirmed",
    )
    return metadata_path, {
        "acquisition_receipt_sha256": _hash_file(receipt_path),
        "raw_archive_bytes": int(raw_path.stat().st_size),
        "raw_archive_sha256": raw_sha256,
        "raw_archive_sha256_recomputed": bool(verify_raw_archive_hash),
        "raw_archive_publisher_md5": raw_entry["publisher_checksum_observed"],
        "metadata_bytes": int(metadata_path.stat().st_size),
        "metadata_sha256": metadata_sha256,
        "raw_schema_audit_sha256": _hash_file(raw_schema_path),
    }


def _read_and_validate_metadata(metadata_path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(
        metadata_path,
        sep="\t",
        usecols=[
            "cell_name",
            "donor",
            "donor_long_id",
            "experiment",
            "day",
        ],
        dtype="string",
    ).rename(
        columns={
            "cell_name": "cell_id",
            "donor": "donor_id",
            "donor_long_id": "line_id",
            "experiment": "experiment_id",
        }
    )
    required = ["cell_id", "donor_id", "line_id", "experiment_id", "day"]
    _require_equal(len(metadata), EXPECTED_CELLS, "metadata.n_cells")
    _require(not bool(metadata[required].isna().any().any()), "Metadata join fields contain nulls")
    _require_equal(int(metadata["cell_id"].nunique()), EXPECTED_CELLS, "metadata.unique_cells")
    _require_equal(int(metadata["donor_id"].nunique()), EXPECTED_DONORS, "metadata.n_donors")
    _require_equal(int(metadata["line_id"].nunique()), EXPECTED_LINES, "metadata.n_lines")
    _require_equal(
        int(metadata["experiment_id"].nunique()),
        EXPECTED_EXPERIMENTS,
        "metadata.n_experiments",
    )
    _require_equal(
        sorted(metadata["day"].astype(str).unique()),
        ["day0", "day1", "day2", "day3"],
        "metadata.day_levels",
    )
    line_to_donor = metadata.groupby("line_id", observed=True)["donor_id"].nunique()
    _require_equal(int(line_to_donor.max()), 1, "metadata.maximum_donors_per_line")
    donor_line_counts = (
        metadata[["donor_id", "line_id"]]
        .drop_duplicates()
        .groupby("donor_id", observed=True)
        .size()
    )
    _require_equal(int((donor_line_counts == 2).sum()), 1, "metadata.two_line_donors")
    _require_equal(int((donor_line_counts == 1).sum()), 124, "metadata.one_line_donors")
    return metadata[required]


def _load_frozen_cohorts(
    *, repository_root: Path, config: Mapping[str, Any], metadata: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    contract_path = repository_root / config["bindings"]["donor_design_contract"][
        "relative_path"
    ]
    freeze_dir = contract_path.parent
    build_path = freeze_dir / "endoderm_benchmark_freeze_build_record_v1.json"
    cohort_path = freeze_dir / "endoderm_donor_cohort_membership_v1.tsv"
    build = json.loads(build_path.read_text(encoding="utf-8"))
    descriptor = build["artifacts"][cohort_path.name]
    _require_equal(
        descriptor["sha256"], _hash_file(cohort_path), "donor_cohort_membership.sha256"
    )
    cohorts = pd.read_csv(cohort_path, sep="\t", dtype="string")
    _require_equal(len(cohorts), EXPECTED_DONORS, "donor_cohort_membership.rows")
    cohorts["primary_complete_support"] = _read_bool(cohorts["primary_complete_support"])
    cohorts["missingness_sensitivity"] = _read_bool(cohorts["missingness_sensitivity"])
    cohorts["all_donors"] = _read_bool(cohorts["all_donors"])
    _require_equal(
        set(cohorts["donor_id"].astype(str)),
        set(metadata["donor_id"].astype(str)),
        "donor_cohort_membership.donor_ids",
    )
    _require_equal(
        int(cohorts["primary_complete_support"].sum()),
        75,
        "donor_cohort_membership.primary_complete_support",
    )
    _require_equal(
        int(cohorts["missingness_sensitivity"].sum()),
        84,
        "donor_cohort_membership.missingness_sensitivity",
    )
    _require(bool(cohorts["all_donors"].all()), "Frozen all-donor cohort is incomplete")
    return cohorts[
        ["donor_id", "primary_complete_support", "missingness_sensitivity", "all_donors"]
    ], {
        "donor_design_contract_sha256": _hash_file(contract_path),
        "donor_freeze_build_record_sha256": _hash_file(build_path),
        "donor_cohort_membership_sha256": _hash_file(cohort_path),
    }


def _read_and_validate_coordinate(
    *, coordinate_dir: Path, metadata: pd.DataFrame, receipt: Mapping[str, Any]
) -> pd.DataFrame:
    coordinate = pd.read_csv(
        coordinate_dir / COORDINATE_FILE,
        sep="\t",
        dtype={
            "cell_id": "string",
            "donor_id": "string",
            "line_id": "string",
            "experiment_id": "string",
            "day": "string",
        },
    )
    required = [
        "cell_id",
        "donor_id",
        "line_id",
        "experiment_id",
        "day",
        "within_day_crossfit_rank",
        "corebench_coordinate",
    ]
    _require_equal(coordinate.columns.tolist(), required, "coordinate.columns")
    _require_equal(len(coordinate), EXPECTED_CELLS, "coordinate.n_cells")
    for column in ["cell_id", "donor_id", "line_id", "experiment_id", "day"]:
        _require(
            bool(
                np.array_equal(
                    coordinate[column].astype(str).to_numpy(),
                    metadata[column].astype(str).to_numpy(),
                )
            ),
            f"Coordinate-to-metadata row mapping differs for {column}",
        )
    values = coordinate["corebench_coordinate"].to_numpy(dtype=float)
    ranks = coordinate["within_day_crossfit_rank"].to_numpy(dtype=float)
    _require(bool(np.isfinite(values).all()), "Coordinate contains non-finite values")
    _require(bool(np.isfinite(ranks).all()), "Within-day rank contains non-finite values")
    _require(bool(((values >= 0) & (values < 1)).all()), "Coordinate is outside [0, 1)")
    _require(bool(((ranks > 0) & (ranks < 1)).all()), "Within-day rank is outside (0, 1)")
    _require_equal(receipt.get("n_cells"), EXPECTED_CELLS, "materialization_receipt.n_cells")
    _require_equal(
        receipt.get("n_coordinate_genes"),
        EXPECTED_COORDINATE_GENES,
        "materialization_receipt.n_coordinate_genes",
    )
    _require_equal(receipt.get("n_gene_folds"), 5, "materialization_receipt.n_gene_folds")
    return coordinate


def _build_donor_bin_tables(
    coordinate: pd.DataFrame,
    cohorts: pd.DataFrame,
    *,
    n_bins: int,
    minimum_cells: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = coordinate.copy()
    values = work["corebench_coordinate"].to_numpy(dtype=float)
    work["bin_id"] = np.minimum(np.floor(values * n_bins).astype(int), n_bins - 1)
    observed = (
        work.groupby(["donor_id", "bin_id"], observed=True)
        .agg(
            cell_count=("cell_id", "size"),
            n_lines=("line_id", "nunique"),
            n_experiments=("experiment_id", "nunique"),
        )
        .reset_index()
    )
    grid = pd.MultiIndex.from_product(
        [sorted(coordinate["donor_id"].astype(str).unique()), range(n_bins)],
        names=["donor_id", "bin_id"],
    ).to_frame(index=False)
    donor_bin = grid.merge(
        observed, on=["donor_id", "bin_id"], how="left", validate="one_to_one"
    ).merge(cohorts, on="donor_id", how="left", validate="many_to_one")
    for column in ("cell_count", "n_lines", "n_experiments"):
        donor_bin[column] = donor_bin[column].fillna(0).astype(int)
    donor_bin["bin_left"] = donor_bin["bin_id"] / n_bins
    donor_bin["bin_right"] = (donor_bin["bin_id"] + 1) / n_bins
    donor_bin["available"] = donor_bin["cell_count"].ge(minimum_cells)
    donor_bin = donor_bin[
        [
            "donor_id",
            "primary_complete_support",
            "missingness_sensitivity",
            "all_donors",
            "bin_id",
            "bin_left",
            "bin_right",
            "cell_count",
            "n_lines",
            "n_experiments",
            "available",
        ]
    ].sort_values(["donor_id", "bin_id"], kind="stable")

    experiment_counts = (
        work.groupby(["bin_id", "experiment_id"], observed=True)
        .size()
        .rename("cell_count")
        .reset_index()
    )
    experiment_counts["cell_share"] = experiment_counts["cell_count"] / experiment_counts.groupby(
        "bin_id", observed=True
    )["cell_count"].transform("sum")
    dominant = experiment_counts.sort_values(
        ["bin_id", "cell_share", "experiment_id"],
        ascending=[True, False, True],
        kind="stable",
    ).groupby("bin_id", observed=True).first()
    top_three = (
        experiment_counts.sort_values(
            ["bin_id", "cell_share"], ascending=[True, False], kind="stable"
        )
        .groupby("bin_id", observed=True)
        .head(3)
        .groupby("bin_id", observed=True)["cell_share"]
        .sum()
    )
    bin_observed = (
        work.groupby("bin_id", observed=True)
        .agg(
            cell_count=("cell_id", "size"),
            n_donors_with_cells=("donor_id", "nunique"),
            n_lines=("line_id", "nunique"),
            n_experiments=("experiment_id", "nunique"),
            n_unique_coordinate_values=("corebench_coordinate", "nunique"),
        )
        .reset_index()
    )
    available_all = donor_bin.groupby("bin_id", observed=True)["available"].sum()
    primary = donor_bin[donor_bin["primary_complete_support"]]
    available_primary = primary.groupby("bin_id", observed=True)["available"].sum()
    primary_with_cells = primary.groupby("bin_id", observed=True)["cell_count"].apply(
        lambda values: int(values.gt(0).sum())
    )
    summary = bin_observed.copy()
    summary["bin_left"] = summary["bin_id"] / n_bins
    summary["bin_right"] = (summary["bin_id"] + 1) / n_bins
    summary["n_available_donors"] = summary["bin_id"].map(available_all).astype(int)
    summary["n_primary_donors_with_cells"] = summary["bin_id"].map(primary_with_cells).astype(int)
    summary["n_available_primary_donors"] = summary["bin_id"].map(available_primary).astype(int)
    summary["dominant_experiment_id"] = summary["bin_id"].map(
        dominant["experiment_id"].astype(str)
    )
    summary["dominant_experiment_cell_share"] = summary["bin_id"].map(
        dominant["cell_share"]
    )
    summary["top_three_experiment_cell_share"] = summary["bin_id"].map(top_three)
    summary = summary[
        [
            "bin_id",
            "bin_left",
            "bin_right",
            "cell_count",
            "n_unique_coordinate_values",
            "n_donors_with_cells",
            "n_available_donors",
            "n_primary_donors_with_cells",
            "n_available_primary_donors",
            "n_lines",
            "n_experiments",
            "dominant_experiment_id",
            "dominant_experiment_cell_share",
            "top_three_experiment_cell_share",
        ]
    ].sort_values("bin_id", kind="stable")
    return donor_bin.reset_index(drop=True), summary.reset_index(drop=True)


def _build_experiment_audit(coordinate: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for day, day_frame in coordinate.groupby("day", sort=True, observed=True):
        values = day_frame["within_day_crossfit_rank"].to_numpy(dtype=float)
        center = float(values.mean())
        total_ss = float(np.square(values - center).sum())
        for experiment_id, group in day_frame.groupby(
            "experiment_id", sort=True, observed=True
        ):
            group_values = group["within_day_crossfit_rank"].to_numpy(dtype=float)
            mean_rank = float(group_values.mean())
            contribution = (
                float(len(group) * (mean_rank - center) ** 2 / total_ss)
                if total_ss > 0
                else math.nan
            )
            rows.append(
                {
                    "day": str(day),
                    "experiment_id": str(experiment_id),
                    "n_cells": int(len(group)),
                    "cell_share_within_day": float(len(group) / len(day_frame)),
                    "mean_within_day_crossfit_rank": mean_rank,
                    "between_experiment_variance_contribution": contribution,
                }
            )
    return pd.DataFrame(rows).sort_values(["day", "experiment_id"], kind="stable")


def _quality_summary(
    coordinate: pd.DataFrame,
    donor_bin: pd.DataFrame,
    bin_summary: pd.DataFrame,
    experiment_audit: pd.DataFrame,
    fold_audit: pd.DataFrame,
    receipt: Mapping[str, Any],
    *,
    minimum_cells: int,
    minimum_donors_per_pseudo_condition: int,
) -> dict[str, Any]:
    tie_sizes = coordinate.groupby("corebench_coordinate", observed=True).size()
    tied = tie_sizes[tie_sizes.gt(1)]
    by_donor = donor_bin.groupby("donor_id", observed=True).agg(
        bins_with_cells=("cell_count", lambda values: int(values.gt(0).sum())),
        available_bins=("available", "sum"),
    )
    primary_ids = set(
        donor_bin.loc[donor_bin["primary_complete_support"], "donor_id"].astype(str)
    )
    primary_by_donor = by_donor.loc[by_donor.index.astype(str).isin(primary_ids)]
    day_rows = []
    for day, day_frame in experiment_audit.groupby("day", sort=True, observed=True):
        contributions = day_frame["between_experiment_variance_contribution"].sort_values(
            ascending=False
        )
        day_rows.append(
            {
                "day": str(day),
                "n_experiments": int(len(day_frame)),
                "maximum_experiment_cell_share": float(
                    day_frame["cell_share_within_day"].max()
                ),
                "experiment_factor_eta_squared": float(contributions.sum()),
                "maximum_single_experiment_variance_contribution": float(
                    contributions.iloc[0]
                ),
                "top_five_experiment_variance_contribution": float(
                    contributions.iloc[:5].sum()
                ),
            }
        )
    maximum_tie_fraction = float(tie_sizes.max() / len(coordinate))
    maximum_bin_experiment_share = float(
        bin_summary["dominant_experiment_cell_share"].max()
    )
    minimum_available = int(bin_summary["n_available_donors"].min())
    minimum_primary_available = int(
        bin_summary["n_available_primary_donors"].min()
    )
    oriented = fold_audit["oriented_donor_median_day_spearman"].to_numpy(dtype=float)
    structural_checks = {
        "all_20_bins_nonempty": bool(len(bin_summary) == 20 and (bin_summary["cell_count"] > 0).all()),
        "every_bin_has_multiple_coordinate_values": bool(
            (bin_summary["n_unique_coordinate_values"] > 1).all()
        ),
        "no_coordinate_value_contains_0_1_percent_of_cells": bool(
            maximum_tie_fraction < 0.001
        ),
        "all_fold_orientations_positive": bool(np.isfinite(oriented).all() and (oriented > 0).all()),
        "all_cross_fold_correlations_positive": bool(
            float(receipt["minimum_cross_fold_spearman"]) > 0
        ),
        "no_single_experiment_has_a_bin_majority": bool(
            maximum_bin_experiment_share < 0.5
        ),
        "minimum_total_available_donors_is_twice_group_minimum": bool(
            minimum_available >= 2 * minimum_donors_per_pseudo_condition
        ),
        "every_primary_donor_has_an_available_bin": bool(
            int(primary_by_donor["available_bins"].min()) > 0
        ),
    }
    return {
        "guardrail_role": "descriptive_cb1_structural_checks_not_statistical_acceptance_thresholds",
        "tie_and_degeneracy": {
            "n_cells": int(len(coordinate)),
            "n_unique_coordinate_values": int(tie_sizes.size),
            "unique_coordinate_fraction": float(tie_sizes.size / len(coordinate)),
            "n_tied_coordinate_values": int(len(tied)),
            "n_cells_in_tie_groups": int(tied.sum()),
            "fraction_cells_in_tie_groups": float(tied.sum() / len(coordinate)),
            "maximum_tie_multiplicity": int(tie_sizes.max()),
            "maximum_tie_cell_fraction": maximum_tie_fraction,
            "minimum_bin_cell_count": int(bin_summary["cell_count"].min()),
            "maximum_bin_cell_count": int(bin_summary["cell_count"].max()),
            "minimum_unique_coordinate_values_per_bin": int(
                bin_summary["n_unique_coordinate_values"].min()
            ),
        },
        "donor_grid_coverage": {
            "n_grid_bins": 20,
            "minimum_cells_for_availability": int(minimum_cells),
            "minimum_donors_per_pseudo_condition_from_bound_design": int(
                minimum_donors_per_pseudo_condition
            ),
            "minimum_donors_with_any_cells_per_bin": int(
                bin_summary["n_donors_with_cells"].min()
            ),
            "maximum_donors_with_any_cells_per_bin": int(
                bin_summary["n_donors_with_cells"].max()
            ),
            "minimum_available_donors_per_bin": minimum_available,
            "maximum_available_donors_per_bin": int(
                bin_summary["n_available_donors"].max()
            ),
            "minimum_available_primary_donors_per_bin": minimum_primary_available,
            "maximum_available_primary_donors_per_bin": int(
                bin_summary["n_available_primary_donors"].max()
            ),
            "minimum_bins_with_cells_per_donor": int(by_donor["bins_with_cells"].min()),
            "median_bins_with_cells_per_donor": float(by_donor["bins_with_cells"].median()),
            "minimum_available_bins_per_donor": int(by_donor["available_bins"].min()),
            "median_available_bins_per_donor": float(by_donor["available_bins"].median()),
            "all_donors_with_zero_available_bins": int(by_donor["available_bins"].eq(0).sum()),
            "minimum_available_bins_per_primary_donor": int(
                primary_by_donor["available_bins"].min()
            ),
            "median_available_bins_per_primary_donor": float(
                primary_by_donor["available_bins"].median()
            ),
            "assignment_specific_common_support_deferred_to_cb2": True,
        },
        "fold_stability": {
            "n_folds": int(len(fold_audit)),
            "minimum_oriented_donor_median_day_spearman": float(oriented.min()),
            "minimum_cross_fold_spearman": float(receipt["minimum_cross_fold_spearman"]),
            "median_cross_fold_spearman": float(receipt["median_cross_fold_spearman"]),
            "condition_or_pseudo_condition_used_for_orientation": False,
        },
        "experiment_structure": {
            "n_experiments": int(coordinate["experiment_id"].nunique()),
            "maximum_global_experiment_cell_share": float(
                coordinate["experiment_id"].value_counts(normalize=True).max()
            ),
            "maximum_single_experiment_cell_share_in_any_bin": maximum_bin_experiment_share,
            "maximum_top_three_experiment_cell_share_in_any_bin": float(
                bin_summary["top_three_experiment_cell_share"].max()
            ),
            "by_day": day_rows,
            "experiment_imbalance_robustness_remains_required_in_cb2": True,
        },
        "structural_checks": structural_checks,
        "all_structural_checks_pass": bool(all(structural_checks.values())),
    }


def validate_and_write_corebench_coordinate(
    *,
    config_path: str | Path,
    repository_root: str | Path,
    freeze_dir: str | Path,
    coordinate_dir: str | Path,
    output_dir: str | Path,
    verify_raw_archive_hash: bool = True,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config_file = Path(config_path).resolve()
    config = load_corebench_config(config_file)
    freeze = Path(freeze_dir).resolve()
    coordinate_output = Path(coordinate_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"CoreBench coordinate validation output exists: {output}")

    freeze_record = validate_corebench_freeze_output(
        config_path=config_file,
        repository_root=root,
        output_dir=freeze,
    )
    materialization_receipt, materialization_record = _validate_materialization_artifacts(
        repository_root=root,
        coordinate_dir=coordinate_output,
        freeze_record=freeze_record,
    )
    metadata_path, source_audit = _validate_source_files(
        repository_root=root,
        config=config,
        verify_raw_archive_hash=verify_raw_archive_hash,
    )
    metadata = _read_and_validate_metadata(metadata_path)
    cohorts, cohort_audit = _load_frozen_cohorts(
        repository_root=root, config=config, metadata=metadata
    )
    coordinate = _read_and_validate_coordinate(
        coordinate_dir=coordinate_output,
        metadata=metadata,
        receipt=materialization_receipt,
    )

    manifest = pd.read_csv(freeze / GENE_FOLD_FILE, sep="\t", dtype="string")
    coordinate_genes = _read_bool(manifest["coordinate_gene"])
    _require_equal(int(coordinate_genes.sum()), EXPECTED_COORDINATE_GENES, "coordinate_genes")
    _require_equal(
        int(manifest.loc[coordinate_genes, "gene_fold"].nunique()), 5, "coordinate_gene_folds"
    )
    exclusion = json.loads((freeze / EXCLUSION_AUDIT_FILE).read_text(encoding="utf-8"))
    _require_equal(
        exclusion.get("coordinate_injection_gene_overlap"),
        0,
        "coordinate_injection_gene_overlap",
    )
    fold_audit = pd.read_csv(coordinate_output / FOLD_AUDIT_FILE, sep="\t")
    _require_equal(len(fold_audit), 5, "fold_audit.rows")
    _require_equal(sorted(fold_audit["fold"].astype(int).tolist()), [1, 2, 3, 4, 5], "fold_audit.folds")

    donor_contract_path = root / config["bindings"]["donor_design_contract"][
        "relative_path"
    ]
    donor_contract = json.loads(donor_contract_path.read_text(encoding="utf-8"))
    n_bins = int(config["statistical_units"]["fixed_common_grid_bins"])
    _require_equal(n_bins, 20, "fixed_common_grid_bins")
    minimum_cells = int(donor_contract["trajectory_preflight"]["minimum_cells_per_donor_bin"])
    minimum_donors = int(
        donor_contract["trajectory_preflight"]["minimum_donors_per_pseudo_condition"]
    )
    donor_bin, bin_summary = _build_donor_bin_tables(
        coordinate,
        cohorts,
        n_bins=n_bins,
        minimum_cells=minimum_cells,
    )
    experiment_audit = _build_experiment_audit(coordinate)
    quality = _quality_summary(
        coordinate,
        donor_bin,
        bin_summary,
        experiment_audit,
        fold_audit,
        materialization_receipt,
        minimum_cells=minimum_cells,
        minimum_donors_per_pseudo_condition=minimum_donors,
    )
    _require(
        quality["all_structural_checks_pass"],
        "CoreBench coordinate failed a CB1 structural validation check",
    )

    cell_order = coordinate[["cell_id"]].copy()
    cell_order.insert(0, "row_index", np.arange(len(cell_order), dtype=int))
    validation_receipt = {
        "schema_name": "trajpathmix_corebench_coordinate_validation_receipt",
        "schema_version": "1.0.0",
        "freeze_id": "trajpathmix_corebench_v1",
        "config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
        "source_materialization_build_record_sha256": _hash_file(
            coordinate_output / MATERIALIZATION_BUILD_RECORD_FILE
        ),
        "source_coordinate_file_sha256": _hash_file(coordinate_output / COORDINATE_FILE),
        "source_fold_audit_sha256": _hash_file(coordinate_output / FOLD_AUDIT_FILE),
        "coordinate_gene_fold_manifest_sha256": _hash_file(freeze / GENE_FOLD_FILE),
        "coordinate_gene_exclusion_audit_sha256": _hash_file(
            freeze / EXCLUSION_AUDIT_FILE
        ),
        "freeze_build_record_sha256": _hash_file(freeze / FREEZE_BUILD_RECORD_FILE),
        "source_files": source_audit,
        "frozen_donor_design": cohort_audit,
        "metadata_join": {
            "n_cells": int(len(metadata)),
            "n_unique_cells": int(metadata["cell_id"].nunique()),
            "n_independent_donors": int(metadata["donor_id"].nunique()),
            "n_nested_lines": int(metadata["line_id"].nunique()),
            "n_experiments": int(metadata["experiment_id"].nunique()),
            "n_days": int(metadata["day"].nunique()),
            "line_to_donor_collision_count": 0,
            "coordinate_metadata_row_mismatch_count": 0,
            "cell_order_exact": True,
        },
        "coordinate_genes": {
            "n_coordinate_genes": int(coordinate_genes.sum()),
            "all_located_by_materializer": True,
            "coordinate_injection_gene_overlap": 0,
            "n_deterministically_rebuilt_folds": 5,
            "freeze_validation_status": freeze_record["validation_status"],
        },
        "experiment_adjustment_interpretation": {
            "frozen_a1_hvg_selection": "experiment_adjusted_residual_variance_rank",
            "pca_input": "normalized_expression_on_frozen_hvg_selected_genes_without_expression_residualization",
            "status": "matches_executable_frozen_a1_contract",
            "clarification": "experiment adjustment applies to the frozen HVG selection; frozen A1 did not define a residualized-expression PCA transform",
        },
        "quality": quality,
        "real_pathway_outcomes_read": False,
        "pathway_scoring_performed": False,
        "pseudo_conditions_generated": False,
        "injection_results_generated": False,
        "cb1_validation_status": "pass_coordinate_materialization_and_structural_quality",
        "next_gate": "stop_after_cb1_pending_separate_authorization_and_contract_clarification_for_cb2",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / f".{output.name}.create.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FileExistsError(f"CoreBench coordinate validation is locked: {lock}") from exc
    temporary: Path | None = None
    try:
        os.close(descriptor)
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
        cell_order.to_csv(
            temporary / CELL_ORDER_FILE, sep="\t", index=False, lineterminator="\n"
        )
        donor_bin.to_csv(
            temporary / DONOR_BIN_FILE, sep="\t", index=False, lineterminator="\n"
        )
        bin_summary.to_csv(
            temporary / BIN_SUMMARY_FILE, sep="\t", index=False, lineterminator="\n"
        )
        experiment_audit.to_csv(
            temporary / EXPERIMENT_AUDIT_FILE,
            sep="\t",
            index=False,
            lineterminator="\n",
        )
        _write_json(validation_receipt, temporary / VALIDATION_RECEIPT_FILE)
        artifacts = {
            path.name: {"sha256": _hash_file(path), "bytes": path.stat().st_size}
            for path in sorted(temporary.iterdir())
        }
        build_record = {
            "schema_name": "trajpathmix_corebench_coordinate_validation_build_record",
            "schema_version": "1.0.0",
            "config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
            "implementation_file": "pyfgsea/trajpathmix_corebench_coordinate_validation.py",
            "implementation_sha256": _hash_file(Path(__file__).resolve()),
            "artifacts": artifacts,
            "source_materialization_implementation_sha256": materialization_record[
                "implementation_sha256"
            ],
            "raw_archive_hash_recomputed": bool(verify_raw_archive_hash),
            "expression_values_read_by_validation": False,
            "pathway_outcomes_read": False,
            "pathway_scoring_performed": False,
            "pseudo_conditions_generated": False,
            "injection_results_generated": False,
            "evidence_revision_mode": "create_only_append_only",
        }
        _write_json(build_record, temporary / VALIDATION_BUILD_RECORD_FILE)
        os.rename(temporary, output)
        temporary = None
        result = dict(build_record)
        result["output_dir"] = str(output)
        result["build_record_sha256"] = _hash_file(
            output / VALIDATION_BUILD_RECORD_FILE
        )
        result["validation_receipt_sha256"] = _hash_file(
            output / VALIDATION_RECEIPT_FILE
        )
        result["cb1_validation_status"] = validation_receipt["cb1_validation_status"]
        return result
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
        lock.unlink(missing_ok=True)


__all__ = [
    "BIN_SUMMARY_FILE",
    "CELL_ORDER_FILE",
    "DONOR_BIN_FILE",
    "EXPERIMENT_AUDIT_FILE",
    "VALIDATION_BUILD_RECORD_FILE",
    "VALIDATION_RECEIPT_FILE",
    "validate_and_write_corebench_coordinate",
]
