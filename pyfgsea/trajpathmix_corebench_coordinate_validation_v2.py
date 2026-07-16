from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import trajpathmix_corebench_coordinate_validation as validation_v1
from .trajpathmix_corebench_coordinate import (
    BUILD_RECORD_FILE as MATERIALIZATION_BUILD_RECORD_FILE,
    COORDINATE_FILE,
    FOLD_AUDIT_FILE,
)
from .trajpathmix_corebench_freeze import (
    BUILD_RECORD_FILE as COREBENCH_FREEZE_BUILD_RECORD_FILE,
    EXCLUSION_AUDIT_FILE,
    FROZEN_CONFIG_PAYLOAD_SHA256,
    GENE_FOLD_FILE,
    load_corebench_config,
    validate_corebench_freeze_output,
)
from .trajpathmix_endoderm_benchmark_freeze import (
    BENCHMARK_CONTRACT_FILE as DONOR_CONTRACT_FILE,
    BUILD_RECORD_FILE as DONOR_FREEZE_BUILD_RECORD_FILE,
    DONOR_COHORT_FILE,
    validate_endoderm_benchmark_freeze_output,
)


CELL_ORDER_FILE = "corebench_cell_order_v2.tsv"
DONOR_BIN_FILE = "corebench_donor_bin_availability_v2.tsv"
BIN_SUMMARY_FILE = "corebench_coordinate_bin_summary_v2.tsv"
EXPERIMENT_AUDIT_FILE = "corebench_coordinate_experiment_audit_v2.tsv"
VALIDATION_RECEIPT_FILE = "corebench_coordinate_validation_receipt_v2.json"
VALIDATION_BUILD_RECORD_FILE = "corebench_coordinate_validation_build_record_v2.json"

VALIDATION_IMPLEMENTATION_FILE = (
    "pyfgsea/trajpathmix_corebench_coordinate_validation_v2.py"
)
SUPPORTING_IMPLEMENTATION_FILE = (
    "pyfgsea/trajpathmix_corebench_coordinate_validation.py"
)
DONOR_FREEZE_CONFIG_FILE = "config/trajpathmix_endoderm_benchmark_freeze_v1.yaml"
SUPERSEDED_V1_BUILD_SHA256 = (
    "73129f72d283f5b299e59c93669ef1a211362f49438bd4f33f7935da028c8a08"
)


def _load_and_validate_frozen_cohorts(
    *,
    repository_root: Path,
    config: Mapping[str, Any],
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    contract_path = (
        repository_root
        / config["bindings"]["donor_design_contract"]["relative_path"]
    ).resolve()
    donor_freeze_dir = contract_path.parent
    validation_v1._require_equal(
        contract_path.name, DONOR_CONTRACT_FILE, "donor_design_contract.file_name"
    )
    donor_freeze = validate_endoderm_benchmark_freeze_output(
        config_path=repository_root / DONOR_FREEZE_CONFIG_FILE,
        repository_root=repository_root,
        output_dir=donor_freeze_dir,
    )
    validation_v1._require_equal(
        donor_freeze.get("validation_status"),
        "pass_donor_level_benchmark_freeze",
        "donor_freeze.validation_status",
    )
    cohort_path = donor_freeze_dir / DONOR_COHORT_FILE
    cohorts = pd.read_csv(cohort_path, sep="\t", dtype="string")
    validation_v1._require_equal(
        len(cohorts), validation_v1.EXPECTED_DONORS, "donor_cohort_membership.rows"
    )
    for column in (
        "primary_complete_support",
        "missingness_sensitivity",
        "all_donors",
    ):
        cohorts[column] = validation_v1._read_bool(cohorts[column])
    validation_v1._require_equal(
        set(cohorts["donor_id"].astype(str)),
        set(metadata["donor_id"].astype(str)),
        "donor_cohort_membership.donor_ids",
    )
    validation_v1._require_equal(
        int(cohorts["primary_complete_support"].sum()),
        75,
        "donor_cohort_membership.primary_complete_support",
    )
    validation_v1._require_equal(
        int(cohorts["missingness_sensitivity"].sum()),
        84,
        "donor_cohort_membership.missingness_sensitivity",
    )
    validation_v1._require(
        bool(cohorts["all_donors"].all()), "Frozen all-donor cohort is incomplete"
    )
    return cohorts[
        ["donor_id", "primary_complete_support", "missingness_sensitivity", "all_donors"]
    ], {
        "donor_design_contract_sha256": validation_v1._hash_file(contract_path),
        "donor_freeze_build_record_sha256": validation_v1._hash_file(
            donor_freeze_dir / DONOR_FREEZE_BUILD_RECORD_FILE
        ),
        "donor_cohort_membership_sha256": validation_v1._hash_file(cohort_path),
        "donor_freeze_validation_status": donor_freeze["validation_status"],
    }


def _read_and_validate_coordinate_formula(
    *,
    coordinate_dir: Path,
    metadata: pd.DataFrame,
    receipt: Mapping[str, Any],
    day_order: list[str],
) -> tuple[pd.DataFrame, float]:
    coordinate = validation_v1._read_and_validate_coordinate(
        coordinate_dir=coordinate_dir,
        metadata=metadata,
        receipt=receipt,
    )
    day_index = coordinate["day"].map(
        {day: index for index, day in enumerate(day_order)}
    )
    validation_v1._require(
        not bool(day_index.isna().any()), "Coordinate contains an unfrozen day label"
    )
    expected = (
        day_index.to_numpy(dtype=float)
        + coordinate["within_day_crossfit_rank"].to_numpy(dtype=float)
    ) / len(day_order)
    observed = coordinate["corebench_coordinate"].to_numpy(dtype=float)
    error = np.abs(observed - expected)
    maximum_error = float(error.max(initial=0.0))
    validation_v1._require(
        bool(np.allclose(observed, expected, rtol=0.0, atol=5e-15)),
        "Coordinate differs from the frozen (day_index + within_day_rank) / 4 formula",
    )
    return coordinate, maximum_error


def _coordinate_quality_review(
    coordinate: pd.DataFrame,
    donor_bin: pd.DataFrame,
    bin_summary: pd.DataFrame,
    experiment_audit: pd.DataFrame,
    fold_audit: pd.DataFrame,
    materialization_receipt: Mapping[str, Any],
    *,
    coordinate_formula_maximum_absolute_error: float,
    minimum_cells: int,
    minimum_donors_per_pseudo_condition: int,
) -> dict[str, Any]:
    legacy = validation_v1._quality_summary(
        coordinate,
        donor_bin,
        bin_summary,
        experiment_audit,
        fold_audit,
        materialization_receipt,
        minimum_cells=minimum_cells,
        minimum_donors_per_pseudo_condition=minimum_donors_per_pseudo_condition,
    )
    oriented = fold_audit["oriented_donor_median_day_spearman"].to_numpy(dtype=float)
    integrity_checks = {
        "frozen_coordinate_formula_reproduced": bool(
            coordinate_formula_maximum_absolute_error <= 5e-15
        ),
        "all_20_bins_nonempty": bool(
            len(bin_summary) == 20 and (bin_summary["cell_count"] > 0).all()
        ),
        "every_bin_has_multiple_coordinate_values": bool(
            (bin_summary["n_unique_coordinate_values"] > 1).all()
        ),
        "all_fold_orientations_positive": bool(
            np.isfinite(oriented).all() and (oriented > 0).all()
        ),
        "all_cross_fold_correlations_positive": bool(
            float(materialization_receipt["minimum_cross_fold_spearman"]) > 0
        ),
    }
    return {
        "policy": {
            "integrity_checks_are_machine_gates": True,
            "tie_donor_and_experiment_metrics_are_descriptive_review_not_preregistered_statistical_thresholds": True,
            "no_cb2_calibration_claim": True,
        },
        "coordinate_formula_maximum_absolute_error": coordinate_formula_maximum_absolute_error,
        "integrity_checks": integrity_checks,
        "all_integrity_checks_pass": bool(all(integrity_checks.values())),
        "tie_and_degeneracy": legacy["tie_and_degeneracy"],
        "donor_grid_coverage": legacy["donor_grid_coverage"],
        "fold_stability": legacy["fold_stability"],
        "experiment_structure": legacy["experiment_structure"],
        "review_findings": {
            "ties": "many_cells_participate_in_small_tie_blocks_maximum_multiplicity_five_without_degenerate_bins",
            "donor_coverage": "primary_75_donors_have_10_to_20_available_bins_all_125_donor_missingness_preserved",
            "experiment_structure": "no_single_experiment_cell_count_majority_but_nontrivial_within_day_association_requires_future_cb2_stress_test",
            "fold_structure": "positive_orientation_and_positive_cross_fold_concordance_no_catastrophic_reversal",
        },
        "coordinate_quality_review_status": "documented_small_ties_primary_coverage_and_nontrivial_experiment_association",
    }


def _validate_validation_directory(
    *, repository_root: Path, output_dir: Path
) -> dict[str, Any]:
    expected_artifacts = {
        CELL_ORDER_FILE,
        DONOR_BIN_FILE,
        BIN_SUMMARY_FILE,
        EXPERIMENT_AUDIT_FILE,
        VALIDATION_RECEIPT_FILE,
    }
    validation_v1._require_equal(
        {path.name for path in output_dir.iterdir() if path.is_file()},
        {*expected_artifacts, VALIDATION_BUILD_RECORD_FILE},
        "validation_v2.output_file_set",
    )
    build = json.loads(
        (output_dir / VALIDATION_BUILD_RECORD_FILE).read_text(encoding="utf-8")
    )
    validation_v1._require_equal(
        build.get("schema_name"),
        "trajpathmix_corebench_coordinate_validation_build_record",
        "validation_v2.build_record.schema_name",
    )
    validation_v1._require_equal(
        build.get("schema_version"), "1.1.0", "validation_v2.build_record.schema_version"
    )
    validation_v1._require_equal(
        build.get("validation_revision"), 2, "validation_v2.build_record.validation_revision"
    )
    validation_v1._require_equal(
        build.get("config_payload_sha256"),
        FROZEN_CONFIG_PAYLOAD_SHA256,
        "validation_v2.build_record.config_payload_sha256",
    )
    validation_v1._require_equal(
        set(build.get("artifacts", {})), expected_artifacts, "validation_v2.artifacts"
    )
    for name, descriptor in build["artifacts"].items():
        validation_v1._verify_artifact(output_dir / name, descriptor, name)
    validation_v1._require_equal(
        validation_v1._hash_file(repository_root / VALIDATION_IMPLEMENTATION_FILE),
        build["implementation_sha256"],
        "validation_v2.implementation_sha256",
    )
    validation_v1._require_equal(
        validation_v1._hash_file(repository_root / SUPPORTING_IMPLEMENTATION_FILE),
        build["supporting_implementation_sha256"],
        "validation_v2.supporting_implementation_sha256",
    )
    validation_v1._require_equal(
        build.get("raw_archive_hash_recomputed"),
        True,
        "validation_v2.raw_archive_hash_recomputed",
    )
    for key in (
        "pathway_outcomes_read",
        "pathway_scoring_performed",
        "pseudo_conditions_generated",
        "injection_results_generated",
    ):
        validation_v1._require_equal(build.get(key), False, f"validation_v2.build_record.{key}")
    receipt = json.loads(
        (output_dir / VALIDATION_RECEIPT_FILE).read_text(encoding="utf-8")
    )
    validation_v1._require_equal(
        receipt.get("schema_version"), "1.1.0", "validation_v2.receipt.schema_version"
    )
    validation_v1._require_equal(
        receipt.get("validation_revision"), 2, "validation_v2.receipt.validation_revision"
    )
    validation_v1._require_equal(
        receipt.get("config_payload_sha256"),
        FROZEN_CONFIG_PAYLOAD_SHA256,
        "validation_v2.receipt.config_payload_sha256",
    )
    validation_v1._require_equal(
        receipt["supersedes_nonfinal_validation_attempt"].get("build_record_sha256"),
        SUPERSEDED_V1_BUILD_SHA256,
        "validation_v2.receipt.superseded_v1_build_sha256",
    )
    validation_v1._require_equal(
        receipt["source_files"].get("raw_archive_sha256_recomputed"),
        True,
        "validation_v2.receipt.raw_archive_sha256_recomputed",
    )
    validation_v1._require_equal(
        receipt["quality"].get("all_integrity_checks_pass"),
        True,
        "validation_v2.receipt.all_integrity_checks_pass",
    )
    validation_v1._require_equal(
        receipt.get("cb1_completion_status"),
        "complete_integrity_validation_with_documented_cb2_risks",
        "validation_v2.receipt.cb1_completion_status",
    )
    for key in (
        "real_pathway_outcomes_read",
        "pathway_scoring_performed",
        "pseudo_conditions_generated",
        "injection_results_generated",
    ):
        validation_v1._require_equal(receipt.get(key), False, f"validation_v2.receipt.{key}")
    result = dict(build)
    result["output_dir"] = str(output_dir)
    result["build_record_sha256"] = validation_v1._hash_file(
        output_dir / VALIDATION_BUILD_RECORD_FILE
    )
    result["validation_receipt_sha256"] = validation_v1._hash_file(
        output_dir / VALIDATION_RECEIPT_FILE
    )
    result["output_validation_status"] = "pass_validation_v2_artifact_integrity"
    return result


def validate_corebench_coordinate_validation_output(
    *, repository_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_dir).resolve()
    if not output.is_dir():
        raise FileNotFoundError(output)
    return _validate_validation_directory(repository_root=root, output_dir=output)


def validate_and_write_corebench_coordinate_v2(
    *,
    config_path: str | Path,
    repository_root: str | Path,
    freeze_dir: str | Path,
    coordinate_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config_file = Path(config_path).resolve()
    config = load_corebench_config(config_file)
    freeze = Path(freeze_dir).resolve()
    coordinate_output = Path(coordinate_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"CoreBench coordinate validation v2 output exists: {output}")

    freeze_record = validate_corebench_freeze_output(
        config_path=config_file,
        repository_root=root,
        output_dir=freeze,
    )
    materialization_receipt, materialization_record = (
        validation_v1._validate_materialization_artifacts(
            repository_root=root,
            coordinate_dir=coordinate_output,
            freeze_record=freeze_record,
        )
    )
    metadata_path, source_audit = validation_v1._validate_source_files(
        repository_root=root,
        config=config,
        verify_raw_archive_hash=True,
    )
    validation_v1._require_equal(
        source_audit["raw_archive_sha256_recomputed"],
        True,
        "validation_v2.raw_archive_sha256_recomputed",
    )
    metadata = validation_v1._read_and_validate_metadata(metadata_path)
    cohorts, cohort_audit = _load_and_validate_frozen_cohorts(
        repository_root=root,
        config=config,
        metadata=metadata,
    )
    day_order = list(config["analysis_coordinate"]["coarse_anchor_order"])
    coordinate, formula_error = _read_and_validate_coordinate_formula(
        coordinate_dir=coordinate_output,
        metadata=metadata,
        receipt=materialization_receipt,
        day_order=day_order,
    )

    manifest = pd.read_csv(freeze / GENE_FOLD_FILE, sep="\t", dtype="string")
    coordinate_genes = validation_v1._read_bool(manifest["coordinate_gene"])
    validation_v1._require_equal(
        int(coordinate_genes.sum()),
        validation_v1.EXPECTED_COORDINATE_GENES,
        "coordinate_genes",
    )
    validation_v1._require_equal(
        int(manifest.loc[coordinate_genes, "gene_fold"].nunique()),
        5,
        "coordinate_gene_folds",
    )
    exclusion = json.loads((freeze / EXCLUSION_AUDIT_FILE).read_text(encoding="utf-8"))
    validation_v1._require_equal(
        exclusion.get("coordinate_injection_gene_overlap"),
        0,
        "coordinate_injection_gene_overlap",
    )
    fold_audit = pd.read_csv(coordinate_output / FOLD_AUDIT_FILE, sep="\t")
    validation_v1._require_equal(len(fold_audit), 5, "fold_audit.rows")

    donor_contract_path = root / config["bindings"]["donor_design_contract"][
        "relative_path"
    ]
    donor_contract = json.loads(donor_contract_path.read_text(encoding="utf-8"))
    n_bins = int(config["statistical_units"]["fixed_common_grid_bins"])
    validation_v1._require_equal(n_bins, 20, "fixed_common_grid_bins")
    minimum_cells = int(
        donor_contract["trajectory_preflight"]["minimum_cells_per_donor_bin"]
    )
    minimum_donors = int(
        donor_contract["trajectory_preflight"]["minimum_donors_per_pseudo_condition"]
    )
    donor_bin, bin_summary = validation_v1._build_donor_bin_tables(
        coordinate,
        cohorts,
        n_bins=n_bins,
        minimum_cells=minimum_cells,
    )
    experiment_audit = validation_v1._build_experiment_audit(coordinate)
    quality = _coordinate_quality_review(
        coordinate,
        donor_bin,
        bin_summary,
        experiment_audit,
        fold_audit,
        materialization_receipt,
        coordinate_formula_maximum_absolute_error=formula_error,
        minimum_cells=minimum_cells,
        minimum_donors_per_pseudo_condition=minimum_donors,
    )
    validation_v1._require(
        quality["all_integrity_checks_pass"],
        "CoreBench coordinate failed a frozen CB1 integrity invariant",
    )

    cell_order = coordinate[["cell_id"]].copy()
    cell_order.insert(0, "row_index", np.arange(len(cell_order), dtype=int))
    validation_receipt = {
        "schema_name": "trajpathmix_corebench_coordinate_validation_receipt",
        "schema_version": "1.1.0",
        "validation_revision": 2,
        "freeze_id": "trajpathmix_corebench_v1",
        "config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
        "supersedes_nonfinal_validation_attempt": {
            "relative_path": "data_external/trajpathmix_corebench_coordinate_validation_v1",
            "build_record_sha256": SUPERSEDED_V1_BUILD_SHA256,
            "reasons": [
                "v1_allowed_skip_raw_hash_to_receive_a_pass_status",
                "v1_did_not_rebuild_validate_the_bound_donor_freeze",
                "v1_mixed_descriptive_coordinate_metrics_with_machine_integrity_gates",
            ],
            "v1_artifacts_preserved": True,
        },
        "source_materialization_build_record_sha256": validation_v1._hash_file(
            coordinate_output / MATERIALIZATION_BUILD_RECORD_FILE
        ),
        "source_coordinate_file_sha256": validation_v1._hash_file(
            coordinate_output / COORDINATE_FILE
        ),
        "source_fold_audit_sha256": validation_v1._hash_file(
            coordinate_output / FOLD_AUDIT_FILE
        ),
        "coordinate_gene_fold_manifest_sha256": validation_v1._hash_file(
            freeze / GENE_FOLD_FILE
        ),
        "coordinate_gene_exclusion_audit_sha256": validation_v1._hash_file(
            freeze / EXCLUSION_AUDIT_FILE
        ),
        "freeze_build_record_sha256": validation_v1._hash_file(
            freeze / COREBENCH_FREEZE_BUILD_RECORD_FILE
        ),
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
        },
        "quality": quality,
        "real_pathway_outcomes_read": False,
        "pathway_scoring_performed": False,
        "pseudo_conditions_generated": False,
        "injection_results_generated": False,
        "cb1_completion_status": "complete_integrity_validation_with_documented_cb2_risks",
        "next_gate": "stop_after_cb1_pending_separate_authorization_and_contract_clarification_for_cb2",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / f".{output.name}.create.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FileExistsError(f"CoreBench coordinate validation v2 is locked: {lock}") from exc
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
        validation_v1._write_json(
            validation_receipt, temporary / VALIDATION_RECEIPT_FILE
        )
        artifacts = {
            path.name: {
                "sha256": validation_v1._hash_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(temporary.iterdir())
        }
        build_record = {
            "schema_name": "trajpathmix_corebench_coordinate_validation_build_record",
            "schema_version": "1.1.0",
            "validation_revision": 2,
            "config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
            "implementation_file": VALIDATION_IMPLEMENTATION_FILE,
            "implementation_sha256": validation_v1._hash_file(Path(__file__).resolve()),
            "supporting_implementation_file": SUPPORTING_IMPLEMENTATION_FILE,
            "supporting_implementation_sha256": validation_v1._hash_file(
                root / SUPPORTING_IMPLEMENTATION_FILE
            ),
            "artifacts": artifacts,
            "source_materialization_implementation_sha256": materialization_record[
                "implementation_sha256"
            ],
            "raw_archive_hash_recomputed": True,
            "donor_freeze_rebuilt_and_validated": True,
            "coordinate_formula_validated": True,
            "expression_values_read_by_validation": False,
            "pathway_outcomes_read": False,
            "pathway_scoring_performed": False,
            "pseudo_conditions_generated": False,
            "injection_results_generated": False,
            "evidence_revision_mode": "create_only_append_only",
        }
        validation_v1._write_json(
            build_record, temporary / VALIDATION_BUILD_RECORD_FILE
        )
        _validate_validation_directory(repository_root=root, output_dir=temporary)
        os.rename(temporary, output)
        temporary = None
        result = validate_corebench_coordinate_validation_output(
            repository_root=root, output_dir=output
        )
        result["cb1_completion_status"] = validation_receipt["cb1_completion_status"]
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
    "validate_and_write_corebench_coordinate_v2",
    "validate_corebench_coordinate_validation_output",
]
