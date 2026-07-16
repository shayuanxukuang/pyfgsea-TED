from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from .trajpathmix_corebench_freeze import (
    FROZEN_CONFIG_PAYLOAD_SHA256,
    GENE_FOLD_FILE,
    load_corebench_config,
    validate_corebench_freeze_output,
)


COORDINATE_FILE = "corebench_fixed_analysis_coordinate_v1.tsv.gz"
FOLD_AUDIT_FILE = "corebench_coordinate_fold_audit_v1.tsv"
RECEIPT_FILE = "corebench_coordinate_materialization_receipt_v1.json"
BUILD_RECORD_FILE = "corebench_coordinate_materialization_build_record_v1.json"


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


def _stable_within_day_midrank(
    values: np.ndarray,
    days: np.ndarray,
    cell_ids: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    days = np.asarray(days, dtype=str)
    cell_ids = np.asarray(cell_ids, dtype=str)
    if values.ndim != 1 or days.shape != values.shape or cell_ids.shape != values.shape:
        raise ValueError("Within-day rank inputs must be aligned one-dimensional arrays")
    if not np.isfinite(values).all():
        raise ValueError("Within-day rank values must be finite")
    result = np.empty(len(values), dtype=np.float64)
    tie_hash = np.array(
        [hashlib.sha256(cell_id.encode("utf-8")).hexdigest() for cell_id in cell_ids]
    )
    for day in sorted(set(days)):
        indices = np.flatnonzero(days == day)
        order = np.lexsort((tie_hash[indices], values[indices]))
        ranks = np.empty(len(indices), dtype=np.float64)
        ranks[order] = (np.arange(len(indices), dtype=np.float64) + 0.5) / len(indices)
        result[indices] = ranks
    return result


def _orient_by_donor_median_day(
    values: np.ndarray,
    donor_ids: np.ndarray,
    days: np.ndarray,
    day_order: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    frame = pd.DataFrame(
        {
            "value": np.asarray(values, dtype=np.float64),
            "donor_id": np.asarray(donor_ids, dtype=str),
            "day": np.asarray(days, dtype=str),
        }
    )
    donor_day = (
        frame.groupby(["donor_id", "day"], observed=True)["value"]
        .median()
        .reset_index()
    )
    medians = donor_day.groupby("day", observed=True)["value"].median()
    if any(day not in medians for day in day_order):
        raise ValueError("Every coarse day is required for coordinate orientation")
    day_medians = np.array([float(medians[day]) for day in day_order])
    if float(np.ptp(day_medians)) <= 1e-12:
        raise ValueError("Coordinate fold has no orientable donor-median day direction")
    rho = float(spearmanr(np.arange(len(day_order)), day_medians).statistic)
    if not math.isfinite(rho) or abs(rho) <= 1e-12:
        raise ValueError("Coordinate fold has no orientable donor-median day direction")
    flipped = rho < 0
    oriented = -np.asarray(values, dtype=np.float64) if flipped else np.asarray(values, dtype=np.float64)
    oriented_medians = -day_medians if flipped else day_medians
    oriented_rho = float(
        spearmanr(np.arange(len(day_order)), oriented_medians).statistic
    )
    return oriented, {
        "raw_donor_median_day_spearman": rho,
        "flipped": bool(flipped),
        "oriented_donor_median_day_spearman": oriented_rho,
        **{
            f"{day}_oriented_donor_median": float(value)
            for day, value in zip(day_order, oriented_medians)
        },
    }


def _read_normalized_expression(
    raw_archive: Path,
    metadata: pd.DataFrame,
    *,
    expected_features: int,
) -> tuple[pd.DataFrame, float]:
    counts = pd.read_csv(
        raw_archive,
        compression="zip",
        index_col=0,
        dtype={str(cell_id): np.float32 for cell_id in metadata["cell_id"]},
    )
    if counts.shape != (expected_features, len(metadata)):
        raise ValueError("CoreBench raw-count matrix shape differs from the schema audit")
    if counts.columns.astype(str).tolist() != metadata["cell_id"].astype(str).tolist():
        raise ValueError("CoreBench raw-count cell order differs from metadata")
    expression = counts.to_numpy(copy=False)
    if np.any(expression < 0) or not np.isfinite(expression).all():
        raise ValueError("CoreBench fractional counts contain negative or nonfinite values")
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
    return counts, reference_library


def materialize_corebench_coordinate(
    *,
    config_path: str | Path,
    repository_root: str | Path,
    freeze_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config_file = Path(config_path).resolve()
    config = load_corebench_config(config_file)
    freeze = Path(freeze_dir).resolve()
    freeze_record = validate_corebench_freeze_output(
        config_path=config_file,
        repository_root=root,
        output_dir=freeze,
    )
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"CoreBench coordinate output exists: {output}")

    receipt_path = root / config["bindings"]["acquisition_receipt"]["relative_path"]
    acquisition_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    raw_entry = next(
        item for item in acquisition_receipt["files"] if item["file_id"] == "endoderm_raw_counts"
    )
    if raw_entry["publisher_checksum_status"] != "verified":
        raise ValueError("CoreBench raw archive publisher checksum is not verified")
    raw_archive = receipt_path.parent / raw_entry["local_relative_path"]
    if _hash_file(raw_archive) != raw_entry["local_sha256"]:
        raise ValueError("CoreBench raw archive SHA-256 differs from acquisition receipt")

    raw_schema_path = root / config["bindings"]["raw_schema_audit"]["relative_path"]
    schema = json.loads(raw_schema_path.read_text(encoding="utf-8"))
    if not schema.get("fractional_count_contract_confirmed"):
        raise ValueError("CoreBench fractional-count schema gate is not satisfied")

    metadata_path = (
        root
        / "data_external/trajpathmix_acquisitions/hipsci_endoderm_125_v2/source/cell_metadata_cols.tsv"
    )
    metadata = pd.read_csv(
        metadata_path,
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
        dtype={"cell_name": "string", "donor": "string", "donor_long_id": "string", "experiment": "string", "day": "string"},
    ).rename(
        columns={
            "cell_name": "cell_id",
            "donor": "donor_id",
            "donor_long_id": "line_id",
            "experiment": "experiment_id",
        }
    )
    day_order = list(config["analysis_coordinate"]["coarse_anchor_order"])
    if sorted(metadata["day"].unique()) != sorted(day_order):
        raise ValueError("CoreBench observed coarse-day set differs from the freeze")

    counts, reference_library = _read_normalized_expression(
        raw_archive,
        metadata,
        expected_features=int(schema["n_features"]),
    )
    manifest = pd.read_csv(freeze / GENE_FOLD_FILE, sep="\t")
    coordinate_manifest = manifest.loc[manifest["coordinate_gene"]].copy()
    row_lookup = pd.Series(np.arange(len(counts.index)), index=counts.index.astype(str))
    if not coordinate_manifest["feature_id"].astype(str).isin(row_lookup.index).all():
        raise ValueError("A frozen coordinate gene is absent from the raw matrix")
    row_indices = coordinate_manifest["feature_id"].astype(str).map(row_lookup).to_numpy(dtype=int)
    expression = counts.to_numpy(copy=False)
    days = metadata["day"].astype(str).to_numpy()
    donors = metadata["donor_id"].astype(str).to_numpy()
    cell_ids = metadata["cell_id"].astype(str).to_numpy()
    fold_spec = config["analysis_coordinate"]["gene_fold_cross_fitting"]
    fold_ranks: list[np.ndarray] = []
    audit_rows: list[dict[str, Any]] = []
    for fold in range(1, int(fold_spec["n_folds"]) + 1):
        training = coordinate_manifest["gene_fold"].ne(fold).to_numpy()
        training_rows = row_indices[training]
        pca = PCA(
            n_components=1,
            svd_solver="randomized",
            random_state=int(fold_spec["seed"]) + fold,
        )
        pc1 = pca.fit_transform(expression[training_rows, :].T).ravel()
        oriented, orientation = _orient_by_donor_median_day(
            pc1,
            donors,
            days,
            day_order,
        )
        ranks = _stable_within_day_midrank(oriented, days, cell_ids)
        fold_ranks.append(ranks)
        audit_rows.append(
            {
                "fold": fold,
                "pca_random_state": int(fold_spec["seed"]) + fold,
                "n_training_coordinate_genes": int(training.sum()),
                "n_held_out_coordinate_genes": int((~training).sum()),
                "pc1_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
                **orientation,
            }
        )
    rank_matrix = np.column_stack(fold_ranks)
    within_day_rank = np.median(rank_matrix, axis=1)
    day_index = pd.Series(days).map({day: index for index, day in enumerate(day_order)}).to_numpy(dtype=float)
    coordinate = (day_index + within_day_rank) / len(day_order)
    if np.any(coordinate < 0) or np.any(coordinate >= 1) or not np.isfinite(coordinate).all():
        raise ValueError("Materialized CoreBench coordinate is outside [0, 1)")

    pairwise = []
    for left in range(rank_matrix.shape[1]):
        for right in range(left + 1, rank_matrix.shape[1]):
            pairwise.append(float(spearmanr(rank_matrix[:, left], rank_matrix[:, right]).statistic))
    coordinate_table = metadata[
        ["cell_id", "donor_id", "line_id", "experiment_id", "day"]
    ].copy()
    coordinate_table["within_day_crossfit_rank"] = within_day_rank
    coordinate_table["corebench_coordinate"] = coordinate
    fold_audit = pd.DataFrame(audit_rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / f".{output.name}.create.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FileExistsError(f"CoreBench coordinate is locked: {lock}") from exc
    temporary: Path | None = None
    try:
        os.close(descriptor)
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
        coordinate_table.to_csv(
            temporary / COORDINATE_FILE,
            sep="\t",
            index=False,
            compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
            lineterminator="\n",
        )
        fold_audit.to_csv(temporary / FOLD_AUDIT_FILE, sep="\t", index=False, lineterminator="\n")
        receipt = {
            "schema_name": "trajpathmix_corebench_coordinate_materialization_receipt",
            "schema_version": "1.0.0",
            "freeze_id": "trajpathmix_corebench_v1",
            "config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
            "corebench_freeze_build_record_sha256": freeze_record["build_record_sha256"],
            "coordinate_file": COORDINATE_FILE,
            "coordinate_file_sha256": _hash_file(temporary / COORDINATE_FILE),
            "n_cells": int(len(coordinate_table)),
            "n_coordinate_genes": int(len(coordinate_manifest)),
            "n_gene_folds": int(rank_matrix.shape[1]),
            "minimum_cross_fold_spearman": float(min(pairwise)),
            "median_cross_fold_spearman": float(np.median(pairwise)),
            "normalization": config["analysis_coordinate"]["within_day_ordering"]["normalization"],
            "normalization_reference_library": reference_library,
            "coordinate_interpretation": config["analysis_coordinate"]["interpretation"],
            "real_pathway_outcomes_read": False,
            "pathway_scoring_performed": False,
            "pseudo_conditions_generated": False,
            "injection_results_generated": False,
            "next_gate": "review_coordinate_receipt_then_authorize_500_replicate_null_smoke",
        }
        _write_json(receipt, temporary / RECEIPT_FILE)
        record = {
            "schema_name": "trajpathmix_corebench_coordinate_materialization_build_record",
            "schema_version": "1.0.0",
            "config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
            "implementation_file": "pyfgsea/trajpathmix_corebench_coordinate.py",
            "implementation_sha256": _hash_file(Path(__file__).resolve()),
            "artifacts": {
                name: {"sha256": _hash_file(temporary / name), "bytes": (temporary / name).stat().st_size}
                for name in (COORDINATE_FILE, FOLD_AUDIT_FILE, RECEIPT_FILE)
            },
            "expression_values_read": True,
            "pathway_outcomes_read": False,
            "pathway_scoring_performed": False,
            "evidence_revision_mode": "create_only_append_only",
        }
        _write_json(record, temporary / BUILD_RECORD_FILE)
        os.rename(temporary, output)
        temporary = None
        result = dict(record)
        result["output_dir"] = str(output)
        result["build_record_sha256"] = _hash_file(output / BUILD_RECORD_FILE)
        result["coordinate_file_sha256"] = receipt["coordinate_file_sha256"]
        result["minimum_cross_fold_spearman"] = receipt["minimum_cross_fold_spearman"]
        return result
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
        lock.unlink(missing_ok=True)


__all__ = [
    "BUILD_RECORD_FILE",
    "COORDINATE_FILE",
    "FOLD_AUDIT_FILE",
    "RECEIPT_FILE",
    "materialize_corebench_coordinate",
]
