from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import zipfile
from typing import Any

import pandas as pd

from pyfgsea.trajpathmix_dataset_acquisition import validate_acquisition_receipt
from pyfgsea.trajpathmix_endoderm_benchmark_freeze import (
    FROZEN_CONFIG_PAYLOAD_SHA256,
    load_endoderm_benchmark_freeze_config,
)


AUDIT_FILE = "endoderm_raw_schema_audit_v1.json"
BUILD_RECORD_FILE = "endoderm_raw_schema_audit_build_record_v1.json"


def _hash_file(path: Path, algorithm: str, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(value: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def inspect_endoderm_raw_csv_zip(
    archive_path: str | Path,
    metadata_path: str | Path,
    *,
    expected_cell_column: str = "cell_name",
    numeric_sample_rows: int = 32,
    numeric_sample_columns: int = 512,
) -> dict[str, Any]:
    """Inspect a verified CSV ZIP without scoring genes or pathways."""

    archive = Path(archive_path).resolve()
    metadata_file = Path(metadata_path).resolve()
    cells = pd.read_csv(
        metadata_file, sep="\t", usecols=[expected_cell_column], dtype="string"
    )[expected_cell_column]
    if cells.isna().any() or not cells.is_unique:
        raise ValueError("Metadata cell IDs must be complete and unique")
    metadata_cells = cells.astype(str).tolist()
    metadata_cell_set = set(metadata_cells)

    csv.field_size_limit(max(csv.field_size_limit(), 2**31 - 1))
    with zipfile.ZipFile(archive) as zipped:
        members = [item for item in zipped.infolist() if not item.is_dir()]
        if len(members) != 1:
            raise ValueError(
                f"Expected exactly one raw-count archive member, observed {len(members)}"
            )
        member = members[0]
        with zipped.open(member, "r") as binary:
            text = (line.decode("utf-8-sig") for line in binary)
            reader = csv.reader(text)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ValueError("Raw-count CSV is empty") from exc
            if len(header) < 2:
                raise ValueError("Raw-count CSV header is too short")
            matrix_cells = header[1:]
            if len(matrix_cells) != len(set(matrix_cells)):
                raise ValueError("Raw-count matrix cell columns are not unique")
            cell_axis_exact_order = matrix_cells == metadata_cells
            cell_axis_exact_set = set(matrix_cells) == metadata_cell_set
            missing_metadata_cells = sorted(metadata_cell_set - set(matrix_cells))
            extra_matrix_cells = sorted(set(matrix_cells) - metadata_cell_set)
            if not cell_axis_exact_set:
                raise ValueError(
                    "Raw-count cell columns do not match metadata one-to-one"
                )

            feature_ids: set[str] = set()
            duplicate_features: list[str] = []
            row_count = 0
            row_width_mismatch_count = 0
            numeric_values_examined = 0
            fractional_values_observed = 0
            negative_values_observed = 0
            nonfinite_values_observed = 0
            for row in reader:
                row_count += 1
                if len(row) != len(header):
                    row_width_mismatch_count += 1
                    continue
                feature_id = row[0]
                if feature_id in feature_ids:
                    duplicate_features.append(feature_id)
                feature_ids.add(feature_id)
                if row_count <= numeric_sample_rows:
                    for value_text in row[1 : 1 + numeric_sample_columns]:
                        value = float(value_text)
                        numeric_values_examined += 1
                        if value != value or value in (float("inf"), float("-inf")):
                            nonfinite_values_observed += 1
                        elif value < 0:
                            negative_values_observed += 1
                        elif abs(value - round(value)) > 1e-12:
                            fractional_values_observed += 1

    if row_width_mismatch_count:
        raise ValueError(
            f"Raw-count CSV has {row_width_mismatch_count} row-width mismatches"
        )
    if duplicate_features:
        raise ValueError("Raw-count feature IDs are not unique")
    if fractional_values_observed == 0:
        raise ValueError("Fractional Salmon values were not observed in the sample")
    if negative_values_observed or nonfinite_values_observed:
        raise ValueError("Invalid numeric values were observed in the raw-count sample")
    return {
        "archive_member_name": member.filename,
        "archive_member_compressed_size_bytes": int(member.compress_size),
        "archive_member_uncompressed_size_bytes": int(member.file_size),
        "archive_member_crc32": f"{member.CRC:08x}",
        "matrix_orientation": "features_by_cells",
        "feature_id_header": header[0],
        "n_features": int(row_count),
        "n_matrix_cell_columns": int(len(matrix_cells)),
        "n_metadata_cells": int(len(metadata_cells)),
        "cell_axis_exact_set": bool(cell_axis_exact_set),
        "cell_axis_exact_order": bool(cell_axis_exact_order),
        "missing_metadata_cell_count": int(len(missing_metadata_cells)),
        "extra_matrix_cell_count": int(len(extra_matrix_cells)),
        "row_width_mismatch_count": int(row_width_mismatch_count),
        "duplicate_feature_count": int(len(duplicate_features)),
        "numeric_values_examined": int(numeric_values_examined),
        "fractional_values_observed": int(fractional_values_observed),
        "negative_values_observed": int(negative_values_observed),
        "nonfinite_values_observed": int(nonfinite_values_observed),
        "fractional_count_contract_confirmed": True,
        "integer_coercion_allowed": False,
    }


def build_endoderm_raw_schema_audit(
    *,
    portfolio_config_path: str | Path,
    benchmark_config_path: str | Path,
    acquisition_dir: str | Path,
    repository_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    benchmark_config_file = Path(benchmark_config_path).resolve()
    benchmark = load_endoderm_benchmark_freeze_config(benchmark_config_file)
    acquisition = Path(acquisition_dir).resolve()
    receipt = validate_acquisition_receipt(
        config_path=portfolio_config_path,
        candidate_id=benchmark["bindings"]["candidate_id"],
        output_dir=acquisition,
    )
    files = {item["file_id"]: item for item in receipt["files"]}
    raw_item = files["endoderm_raw_counts"]
    metadata_item = files["endoderm_cell_metadata"]
    archive_path = acquisition / raw_item["local_relative_path"]
    metadata_path = acquisition / metadata_item["local_relative_path"]
    schema = inspect_endoderm_raw_csv_zip(archive_path, metadata_path)
    audit = {
        "schema_name": "trajpathmix_endoderm_raw_schema_audit",
        "schema_version": "1.0.0",
        "candidate_id": benchmark["bindings"]["candidate_id"],
        "benchmark_freeze_config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
        "acquisition_receipt_sha256": receipt["receipt_sha256"],
        "raw_archive_relative_path": str(archive_path.relative_to(root)).replace(
            "\\", "/"
        ),
        "raw_archive_size_bytes": int(raw_item["size_bytes"]),
        "raw_archive_publisher_md5": raw_item["publisher_checksum_observed"],
        "raw_archive_local_sha256": raw_item["local_sha256"],
        "metadata_relative_path": str(metadata_path.relative_to(root)).replace(
            "\\", "/"
        ),
        "metadata_local_sha256": metadata_item["local_sha256"],
        **schema,
        "day_used_for_trajectory": False,
        "deposited_trajectory_fields_used_for_trajectory": False,
        "pathway_outcomes_read": False,
        "pathway_scoring_performed": False,
        "next_gate": "blinded_trajectory_and_outcome_blind_estimability_preflight",
    }
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Endoderm raw-schema output exists: {output}")
    output.mkdir(parents=True)
    _write_json(audit, output / AUDIT_FILE)
    record = {
        "schema_name": "trajpathmix_endoderm_raw_schema_audit_build_record",
        "schema_version": "1.0.0",
        "candidate_id": benchmark["bindings"]["candidate_id"],
        "implementation_file": "pyfgsea/trajpathmix_endoderm_raw_schema.py",
        "implementation_sha256": _hash_file(Path(__file__).resolve(), "sha256"),
        "benchmark_config_file": str(benchmark_config_file.relative_to(root)).replace(
            "\\", "/"
        ),
        "benchmark_config_file_sha256": _hash_file(
            benchmark_config_file, "sha256"
        ),
        "benchmark_config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
        "acquisition_receipt_sha256": receipt["receipt_sha256"],
        "artifacts": {
            AUDIT_FILE: {
                "schema_name": audit["schema_name"],
                "sha256": _hash_file(output / AUDIT_FILE, "sha256"),
            }
        },
        "expression_matrix_schema_read": True,
        "expression_values_sampled_for_fractional_contract_only": True,
        "pathway_outcomes_read": False,
        "pathway_scoring_performed": False,
    }
    _write_json(record, output / BUILD_RECORD_FILE)
    result = dict(record)
    result["output_dir"] = str(output)
    result["build_record_sha256"] = _hash_file(output / BUILD_RECORD_FILE, "sha256")
    result["validation_status"] = "pass_raw_schema_and_one_to_one_cell_join"
    return result


__all__ = [
    "AUDIT_FILE",
    "BUILD_RECORD_FILE",
    "build_endoderm_raw_schema_audit",
    "inspect_endoderm_raw_csv_zip",
]
