"""Contracts and staging utilities for the versioned T21 data product.

The public deposits are large and heterogeneous.  This module deliberately
separates evidence inventory, count staging, and release finalization so that
an interrupted or metadata-only run cannot leave an apparently complete data
product behind.  All biological donor identifiers remain accession-namespaced
until a correspondence has direct supporting evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
from hashlib import sha256
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import requests
from scipy import sparse
from scipy.io import mmread

from .t21_expression_preprocessing import (
    FORMAL_EXPRESSION_CONTRACT_VERSION,
    FORMAL_EXPRESSION_TARGET_SUM,
    formal_expression_preprocessing_contract,
    formal_expression_preprocessing_contract_sha256,
    formal_expression_preprocessing_source_sha256,
    validate_t21_formal_expression,
)


PRODUCT_VERSION = "1.0.0"
PRIMARY_ACCESSION = "E-MTAB-13067"
ACCESSION_MODALITY = {
    "E-MTAB-13067": "scRNA",
    "E-MTAB-13070": "multiome",
    "E-MTAB-13062": "Visium",
}

FINAL_PRODUCT_NAMES = {
    "scrna": "t21_scRNA_analysis_ready_v1.h5ad",
    "donor_design": "t21_donor_design_v1.tsv",
    "trajectory": "t21_trajectory_draws_v1.zarr",
    "fates": "t21_fate_probabilities_v1.parquet",
    "provenance": "t21_data_provenance_manifest.json",
}

REQUIRED_SCRNA_OBS = (
    "cell_id",
    "donor_id",
    "source_donor_id",
    "condition",
    "condition_original",
    "pcw",
    "sex",
    "tissue",
    "sort_gate",
    "sort_gate_original",
    "library_id",
    "technical_batch",
    "original_cell_type",
    "original_cell_type_source",
    "analysis_cell_type",
    "lineage_inclusion",
    "lineage_inclusion_reason",
    "original_barcode",
)

REQUIRED_SCRNA_VAR = (
    "gene_id_original",
    "gene_symbol",
    "feature_type",
    "chromosome",
    "is_chr21",
    "genome_build",
    "gene_mapping_status",
)

REQUIRED_DONOR_DESIGN_COLUMNS = (
    "donor_id",
    "condition",
    "pcw",
    "sex",
    "available_tissues",
    "available_sort_gates",
    "available_modalities",
    "number_of_cells_by_gate",
    "number_of_cells_in_primary_lineage",
    "donor_correspondence_status",
    "trajectory_bin_coverage_fraction",
)

REQUIRED_EXTERNAL_2021_OVERLAP_COLUMNS = (
    "audit_schema_version",
    "donor_id_2021",
    "processing_unit_type",
    "member_donor_ids",
    "condition",
    "sex",
    "reported_age",
    "pcw_numeric",
    "direct_accessions",
    "pooled_membership_accessions",
    "source_library_ids",
    "post_qc_cells",
    "cell_count_scope",
    "candidate_2024_scrna_donor_ids",
    "candidate_2024_person_tokens",
    "candidate_match_rule_id",
    "overlap_status",
    "public_crosswalk_available",
    "genetic_identity_verified",
    "analysis_unit_eligible",
    "independent_replication_eligible",
    "analysis_action",
    "evidence_source_ids",
)

REQUIRED_EXTERNAL_2021_SOURCE_COLUMNS = (
    "source_schema_version",
    "source_id",
    "role",
    "source_type",
    "accession",
    "url",
    "original_file_name",
    "version_or_commit",
    "repository_relative_path",
    "retrieval_status",
    "retrieved_at_utc",
    "bytes",
    "bytes_status",
    "sha256",
    "sha256_status",
    "file_format",
    "delimiter",
    "orientation",
    "value_semantics",
    "cell_id_join_key",
    "pool_handling",
    "donor_scope",
    "formal_use_status",
    "formal_use",
    "validation_status",
)

REQUIRED_EXTERNAL_2021_CONSTRAINT_COLUMNS = (
    "constraint_schema_version",
    "constraint_id",
    "constraint_scope",
    "constraint_status",
    "value_type",
    "required_value",
    "unit",
    "gate_action",
    "evidence_source_ids",
)

FATE_PROBABILITY_COLUMNS = (
    "erythroid_probability",
    "megakaryocyte_probability",
    "myeloid_probability",
    "other_probability",
)

REQUIRED_FATE_COLUMNS = (
    "cell_id",
    *FATE_PROBABILITY_COLUMNS,
    "fate_eligible",
    "fate_ineligibility_reason",
    "fate_model_id",
    "trajectory_draw_id",
    "terminal_definition_hash",
)

_CONDITION_MAP = {
    "normal": "disomy",
    "healthy": "disomy",
    "disomy": "disomy",
    "trisomy 21": "T21",
    "down syndrome": "T21",
    "down's syndrome": "T21",
    "t21": "T21",
    "ts21": "T21",
}

_GATE_MAP = {
    "CD34+/Lin-": "cd34_pos_lin_neg",
    "CD34+": "cd34_pos",
    "CD45+": "cd45_pos",
    "CD235a-": "cd235a_neg",
    "CD45+ (+) CD45-/CD235a-/CD71-": "cd45_pos_plus_triple_negative",
    "CD45-/CD235a-/CD71-": "cd45_neg_cd235a_neg_cd71_neg",
}

_GATE_PRIORITY = ("cd45_pos", "cd235a_neg")


def utc_now() -> str:
    """Return a seconds-resolution UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_json(value: Any) -> str:
    """Serialize a value in the stable form used by hashes and TSV cells."""

    def normalize_numpy(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(
            f"Object of type {type(item).__name__} is not JSON serializable"
        )

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=normalize_numpy,
    )


def expression_contract_metadata(
    matrix: Any,
    *,
    counts: Any,
) -> dict[str, Any]:
    validation = validate_t21_formal_expression(counts, matrix)
    return {
        "contract": formal_expression_preprocessing_contract(),
        "validation": validation,
    }


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(path: str | Path) -> str:
    """Hash a directory from sorted relative paths and per-file SHA256 values."""
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"Tree digest requires a directory: {root}")
    digest = sha256()
    for file_path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = file_path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(file_path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def cell_id_set_hash(cell_ids: Iterable[str]) -> str:
    """Hash a sorted cell-ID set, rejecting duplicates."""
    values = [str(value) for value in cell_ids]
    if len(values) != len(set(values)):
        raise ValueError("Cell IDs are not unique")
    return sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def ordered_id_hash(values: Iterable[str]) -> str:
    """Hash IDs in their stored order."""
    return sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def canonicalize_condition(value: Any) -> str:
    raw = str(value).strip()
    key = re.sub(r"\s+", " ", raw).lower()
    if key not in _CONDITION_MAP:
        raise ValueError(f"Unrecognized condition {raw!r}")
    return _CONDITION_MAP[key]


def _ascii_minus(value: Any) -> str:
    return str(value).strip().replace("−", "-").replace("–", "-").replace("—", "-")


def canonicalize_sort_gate(value: Any) -> str:
    """Map the six deposited gates without collapsing biologically distinct gates."""
    raw = _ascii_minus(value)
    raw = re.sub(r"\s+", " ", raw)
    if raw not in _GATE_MAP:
        raise ValueError(f"Unrecognized sort gate {value!r}; update the explicit mapping")
    return _GATE_MAP[raw]


def parse_pcw(value: Any) -> float:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(value))
    if match is None:
        raise ValueError(f"Could not parse post-conception weeks from {value!r}")
    return float(match.group(1))


def make_cell_id(accession: str, library_id: str, barcode: str) -> str:
    for label, value in {
        "accession": accession,
        "library_id": library_id,
        "barcode": barcode,
    }.items():
        if not str(value).strip() or "|" in str(value):
            raise ValueError(f"Invalid {label} for deterministic cell ID: {value!r}")
    return f"{accession}|{library_id}|{barcode}"


def namespaced_donor_id(accession: str, source_donor_id: str) -> str:
    return f"{accession}:{source_donor_id}"


def source_person_token(value: Any) -> str:
    """Normalize the explicit HDBR/person number embedded in an original name.

    This is intentionally limited to the deposited ``original source name``
    field.  It is not derived from the accession-local public donor label.
    """
    numbers = re.findall(r"(?<!\d)(\d{4,6})(?!\d)", str(value))
    if not numbers:
        return "not_reported"
    return sorted(numbers, key=lambda item: (-len(item), item))[0]


def _unique_value(group: pd.DataFrame, column: str, *, allow_missing: bool = False) -> str:
    if column not in group:
        if allow_missing:
            return ""
        raise ValueError(f"Required SDRF column is missing: {column}")
    values = group[column].dropna().astype(str).str.strip()
    values = values[values.ne("")].drop_duplicates().tolist()
    if not values and allow_missing:
        return ""
    if len(values) != 1:
        raise ValueError(
            f"{column!r} is not invariant within {group['Source Name'].iloc[0]!r}: {values}"
        )
    return values[0]


def _json_unique(group: pd.DataFrame, columns: Sequence[str]) -> str:
    values: set[str] = set()
    for column in columns:
        if column in group:
            values.update(
                value
                for value in group[column].dropna().astype(str).str.strip()
                if value
            )
    return stable_json(sorted(values))


def read_sdrf_library_manifest(path: str | Path, accession: str) -> pd.DataFrame:
    """Build one verified library row per SDRF ``Source Name``.

    The SDRF contains one row per FASTQ, so source-level fields and processed
    file names are checked for invariance before deduplication.
    """
    path = Path(path)
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if "Source Name" not in frame:
        raise ValueError(f"SDRF has no Source Name column: {path}")
    derived_columns = [
        column for column in frame.columns if column.startswith("Derived Array Data File")
    ]
    if not derived_columns:
        raise ValueError(f"SDRF has no processed-data columns: {path}")
    run_columns = [
        column
        for column in frame.columns
        if column in {"Assay Name", "Comment[ENA_RUN]", "Comment[technical replicate group]"}
    ]
    rows: list[dict[str, Any]] = []
    for source_name, group in frame.groupby("Source Name", sort=True, dropna=False):
        source_name = str(source_name).strip()
        source_donor = _unique_value(group, "Characteristics[individual]")
        condition_original = _unique_value(group, "Characteristics[disease]")
        tissue = _unique_value(group, "Characteristics[organism part]").lower()
        pcw_original = _unique_value(group, "Characteristics[age]")
        gate_original = _unique_value(
            group, "Characteristics[immunophenotype]", allow_missing=True
        )
        processed_files = json.loads(_json_unique(group, derived_columns))
        original_source_column = next(
            (
                column
                for column in (
                    "Comment[original source name]",
                    "Characteristics[original source name]",
                )
                if column in group
            ),
            "",
        )
        original_source_name = (
            _unique_value(group, original_source_column, allow_missing=True)
            if original_source_column
            else ""
        )
        modality = ACCESSION_MODALITY.get(accession, "unknown")
        assay_component = ""
        if modality == "multiome":
            if source_name.endswith("_GEX"):
                assay_component = "RNA"
            elif source_name.endswith("_ATAC"):
                assay_component = "ATAC"
        rows.append(
            {
                "accession": accession,
                "library_id": source_name,
                "biological_library_id": re.sub(r"_(?:GEX|ATAC)$", "", source_name),
                "source_donor_id": source_donor,
                "donor_id": namespaced_donor_id(accession, source_donor),
                "condition_original": condition_original,
                "condition": canonicalize_condition(condition_original),
                "pcw_original": pcw_original,
                "pcw": parse_pcw(pcw_original),
                "sex": "not_reported",
                "tissue": tissue,
                "sort_gate_original": gate_original or "not_applicable",
                "sort_gate": (
                    canonicalize_sort_gate(gate_original)
                    if gate_original
                    else "not_applicable"
                ),
                "modality": modality,
                "assay_component": assay_component,
                "original_source_name": original_source_name,
                "original_source_name_field": original_source_column or "not_reported",
                "source_person_token": source_person_token(original_source_name),
                "processed_files": stable_json(processed_files),
                "technical_batch": "not_resolved",
                "technical_batch_evidence": _json_unique(group, run_columns),
                "sdrf_path": path.as_posix(),
            }
        )
    result = pd.DataFrame(rows).sort_values(["accession", "library_id"]).reset_index(drop=True)
    if result["library_id"].duplicated().any():
        raise AssertionError("SDRF library IDs are not unique")
    return result


def read_author_scrna_metadata(path: str | Path) -> pd.DataFrame:
    """Read the public author design table without treating it as deposited truth."""
    source = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = {"organ", "sorting", "age", "environment", "ID1", "ID2"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Author metadata is missing columns: {sorted(missing)}")
    result = pd.DataFrame(
        {
            "original_source_name": source["ID2"].astype(str),
            "author_internal_donor_id": source["ID1"].astype(str),
            "author_tissue": source["organ"].str.lower(),
            "author_sort_gate_original": source["sorting"].map(_ascii_minus),
            "author_sort_gate": source["sorting"].map(canonicalize_sort_gate),
            "author_pcw": source["age"].map(parse_pcw),
            "author_condition": source["environment"].map(canonicalize_condition),
            "author_estimated_cells": pd.to_numeric(
                source.get("# estimated cells", ""), errors="coerce"
            ).astype("Int64"),
            "author_isolated_cells": pd.to_numeric(
                source.get("# isolated cells", ""), errors="coerce"
            ).astype("Int64"),
        }
    )
    if result["original_source_name"].duplicated().any():
        raise ValueError("Author metadata original source names are not unique")
    return result.sort_values("original_source_name").reset_index(drop=True)


def read_supplementary_library_qc(path: str | Path) -> pd.DataFrame:
    """Read Nature Supplementary Tables 1-2 as library-level QC evidence.

    In the published workbook the column named ``Tissue`` contains the
    condition, while ``Organ`` contains the anatomical tissue.  This function
    encodes that documented exception rather than guessing from the header.
    """
    path = Path(path)
    frames = []
    for sheet_name in ("Suppl. Tab 1", "Suppl. Tab 2"):
        frame = pd.read_excel(
            path,
            sheet_name=sheet_name,
            header=2,
            engine="openpyxl",
        )
        required = {
            "Sample ID",
            "Tissue",
            "Sex",
            "Organ",
            "Sorting strategy",
            "Age",
            "Median # genes per cell after QC",
            "Median # UMIs per cell after QC",
            "# cells after QC",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{sheet_name} is missing columns: {sorted(missing)}")
        frame = frame.loc[frame["Sample ID"].notna()].copy()
        frame["supplement_sheet"] = sheet_name
        frame["supplement_row"] = frame.index + 4
        frames.append(frame)
    source = pd.concat(frames, ignore_index=True)
    sample_ids = source["Sample ID"].astype(str).str.strip()
    original_names = sample_ids.str.replace(r"^\d+\s+", "", regex=True)
    sex_map = {"F": "female", "M": "male"}
    result = pd.DataFrame(
        {
            "original_source_name": original_names,
            "supplement_sample_id": sample_ids,
            "supplement_condition_original": source["Tissue"].astype(str),
            "supplement_condition": source["Tissue"].map(canonicalize_condition),
            "supplement_sex_original": source["Sex"].astype(str),
            "supplement_sex": source["Sex"].map(sex_map),
            "supplement_tissue": source["Organ"].astype(str).str.lower(),
            "supplement_sort_gate_original": source["Sorting strategy"].map(
                _ascii_minus
            ),
            "supplement_sort_gate": source["Sorting strategy"].map(
                canonicalize_sort_gate
            ),
            "supplement_pcw": source["Age"].map(parse_pcw),
            "post_qc_median_genes": pd.to_numeric(
                source["Median # genes per cell after QC"], errors="raise"
            ),
            "post_qc_median_umis": pd.to_numeric(
                source["Median # UMIs per cell after QC"], errors="raise"
            ),
            "post_qc_expected_cells": pd.to_numeric(
                source["# cells after QC"], errors="raise"
            ).astype(np.int64),
            "supplement_sheet": source["supplement_sheet"],
            "supplement_row": source["supplement_row"].astype(np.int64),
        }
    )
    if len(result) != 86 or result["original_source_name"].duplicated().any():
        raise ValueError(
            "Supplementary Tables 1-2 must contain 86 unique scRNA libraries"
        )
    if result["supplement_sex"].isna().any():
        raise ValueError("Supplement contains an unrecognized sex value")
    return result.sort_values("original_source_name").reset_index(drop=True)


def attach_author_metadata(
    libraries: pd.DataFrame, author_metadata: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join and audit the public SDRF against the public author design table."""
    primary = libraries[libraries["accession"].eq(PRIMARY_ACCESSION)].copy()
    other = libraries[~libraries["accession"].eq(PRIMARY_ACCESSION)].copy()
    merged = primary.merge(
        author_metadata, on="original_source_name", how="left", validate="one_to_one"
    )
    comparison_columns = {
        "condition": "author_condition",
        "pcw": "author_pcw",
        "tissue": "author_tissue",
        "sort_gate": "author_sort_gate",
    }
    discrepancy_rows: list[dict[str, Any]] = []
    statuses: list[str] = []
    for _, row in merged.iterrows():
        fields = [
            field
            for field, author_field in comparison_columns.items()
            if pd.isna(row.get(author_field)) or str(row[field]) != str(row[author_field])
        ]
        status = "concordant" if not fields else "conflict_unresolved"
        statuses.append(status)
        for field in fields:
            discrepancy_rows.append(
                {
                    "accession": PRIMARY_ACCESSION,
                    "library_id": row["library_id"],
                    "original_source_name": row["original_source_name"],
                    "field": field,
                    "sdrf_value": row[field],
                    "author_value": row.get(comparison_columns[field], ""),
                    "status": "conflict_unresolved",
                    "analysis_action": "do_not_silently_pool_conflicting_library",
                }
            )
    merged["author_metadata_status"] = statuses
    result = pd.concat([merged, other], ignore_index=True, sort=False)
    result["author_metadata_status"] = result["author_metadata_status"].fillna(
        "not_applicable"
    )
    discrepancies = pd.DataFrame(
        discrepancy_rows,
        columns=[
            "accession",
            "library_id",
            "original_source_name",
            "field",
            "sdrf_value",
            "author_value",
            "status",
            "analysis_action",
        ],
    )
    return result.sort_values(["accession", "library_id"]).reset_index(drop=True), discrepancies


def attach_supplementary_library_qc(
    libraries: pd.DataFrame, supplement: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach sex/QC targets and resolve the single deposited gate discrepancy."""
    primary = libraries[libraries["accession"].eq(PRIMARY_ACCESSION)].copy()
    other = libraries[~libraries["accession"].eq(PRIMARY_ACCESSION)].copy()
    merged = primary.merge(
        supplement, on="original_source_name", how="left", validate="one_to_one"
    )
    if merged["supplement_sample_id"].isna().any():
        missing = merged.loc[
            merged["supplement_sample_id"].isna(), "original_source_name"
        ].tolist()
        raise ValueError(f"Supplement does not cover scRNA libraries: {missing}")
    for field, supplement_field in (
        ("condition", "supplement_condition"),
        ("pcw", "supplement_pcw"),
        ("tissue", "supplement_tissue"),
    ):
        mismatch = merged[field].astype(str).ne(merged[supplement_field].astype(str))
        if mismatch.any():
            raise ValueError(
                f"Supplement disagrees with SDRF for {field}: "
                f"{merged.loc[mismatch, 'library_id'].tolist()}"
            )
    resolution_rows = []
    statuses = []
    resolved_gates = []
    for _, row in merged.iterrows():
        sdrf_gate = str(row["sort_gate"])
        supplement_gate = str(row["supplement_sort_gate"])
        author_gate = str(row.get("author_sort_gate", ""))
        if sdrf_gate == supplement_gate and (
            not author_gate or author_gate == supplement_gate
        ):
            status = "concordant_three_source" if author_gate else "concordant"
        elif supplement_gate == author_gate and supplement_gate != sdrf_gate:
            status = "resolved_by_paper_supplement_and_author_pipeline"
        else:
            status = "conflict_unresolved"
        statuses.append(status)
        resolved_gates.append(
            supplement_gate if status != "conflict_unresolved" else "not_resolved"
        )
        if status != "concordant_three_source":
            resolution_rows.append(
                {
                    "library_id": row["library_id"],
                    "original_source_name": row["original_source_name"],
                    "sort_gate_sdrf": sdrf_gate,
                    "sort_gate_author": author_gate,
                    "sort_gate_supplement": supplement_gate,
                    "sort_gate_analysis": resolved_gates[-1],
                    "resolution_status": status,
                    "resolution_evidence": (
                        f"Nature:{row['supplement_sheet']}:row_{int(row['supplement_row'])};"
                        "author_scRNA.metadata.txt"
                    ),
                }
            )
    merged["sort_gate_sdrf"] = merged["sort_gate"]
    merged["sort_gate_sdrf_original"] = merged["sort_gate_original"]
    merged["sort_gate_resolution_status"] = statuses
    merged["sort_gate"] = resolved_gates
    merged["sort_gate_original"] = merged["supplement_sort_gate_original"]
    merged["sex"] = merged["supplement_sex"]
    merged["sex_source"] = (
        "Nature_s41586-024-07946-4_Supplementary_Tables_1_2"
    )
    other["sort_gate_sdrf"] = other["sort_gate"]
    other["sort_gate_sdrf_original"] = other["sort_gate_original"]
    other["sort_gate_resolution_status"] = "not_applicable"
    other["sex_source"] = "not_reported_in_official_sdrf_or_ena"
    result = pd.concat([merged, other], ignore_index=True, sort=False)
    resolution = pd.DataFrame(
        resolution_rows,
        columns=[
            "library_id",
            "original_source_name",
            "sort_gate_sdrf",
            "sort_gate_author",
            "sort_gate_supplement",
            "sort_gate_analysis",
            "resolution_status",
            "resolution_evidence",
        ],
    )
    return result.sort_values(["accession", "library_id"]).reset_index(drop=True), resolution


def read_biostudies_file_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype={"path": str})
    required = {"accession", "section_type", "path", "size", "file_type"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"File manifest is missing columns: {sorted(missing)}")
    frame["size"] = pd.to_numeric(frame["size"], errors="raise").astype(np.int64)
    if frame.duplicated(["accession", "path"]).any():
        raise ValueError("File manifest contains duplicate accession/path rows")
    return frame


def validate_scrna_triplets(
    libraries: pd.DataFrame, files: pd.DataFrame
) -> pd.DataFrame:
    """Verify that every scRNA SDRF library has exactly one 10X file triplet."""
    scrna = libraries[
        libraries["accession"].eq(PRIMARY_ACCESSION)
    ].copy()
    file_names = set(
        files.loc[files["accession"].eq(PRIMARY_ACCESSION), "path"].astype(str)
    )
    rows: list[dict[str, Any]] = []
    for _, library in scrna.iterrows():
        names = json.loads(library["processed_files"])
        matches = {
            "matrix": [name for name in names if name.endswith("-matrix.mtx.gz")],
            "barcodes": [name for name in names if name.endswith("-barcodes.tsv.gz")],
            "features": [name for name in names if name.endswith("-features.tsv.gz")],
        }
        present = {
            role: bool(len(values) == 1 and values[0] in file_names)
            for role, values in matches.items()
        }
        rows.append(
            {
                "accession": PRIMARY_ACCESSION,
                "library_id": library["library_id"],
                "matrix_file": matches["matrix"][0] if len(matches["matrix"]) == 1 else "",
                "barcodes_file": matches["barcodes"][0]
                if len(matches["barcodes"]) == 1
                else "",
                "features_file": matches["features"][0]
                if len(matches["features"]) == 1
                else "",
                "matrix_present_in_manifest": present["matrix"],
                "barcodes_present_in_manifest": present["barcodes"],
                "features_present_in_manifest": present["features"],
                "triplet_status": "complete" if all(present.values()) else "invalid",
            }
        )
    result = pd.DataFrame(rows).sort_values("library_id").reset_index(drop=True)
    return result


def build_source_file_inventory(
    files: pd.DataFrame,
    libraries: pd.DataFrame,
    raw_root: str | Path,
    *,
    hash_present: bool = False,
) -> pd.DataFrame:
    """Add official URLs, local validation, and library ownership to a file list."""
    raw_root = Path(raw_root)
    ownership: dict[tuple[str, str], set[str]] = {}
    biological_ownership: dict[tuple[str, str], set[str]] = {}
    for _, row in libraries.iterrows():
        for name in json.loads(row["processed_files"]):
            key = (str(row["accession"]), str(name))
            ownership.setdefault(key, set()).add(str(row["library_id"]))
            biological_ownership.setdefault(key, set()).add(
                str(row["biological_library_id"])
            )
    rows = []
    for _, row in files.iterrows():
        accession = str(row["accession"])
        name = str(row["path"])
        local = raw_root / accession / name
        exists = local.is_file()
        expected = int(row["size"])
        local_size = local.stat().st_size if exists else 0
        size_ok = bool(exists and local_size == expected)
        rows.append(
            {
                "accession": accession,
                "section_type": row["section_type"],
                "file_name": name,
                "library_id": (
                    next(iter(ownership[(accession, name)]))
                    if len(ownership.get((accession, name), set())) == 1
                    else "shared_assay_components"
                    if ownership.get((accession, name))
                    else "not_library_scoped"
                ),
                "library_ids": stable_json(
                    sorted(ownership.get((accession, name), set()))
                ),
                "biological_library_ids": stable_json(
                    sorted(biological_ownership.get((accession, name), set()))
                ),
                "expected_size_bytes": expected,
                "source_url": f"https://www.ebi.ac.uk/biostudies/files/{accession}/{name}",
                "local_relative_path": (Path("raw") / accession / name).as_posix(),
                "local_exists": exists,
                "local_size_bytes": local_size,
                "size_ok": size_ok,
                "sha256": sha256_file(local) if hash_present and size_ok else "",
                "download_status": "complete" if size_ok else "missing_or_invalid",
            }
        )
    return pd.DataFrame(rows).sort_values(["accession", "file_name"]).reset_index(drop=True)


def build_sampling_frame_audit(libraries: pd.DataFrame) -> pd.DataFrame:
    """Audit exact fetal-liver sort gates before any outcome analysis."""
    primary = libraries[
        libraries["accession"].eq(PRIMARY_ACCESSION)
        & libraries["tissue"].eq("liver")
    ].copy()
    rows: list[dict[str, Any]] = []
    for gate, group in primary.groupby("sort_gate", sort=True):
        donor_gate = group.drop_duplicates(["donor_id", "sort_gate"])
        counts = donor_gate.groupby("condition")["donor_id"].nunique().to_dict()
        n_t21 = int(counts.get("T21", 0))
        n_disomy = int(counts.get("disomy", 0))
        n_total = n_t21 + n_disomy
        label_space = math.comb(n_total, n_disomy) if n_total else 0
        resolution = group.get(
            "sort_gate_resolution_status", pd.Series(index=group.index, dtype=str)
        )
        gate_conflicts_resolved = int(
            resolution.eq("resolved_by_paper_supplement_and_author_pipeline").sum()
        )
        gate_conflicts_unresolved = int(resolution.eq("conflict_unresolved").sum())
        eligible = (
            n_t21 >= 3
            and n_disomy >= 3
            and label_space >= 100
            and gate_conflicts_unresolved == 0
        )
        rows.append(
            {
                "sampling_frame_id": f"{PRIMARY_ACCESSION}:fetal_liver:{gate}",
                "accession": PRIMARY_ACCESSION,
                "tissue": "liver",
                "sort_gate": gate,
                "sort_gate_original_values": stable_json(
                    sorted(group["sort_gate_original"].astype(str).unique())
                ),
                "n_libraries": int(group["library_id"].nunique()),
                "n_t21_donors": n_t21,
                "n_disomy_donors": n_disomy,
                "n_total_donors": n_total,
                "condition_label_space_size": label_space,
                "minimum_exact_p": (1.0 / label_space) if label_space else np.nan,
                "n_cross_source_gate_conflicts_resolved": gate_conflicts_resolved,
                "n_cross_source_gate_conflicts_unresolved": gate_conflicts_unresolved,
                "formal_discovery_eligible": eligible,
                "sampling_frame_locked": False,
                "role_recommendation": "not_eligible",
                "occupancy_claim_ceiling": "within_fixed_gate_relative_occupancy_only",
                "failure_reason": (
                    ""
                    if eligible
                    else "requires_at_least_3_donors_per_condition_and_100_labelings"
                ),
            }
        )
    audit = pd.DataFrame(rows)
    eligible_indices = audit.index[audit["formal_discovery_eligible"]].tolist()
    ordered = sorted(
        eligible_indices,
        key=lambda index: (
            _GATE_PRIORITY.index(audit.loc[index, "sort_gate"])
            if audit.loc[index, "sort_gate"] in _GATE_PRIORITY
            else len(_GATE_PRIORITY),
            -int(audit.loc[index, "condition_label_space_size"]),
        ),
    )
    if ordered:
        audit.loc[ordered[0], "role_recommendation"] = "candidate_primary_pending_lock"
        for index in ordered[1:]:
            audit.loc[index, "role_recommendation"] = "candidate_sensitivity_pending_lock"
    return audit.sort_values("sort_gate").reset_index(drop=True)


def build_donor_design_preflight(libraries: pd.DataFrame) -> pd.DataFrame:
    """Create namespaced donor rows without inventing cross-accession matches."""
    rows: list[dict[str, Any]] = []
    for donor_id, group in libraries.groupby("donor_id", sort=True):
        invariant = {}
        for column in ("accession", "source_donor_id", "condition", "pcw", "sex"):
            values = group[column].drop_duplicates().tolist()
            if len(values) != 1:
                raise ValueError(f"Donor {donor_id!r} has inconsistent {column}: {values}")
            invariant[column] = values[0]
        modalities = sorted(set(group["modality"].astype(str)))
        rows.append(
            {
                "donor_id": donor_id,
                "source_donor_id": invariant["source_donor_id"],
                "source_donor_ids": stable_json([invariant["source_donor_id"]]),
                "source_accessions": stable_json([invariant["accession"]]),
                "source_person_tokens": stable_json(
                    sorted(
                        set(group["source_person_token"].astype(str))
                        - {"not_reported"}
                    )
                ),
                "cross_modal_person_id": (
                    "HDBR:"
                    + sorted(
                        set(group["source_person_token"].astype(str))
                        - {"not_reported"}
                    )[0]
                    if len(
                        set(group["source_person_token"].astype(str))
                        - {"not_reported"}
                    )
                    == 1
                    else "not_resolved"
                ),
                "condition": invariant["condition"],
                "pcw": float(invariant["pcw"]),
                "sex": invariant["sex"],
                "available_tissues": stable_json(sorted(set(group["tissue"].astype(str)))),
                "available_sort_gates": stable_json(
                    sorted(set(group["sort_gate"].astype(str)) - {"not_applicable"})
                ),
                "available_modalities": stable_json(modalities),
                "number_of_cells_by_gate": stable_json({}),
                "number_of_cells_in_primary_lineage": pd.NA,
                "n_scrna_libraries": int(
                    group.loc[group["modality"].eq("scRNA"), "library_id"].nunique()
                ),
                "primary_sampling_frame_id": "not_locked",
                "number_of_cells_in_primary_sampling_frame": pd.NA,
                "primary_trajectory_draw_id": "not_built",
                "n_trajectory_bins_planned": pd.NA,
                "n_trajectory_bins_observed": pd.NA,
                "trajectory_bin_coverage_fraction": np.nan,
                "donor_correspondence_status": "accession_namespaced_unmerged",
                "correspondence_evidence_ids": stable_json([]),
                "independent_replication_eligible": False,
                "unresolved_overlap_reason": "cross_accession_identity_not_directly_confirmed",
                "design_stage": "metadata_preflight_counts_not_staged",
            }
        )
    return pd.DataFrame(rows).sort_values("donor_id").reset_index(drop=True)


def build_donor_correspondence_audit(donor_design: pd.DataFrame) -> pd.DataFrame:
    """Audit same-string donor IDs across accessions without auto-merging them."""
    rows: list[dict[str, Any]] = []
    records = donor_design.to_dict("records")
    for left_index in range(len(records)):
        for right_index in range(left_index + 1, len(records)):
            left, right = records[left_index], records[right_index]
            left_accession = json.loads(left["source_accessions"])[0]
            right_accession = json.loads(right["source_accessions"])[0]
            if left_accession == right_accession:
                continue
            same_public_id = left["source_donor_id"] == right["source_donor_id"]
            left_tokens = set(json.loads(left["source_person_tokens"]))
            right_tokens = set(json.loads(right["source_person_tokens"]))
            shared_tokens = sorted(left_tokens.intersection(right_tokens))
            if not same_public_id and not shared_tokens:
                continue
            condition_match = left["condition"] == right["condition"]
            pcw_match = float(left["pcw"]) == float(right["pcw"])
            if shared_tokens and condition_match and pcw_match:
                status = "confirmed_metadata_same_biological_donor"
                action = "merge_modalities_only_under_cross_modal_person_id"
                evidence = (
                    "SDRF_original_source_name_person_token:HDBR:" + shared_tokens[0]
                )
            elif same_public_id and condition_match and pcw_match:
                status = "metadata_concordant_candidate_match"
                action = "keep_separate_until_direct_donor_crosswalk"
                evidence = "accession_local_donor_label_plus_condition_pcw_only"
            else:
                status = "conflict_unresolved"
                action = "forbid_merge"
                evidence = "accession_local_donor_label_collision"
            rows.append(
                {
                    "source_donor_id": left["source_donor_id"],
                    "left_donor_id": left["donor_id"],
                    "right_donor_id": right["donor_id"],
                    "left_condition": left["condition"],
                    "right_condition": right["condition"],
                    "condition_match": condition_match,
                    "left_pcw": left["pcw"],
                    "right_pcw": right["pcw"],
                    "pcw_match": pcw_match,
                    "shared_source_person_tokens": stable_json(shared_tokens),
                    "correspondence_status": status,
                    "linkage_method": (
                        "exact_accession_donor_label_plus_normalized_original_source_"
                        "person_token_plus_exact_pcw_and_condition"
                        if status.startswith("confirmed_metadata")
                        else "accession_local_label_audit"
                    ),
                    "genetic_identity_verified": False,
                    "analysis_action": action,
                    "direct_evidence_id": evidence,
                }
            )
    columns = [
        "source_donor_id",
        "left_donor_id",
        "right_donor_id",
        "left_condition",
        "right_condition",
        "condition_match",
        "left_pcw",
        "right_pcw",
        "pcw_match",
        "shared_source_person_tokens",
        "correspondence_status",
        "linkage_method",
        "genetic_identity_verified",
        "analysis_action",
        "direct_evidence_id",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["source_donor_id", "left_donor_id", "right_donor_id"]
    ).reset_index(drop=True)


def _open_maybe_gzip(path: Path):
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def _read_tsv_no_header(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None, dtype=str, compression="infer")


def read_10x_triplet(
    matrix_path: str | Path,
    barcodes_path: str | Path,
    features_path: str | Path,
    library_metadata: Mapping[str, Any],
    *,
    selected_barcodes: Iterable[str] | None = None,
) -> ad.AnnData:
    """Read a deposited 10X triplet as a sparse, metadata-complete shard."""
    matrix_path, barcodes_path, features_path = map(
        Path, (matrix_path, barcodes_path, features_path)
    )
    with _open_maybe_gzip(matrix_path) as handle:
        matrix = mmread(handle)
    barcodes = _read_tsv_no_header(barcodes_path).iloc[:, 0].astype(str)
    features = _read_tsv_no_header(features_path)
    raw_matrix = sparse.csc_matrix(matrix)
    if raw_matrix.shape != (len(features), len(barcodes)):
        raise ValueError(
            f"10X dimensions disagree: matrix={raw_matrix.shape}, "
            f"barcodes={len(barcodes)}, features={len(features)}"
        )
    if selected_barcodes is None:
        if len(barcodes) > 1_000_000:
            raise ValueError(
                "Raw barcode space exceeds one million columns; run the "
                "pre-registered per-library cell-calling stage first"
            )
        selected = np.ones(len(barcodes), dtype=bool)
    else:
        selected_values = {str(value) for value in selected_barcodes}
        if not selected_values:
            raise ValueError("Cell-calling selection contains no barcodes")
        selected = barcodes.isin(selected_values).to_numpy()
        missing = selected_values.difference(set(barcodes[selected]))
        if missing:
            raise ValueError(
                f"Cell-calling selection contains {len(missing)} unknown barcodes"
            )
    barcodes = barcodes.loc[selected].reset_index(drop=True)
    counts = raw_matrix[:, selected].transpose().tocsr()
    if counts.data.size:
        if np.any(~np.isfinite(counts.data)) or np.any(counts.data < 0):
            raise ValueError("Counts contain negative or non-finite values")
        if not np.array_equal(counts.data, np.rint(counts.data)):
            raise ValueError("Counts are not integer-valued")
    max_count = float(counts.data.max()) if counts.data.size else 0.0
    counts = counts.astype(np.int32 if max_count <= np.iinfo(np.int32).max else np.int64)
    gene_ids = features.iloc[:, 0].astype(str)
    if gene_ids.duplicated().any():
        duplicated = gene_ids[gene_ids.duplicated()].head().tolist()
        raise ValueError(f"Feature IDs are not unique; refusing make_unique(): {duplicated}")
    symbols = features.iloc[:, 1].astype(str) if features.shape[1] >= 2 else gene_ids
    feature_type = (
        features.iloc[:, 2].astype(str)
        if features.shape[1] >= 3
        else pd.Series("Gene Expression", index=features.index)
    )
    accession = str(library_metadata["accession"])
    library_id = str(library_metadata["library_id"])
    ids = [make_cell_id(accession, library_id, barcode) for barcode in barcodes]
    obs = pd.DataFrame(index=pd.Index(ids, name="cell_id"))
    obs["cell_id"] = ids
    obs["donor_id"] = str(library_metadata["donor_id"])
    obs["source_donor_id"] = str(library_metadata["source_donor_id"])
    obs["condition"] = str(library_metadata["condition"])
    obs["condition_original"] = str(library_metadata["condition_original"])
    obs["pcw"] = float(library_metadata["pcw"])
    obs["sex"] = str(library_metadata.get("sex", "not_reported"))
    obs["tissue"] = str(library_metadata["tissue"])
    obs["sort_gate"] = str(library_metadata["sort_gate"])
    obs["sort_gate_original"] = str(library_metadata["sort_gate_original"])
    obs["library_id"] = library_id
    obs["technical_batch"] = str(library_metadata.get("technical_batch", "not_resolved"))
    obs["original_cell_type"] = "not_available_publicly"
    obs["original_cell_type_source"] = "public_count_matrix_has_no_cell_labels"
    obs["analysis_cell_type"] = "not_assigned"
    obs["lineage_inclusion"] = False
    obs["lineage_inclusion_reason"] = "annotation_plan_not_applied"
    obs["original_barcode"] = barcodes.to_numpy()
    var = pd.DataFrame(index=pd.Index(gene_ids, name="gene_id"))
    var["gene_id_original"] = gene_ids.to_numpy()
    var["gene_symbol"] = symbols.to_numpy()
    var["feature_type"] = feature_type.to_numpy()
    var["chromosome"] = "not_mapped"
    var["is_chr21"] = "not_mapped"
    var["genome_build"] = "GRCh38_deposit_annotation_unverified"
    var["gene_mapping_status"] = "not_mapped"
    # Count shards preserve their historical raw-X semantics.  The immutable
    # formal expression transform is applied only during annotated assembly,
    # after the hash-pinned count-shard evidence has been validated.
    result = ad.AnnData(X=counts.astype(np.float32), obs=obs, var=var)
    result.layers["counts"] = counts
    result.uns["t21_data_product"] = {
        "schema_version": PRODUCT_VERSION,
        "stage": "counts_shard_unannotated",
        "accession": accession,
        "library_id": library_id,
    }
    return result


def _validate_sparse_integer_counts(counts: Any) -> None:
    if not sparse.issparse(counts):
        raise ValueError('layers["counts"] must be sparse')
    if not np.issubdtype(counts.dtype, np.integer):
        raise ValueError('layers["counts"] must use an integer dtype')
    values = counts.data
    if values.size:
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Counts must be finite and non-negative")
        if not np.array_equal(values, np.rint(values)):
            raise ValueError("Counts must be exactly integer-valued")


def _validate_formal_expression_contract(
    adata: ad.AnnData, counts: Any
) -> tuple[Mapping[str, Any], str]:
    validation = validate_t21_formal_expression(counts, adata.X)
    product_metadata = adata.uns.get("t21_data_product")
    expression_metadata = (
        product_metadata.get("expression_contract")
        if isinstance(product_metadata, Mapping)
        else None
    )
    if not isinstance(expression_metadata, Mapping):
        raise ValueError("H5AD contract lacks its formal expression contract")
    expected_metadata = expression_contract_metadata(adata.X, counts=counts)
    if stable_json(dict(expression_metadata)) != stable_json(expected_metadata):
        raise ValueError("H5AD formal expression metadata differs from fresh validation")
    return expression_metadata, str(validation["expression_csr_semantic_sha256"])


def validate_scrna_contract(
    adata: ad.AnnData,
    *,
    strict_analysis_labels: bool = True,
    require_formal_expression: bool | None = None,
) -> dict[str, Any]:
    missing_obs = sorted(set(REQUIRED_SCRNA_OBS).difference(adata.obs.columns))
    missing_var = sorted(set(REQUIRED_SCRNA_VAR).difference(adata.var.columns))
    if missing_obs or missing_var:
        raise ValueError(f"H5AD contract missing obs={missing_obs}, var={missing_var}")
    if "counts" not in adata.layers:
        raise ValueError('H5AD contract requires layers["counts"]')
    counts = adata.layers["counts"]
    counts_is_backed = hasattr(counts, "to_memory") and not sparse.issparse(counts)
    if counts_is_backed:
        counts_dtype = np.dtype(counts.dtype)
        if counts_dtype == np.dtype(bool) or not np.issubdtype(
            counts_dtype, np.integer
        ):
            raise ValueError('Backed layers["counts"] must use an integer dtype')
    else:
        _validate_sparse_integer_counts(counts)
    if counts.shape != adata.shape:
        raise ValueError("Counts and X must have the same shape")
    if require_formal_expression is None:
        require_formal_expression = bool(strict_analysis_labels)
    expression_metadata: Mapping[str, Any] | None = None
    x_semantic_sha256: str | None = None
    if require_formal_expression:
        if counts_is_backed:
            raise ValueError(
                "Backed formal expression requires explicit chunk validation"
            )
        expression_metadata, x_semantic_sha256 = _validate_formal_expression_contract(
            adata, counts
        )
    if not adata.obs_names.is_unique or not adata.obs["cell_id"].is_unique:
        raise ValueError("Cell IDs must be globally unique")
    if not np.array_equal(
        adata.obs_names.astype(str).to_numpy(), adata.obs["cell_id"].astype(str).to_numpy()
    ):
        raise ValueError("obs_names must exactly equal obs['cell_id']")
    if not adata.var_names.is_unique:
        raise ValueError("Feature IDs must be unique")
    for column in REQUIRED_SCRNA_OBS:
        values = adata.obs[column]
        if values.isna().any():
            raise ValueError(f"Required obs field {column!r} contains missing values")
        if not pd.api.types.is_bool_dtype(values.dtype):
            if values.astype(str).str.strip().eq("").any():
                raise ValueError(f"Required obs field {column!r} contains empty values")
    for column in REQUIRED_SCRNA_VAR:
        values = adata.var[column]
        if values.isna().any():
            raise ValueError(f"Required var field {column!r} contains missing values")
        if not pd.api.types.is_bool_dtype(values.dtype):
            if values.astype(str).str.strip().eq("").any():
                raise ValueError(f"Required var field {column!r} contains empty values")
    expected_cell_ids = np.asarray(
        [
            make_cell_id(PRIMARY_ACCESSION, library_id, barcode)
            for library_id, barcode in zip(
                adata.obs["library_id"].astype(str),
                adata.obs["original_barcode"].astype(str),
            )
        ]
    )
    if not np.array_equal(adata.obs["cell_id"].astype(str).to_numpy(), expected_cell_ids):
        raise ValueError(
            "cell_id must equal E-MTAB-13067|library_id|original_barcode"
        )
    expected_donors = (
        PRIMARY_ACCESSION + ":" + adata.obs["source_donor_id"].astype(str)
    ).to_numpy()
    if not np.array_equal(adata.obs["donor_id"].astype(str).to_numpy(), expected_donors):
        raise ValueError("donor_id must remain accession-namespaced in the scRNA product")
    conditions = set(adata.obs["condition"].astype(str))
    if not conditions.issubset({"T21", "disomy"}):
        raise ValueError(f"Unexpected canonical conditions: {sorted(conditions)}")
    if not pd.api.types.is_bool_dtype(adata.obs["lineage_inclusion"].dtype):
        raise ValueError("lineage_inclusion must use a boolean dtype")
    pcw = pd.to_numeric(adata.obs["pcw"], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(pcw)) or np.any(pcw <= 0):
        raise ValueError("pcw must be finite and positive")
    library_invariants = (
        "donor_id",
        "source_donor_id",
        "condition",
        "condition_original",
        "pcw",
        "sex",
        "tissue",
        "sort_gate",
        "sort_gate_original",
        "technical_batch",
    )
    for library_id, group in adata.obs.groupby("library_id", observed=True, sort=False):
        varying = [column for column in library_invariants if group[column].nunique() != 1]
        if varying:
            raise ValueError(
                f"Library {library_id!r} has non-invariant metadata fields: {varying}"
            )
    if strict_analysis_labels:
        analysis_labels = (
            adata.obs["analysis_cell_type"].astype(str).str.strip().str.lower()
        )
        if analysis_labels.isin(
            {"", "not_assigned", "unknown", "not_available", "not_available_publicly"}
        ).any():
            raise ValueError("Analysis cell types have not been assigned")
        technical_batches = (
            adata.obs["technical_batch"].astype(str).str.strip().str.lower()
        )
        if technical_batches.isin(
            {"", "unknown", "not_resolved", "not_available", "not_reported"}
        ).any():
            raise ValueError("Technical batch handling has not been resolved")
        if not adata.obs["lineage_inclusion"].astype(bool).any():
            raise ValueError("No cells pass the pre-registered lineage rule")
        chromosome = adata.var["chromosome"].astype(str).str.strip()
        mapping_status = adata.var["gene_mapping_status"].astype(str).str.strip()
        genome_build = adata.var["genome_build"].astype(str).str.strip()
        if chromosome.str.lower().isin(["", "not_mapped"]).any():
            raise ValueError("Formal H5AD requires resolved chromosome annotations")
        if mapping_status.str.lower().isin(["", "not_mapped"]).any():
            raise ValueError("Formal H5AD requires an explicit gene mapping status")
        if genome_build.str.lower().str.contains("unverified|not_mapped", regex=True).any():
            raise ValueError("Formal H5AD requires a verified reference genome build")
        if not pd.api.types.is_bool_dtype(adata.var["is_chr21"].dtype):
            raise ValueError("Formal H5AD var['is_chr21'] must use a boolean dtype")
        chromosome_token = chromosome.str.lower().str.removeprefix("chr")
        expected_chr21 = chromosome_token.eq("21").to_numpy()
        if not np.array_equal(adata.var["is_chr21"].to_numpy(dtype=bool), expected_chr21):
            raise ValueError("var['is_chr21'] disagrees with var['chromosome']")
    return {
        "n_cells": int(adata.n_obs),
        "n_features": int(adata.n_vars),
        "cell_id_set_hash": cell_id_set_hash(adata.obs_names.astype(str)),
        "cell_id_order_hash": ordered_id_hash(adata.obs_names.astype(str)),
        "gene_order_hash": ordered_id_hash(adata.var_names.astype(str)),
        "strict_analysis_labels": strict_analysis_labels,
        "formal_expression_validated": bool(require_formal_expression),
        "expression_contract_version": FORMAL_EXPRESSION_CONTRACT_VERSION,
        "expression_target_sum": FORMAL_EXPRESSION_TARGET_SUM,
        "expression_contract_sha256": (
            formal_expression_preprocessing_contract_sha256()
        ),
        "expression_implementation_sha256": (
            formal_expression_preprocessing_source_sha256()
        ),
        "x_semantic_sha256": x_semantic_sha256,
    }


def validate_donor_design(
    frame: pd.DataFrame, *, scrna_obs: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Validate the one-row-per-donor release design and optional scRNA counts."""
    missing = sorted(set(REQUIRED_DONOR_DESIGN_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Donor design is missing columns: {missing}")
    donor_ids = frame["donor_id"].astype(str).str.strip()
    if donor_ids.eq("").any() or donor_ids.duplicated().any():
        raise ValueError("Donor design donor_id values must be non-empty and unique")
    if not set(frame["condition"].astype(str)).issubset({"T21", "disomy"}):
        raise ValueError("Donor design conditions must use canonical T21/disomy labels")
    pcw = pd.to_numeric(frame["pcw"], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(pcw)) or np.any(pcw <= 0):
        raise ValueError("Donor design pcw values must be finite and positive")
    for column in (
        "sex",
        "donor_correspondence_status",
    ):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Donor design field {column!r} must be non-empty")

    decoded: dict[str, list[Any]] = {}
    for column, expected_type in (
        ("available_tissues", list),
        ("available_sort_gates", list),
        ("available_modalities", list),
        ("number_of_cells_by_gate", dict),
    ):
        values = []
        for value in frame[column]:
            try:
                parsed = json.loads(str(value))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Donor design {column!r} must contain JSON") from exc
            if not isinstance(parsed, expected_type):
                raise ValueError(
                    f"Donor design {column!r} must contain {expected_type.__name__} values"
                )
            values.append(parsed)
        decoded[column] = values
    for index, counts in enumerate(decoded["number_of_cells_by_gate"]):
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in counts.values()
        ):
            raise ValueError(f"Donor design row {index} has invalid gate cell counts")
    primary_counts = pd.to_numeric(
        frame["number_of_cells_in_primary_lineage"], errors="coerce"
    ).to_numpy(dtype=float)
    if (
        np.any(~np.isfinite(primary_counts))
        or np.any(primary_counts < 0)
        or not np.array_equal(primary_counts, np.rint(primary_counts))
    ):
        raise ValueError("Primary-lineage donor cell counts must be non-negative integers")
    coverage = pd.to_numeric(
        frame["trajectory_bin_coverage_fraction"], errors="coerce"
    ).to_numpy(dtype=float)
    if np.any(~np.isfinite(coverage)) or np.any((coverage < 0) | (coverage > 1)):
        raise ValueError("Trajectory-bin coverage must be finite and within [0, 1]")

    if scrna_obs is not None:
        if "donor_id" not in scrna_obs or "sort_gate" not in scrna_obs:
            raise ValueError("scRNA obs lacks donor_id/sort_gate for donor-design validation")
        observed_donors = set(scrna_obs["donor_id"].astype(str))
        design_donors = set(donor_ids)
        if not observed_donors.issubset(design_donors):
            raise ValueError("scRNA donor IDs are missing from the donor design")
        design_index = frame.assign(_donor_id=donor_ids).set_index("_donor_id")
        count_maps = dict(zip(donor_ids, decoded["number_of_cells_by_gate"]))
        for donor_id, obs_group in scrna_obs.groupby("donor_id", observed=True):
            observed_by_gate = {
                str(gate): int(count)
                for gate, count in obs_group["sort_gate"].value_counts().items()
            }
            declared = {
                str(gate): int(count)
                for gate, count in count_maps[str(donor_id)].items()
                if int(count) > 0
            }
            if observed_by_gate != declared:
                raise ValueError(f"Gate cell counts disagree for donor {donor_id!r}")
            if "lineage_inclusion" in scrna_obs:
                observed_primary = int(obs_group["lineage_inclusion"].astype(bool).sum())
                declared_primary = int(
                    design_index.loc[str(donor_id), "number_of_cells_in_primary_lineage"]
                )
                if observed_primary != declared_primary:
                    raise ValueError(
                        f"Primary-lineage cell count disagrees for donor {donor_id!r}"
                    )
    return {
        "n_donors": int(len(frame)),
        "donor_set_hash": cell_id_set_hash(donor_ids),
        "n_t21": int(frame["condition"].astype(str).eq("T21").sum()),
        "n_disomy": int(frame["condition"].astype(str).eq("disomy").sum()),
    }


def _json_string_array(value: Any, label: str) -> list[str]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must contain a JSON array") from exc
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise ValueError(f"{label} must contain a JSON string array")
    if any(not item.strip() for item in parsed) or len(parsed) != len(set(parsed)):
        raise ValueError(f"{label} contains empty or duplicate values")
    return parsed


def _strict_boolean_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = frame[column]
    if pd.api.types.is_bool_dtype(values.dtype):
        if values.isna().any():
            raise ValueError(f"{column} may not contain missing values")
        return values.to_numpy(dtype=bool)
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError(f"{column} must contain only true/false")
    return normalized.eq("true").to_numpy(dtype=bool)


def validate_external_2021_evidence(
    overlap: pd.DataFrame,
    sources: pd.DataFrame,
    constraints: pd.DataFrame,
    *,
    repository_root: str | Path | None = None,
    current_donor_design: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Validate the 2021 external-design audit without asserting independence.

    A successful integrity check intentionally retains an unresolved independence
    ceiling.  It proves that the ambiguity, pooled-cell exclusion, and 4-vs-2
    permutation limits are represented faithfully; it does not prove that the
    2021 donors are genetically distinct from E-MTAB-13067 donors.
    """
    missing_overlap = sorted(
        set(REQUIRED_EXTERNAL_2021_OVERLAP_COLUMNS).difference(overlap.columns)
    )
    missing_sources = sorted(
        set(REQUIRED_EXTERNAL_2021_SOURCE_COLUMNS).difference(sources.columns)
    )
    missing_constraints = sorted(
        set(REQUIRED_EXTERNAL_2021_CONSTRAINT_COLUMNS).difference(
            constraints.columns
        )
    )
    if missing_overlap or missing_sources or missing_constraints:
        raise ValueError(
            "External 2021 evidence contract is incomplete: "
            f"overlap={missing_overlap}, sources={missing_sources}, "
            f"constraints={missing_constraints}"
        )

    donor_ids = overlap["donor_id_2021"].astype(str).str.strip()
    source_ids = sources["source_id"].astype(str).str.strip()
    constraint_ids = constraints["constraint_id"].astype(str).str.strip()
    for label, values in (
        ("donor_id_2021", donor_ids),
        ("source_id", source_ids),
        ("constraint_id", constraint_ids),
    ):
        if values.eq("").any() or values.duplicated().any():
            raise ValueError(f"External 2021 {label} values must be non-empty and unique")

    array_columns = (
        "member_donor_ids",
        "direct_accessions",
        "pooled_membership_accessions",
        "source_library_ids",
        "candidate_2024_scrna_donor_ids",
        "candidate_2024_person_tokens",
        "evidence_source_ids",
    )
    overlap_arrays = {
        column: [
            _json_string_array(value, f"overlap.{column}")
            for value in overlap[column]
        ]
        for column in array_columns
    }
    constraint_evidence = [
        _json_string_array(value, "constraints.evidence_source_ids")
        for value in constraints["evidence_source_ids"]
    ]
    known_sources = set(source_ids)
    referenced_sources = {
        source
        for values in overlap_arrays["evidence_source_ids"] + constraint_evidence
        for source in values
    }
    unknown_sources = sorted(referenced_sources.difference(known_sources))
    if unknown_sources:
        raise ValueError(f"External 2021 evidence references unknown sources: {unknown_sources}")

    boolean_columns = (
        "public_crosswalk_available",
        "genetic_identity_verified",
        "analysis_unit_eligible",
        "independent_replication_eligible",
    )
    booleans = {
        column: _strict_boolean_values(overlap, column) for column in boolean_columns
    }
    biological = overlap["processing_unit_type"].astype(str).eq("biological_donor")
    pooled = overlap["processing_unit_type"].astype(str).eq(
        "pooled_rerun_of_D1_D2_D3"
    )
    expected_biological = {"D1", "D2", "D3", "D4", "F38", "F45"}
    if set(donor_ids[biological]) != expected_biological or int(pooled.sum()) != 1:
        raise ValueError("External 2021 audit must contain six donors and one DSOXPool row")
    pool_indices = np.flatnonzero(pooled.to_numpy())
    pool_index = int(pool_indices[0])
    if donor_ids.iloc[pool_index] != "DSOXPool":
        raise ValueError("The pooled processing row must be named DSOXPool")
    if set(overlap_arrays["member_donor_ids"][pool_index]) != {"D1", "D2", "D3"}:
        raise ValueError("DSOXPool must identify D1/D2/D3 as unresolved members")

    biological_frame = overlap.loc[biological]
    condition_counts = biological_frame["condition"].astype(str).value_counts().to_dict()
    if condition_counts != {"T21": 4, "disomy": 2}:
        raise ValueError(f"Expected external 2021 4-vs-2 design, found {condition_counts}")
    biological_indices = np.flatnonzero(biological.to_numpy())
    if (
        booleans["public_crosswalk_available"][biological_indices].any()
        or booleans["genetic_identity_verified"][biological_indices].any()
        or not booleans["analysis_unit_eligible"][biological_indices].all()
        or booleans["independent_replication_eligible"][biological_indices].any()
    ):
        raise ValueError("External biological donors violate the unresolved-independence ceiling")
    if not biological_frame["overlap_status"].astype(str).eq(
        "unresolved_public_crosswalk_missing"
    ).all():
        raise ValueError("All external biological donors must retain unresolved overlap status")
    if not biological_frame["analysis_action"].astype(str).eq(
        "cross_study_cross_processing_validation_only"
    ).all():
        raise ValueError("External biological donors exceed their allowed analysis action")
    if (
        booleans["analysis_unit_eligible"][pool_index]
        or booleans["independent_replication_eligible"][pool_index]
        or overlap.iloc[pool_index]["overlap_status"]
        != "unresolved_pool_demultiplex_crosswalk"
    ):
        raise ValueError("DSOXPool must remain excluded pending genotype demultiplexing")

    post_qc = pd.to_numeric(overlap["post_qc_cells"], errors="coerce").to_numpy()
    if (
        np.any(~np.isfinite(post_qc))
        or np.any(post_qc <= 0)
        or not np.array_equal(post_qc, np.rint(post_qc))
    ):
        raise ValueError("External post-QC cell counts must be positive integers")
    t21_direct = int(post_qc[biological.to_numpy() & overlap["condition"].eq("T21")].sum())
    if t21_direct != 13949 or int(post_qc[pool_index]) != 2794:
        raise ValueError("External 2021 T21 direct/pool cell-count scopes changed")

    if current_donor_design is not None:
        if "donor_id" not in current_donor_design:
            raise ValueError("Current donor design lacks donor_id for candidate validation")
        design = current_donor_design.copy()
        design["donor_id"] = design["donor_id"].astype(str)
        design_index = design.set_index("donor_id", drop=False)
        for row_index, row in overlap.iterrows():
            candidates = overlap_arrays["candidate_2024_scrna_donor_ids"][row_index]
            candidate_tokens = set(
                overlap_arrays["candidate_2024_person_tokens"][row_index]
            )
            if not candidates:
                if candidate_tokens:
                    raise ValueError("Candidate person tokens exist without candidate donors")
                continue
            missing_candidates = sorted(set(candidates).difference(design_index.index))
            if missing_candidates:
                raise ValueError(
                    f"External audit candidates are absent from donor design: {missing_candidates}"
                )
            observed_tokens: set[str] = set()
            for candidate in candidates:
                candidate_row = design_index.loc[candidate]
                if str(candidate_row["condition"]) != str(row["condition"]):
                    raise ValueError("External candidate condition does not match")
                if str(candidate_row["sex"]).lower() != str(row["sex"]).lower():
                    raise ValueError("External candidate sex does not match")
                candidate_pcw = float(candidate_row["pcw"])
                source_pcw = float(row["pcw_numeric"])
                rule = str(row["candidate_match_rule_id"])
                if "rounded_reported_pcw" in rule:
                    age_matches = round(candidate_pcw) == round(source_pcw)
                else:
                    age_matches = math.isclose(candidate_pcw, source_pcw, abs_tol=1e-9)
                if not age_matches:
                    raise ValueError("External candidate PCW does not match its declared rule")
                if "source_person_tokens" in design_index:
                    observed_tokens.update(
                        _json_string_array(
                            candidate_row["source_person_tokens"],
                            "donor_design.source_person_tokens",
                        )
                    )
            if candidate_tokens and candidate_tokens != observed_tokens:
                raise ValueError("External candidate person tokens disagree with donor design")

    required_source_fields = (
        "role",
        "source_type",
        "accession",
        "url",
        "original_file_name",
        "retrieval_status",
        "formal_use_status",
        "formal_use",
        "validation_status",
    )
    for column in required_source_fields:
        if sources[column].isna().any() or sources[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"External source field {column!r} must be non-empty")
    allowed_retrieval = {
        "reference_url_only_not_retrieved",
        "not_retrieved_into_product",
        "retrieved_local_file",
        "cloned_pinned_commit",
    }
    observed_retrieval = set(sources["retrieval_status"].astype(str))
    if not observed_retrieval.issubset(allowed_retrieval):
        raise ValueError(f"Unknown external retrieval statuses: {observed_retrieval}")
    root = Path(repository_root).resolve() if repository_root is not None else None
    for _, source in sources.iterrows():
        status = str(source["retrieval_status"])
        relative_text = str(source["repository_relative_path"]).strip()
        if status in {"retrieved_local_file", "cloned_pinned_commit"}:
            if root is None or not relative_text:
                raise ValueError("Retrieved external evidence requires repository_root/path")
            relative_path = Path(relative_text)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError("External evidence path must be repository-relative")
            local_path = (root / relative_path).resolve()
            try:
                local_path.relative_to(root)
            except ValueError as exc:
                raise ValueError("External evidence path escapes repository root") from exc
            if status == "retrieved_local_file":
                if not local_path.is_file():
                    raise ValueError(f"Retrieved external file is missing: {relative_text}")
                declared_bytes = int(str(source["bytes"]))
                digest = str(source["sha256"]).lower()
                if declared_bytes != local_path.stat().st_size or not re.fullmatch(
                    r"[0-9a-f]{64}", digest
                ):
                    raise ValueError("External file bytes/SHA256 declaration is invalid")
                if sha256_file(local_path) != digest:
                    raise ValueError(f"External file digest changed: {relative_text}")
                if not str(source["retrieved_at_utc"]).strip():
                    raise ValueError("Retrieved external file lacks retrieval timestamp")
            else:
                commit = str(source["version_or_commit"]).lower()
                if not local_path.is_dir() or not re.fullmatch(r"[0-9a-f]{40}", commit):
                    raise ValueError("Pinned external repository path/commit is invalid")
                try:
                    observed_commit = subprocess.run(
                        ["git", "-C", str(local_path), "rev-parse", "HEAD"],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    ).stdout.strip().lower()
                    dirty = subprocess.run(
                        [
                            "git",
                            "-C",
                            str(local_path),
                            "status",
                            "--porcelain=v1",
                            "--untracked-files=all",
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    ).stdout.strip()
                except (OSError, subprocess.SubprocessError) as exc:
                    raise ValueError(
                        "Pinned external repository cannot be verified with git"
                    ) from exc
                if observed_commit != commit or dirty:
                    raise ValueError(
                        "Pinned external repository commit/cleanliness changed"
                    )
        else:
            if relative_text or str(source["sha256"]).strip():
                raise ValueError("Unretrieved external evidence may not declare a local path/SHA")
            remote_bytes = str(source["bytes"]).strip()
            if remote_bytes:
                if (
                    int(remote_bytes) <= 0
                    or str(source["bytes_status"]) != "remote_content_length_verified"
                ):
                    raise ValueError("Unretrieved source bytes require a verified remote size")
            if str(source["sha256_status"]) not in {
                "not_computed_not_retrieved",
                "not_applicable_reference_page",
                "not_applicable_git_commit_pinned_but_repository_not_retained",
            }:
                raise ValueError("Unretrieved source SHA status is not fail-closed")

    indexed_sources = sources.assign(_source_id=source_ids).set_index("_source_id")
    t21_counts = indexed_sources.loc["emt9389_t21_raw_counts"]
    healthy_counts = indexed_sources.loc["emt9389_healthy_raw_counts"]
    if "exclude_all_DSOXPool_cells" not in str(t21_counts["pool_handling"]):
        raise ValueError("External T21 counts do not enforce DSOXPool exclusion")
    healthy_scope = str(healthy_counts["donor_scope"])
    if "effect_validation_F38_F45_only" not in healthy_scope or "all_9_healthy_donors" not in healthy_scope:
        raise ValueError("External healthy counts conflate effect and reference scopes")
    t21_portal = indexed_sources.loc["cellatlas_t21_h5ad"]
    if "never_count_pseudobulk" not in str(t21_portal["formal_use"]).lower():
        raise ValueError("Transformed T21 portal H5AD X is not prohibited for pseudobulk")
    healthy_portal = indexed_sources.loc["cellatlas_healthy_h5ad"]
    if (
        "metadata_and_labels_only" not in str(healthy_portal["formal_use"]).lower()
        or "x_is_transformed" not in str(healthy_portal["value_semantics"]).lower()
        or "metadata_only_blocked" not in str(healthy_portal["formal_use_status"]).lower()
    ):
        raise ValueError("Healthy portal H5AD is not safely restricted to metadata")

    constraint_index = constraints.assign(_constraint_id=constraint_ids).set_index(
        "_constraint_id"
    )
    expected_constraints = {
        "biological_donor_count_T21": ("integer", "4"),
        "biological_donor_count_disomy": ("integer", "2"),
        "whole_donor_permutation_unit": ("string", "whole_donor"),
        "condition_label_assignment_space": ("integer", "15"),
        "minimum_unrandomized_exact_p": ("decimal", "0.0666666666666667"),
        "age_matched_effect_control_ids": ("json_array", '["F38","F45"]'),
        "dsoxpool_analysis_eligible": ("boolean", "false"),
        "healthy_reference_donor_count": ("integer", "9"),
        "published_condition_graphs_shared": ("boolean", "false"),
        "public_ready_per_cell_pseudotime_available": ("boolean", "false"),
        "absolute_onset_coordinate_comparable": ("boolean", "false"),
        "external_independent_replication_eligible": ("boolean", "false"),
    }
    missing_expected = sorted(set(expected_constraints).difference(constraint_index.index))
    if missing_expected:
        raise ValueError(f"External design constraints are missing: {missing_expected}")
    constraint_status = constraints["constraint_status"].astype(str)
    expected_unresolved = constraint_ids.eq(
        "external_independent_replication_eligible"
    )
    if not constraint_status[~expected_unresolved].eq("locked").all() or not constraint_status[
        expected_unresolved
    ].eq("unresolved").all():
        raise ValueError(
            "External design constraints must be locked except the explicit "
            "independence ceiling"
        )
    for constraint_id, (value_type, expected_value) in expected_constraints.items():
        row = constraint_index.loc[constraint_id]
        observed_value = str(row["required_value"])
        if str(row["value_type"]) != value_type:
            raise ValueError(f"Constraint {constraint_id} changed value type")
        if constraint_id == "minimum_unrandomized_exact_p":
            matches = math.isclose(float(observed_value), 1.0 / 15.0, abs_tol=1e-15)
        elif value_type == "json_array":
            matches = _json_string_array(observed_value, constraint_id) == ["F38", "F45"]
        else:
            matches = observed_value == expected_value
        if not matches:
            raise ValueError(f"Constraint {constraint_id} changed its locked value")
    restricted = constraint_index.loc["restricted_permutation_assignment_space"]
    if "less_than_or_equal_to_15" not in str(restricted["required_value"]):
        raise ValueError("Restricted external permutation space is not bounded by 15")

    pending_prerequisite = sources["formal_use_status"].astype(str).str.contains(
        "blocked_until|pending_release|pending_repository", regex=True
    )
    retrieval_evidence_complete = sources["retrieval_status"].astype(str).isin(
        {"retrieved_local_file", "cloned_pinned_commit"}
    )
    # ``formal_use_status`` is a frozen design-semantic field, so the retrieval
    # utility deliberately does not rewrite legacy ``pending_*`` wording.  Once
    # this validator has checked the local bytes/SHA256 or pinned-repository
    # declaration above, retrieval status is the authoritative acquisition
    # state.  Formal raw-count/H5AD inputs that remain absent still stay pending.
    pending = pending_prerequisite & ~retrieval_evidence_complete
    return {
        "audit_integrity_status": "pass",
        "independence_status": "unresolved_public_crosswalk_missing",
        "independent_replication_eligible": False,
        "allowed_analysis_action": "cross_study_cross_processing_validation_only",
        "n_biological_donors": int(biological.sum()),
        "n_t21": 4,
        "n_disomy": 2,
        "n_pooled_processing_units_excluded": 1,
        "condition_label_assignment_space": 15,
        "minimum_unrandomized_exact_p": 1.0 / 15.0,
        "n_source_records": int(len(sources)),
        "n_pending_formal_sources": int(pending.sum()),
        "external_source_provenance_complete": not bool(pending.any()),
    }


def _finite_metric(mapping: Mapping[str, Any], key: str) -> float:
    if key not in mapping:
        raise ValueError(f"Calibration report is missing metric {key!r}")
    value = float(mapping[key])
    if not np.isfinite(value):
        raise ValueError(f"Calibration metric {key!r} must be finite")
    return value


def _probability_metric(mapping: Mapping[str, Any], key: str) -> float:
    value = _finite_metric(mapping, key)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Calibration probability {key!r} must be within [0, 1]")
    return value


def validate_pre_unblinding_calibration(
    report: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    expected_bindings: Mapping[str, Any],
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Recompute the frozen six-scenario unblinding decision from report metrics."""
    if policy.get("schema_name") != "t21_pre_unblinding_calibration_acceptance_policy":
        raise ValueError("Unexpected calibration acceptance policy schema")
    if not bool(policy.get("outcome_blinded_at_freeze")) or bool(
        policy.get("real_pathway_results_may_be_read")
    ):
        raise ValueError("Calibration policy was not frozen behind a blind")
    policy_version = str(policy.get("schema_version", ""))
    profile_policy = policy_version.startswith("2.")
    design_binding = policy.get("design_binding")
    if not isinstance(design_binding, Mapping):
        raise ValueError("Calibration policy design_binding must be an object")
    analysis_relative = str(design_binding.get("analysis_plan", ""))
    analysis_sha256 = str(design_binding.get("analysis_plan_sha256", ""))
    runner_relative = str(design_binding.get("calibration_runner_spec", ""))
    runner_sha256 = str(design_binding.get("calibration_runner_spec_sha256", ""))
    report_schema_relative = str(
        design_binding.get("calibration_report_schema", "")
    )
    report_schema_sha256 = str(
        design_binding.get("calibration_report_schema_sha256", "")
    )
    common_binding_valid = (
        bool(re.fullmatch(r"[0-9a-f]{64}", runner_sha256))
        and design_binding.get("pathway_universe_must_be_hashed") is True
        and design_binding.get("pathway_universe_logical_hash_required") is True
    )
    if profile_policy:
        profile_binding_valid = (
            analysis_relative == "config/t21_data_product_v1.yaml"
            and runner_relative == "config/t21_preunblinding_calibration_runner_v2.yaml"
            and report_schema_relative
            == "schemas/t21_calibration_report_v2.schema.json"
            and bool(re.fullmatch(r"[0-9a-f]{64}", analysis_sha256))
            and bool(re.fullmatch(r"[0-9a-f]{64}", report_schema_sha256))
            and design_binding.get("required_fixed_common_grid_bins") == 20
            and design_binding.get("outcome_blind_design_profile_required") is True
            and design_binding.get("design_profile_file_and_payload_hash_required") is True
            and design_binding.get("scalar_only_path_free_bindings_required") is True
        )
        if not common_binding_valid or not profile_binding_valid:
            raise ValueError("Calibration v2 policy does not bind the blind design profile")
        if policy.get("publication_execution_contract") != {
            "phase": "final",
            "seed": 20260713,
            "chunk_size": 32,
            "development_override": False,
            "publication_minima_satisfied": True,
            "complete_null_replicates": 10000,
            "scenario_replicates": 2000,
            "power_replicates_per_point": 1000,
        }:
            raise ValueError("Calibration v2 publication execution contract changed")
    elif not (
        common_binding_valid
        and design_binding.get("canonical_pathway_universe")
        == "reference/t21_pathway_universe_v1.tsv"
        and runner_relative == "config/t21_preunblinding_calibration_runner_v1.yaml"
    ):
        raise ValueError("Calibration policy does not bind the canonical pathway universe")
    if report.get("schema_name") != "t21_pre_unblinding_calibration_report":
        raise ValueError("Unexpected calibration report schema")
    report_version = str(report.get("schema_version", ""))
    if (profile_policy or report_version) and report_version.split(
        ".", 1
    )[0] != policy_version.split(".", 1)[0]:
        raise ValueError("Calibration report and acceptance policy major versions differ")
    if report.get("outcome_blinded") is not True:
        raise ValueError("Calibration report must affirm outcome blinding")
    if profile_policy and (
        report.get("calibration_stage") != "final"
        or report.get("publication_minima_satisfied") is not True
    ):
        raise ValueError("Calibration v2 report is not a publication-scale final run")

    bindings = report.get("input_bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("Calibration report input_bindings must be an object")
    pathway_binding_keys = {
        "pathway_universe_sha256",
        "pathway_universe_logical_sha256",
    }
    if not profile_policy:
        pathway_binding_keys.add("pathway_universe_relative_path")
    if not pathway_binding_keys.issubset(expected_bindings):
        raise ValueError(
            "Validator must receive canonical pathway path, file SHA, and logical SHA"
        )
    if profile_policy:
        policy_to_expected = {
            "analysis_plan_sha256": "analysis_plan_sha256",
            "calibration_runner_spec_sha256": "runner_spec_sha256",
            "calibration_report_schema_sha256": "calibration_report_schema_sha256",
        }
        for policy_key, expected_key in policy_to_expected.items():
            if design_binding.get(policy_key) != expected_bindings.get(expected_key):
                raise ValueError(
                    f"Calibration {policy_key} differs from the frozen blind binding"
                )
    elif expected_bindings.get("runner_spec_sha256") != design_binding.get(
        "calibration_runner_spec_sha256"
    ):
        raise ValueError("Calibration runner spec differs from the frozen policy")
    if not profile_policy and (
        bindings.get("pathway_universe_relative_path")
        != "reference/t21_pathway_universe_v1.tsv"
    ):
        raise ValueError("Calibration report does not bind the canonical pathway path")
    for key in (
        "pathway_universe_sha256",
        "pathway_universe_logical_sha256",
    ):
        value = bindings.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"Calibration pathway binding {key!r} is invalid")
    code_commit = bindings.get("code_commit")
    if not isinstance(code_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", code_commit.lower()
    ):
        raise ValueError("Calibration code_commit must be a full Git commit")
    code_dirty = bindings.get("code_dirty")
    if not isinstance(code_dirty, bool):
        raise ValueError("Calibration code_dirty must be boolean")
    for key, expected in expected_bindings.items():
        if bindings.get(key) != expected:
            raise ValueError(f"Calibration input binding differs for {key!r}")
    if code_dirty:
        patch_hash = str(bindings.get("code_patch_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", patch_hash):
            raise ValueError("Dirty calibration code requires a complete patch SHA256")
        if profile_policy:
            raise ValueError("Publication-scale calibration v2 requires clean committed code")
    if profile_policy:
        required_profile_binding_keys = {
            "bindings_schema_version",
            "design_profile_sha256",
            "design_profile_payload_sha256",
            "analysis_plan_sha256",
            "runner_spec_sha256",
            "calibration_report_schema_sha256",
        }
        if not required_profile_binding_keys.issubset(expected_bindings):
            raise ValueError("Calibration v2 expected bindings omit the blind profile")
        if any(
            str(key).lower().endswith("_path")
            or "relative_path" in str(key).lower()
            for key in bindings
        ):
            raise ValueError("Calibration v2 bindings may not expose filesystem paths")
        execution = report.get("execution")
        usage = report.get("design_profile_usage")
        if not isinstance(execution, Mapping) or not isinstance(usage, Mapping):
            raise ValueError("Calibration v2 report lacks profile-use evidence")
        benchmark = execution.get("publication_runner_benchmark")
        benchmark_valid = (
            isinstance(benchmark, Mapping)
            and math.isfinite(float(benchmark.get("wall_clock_seconds", math.nan)))
            and float(benchmark.get("wall_clock_seconds", 0.0)) > 0
            and math.isfinite(
                float(benchmark.get("replicate_work_units_per_second", math.nan))
            )
            and float(benchmark.get("replicate_work_units_per_second", 0.0)) > 0
            and int(benchmark.get("replicate_work_units", 0)) > 0
            and math.isclose(
                float(benchmark.get("replicate_work_units_per_second", 0.0)),
                int(benchmark.get("replicate_work_units", 0))
                / float(benchmark.get("wall_clock_seconds", math.inf)),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            and int(benchmark.get("seed", -1)) == 20260713
            and int(benchmark.get("chunk_size", -1)) == 32
            and benchmark.get("vectorized_shared_freedman_lane_batch_used") is True
            and int(benchmark.get("mapping_batch_size", -1)) == 16
        )
        if (
            not benchmark_valid
            or execution.get("real_pathway_results_read") is not False
            or execution.get("phase") != "final"
            or execution.get("development_override") is not False
            or int(execution.get("seed", -1)) != 20260713
            or int(execution.get("chunk_size", -1)) != 32
            or execution.get("design_profile_used") is not True
            or execution.get("n_curve_bins") != 20
            or not 2 <= int(execution.get("selected_n_curve_bins", 0)) <= 20
            or execution.get("condition_label_space_size") != 680
            or execution.get("label_space_exhaustive") is not True
            or execution.get(
                "finite_sample_exactness_with_continuous_covariates_claimed"
            )
            is not False
            or execution.get("formal_regulation_shared_kernel_used") is not True
            or execution.get("formal_timing_shared_kernel_used") is not True
            or execution.get("formal_power_shared_kernel_used") is not True
            or execution.get("formal_loco_shared_kernel_used") is not True
            or execution.get("formal_loco_fixed_s_support_gate_used") is not True
            or execution.get("vectorized_shared_freedman_lane_batch_used") is not True
            or execution.get("mapping_batch_size") != 16
            or execution.get("maximum_replicate_chunk_size") != 32
            or not 1 <= int(execution.get("chunk_size", 0)) <= 32
            or execution.get(
                "formal_occupancy_fate_decomposition_kernel_used"
            )
            is not True
            or execution.get("shared_state_population_draw_used") is not True
            or execution.get("chr21_gene_level_total_trans_projection_used")
            is not True
            or execution.get("fixed_20_bin_source_grid_verified") is not True
            or execution.get("selected_support_design_valid") is not True
            or execution.get("onset_interval_coverage_evaluated") is not False
            or execution.get("onset_confidence_interval_claim_unlocked") is not False
            or execution.get("covariate_stress_signature_effect_injected") is not True
            or usage.get("fixed_common_grid_bins") != 20
            or not 2 <= int(usage.get("selected_common_grid_bins", 0)) <= 20
            or usage.get("fixed_20_bin_source_grid_verified") is not True
            or usage.get("selected_support_design_valid") is not True
            or usage.get("raw_rows_carry_profile_and_parameter_hashes") is not True
            or usage.get("covariate_stress_signature_effect_injected") is not True
            or usage.get("sensitivity_signature_matrix_sha256")
            != execution.get("sensitivity_signature_matrix_sha256")
            or int(usage.get("n_sensitivity_signature_components", 0))
            != int(execution.get("n_sensitivity_signature_components", -1))
            or not 1
            <= int(usage.get("n_sensitivity_signature_components", 0))
            <= 3
            or any(
                not re.fullmatch(r"[0-9a-f]{64}", str(usage.get(key, "")))
                or usage.get(key) != execution.get(key)
                for key in (
                    "canonical_donor_design_spec_sha256",
                    "canonical_reduced_design_sha256",
                    "canonical_terms_sha256",
                    "canonical_encoding_sha256",
                )
            )
            or usage.get("profile_file_sha256")
            != bindings.get("design_profile_sha256")
            or usage.get("profile_payload_sha256")
            != bindings.get("design_profile_payload_sha256")
        ):
            raise ValueError("Calibration v2 report does not prove formal blind profile use")

    replicate_minima = policy["replicate_minima"]
    thresholds = policy["acceptance_thresholds"]
    expected_scenarios = list(policy["scenario_names"])
    scenario_rows = report.get("scenario_metrics")
    if not isinstance(scenario_rows, list):
        raise ValueError("Calibration scenario_metrics must be a list")
    scenario_names = [str(row.get("scenario", "")) for row in scenario_rows]
    if len(scenario_names) != len(set(scenario_names)) or set(scenario_names) != set(
        expected_scenarios
    ):
        raise ValueError("Calibration scenarios must exactly match the frozen six scenarios")
    scenarios = {str(row["scenario"]): row for row in scenario_rows}

    for scenario, row in scenarios.items():
        observed_replicates = int(row.get("n_replicates", 0))
        if profile_policy:
            publication = policy["publication_execution_contract"]
            expected_replicates = int(
                publication["complete_null_replicates"]
                if scenario == "complete_null"
                else publication["scenario_replicates"]
            )
            if observed_replicates != expected_replicates:
                raise ValueError(
                    f"Calibration scenario {scenario!r} has {observed_replicates} "
                    f"replicates; exactly {expected_replicates} are frozen"
                )
            continue
        minimum = int(
            replicate_minima["final_null"]
            if scenario == "complete_null"
            else replicate_minima["scenario_screen"]
        )
        if observed_replicates < minimum:
            raise ValueError(
                f"Calibration scenario {scenario!r} has {observed_replicates} "
                f"replicates; at least {minimum} are required"
            )

    null = scenarios["complete_null"]
    null_checks = {
        "empirical_fwer": "empirical_fwer_max",
        "empirical_fdr": "empirical_fdr_max",
        "onset_false_positive_rate": "onset_false_positive_rate_max",
        "duration_false_positive_rate": "duration_false_positive_rate_max",
    }
    for metric, threshold in null_checks.items():
        if _probability_metric(null, metric) > float(thresholds[threshold]):
            raise ValueError(f"Complete-null {metric} exceeds its frozen threshold")
    coverage_metric = (
        "pointwise_curve_coverage" if profile_policy else "confidence_coverage"
    )
    coverage_threshold = (
        "complete_null_pointwise_curve_coverage_min"
        if profile_policy
        else "confidence_coverage_min"
    )
    if _probability_metric(null, coverage_metric) < float(
        thresholds[coverage_threshold]
    ):
        raise ValueError("Complete-null pointwise curve coverage is below threshold")
    if profile_policy:
        claim_boundary = policy.get("claim_boundary", {})
        if (
            claim_boundary.get("onset_interval_coverage_evaluated") is not False
            or claim_boundary.get("onset_confidence_interval_claim_unlocked")
            is not False
        ):
            raise ValueError("Calibration v2 must not imply onset-interval coverage")

    scenario_checks = (
        (
            "occupancy_only",
            "regulation_false_discovery_rate",
            "occupancy_only_regulation_false_discovery_rate_max",
        ),
        (
            "fate_only",
            "regulation_false_discovery_rate",
            "fate_only_regulation_false_discovery_rate_max",
        ),
        (
            "trajectory_speed_or_mapping_only",
            "regulation_false_discovery_rate",
            "speed_only_regulation_false_discovery_rate_max",
        ),
        (
            "trajectory_speed_or_mapping_only",
            "false_timing_shift_rate",
            "speed_only_false_timing_shift_rate_max",
        ),
        (
            "covariate_condition_association",
            "empirical_fwer",
            "covariate_extreme_donor_empirical_fwer_max",
        ),
        (
            "chr21_dosage_only",
            "trans_false_discovery_rate",
            "chr21_only_trans_false_discovery_rate_max",
        ),
    )
    for scenario, metric, threshold in scenario_checks:
        if _probability_metric(scenarios[scenario], metric) > float(thresholds[threshold]):
            raise ValueError(f"Scenario {scenario!r} metric {metric!r} exceeds threshold")
    if profile_policy:
        positive_detector_checks = (
            (
                "occupancy_only",
                "occupancy_detection_rate",
                "occupancy_signal_detection_rate_min",
            ),
            (
                "fate_only",
                "fate_detection_rate",
                "fate_signal_detection_rate_min",
            ),
            (
                "chr21_dosage_only",
                "chr21_total_positive_control_detection_rate",
                "chr21_total_positive_control_detection_rate_min",
            ),
        )
        for scenario, metric, threshold in positive_detector_checks:
            if _probability_metric(scenarios[scenario], metric) < float(
                thresholds[threshold]
            ):
                raise ValueError(
                    f"Scenario {scenario!r} positive detector {metric!r} is below threshold"
                )
        detector_false_positive_checks = (
            (
                "complete_null",
                "occupancy_detection_rate",
                "complete_null_occupancy_detection_rate_max",
            ),
            (
                "complete_null",
                "fate_detection_rate",
                "complete_null_fate_detection_rate_max",
            ),
            (
                "occupancy_only",
                "fate_detection_rate",
                "occupancy_only_fate_detection_rate_max",
            ),
            (
                "fate_only",
                "occupancy_detection_rate",
                "fate_only_occupancy_detection_rate_max",
            ),
        )
        for scenario, metric, threshold in detector_false_positive_checks:
            if _probability_metric(scenarios[scenario], metric) > float(
                thresholds[threshold]
            ):
                raise ValueError(
                    f"Scenario {scenario!r} detector {metric!r} exceeds threshold"
                )

    power = report.get("power_metrics")
    if not isinstance(power, Mapping):
        raise ValueError("Calibration power_metrics must be an object")
    if int(power.get("n_replicates_per_point", 0)) < int(
        replicate_minima["power_per_point"]
    ):
        raise ValueError("Calibration power grid has too few replicates per point")
    if profile_policy and int(power.get("n_replicates_per_point", 0)) != int(
        policy["publication_execution_contract"]["power_replicates_per_point"]
    ):
        raise ValueError("Calibration v2 power grid changed its frozen replicate count")
    target_effect = _finite_metric(power, "target_effect_standardized")
    if not math.isclose(
        target_effect,
        float(policy["power_target"]["standardized_effect"]),
        abs_tol=1e-12,
    ):
        raise ValueError("Calibration power target differs from the frozen policy")
    lower_checks = {
        "power_at_target_effect": "power_at_target_effect_min",
        "leave_one_control_out_power": "leave_one_control_out_power_min",
    }
    upper_checks = {
        "minimum_detectable_effect_standardized": "minimum_detectable_effect_max_standardized",
        "minimum_resolvable_onset_shift": "minimum_resolvable_onset_shift_max",
        "extreme_donor_empirical_fwer": "covariate_extreme_donor_empirical_fwer_max",
    }
    for metric, threshold in lower_checks.items():
        if _probability_metric(power, metric) < float(thresholds[threshold]):
            raise ValueError(f"Calibration power metric {metric!r} is below threshold")
    for metric, threshold in upper_checks.items():
        value = _finite_metric(power, metric)
        if value < 0 or ("fwer" in metric and value > 1):
            raise ValueError(f"Calibration metric {metric!r} is outside its domain")
        if value > float(thresholds[threshold]):
            raise ValueError(f"Calibration power metric {metric!r} exceeds threshold")

    artifacts = report.get("output_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Calibration report must register result artifacts")
    root = Path(repository_root).resolve() if repository_root is not None else None
    seen_paths: set[str] = set()
    for artifact in artifacts:
        relative_text = str(artifact.get("relative_path", ""))
        if not relative_text or relative_text in seen_paths:
            raise ValueError("Calibration artifact paths must be non-empty and unique")
        seen_paths.add(relative_text)
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Calibration artifact path must be repository-relative")
        digest = str(artifact.get("sha256", "")).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Calibration artifact SHA256 is invalid")
        if root is not None:
            local = (root / relative).resolve()
            try:
                local.relative_to(root)
            except ValueError as exc:
                raise ValueError("Calibration artifact escapes repository root") from exc
            if not local.is_file():
                raise ValueError(f"Calibration artifact is missing: {relative_text}")
            if int(artifact.get("bytes", -1)) != local.stat().st_size:
                raise ValueError("Calibration artifact byte count changed")
            if sha256_file(local) != digest:
                raise ValueError("Calibration artifact digest changed")
    raw_verification: dict[str, Any] | None = None
    if profile_policy:
        if root is None:
            raise ValueError(
                "Calibration v2 validation requires repository_root for raw evidence"
            )
        bound_files = (
            ("analysis plan", analysis_relative, analysis_sha256),
            ("runner spec", runner_relative, runner_sha256),
            ("report schema", report_schema_relative, report_schema_sha256),
        )
        resolved_bound_files: dict[str, Path] = {}
        for label, relative_text, expected_sha256 in bound_files:
            local_path = (root / relative_text).resolve()
            try:
                local_path.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"Calibration v2 {label} escapes repository root"
                ) from exc
            if (
                not local_path.is_file()
                or sha256_file(local_path) != expected_sha256
            ):
                raise ValueError(
                    f"Calibration v2 {label} differs from the frozen policy"
                )
            resolved_bound_files[label] = local_path
        runner_path = resolved_bound_files["runner spec"]
        from .t21_preunblinding_calibration import (
            validate_pre_unblinding_calibration_artifacts,
        )

        raw_verification = validate_pre_unblinding_calibration_artifacts(
            report,
            repository_root=root,
            runner_spec_path=runner_path,
        )
        if raw_verification.get("design_profile_usage_verified") is not True:
            raise ValueError("Calibration v2 raw evidence did not verify profile use")
    result = {
        "status": "pass",
        "policy_id": str(policy["policy_id"]),
        "report_id": str(report["report_id"]),
        "n_scenarios": len(scenarios),
        "final_null_replicates": int(null["n_replicates"]),
        "power_replicates_per_point": int(power["n_replicates_per_point"]),
        "minimum_detectable_effect_standardized": float(
            power["minimum_detectable_effect_standardized"]
        ),
        "minimum_resolvable_onset_shift": float(
            power["minimum_resolvable_onset_shift"]
        ),
    }
    if raw_verification is not None:
        result["raw_replicate_verification"] = raw_verification
        result["publication_calibration_eligible"] = True
    return result


def validate_unblinding_decision(
    decision: Mapping[str, Any], *, report_sha256: str, policy_sha256: str
) -> dict[str, Any]:
    """Validate the human-attested decision that opens real pathway outcomes."""
    if decision.get("schema_name") != "t21_unblinding_decision":
        raise ValueError("Unexpected unblinding decision schema")
    if decision.get("decision") != "unlock_real_pathway_results":
        raise ValueError("Unblinding decision does not authorize outcome access")
    if decision.get("real_pathway_outcomes_inspected_before_decision") is not False:
        raise ValueError("Unblinding decision records prior outcome inspection")
    if str(decision.get("calibration_report_sha256", "")) != report_sha256:
        raise ValueError("Unblinding decision is bound to a different report")
    if str(decision.get("calibration_policy_sha256", "")) != policy_sha256:
        raise ValueError("Unblinding decision is bound to a different policy")
    for key in ("decision_id", "decided_at_utc", "approver", "rationale"):
        if not str(decision.get(key, "")).strip():
            raise ValueError(f"Unblinding decision field {key!r} must be non-empty")
    decided_at = str(decision["decided_at_utc"]).strip()
    try:
        parsed_decision_time = datetime.fromisoformat(
            decided_at[:-1] + "+00:00" if decided_at.endswith("Z") else decided_at
        )
    except ValueError as exc:
        raise ValueError(
            "Unblinding decision decided_at_utc must be an ISO-8601 timestamp"
        ) from exc
    if (
        parsed_decision_time.tzinfo is None
        or parsed_decision_time.utcoffset() != timezone.utc.utcoffset(
            parsed_decision_time
        )
    ):
        raise ValueError(
            "Unblinding decision decided_at_utc must be timezone-aware UTC"
        )
    return {
        "status": "pass",
        "decision_id": str(decision["decision_id"]),
        "approver": str(decision["approver"]),
    }


def apply_author_qc_v1(
    adata: ad.AnnData,
    *,
    run_scrublet: bool = True,
    random_state: int = 0,
) -> tuple[ad.AnnData, pd.DataFrame]:
    """Apply the public author QC thresholds with an explicit removal ledger.

    Cell calling is intentionally outside this function.  The input must
    already contain only EmptyDrops-called barcodes.  Scrublet parameters
    mirror the public code (Scrublet 0.2.3, expected doublet rate 0.06,
    30 PCs, and Scrublet's default seed 0).  This remains a reconstruction,
    not an author label or author-processed object.
    """
    if "counts" not in adata.layers:
        raise ValueError('QC requires layers["counts"]')
    counts = sparse.csr_matrix(adata.layers["counts"])
    total_counts = np.asarray(counts.sum(axis=1)).ravel()
    n_genes = np.asarray(counts.getnnz(axis=1)).ravel()
    symbols = adata.var["gene_symbol"].astype(str)
    mt_mask = symbols.str.upper().str.startswith("MT-").to_numpy()
    mt_counts = (
        np.asarray(counts[:, mt_mask].sum(axis=1)).ravel()
        if mt_mask.any()
        else np.zeros(adata.n_obs, dtype=float)
    )
    pct_mt = np.divide(
        100.0 * mt_counts,
        total_counts,
        out=np.zeros_like(total_counts, dtype=float),
        where=total_counts > 0,
    )
    adata.obs["total_counts"] = total_counts
    adata.obs["n_genes_by_counts"] = n_genes
    adata.obs["pct_counts_mt"] = pct_mt
    rules = (
        ("min_genes_gt_250", n_genes > 250),
        ("max_genes_lt_8500", n_genes < 8500),
        ("min_counts_gt_750", total_counts > 750),
        ("max_counts_lt_110000", total_counts < 110000),
        ("pct_mito_lt_20", pct_mt < 20.0),
    )
    keep = np.ones(adata.n_obs, dtype=bool)
    ledger_rows = []
    for rule, passes in rules:
        before = int(keep.sum())
        newly_removed = keep & ~passes
        keep &= passes
        ledger_rows.append(
            {
                "step": rule,
                "n_before": before,
                "n_removed": int(newly_removed.sum()),
                "n_after": int(keep.sum()),
                "implementation": "public_author_threshold_reimplementation",
            }
        )
    filtered = adata[keep].copy()
    if run_scrublet:
        import scrublet as scrublet_module

        scrublet_version = importlib_metadata.version("scrublet")
        if scrublet_version != "0.2.3":
            raise RuntimeError(
                f"Pinned Scrublet 0.2.3 is required, found {scrublet_version}"
            )

        scrublet = scrublet_module.Scrublet(
            sparse.csr_matrix(filtered.layers["counts"]),
            expected_doublet_rate=0.06,
            random_state=random_state,
        )
        doublet_score, predicted = scrublet.scrub_doublets(
            min_counts=2,
            min_cells=3,
            min_gene_variability_pctl=85,
            n_prin_comps=30,
            verbose=False,
        )
        predicted = np.asarray(predicted, dtype=bool)
        filtered.obs["doublet_score"] = np.asarray(doublet_score, dtype=float)
        filtered.obs["predicted_doublet"] = predicted
        before = filtered.n_obs
        filtered = filtered[~predicted].copy()
        ledger_rows.append(
            {
                "step": "scrublet_expected_doublet_rate_0.06",
                "n_before": int(before),
                "n_removed": int(predicted.sum()),
                "n_after": int(filtered.n_obs),
                "implementation": (
                    "scrublet_0.2.3_public_parameters_min_counts2_min_cells3_"
                    "variability85_30pcs_seed_"
                    + str(random_state)
                ),
            }
        )
    filtered.uns.setdefault("t21_data_product", {})["qc"] = {
        "plan_id": "t21_author_thresholds_pinned_scrublet_reconstruction_v1",
        "cell_calling_required": "DropletUtils_emptyDrops_seed100_FDR0.001_plus_inflection",
        "thresholds": {
            "min_genes": 250,
            "max_genes": 8500,
            "min_counts": 750,
            "max_counts": 110000,
            "count_and_gene_bounds": "strict_gt_min_and_strict_lt_max",
            "max_pct_mito_strict": 20.0,
            "scrublet_expected_doublet_rate": 0.06,
            "scrublet_version": "0.2.3",
            "scrublet_min_counts": 2,
            "scrublet_min_cells": 3,
            "scrublet_min_gene_variability_pctl": 85,
            "scrublet_n_prin_comps": 30,
            "scrublet_random_state": random_state,
        },
        "scrublet_applied": bool(run_scrublet),
        "scrublet_runtime_version": (
            importlib_metadata.version("scrublet") if run_scrublet else "not_run"
        ),
    }
    return filtered, pd.DataFrame(ledger_rows)


def validate_fate_probabilities(
    frame: pd.DataFrame, *, expected_cell_ids: Iterable[str] | None = None, atol: float = 1e-6
) -> dict[str, Any]:
    missing = sorted(set(REQUIRED_FATE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Fate table is missing columns: {missing}")
    if frame["cell_id"].astype(str).duplicated().any():
        raise ValueError("Fate table cell IDs are not unique")
    if frame["cell_id"].isna().any() or frame["cell_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Fate table cell IDs must be non-empty")
    if not pd.api.types.is_bool_dtype(frame["fate_eligible"].dtype):
        raise ValueError("fate_eligible must use a boolean dtype")
    if frame["fate_eligible"].isna().any():
        raise ValueError("fate_eligible may not contain missing values")
    eligible = frame["fate_eligible"]
    for column in ("fate_model_id", "trajectory_draw_id", "terminal_definition_hash"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Fate field {column!r} must be non-empty")
    if (
        frame.loc[~eligible, "fate_ineligibility_reason"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):
        raise ValueError("Ineligible cells require a fate_ineligibility_reason")
    probabilities = frame.loc[:, FATE_PROBABILITY_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    eligible_values = probabilities.loc[eligible].to_numpy(dtype=float)
    if eligible_values.size:
        if np.any(~np.isfinite(eligible_values)):
            raise ValueError("Eligible fate probabilities must be finite")
        if np.any((eligible_values < -atol) | (eligible_values > 1 + atol)):
            raise ValueError("Fate probabilities are outside [0, 1]")
        if not np.allclose(eligible_values.sum(axis=1), 1.0, atol=atol, rtol=0):
            raise ValueError("Eligible fate probabilities do not sum to one")
    if probabilities.loc[~eligible].notna().any(axis=None):
        raise ValueError("Ineligible cells must have null fate probabilities")
    if expected_cell_ids is not None:
        expected = {str(value) for value in expected_cell_ids}
        observed = set(frame["cell_id"].astype(str))
        if observed != expected:
            raise ValueError(
                f"Fate/Zarr cell-ID sets differ: missing={len(expected-observed)}, "
                f"unexpected={len(observed-expected)}"
            )
    return {
        "n_rows": int(len(frame)),
        "n_eligible": int(eligible.sum()),
        "cell_id_set_hash": cell_id_set_hash(frame["cell_id"].astype(str)),
        "trajectory_draw_ids": sorted(set(frame["trajectory_draw_id"].astype(str))),
        "terminal_definition_hashes": sorted(
            set(frame["terminal_definition_hash"].astype(str))
        ),
    }


def validate_trajectory_arrays(
    *,
    cell_ids: Sequence[str],
    draw_ids: Sequence[str],
    pseudotime: np.ndarray,
    mapped: np.ndarray,
    donor_ids: Sequence[str],
    bin_left: np.ndarray,
    bin_center: np.ndarray,
    bin_right: np.ndarray,
    donor_bin_cell_count: np.ndarray,
    donor_bin_available: np.ndarray,
    draw_metadata: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the in-memory contract used to write the trajectory Zarr."""
    n_cells, n_draws = len(cell_ids), len(draw_ids)
    if n_cells == 0 or n_draws == 0 or len(donor_ids) == 0:
        raise ValueError("Trajectory axes may not be empty")
    for label, values in (
        ("cell_id", cell_ids),
        ("trajectory_draw_id", draw_ids),
        ("donor_id", donor_ids),
    ):
        normalized = [str(value).strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError(f"Trajectory {label} axis contains empty IDs")
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"Trajectory {label} axis contains duplicate IDs")
    if pseudotime.shape != (n_cells, n_draws) or mapped.shape != (n_cells, n_draws):
        raise ValueError("pseudotime/mapped must have shape cell x trajectory_draw")
    if mapped.dtype != bool:
        raise ValueError("mapped must be boolean")
    mapped_values = pseudotime[mapped]
    if mapped_values.size and (
        np.any(~np.isfinite(mapped_values))
        or np.any(mapped_values < 0)
        or np.any(mapped_values > 1)
    ):
        raise ValueError("Mapped pseudotime values must be finite and within [0, 1]")
    if np.any(~mapped & ~np.isnan(pseudotime)):
        raise ValueError("Unmapped pseudotime entries must be NaN")
    if not (
        bin_left.ndim == bin_center.ndim == bin_right.ndim == 1
        and len(bin_left) == len(bin_center) == len(bin_right)
        and len(bin_left) > 0
        and np.all(np.isfinite(bin_left))
        and np.all(np.isfinite(bin_center))
        and np.all(np.isfinite(bin_right))
        and np.all((0 <= bin_left) & (bin_left < bin_center))
        and np.all((bin_center < bin_right) & (bin_right <= 1))
        and np.all(np.diff(bin_left) > 0)
        and np.all(np.diff(bin_center) > 0)
        and np.all(np.diff(bin_right) > 0)
        and np.allclose(bin_left[1:], bin_right[:-1], rtol=0, atol=1e-8)
    ):
        raise ValueError("Pseudotime bin axes are invalid")
    expected_shape = (len(donor_ids), len(bin_center), n_draws)
    if donor_bin_cell_count.shape != expected_shape:
        raise ValueError(f"donor-bin cell counts must have shape {expected_shape}")
    if donor_bin_available.shape != expected_shape or donor_bin_available.dtype != bool:
        raise ValueError("donor-bin availability has wrong shape or dtype")
    if not np.issubdtype(donor_bin_cell_count.dtype, np.integer):
        raise ValueError("donor-bin cell counts must use an integer dtype")
    if np.any(donor_bin_cell_count < 0) or not np.array_equal(
        donor_bin_cell_count, np.rint(donor_bin_cell_count)
    ):
        raise ValueError("donor-bin cell counts must be non-negative integers")
    if not np.array_equal(donor_bin_available, donor_bin_cell_count > 0):
        raise ValueError("donor-bin availability must equal cell_count > 0")
    if len(draw_metadata) != n_draws:
        raise ValueError("One metadata record is required per trajectory draw")
    required_draw = {
        "trajectory_draw_id",
        "method",
        "method_version",
        "parameters_json",
        "parameters_hash",
        "root_definition_id",
        "root_cell_set_hash",
        "terminal_definition_id",
        "terminal_definition_hash",
        "seed",
        "rng",
        "used_condition_information",
        "used_candidate_pathway_genes",
        "orientation",
        "status",
        "correlation_with_primary",
    }
    for index, metadata in enumerate(draw_metadata):
        missing = required_draw.difference(metadata)
        if missing:
            raise ValueError(f"Trajectory draw {index} metadata is missing {sorted(missing)}")
        if str(metadata["trajectory_draw_id"]) != str(draw_ids[index]):
            raise ValueError("Trajectory draw metadata order does not match the draw axis")
        for field in ("used_condition_information", "used_candidate_pathway_genes"):
            if not isinstance(metadata[field], (bool, np.bool_)):
                raise ValueError(f"Trajectory draw {field} must be boolean")
        if metadata["used_condition_information"]:
            raise ValueError("Primary trajectory draws may not use condition information")
        if metadata["used_candidate_pathway_genes"]:
            raise ValueError("Primary trajectory draws may not use candidate pathway genes")
        parameters_json = str(metadata["parameters_json"])
        try:
            parsed_parameters = json.loads(parameters_json)
        except json.JSONDecodeError as exc:
            raise ValueError("Trajectory parameters_json is not valid JSON") from exc
        if stable_json(parsed_parameters) != parameters_json:
            raise ValueError("Trajectory parameters_json must use canonical stable JSON")
        expected_parameters_hash = sha256(parameters_json.encode("utf-8")).hexdigest()
        if str(metadata["parameters_hash"]) != expected_parameters_hash:
            raise ValueError("Trajectory parameters_hash does not match parameters_json")
        correlation = float(metadata["correlation_with_primary"])
        if not np.isfinite(correlation) or not -1.0 <= correlation <= 1.0:
            raise ValueError("correlation_with_primary must be finite and within [-1, 1]")
    return {
        "n_cells": n_cells,
        "n_draws": n_draws,
        "n_donors": len(donor_ids),
        "n_bins": len(bin_center),
        "cell_id_set_hash": cell_id_set_hash(cell_ids),
        "cell_id_order_hash": ordered_id_hash(cell_ids),
        "donor_set_hash": cell_id_set_hash(donor_ids),
        "trajectory_draw_ids": sorted(str(value) for value in draw_ids),
        "trajectory_draw_id_order_hash": ordered_id_hash(draw_ids),
        "terminal_definition_hashes": sorted(
            {str(metadata["terminal_definition_hash"]) for metadata in draw_metadata}
        ),
        "grid_hash": sha256(
            stable_json(
                {
                    "bin_left": np.asarray(bin_left, dtype=float).tolist(),
                    "bin_center": np.asarray(bin_center, dtype=float).tolist(),
                    "bin_right": np.asarray(bin_right, dtype=float).tolist(),
                }
            ).encode("utf-8")
        ).hexdigest(),
    }


def _unicode_array(values: Iterable[Any]) -> np.ndarray:
    strings = [str(value) for value in values]
    width = max((len(value) for value in strings), default=1)
    return np.asarray(strings, dtype=f"U{max(width, 1)}")


def write_trajectory_zarr(
    path: str | Path,
    *,
    cell_ids: Sequence[str],
    draw_ids: Sequence[str],
    pseudotime: np.ndarray,
    mapped: np.ndarray,
    donor_ids: Sequence[str],
    bin_left: np.ndarray,
    bin_center: np.ndarray,
    bin_right: np.ndarray,
    donor_bin_cell_count: np.ndarray,
    donor_bin_available: np.ndarray,
    draw_metadata: Sequence[Mapping[str, Any]],
    overwrite: bool = False,
) -> Path:
    """Validate and atomically write the fixed trajectory-draw Zarr layout."""
    validate_trajectory_arrays(
        cell_ids=cell_ids,
        draw_ids=draw_ids,
        pseudotime=np.asarray(pseudotime),
        mapped=np.asarray(mapped),
        donor_ids=donor_ids,
        bin_left=np.asarray(bin_left),
        bin_center=np.asarray(bin_center),
        bin_right=np.asarray(bin_right),
        donor_bin_cell_count=np.asarray(donor_bin_cell_count),
        donor_bin_available=np.asarray(donor_bin_available),
        draw_metadata=draw_metadata,
    )
    import zarr

    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale trajectory staging directory exists: {temporary}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Trajectory product already exists: {path}")
    try:
        group = zarr.open_group(temporary, mode="w", zarr_format=2)
    except TypeError:  # zarr 2.x compatibility
        group = zarr.open_group(temporary, mode="w", zarr_version=2)
    group.attrs.update(
        {
            "schema_name": "t21_trajectory_draws",
            "schema_version": PRODUCT_VERSION,
            "axis_order": "cell_by_trajectory_draw",
            "created_at_utc": utc_now(),
        }
    )

    def put(name: str, values: np.ndarray) -> None:
        array = np.asarray(values)
        group.create_array(name, data=array, overwrite=True)

    put("axes/cell_id", _unicode_array(cell_ids))
    put("axes/trajectory_draw_id", _unicode_array(draw_ids))
    put("pseudotime", np.asarray(pseudotime, dtype=np.float32))
    put("mapped", np.asarray(mapped, dtype=bool))
    put("axes/donor_id", _unicode_array(donor_ids))
    put("axes/bin_left", np.asarray(bin_left, dtype=np.float32))
    put("axes/bin_center", np.asarray(bin_center, dtype=np.float32))
    put("axes/bin_right", np.asarray(bin_right, dtype=np.float32))
    put("donor_bin/cell_count", np.asarray(donor_bin_cell_count, dtype=np.int64))
    put("donor_bin/available", np.asarray(donor_bin_available, dtype=bool))
    put(
        "draw_metadata/json",
        _unicode_array(stable_json(dict(record)) for record in draw_metadata),
    )
    if path.exists():
        backup = path.with_name(path.name + ".previous")
        if backup.exists():
            raise FileExistsError(f"Trajectory backup path already exists: {backup}")
        os.replace(path, backup)
        try:
            os.replace(temporary, path)
        except Exception:
            os.replace(backup, path)
            raise
        return path
    os.replace(temporary, path)
    return path


def validate_trajectory_zarr(path: str | Path) -> dict[str, Any]:
    """Validate a stored trajectory product and attach its tree digest."""
    import zarr

    path = Path(path)
    group = zarr.open_group(path, mode="r")
    required = (
        "axes/cell_id",
        "axes/trajectory_draw_id",
        "pseudotime",
        "mapped",
        "axes/donor_id",
        "axes/bin_left",
        "axes/bin_center",
        "axes/bin_right",
        "donor_bin/cell_count",
        "donor_bin/available",
        "draw_metadata/json",
    )
    missing = [name for name in required if name not in group]
    if missing:
        raise ValueError(f"Trajectory Zarr is missing arrays: {missing}")
    metadata = [
        json.loads(str(value)) for value in group["draw_metadata/json"][:].tolist()
    ]
    result = validate_trajectory_arrays(
        cell_ids=[str(value) for value in group["axes/cell_id"][:]],
        draw_ids=[str(value) for value in group["axes/trajectory_draw_id"][:]],
        pseudotime=np.asarray(group["pseudotime"][:]),
        mapped=np.asarray(group["mapped"][:]),
        donor_ids=[str(value) for value in group["axes/donor_id"][:]],
        bin_left=np.asarray(group["axes/bin_left"][:]),
        bin_center=np.asarray(group["axes/bin_center"][:]),
        bin_right=np.asarray(group["axes/bin_right"][:]),
        donor_bin_cell_count=np.asarray(group["donor_bin/cell_count"][:]),
        donor_bin_available=np.asarray(group["donor_bin/available"][:]),
        draw_metadata=metadata,
    )
    result["tree_digest_sha256"] = tree_digest(path)
    return result


def validate_trajectory_scrna_alignment(
    path: str | Path, scrna_obs: pd.DataFrame
) -> dict[str, Any]:
    """Recompute every donor-by-bin count from H5AD donor assignments."""
    import zarr

    if "donor_id" not in scrna_obs:
        raise ValueError("scRNA obs lacks donor_id for trajectory count validation")
    if not scrna_obs.index.is_unique:
        raise ValueError("scRNA obs cell IDs must be unique")
    path = Path(path)
    summary = validate_trajectory_zarr(path)
    group = zarr.open_group(path, mode="r")
    cell_ids = np.asarray([str(value) for value in group["axes/cell_id"][:]])
    donor_ids = np.asarray([str(value) for value in group["axes/donor_id"][:]])
    observed_cells = set(scrna_obs.index.astype(str))
    if set(cell_ids) != observed_cells:
        raise ValueError("Trajectory and scRNA cell-ID sets differ")
    obs_index = scrna_obs.copy()
    obs_index.index = obs_index.index.astype(str)
    cell_donors = obs_index.loc[cell_ids, "donor_id"].astype(str).to_numpy()
    observed_donors = set(cell_donors)
    if set(donor_ids) != observed_donors:
        raise ValueError("Trajectory donor axis and scRNA donor assignments differ")
    donor_lookup = {donor_id: index for index, donor_id in enumerate(donor_ids)}
    donor_index = np.asarray([donor_lookup[donor_id] for donor_id in cell_donors])

    bin_left = np.asarray(group["axes/bin_left"][:], dtype=float)
    bin_right = np.asarray(group["axes/bin_right"][:], dtype=float)
    edges = np.concatenate(([bin_left[0]], bin_right))
    pseudotime = np.asarray(group["pseudotime"][:], dtype=float)
    mapped = np.asarray(group["mapped"][:], dtype=bool)
    declared = np.asarray(group["donor_bin/cell_count"][:], dtype=np.int64)
    recomputed = np.zeros_like(declared)
    for draw_index in range(pseudotime.shape[1]):
        selected = mapped[:, draw_index]
        values = pseudotime[selected, draw_index]
        bin_index = np.searchsorted(edges, values, side="right") - 1
        bin_index = np.clip(bin_index, 0, len(bin_left) - 1)
        np.add.at(
            recomputed[:, :, draw_index],
            (donor_index[selected], bin_index),
            1,
        )
    if not np.array_equal(recomputed, declared):
        disagreement = int(np.abs(recomputed - declared).sum())
        raise ValueError(
            f"Trajectory donor-bin counts disagree with H5AD by {disagreement} cells"
        )
    return {
        **summary,
        "scrna_donor_set_hash": cell_id_set_hash(observed_donors),
        "donor_bin_counts_recomputed": True,
    }


def formal_t21_analysis_view(
    adata: ad.AnnData,
    *,
    trajectory_path: str | Path,
    fates_path: str | Path,
) -> dict[str, Any]:
    """Load and align the one permitted formal T21 cell-analysis view.

    The H5AD remains the immutable trajectory input.  Pseudotime and eligibility
    are read only from the profile-bound trajectory/fate artifacts and aligned by
    global cell ID; callers cannot supply an alternative pseudotime or mask.
    """
    import zarr

    validate_trajectory_scrna_alignment(trajectory_path, adata.obs)
    fates = pd.read_parquet(fates_path)
    fate_summary = validate_fate_probabilities(
        fates, expected_cell_ids=adata.obs_names.astype(str)
    )
    group = zarr.open_group(Path(trajectory_path), mode="r")
    primary_draw_id = str(group.attrs.get("primary_trajectory_draw_id", "")).strip()
    if not primary_draw_id:
        raise ValueError("Trajectory Zarr lacks primary_trajectory_draw_id")
    draw_ids = [str(value) for value in group["axes/trajectory_draw_id"][:]]
    if draw_ids.count(primary_draw_id) != 1:
        raise ValueError("Trajectory primary draw is not unique on the draw axis")
    fate_draw_ids = list(fate_summary["trajectory_draw_ids"])
    if fate_draw_ids != [primary_draw_id]:
        raise ValueError("Fate table is not bound to the trajectory primary draw")
    primary_index = draw_ids.index(primary_draw_id)
    trajectory_cell_ids = pd.Index(
        [str(value) for value in group["axes/cell_id"][:]], name="cell_id"
    )
    obs_cell_ids = pd.Index(adata.obs_names.astype(str), name="cell_id")
    trajectory_positions = trajectory_cell_ids.get_indexer(obs_cell_ids)
    if np.any(trajectory_positions < 0):  # pragma: no cover - alignment guarded
        raise ValueError("H5AD contains cells absent from the trajectory product")
    trajectory_mapped = np.asarray(
        group["mapped"][:, primary_index], dtype=bool
    )[trajectory_positions]
    pseudotime = np.asarray(
        group["pseudotime"][:, primary_index], dtype=float
    )[trajectory_positions]
    if not np.array_equal(trajectory_mapped, np.isfinite(pseudotime)):
        raise ValueError("Primary trajectory mapped mask and pseudotime finiteness differ")
    fate_index = fates.assign(cell_id=fates["cell_id"].astype(str)).set_index(
        "cell_id", drop=False
    )
    fate_eligible = fate_index.loc[obs_cell_ids, "fate_eligible"].to_numpy(dtype=bool)
    lineage = adata.obs["lineage_inclusion"]
    if not pd.api.types.is_bool_dtype(lineage.dtype) or lineage.isna().any():
        raise ValueError("lineage_inclusion must be a complete boolean vector")
    lineage_values = lineage.to_numpy(dtype=bool)
    if not np.array_equal(lineage_values, trajectory_mapped):
        raise ValueError("Frozen lineage and primary trajectory eligibility differ")
    if not np.array_equal(lineage_values, fate_eligible):
        raise ValueError("Frozen lineage and primary fate eligibility differ")
    analysis_mask = lineage_values & trajectory_mapped & fate_eligible
    if not np.any(analysis_mask):
        raise ValueError("The formal T21 analysis view contains no cells")
    analysis_cell_ids = obs_cell_ids[analysis_mask]
    return {
        "analysis_mask": analysis_mask,
        "primary_pseudotime": pseudotime,
        "primary_trajectory_draw_id": primary_draw_id,
        "primary_trajectory_draw_id_sha256": sha256(
            primary_draw_id.encode("utf-8")
        ).hexdigest(),
        "n_analysis_cells": int(analysis_mask.sum()),
        "analysis_cell_set_hash": cell_id_set_hash(analysis_cell_ids),
        "analysis_cell_order_hash": ordered_id_hash(analysis_cell_ids),
        "lineage_primary_trajectory_fate_masks_identical": True,
        "trajectory_tree_digest_sha256": tree_digest(trajectory_path),
        "trajectory_cell_set_hash": cell_id_set_hash(trajectory_cell_ids),
        "trajectory_donor_set_hash": cell_id_set_hash(
            [str(value) for value in group["axes/donor_id"][:]]
        ),
        "fates_file_sha256": sha256_file(fates_path),
        "fates_cell_set_hash": str(fate_summary["cell_id_set_hash"]),
    }


def write_fate_probabilities(
    frame: pd.DataFrame, path: str | Path, *, expected_cell_ids: Iterable[str] | None = None
) -> Path:
    """Validate and atomically write soft fate probabilities as Parquet."""
    validate_fate_probabilities(frame, expected_cell_ids=expected_cell_ids)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    os.replace(temporary, path)
    return path


@dataclass(frozen=True)
class DownloadRecord:
    accession: str
    file_name: str
    source_url: str
    destination: str
    expected_size_bytes: int
    local_size_bytes: int
    status: str
    sha256: str
    etag: str
    last_modified: str
    downloaded_at_utc: str
    error: str = ""


def download_with_resume(
    url: str,
    destination: str | Path,
    expected_size: int,
    *,
    accession: str,
    retries: int = 5,
    timeout: tuple[int, int] = (30, 240),
) -> DownloadRecord:
    """Download through a ``.part`` file, validate size/hash, and atomically publish."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if destination.exists() and destination.stat().st_size == expected_size:
        return DownloadRecord(
            accession=accession,
            file_name=destination.name,
            source_url=url,
            destination=destination.as_posix(),
            expected_size_bytes=expected_size,
            local_size_bytes=expected_size,
            status="already_complete",
            sha256=sha256_file(destination),
            etag="",
            last_modified="",
            downloaded_at_utc=utc_now(),
        )
    if destination.exists():
        os.replace(destination, partial)
    if partial.exists() and partial.stat().st_size == expected_size:
        digest = sha256_file(partial)
        os.replace(partial, destination)
        return DownloadRecord(
            accession=accession,
            file_name=destination.name,
            source_url=url,
            destination=destination.as_posix(),
            expected_size_bytes=expected_size,
            local_size_bytes=expected_size,
            status="recovered_complete_partial",
            sha256=digest,
            etag="",
            last_modified="",
            downloaded_at_utc=utc_now(),
        )
    last_error = ""
    response_headers: Mapping[str, str] = {}
    for attempt in range(1, retries + 1):
        existing = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        mode = "ab" if existing else "wb"
        try:
            with requests.get(
                url, stream=True, headers=headers, allow_redirects=True, timeout=timeout
            ) as response:
                if existing and response.status_code == 200:
                    mode = "wb"
                    existing = 0
                response.raise_for_status()
                response_headers = response.headers
                with partial.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            local_size = partial.stat().st_size
            if local_size != expected_size:
                last_error = f"size_mismatch:{local_size}!={expected_size}"
                continue
            digest = sha256_file(partial)
            os.replace(partial, destination)
            return DownloadRecord(
                accession=accession,
                file_name=destination.name,
                source_url=url,
                destination=destination.as_posix(),
                expected_size_bytes=expected_size,
                local_size_bytes=local_size,
                status="downloaded",
                sha256=digest,
                etag=response_headers.get("ETag", ""),
                last_modified=response_headers.get("Last-Modified", ""),
                downloaded_at_utc=utc_now(),
            )
        except Exception as exc:  # noqa: BLE001 - persisted as evidence
            last_error = f"attempt_{attempt}:{type(exc).__name__}:{exc}"
    local_size = partial.stat().st_size if partial.exists() else 0
    return DownloadRecord(
        accession=accession,
        file_name=destination.name,
        source_url=url,
        destination=destination.as_posix(),
        expected_size_bytes=expected_size,
        local_size_bytes=local_size,
        status="failed",
        sha256="",
        etag=response_headers.get("ETag", ""),
        last_modified=response_headers.get("Last-Modified", ""),
        downloaded_at_utc=utc_now(),
        error=last_error,
    )


def download_records_frame(records: Sequence[DownloadRecord]) -> pd.DataFrame:
    return pd.DataFrame([asdict(record) for record in records])


def write_preflight_provenance(
    output_path: str | Path,
    *,
    repository_root: str | Path,
    command: Sequence[str],
    sources: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
    validation_gates: Sequence[Mapping[str, Any]],
    blockers: Sequence[Mapping[str, Any]],
    source_repositories: Sequence[Mapping[str, Any]] = (),
) -> Path:
    """Write an explicitly non-final manifest for an inventory/staging run."""
    repository_root = Path(repository_root)
    manifest = {
        "schema_name": "t21_data_provenance_manifest",
        "schema_version": PRODUCT_VERSION,
        "release_id": "t21_data_product_v1_preflight",
        "release_status": "incomplete_preflight_not_for_biological_discovery",
        "created_at_utc": utc_now(),
        "repository_relative_paths_only": True,
        "command": list(command),
        "sources": list(sources),
        "source_repositories": list(source_repositories),
        "transformations": [
            {
                "step_id": "inventory_and_design_audit",
                "script": "scripts/build_t21_data_product.py",
                "parameters": {"outcome_data_inspected": False},
                "seed": None,
            }
        ],
        "outputs": list(outputs),
        "cross_artifact_contract": {
            "sampling_frame_id": "not_locked",
            "primary_trajectory_draw_id": "not_built",
            "cell_id_set_hash": "not_available_before_count_staging",
            "gene_order_hash": "not_available_before_count_staging",
            "donor_set_hash": "not_final_before_crosswalk_resolution",
            "scrna_donor_set_hash": "not_available_before_count_staging",
            "trajectory_grid_hash": "not_available_before_trajectory_reconstruction",
        },
        "validation_gates": list(validation_gates),
        "blockers": list(blockers),
        "claim_policy": {
            "allowed": [
                "within_fixed_gate_relative_occupancy",
                "metadata_only_sampling_frame_feasibility",
                "cross_study_cross_processing_validation_with_possible_donor_reuse",
            ],
            "forbidden": [
                "pooled_sort_gate_tissue_occupancy",
                "unconfirmed_cross_accession_donor_replication",
                "independent_2021_external_replication_until_genetic_or_author_crosswalk_resolution",
                "cross_study_donor_merge_or_combined_p_values_while_overlap_is_unresolved",
                "pathway_discovery_before_calibration_unblinding_gate",
                "T21_or_GATA1_claim_above_level_3_5_without_matched_functional_gate",
            ],
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    return output_path


def artifact_record(path: str | Path, root: str | Path, role: str) -> dict[str, Any]:
    path, root = Path(path), Path(root)
    if path.is_dir():
        return {
            "role": role,
            "format": "zarr",
            "relative_path": path.relative_to(root).as_posix(),
            "bytes": sum(p.stat().st_size for p in path.rglob("*") if p.is_file()),
            "tree_digest_sha256": tree_digest(path),
        }
    return {
        "role": role,
        "format": path.suffix.lstrip("."),
        "relative_path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_copy_file(source: str | Path, destination: str | Path) -> None:
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
