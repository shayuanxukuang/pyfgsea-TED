"""Fail-closed, condition-blind reference annotation for the T21 product.

This module deliberately does not deserialize or execute the reference model.
It supports three auditable stages:

1. export count shards as condition-stripped H5AD inputs with anonymous row IDs;
2. import a prediction table produced in a controlled external environment; and
3. adjudicate the predictions against a frozen label map and exact cell keys.

The model file is treated only as an immutable byte sequence for SHA256
verification.  Condition, diagnosis and pathway-outcome columns are not
accepted by the prediction import contract.
"""

from __future__ import annotations

from hashlib import sha256
import hmac
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
from typing import Any, Iterable, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import yaml

from .t21_data_product import sha256_file, utc_now


PRIMARY_ACCESSION = "E-MTAB-13067"
PRIMARY_SAMPLING_FRAME_ID = "t21_fetal_liver_cd45_primary_v1"
PRIMARY_ASSEMBLY_SAMPLING_FRAME = "cd45_pos"
CD235A_NEG_SENSITIVITY_SAMPLING_FRAME_ID = (
    "t21_fetal_liver_cd235a_neg_sensitivity_v1"
)
CD235A_NEG_ASSEMBLY_SAMPLING_FRAME = "cd235a_neg"
# Backward-compatible public alias. New sensitivity workflows pass their frame
# explicitly so a CD45+ artifact can never be linked into the CD235a- chain.
SAMPLING_FRAME_ID = PRIMARY_SAMPLING_FRAME_ID
REGISTERED_ASSEMBLY_SAMPLING_FRAMES = {
    PRIMARY_SAMPLING_FRAME_ID: PRIMARY_ASSEMBLY_SAMPLING_FRAME,
    CD235A_NEG_SENSITIVITY_SAMPLING_FRAME_ID: (
        CD235A_NEG_ASSEMBLY_SAMPLING_FRAME
    ),
}
ANNOTATION_TABLE_NAME = "t21_cell_annotations_condition_blind_reference_v1.tsv"
ADJUDICATION_RECORD_NAME = "t21_cell_annotation_adjudication_v1.json"
BLINDING_RECORD_NAME = "t21_condition_blind_annotation_input_record_v1.json"
CELL_MANIFEST_NAME = "t21_annotation_cell_manifest_v1.tsv"
RECORD_CHECKSUM_SUFFIX = ".sha256"
PREDICTION_PLAN_SCHEMA_NAME = "t21_condition_blind_celltypist_prediction_plan"
FEATURE_AUDIT_SCHEMA_NAME = "t21_condition_blind_celltypist_feature_audit"
EXPECTED_CELLTYPIST_MODEL_FEATURES = 6971

CELL_MANIFEST_COLUMNS = (
    "annotation_row_id",
    "cell_id",
    "library_id",
    "original_barcode",
)
PREDICTION_COLUMNS = (
    "annotation_row_id",
    "reference_label",
    "mapping_confidence",
    "annotation_uncertain",
    "predicted_doublet",
)
ANNOTATION_COLUMNS = (
    "cell_id",
    "library_id",
    "original_barcode",
    "original_cell_type",
    "original_cell_type_source",
    "analysis_cell_type",
    "lineage_inclusion",
    "lineage_inclusion_reason",
    "mapping_confidence",
    "annotation_uncertain",
    "predicted_doublet",
    "annotation_row_id",
    "annotation_plan_id",
    "reference_mapping_id",
    "annotation_run_id",
    "model_id",
    "model_sha256",
    "label_mapping_sha256",
    "prediction_table_sha256",
    "cell_manifest_sha256",
)
REFERENCE_AUDIT_COLUMNS = frozenset(ANNOTATION_COLUMNS).difference(
    {
        "cell_id",
        "original_cell_type",
        "original_cell_type_source",
        "analysis_cell_type",
        "lineage_inclusion",
        "lineage_inclusion_reason",
    }
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ROW_ID_RE = re.compile(r"ar_[0-9a-f]{32}")
_FORBIDDEN_FEATURE_TOKENS = (
    "condition",
    "diagnosis",
    "case_control",
    "outcome",
    "pathway",
    "trisomy",
    "disomy",
)
_ALLOWED_BLINDED_VAR_COLUMNS = (
    "gene_id_original",
    "gene_symbol",
    "feature_type",
)


def _nonempty_contract_value(value: object, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} must be non-empty")
    return text


def _registered_sampling_frame_contract(
    sampling_frame_id: object,
    assembly_sampling_frame: object,
) -> tuple[str, str]:
    frame_id = _nonempty_contract_value(sampling_frame_id, "sampling_frame_id")
    assembly_frame = _nonempty_contract_value(
        assembly_sampling_frame, "assembly_sampling_frame"
    )
    registered = REGISTERED_ASSEMBLY_SAMPLING_FRAMES.get(frame_id)
    if registered is None or assembly_frame != registered:
        raise ValueError(
            "Sampling-frame ID and assembly sampling frame are not a registered pair"
        )
    return frame_id, assembly_frame


def _atomic_write_tsv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False, na_rep="")
    os.replace(temporary, path)
    return path


def _atomic_write_json(value: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _strict_bool(values: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        if values.isna().any():
            raise ValueError(f"{label} contains missing booleans")
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError(f"{label} must contain only true/false values")
    return normalized.eq("true")


def _safe_relative(path: Path, repository_root: Path) -> str:
    resolved = path.resolve()
    root = repository_root.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Evidence path is outside the repository: {resolved}") from exc


def _resolve_relative(relative_path: object, repository_root: Path) -> Path:
    text = str(relative_path)
    if not text or "\\" in text or Path(text).is_absolute():
        raise ValueError(f"Evidence relative_path is not canonical: {text!r}")
    resolved = (repository_root.resolve() / text).resolve()
    if _safe_relative(resolved, repository_root) != text:
        raise ValueError(f"Evidence relative_path escapes or is non-canonical: {text!r}")
    return resolved


def _require_sha256(value: object, label: str) -> str:
    digest = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return digest


def _file_record(path: Path, repository_root: Path, **extra: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "relative_path": _safe_relative(path, repository_root),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }
    record.update(extra)
    return record


def _verify_file_record(
    record: Mapping[str, Any],
    repository_root: Path,
    *,
    label: str,
    expected_path: Path | None = None,
) -> Path:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} must be a file record")
    missing = {"relative_path", "bytes", "sha256"}.difference(record)
    if missing:
        raise ValueError(f"{label} is missing file fields: {sorted(missing)}")
    path = _resolve_relative(record["relative_path"], repository_root)
    if expected_path is not None and path != expected_path.resolve():
        raise ValueError(f"{label} path differs from the supplied input")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    try:
        declared_bytes = int(record["bytes"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}.bytes must be an integer") from exc
    if declared_bytes != path.stat().st_size:
        raise ValueError(f"{label} byte size changed")
    digest = _require_sha256(record["sha256"], f"{label}.sha256")
    if digest != sha256_file(path):
        raise ValueError(f"{label} SHA256 changed")
    return path


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def _validate_exact_columns(
    frame: pd.DataFrame, expected: Iterable[str], label: str
) -> None:
    expected_set = set(expected)
    observed_set = set(map(str, frame.columns))
    if observed_set != expected_set:
        raise ValueError(
            f"{label} columns differ from the condition-blind contract: "
            f"missing={sorted(expected_set-observed_set)}, "
            f"unexpected={sorted(observed_set-expected_set)}"
        )


def validate_cell_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the exact anonymous-to-public cell identity bridge."""
    _validate_exact_columns(frame, CELL_MANIFEST_COLUMNS, "Cell manifest")
    result = frame.loc[:, CELL_MANIFEST_COLUMNS].copy()
    for column in CELL_MANIFEST_COLUMNS:
        result[column] = result[column].astype(str).str.strip()
        if result[column].eq("").any():
            raise ValueError(f"Cell manifest field {column!r} must be non-empty")
    if result["annotation_row_id"].duplicated().any():
        raise ValueError("Cell manifest contains duplicate anonymous row IDs")
    if not result["annotation_row_id"].map(lambda value: bool(_ROW_ID_RE.fullmatch(value))).all():
        raise ValueError("Cell manifest anonymous row IDs have an invalid format")
    if result["cell_id"].duplicated().any():
        raise ValueError("Cell manifest contains duplicate cell IDs")
    if result.duplicated(["library_id", "original_barcode"]).any():
        raise ValueError("Cell manifest contains duplicate library/barcode keys")
    expected_ids = (
        PRIMARY_ACCESSION
        + "|"
        + result["library_id"]
        + "|"
        + result["original_barcode"]
    )
    if not result["cell_id"].eq(expected_ids).all():
        raise ValueError("Cell manifest violates the exact accession/library/barcode cell ID contract")
    return result


def validate_prediction_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a prediction-only table with no condition-bearing columns."""
    _validate_exact_columns(frame, PREDICTION_COLUMNS, "Prediction table")
    result = frame.loc[:, PREDICTION_COLUMNS].copy()
    result["annotation_row_id"] = result["annotation_row_id"].astype(str).str.strip()
    result["reference_label"] = result["reference_label"].astype(str).str.strip()
    if result[["annotation_row_id", "reference_label"]].eq("").any().any():
        raise ValueError("Prediction row IDs and reference labels must be non-empty")
    if result["annotation_row_id"].duplicated().any():
        raise ValueError("Prediction table contains duplicate anonymous row IDs")
    if not result["annotation_row_id"].map(lambda value: bool(_ROW_ID_RE.fullmatch(value))).all():
        raise ValueError("Prediction table anonymous row IDs have an invalid format")
    confidence = pd.to_numeric(result["mapping_confidence"], errors="coerce")
    if confidence.isna().any() or (~np.isfinite(confidence.to_numpy(dtype=float))).any():
        raise ValueError("Prediction mapping_confidence must be finite")
    if ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("Prediction mapping_confidence must lie in [0,1]")
    result["mapping_confidence"] = confidence.astype(float)
    result["annotation_uncertain"] = _strict_bool(
        result["annotation_uncertain"], "predictions.annotation_uncertain"
    )
    result["predicted_doublet"] = _strict_bool(
        result["predicted_doublet"], "predictions.predicted_doublet"
    )
    return result


def validate_reference_mapping(
    path: Path,
    *,
    expected_sampling_frame_id: str = SAMPLING_FRAME_ID,
) -> dict[str, Any]:
    """Validate the frozen source-label mapping without reading the model."""
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "schema_name",
        "schema_version",
        "mapping_id",
        "status",
        "frozen_at_utc",
        "outcome_blinded_at_freeze",
        "real_pathway_results_inspected",
        "sampling_frame_id",
        "annotation_plan_id",
        "model",
        "annotation_policy",
        "reference_label_mapping",
        "predicted_doublet_reference_labels",
        "uncertain_reference_labels",
        "lineage_rule",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Reference mapping is missing its frozen contract")
    expected_frame_id = _nonempty_contract_value(
        expected_sampling_frame_id, "expected_sampling_frame_id"
    )
    if (
        value["schema_name"] != "t21_condition_blind_reference_label_mapping"
        or not str(value["schema_version"]).startswith("1.")
        or value["status"] != "frozen"
        or value["outcome_blinded_at_freeze"] is not True
        or value["real_pathway_results_inspected"] is not False
        or value["sampling_frame_id"] != expected_frame_id
    ):
        raise ValueError("Reference mapping is not a frozen, outcome-blind T21 mapping")
    if not str(value["mapping_id"]).strip() or not str(value["annotation_plan_id"]).strip():
        raise ValueError("Reference mapping IDs must be non-empty")

    model = value["model"]
    expected_model_fields = {
        "model_id",
        "expected_file_name",
        "sha256",
        "serialization",
        "expected_reference_labels",
        "adjudicator_loading_policy",
    }
    if not isinstance(model, dict) or set(model) != expected_model_fields:
        raise ValueError("Reference mapping model contract must be a mapping")
    for field in ("model_id", "expected_file_name", "sha256", "serialization"):
        if not str(model.get(field, "")).strip():
            raise ValueError(f"Reference mapping model field {field!r} is empty")
    _require_sha256(model["sha256"], "reference_mapping.model.sha256")
    try:
        expected_reference_labels = int(model["expected_reference_labels"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Reference mapping must pin the model label count") from exc
    if expected_reference_labels < 1:
        raise ValueError("Reference mapping model label count must be positive")
    if model.get("adjudicator_loading_policy") != "hash_only_never_deserialize":
        raise ValueError("Adjudicator may not deserialize the reference model")

    policy = value["annotation_policy"]
    expected_policy_fields = {
        "condition_used_for_annotation",
        "candidate_pathway_genes_used_for_annotation",
        "cell_identifier_used_as_model_feature",
        "library_identifier_used_as_model_feature",
        "model_feature_space",
        "minimum_mapping_confidence",
        "unmapped_reference_label_policy",
        "source_label_preservation",
        "excluded_analysis_cell_type",
        "uncertain_analysis_cell_type",
    }
    if not isinstance(policy, dict) or set(policy) != expected_policy_fields:
        raise ValueError("Reference annotation policy must be a mapping")
    false_fields = (
        "condition_used_for_annotation",
        "candidate_pathway_genes_used_for_annotation",
        "cell_identifier_used_as_model_feature",
        "library_identifier_used_as_model_feature",
    )
    if any(policy.get(field) is not False for field in false_fields):
        raise ValueError("Reference mapping permits a forbidden condition or identifier feature")
    if policy.get("model_feature_space") != "gene_expression_only":
        raise ValueError("Reference mapping model feature space must be gene_expression_only")
    if policy.get("unmapped_reference_label_policy") != "fail_closed":
        raise ValueError("Reference mapping must fail closed on unmapped labels")
    if policy.get("source_label_preservation") != "original_cell_type_must_equal_reference_label":
        raise ValueError("Reference mapping must preserve raw reference labels")
    try:
        threshold = float(policy["minimum_mapping_confidence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Reference mapping confidence threshold is invalid") from exc
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Reference mapping confidence threshold must lie in [0,1]")

    mapping = value["reference_label_mapping"]
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Reference label mapping must be non-empty")
    normalized_mapping = {str(key).strip(): str(mapped).strip() for key, mapped in mapping.items()}
    if any(not key or not mapped for key, mapped in normalized_mapping.items()):
        raise ValueError("Reference label mapping contains an empty label")
    if len(normalized_mapping) != expected_reference_labels:
        raise ValueError("Reference label mapping does not enumerate the pinned model classes")
    for source_label in normalized_mapping:
        lowered = source_label.lower()
        if any(token in lowered for token in _FORBIDDEN_FEATURE_TOKENS):
            raise ValueError("Reference label mapping contains a condition/outcome label")

    doublets = list(map(str, value["predicted_doublet_reference_labels"]))
    uncertain = list(map(str, value["uncertain_reference_labels"]))
    if len(doublets) != len(set(doublets)) or len(uncertain) != len(set(uncertain)):
        raise ValueError("Reference exclusion-label lists must be unique")
    if not set(doublets).issubset(normalized_mapping) or not set(uncertain).issubset(
        normalized_mapping
    ):
        raise ValueError("Reference exclusion-label lists contain unmapped labels")
    uncertain_type = str(policy.get("uncertain_analysis_cell_type", ""))
    sentinel_labels = set(doublets).union(uncertain)
    if any(normalized_mapping[label] != uncertain_type for label in sentinel_labels):
        raise ValueError("Doublet/uncertain source labels must map to the uncertain analysis type")

    lineage = value["lineage_rule"]
    if not isinstance(lineage, dict) or set(lineage) != {
        "include_analysis_cell_types"
    }:
        raise ValueError("Reference mapping lineage rule has unexpected fields")
    included = lineage.get("include_analysis_cell_types") if isinstance(lineage, dict) else None
    if not isinstance(included, list) or not included or len(included) != len(set(included)):
        raise ValueError("Reference mapping lineage types must be unique and non-empty")
    if not set(map(str, included)).issubset(set(normalized_mapping.values())):
        raise ValueError("Reference mapping does not define every included lineage type")
    value["reference_label_mapping"] = normalized_mapping
    return value


def validate_annotation_plan_for_reference(
    path: Path,
    *,
    repository_root: Path,
    mapping_path: Path,
    mapping: Mapping[str, Any],
    expected_sampling_frame_id: str = SAMPLING_FRAME_ID,
) -> dict[str, Any]:
    """Validate the annotation-plan binding to the frozen reference map."""
    plan = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("Annotation plan must be a mapping")
    expected_frame_id = _nonempty_contract_value(
        expected_sampling_frame_id, "expected_sampling_frame_id"
    )
    if mapping.get("sampling_frame_id") != expected_frame_id:
        raise ValueError("Reference map belongs to another sampling frame")
    if (
        plan.get("schema_name") != "t21_cell_annotation_and_lineage_plan"
        or plan.get("status") != "frozen"
        or plan.get("outcome_blinded_at_freeze") is not True
        or plan.get("real_pathway_results_inspected") is not False
        or plan.get("sampling_frame_id") != expected_frame_id
        or plan.get("sampling_frame_id") != mapping.get("sampling_frame_id")
        or plan.get("plan_id") != mapping.get("annotation_plan_id")
    ):
        raise ValueError("Annotation plan and reference map do not share the frozen contract")
    if expected_frame_id == CD235A_NEG_SENSITIVITY_SAMPLING_FRAME_ID and (
        plan.get("analysis_role") != "sensitivity_only"
        or plan.get("pooling_with_primary_allowed") is not False
        or plan.get("primary_discovery_claim_allowed") is not False
    ):
        raise ValueError("CD235a- annotation plan exceeds its sensitivity-only role")
    policy = plan.get("annotation_policy")
    if not isinstance(policy, dict) or any(
        policy.get(field) is not False
        for field in (
            "condition_used_for_annotation",
            "candidate_pathway_genes_used_for_annotation",
        )
    ):
        raise ValueError("Annotation plan permits condition or pathway leakage")
    contracts = plan.get("source_contracts")
    reference = contracts.get("condition_blind_reference") if isinstance(contracts, dict) else None
    if not isinstance(reference, dict):
        raise ValueError("Annotation plan lacks the condition-blind reference source contract")
    expected_relative = _safe_relative(mapping_path, repository_root)
    if (
        reference.get("original_cell_type_source_prefix")
        != "condition_blind_reference_mapping:"
        or reference.get("mapping_schema_name") != mapping.get("schema_name")
        or reference.get("mapping_id") != mapping.get("mapping_id")
        or reference.get("mapping_relative_path") != expected_relative
        or str(reference.get("mapping_sha256", "")).lower() != sha256_file(mapping_path)
        or reference.get("annotation_adjudication_required") is not True
        or reference.get("annotation_adjudication_checksum_sidecar_required") is not True
    ):
        raise ValueError("Annotation plan reference-source hash/path binding changed")
    requirements = plan.get("release_requirements")
    if not isinstance(requirements, dict) or requirements.get(
        "annotation_adjudication_sidecar_required_for_reference_mapping"
    ) is not True:
        raise ValueError("Annotation plan does not require the reference adjudication sidecar")
    plan_included = plan.get("lineage_rule", {}).get("include_analysis_cell_types")
    mapping_included = mapping.get("lineage_rule", {}).get("include_analysis_cell_types")
    if list(map(str, plan_included or [])) != list(map(str, mapping_included or [])):
        raise ValueError("Annotation plan and reference mapping lineage rules differ")
    return plan


def _validate_count_matrix(matrix: Any, label: str) -> Any:
    if sparse.issparse(matrix):
        values = np.asarray(matrix.data)
    else:
        values = np.asarray(matrix)
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{label} counts must be numeric")
    numeric = values.astype(float, copy=False)
    if np.any(~np.isfinite(numeric)) or np.any(numeric < 0):
        raise ValueError(f"{label} counts must be finite and non-negative")
    if not np.array_equal(numeric, np.rint(numeric)):
        raise ValueError(f"{label} counts must be exact integers")
    return matrix


def _validate_blinded_expression_h5ad(path: Path) -> dict[str, Any]:
    """Inspect an exported model input and reject any metadata leakage."""
    value = ad.read_h5ad(path, backed="r")
    try:
        if list(value.obs.columns) != ["annotation_row_id"]:
            raise ValueError("Blinded expression H5AD contains non-anonymous obs metadata")
        row_ids = value.obs["annotation_row_id"].astype(str)
        if not value.obs_names.astype(str).equals(pd.Index(row_ids)):
            raise ValueError("Blinded expression H5AD obs_names differ from anonymous row IDs")
        if row_ids.duplicated().any() or not row_ids.map(
            lambda item: bool(_ROW_ID_RE.fullmatch(item))
        ).all():
            raise ValueError("Blinded expression H5AD contains invalid anonymous row IDs")
        if tuple(value.var.columns) != _ALLOWED_BLINDED_VAR_COLUMNS:
            raise ValueError("Blinded expression H5AD contains an unapproved feature annotation")
        if any(
            len(collection)
            for collection in (value.obsm, value.varm, value.obsp, value.varp, value.layers)
        ):
            raise ValueError("Blinded expression H5AD contains auxiliary matrices or metadata")
        expected_uns = {"t21_condition_blind_annotation_input"}
        if set(value.uns) != expected_uns:
            raise ValueError("Blinded expression H5AD contains unexpected unstructured metadata")
        uns = value.uns["t21_condition_blind_annotation_input"]
        if not isinstance(uns, Mapping) or (
            uns.get("schema_name") != "t21_condition_blind_expression_input"
            or uns.get("condition_metadata_present") is not False
            or uns.get("pathway_outcomes_present") is not False
            or uns.get("cell_identifiers_anonymous") is not True
        ):
            raise ValueError("Blinded expression H5AD has an invalid blind-state record")
        return {
            "n_cells": int(value.n_obs),
            "n_genes": int(value.n_vars),
            "row_ids": row_ids.tolist(),
        }
    finally:
        if getattr(value, "file", None) is not None:
            value.file.close()


def _validate_export_assembly_spec(
    path: Path,
    *,
    source_records: Sequence[Mapping[str, Any]],
    expected_sampling_frame: str = PRIMARY_ASSEMBLY_SAMPLING_FRAME,
) -> pd.DataFrame:
    """Bind a blinded export to one exact, immutable assembly spec."""
    expected_frame = _nonempty_contract_value(
        expected_sampling_frame, "expected_sampling_frame"
    )
    spec = _read_tsv(path)
    expected_columns = {
        "library_order",
        "library_id",
        "sampling_frame",
        "tissue",
        "shard_sha256",
        "include",
        "include_reason",
    }
    _validate_exact_columns(spec, expected_columns, "Assembly spec")
    spec["include"] = _strict_bool(spec["include"], "assembly_spec.include")
    if not spec["include"].all():
        raise ValueError("Annotation export assembly spec may not contain excluded rows")
    order = pd.to_numeric(spec["library_order"], errors="coerce").to_numpy()
    if not np.array_equal(order, np.arange(len(spec))):
        raise ValueError("Annotation export assembly order must be contiguous from zero")
    if (
        spec["library_id"].astype(str).duplicated().any()
        or not spec["sampling_frame"].astype(str).eq(expected_frame).all()
        or not spec["tissue"].astype(str).eq("liver").all()
        or spec["include_reason"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError(
            "Annotation export assembly spec is not the locked "
            f"{expected_frame!r} frame"
        )
    if len(spec) != len(source_records):
        raise ValueError("Annotation export shard count differs from the assembly spec")
    for index, (row, source) in enumerate(zip(spec.itertuples(index=False), source_records)):
        declared_hash = _require_sha256(
            getattr(row, "shard_sha256"), f"assembly_spec[{index}].shard_sha256"
        )
        if (
            int(getattr(row, "library_order")) != index
            or str(getattr(row, "library_id")) != str(source["library_id"])
            or declared_hash != str(source["sha256"])
            or int(source["shard_order"]) != index
        ):
            raise ValueError(
                "Annotation export source shard order/hash differs from the assembly spec"
            )
    return spec


def export_condition_blind_inputs(
    *,
    repository_root: Path,
    shard_paths: Sequence[Path],
    output_dir: Path,
    private_output_dir: Path,
    assembly_spec_path: Path,
    expected_libraries: int | None = None,
    blinding_key: bytes | None = None,
    sampling_frame_id: str = SAMPLING_FRAME_ID,
    assembly_sampling_frame: str = PRIMARY_ASSEMBLY_SAMPLING_FRAME,
) -> dict[str, Any]:
    """Export condition-stripped model inputs and an exact private join bridge.

    The returned H5AD files contain only expression, three feature columns and
    anonymous row IDs. The cell manifest and blinding record are evidence
    artifacts and must not be supplied as model features.  The public predictor
    input directory and private evidence directory are disjoint and non-nested,
    so passing the former to the predictor cannot accidentally expose the cell
    manifest through an adjacent file scan.
    """
    root = repository_root.resolve()
    frame_id, assembly_frame = _registered_sampling_frame_contract(
        sampling_frame_id,
        assembly_sampling_frame,
    )
    paths = [path.resolve() for path in shard_paths]
    if not paths:
        raise ValueError("At least one count shard is required")
    if expected_libraries is not None and len(paths) != int(expected_libraries):
        raise ValueError(
            f"Expected {expected_libraries} count shards, observed {len(paths)}"
        )
    if len(paths) != len(set(paths)):
        raise ValueError("Condition-blind export received duplicate shard paths")
    for path in paths:
        _safe_relative(path, root)
        if not path.is_file():
            raise FileNotFoundError(path)
    target = output_dir.resolve()
    private_target = private_output_dir.resolve()
    assembly_spec = assembly_spec_path.resolve()
    for evidence_path in (target, private_target, assembly_spec):
        _safe_relative(evidence_path, root)
    if not assembly_spec.is_file():
        raise FileNotFoundError(assembly_spec)
    if (
        target == private_target
        or target in private_target.parents
        or private_target in target.parents
    ):
        raise ValueError("Predictor inputs and private annotation evidence must be disjoint")
    for immutable_target in (target, private_target):
        if immutable_target.exists():
            raise FileExistsError(
                f"Immutable annotation export directory exists: {immutable_target}"
            )
    key = blinding_key if blinding_key is not None else secrets.token_bytes(32)
    if not isinstance(key, bytes) or len(key) < 16:
        raise ValueError("Blinding key must contain at least 16 bytes")
    target.parent.mkdir(parents=True, exist_ok=True)
    private_target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".tmp-{target.name}-{secrets.token_hex(8)}"
    private_temporary = private_target.parent / (
        f".tmp-{private_target.name}-{secrets.token_hex(8)}"
    )
    temporary.mkdir(parents=False, exist_ok=False)
    private_temporary.mkdir(parents=False, exist_ok=False)
    blind_root = temporary

    source_records: list[dict[str, Any]] = []
    blinded_records: list[dict[str, Any]] = []
    manifest_parts: list[pd.DataFrame] = []
    seen_libraries: set[str] = set()
    reference_var_names: np.ndarray | None = None
    reference_var_frame: pd.DataFrame | None = None
    try:
        for shard_order, shard_path in enumerate(paths):
            shard = ad.read_h5ad(shard_path)
            required_obs = {"cell_id", "library_id", "original_barcode"}
            if required_obs.difference(shard.obs.columns):
                raise ValueError(f"Count shard lacks exact cell keys: {shard_path}")
            if not shard.obs_names.astype(str).equals(
                pd.Index(shard.obs["cell_id"].astype(str))
            ):
                raise ValueError("Count shard obs_names differ from its exact cell_id column")
            libraries = shard.obs["library_id"].astype(str)
            if libraries.nunique() != 1:
                raise ValueError("Each count shard must contain exactly one library")
            library_id = str(libraries.iloc[0])
            if library_id in seen_libraries:
                raise ValueError(f"Duplicate library across count shards: {library_id}")
            seen_libraries.add(library_id)
            cells = pd.DataFrame(
                {
                    "cell_id": shard.obs["cell_id"].astype(str).to_numpy(),
                    "library_id": libraries.to_numpy(),
                    "original_barcode": shard.obs["original_barcode"].astype(str).to_numpy(),
                }
            )
            row_ids = [
                "ar_"
                + hmac.new(key, cell_id.encode("utf-8"), "sha256").hexdigest()[:32]
                for cell_id in cells["cell_id"]
            ]
            cells.insert(0, "annotation_row_id", row_ids)
            cells = validate_cell_manifest(cells)
            manifest_parts.append(cells)

            var_names = shard.var_names.astype(str).to_numpy()
            if reference_var_names is None:
                reference_var_names = var_names.copy()
            elif not np.array_equal(reference_var_names, var_names):
                raise ValueError("Count shards do not have identical ordered gene axes")
            missing_var = set(_ALLOWED_BLINDED_VAR_COLUMNS).difference(shard.var.columns)
            if missing_var:
                raise ValueError(f"Count shard lacks feature fields: {sorted(missing_var)}")
            selected_var = shard.var.loc[:, _ALLOWED_BLINDED_VAR_COLUMNS].copy()
            if selected_var.isna().any().any():
                raise ValueError("Count shard contains an empty model feature annotation")
            selected_var = selected_var.astype(str).apply(lambda column: column.str.strip())
            if selected_var.eq("").any().any():
                raise ValueError("Count shard contains an empty model feature annotation")
            if reference_var_frame is None:
                reference_var_frame = selected_var.copy()
            elif not selected_var.equals(reference_var_frame):
                raise ValueError("Count shards have different ordered feature annotations")
            matrix = shard.layers["counts"] if "counts" in shard.layers else shard.X
            _validate_count_matrix(matrix, str(shard_path))
            blind_obs = pd.DataFrame(
                {"annotation_row_id": row_ids},
                index=pd.Index(row_ids, name="annotation_row_id"),
            )
            blind_var = shard.var.loc[:, _ALLOWED_BLINDED_VAR_COLUMNS].copy()
            blind = ad.AnnData(X=matrix.copy(), obs=blind_obs, var=blind_var)
            blind.uns["t21_condition_blind_annotation_input"] = {
                "schema_name": "t21_condition_blind_expression_input",
                "schema_version": "1.0.0",
                "annotation_shard_id": f"annotation_shard_{shard_order:03d}",
                "condition_metadata_present": False,
                "pathway_outcomes_present": False,
                "cell_identifiers_anonymous": True,
            }
            blind_path = blind_root / f"annotation_shard_{shard_order:03d}.h5ad"
            blind.write_h5ad(blind_path, compression="gzip")
            inspected = _validate_blinded_expression_h5ad(blind_path)
            if inspected["row_ids"] != row_ids:
                raise ValueError("Blinded expression row order changed while writing")
            source_records.append(
                _file_record(
                    shard_path,
                    root,
                    shard_order=shard_order,
                    library_id=library_id,
                    n_cells=int(shard.n_obs),
                )
            )
            final_blind_path = target / blind_path.name
            blinded_record = _file_record(
                blind_path,
                temporary,
                role="condition_blind_expression_h5ad",
                shard_order=shard_order,
                n_cells=int(shard.n_obs),
                n_genes=int(shard.n_vars),
            )
            blinded_record["relative_path"] = _safe_relative(final_blind_path, root)
            blinded_records.append(blinded_record)

        manifest = validate_cell_manifest(pd.concat(manifest_parts, ignore_index=True))
        _validate_export_assembly_spec(
            assembly_spec,
            source_records=source_records,
            expected_sampling_frame=assembly_frame,
        )
        manifest_path = private_temporary / CELL_MANIFEST_NAME
        _atomic_write_tsv(manifest, manifest_path)
        final_manifest_path = private_target / CELL_MANIFEST_NAME
        manifest_record = _file_record(manifest_path, private_temporary)
        manifest_record["relative_path"] = _safe_relative(final_manifest_path, root)
        record: dict[str, Any] = {
            "schema_name": "t21_condition_blind_annotation_input_record",
            "schema_version": "1.1.0",
            "created_at_utc": utc_now(),
            "sampling_frame_id": frame_id,
            "outcome_blinded": True,
            "real_pathway_results_inspected": False,
            "condition_metadata_exported_to_model": False,
            "candidate_pathway_results_exported_to_model": False,
            "blinding": {
                "anonymous_id_method": "HMAC-SHA256_truncated_128bit",
                "blinding_key_sha256": sha256(key).hexdigest(),
                "blinding_key_exported_to_predictor": False,
                "cell_manifest_exported_to_predictor": False,
            },
            "assembly_spec": _file_record(assembly_spec, root),
            "source_shards": source_records,
            "outputs": {
                "cell_manifest": manifest_record,
                "expression_inputs": blinded_records,
            },
            "summary": {
                "n_libraries": len(source_records),
                "n_cells": len(manifest),
                "n_genes": int(
                    len(reference_var_names) if reference_var_names is not None else 0
                ),
                "n_duplicate_cell_ids": 0,
                "n_duplicate_library_barcodes": 0,
            },
        }
        record_path = private_temporary / BLINDING_RECORD_NAME
        _atomic_write_json(record, record_path)
        os.replace(temporary, target)
        os.replace(private_temporary, private_target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if private_temporary.exists():
            shutil.rmtree(private_temporary)
        if target.exists() and not private_target.exists():
            shutil.rmtree(target)
        raise
    return record


def validate_blinding_record(
    path: Path,
    *,
    repository_root: Path,
    expected_sampling_frame_id: str = SAMPLING_FRAME_ID,
    expected_assembly_sampling_frame: str = PRIMARY_ASSEMBLY_SAMPLING_FRAME,
) -> dict[str, Any]:
    """Validate the condition-stripped export and all of its hash bindings."""
    frame_id, assembly_frame = _registered_sampling_frame_contract(
        expected_sampling_frame_id,
        expected_assembly_sampling_frame,
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    expected_top_level = {
        "schema_name",
        "schema_version",
        "created_at_utc",
        "sampling_frame_id",
        "outcome_blinded",
        "real_pathway_results_inspected",
        "condition_metadata_exported_to_model",
        "candidate_pathway_results_exported_to_model",
        "blinding",
        "assembly_spec",
        "source_shards",
        "outputs",
        "summary",
    }
    if not isinstance(value, dict) or set(value) != expected_top_level or (
        value.get("schema_name") != "t21_condition_blind_annotation_input_record"
        or not str(value.get("schema_version", "")).startswith("1.")
        or value.get("sampling_frame_id") != frame_id
        or value.get("outcome_blinded") is not True
        or value.get("real_pathway_results_inspected") is not False
        or value.get("condition_metadata_exported_to_model") is not False
        or value.get("candidate_pathway_results_exported_to_model") is not False
    ):
        raise ValueError("Annotation blinding record has an invalid blind-state contract")
    blinding = value.get("blinding")
    if not isinstance(blinding, dict) or set(blinding) != {
        "anonymous_id_method",
        "blinding_key_sha256",
        "blinding_key_exported_to_predictor",
        "cell_manifest_exported_to_predictor",
    } or (
        blinding.get("anonymous_id_method") != "HMAC-SHA256_truncated_128bit"
        or blinding.get("blinding_key_exported_to_predictor") is not False
        or blinding.get("cell_manifest_exported_to_predictor") is not False
    ):
        raise ValueError("Annotation blinding record permits an identity leak")
    _require_sha256(blinding.get("blinding_key_sha256"), "blinding_key_sha256")

    assembly_spec_path = _verify_file_record(
        value.get("assembly_spec", {}),
        repository_root,
        label="annotation export assembly spec",
    )
    if set(value["assembly_spec"]) != {"relative_path", "bytes", "sha256"}:
        raise ValueError("Annotation export assembly-spec record has unexpected fields")

    outputs = value.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {
        "cell_manifest",
        "expression_inputs",
    }:
        raise ValueError("Annotation blinding record omits outputs")
    manifest_path = _verify_file_record(
        outputs.get("cell_manifest", {}),
        repository_root,
        label="blinding cell manifest",
    )
    if set(outputs["cell_manifest"]) != {"relative_path", "bytes", "sha256"}:
        raise ValueError("Annotation blinding cell-manifest record has unexpected fields")
    manifest = validate_cell_manifest(_read_tsv(manifest_path))
    expression_records = outputs.get("expression_inputs")
    if not isinstance(expression_records, list) or not expression_records:
        raise ValueError("Annotation blinding record has no expression inputs")
    observed_row_ids: list[str] = []
    seen_paths: set[Path] = set()
    expression_parents: set[Path] = set()
    for index, record in enumerate(expression_records):
        if not isinstance(record, dict) or set(record) != {
            "relative_path",
            "bytes",
            "sha256",
            "role",
            "shard_order",
            "n_cells",
            "n_genes",
        }:
            raise ValueError("Annotation blinding expression record has unexpected fields")
        expression_path = _verify_file_record(
            record,
            repository_root,
            label=f"blinded expression input {index}",
        )
        if expression_path in seen_paths:
            raise ValueError("Annotation blinding record repeats an expression input")
        seen_paths.add(expression_path)
        expression_parents.add(expression_path.parent)
        if record.get("role") != "condition_blind_expression_h5ad":
            raise ValueError("Annotation blinding record contains an unapproved input role")
        inspected = _validate_blinded_expression_h5ad(expression_path)
        if int(record.get("n_cells", -1)) != inspected["n_cells"] or int(
            record.get("n_genes", -1)
        ) != inspected["n_genes"]:
            raise ValueError("Blinded expression dimensions differ from their record")
        observed_row_ids.extend(inspected["row_ids"])
    if observed_row_ids != manifest["annotation_row_id"].tolist():
        raise ValueError("Blinded expression row IDs/order differ from the cell manifest")
    if len(expression_parents) != 1:
        raise ValueError("Blinded expression inputs must occupy one isolated directory")
    expression_parent = next(iter(expression_parents))
    manifest_parent = manifest_path.parent
    if (
        expression_parent == manifest_parent
        or expression_parent in manifest_parent.parents
        or manifest_parent in expression_parent.parents
    ):
        raise ValueError("Private cell manifest is not isolated from predictor inputs")
    input_entries = set(expression_parent.iterdir())
    if input_entries != seen_paths:
        raise ValueError("Predictor input directory contains an unregistered file")

    source_records = value.get("source_shards")
    if not isinstance(source_records, list) or len(source_records) != len(expression_records):
        raise ValueError("Annotation blinding record source/output shard counts differ")
    source_libraries: set[str] = set()
    for index, record in enumerate(source_records):
        _verify_file_record(
            record,
            repository_root,
            label=f"source count shard {index}",
        )
        library_id = str(record.get("library_id", ""))
        if not library_id or library_id in source_libraries:
            raise ValueError("Annotation blinding source library IDs must be unique")
        source_libraries.add(library_id)
        if int(record.get("shard_order", -1)) != index:
            raise ValueError("Annotation blinding source shard order is not contiguous")
    _validate_export_assembly_spec(
        assembly_spec_path,
        source_records=source_records,
        expected_sampling_frame=assembly_frame,
    )
    summary = value.get("summary")
    if not isinstance(summary, dict) or (
        int(summary.get("n_libraries", -1)) != len(expression_records)
        or int(summary.get("n_cells", -1)) != len(manifest)
        or int(summary.get("n_duplicate_cell_ids", -1)) != 0
        or int(summary.get("n_duplicate_library_barcodes", -1)) != 0
    ):
        raise ValueError("Annotation blinding summary does not close")
    return value


def _stable_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def validate_celltypist_prediction_plan(
    path: Path,
    *,
    repository_root: Path,
    expected_mapping_path: Path | None = None,
    expected_sampling_frame_id: str = SAMPLING_FRAME_ID,
) -> dict[str, Any]:
    """Validate the frozen outcome-blind predictor method and hash bindings."""
    frame_id = _nonempty_contract_value(
        expected_sampling_frame_id, "expected_sampling_frame_id"
    )
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected_top = {
        "schema_name",
        "schema_version",
        "plan_id",
        "status",
        "frozen_at_utc",
        "outcome_blinded_at_freeze",
        "real_pathway_results_inspected",
        "sampling_frame_id",
        "input_contract",
        "model",
        "software_contract",
        "prediction",
        "provenance_contract",
    }
    if not isinstance(value, dict) or set(value) != expected_top or (
        value.get("schema_name") != PREDICTION_PLAN_SCHEMA_NAME
        or not str(value.get("schema_version", "")).startswith("1.")
        or value.get("status") != "frozen"
        or value.get("outcome_blinded_at_freeze") is not True
        or value.get("real_pathway_results_inspected") is not False
        or value.get("sampling_frame_id") != frame_id
        or not str(value.get("plan_id", "")).strip()
    ):
        raise ValueError("CellTypist prediction plan is not frozen and outcome-blind")
    inputs = value.get("input_contract")
    required_inputs = {
        "file_name_pattern",
        "matrix",
        "matrix_semantics",
        "anonymous_cell_identifier",
        "feature_identifier",
        "gene_symbol_normalization",
        "empty_gene_symbol_policy",
        "duplicate_gene_symbol_policy",
        "duplicate_collapse_total_count_policy",
        "normalization",
        "normalization_target_sum",
        "zero_count_cell_policy",
    }
    if not isinstance(inputs, dict) or set(inputs) != required_inputs or any(
        inputs.get(field) != expected
        for field, expected in {
            "matrix": "X",
            "matrix_semantics": "raw_nonnegative_integer_counts",
            "anonymous_cell_identifier": "annotation_row_id",
            "feature_identifier": "gene_symbol",
            "gene_symbol_normalization": "strip_surrounding_whitespace_only",
            "empty_gene_symbol_policy": "fail_closed",
            "duplicate_gene_symbol_policy": (
                "sum_raw_counts_preserve_first_symbol_order"
            ),
            "duplicate_collapse_total_count_policy": "exact_conservation",
            "normalization": "normalize_total_target_sum_10000_then_log1p",
            "normalization_target_sum": 10_000,
            "zero_count_cell_policy": "fail_closed",
        }.items()
    ):
        raise ValueError("CellTypist input preprocessing differs from the frozen plan")
    try:
        re.compile(str(inputs["file_name_pattern"]))
    except re.error as exc:
        raise ValueError("CellTypist input filename pattern is invalid") from exc

    model = value.get("model")
    required_model = {
        "model_id",
        "expected_file_name",
        "serialization",
        "sha256",
        "expected_model_features",
        "feature_match",
        "feature_overlap_policy",
        "required_feature_overlap",
    }
    if not isinstance(model, dict) or set(model) != required_model or (
        model.get("serialization") != "pickle"
        or int(model.get("expected_model_features", -1))
        != EXPECTED_CELLTYPIST_MODEL_FEATURES
        or model.get("feature_match") != "exact_case_sensitive_gene_symbol"
        or model.get("feature_overlap_policy") != "require_all_model_features"
        or model.get("required_feature_overlap") != "6971/6971"
    ):
        raise ValueError("CellTypist model feature contract differs from the frozen plan")
    _require_sha256(model.get("sha256"), "prediction_plan.model.sha256")

    software_contract = value.get("software_contract")
    if not isinstance(software_contract, dict) or set(software_contract) != {
        "celltypist_version",
        "scikit_learn_version_recorded",
        "anndata_version_recorded",
    } or (
        software_contract.get("celltypist_version") != "1.7.1"
        or software_contract.get("scikit_learn_version_recorded") is not True
        or software_contract.get("anndata_version_recorded") is not True
    ):
        raise ValueError("CellTypist software contract is not frozen to version 1.7.1")

    prediction = value.get("prediction")
    required_prediction = {
        "annotation_mode",
        "celltypist_api_mode",
        "majority_voting",
        "model_input_matrix",
        "mapping_confidence",
        "minimum_mapping_confidence",
        "annotation_uncertain_rule",
        "predicted_doublet_rule",
        "predicted_doublet_reference_labels",
        "uncertain_reference_labels",
    }
    if not isinstance(prediction, dict) or set(prediction) != required_prediction or (
        prediction.get("annotation_mode") != "best_match"
        or prediction.get("celltypist_api_mode") != "best match"
        or prediction.get("majority_voting") is not False
        or prediction.get("model_input_matrix") != "X"
        or prediction.get("mapping_confidence") != "predicted_class_probability"
        or float(prediction.get("minimum_mapping_confidence", -1)) != 0.5
        or prediction.get("annotation_uncertain_rule")
        != "confidence_below_threshold_or_predeclared_uncertain_label"
        or prediction.get("predicted_doublet_rule") != "predeclared_doublet_label"
    ):
        raise ValueError("CellTypist prediction settings differ from the frozen plan")
    doublets = list(map(str, prediction["predicted_doublet_reference_labels"]))
    uncertain = list(map(str, prediction["uncertain_reference_labels"]))
    if (
        not doublets
        or len(doublets) != len(set(doublets))
        or len(uncertain) != len(set(uncertain))
        or set(doublets).intersection(uncertain)
    ):
        raise ValueError("CellTypist doublet/uncertain rules are not unique and disjoint")

    provenance = value.get("provenance_contract")
    required_provenance = {
        "reference_mapping_relative_path",
        "reference_mapping_sha256",
        "runner_code_sha256_required",
        "prediction_plan_sha256_required",
        "celltypist_version_required",
        "sklearn_version_required",
        "anndata_version_required",
        "per_shard_duplicate_collapse_audit_required",
        "per_shard_exact_feature_overlap_required",
        "strict_five_column_prediction_table",
    }
    if not isinstance(provenance, dict) or set(provenance) != required_provenance:
        raise ValueError("CellTypist prediction provenance contract differs")
    boolean_fields = required_provenance.difference(
        {"reference_mapping_relative_path", "reference_mapping_sha256"}
    )
    if any(provenance.get(field) is not True for field in boolean_fields):
        raise ValueError("CellTypist prediction provenance contract is incomplete")
    mapping_path = _resolve_relative(
        provenance.get("reference_mapping_relative_path"), repository_root
    )
    if expected_mapping_path is not None and mapping_path != expected_mapping_path.resolve():
        raise ValueError("CellTypist prediction plan references another label mapping")
    if not mapping_path.is_file() or sha256_file(mapping_path) != _require_sha256(
        provenance.get("reference_mapping_sha256"),
        "prediction_plan.reference_mapping_sha256",
    ):
        raise ValueError("CellTypist prediction plan label-mapping hash differs")
    return value


def validate_celltypist_feature_audit(
    path: Path,
    *,
    repository_root: Path,
    plan_path: Path,
    runner_script_path: Path,
    model_path: Path,
    expression_records: Sequence[Mapping[str, Any]],
    expected_sampling_frame_id: str = SAMPLING_FRAME_ID,
) -> dict[str, Any]:
    """Recompute feature/collapse facts without deserializing the model pickle."""
    plan = validate_celltypist_prediction_plan(
        plan_path,
        repository_root=repository_root,
        expected_sampling_frame_id=expected_sampling_frame_id,
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    expected_top = {
        "schema_name",
        "schema_version",
        "plan_id",
        "runner_script_sha256",
        "prediction_plan_sha256",
        "model_id",
        "model_sha256",
        "model_features_ordered_sha256",
        "model_features",
        "inputs",
        "summary",
    }
    if not isinstance(value, dict) or set(value) != expected_top or (
        value.get("schema_name") != FEATURE_AUDIT_SCHEMA_NAME
        or not str(value.get("schema_version", "")).startswith("1.")
        or value.get("plan_id") != plan["plan_id"]
        or value.get("runner_script_sha256") != sha256_file(runner_script_path)
        or value.get("prediction_plan_sha256") != sha256_file(plan_path)
        or value.get("model_id") != plan["model"]["model_id"]
        or value.get("model_sha256") != sha256_file(model_path)
    ):
        raise ValueError("CellTypist feature audit hash/plan/model bindings differ")
    model_features = list(map(str, value.get("model_features", [])))
    if (
        len(model_features) != EXPECTED_CELLTYPIST_MODEL_FEATURES
        or len(set(model_features)) != EXPECTED_CELLTYPIST_MODEL_FEATURES
        or any(not feature.strip() for feature in model_features)
        or value.get("model_features_ordered_sha256")
        != _stable_sha256(model_features)
    ):
        raise ValueError("CellTypist feature audit does not enumerate 6971 unique features")
    input_audits = value.get("inputs")
    if not isinstance(input_audits, list) or len(input_audits) != len(expression_records):
        raise ValueError("CellTypist feature-audit input set differs")
    expected_fields = {
        "file_name",
        "bytes",
        "sha256",
        "n_cells",
        "n_features_original",
        "n_gene_symbols_unique",
        "n_duplicate_symbol_groups",
        "n_duplicate_feature_rows_collapsed",
        "raw_count_sum",
        "collapsed_count_sum",
        "n_model_features_total",
        "n_model_features_matched",
        "n_model_features_missing",
        "missing_model_features",
        "feature_overlap",
    }
    aggregate = {
        "n_cells": 0,
        "n_duplicate_symbol_groups": 0,
        "n_duplicate_feature_rows_collapsed": 0,
        "raw_count_sum": 0,
        "collapsed_count_sum": 0,
    }
    for index, (audit, expression_record) in enumerate(
        zip(input_audits, expression_records)
    ):
        if not isinstance(audit, dict) or set(audit) != expected_fields:
            raise ValueError("CellTypist per-shard feature audit fields differ")
        expression_path = _verify_file_record(
            expression_record,
            repository_root,
            label=f"feature-audit expression input {index}",
        )
        if (
            audit["file_name"] != expression_path.name
            or int(audit["bytes"]) != expression_path.stat().st_size
            or audit["sha256"] != sha256_file(expression_path)
            or int(audit["n_model_features_total"])
            != EXPECTED_CELLTYPIST_MODEL_FEATURES
            or int(audit["n_model_features_matched"])
            != EXPECTED_CELLTYPIST_MODEL_FEATURES
            or int(audit["n_model_features_missing"]) != 0
            or audit["missing_model_features"] != []
            or audit["feature_overlap"] != "6971/6971"
            or int(audit["raw_count_sum"]) != int(audit["collapsed_count_sum"])
        ):
            raise ValueError("CellTypist per-shard feature overlap/collapse audit differs")
        source = ad.read_h5ad(expression_path, backed="r")
        try:
            symbols = source.var["gene_symbol"].astype(str).str.strip()
            counts = symbols.value_counts(sort=False)
            recomputed = {
                "n_cells": int(source.n_obs),
                "n_features_original": int(source.n_vars),
                "n_gene_symbols_unique": int(symbols.nunique()),
                "n_duplicate_symbol_groups": int(counts.gt(1).sum()),
                "n_duplicate_feature_rows_collapsed": int(
                    source.n_vars - symbols.nunique()
                ),
            }
            if symbols.eq("").any() or not set(model_features).issubset(set(symbols)):
                raise ValueError("CellTypist audited model features are absent from an input")
            for field, expected in recomputed.items():
                if int(audit[field]) != expected:
                    raise ValueError(
                        f"CellTypist feature audit field {field!r} differs for shard {index}"
                    )
        finally:
            if getattr(source, "file", None) is not None:
                source.file.close()
        for field in aggregate:
            aggregate[field] += int(audit[field])
    summary = value.get("summary")
    expected_summary = {
        "n_expression_inputs",
        "n_cells",
        "n_model_features_total",
        "n_model_features_matched_per_shard",
        "n_model_features_missing_per_shard",
        "required_feature_overlap",
        "n_duplicate_symbol_groups",
        "n_duplicate_feature_rows_collapsed",
        "raw_count_sum",
        "collapsed_count_sum",
    }
    if not isinstance(summary, dict) or set(summary) != expected_summary or (
        int(summary.get("n_expression_inputs", -1)) != len(expression_records)
        or int(summary.get("n_model_features_total", -1))
        != EXPECTED_CELLTYPIST_MODEL_FEATURES
        or int(summary.get("n_model_features_matched_per_shard", -1))
        != EXPECTED_CELLTYPIST_MODEL_FEATURES
        or int(summary.get("n_model_features_missing_per_shard", -1)) != 0
        or summary.get("required_feature_overlap") != "6971/6971"
        or any(int(summary.get(field, -1)) != total for field, total in aggregate.items())
    ):
        raise ValueError("CellTypist feature-audit summary does not close")
    return value


_PREDICTION_RUN_KEYS = {
    "schema_name",
    "schema_version",
    "run_id",
    "created_at_utc",
    "sampling_frame_id",
    "outcome_blinded",
    "real_pathway_results_inspected",
    "condition_used_for_annotation",
    "candidate_pathway_genes_used_for_annotation",
    "cell_identifier_used_as_feature",
    "library_identifier_used_as_feature",
    "cell_manifest_used_for_join_only_after_prediction",
    "model_feature_space",
    "blinding_record",
    "cell_manifest",
    "expression_inputs",
    "model",
    "prediction_plan",
    "runner_script",
    "feature_audit",
    "prediction_table",
    "software",
    "runtime",
}


def _validate_prediction_software_runtime(
    software: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    runner_script_sha256: str,
    prediction_plan_sha256: str,
) -> None:
    required_software = {
        "name",
        "version",
        "celltypist_version",
        "sklearn_version",
        "anndata_version",
        "annotation_mode",
        "majority_voting",
        "input_normalization",
        "normalization_target_sum",
        "model_input_matrix",
        "feature_identifier",
        "duplicate_gene_symbol_policy",
        "mapping_confidence",
        "runner_script_sha256",
        "prediction_plan_sha256",
    }
    if not isinstance(software, Mapping) or set(software) != required_software:
        raise ValueError("Prediction software/method settings are incomplete")
    if (
        software.get("name") != "CellTypist"
        or software.get("version") != "1.7.1"
        or software.get("version") != software.get("celltypist_version")
        or any(
            not str(software.get(field, "")).strip()
            for field in ("celltypist_version", "sklearn_version", "anndata_version")
        )
    ):
        raise ValueError("CellTypist/sklearn/anndata versions must be recorded")
    if (
        software.get("annotation_mode") != "best_match"
        or software.get("majority_voting") is not False
        or software.get("input_normalization")
        != "normalize_total_target_sum_10000_then_log1p"
        or float(software.get("normalization_target_sum", -1)) != 10_000.0
        or software.get("model_input_matrix") != "X"
        or software.get("feature_identifier") != "gene_symbol"
        or software.get("duplicate_gene_symbol_policy")
        != "sum_raw_counts_preserve_first_symbol_order"
        or software.get("mapping_confidence") != "predicted_class_probability"
        or software.get("runner_script_sha256") != runner_script_sha256
        or software.get("prediction_plan_sha256") != prediction_plan_sha256
    ):
        raise ValueError("Prediction software settings differ from the frozen method")
    required_runtime = {
        "python_version",
        "platform",
        "implementation",
        "python_executable_sha256",
        "n_expression_inputs",
        "n_prediction_rows",
    }
    if not isinstance(runtime, Mapping) or set(runtime) != required_runtime:
        raise ValueError("Prediction runtime must be a mapping")
    for field in ("python_version", "platform", "implementation"):
        if not str(runtime.get(field, "")).strip():
            raise ValueError(f"Prediction runtime field {field!r} must be recorded")
    _require_sha256(
        runtime.get("python_executable_sha256"),
        "prediction runtime python_executable_sha256",
    )
    for field in ("n_expression_inputs", "n_prediction_rows"):
        try:
            value = int(runtime[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Prediction runtime field {field!r} must be an integer") from exc
        if value < 1 or value != runtime[field]:
            raise ValueError(f"Prediction runtime field {field!r} must be positive")


def register_prediction_run(
    *,
    repository_root: Path,
    blinding_record_path: Path,
    prediction_path: Path,
    model_path: Path,
    prediction_plan_path: Path,
    runner_script_path: Path,
    feature_audit_path: Path,
    output_path: Path,
    run_id: str,
    model_id: str,
    model_serialization: str,
    software: Mapping[str, Any],
    runtime: Mapping[str, Any],
    attest_condition_blind: bool,
    attest_no_pathway_outcomes: bool,
    attest_identifiers_not_features: bool,
    attest_cell_manifest_postprediction_only: bool,
    sampling_frame_id: str = SAMPLING_FRAME_ID,
    assembly_sampling_frame: str = PRIMARY_ASSEMBLY_SAMPLING_FRAME,
) -> dict[str, Any]:
    """Register externally produced predictions with explicit blind-state attestations."""
    if not all(
        (
            attest_condition_blind,
            attest_no_pathway_outcomes,
            attest_identifiers_not_features,
            attest_cell_manifest_postprediction_only,
        )
    ):
        raise ValueError("All condition-blind prediction attestations are required")
    frame_id, assembly_frame = _registered_sampling_frame_contract(
        sampling_frame_id,
        assembly_sampling_frame,
    )
    root = repository_root.resolve()
    blinding_path = blinding_record_path.resolve()
    predictions_path = prediction_path.resolve()
    reference_model_path = model_path.resolve()
    frozen_prediction_plan_path = prediction_plan_path.resolve()
    predictor_runner_path = runner_script_path.resolve()
    predictor_feature_audit_path = feature_audit_path.resolve()
    target = output_path.resolve()
    for path in (
        blinding_path,
        predictions_path,
        reference_model_path,
        frozen_prediction_plan_path,
        predictor_runner_path,
        predictor_feature_audit_path,
        target,
    ):
        _safe_relative(path, root)
    if target.exists():
        raise FileExistsError(f"Immutable prediction run manifest exists: {target}")
    blinding = validate_blinding_record(
        blinding_path,
        repository_root=root,
        expected_sampling_frame_id=frame_id,
        expected_assembly_sampling_frame=assembly_frame,
    )
    prediction_table = validate_prediction_table(_read_tsv(predictions_path))
    prediction_plan = validate_celltypist_prediction_plan(
        frozen_prediction_plan_path,
        repository_root=root,
        expected_sampling_frame_id=frame_id,
    )
    if (
        reference_model_path.name != prediction_plan["model"]["expected_file_name"]
        or sha256_file(reference_model_path) != prediction_plan["model"]["sha256"]
        or str(model_id) != prediction_plan["model"]["model_id"]
        or str(model_serialization) != prediction_plan["model"]["serialization"]
    ):
        raise ValueError("Prediction model differs from the frozen CellTypist plan")
    if not predictor_runner_path.is_file() or not predictor_feature_audit_path.is_file():
        raise FileNotFoundError("Prediction runner or feature audit is missing")
    if not str(run_id).strip() or not str(model_id).strip() or not str(
        model_serialization
    ).strip():
        raise ValueError("Prediction run/model identifiers must be non-empty")
    runner_hash = sha256_file(predictor_runner_path)
    plan_hash = sha256_file(frozen_prediction_plan_path)
    _validate_prediction_software_runtime(
        software,
        runtime,
        runner_script_sha256=runner_hash,
        prediction_plan_sha256=plan_hash,
    )
    if (
        int(runtime["n_expression_inputs"])
        != len(blinding["outputs"]["expression_inputs"])
        or int(runtime["n_prediction_rows"]) != len(prediction_table)
    ):
        raise ValueError("Prediction runtime input/output dimensions differ")
    validate_celltypist_feature_audit(
        predictor_feature_audit_path,
        repository_root=root,
        plan_path=frozen_prediction_plan_path,
        runner_script_path=predictor_runner_path,
        model_path=reference_model_path,
        expression_records=blinding["outputs"]["expression_inputs"],
        expected_sampling_frame_id=frame_id,
    )
    value: dict[str, Any] = {
        "schema_name": "t21_condition_blind_reference_prediction_run",
        "schema_version": "1.0.0",
        "run_id": str(run_id),
        "created_at_utc": utc_now(),
        "sampling_frame_id": frame_id,
        "outcome_blinded": True,
        "real_pathway_results_inspected": False,
        "condition_used_for_annotation": False,
        "candidate_pathway_genes_used_for_annotation": False,
        "cell_identifier_used_as_feature": False,
        "library_identifier_used_as_feature": False,
        "cell_manifest_used_for_join_only_after_prediction": True,
        "model_feature_space": "gene_expression_only",
        "blinding_record": _file_record(blinding_path, root),
        "cell_manifest": dict(blinding["outputs"]["cell_manifest"]),
        "expression_inputs": [
            dict(record) for record in blinding["outputs"]["expression_inputs"]
        ],
        "model": _file_record(
            reference_model_path,
            root,
            model_id=str(model_id),
            serialization=str(model_serialization),
            deserialized_by_adjudicator=False,
        ),
        "prediction_plan": _file_record(
            frozen_prediction_plan_path,
            root,
            plan_id=str(prediction_plan["plan_id"]),
        ),
        "runner_script": _file_record(predictor_runner_path, root),
        "feature_audit": _file_record(predictor_feature_audit_path, root),
        "prediction_table": _file_record(predictions_path, root),
        "software": dict(software),
        "runtime": dict(runtime),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".tmp-{secrets.token_hex(8)}")
    try:
        _atomic_write_json(value, temporary)
        validate_prediction_run_manifest(
            temporary,
            repository_root=root,
            expected_blinding_record_path=blinding_path,
            expected_cell_manifest_path=_resolve_relative(
                value["cell_manifest"]["relative_path"], root
            ),
            expected_prediction_path=predictions_path,
            expected_model_path=reference_model_path,
            expected_prediction_plan_path=frozen_prediction_plan_path,
            expected_runner_script_path=predictor_runner_path,
            expected_feature_audit_path=predictor_feature_audit_path,
            expected_sampling_frame_id=frame_id,
            expected_assembly_sampling_frame=assembly_frame,
        )
        os.replace(temporary, target)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return value


def validate_prediction_run_manifest(
    path: Path,
    *,
    repository_root: Path,
    expected_blinding_record_path: Path | None = None,
    expected_cell_manifest_path: Path | None = None,
    expected_prediction_path: Path | None = None,
    expected_model_path: Path | None = None,
    expected_prediction_plan_path: Path | None = None,
    expected_runner_script_path: Path | None = None,
    expected_feature_audit_path: Path | None = None,
    expected_sampling_frame_id: str = SAMPLING_FRAME_ID,
    expected_assembly_sampling_frame: str = PRIMARY_ASSEMBLY_SAMPLING_FRAME,
) -> dict[str, Any]:
    """Validate external prediction provenance and condition-blind attestations."""
    frame_id, assembly_frame = _registered_sampling_frame_contract(
        expected_sampling_frame_id,
        expected_assembly_sampling_frame,
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != _PREDICTION_RUN_KEYS:
        observed = set(value) if isinstance(value, dict) else set()
        raise ValueError(
            "Prediction run manifest fields differ from the fail-closed schema: "
            f"missing={sorted(_PREDICTION_RUN_KEYS-observed)}, "
            f"unexpected={sorted(observed-_PREDICTION_RUN_KEYS)}"
        )
    if (
        value["schema_name"] != "t21_condition_blind_reference_prediction_run"
        or not str(value["schema_version"]).startswith("1.")
        or not str(value["run_id"]).strip()
        or not str(value["created_at_utc"]).strip()
        or value["sampling_frame_id"] != frame_id
        or value["outcome_blinded"] is not True
        or value["real_pathway_results_inspected"] is not False
        or value["condition_used_for_annotation"] is not False
        or value["candidate_pathway_genes_used_for_annotation"] is not False
        or value["cell_identifier_used_as_feature"] is not False
        or value["library_identifier_used_as_feature"] is not False
        or value["cell_manifest_used_for_join_only_after_prediction"] is not True
        or value["model_feature_space"] != "gene_expression_only"
    ):
        raise ValueError("Prediction run manifest violates the condition-blind contract")

    blinding_path = _verify_file_record(
        value["blinding_record"],
        repository_root,
        label="prediction blinding record",
        expected_path=expected_blinding_record_path,
    )
    plain_file_fields = {"relative_path", "bytes", "sha256"}
    for role in (
        "blinding_record",
        "cell_manifest",
        "prediction_table",
        "runner_script",
        "feature_audit",
    ):
        if not isinstance(value[role], dict) or set(value[role]) != plain_file_fields:
            raise ValueError(f"Prediction {role} record has unexpected fields")
    blinding = validate_blinding_record(
        blinding_path,
        repository_root=repository_root,
        expected_sampling_frame_id=frame_id,
        expected_assembly_sampling_frame=assembly_frame,
    )
    manifest_path = _verify_file_record(
        value["cell_manifest"],
        repository_root,
        label="prediction cell manifest",
        expected_path=expected_cell_manifest_path,
    )
    blinding_manifest = blinding["outputs"]["cell_manifest"]
    if (
        value["cell_manifest"]["relative_path"] != blinding_manifest["relative_path"]
        or value["cell_manifest"]["sha256"] != blinding_manifest["sha256"]
    ):
        raise ValueError("Prediction cell manifest differs from the blinding record")
    validate_cell_manifest(_read_tsv(manifest_path))

    expression_inputs = value["expression_inputs"]
    blinded_inputs = blinding["outputs"]["expression_inputs"]
    if not isinstance(expression_inputs, list) or len(expression_inputs) != len(blinded_inputs):
        raise ValueError("Prediction expression inputs differ from the blinded export")
    for index, (declared, blinded) in enumerate(zip(expression_inputs, blinded_inputs)):
        if not isinstance(declared, dict) or set(declared) != {
            "relative_path",
            "bytes",
            "sha256",
            "role",
            "shard_order",
            "n_cells",
            "n_genes",
        }:
            raise ValueError("Prediction expression input has unexpected fields")
        _verify_file_record(
            declared,
            repository_root,
            label=f"prediction expression input {index}",
        )
        for field in ("relative_path", "bytes", "sha256", "role", "shard_order"):
            if declared.get(field) != blinded.get(field):
                raise ValueError("Prediction expression input differs from the blinding record")

    model = value["model"]
    if not isinstance(model, dict) or set(model) != {
        "model_id",
        "relative_path",
        "bytes",
        "sha256",
        "serialization",
        "deserialized_by_adjudicator",
    }:
        raise ValueError("Prediction run model record has invalid fields")
    model_path = _verify_file_record(
        model,
        repository_root,
        label="prediction reference model",
        expected_path=expected_model_path,
    )
    if (
        not str(model["model_id"]).strip()
        or not str(model["serialization"]).strip()
        or model["deserialized_by_adjudicator"] is not False
    ):
        raise ValueError("Prediction model provenance is incomplete")

    prediction_plan_record = value["prediction_plan"]
    if not isinstance(prediction_plan_record, dict) or set(prediction_plan_record) != {
        "relative_path",
        "bytes",
        "sha256",
        "plan_id",
    }:
        raise ValueError("Prediction plan record has invalid fields")
    prediction_plan_path = _verify_file_record(
        prediction_plan_record,
        repository_root,
        label="frozen CellTypist prediction plan",
        expected_path=expected_prediction_plan_path,
    )
    prediction_plan = validate_celltypist_prediction_plan(
        prediction_plan_path,
        repository_root=repository_root,
        expected_sampling_frame_id=frame_id,
    )
    if prediction_plan_record["plan_id"] != prediction_plan["plan_id"]:
        raise ValueError("Prediction plan ID differs from the frozen plan")
    if (
        model["model_id"] != prediction_plan["model"]["model_id"]
        or model["sha256"] != prediction_plan["model"]["sha256"]
        or model["serialization"] != prediction_plan["model"]["serialization"]
        or model_path.name != prediction_plan["model"]["expected_file_name"]
    ):
        raise ValueError("Prediction model differs from the frozen plan")

    runner_script_path = _verify_file_record(
        value["runner_script"],
        repository_root,
        label="CellTypist prediction runner",
        expected_path=expected_runner_script_path,
    )
    feature_audit_path = _verify_file_record(
        value["feature_audit"],
        repository_root,
        label="CellTypist feature audit",
        expected_path=expected_feature_audit_path,
    )

    prediction_path = _verify_file_record(
        value["prediction_table"],
        repository_root,
        label="reference prediction table",
        expected_path=expected_prediction_path,
    )
    prediction_table = validate_prediction_table(_read_tsv(prediction_path))
    software = value["software"]
    runtime = value["runtime"]
    _validate_prediction_software_runtime(
        software,
        runtime,
        runner_script_sha256=sha256_file(runner_script_path),
        prediction_plan_sha256=sha256_file(prediction_plan_path),
    )
    if (
        int(runtime["n_expression_inputs"]) != len(expression_inputs)
        or int(runtime["n_prediction_rows"]) != len(prediction_table)
    ):
        raise ValueError("Prediction runtime input/output dimensions differ")
    validate_celltypist_feature_audit(
        feature_audit_path,
        repository_root=repository_root,
        plan_path=prediction_plan_path,
        runner_script_path=runner_script_path,
        model_path=model_path,
        expression_records=expression_inputs,
        expected_sampling_frame_id=frame_id,
    )
    value["_resolved_paths"] = {
        "blinding_record": blinding_path,
        "cell_manifest": manifest_path,
        "model": model_path,
        "prediction_table": prediction_path,
        "prediction_plan": prediction_plan_path,
        "runner_script": runner_script_path,
        "feature_audit": feature_audit_path,
    }
    return value


def _lineage_reason(
    *,
    input_uncertain: bool,
    predeclared_uncertain: bool,
    confidence_below_threshold: bool,
    predicted_doublet: bool,
    included: bool,
) -> str:
    reasons: list[str] = []
    if predicted_doublet:
        reasons.append("predicted_doublet")
    if input_uncertain:
        reasons.append("prediction_run_marked_uncertain")
    if predeclared_uncertain:
        reasons.append("reference_label_predeclared_uncertain")
    if confidence_below_threshold:
        reasons.append("mapping_confidence_below_frozen_threshold")
    if not reasons:
        reasons.append(
            "included_by_frozen_lineage_rule"
            if included
            else "analysis_cell_type_outside_frozen_lineage"
        )
    return ";".join(reasons)


def build_reference_annotation_table(
    *,
    cell_manifest: pd.DataFrame,
    predictions: pd.DataFrame,
    mapping: Mapping[str, Any],
    annotation_plan: Mapping[str, Any],
    annotation_run_id: str,
    model_id: str,
    model_sha256: str,
    label_mapping_sha256: str,
    prediction_table_sha256: str,
    cell_manifest_sha256: str,
) -> pd.DataFrame:
    """Join predictions to exact cell keys and apply the frozen lineage rule."""
    cells = validate_cell_manifest(cell_manifest)
    predicted = validate_prediction_table(predictions)
    missing = set(cells["annotation_row_id"]).difference(predicted["annotation_row_id"])
    extra = set(predicted["annotation_row_id"]).difference(cells["annotation_row_id"])
    if missing or extra:
        raise ValueError(
            "Prediction/cell-manifest anonymous IDs differ: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    merged = cells.merge(
        predicted,
        on="annotation_row_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if merged["annotation_row_id"].tolist() != cells["annotation_row_id"].tolist():
        raise ValueError("Prediction join changed the canonical cell order")

    label_mapping = {
        str(label): str(target)
        for label, target in mapping["reference_label_mapping"].items()
    }
    unmapped = sorted(set(merged["reference_label"]).difference(label_mapping))
    if unmapped:
        raise ValueError(f"Reference predictions contain unmapped labels: {unmapped}")
    analysis_type = merged["reference_label"].map(label_mapping)
    policy = mapping["annotation_policy"]
    threshold = float(policy["minimum_mapping_confidence"])
    confidence_below = merged["mapping_confidence"].astype(float).lt(threshold)
    predeclared_uncertain = merged["reference_label"].isin(
        set(map(str, mapping["uncertain_reference_labels"]))
    )
    predeclared_doublet = merged["reference_label"].isin(
        set(map(str, mapping["predicted_doublet_reference_labels"]))
    )
    input_uncertain = merged["annotation_uncertain"].astype(bool)
    input_doublet = merged["predicted_doublet"].astype(bool)
    final_uncertain = (
        input_uncertain
        | predeclared_uncertain
        | confidence_below
        | analysis_type.eq(str(policy["uncertain_analysis_cell_type"]))
    )
    final_doublet = input_doublet | predeclared_doublet
    included_types = set(
        map(str, annotation_plan["lineage_rule"]["include_analysis_cell_types"])
    )
    inclusion = analysis_type.isin(included_types) & ~final_uncertain & ~final_doublet
    reasons = [
        _lineage_reason(
            input_uncertain=bool(input_uncertain.iloc[index]),
            predeclared_uncertain=bool(predeclared_uncertain.iloc[index]),
            confidence_below_threshold=bool(confidence_below.iloc[index]),
            predicted_doublet=bool(final_doublet.iloc[index]),
            included=bool(inclusion.iloc[index]),
        )
        for index in range(len(merged))
    ]
    source_value = "condition_blind_reference_mapping:" + str(mapping["mapping_id"])
    output = pd.DataFrame(
        {
            "cell_id": merged["cell_id"],
            "library_id": merged["library_id"],
            "original_barcode": merged["original_barcode"],
            "original_cell_type": merged["reference_label"],
            "original_cell_type_source": source_value,
            "analysis_cell_type": analysis_type,
            "lineage_inclusion": inclusion.astype(bool),
            "lineage_inclusion_reason": reasons,
            "mapping_confidence": merged["mapping_confidence"].astype(float),
            "annotation_uncertain": final_uncertain.astype(bool),
            "predicted_doublet": final_doublet.astype(bool),
            "annotation_row_id": merged["annotation_row_id"],
            "annotation_plan_id": str(annotation_plan["plan_id"]),
            "reference_mapping_id": str(mapping["mapping_id"]),
            "annotation_run_id": str(annotation_run_id),
            "model_id": str(model_id),
            "model_sha256": _require_sha256(model_sha256, "model_sha256"),
            "label_mapping_sha256": _require_sha256(
                label_mapping_sha256, "label_mapping_sha256"
            ),
            "prediction_table_sha256": _require_sha256(
                prediction_table_sha256, "prediction_table_sha256"
            ),
            "cell_manifest_sha256": _require_sha256(
                cell_manifest_sha256, "cell_manifest_sha256"
            ),
        },
        columns=ANNOTATION_COLUMNS,
    )
    if output["original_cell_type"].tolist() != merged["reference_label"].tolist():
        raise ValueError("Raw reference labels changed during adjudication")
    return output


def adjudicate_reference_annotations(
    *,
    repository_root: Path,
    cell_manifest_path: Path,
    prediction_path: Path,
    prediction_run_manifest_path: Path,
    blinding_record_path: Path,
    model_path: Path,
    mapping_path: Path,
    annotation_plan_path: Path,
    output_dir: Path,
    sampling_frame_id: str = SAMPLING_FRAME_ID,
    assembly_sampling_frame: str = PRIMARY_ASSEMBLY_SAMPLING_FRAME,
) -> dict[str, Any]:
    """Create an immutable, hash-bound formal reference-annotation bundle."""
    frame_id, assembly_frame = _registered_sampling_frame_contract(
        sampling_frame_id,
        assembly_sampling_frame,
    )
    root = repository_root.resolve()
    supplied_paths = {
        "cell_manifest": cell_manifest_path.resolve(),
        "prediction_table": prediction_path.resolve(),
        "prediction_run_manifest": prediction_run_manifest_path.resolve(),
        "blinding_record": blinding_record_path.resolve(),
        "model": model_path.resolve(),
        "label_mapping": mapping_path.resolve(),
        "annotation_plan": annotation_plan_path.resolve(),
    }
    for role, path in supplied_paths.items():
        _safe_relative(path, root)
        if not path.is_file():
            raise FileNotFoundError(f"Annotation input {role!r} is missing: {path}")
    target = output_dir.resolve()
    _safe_relative(target, root)
    if target.exists():
        raise FileExistsError(f"Immutable annotation bundle exists: {target}")

    mapping = validate_reference_mapping(
        supplied_paths["label_mapping"],
        expected_sampling_frame_id=frame_id,
    )
    plan = validate_annotation_plan_for_reference(
        supplied_paths["annotation_plan"],
        repository_root=root,
        mapping_path=supplied_paths["label_mapping"],
        mapping=mapping,
        expected_sampling_frame_id=frame_id,
    )
    if model_path.name != str(mapping["model"]["expected_file_name"]):
        raise ValueError("Reference model filename differs from the frozen mapping")
    model_hash = sha256_file(supplied_paths["model"])
    if model_hash != str(mapping["model"]["sha256"]).lower():
        raise ValueError("Reference model SHA256 differs from the frozen mapping")
    run = validate_prediction_run_manifest(
        supplied_paths["prediction_run_manifest"],
        repository_root=root,
        expected_blinding_record_path=supplied_paths["blinding_record"],
        expected_cell_manifest_path=supplied_paths["cell_manifest"],
        expected_prediction_path=supplied_paths["prediction_table"],
        expected_model_path=supplied_paths["model"],
        expected_sampling_frame_id=frame_id,
        expected_assembly_sampling_frame=assembly_frame,
    )
    if (
        run["model"]["model_id"] != mapping["model"]["model_id"]
        or run["model"]["sha256"] != model_hash
        or run["model"]["serialization"] != mapping["model"]["serialization"]
    ):
        raise ValueError("Prediction run used another reference model")
    prediction_plan_path = run["_resolved_paths"]["prediction_plan"]
    prediction_plan = validate_celltypist_prediction_plan(
        prediction_plan_path,
        repository_root=root,
        expected_mapping_path=supplied_paths["label_mapping"],
        expected_sampling_frame_id=frame_id,
    )
    prediction_rules = prediction_plan["prediction"]
    if (
        float(prediction_rules["minimum_mapping_confidence"])
        != float(mapping["annotation_policy"]["minimum_mapping_confidence"])
        or list(map(str, prediction_rules["predicted_doublet_reference_labels"]))
        != list(map(str, mapping["predicted_doublet_reference_labels"]))
        or list(map(str, prediction_rules["uncertain_reference_labels"]))
        != list(map(str, mapping["uncertain_reference_labels"]))
    ):
        raise ValueError("Prediction uncertainty/doublet rules differ from the label mapping")

    cell_manifest = validate_cell_manifest(_read_tsv(supplied_paths["cell_manifest"]))
    predictions = validate_prediction_table(_read_tsv(supplied_paths["prediction_table"]))
    mapping_hash = sha256_file(supplied_paths["label_mapping"])
    plan_hash = sha256_file(supplied_paths["annotation_plan"])
    prediction_hash = sha256_file(supplied_paths["prediction_table"])
    cell_manifest_hash = sha256_file(supplied_paths["cell_manifest"])
    annotations = build_reference_annotation_table(
        cell_manifest=cell_manifest,
        predictions=predictions,
        mapping=mapping,
        annotation_plan=plan,
        annotation_run_id=str(run["run_id"]),
        model_id=str(run["model"]["model_id"]),
        model_sha256=model_hash,
        label_mapping_sha256=mapping_hash,
        prediction_table_sha256=prediction_hash,
        cell_manifest_sha256=cell_manifest_hash,
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".tmp-{target.name}-{secrets.token_hex(8)}"
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        annotation_path = temporary / ANNOTATION_TABLE_NAME
        _atomic_write_tsv(annotations, annotation_path)
        final_annotation_path = target / ANNOTATION_TABLE_NAME
        output_record = _file_record(annotation_path, temporary)
        output_record["relative_path"] = _safe_relative(final_annotation_path, root)
        record: dict[str, Any] = {
            "schema_name": "t21_cell_annotation_adjudication_record",
            "schema_version": "1.0.0",
            "status": "pass_condition_blind_reference_annotation_adjudicated",
            "created_at_utc": utc_now(),
            "sampling_frame_id": frame_id,
            "outcome_blinded": True,
            "real_pathway_results_inspected": False,
            "condition_used_for_annotation": False,
            "candidate_pathway_genes_used_for_annotation": False,
            "author_label_claim_allowed": False,
            "original_cell_type_source": annotations[
                "original_cell_type_source"
            ].iloc[0],
            "inputs": {
                role: _file_record(path, root)
                for role, path in supplied_paths.items()
            },
            "prediction_run": {
                "run_id": run["run_id"],
                "software": run["software"],
                "runtime": run["runtime"],
                "model_feature_space": run["model_feature_space"],
                "expression_inputs": run["expression_inputs"],
                "prediction_plan": run["prediction_plan"],
                "runner_script": run["runner_script"],
                "feature_audit": run["feature_audit"],
            },
            "model": {
                "model_id": run["model"]["model_id"],
                "relative_path": _safe_relative(supplied_paths["model"], root),
                "bytes": int(supplied_paths["model"].stat().st_size),
                "sha256": model_hash,
                "serialization": run["model"]["serialization"],
                "deserialized_by_adjudicator": False,
            },
            "label_mapping": {
                "mapping_id": mapping["mapping_id"],
                "relative_path": _safe_relative(supplied_paths["label_mapping"], root),
                "bytes": int(supplied_paths["label_mapping"].stat().st_size),
                "sha256": mapping_hash,
            },
            "annotation_plan": {
                "plan_id": plan["plan_id"],
                "relative_path": _safe_relative(supplied_paths["annotation_plan"], root),
                "bytes": int(supplied_paths["annotation_plan"].stat().st_size),
                "sha256": plan_hash,
            },
            "exact_join": {
                "key": [
                    "annotation_row_id",
                    "cell_id",
                    "library_id",
                    "original_barcode",
                ],
                "n_cell_manifest_rows": len(cell_manifest),
                "n_prediction_rows": len(predictions),
                "n_output_rows": len(annotations),
                "n_duplicate_cell_ids": 0,
                "n_duplicate_library_barcodes": 0,
                "n_missing_predictions": 0,
                "n_extra_predictions": 0,
            },
            "output": output_record,
            "record_checksum_policy": "external_sha256_sidecar",
            "claim_boundary": (
                "condition-blind reference annotation only; not author labels and not "
                "functional mechanism evidence"
            ),
        }
        record_path = temporary / ADJUDICATION_RECORD_NAME
        _atomic_write_json(record, record_path)
        record_hash = sha256_file(record_path)
        checksum_path = record_path.with_name(record_path.name + RECORD_CHECKSUM_SUFFIX)
        checksum_path.write_text(
            f"{record_hash}  {ADJUDICATION_RECORD_NAME}\n", encoding="ascii"
        )
        os.replace(temporary, target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return record


def _read_record_checksum(record_path: Path) -> str:
    checksum_path = record_path.with_name(record_path.name + RECORD_CHECKSUM_SUFFIX)
    if not checksum_path.is_file():
        raise FileNotFoundError("Annotation adjudication checksum sidecar is missing")
    fields = checksum_path.read_text(encoding="ascii").strip().split()
    if len(fields) != 2 or fields[1] != record_path.name:
        raise ValueError("Annotation adjudication checksum sidecar is malformed")
    digest = _require_sha256(fields[0], "annotation adjudication record checksum")
    if digest != sha256_file(record_path):
        raise ValueError("Annotation adjudication record checksum differs")
    return digest


def validate_reference_annotation_bundle(
    *,
    repository_root: Path,
    annotation_path: Path,
    adjudication_record_path: Path,
    mapping_path: Path,
    annotation_plan_path: Path,
    expected_sampling_frame_id: str = SAMPLING_FRAME_ID,
    expected_assembly_sampling_frame: str = PRIMARY_ASSEMBLY_SAMPLING_FRAME,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Recompute and validate a formal reference-annotation bundle."""
    frame_id, assembly_frame = _registered_sampling_frame_contract(
        expected_sampling_frame_id,
        expected_assembly_sampling_frame,
    )
    root = repository_root.resolve()
    annotations_path = annotation_path.resolve()
    record_path = adjudication_record_path.resolve()
    expected_mapping_path = mapping_path.resolve()
    expected_plan_path = annotation_plan_path.resolve()
    for path in (
        annotations_path,
        record_path,
        expected_mapping_path,
        expected_plan_path,
    ):
        _safe_relative(path, root)
        if not path.is_file():
            raise FileNotFoundError(path)
    _read_record_checksum(record_path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or (
        record.get("schema_name") != "t21_cell_annotation_adjudication_record"
        or not str(record.get("schema_version", "")).startswith("1.")
        or record.get("status")
        != "pass_condition_blind_reference_annotation_adjudicated"
        or record.get("sampling_frame_id") != frame_id
        or record.get("outcome_blinded") is not True
        or record.get("real_pathway_results_inspected") is not False
        or record.get("condition_used_for_annotation") is not False
        or record.get("candidate_pathway_genes_used_for_annotation") is not False
        or record.get("author_label_claim_allowed") is not False
        or record.get("record_checksum_policy") != "external_sha256_sidecar"
    ):
        raise ValueError("Reference annotation adjudication record has an invalid contract")
    inputs = record.get("inputs")
    expected_roles = {
        "cell_manifest",
        "prediction_table",
        "prediction_run_manifest",
        "blinding_record",
        "model",
        "label_mapping",
        "annotation_plan",
    }
    if not isinstance(inputs, dict) or set(inputs) != expected_roles:
        raise ValueError("Reference annotation adjudication input roles differ")
    resolved_inputs = {
        role: _verify_file_record(
            inputs[role],
            root,
            label=f"annotation adjudication input {role}",
            expected_path=(
                expected_mapping_path
                if role == "label_mapping"
                else expected_plan_path
                if role == "annotation_plan"
                else None
            ),
        )
        for role in expected_roles
    }
    output_path = _verify_file_record(
        record.get("output", {}),
        root,
        label="adjudicated annotation table",
        expected_path=annotations_path,
    )
    mapping = validate_reference_mapping(
        expected_mapping_path,
        expected_sampling_frame_id=frame_id,
    )
    plan = validate_annotation_plan_for_reference(
        expected_plan_path,
        repository_root=root,
        mapping_path=expected_mapping_path,
        mapping=mapping,
        expected_sampling_frame_id=frame_id,
    )
    model_path = resolved_inputs["model"]
    if (
        model_path.name != mapping["model"]["expected_file_name"]
        or sha256_file(model_path) != mapping["model"]["sha256"]
    ):
        raise ValueError("Adjudicated annotation model differs from the frozen mapping")
    run = validate_prediction_run_manifest(
        resolved_inputs["prediction_run_manifest"],
        repository_root=root,
        expected_blinding_record_path=resolved_inputs["blinding_record"],
        expected_cell_manifest_path=resolved_inputs["cell_manifest"],
        expected_prediction_path=resolved_inputs["prediction_table"],
        expected_model_path=model_path,
        expected_sampling_frame_id=frame_id,
        expected_assembly_sampling_frame=assembly_frame,
    )
    prediction_plan = validate_celltypist_prediction_plan(
        run["_resolved_paths"]["prediction_plan"],
        repository_root=root,
        expected_mapping_path=expected_mapping_path,
        expected_sampling_frame_id=frame_id,
    )
    prediction_rules = prediction_plan["prediction"]
    if (
        float(prediction_rules["minimum_mapping_confidence"])
        != float(mapping["annotation_policy"]["minimum_mapping_confidence"])
        or list(map(str, prediction_rules["predicted_doublet_reference_labels"]))
        != list(map(str, mapping["predicted_doublet_reference_labels"]))
        or list(map(str, prediction_rules["uncertain_reference_labels"]))
        != list(map(str, mapping["uncertain_reference_labels"]))
    ):
        raise ValueError("Adjudicated prediction rules differ from the frozen mapping")
    expected_prediction_run = {
        "run_id": run["run_id"],
        "software": run["software"],
        "runtime": run["runtime"],
        "model_feature_space": run["model_feature_space"],
        "expression_inputs": run["expression_inputs"],
        "prediction_plan": run["prediction_plan"],
        "runner_script": run["runner_script"],
        "feature_audit": run["feature_audit"],
    }
    if record.get("prediction_run") != expected_prediction_run:
        raise ValueError("Adjudication prediction runner/plan/audit evidence differs")
    if (
        record.get("model", {}).get("sha256") != sha256_file(model_path)
        or record.get("model", {}).get("deserialized_by_adjudicator") is not False
        or record.get("label_mapping", {}).get("sha256")
        != sha256_file(expected_mapping_path)
        or record.get("annotation_plan", {}).get("sha256")
        != sha256_file(expected_plan_path)
    ):
        raise ValueError("Adjudication model/mapping/plan hashes do not close")

    cells = validate_cell_manifest(_read_tsv(resolved_inputs["cell_manifest"]))
    predictions = validate_prediction_table(_read_tsv(resolved_inputs["prediction_table"]))
    recomputed = build_reference_annotation_table(
        cell_manifest=cells,
        predictions=predictions,
        mapping=mapping,
        annotation_plan=plan,
        annotation_run_id=str(run["run_id"]),
        model_id=str(run["model"]["model_id"]),
        model_sha256=sha256_file(model_path),
        label_mapping_sha256=sha256_file(expected_mapping_path),
        prediction_table_sha256=sha256_file(resolved_inputs["prediction_table"]),
        cell_manifest_sha256=sha256_file(resolved_inputs["cell_manifest"]),
    )
    observed = _read_tsv(output_path)
    _validate_exact_columns(observed, ANNOTATION_COLUMNS, "Adjudicated annotation table")
    observed["lineage_inclusion"] = _strict_bool(
        observed["lineage_inclusion"], "annotations.lineage_inclusion"
    )
    observed["annotation_uncertain"] = _strict_bool(
        observed["annotation_uncertain"], "annotations.annotation_uncertain"
    )
    observed["predicted_doublet"] = _strict_bool(
        observed["predicted_doublet"], "annotations.predicted_doublet"
    )
    observed["mapping_confidence"] = pd.to_numeric(
        observed["mapping_confidence"], errors="raise"
    ).astype(float)
    pd.testing.assert_frame_equal(
        observed.loc[:, ANNOTATION_COLUMNS].reset_index(drop=True),
        recomputed.loc[:, ANNOTATION_COLUMNS].reset_index(drop=True),
        check_dtype=True,
        check_exact=True,
    )
    join = record.get("exact_join")
    if not isinstance(join, dict) or any(
        int(join.get(field, -1)) != expected
        for field, expected in {
            "n_cell_manifest_rows": len(cells),
            "n_prediction_rows": len(predictions),
            "n_output_rows": len(observed),
            "n_duplicate_cell_ids": 0,
            "n_duplicate_library_barcodes": 0,
            "n_missing_predictions": 0,
            "n_extra_predictions": 0,
        }.items()
    ):
        raise ValueError("Reference annotation exact-join summary does not close")
    expected_source = "condition_blind_reference_mapping:" + str(mapping["mapping_id"])
    if record.get("original_cell_type_source") != expected_source or not observed[
        "original_cell_type_source"
    ].eq(expected_source).all():
        raise ValueError("Reference annotations do not explicitly identify their source")
    return record, observed, mapping


def validate_reference_annotation_semantics(
    annotations: pd.DataFrame,
    *,
    plan: Mapping[str, Any],
    mapping: Mapping[str, Any],
) -> set[str]:
    """Validate source-specific fields before copying annotations into H5AD."""
    required = set(ANNOTATION_COLUMNS)
    missing = required.difference(annotations.columns)
    if missing:
        raise ValueError(f"Reference annotations omit audit fields: {sorted(missing)}")
    expected_source = "condition_blind_reference_mapping:" + str(mapping["mapping_id"])
    if not annotations["original_cell_type_source"].astype(str).eq(expected_source).all():
        raise ValueError("Reference annotations mix or misidentify label sources")
    raw = annotations["original_cell_type"].astype(str)
    mapped = raw.map(mapping["reference_label_mapping"])
    if mapped.isna().any():
        raise ValueError("Reference annotations contain a source label outside the frozen map")
    if not annotations["analysis_cell_type"].astype(str).eq(mapped.astype(str)).all():
        raise ValueError("Reference analysis_cell_type differs from the frozen label map")
    confidence = pd.to_numeric(annotations["mapping_confidence"], errors="coerce")
    uncertain = _strict_bool(annotations["annotation_uncertain"], "annotation_uncertain")
    doublet = _strict_bool(annotations["predicted_doublet"], "predicted_doublet")
    inclusion = _strict_bool(annotations["lineage_inclusion"], "lineage_inclusion")
    threshold = float(mapping["annotation_policy"]["minimum_mapping_confidence"])
    included_types = set(map(str, plan["lineage_rule"]["include_analysis_cell_types"]))
    expected = mapped.astype(str).isin(included_types) & confidence.ge(threshold) & ~uncertain & ~doublet
    if confidence.isna().any() or not inclusion.eq(expected).all():
        raise ValueError("Reference lineage inclusion differs from the frozen rule")
    if annotations.loc[~inclusion, "lineage_inclusion_reason"].astype(str).str.strip().eq("").any():
        raise ValueError("Every excluded reference-annotated cell requires a reason")
    return set(ANNOTATION_COLUMNS)


def classify_annotation_source(
    annotations: pd.DataFrame,
    *,
    plan: Mapping[str, Any],
) -> str:
    """Classify a whole annotation table as author or adjudicated reference labels."""
    if "original_cell_type_source" not in annotations.columns or annotations.empty:
        raise ValueError("Cell annotations omit their label source")
    sources = annotations["original_cell_type_source"].astype(str).str.strip()
    if sources.eq("").any() or sources.nunique() != 1:
        raise ValueError("A formal annotation table must use exactly one explicit source")
    contracts = plan.get("source_contracts")
    if not isinstance(contracts, Mapping):
        raise ValueError("Annotation plan omits source-specific contracts")
    author = contracts.get("author")
    reference = contracts.get("condition_blind_reference")
    if not isinstance(author, Mapping) or not isinstance(reference, Mapping):
        raise ValueError("Annotation plan source contracts are incomplete")
    source = str(sources.iloc[0])
    allowed_author = set(map(str, author.get("original_cell_type_source_values", [])))
    reference_prefix = str(reference.get("original_cell_type_source_prefix", ""))
    if source in allowed_author:
        return "author"
    if reference_prefix and source.startswith(reference_prefix):
        return "condition_blind_reference"
    raise ValueError(f"Cell annotations use an unregistered label source: {source!r}")


def validate_author_annotation_semantics(
    annotations: pd.DataFrame,
    *,
    plan: Mapping[str, Any],
) -> set[str]:
    """Preserve the legacy author-label mapping while making its source explicit."""
    if classify_annotation_source(annotations, plan=plan) != "author":
        raise ValueError("Author annotation validation received another label source")
    required = {
        "cell_id",
        "original_cell_type",
        "original_cell_type_source",
        "analysis_cell_type",
        "lineage_inclusion",
        "lineage_inclusion_reason",
    }
    missing = required.difference(annotations.columns)
    if missing:
        raise ValueError(f"Author annotations omit fields: {sorted(missing)}")
    mapping = {str(key): str(value) for key, value in plan["label_mapping"].items()}
    original = annotations["original_cell_type"].astype(str)
    mapped = original.map(mapping)
    known = mapped.notna()
    if not annotations.loc[known, "analysis_cell_type"].astype(str).eq(
        mapped.loc[known].astype(str)
    ).all():
        raise ValueError("Cell annotations violate the frozen author-label mapping")
    inclusion = _strict_bool(annotations["lineage_inclusion"], "lineage_inclusion")
    if inclusion.loc[~known].any():
        raise ValueError("Unmapped author labels cannot enter the formal lineage")
    included_types = set(map(str, plan["lineage_rule"]["include_analysis_cell_types"]))
    expected = annotations["analysis_cell_type"].astype(str).isin(included_types) & known
    if not inclusion.eq(expected).all():
        raise ValueError("Author-label lineage inclusion differs from the frozen rule")
    if annotations.loc[~inclusion, "lineage_inclusion_reason"].astype(str).str.strip().eq("").any():
        raise ValueError("Every excluded author-annotated cell requires a reason")
    return required
