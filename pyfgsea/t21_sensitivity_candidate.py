from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence
from uuid import uuid4

import anndata as ad
import numpy as np
import pandas as pd
import yaml

from .t21_data_product import (
    FINAL_PRODUCT_NAMES,
    artifact_record,
    expression_contract_metadata,
    sha256_file,
    stable_json,
    utc_now,
    validate_scrna_contract,
)
from .t21_expression_preprocessing import (
    formal_expression_preprocessing_contract,
    formal_expression_preprocessing_source_sha256,
    normalize_t21_formal_expression,
)
from .t21_reference_annotation import (
    REFERENCE_AUDIT_COLUMNS,
    validate_reference_annotation_bundle,
    validate_reference_annotation_semantics,
)
from .t21_sensitivity_cell_call_migration import (
    TARGET_POLICY_SHA256,
    validate_cd235a_neg_cell_call_migration,
)
from .t21_trajectory_fate import (
    _capture_trajectory_code_bindings,
    build_t21_trajectory_fate_products,
    validate_built_trajectory_fate_directory,
)


SAMPLING_FRAME_ID = "t21_fetal_liver_cd235a_neg_sensitivity_v1"
ANALYSIS_ROLE = "sensitivity_only"
EXPECTED_LIBRARIES = 18
EXPECTED_CELLS = 371_259
EXPECTED_FEATURES = 33_538
EXPECTED_DONORS = 16
EXPECTED_CASES = 13
EXPECTED_CONTROLS = 3
V1_POLICY_RELATIVE_PATH = (
    "config/t21_preprocessing_adjudication_cd235a_neg_sensitivity_v1.yaml"
)
V1_POLICY_SHA256 = (
    "58fb284601059cb195aaa4264bd8a439c805878f04a537f34b893603c7f0a0c4"
)
V2_POLICY_RELATIVE_PATH = (
    "config/t21_preprocessing_adjudication_cd235a_neg_sensitivity_v2.yaml"
)
NAMESPACE_RELATIVE_PATH = (
    "data_external/t21_data_product_v1/staging/sensitivity/"
    f"{SAMPLING_FRAME_ID}"
)
AUDIT_RELATIVE_PATH = (
    "data_external/t21_data_product_v1/audit/sensitivity/"
    f"{SAMPLING_FRAME_ID}"
)
ASSEMBLY_SPEC_NAME = "t21_scrna_assembly_spec_candidate_v1.tsv"
BATCH_DEFINITION_NAME = "t21_technical_batch_resolution_v1.tsv"
ASSEMBLED_H5AD_NAME = FINAL_PRODUCT_NAMES["scrna"]
ASSEMBLY_RECORD_NAME = "t21_scrna_assembly_record_v1.json"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _atomic_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, sep="\t", index=False, lineterminator="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _relative(path: Path, repository_root: Path) -> str:
    return path.resolve().relative_to(repository_root.resolve()).as_posix()


def _file_record(path: Path, repository_root: Path) -> dict[str, Any]:
    return {
        "relative_path": _relative(path, repository_root),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _namespace_paths(repository_root: Path) -> dict[str, Path]:
    root = repository_root.resolve()
    namespace = root / NAMESPACE_RELATIVE_PATH
    audit = root / AUDIT_RELATIVE_PATH
    return {
        "namespace": namespace,
        "audit": audit,
        "shard_root": namespace / "scrna_qc_shards",
        "shard_ledger": namespace / "scrna_qc_shards" / "t21_count_shard_ledger_v1.tsv",
        "assembly_spec": namespace / ASSEMBLY_SPEC_NAME,
        "batch_definition": audit / BATCH_DEFINITION_NAME,
        "cell_call_adjudication": audit / "t21_cell_call_adjudication_v1.json",
        "v2_migration": audit / "t21_cell_call_policy_v2_migration.json",
        "v2_revalidation": audit / "t21_cell_call_policy_v2_revalidation.tsv",
        "candidate_root": namespace / "scrna_assemblies",
    }


def _strict_bool(values: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        if values.isna().any():
            raise ValueError(f"{label} contains missing booleans")
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError(f"{label} must contain only true/false")
    return normalized.eq("true")


def build_assembly_spec_from_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    required = {
        "library_id",
        "status",
        "n_cells",
        "n_features",
        "shard",
        "shard_sha256",
        "sampling_frame",
        "tissue",
        "preprocessing_policy_sha256",
    }
    missing = sorted(required.difference(ledger.columns))
    if missing:
        raise ValueError(f"Sensitivity shard ledger lacks columns: {missing}")
    result = ledger.copy()
    if len(result) != EXPECTED_LIBRARIES or result["library_id"].astype(str).duplicated().any():
        raise ValueError("Sensitivity shard ledger is not the exact 18-library frame")
    if not result["status"].astype(str).str.startswith("pass_").all():
        raise ValueError("Sensitivity shard ledger contains a non-pass row")
    if not result["sampling_frame"].astype(str).eq("cd235a_neg").all():
        raise ValueError("Sensitivity shard ledger contains another sampling frame")
    if not result["tissue"].astype(str).eq("liver").all():
        raise ValueError("Sensitivity shard ledger contains another tissue")
    if not result["preprocessing_policy_sha256"].astype(str).str.lower().eq(
        V1_POLICY_SHA256
    ).all():
        raise ValueError("Sensitivity shard ledger uses another preprocessing policy")
    cells = pd.to_numeric(result["n_cells"], errors="raise").astype(int)
    features = pd.to_numeric(result["n_features"], errors="raise").astype(int)
    if int(cells.sum()) != EXPECTED_CELLS or not features.eq(EXPECTED_FEATURES).all():
        raise ValueError("Sensitivity shard dimensions differ from frozen evidence")
    if not result["shard_sha256"].astype(str).str.lower().map(
        lambda value: bool(_SHA256_RE.fullmatch(value))
    ).all():
        raise ValueError("Sensitivity shard ledger has a malformed SHA256")
    return pd.DataFrame(
        {
            "library_order": np.arange(len(result), dtype=int),
            "library_id": result["library_id"].astype(str).to_numpy(),
            "sampling_frame": "cd235a_neg",
            "tissue": "liver",
            "shard_sha256": result["shard_sha256"].astype(str).str.lower().to_numpy(),
            "include": True,
            "include_reason": "locked_sensitivity_complete_validated_shard",
        }
    )


def build_batch_definition(library_ids: Sequence[str]) -> pd.DataFrame:
    libraries = [str(value) for value in library_ids]
    if len(libraries) != EXPECTED_LIBRARIES or len(libraries) != len(set(libraries)):
        raise ValueError("Sensitivity batch definition requires 18 unique libraries")
    return pd.DataFrame(
        {
            "library_id": libraries,
            "technical_batch": "omitted_not_identifiable",
            "batch_definition_status": "explicitly_omitted_not_identifiable",
            "evidence": (
                "official metadata contain no identifiable sensitivity-frame batch; "
                "no batch coefficient is fabricated"
            ),
        }
    )


def validate_sensitivity_preprocessing_evidence(
    repository_root: str | Path,
    *,
    validate_raw_axis: bool = True,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    paths = _namespace_paths(root)
    v1_policy = root / V1_POLICY_RELATIVE_PATH
    v2_policy = root / V2_POLICY_RELATIVE_PATH
    if sha256_file(v1_policy) != V1_POLICY_SHA256:
        raise ValueError("Sensitivity v1 preprocessing policy SHA256 changed")
    if sha256_file(v2_policy) != TARGET_POLICY_SHA256:
        raise ValueError("Sensitivity v2 revalidation policy SHA256 changed")
    policy = yaml.safe_load(v1_policy.read_text(encoding="utf-8"))
    if not isinstance(policy, Mapping) or (
        policy.get("sampling_frame_id") != SAMPLING_FRAME_ID
        or policy.get("analysis_role") != ANALYSIS_ROLE
        or int(policy.get("expected_libraries", -1)) != EXPECTED_LIBRARIES
        or policy.get("real_pathway_results_inspected") is not False
    ):
        raise ValueError("Sensitivity v1 policy violates the frozen frame contract")
    if validate_raw_axis:
        migration = validate_cd235a_neg_cell_call_migration(
            paths["v2_migration"], repository_root=root
        )
    else:
        migration = json.loads(paths["v2_migration"].read_text(encoding="utf-8"))
        if (
            migration.get("status") != "pass_outcome_blind_hash_bound_revalidation"
            or migration.get("real_pathway_results_inspected") is not False
            or migration.get("sampling_frame_id") != SAMPLING_FRAME_ID
        ):
            raise ValueError("Sensitivity v2 migration record is invalid")
        ledger_record = migration.get("revalidation_ledger", {})
        if sha256_file(paths["v2_revalidation"]) != ledger_record.get("sha256"):
            raise ValueError("Sensitivity v2 revalidation ledger SHA256 changed")
    adjudication = json.loads(
        paths["cell_call_adjudication"].read_text(encoding="utf-8")
    )
    if (
        adjudication.get("sampling_frame_id") != SAMPLING_FRAME_ID
        or adjudication.get("analysis_role") != ANALYSIS_ROLE
        or adjudication.get("outcome_blinded") is not True
        or adjudication.get("real_pathway_results_inspected") is not False
        or int(adjudication.get("n_libraries", -1)) != EXPECTED_LIBRARIES
        or adjudication.get("sensitivity_v2_migration", {}).get("validation", {}).get(
            "status"
        )
        != "pass"
    ):
        raise ValueError("Sensitivity cell-call adjudication is not canonical")
    ledger = pd.read_csv(
        paths["shard_ledger"], sep="\t", dtype=str, keep_default_na=False
    )
    spec = build_assembly_spec_from_ledger(ledger)
    for row in ledger.itertuples(index=False):
        shard = root / str(row.shard)
        if (
            not shard.is_file()
            or shard.resolve().parent != paths["shard_root"].resolve()
            or sha256_file(shard) != str(row.shard_sha256).lower()
        ):
            raise ValueError(f"Sensitivity shard path/hash changed: {row.library_id}")
    return {
        "status": "pass",
        "sampling_frame_id": SAMPLING_FRAME_ID,
        "analysis_role": ANALYSIS_ROLE,
        "real_pathway_outcomes_read": False,
        "n_libraries": EXPECTED_LIBRARIES,
        "n_cells": EXPECTED_CELLS,
        "n_features": EXPECTED_FEATURES,
        "assembly_spec": spec,
        "ledger": ledger,
        "migration_id": str(migration.get("migration_id", "")),
        "input_bindings": {
            "v1_policy": _file_record(v1_policy, root),
            "v2_policy": _file_record(v2_policy, root),
            "v2_migration": _file_record(paths["v2_migration"], root),
            "v2_revalidation": _file_record(paths["v2_revalidation"], root),
            "cell_call_adjudication": _file_record(
                paths["cell_call_adjudication"], root
            ),
            "count_shard_ledger": _file_record(paths["shard_ledger"], root),
        },
    }


def prepare_sensitivity_assembly_inputs(
    repository_root: str | Path,
    *,
    validate_raw_axis: bool = True,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    paths = _namespace_paths(root)
    evidence = validate_sensitivity_preprocessing_evidence(
        root, validate_raw_axis=validate_raw_axis
    )
    for output in (paths["assembly_spec"], paths["batch_definition"]):
        if output.exists():
            raise FileExistsError(f"Immutable sensitivity design input exists: {output}")
    spec = evidence["assembly_spec"]
    batch = build_batch_definition(spec["library_id"].astype(str).tolist())
    _atomic_tsv(spec, paths["assembly_spec"])
    _atomic_tsv(batch, paths["batch_definition"])
    return {
        "status": "pass_prepared",
        "sampling_frame_id": SAMPLING_FRAME_ID,
        "real_pathway_outcomes_read": False,
        "assembly_spec": _file_record(paths["assembly_spec"], root),
        "batch_definition": _file_record(paths["batch_definition"], root),
        "input_bindings": evidence["input_bindings"],
    }


def _concat_on_disk(
    inputs: Sequence[Path], output: Path, *, reference_var: pd.DataFrame
) -> ad.AnnData:
    from anndata.experimental import concat_on_disk

    concat_on_disk(inputs, output, axis=0, join="inner", merge=None, uns_merge=None)
    assembled = ad.read_h5ad(output)
    expected_names = reference_var.index.astype(str).to_numpy()
    if not np.array_equal(assembled.var_names.astype(str).to_numpy(), expected_names):
        raise ValueError("Sensitivity on-disk concatenation changed the feature axis")
    assembled.var = reference_var.copy(deep=True)
    return assembled


def _git_state(repository_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def assemble_sensitivity_candidate(
    repository_root: str | Path,
    *,
    candidate_id: str,
    annotation_path: str | Path,
    adjudication_record_path: str | Path,
    mapping_path: str | Path,
    annotation_plan_path: str | Path,
    gene_map_path: str | Path,
    validate_raw_axis: bool = False,
    command: Sequence[str] | None = None,
) -> Path:
    root = Path(repository_root).resolve()
    paths = _namespace_paths(root)
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{7,127}", candidate_id):
        raise ValueError("Sensitivity candidate_id is not a safe immutable identifier")
    candidate = paths["candidate_root"] / candidate_id
    if candidate.exists():
        raise FileExistsError(f"Immutable sensitivity candidate exists: {candidate}")
    evidence = validate_sensitivity_preprocessing_evidence(
        root, validate_raw_axis=validate_raw_axis
    )
    if not paths["assembly_spec"].is_file() or not paths["batch_definition"].is_file():
        raise FileNotFoundError("Sensitivity assembly spec/batch definition is missing")
    spec = pd.read_csv(
        paths["assembly_spec"], sep="\t", dtype=str, keep_default_na=False
    )
    expected_spec = evidence["assembly_spec"].astype(str)
    observed_spec = spec.astype(str)
    if list(observed_spec.columns) != list(expected_spec.columns) or not observed_spec.equals(
        expected_spec
    ):
        raise ValueError("Sensitivity assembly spec differs from the shard ledger")
    batch = pd.read_csv(
        paths["batch_definition"], sep="\t", dtype=str, keep_default_na=False
    )
    expected_batch = build_batch_definition(spec["library_id"].tolist()).astype(str)
    if list(batch.columns) != list(expected_batch.columns) or not batch.astype(str).equals(
        expected_batch
    ):
        raise ValueError("Sensitivity batch definition differs from the frozen omission")

    annotation_path = Path(annotation_path).resolve()
    adjudication_record_path = Path(adjudication_record_path).resolve()
    mapping_path = Path(mapping_path).resolve()
    annotation_plan_path = Path(annotation_plan_path).resolve()
    gene_map_path = Path(gene_map_path).resolve()
    record, annotations, mapping = validate_reference_annotation_bundle(
        repository_root=root,
        annotation_path=annotation_path,
        adjudication_record_path=adjudication_record_path,
        mapping_path=mapping_path,
        annotation_plan_path=annotation_plan_path,
        expected_sampling_frame_id=SAMPLING_FRAME_ID,
        expected_assembly_sampling_frame="cd235a_neg",
    )
    if (
        record.get("sampling_frame_id") != SAMPLING_FRAME_ID
        or len(annotations) != EXPECTED_CELLS
    ):
        raise ValueError("Sensitivity annotation bundle has another frame or cell count")
    plan = yaml.safe_load(annotation_plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, Mapping) or plan.get("sampling_frame_id") != SAMPLING_FRAME_ID:
        raise ValueError("Sensitivity annotation plan has another frame")
    validate_reference_annotation_semantics(annotations, plan=plan, mapping=mapping)
    annotations["lineage_inclusion"] = _strict_bool(
        annotations["lineage_inclusion"], "annotations.lineage_inclusion"
    )
    annotation_index = annotations.assign(
        cell_id=annotations["cell_id"].astype(str)
    ).set_index("cell_id", drop=False)
    if annotation_index.index.duplicated().any():
        raise ValueError("Sensitivity annotations contain duplicate cell IDs")

    gene_map = pd.read_csv(
        gene_map_path, sep="\t", dtype=str, keep_default_na=False
    )
    if len(gene_map) != EXPECTED_FEATURES or gene_map["gene_id_original"].duplicated().any():
        raise ValueError("Sensitivity gene map is not the frozen 33,538-feature axis")
    gene_map["is_chr21"] = _strict_bool(gene_map["is_chr21"], "gene_map.is_chr21")
    gene_index = gene_map.set_index("gene_id_original", drop=False)

    paths["candidate_root"].mkdir(parents=True, exist_ok=True)
    temporary = paths["candidate_root"] / f".tmp-{candidate_id}-{uuid4().hex}"
    temporary.mkdir(parents=False, exist_ok=False)
    annotated_root = temporary / "annotated_shards"
    annotated_root.mkdir()
    annotated_paths: list[Path] = []
    shard_records: list[dict[str, Any]] = []
    all_cells: set[str] = set()
    reference_var: pd.DataFrame | None = None
    metadata = {
        "schema_version": "1.0.0",
        "stage": "diagnostic_sensitivity_analysis_ready_scrna_candidate",
        "candidate_id": candidate_id,
        "sampling_frame_id": SAMPLING_FRAME_ID,
        "analysis_role": ANALYSIS_ROLE,
        "pooling_with_primary_allowed": False,
        "primary_discovery_claim_allowed": False,
        "real_pathway_outcomes_inspected": False,
        "assembly_spec_sha256": sha256_file(paths["assembly_spec"]),
        "batch_definition_sha256": sha256_file(paths["batch_definition"]),
        "annotations_sha256": sha256_file(annotation_path),
        "annotation_plan_sha256": sha256_file(annotation_plan_path),
        "annotation_adjudication_sha256": sha256_file(adjudication_record_path),
        "reference_label_mapping_sha256": sha256_file(mapping_path),
        "gene_map_sha256": sha256_file(gene_map_path),
        "preprocessing_evidence": evidence["input_bindings"],
        "expression_transform": formal_expression_preprocessing_contract(),
        "expression_implementation_sha256": formal_expression_preprocessing_source_sha256(),
    }
    try:
        ledger_index = evidence["ledger"].set_index("library_id", drop=False)
        for row in spec.itertuples(index=False):
            library_id = str(row.library_id)
            ledger_row = ledger_index.loc[library_id]
            shard_path = root / str(ledger_row["shard"])
            shard = ad.read_h5ad(shard_path)
            if shard.shape[1] != EXPECTED_FEATURES:
                raise ValueError(f"Sensitivity shard feature count changed: {library_id}")
            cell_ids = shard.obs_names.astype(str).tolist()
            if all_cells.intersection(cell_ids):
                raise ValueError("Sensitivity shards contain duplicate global cell IDs")
            if not set(cell_ids).issubset(annotation_index.index):
                raise ValueError(f"Sensitivity annotations omit cells from {library_id}")
            joined = annotation_index.loc[cell_ids]
            for key in ("library_id", "original_barcode"):
                if not np.array_equal(
                    shard.obs[key].astype(str).to_numpy(),
                    joined[key].astype(str).to_numpy(),
                ):
                    raise ValueError(f"Sensitivity annotation {key} join changed")
            copy_columns = {
                "original_cell_type",
                "original_cell_type_source",
                "analysis_cell_type",
                "lineage_inclusion",
                "lineage_inclusion_reason",
                *REFERENCE_AUDIT_COLUMNS,
            }
            for column in copy_columns:
                shard.obs[column] = joined[column].to_numpy()
            shard.obs["lineage_inclusion"] = shard.obs["lineage_inclusion"].astype(bool)
            shard.obs["annotation_uncertain"] = shard.obs["annotation_uncertain"].astype(bool)
            shard.obs["predicted_doublet"] = shard.obs["predicted_doublet"].astype(bool)
            shard.obs["mapping_confidence"] = pd.to_numeric(
                shard.obs["mapping_confidence"], errors="raise"
            ).astype(float)
            shard.obs["technical_batch"] = "omitted_not_identifiable"
            var_names = shard.var_names.astype(str)
            if set(var_names) != set(gene_index.index) or len(var_names) != len(gene_index):
                raise ValueError("Sensitivity shard feature IDs differ from gene map")
            mapped_genes = gene_index.loc[var_names]
            for column in gene_map.columns:
                if column != "gene_id_original":
                    shard.var[column] = mapped_genes[column].to_numpy()
            shard.var["is_chr21"] = shard.var["is_chr21"].astype(bool)
            if reference_var is None:
                reference_var = shard.var.copy(deep=True)
            elif not shard.var.equals(reference_var):
                raise ValueError("Sensitivity annotated shards have different var metadata")
            shard.X = normalize_t21_formal_expression(shard.layers["counts"])
            shard_metadata = dict(metadata)
            shard_metadata["expression_contract"] = expression_contract_metadata(
                shard.X, counts=shard.layers["counts"]
            )
            shard.uns["t21_data_product"] = shard_metadata
            validate_scrna_contract(
                shard,
                strict_analysis_labels=True,
                require_formal_expression=True,
            )
            annotated_path = annotated_root / f"{library_id}.annotated.h5ad"
            shard.write_h5ad(annotated_path, compression="gzip")
            annotated_paths.append(annotated_path)
            all_cells.update(cell_ids)
            shard_records.append(
                {
                    "library_id": library_id,
                    "n_cells": int(shard.n_obs),
                    "source": _file_record(shard_path, root),
                }
            )
        if set(annotation_index.index) != all_cells or len(all_cells) != EXPECTED_CELLS:
            raise ValueError("Sensitivity annotations and shard cell set do not close")
        if reference_var is None:
            raise RuntimeError("Sensitivity assembly produced no annotated shard")
        output = temporary / ASSEMBLED_H5AD_NAME
        assembled = _concat_on_disk(
            annotated_paths, output, reference_var=reference_var
        )
        metadata["expression_contract"] = expression_contract_metadata(
            assembled.X, counts=assembled.layers["counts"]
        )
        assembled.uns["t21_data_product"] = metadata
        assembled.write_h5ad(output, compression="gzip")
        contract = validate_scrna_contract(
            assembled,
            strict_analysis_labels=True,
            require_formal_expression=True,
        )
        conditions = assembled.obs[["donor_id", "condition"]].drop_duplicates()
        if conditions["donor_id"].duplicated().any():
            raise ValueError("A sensitivity donor has multiple conditions")
        counts = conditions["condition"].value_counts()
        if (
            len(conditions) != EXPECTED_DONORS
            or int(counts.get("T21", 0)) != EXPECTED_CASES
            or int(counts.get("disomy", 0)) != EXPECTED_CONTROLS
        ):
            raise ValueError("Sensitivity assembled donor design differs from 13:3")
        shutil.rmtree(annotated_root)
        code_paths = [Path(__file__).resolve()]
        code_bindings = [
            {"path": _relative(path, root), "sha256": sha256_file(path)}
            for path in code_paths
        ]
        output_record = artifact_record(output, root, "scrna")
        output_record["relative_path"] = _relative(
            candidate / ASSEMBLED_H5AD_NAME, root
        )
        assembly_record = {
            "schema_name": "t21_sensitivity_scrna_assembly_record",
            "schema_version": "1.0.0",
            "candidate_id": candidate_id,
            "created_at_utc": utc_now(),
            "sampling_frame_id": SAMPLING_FRAME_ID,
            "analysis_role": ANALYSIS_ROLE,
            "outcome_blinded": True,
            "real_pathway_outcomes_read": False,
            "pooling_with_primary_allowed": False,
            "formal_release_allowed": False,
            "command": list(command or sys.argv),
            "git": _git_state(root),
            "code_bindings": code_bindings,
            "implementation_sha256": sha256(
                stable_json(code_bindings).encode("utf-8")
            ).hexdigest(),
            "inputs": {
                "assembly_spec": _file_record(paths["assembly_spec"], root),
                "batch_definition": _file_record(paths["batch_definition"], root),
                "annotations": _file_record(annotation_path, root),
                "annotation_adjudication": _file_record(
                    adjudication_record_path, root
                ),
                "reference_mapping": _file_record(mapping_path, root),
                "annotation_plan": _file_record(annotation_plan_path, root),
                "gene_map": _file_record(gene_map_path, root),
                **evidence["input_bindings"],
            },
            "shards": shard_records,
            "output": output_record,
            "contract": contract,
            "assembly_metadata": metadata,
        }
        _atomic_json(assembly_record, temporary / ASSEMBLY_RECORD_NAME)
        os.replace(temporary, candidate)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    return candidate


def _validate_persisted_expression_binding(
    adata: ad.AnnData, assembly_record: Mapping[str, Any]
) -> None:
    """Validate an immutable expression contract without materializing backed X.

    The assembly builder performs value-level validation before publication and
    binds the exact H5AD by SHA256.  A backed AnnData exposes ``X`` as an anndata
    CSR dataset rather than a scipy CSR matrix, so the in-memory validator cannot
    be applied to that container directly.  Verify the persisted encoding and the
    exact contract/hash bindings recorded at publication instead.
    """

    expression = adata.X
    if (
        getattr(expression, "format", None) != "csr"
        or np.dtype(expression.dtype) != np.dtype("float32")
        or tuple(expression.shape) != tuple(adata.shape)
    ):
        raise ValueError("Sensitivity backed expression is not persisted float32 CSR")
    metadata = adata.uns.get("t21_data_product", {})
    persisted = metadata.get("expression_contract")
    recorded = assembly_record.get("assembly_metadata", {}).get(
        "expression_contract"
    )
    if not isinstance(persisted, Mapping) or stable_json(persisted) != stable_json(
        recorded
    ):
        raise ValueError("Sensitivity persisted expression contract changed")
    validation = persisted.get("validation", {})
    contract = assembly_record.get("contract", {})
    expected = {
        "formal_expression_validated": True,
        "expression_contract_version": validation.get("schema_version"),
        "expression_contract_sha256": validation.get("contract_sha256"),
        "expression_implementation_sha256": validation.get(
            "implementation_source_sha256"
        ),
        "x_semantic_sha256": validation.get("expression_csr_semantic_sha256"),
        "n_cells": int(adata.n_obs),
        "n_features": int(adata.n_vars),
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"Sensitivity assembly expression binding changed for {key}")


def build_sensitivity_trajectory(
    repository_root: str | Path,
    *,
    candidate_dir: str | Path,
    donor_design_base_path: str | Path,
    plan_path: str | Path,
    analysis_plan_path: str | Path,
    cli_path: str | Path,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    paths = _namespace_paths(root)
    candidate = Path(candidate_dir).resolve()
    if candidate.parent != paths["candidate_root"].resolve():
        raise ValueError("Sensitivity trajectory candidate is outside its namespace")
    h5ad_path = candidate / ASSEMBLED_H5AD_NAME
    assembly_record_path = candidate / ASSEMBLY_RECORD_NAME
    if not h5ad_path.is_file() or not assembly_record_path.is_file():
        raise FileNotFoundError("Sensitivity assembly candidate is incomplete")
    assembly_record = json.loads(assembly_record_path.read_text(encoding="utf-8"))
    if (
        assembly_record.get("sampling_frame_id") != SAMPLING_FRAME_ID
        or assembly_record.get("analysis_role") != ANALYSIS_ROLE
        or assembly_record.get("real_pathway_outcomes_read") is not False
        or assembly_record.get("formal_release_allowed") is not False
        or assembly_record.get("output", {}).get("sha256") != sha256_file(h5ad_path)
    ):
        raise ValueError("Sensitivity assembly record violates the diagnostic contract")
    existing = set(path.name for path in candidate.iterdir())
    allowed = {ASSEMBLED_H5AD_NAME, ASSEMBLY_RECORD_NAME}
    if existing != allowed:
        raise FileExistsError("Sensitivity candidate already contains trajectory artifacts")

    def h5ad_validator(adata: ad.AnnData) -> None:
        summary = validate_scrna_contract(
            adata, strict_analysis_labels=True, require_formal_expression=False
        )
        _validate_persisted_expression_binding(adata, assembly_record)
        metadata = adata.uns.get("t21_data_product", {})
        if (
            int(summary["n_cells"]) != EXPECTED_CELLS
            or metadata.get("sampling_frame_id") != SAMPLING_FRAME_ID
            or metadata.get("analysis_role") != ANALYSIS_ROLE
            or metadata.get("real_pathway_outcomes_inspected") is not False
        ):
            raise ValueError("Sensitivity H5AD violates trajectory input scope")

    temporary = candidate.parent / f".tmp-{candidate.name}-trajectory-{uuid4().hex}"
    temporary.mkdir(parents=False, exist_ok=False)
    code_bindings = _capture_trajectory_code_bindings(
        root, cli_path=Path(cli_path).resolve()
    )
    try:
        record = build_t21_trajectory_fate_products(
            h5ad_path=h5ad_path,
            donor_design_base_path=donor_design_base_path,
            plan_path=plan_path,
            analysis_plan_path=analysis_plan_path,
            output_dir=temporary,
            repository_root=root,
            prebound_code_bindings=code_bindings,
            command=list(command or sys.argv),
            cli_path=Path(cli_path).resolve(),
            expected_h5ad_sha256=sha256_file(h5ad_path),
            h5ad_validator=h5ad_validator,
            logical_input_names={
                "assembled_scrna_h5ad": ASSEMBLED_H5AD_NAME,
                "base_donor_design": Path(donor_design_base_path).name,
                "frozen_trajectory_fate_plan": Path(plan_path).name,
                "master_t21_analysis_plan": Path(analysis_plan_path).name,
            },
        )
        adata = ad.read_h5ad(h5ad_path, backed="r")
        try:
            validate_built_trajectory_fate_directory(
                temporary, scrna_obs=adata.obs
            )
        finally:
            adata.file.close()
        for output in temporary.iterdir():
            destination = candidate / output.name
            if destination.exists():
                raise FileExistsError(destination)
            os.replace(output, destination)
        temporary.rmdir()
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    return record


def validate_sensitivity_candidate(
    repository_root: str | Path, *, candidate_dir: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    paths = _namespace_paths(root)
    candidate = Path(candidate_dir).resolve()
    if candidate.parent != paths["candidate_root"].resolve():
        raise ValueError("Sensitivity candidate is outside its namespace")
    h5ad_path = candidate / ASSEMBLED_H5AD_NAME
    record_path = candidate / ASSEMBLY_RECORD_NAME
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (
        record.get("sampling_frame_id") != SAMPLING_FRAME_ID
        or record.get("analysis_role") != ANALYSIS_ROLE
        or record.get("formal_release_allowed") is not False
        or record.get("real_pathway_outcomes_read") is not False
        or record.get("output", {}).get("sha256") != sha256_file(h5ad_path)
    ):
        raise ValueError("Sensitivity assembly record is invalid")
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        scrna = validate_scrna_contract(
            adata, strict_analysis_labels=True, require_formal_expression=False
        )
        _validate_persisted_expression_binding(adata, record)
        scrna = dict(scrna)
        scrna["formal_expression_validated"] = True
        scrna["x_semantic_sha256"] = record["contract"]["x_semantic_sha256"]
        scrna["formal_expression_validation_mode"] = (
            "publication_value_validation_plus_exact_file_sha256_and_"
            "persisted_contract_binding"
        )
        trajectory = validate_built_trajectory_fate_directory(
            candidate, scrna_obs=adata.obs
        )
    finally:
        adata.file.close()
    return {
        "status": "pass_diagnostic_sensitivity_candidate",
        "sampling_frame_id": SAMPLING_FRAME_ID,
        "analysis_role": ANALYSIS_ROLE,
        "real_pathway_outcomes_read": False,
        "formal_release_allowed": False,
        "scrna": scrna,
        "trajectory": trajectory,
    }
