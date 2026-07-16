from __future__ import annotations

from datetime import datetime, timezone
import gzip
from hashlib import sha256
from io import StringIO
import json
from math import comb
import os
from pathlib import Path
import re
from typing import Any, Mapping
from uuid import uuid4

import pandas as pd
import yaml


POLICY_SCHEMA_NAME = "t21_cd235a_neg_cell_call_revalidation_policy"
POLICY_SCHEMA_VERSION = "2.0.0"
MIGRATION_SCHEMA_NAME = "t21_cd235a_neg_cell_call_policy_migration"
MIGRATION_SCHEMA_VERSION = "2.0.0"

TARGET_POLICY_RELATIVE_PATH = (
    "config/t21_preprocessing_adjudication_cd235a_neg_sensitivity_v2.yaml"
)
TARGET_POLICY_SHA256 = (
    "2f7e109e61f682c8d4783479f3ba7a9cc0ed9d9ed0e35a9e03c906f5ab0ec074"
)
SOURCE_V1_POLICY_SHA256 = (
    "58fb284601059cb195aaa4264bd8a439c805878f04a537f34b893603c7f0a0c4"
)
SOURCE_V1_LEDGER_SHA256 = (
    "b9611e4c6aad7303e89c0285f2ca4adf4f7f646b4d35893483301a3bceaa4c14"
)
CLI_RELATIVE_PATH = "scripts/migrate_t21_sensitivity_cell_calls_v2.py"
MODULE_RELATIVE_PATH = "pyfgsea/t21_sensitivity_cell_call_migration.py"

FORMAL_ACCEPTANCE_GATES = (
    "selected_cells_positive",
    "selected_barcodes_nonempty_and_unique",
    "each_selected_barcode_occurs_exactly_once_on_hash_bound_raw_barcode_axis",
)
HASH_RECHECK_CHECKPOINTS = (
    "before_revalidation",
    "after_revalidation_before_publish",
    "after_publish",
)
REVALIDATION_COLUMNS = (
    "library_id",
    "donor_id",
    "condition",
    "run_id",
    "source_status",
    "migration_status",
    "n_selected",
    "selected_barcodes_nonempty",
    "selected_barcodes_unique",
    "source_call_relative_path",
    "source_call_bytes",
    "source_call_sha256_before",
    "source_call_sha256_after",
    "raw_barcode_axis_relative_path",
    "raw_barcode_axis_bytes",
    "raw_barcode_axis_sha256_bound_by_source_ledger",
    "raw_barcode_axis_sha256_before",
    "raw_barcode_axis_sha256_after",
    "raw_barcode_axis_rows_scanned",
    "selected_missing_from_raw_axis",
    "selected_repeated_on_raw_axis",
    "selected_exactly_once_on_raw_axis",
    "author_isolated_cells",
    "selected_minus_author_isolated",
    "selected_exceeds_author_isolated",
    "author_isolated_cells_role",
    "formal_acceptance_pass",
    "calls_recomputed",
    "outcome_blinded",
    "real_pathway_results_inspected",
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SOURCE_LEDGER_COLUMNS = {
    "library_id",
    "status",
    "n_selected",
    "output",
    "output_sha256",
    "input_signature_schema",
    "preprocessing_policy_sha256",
    "run_id",
    "sampling_frame",
    "tissue",
    "barcodes_sha256",
}
_CALL_COLUMNS = {"barcode", "selected"}


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sha256(value: Any, label: str) -> str:
    digest = str(value).lower()
    if not _SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return digest


def _strict_int(value: Any, label: str) -> int:
    text = str(value).strip()
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError(f"{label} must be a non-negative integer")
    return int(text)


def _resolve_relative(
    relative_path: Any,
    root: Path,
    *,
    label: str,
    require_file: bool = True,
) -> Path:
    value = Path(str(relative_path))
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"{label} path escapes the repository")
    resolved = (root / value).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes the repository") from exc
    if require_file and not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("Evidence path escapes the repository") from exc


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Sensitivity cell-call v2 policy must be one YAML mapping")
    return value


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def _policy_outputs(policy: Mapping[str, Any], root: Path) -> tuple[Path, Path]:
    outputs = _mapping(policy.get("outputs"), "Policy outputs")
    record = _resolve_relative(
        outputs.get("migration_json"),
        root,
        label="Migration JSON",
        require_file=False,
    )
    ledger = _resolve_relative(
        outputs.get("revalidation_tsv"),
        root,
        label="Revalidation TSV",
        require_file=False,
    )
    if record == ledger:
        raise ValueError("Migration JSON and revalidation TSV paths overlap")
    return record, ledger


def load_frozen_v2_policy(
    policy_path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    path = Path(policy_path).resolve()
    if _relative(path, root) != TARGET_POLICY_RELATIVE_PATH:
        raise ValueError("Sensitivity cell-call migration requires the frozen v2 policy")
    observed_policy_sha = sha256_file(path)
    if observed_policy_sha != TARGET_POLICY_SHA256:
        raise ValueError("Sensitivity cell-call v2 policy differs from its frozen SHA256")
    policy = _read_yaml(path)
    if (
        policy.get("schema_name") != POLICY_SCHEMA_NAME
        or policy.get("schema_version") != POLICY_SCHEMA_VERSION
        or policy.get("policy_id")
        != "t21_cd235a_neg_pinned_public_reconstruction_sensitivity_v2"
        or policy.get("outcome_blinded_at_freeze") is not True
        or policy.get("real_pathway_results_inspected") is not False
        or policy.get("pathway_artifacts_read_for_amendment") is not False
        or policy.get("cell_call_counts_inspected_for_amendment") is not True
        or policy.get("isolated_cell_count_exceedances_inspected_for_amendment")
        is not True
        or policy.get("analysis_role") != "sensitivity_only"
        or policy.get("artifact_namespace")
        != "t21_fetal_liver_cd235a_neg_sensitivity_v1"
        or policy.get("sampling_frame_id")
        != "t21_fetal_liver_cd235a_neg_sensitivity_v1"
        or policy.get("accession") != "E-MTAB-13067"
        or policy.get("sampling_frame") != "cd235a_neg"
        or policy.get("tissue") != "liver"
        or int(policy.get("expected_libraries", -1)) != 18
    ):
        raise ValueError("Sensitivity cell-call v2 policy changes its fixed frame")

    expected_design = {
        "expected_donors": 16,
        "expected_t21_donors": 13,
        "expected_disomy_donors": 3,
        "exact_condition_label_assignments": 560,
        "ledger_unit": "library",
        "donor_design_role": "fixed_gate_sensitivity_only",
        "pooling_with_primary_allowed": False,
    }
    if dict(_mapping(policy.get("design_metadata"), "Design metadata")) != expected_design:
        raise ValueError("Sensitivity cell-call v2 policy changes its donor design")

    source = _mapping(policy.get("source_evidence"), "Source evidence")
    source_policy = _mapping(source.get("v1_policy"), "Source v1 policy")
    source_ledger = _mapping(
        source.get("v1_diagnostic_ledger"), "Source v1 diagnostic ledger"
    )
    if (
        source.get("calls_recomputed") is not False
        or source.get("calls_modified") is not False
        or _sha256(source_policy.get("sha256"), "Source v1 policy SHA256")
        != SOURCE_V1_POLICY_SHA256
        or _sha256(source_ledger.get("sha256"), "Source v1 ledger SHA256")
        != SOURCE_V1_LEDGER_SHA256
        or source.get("source_run_id") != "20260713T172320Z_54304"
        or source.get("expected_input_signature_schema")
        != "t21_cell_call_inputs_v2"
        or list(source.get("accepted_source_statuses", []))
        != [
            "diagnostic_author_count_matched",
            "diagnostic_author_count_mismatch",
        ]
    ):
        raise ValueError("Sensitivity v2 policy changes its immutable v1 evidence")

    revalidation = _mapping(
        policy.get("cell_call_revalidation"), "Cell-call revalidation"
    )
    raw_axis = _mapping(revalidation.get("raw_barcode_axis"), "Raw barcode axis")
    isolated = _mapping(
        revalidation.get("author_isolated_cells"), "Author isolated cells"
    )
    if (
        tuple(revalidation.get("formal_acceptance_gates", []))
        != FORMAL_ACCEPTANCE_GATES
        or set(revalidation.get("source_table_required_columns", []))
        != _CALL_COLUMNS
        or revalidation.get("selected_field_requirement")
        != "every_published_row_is_selected"
        or raw_axis.get("format")
        != "gzip_or_plain_utf8_tsv_first_column_no_header"
        or raw_axis.get("source_ledger_sha256_column") != "barcodes_sha256"
        or raw_axis.get("occurrence_requirement")
        != "exactly_once_per_selected_barcode"
        or raw_axis.get("nonselected_barcode_uniqueness_required") is not False
        or isolated.get("role") != "diagnostic_only"
        or isolated.get("formal_upper_bound") is not False
        or isolated.get("exceedance_is_a_failure") is not False
        or isolated.get("record_difference_and_exceedance") is not True
    ):
        raise ValueError("Sensitivity v2 policy changes its formal acceptance gates")

    integrity = _mapping(policy.get("integrity"), "Integrity contract")
    if (
        integrity.get("before_after_hash_recheck_required") is not True
        or tuple(integrity.get("recheck_checkpoints", []))
        != HASH_RECHECK_CHECKPOINTS
        or integrity.get("source_files_may_be_modified") is not False
        or integrity.get("output_files_must_not_preexist") is not True
    ):
        raise ValueError("Sensitivity v2 policy weakens hash revalidation")
    interpretation = _mapping(policy.get("interpretation"), "Interpretation")
    if (
        interpretation.get("author_pipeline_exact_reproduction_claim_allowed")
        is not False
        or interpretation.get("primary_discovery_claim_allowed") is not False
        or interpretation.get("pooling_with_primary_allowed") is not False
        or interpretation.get("isolated_cell_count_may_affect_acceptance") is not False
        or interpretation.get("pathway_outcomes_may_influence_revalidation") is not False
    ):
        raise ValueError("Sensitivity v2 policy would fail open or overclaim")

    locked = _mapping(policy.get("locked_metadata"), "Locked metadata")
    for key in ("library_manifest", "triplet_audit"):
        entry = _mapping(locked.get(key), f"Locked metadata {key}")
        _sha256(entry.get("sha256"), f"Locked metadata {key} SHA256")
        _resolve_relative(entry.get("relative_path"), root, label=key)
    _resolve_relative(
        locked.get("raw_data_root"),
        root,
        label="Raw data root",
        require_file=False,
    )
    _resolve_relative(source_policy.get("relative_path"), root, label="Source v1 policy")
    _resolve_relative(
        source_ledger.get("relative_path"), root, label="Source v1 diagnostic ledger"
    )
    _resolve_relative(
        source.get("source_output_root"),
        root,
        label="Source output root",
        require_file=False,
    )
    _policy_outputs(policy, root)
    return policy


def _add_watch(
    watched: dict[str, dict[str, Any]],
    role: str,
    path: Path,
    root: Path,
) -> dict[str, Any]:
    if role in watched:
        raise ValueError(f"Duplicate watched role: {role}")
    if not path.is_file():
        raise FileNotFoundError(path)
    entry = {
        "relative_path": _relative(path, root),
        "bytes": path.stat().st_size,
        "sha256_before": sha256_file(path),
    }
    watched[role] = entry
    return entry


def _recheck_watched(
    watched: Mapping[str, Mapping[str, Any]], root: Path, *, checkpoint: str
) -> None:
    for role, entry in watched.items():
        path = _resolve_relative(entry["relative_path"], root, label=role)
        if (
            path.stat().st_size != int(entry["bytes"])
            or sha256_file(path) != entry["sha256_before"]
        ):
            raise RuntimeError(f"Watched input changed at {checkpoint}: {role}")


def _binding(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "relative_path": str(entry["relative_path"]),
        "bytes": int(entry["bytes"]),
        "sha256_before": str(entry["sha256_before"]),
        "sha256_after": str(entry["sha256_before"]),
    }


def _load_context(
    policy_path: Path, root: Path, cli_path: Path
) -> dict[str, Any]:
    policy = load_frozen_v2_policy(policy_path, repository_root=root)
    cli_path = cli_path.resolve()
    module_path = Path(__file__).resolve()
    if _relative(cli_path, root) != CLI_RELATIVE_PATH or not cli_path.is_file():
        raise ValueError("Migration CLI is not the repository-frozen implementation")
    if _relative(module_path, root) != MODULE_RELATIVE_PATH:
        raise ValueError("Migration module is not repository controlled")

    source = _mapping(policy["source_evidence"], "Source evidence")
    locked = _mapping(policy["locked_metadata"], "Locked metadata")
    source_policy_entry = _mapping(source["v1_policy"], "Source v1 policy")
    source_ledger_entry = _mapping(
        source["v1_diagnostic_ledger"], "Source v1 diagnostic ledger"
    )
    manifest_entry = _mapping(locked["library_manifest"], "Library manifest")
    triplet_entry = _mapping(locked["triplet_audit"], "Triplet audit")
    paths = {
        "target_v2_policy": policy_path.resolve(),
        "source_v1_policy": _resolve_relative(
            source_policy_entry["relative_path"], root, label="Source v1 policy"
        ),
        "source_v1_diagnostic_ledger": _resolve_relative(
            source_ledger_entry["relative_path"],
            root,
            label="Source v1 diagnostic ledger",
        ),
        "locked_library_manifest": _resolve_relative(
            manifest_entry["relative_path"], root, label="Library manifest"
        ),
        "locked_triplet_audit": _resolve_relative(
            triplet_entry["relative_path"], root, label="Triplet audit"
        ),
        "migration_module": module_path,
        "migration_cli": cli_path,
    }
    watched: dict[str, dict[str, Any]] = {}
    for role, path in paths.items():
        _add_watch(watched, role, path, root)
    expected_hashes = {
        "target_v2_policy": TARGET_POLICY_SHA256,
        "source_v1_policy": SOURCE_V1_POLICY_SHA256,
        "source_v1_diagnostic_ledger": SOURCE_V1_LEDGER_SHA256,
        "locked_library_manifest": _sha256(
            manifest_entry["sha256"], "Library manifest SHA256"
        ),
        "locked_triplet_audit": _sha256(
            triplet_entry["sha256"], "Triplet audit SHA256"
        ),
    }
    for role, expected in expected_hashes.items():
        if watched[role]["sha256_before"] != expected:
            raise ValueError(f"Hash-bound input differs: {role}")

    libraries = _read_tsv(paths["locked_library_manifest"])
    required_library_columns = {
        "accession",
        "library_id",
        "donor_id",
        "condition",
        "tissue",
        "sort_gate",
        "author_isolated_cells",
    }
    missing = sorted(required_library_columns.difference(libraries.columns))
    if missing:
        raise ValueError(f"Library manifest is missing columns: {missing}")
    selected = libraries[
        libraries["accession"].eq(policy["accession"])
        & libraries["tissue"].eq(policy["tissue"])
        & libraries["sort_gate"].eq(policy["sampling_frame"])
    ].copy()
    if "sort_gate_resolution_status" in selected:
        selected = selected[
            ~selected["sort_gate_resolution_status"].eq("conflict_unresolved")
        ].copy()
    if (
        len(selected) != int(policy["expected_libraries"])
        or selected["library_id"].duplicated().any()
    ):
        raise ValueError("Locked CD235a-negative library frame is not exact")
    donor_condition = selected[["donor_id", "condition"]].drop_duplicates()
    if donor_condition["donor_id"].duplicated().any():
        raise ValueError("One sensitivity donor has multiple condition labels")
    design = _mapping(policy["design_metadata"], "Design metadata")
    n_donors = donor_condition["donor_id"].nunique()
    n_t21 = int(donor_condition["condition"].eq("T21").sum())
    n_disomy = int(donor_condition["condition"].eq("disomy").sum())
    if (
        n_donors != int(design["expected_donors"])
        or n_t21 != int(design["expected_t21_donors"])
        or n_disomy != int(design["expected_disomy_donors"])
        or comb(n_donors, n_disomy)
        != int(design["exact_condition_label_assignments"])
    ):
        raise ValueError("Locked sensitivity donor design differs from 13/3 and 560")

    triplets = _read_tsv(paths["locked_triplet_audit"])
    required_triplet_columns = {"library_id", "triplet_status", "barcodes_file"}
    missing = sorted(required_triplet_columns.difference(triplets.columns))
    if missing:
        raise ValueError(f"Triplet audit is missing columns: {missing}")
    triplets = triplets[triplets["library_id"].isin(selected["library_id"])].copy()
    if (
        len(triplets) != len(selected)
        or triplets["library_id"].duplicated().any()
        or not triplets["triplet_status"].eq("complete").all()
    ):
        raise ValueError("Raw barcode axes do not exactly cover the locked frame")

    source_ledger = _read_tsv(paths["source_v1_diagnostic_ledger"])
    missing = sorted(_SOURCE_LEDGER_COLUMNS.difference(source_ledger.columns))
    if missing:
        raise ValueError(f"Source diagnostic ledger is missing columns: {missing}")
    expected_ids = set(selected["library_id"])
    observed_ids = set(source_ledger["library_id"])
    if (
        len(source_ledger) != len(selected)
        or source_ledger["library_id"].duplicated().any()
        or observed_ids != expected_ids
    ):
        raise ValueError("Source diagnostic ledger does not exactly cover 18 libraries")
    if (
        source_ledger["run_id"].nunique() != 1
        or not source_ledger["run_id"].eq(source["source_run_id"]).all()
        or not source_ledger["sampling_frame"].eq(policy["sampling_frame"]).all()
        or not source_ledger["tissue"].eq(policy["tissue"]).all()
        or not source_ledger["status"]
        .isin(source["accepted_source_statuses"])
        .all()
        or not source_ledger["input_signature_schema"]
        .eq(source["expected_input_signature_schema"])
        .all()
        or not source_ledger["preprocessing_policy_sha256"]
        .str.lower()
        .eq(SOURCE_V1_POLICY_SHA256)
        .all()
    ):
        raise ValueError("Source diagnostic ledger mixes policies, runs, or frames")

    source_output_root = _resolve_relative(
        source["source_output_root"],
        root,
        label="Source output root",
        require_file=False,
    )
    raw_data_root = _resolve_relative(
        locked["raw_data_root"],
        root,
        label="Raw data root",
        require_file=False,
    )
    if not source_output_root.is_dir() or not raw_data_root.is_dir():
        raise FileNotFoundError("Source output or raw data root is missing")
    record_path, revalidation_path = _policy_outputs(policy, root)
    return {
        "root": root,
        "policy": policy,
        "policy_path": policy_path.resolve(),
        "source_ledger": source_ledger,
        "libraries": selected.set_index("library_id", drop=False),
        "triplets": triplets.set_index("library_id", drop=False),
        "source_output_root": source_output_root,
        "raw_data_root": raw_data_root,
        "record_path": record_path,
        "revalidation_path": revalidation_path,
        "watched": watched,
    }


def _read_selected_barcodes(path: Path, expected_n: int, library_id: str) -> list[str]:
    calls = _read_tsv(path)
    missing = sorted(_CALL_COLUMNS.difference(calls.columns))
    if missing:
        raise ValueError(f"Cell-call table is missing columns for {library_id}: {missing}")
    barcodes = calls["barcode"].astype(str).str.strip()
    selected = calls["selected"].astype(str).str.strip().str.lower()
    if not selected.isin(["true", "false"]).all() or not selected.eq("true").all():
        raise ValueError(f"Cell-call table contains non-selected rows for {library_id}")
    if len(calls) < 1 or len(calls) != expected_n:
        raise ValueError(f"Selected cell count is not positive or ledger-bound for {library_id}")
    if barcodes.eq("").any() or barcodes.duplicated().any():
        raise ValueError(f"Selected barcodes are empty or duplicated for {library_id}")
    return barcodes.tolist()


def _scan_selected_barcode_occurrences(
    path: Path, selected_barcodes: list[str]
) -> tuple[int, int, int]:
    counts = dict.fromkeys(selected_barcodes, 0)
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    n_rows = 0
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        for line in handle:
            n_rows += 1
            barcode = line.rstrip("\r\n").split("\t", 1)[0]
            if barcode in counts:
                counts[barcode] += 1
    missing = sum(value == 0 for value in counts.values())
    repeated = sum(value > 1 for value in counts.values())
    return n_rows, missing, repeated


def _revalidate_rows(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    root = context["root"]
    watched = context["watched"]
    libraries = context["libraries"]
    triplets = context["triplets"]
    rows: list[dict[str, Any]] = []
    for _, ledger_row in context["source_ledger"].sort_values("library_id").iterrows():
        library_id = str(ledger_row["library_id"])
        library = libraries.loc[library_id]
        triplet = triplets.loc[library_id]

        call_path = _resolve_relative(
            ledger_row["output"], root, label=f"{library_id} source call table"
        )
        expected_call_path = (
            context["source_output_root"] / f"{library_id}.emptydrops.tsv"
        ).resolve()
        if call_path != expected_call_path:
            raise ValueError(f"Source call path differs from the locked run for {library_id}")
        barcode_filename = Path(str(triplet["barcodes_file"]))
        if (
            barcode_filename.is_absolute()
            or ".." in barcode_filename.parts
            or barcode_filename.name != str(triplet["barcodes_file"])
        ):
            raise ValueError(f"Raw barcode filename is unsafe for {library_id}")
        raw_axis_path = (context["raw_data_root"] / barcode_filename).resolve()
        try:
            raw_axis_path.relative_to(context["raw_data_root"])
        except ValueError as exc:
            raise ValueError(f"Raw barcode axis escapes its root for {library_id}") from exc

        call_watch = _add_watch(
            watched, f"source_call:{library_id}", call_path, root
        )
        axis_watch = _add_watch(
            watched, f"raw_barcode_axis:{library_id}", raw_axis_path, root
        )
        expected_call_sha = _sha256(
            ledger_row["output_sha256"], f"{library_id} source call SHA256"
        )
        expected_axis_sha = _sha256(
            ledger_row["barcodes_sha256"], f"{library_id} raw barcode SHA256"
        )
        if call_watch["sha256_before"] != expected_call_sha:
            raise ValueError(f"Source call table hash differs for {library_id}")
        if axis_watch["sha256_before"] != expected_axis_sha:
            raise ValueError(f"Raw barcode axis hash differs for {library_id}")

        n_selected = _strict_int(ledger_row["n_selected"], f"{library_id} n_selected")
        selected_barcodes = _read_selected_barcodes(
            call_path, n_selected, library_id
        )
        axis_rows, missing, repeated = _scan_selected_barcode_occurrences(
            raw_axis_path, selected_barcodes
        )
        if missing or repeated:
            raise ValueError(
                "Selected barcode raw-axis occurrence gate failed for "
                f"{library_id}: missing={missing}, repeated={repeated}"
            )
        isolated = _strict_int(
            library["author_isolated_cells"],
            f"{library_id} author_isolated_cells",
        )
        rows.append(
            {
                "library_id": library_id,
                "donor_id": str(library["donor_id"]),
                "condition": str(library["condition"]),
                "run_id": str(ledger_row["run_id"]),
                "source_status": str(ledger_row["status"]),
                "migration_status": "pass_hash_bound_cell_call_revalidation_v2",
                "n_selected": n_selected,
                "selected_barcodes_nonempty": True,
                "selected_barcodes_unique": True,
                "source_call_relative_path": _relative(call_path, root),
                "source_call_bytes": int(call_watch["bytes"]),
                "source_call_sha256_before": expected_call_sha,
                "source_call_sha256_after": expected_call_sha,
                "raw_barcode_axis_relative_path": _relative(raw_axis_path, root),
                "raw_barcode_axis_bytes": int(axis_watch["bytes"]),
                "raw_barcode_axis_sha256_bound_by_source_ledger": expected_axis_sha,
                "raw_barcode_axis_sha256_before": expected_axis_sha,
                "raw_barcode_axis_sha256_after": expected_axis_sha,
                "raw_barcode_axis_rows_scanned": axis_rows,
                "selected_missing_from_raw_axis": 0,
                "selected_repeated_on_raw_axis": 0,
                "selected_exactly_once_on_raw_axis": True,
                "author_isolated_cells": isolated,
                "selected_minus_author_isolated": n_selected - isolated,
                "selected_exceeds_author_isolated": n_selected > isolated,
                "author_isolated_cells_role": "diagnostic_only",
                "formal_acceptance_pass": True,
                "calls_recomputed": False,
                "outcome_blinded": True,
                "real_pathway_results_inspected": False,
            }
        )
    return rows


def _revalidation_text(rows: list[dict[str, Any]]) -> str:
    frame = pd.DataFrame(rows, columns=REVALIDATION_COLUMNS)
    buffer = StringIO()
    frame.to_csv(buffer, sep="\t", index=False, lineterminator="\n")
    return buffer.getvalue()


def _global_bindings(watched: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        role: _binding(watched[role])
        for role in (
            "target_v2_policy",
            "source_v1_policy",
            "source_v1_diagnostic_ledger",
            "locked_library_manifest",
            "locked_triplet_audit",
        )
    }


def _migration_id(
    *, target_policy_sha256: str, source_ledger_sha256: str, tsv_sha256: str
) -> str:
    payload = {
        "target_policy_sha256": target_policy_sha256,
        "source_ledger_sha256": source_ledger_sha256,
        "revalidation_tsv_sha256": tsv_sha256,
    }
    return "t21-cd235a-neg-cell-call-v2-" + sha256(
        stable_json(payload).encode("utf-8")
    ).hexdigest()[:16]


def _record_payload_sha256(record: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "integrity"}
    return sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _record_value(
    context: Mapping[str, Any], rows: list[dict[str, Any]], tsv_path: Path
) -> dict[str, Any]:
    watched = context["watched"]
    tsv_sha = sha256_file(tsv_path)
    policy_sha = watched["target_v2_policy"]["sha256_before"]
    source_ledger_sha = watched["source_v1_diagnostic_ledger"]["sha256_before"]
    return {
        "schema_name": MIGRATION_SCHEMA_NAME,
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "migration_id": _migration_id(
            target_policy_sha256=policy_sha,
            source_ledger_sha256=source_ledger_sha,
            tsv_sha256=tsv_sha,
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass_outcome_blind_hash_bound_revalidation",
        "outcome_blinded": True,
        "real_pathway_results_inspected": False,
        "pathway_artifacts_read": False,
        "cell_call_counts_inspected_for_policy_amendment": True,
        "isolated_cell_count_exceedances_inspected_for_policy_amendment": True,
        "calls_recomputed": False,
        "calls_modified": False,
        "author_isolated_cells_used_for_acceptance": False,
        "analysis_role": "sensitivity_only",
        "sampling_frame_id": "t21_fetal_liver_cd235a_neg_sensitivity_v1",
        "formal_acceptance_gates": list(FORMAL_ACCEPTANCE_GATES),
        "input_bindings": _global_bindings(watched),
        "implementation": {
            "module": _binding(watched["migration_module"]),
            "cli": _binding(watched["migration_cli"]),
        },
        "hash_recheck": {
            "checkpoints": list(HASH_RECHECK_CHECKPOINTS),
            "watched_file_count": len(watched),
            "all_before_after_hashes_identical": True,
            "post_publish_recheck_completed": True,
            "per_library_hashes_recorded_in_revalidation_tsv": True,
        },
        "revalidation_ledger": {
            "relative_path": _relative(tsv_path, context["root"]),
            "bytes": tsv_path.stat().st_size,
            "sha256": tsv_sha,
            "n_libraries": len(rows),
        },
        "summary": {
            "n_libraries": len(rows),
            "n_selected_total": sum(int(row["n_selected"]) for row in rows),
            "n_author_isolated_count_exceedances": sum(
                bool(row["selected_exceeds_author_isolated"]) for row in rows
            ),
            "n_raw_axis_missing_selected_barcodes": 0,
            "n_raw_axis_repeated_selected_barcodes": 0,
            "all_formal_acceptance_gates_passed": True,
        },
        "migration_scope": {
            "source_calls_copied": False,
            "source_calls_rewritten": False,
            "new_cell_calls_computed": False,
            "evidence_outputs_only": ["migration_json", "revalidation_tsv"],
        },
    }


def _write_temp(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    return temporary


def migrate_cd235a_neg_cell_calls_v2(
    *,
    policy_path: str | Path,
    migration_json_path: str | Path,
    revalidation_tsv_path: str | Path,
    migration_cli_path: str | Path,
    repository_root: str | Path,
) -> tuple[Path, Path]:
    root = Path(repository_root).resolve()
    context = _load_context(
        Path(policy_path).resolve(), root, Path(migration_cli_path).resolve()
    )
    record_path = Path(migration_json_path).resolve()
    tsv_path = Path(revalidation_tsv_path).resolve()
    if record_path != context["record_path"] or tsv_path != context["revalidation_path"]:
        raise ValueError("Migration output paths differ from the frozen v2 policy")
    if record_path.exists() or tsv_path.exists():
        raise FileExistsError("Migration JSON or revalidation TSV already exists")

    rows = _revalidate_rows(context)
    _recheck_watched(
        context["watched"], root, checkpoint="after_revalidation_before_publish"
    )
    tsv_temp: Path | None = None
    record_temp: Path | None = None
    published: list[Path] = []
    try:
        tsv_temp = _write_temp(tsv_path, _revalidation_text(rows))
        record = _record_value(context, rows, tsv_temp)
        record["revalidation_ledger"]["relative_path"] = _relative(tsv_path, root)
        record["integrity"] = {
            "record_payload_sha256": _record_payload_sha256(record)
        }
        record_temp = _write_temp(
            record_path,
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
        )
        _recheck_watched(
            context["watched"], root, checkpoint="after_revalidation_before_publish"
        )
        os.replace(tsv_temp, tsv_path)
        tsv_temp = None
        published.append(tsv_path)
        os.replace(record_temp, record_path)
        record_temp = None
        published.append(record_path)
        _recheck_watched(context["watched"], root, checkpoint="after_publish")
    except BaseException:
        if tsv_temp is not None:
            tsv_temp.unlink(missing_ok=True)
        if record_temp is not None:
            record_temp.unlink(missing_ok=True)
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    return record_path, tsv_path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Migration JSON must contain one object")
    return value


def validate_cd235a_neg_cell_call_migration(
    migration_json_path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    record_path = Path(migration_json_path).resolve()
    if not record_path.is_file():
        raise FileNotFoundError(record_path)
    record = _load_json(record_path)
    integrity = _mapping(record.get("integrity"), "Migration JSON integrity")
    if integrity.get("record_payload_sha256") != _record_payload_sha256(record):
        raise ValueError("Migration JSON payload SHA256 differs")
    try:
        created_at = datetime.fromisoformat(str(record.get("created_at_utc", "")))
    except ValueError as exc:
        raise ValueError("Migration JSON creation time is invalid") from exc
    if created_at.tzinfo is None:
        raise ValueError("Migration JSON creation time must include a timezone")
    if (
        record.get("schema_name") != MIGRATION_SCHEMA_NAME
        or record.get("schema_version") != MIGRATION_SCHEMA_VERSION
        or record.get("status") != "pass_outcome_blind_hash_bound_revalidation"
        or record.get("outcome_blinded") is not True
        or record.get("real_pathway_results_inspected") is not False
        or record.get("pathway_artifacts_read") is not False
        or record.get("cell_call_counts_inspected_for_policy_amendment") is not True
        or record.get(
            "isolated_cell_count_exceedances_inspected_for_policy_amendment"
        )
        is not True
        or record.get("calls_recomputed") is not False
        or record.get("calls_modified") is not False
        or record.get("author_isolated_cells_used_for_acceptance") is not False
        or record.get("analysis_role") != "sensitivity_only"
        or record.get("sampling_frame_id")
        != "t21_fetal_liver_cd235a_neg_sensitivity_v1"
        or tuple(record.get("formal_acceptance_gates", []))
        != FORMAL_ACCEPTANCE_GATES
    ):
        raise ValueError("Migration JSON violates the outcome-blind v2 contract")

    bindings = _mapping(record.get("input_bindings"), "Input bindings")
    policy_binding = _mapping(bindings.get("target_v2_policy"), "Target v2 policy")
    policy_path = _resolve_relative(
        policy_binding.get("relative_path"), root, label="Target v2 policy"
    )
    implementation = _mapping(record.get("implementation"), "Implementation")
    cli_binding = _mapping(implementation.get("cli"), "Migration CLI")
    cli_path = _resolve_relative(
        cli_binding.get("relative_path"), root, label="Migration CLI"
    )
    context = _load_context(policy_path, root, cli_path)
    if record_path != context["record_path"]:
        raise ValueError("Migration JSON is outside its frozen policy path")

    expected_global = _global_bindings(context["watched"])
    if stable_json(bindings) != stable_json(expected_global):
        raise ValueError("Migration input bindings differ from current frozen inputs")
    expected_implementation = {
        "module": _binding(context["watched"]["migration_module"]),
        "cli": _binding(context["watched"]["migration_cli"]),
    }
    if stable_json(implementation) != stable_json(expected_implementation):
        raise ValueError("Migration implementation bindings changed")

    ledger_entry = _mapping(record.get("revalidation_ledger"), "Revalidation ledger")
    tsv_path = _resolve_relative(
        ledger_entry.get("relative_path"), root, label="Revalidation TSV"
    )
    if tsv_path != context["revalidation_path"]:
        raise ValueError("Revalidation TSV is outside its frozen policy path")
    if (
        tsv_path.stat().st_size != int(ledger_entry.get("bytes", -1))
        or sha256_file(tsv_path) != ledger_entry.get("sha256")
    ):
        raise ValueError("Revalidation TSV bytes or SHA256 changed")

    rows = _revalidate_rows(context)
    expected_text = _revalidation_text(rows)
    if tsv_path.read_text(encoding="utf-8") != expected_text:
        raise ValueError("Revalidation TSV is not reproducible from hash-bound inputs")
    _recheck_watched(context["watched"], root, checkpoint="validator_after_rescan")

    expected_hash_recheck = {
        "checkpoints": list(HASH_RECHECK_CHECKPOINTS),
        "watched_file_count": len(context["watched"]),
        "all_before_after_hashes_identical": True,
        "post_publish_recheck_completed": True,
        "per_library_hashes_recorded_in_revalidation_tsv": True,
    }
    if stable_json(record.get("hash_recheck")) != stable_json(expected_hash_recheck):
        raise ValueError("Migration hash-recheck evidence is incomplete")
    expected_summary = {
        "n_libraries": len(rows),
        "n_selected_total": sum(int(row["n_selected"]) for row in rows),
        "n_author_isolated_count_exceedances": sum(
            bool(row["selected_exceeds_author_isolated"]) for row in rows
        ),
        "n_raw_axis_missing_selected_barcodes": 0,
        "n_raw_axis_repeated_selected_barcodes": 0,
        "all_formal_acceptance_gates_passed": True,
    }
    if stable_json(record.get("summary")) != stable_json(expected_summary):
        raise ValueError("Migration summary differs from revalidated evidence")
    expected_id = _migration_id(
        target_policy_sha256=TARGET_POLICY_SHA256,
        source_ledger_sha256=SOURCE_V1_LEDGER_SHA256,
        tsv_sha256=sha256_file(tsv_path),
    )
    if record.get("migration_id") != expected_id:
        raise ValueError("Migration identifier differs from its immutable inputs")
    if int(ledger_entry.get("n_libraries", -1)) != len(rows):
        raise ValueError("Migration ledger library count differs")
    scope = record.get("migration_scope")
    if scope != {
        "source_calls_copied": False,
        "source_calls_rewritten": False,
        "new_cell_calls_computed": False,
        "evidence_outputs_only": ["migration_json", "revalidation_tsv"],
    }:
        raise ValueError("Migration scope permits cell-call mutation")
    return {
        "status": "pass",
        "migration_id": expected_id,
        "n_libraries": len(rows),
        "n_selected_total": expected_summary["n_selected_total"],
        "n_author_isolated_count_exceedances": expected_summary[
            "n_author_isolated_count_exceedances"
        ],
        "source_v1_policy_sha256": SOURCE_V1_POLICY_SHA256,
        "source_v1_ledger_sha256": SOURCE_V1_LEDGER_SHA256,
        "target_v2_policy_sha256": TARGET_POLICY_SHA256,
    }
