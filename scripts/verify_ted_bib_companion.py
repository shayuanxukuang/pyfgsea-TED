#!/usr/bin/env python3
"""Fail-closed verification for a built TED BIB v1.1.0 companion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree

from jsonschema import Draft202012Validator


RELEASE_VERSION = "1.1.0"
PACKAGES = ("core", "native_outputs", "stability_shards")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
FOCUSED_TEST_FILES = [
    "tests/test_ted_evidence.py",
    "tests/test_ted_schema.py",
    "tests/test_ted_flagship.py",
    "tests/test_gse171964_replication_freeze.py",
    "tests/test_nearest_method_benchmark.py",
]
FOCUSED_EXCLUDED_TESTS = [
    "test_parallel_evidence_types_never_upgrade_event_e_code",
    "test_canonical_parallel_evidence_schema_matches_embedded_v2_contract",
    "test_canonical_parallel_evidence_schema_enforces_controls_and_replication_id",
    "test_canonical_replication_facets_schema_matches_embedded_v2_contract",
    "test_canonical_replication_facets_keep_event_and_outcome_results_separate",
]


class CompanionVerificationError(RuntimeError):
    """A package member, manifest, or semantic contract failed verification."""


def archive_name(package: str) -> str:
    return (
        f"ted-bib-companion-v{RELEASE_VERSION}-"
        f"{package.replace('_', '-')}.zip"
    )


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_member(value: str, *, context: str) -> str:
    raw = value.strip().replace("\\", "/")
    path = PurePosixPath(raw)
    if (
        not raw
        or path.is_absolute()
        or ":" in raw
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise CompanionVerificationError(
            f"{context}: unsafe package member path {value!r}"
        )
    return path.as_posix()


def read_tsv_bytes(payload: bytes, *, context: str) -> list[dict[str, str]]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CompanionVerificationError(
            f"{context}: TSV is not UTF-8"
        ) from exc
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    if not reader.fieldnames:
        raise CompanionVerificationError(f"{context}: TSV has no header")
    try:
        return list(reader)
    except csv.Error as exc:
        raise CompanionVerificationError(
            f"{context}: invalid TSV: {exc}"
        ) from exc


def verify_outer_manifest(bundle_dir: Path) -> dict[str, dict[str, object]]:
    manifest_path = bundle_dir / "ARCHIVE_MANIFEST.tsv"
    if not manifest_path.is_file():
        raise CompanionVerificationError(
            f"missing outer manifest: {manifest_path}"
        )
    rows = read_tsv_bytes(
        manifest_path.read_bytes(),
        context="ARCHIVE_MANIFEST.tsv",
    )
    expected_fields = {"file", "bytes", "sha256", "role"}
    if not rows and manifest_path.stat().st_size == 0:
        raise CompanionVerificationError("ARCHIVE_MANIFEST.tsv is empty")
    entries: dict[str, dict[str, object]] = {}
    for row_number, row in enumerate(rows, start=2):
        if not expected_fields.issubset(row):
            raise CompanionVerificationError(
                "ARCHIVE_MANIFEST.tsv is missing required columns"
            )
        name = normalize_member(
            row["file"],
            context=f"ARCHIVE_MANIFEST.tsv row {row_number}",
        )
        if "/" in name:
            raise CompanionVerificationError(
                f"outer manifest member must be a top-level file: {name}"
            )
        if name in entries:
            raise CompanionVerificationError(
                f"duplicate outer manifest entry: {name}"
            )
        try:
            expected_size = int(row["bytes"])
        except ValueError as exc:
            raise CompanionVerificationError(
                f"invalid byte count for outer member {name}"
            ) from exc
        checksum = row["sha256"].strip().lower()
        if expected_size < 0 or not SHA256_RE.fullmatch(checksum):
            raise CompanionVerificationError(
                f"invalid integrity fields for outer member {name}"
            )
        path = bundle_dir / name
        if not path.is_file():
            raise CompanionVerificationError(f"missing outer member: {name}")
        actual_size = path.stat().st_size
        actual_sha256 = sha256_path(path)
        if actual_size != expected_size or actual_sha256 != checksum:
            raise CompanionVerificationError(
                f"outer member integrity mismatch: {name}"
            )
        entries[name] = {
            "bytes": actual_size,
            "sha256": actual_sha256,
            "role": row["role"],
        }
    actual_names = {
        path.name for path in bundle_dir.iterdir() if path.is_file()
    }
    expected_names = set(entries) | {"ARCHIVE_MANIFEST.tsv"}
    if actual_names != expected_names:
        raise CompanionVerificationError(
            "outer bundle has missing or unmanifested files: "
            f"expected {sorted(expected_names)}, found {sorted(actual_names)}"
        )
    required = {
        archive_name(package) for package in PACKAGES
    } | {"BUILD_INDEX.json"}
    if set(entries) != required:
        raise CompanionVerificationError(
            f"outer manifest members differ from required set: {sorted(required)}"
        )
    return entries


def hash_zip_member(
    archive: zipfile.ZipFile,
    member: str,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with archive.open(member, "r") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def verify_package_archive(
    path: Path,
    package: str,
) -> tuple[dict[str, dict[str, object]], dict[str, bytes]]:
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise CompanionVerificationError(
                f"{path.name}: duplicate ZIP member names"
            )
        normalized = {
            normalize_member(name, context=path.name) for name in names
        }
        if normalized != set(names):
            raise CompanionVerificationError(
                f"{path.name}: ZIP contains non-canonical member paths"
            )
        if "PACKAGE_MANIFEST.tsv" not in names:
            raise CompanionVerificationError(
                f"{path.name}: PACKAGE_MANIFEST.tsv is missing"
            )
        manifest_payload = archive.read("PACKAGE_MANIFEST.tsv")
        rows = read_tsv_bytes(
            manifest_payload,
            context=f"{path.name}:PACKAGE_MANIFEST.tsv",
        )
        required_fields = {
            "path",
            "bytes",
            "sha256",
            "source_rule",
            "role",
        }
        entries: dict[str, dict[str, object]] = {}
        selected_payloads: dict[str, bytes] = {
            "PACKAGE_MANIFEST.tsv": manifest_payload
        }
        for row_number, row in enumerate(rows, start=2):
            if not required_fields.issubset(row):
                raise CompanionVerificationError(
                    f"{path.name}: package manifest lacks required columns"
                )
            member = normalize_member(
                row["path"],
                context=f"{path.name} manifest row {row_number}",
            )
            if member == "PACKAGE_MANIFEST.tsv":
                raise CompanionVerificationError(
                    f"{path.name}: package manifest must exclude itself"
                )
            if member in entries:
                raise CompanionVerificationError(
                    f"{path.name}: duplicate package manifest entry {member}"
                )
            try:
                expected_size = int(row["bytes"])
            except ValueError as exc:
                raise CompanionVerificationError(
                    f"{path.name}: invalid byte count for {member}"
                ) from exc
            checksum = row["sha256"].strip().lower()
            if expected_size < 0 or not SHA256_RE.fullmatch(checksum):
                raise CompanionVerificationError(
                    f"{path.name}: invalid integrity fields for {member}"
                )
            if member not in names:
                raise CompanionVerificationError(
                    f"{path.name}: manifest member is absent: {member}"
                )
            info = archive.getinfo(member)
            if info.file_size != expected_size:
                raise CompanionVerificationError(
                    f"{path.name}: ZIP size mismatch for {member}"
                )
            actual_size, actual_checksum = hash_zip_member(archive, member)
            if (
                actual_size != expected_size
                or actual_checksum != checksum
            ):
                raise CompanionVerificationError(
                    f"{path.name}: integrity mismatch for {member}"
                )
            entries[member] = {
                "bytes": actual_size,
                "sha256": actual_checksum,
                "source_rule": row["source_rule"],
                "role": row["role"],
            }
            capture = (
                package == "core"
                and member.endswith((".json", ".tsv", ".xml", ".txt"))
            ) or member.endswith("manifest.tsv") or member == "PACKAGE_METADATA.json"
            if capture and actual_size <= 32 * 1024 * 1024:
                selected_payloads[member] = archive.read(member)
        expected_names = set(entries) | {"PACKAGE_MANIFEST.tsv"}
        if set(names) != expected_names:
            raise CompanionVerificationError(
                f"{path.name}: ZIP has missing or unmanifested members"
            )
    return entries, selected_payloads


def parse_int(value: str, *, context: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise CompanionVerificationError(
            f"{context}: expected integer, got {value!r}"
        ) from exc


def verify_focused_81(
    core_payloads: dict[str, bytes],
    analysis_lock_commit: str,
) -> None:
    junit_path = "results/ted_bib_focused_81/focused_81_junit.xml"
    summary_path = (
        "results/ted_bib_focused_81/focused_81_terminal_summary.txt"
    )
    collection_path = (
        "results/ted_bib_focused_81/focused_81_collection.txt"
    )
    command_path = "results/ted_bib_focused_81/focused_81_command.json"
    try:
        root = ElementTree.fromstring(core_payloads[junit_path])
    except (KeyError, ElementTree.ParseError) as exc:
        raise CompanionVerificationError(
            "focused-81 JUnit evidence is missing or invalid"
        ) from exc
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        raise CompanionVerificationError(
            "focused-81 JUnit contains no test suites"
        )
    tests = sum(parse_int(suite.get("tests", "0"), context="JUnit tests") for suite in suites)
    failures = sum(
        parse_int(suite.get("failures", "0"), context="JUnit failures")
        for suite in suites
    )
    errors = sum(
        parse_int(suite.get("errors", "0"), context="JUnit errors")
        for suite in suites
    )
    skipped = sum(
        parse_int(suite.get("skipped", "0"), context="JUnit skipped")
        for suite in suites
    )
    if tests != 81 or failures != 0 or errors != 0 or skipped != 0:
        raise CompanionVerificationError(
            "focused test evidence must report exactly 81 tests, "
            "0 failures, 0 errors, 0 skipped; found "
            f"{tests}, {failures}, {errors}, {skipped}"
        )
    try:
        summary = core_payloads[summary_path].decode("utf-8")
    except (KeyError, UnicodeDecodeError) as exc:
        raise CompanionVerificationError(
            "focused-81 terminal summary is missing or not UTF-8"
        ) from exc
    if re.search(r"\b81 passed\b", summary) is None:
        raise CompanionVerificationError(
            "focused-81 terminal summary does not contain '81 passed'"
        )
    try:
        collection = core_payloads[collection_path].decode("utf-8")
        command = json.loads(core_payloads[command_path].decode("utf-8"))
    except (
        KeyError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise CompanionVerificationError(
            "focused-81 collection or command evidence is missing or invalid"
        ) from exc
    if not isinstance(command, dict):
        raise CompanionVerificationError(
            "focused-81 command evidence must be a JSON object"
        )
    node_ids = [
        line
        for line in collection.splitlines()
        if "::test_" in line and not line.startswith((" ", "="))
    ]
    selection_payload = ("\n".join(node_ids) + "\n").encode("utf-8")
    expected_result = {
        "tests": 81,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }
    selection = command.get("selection")
    runtime = command.get("runtime")
    environment_controls = command.get("environment_controls")
    evidence_sha256 = command.get("evidence_sha256")
    if (
        command.get("evidence_kind") != "ted_v1.1.0_focused_81"
        or command.get("analysis_lock_commit") != analysis_lock_commit
        or command.get("repository_dirty") is not False
        or command.get("result") != expected_result
        or not isinstance(command.get("argv"), list)
        or not isinstance(selection, dict)
        or selection.get("test_files") != FOCUSED_TEST_FILES
        or selection.get("excluded_v1_1_extension_tests")
        != FOCUSED_EXCLUDED_TESTS
        or selection.get("collected") != 81
        or selection.get("node_ids") != node_ids
        or selection.get("selection_manifest_sha256")
        != hashlib.sha256(selection_payload).hexdigest()
        or not isinstance(runtime, dict)
        or not isinstance(runtime.get("pyfgsea"), dict)
        or runtime["pyfgsea"].get("version") != "0.2.0"
        or not runtime["pyfgsea"].get("import_origin")
        or not runtime["pyfgsea"].get("import_root")
        or not isinstance(environment_controls, dict)
        or environment_controls.get("isolated_child_python") is not True
        or environment_controls.get("PYTHONPATH") != "removed"
        or not environment_controls.get(
            "installed_distribution_root_inserted_by_guard"
        )
        or not isinstance(evidence_sha256, dict)
    ):
        raise CompanionVerificationError(
            "focused-81 command record disagrees with the locked selection, "
            "runtime, result, or analysis commit"
        )
    expected_hashes = {
        "collection": hashlib.sha256(core_payloads[collection_path]).hexdigest(),
        "junit": hashlib.sha256(core_payloads[junit_path]).hexdigest(),
        "terminal_summary": hashlib.sha256(
            core_payloads[summary_path]
        ).hexdigest(),
    }
    if evidence_sha256 != expected_hashes:
        raise CompanionVerificationError(
            "focused-81 command record hashes disagree with packaged evidence"
        )


def verify_common_task(core_payloads: dict[str, bytes]) -> None:
    base = "results/ted_manuscript_machine_readable_v2/"
    try:
        status = json.loads(
            core_payloads[f"{base}common_task_status.json"].decode("utf-8")
        )
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompanionVerificationError(
            "common-task status is missing or invalid"
        ) from exc
    methods = {
        "TIPS",
        "scTransient",
        "tradeSeq",
        "score_then_smooth",
        "TED",
    }
    if (
        status.get("expected_tasks") != 480
        or status.get("completed_tasks") != 480
        or status.get("complete") is not True
        or set(status.get("methods", [])) != methods
    ):
        raise CompanionVerificationError(
            "common-task status does not certify 480/480 tasks and five methods"
        )
    registry_path = f"{base}common_task_scenario_registry.tsv"
    try:
        registry = read_tsv_bytes(
            core_payloads[registry_path],
            context=registry_path,
        )
    except KeyError as exc:
        raise CompanionVerificationError(
            "common-task scenario registry is missing"
        ) from exc
    try:
        task_count = sum(int(row["replicates"]) for row in registry)
    except (KeyError, ValueError) as exc:
        raise CompanionVerificationError(
            "scenario registry lacks valid replicates counts"
        ) from exc
    if task_count != 480:
        raise CompanionVerificationError(
            f"scenario registry expands to {task_count} tasks, expected 480"
        )


def load_schema(
    core_payloads: dict[str, bytes],
    path: str,
) -> dict[str, object]:
    try:
        schema = json.loads(core_payloads[path].decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompanionVerificationError(
            f"schema is missing or invalid: {path}"
        ) from exc
    if not isinstance(schema, dict):
        raise CompanionVerificationError(f"schema is not an object: {path}")
    return schema


def verify_schema(core_payloads: dict[str, bytes]) -> None:
    path = "schemas/ted_event_report_v2.schema.json"
    schema = load_schema(core_payloads, path)
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise CompanionVerificationError(
            "TED event v2 schema properties must be an object"
        )
    required = {
        "parallel_evidence_records",
        "event_replication_eligibility_status",
        "event_replication_test_status",
        "event_replication_status",
        "outcome_replication_status",
    }
    if not required.issubset(properties):
        raise CompanionVerificationError(
            "TED event v2 schema lacks parallel-evidence or replication facets"
        )
    parallel_path = "schemas/parallel_evidence_record_v1.schema.json"
    parallel = load_schema(core_payloads, parallel_path)
    parallel_properties = parallel.get("properties", {})
    if not isinstance(parallel_properties, dict):
        raise CompanionVerificationError(
            "parallel evidence record schema properties must be an object"
        )
    parallel_required = {
        "record_id",
        "evidence_type",
        "status",
        "independence_context",
        "controls_pass",
        "replication_status",
        "reason_codes",
    }
    if not parallel_required.issubset(parallel_properties):
        raise CompanionVerificationError(
            "parallel evidence record schema lacks required typed fields"
        )
    evidence_enum = (
        parallel_properties.get("evidence_type", {}).get("enum", [])
        if isinstance(parallel_properties.get("evidence_type"), dict)
        else []
    )
    if set(evidence_enum) != {
        "orthogonal_outcome",
        "intervention_reversal",
        "matched_rescue",
    }:
        raise CompanionVerificationError(
            "parallel evidence record schema lacks the three evidence types"
        )
    replication_path = "schemas/replication_facets_v1.schema.json"
    replication = load_schema(core_payloads, replication_path)
    replication_properties = replication.get("properties", {})
    if not isinstance(replication_properties, dict):
        raise CompanionVerificationError(
            "replication facets schema properties must be an object"
        )
    replication_required = {
        "event_replication_eligibility_status",
        "event_replication_test_status",
        "event_replication_status",
        "outcome_replication_status",
        "replication_dataset_id",
    }
    if not replication_required.issubset(replication_properties):
        raise CompanionVerificationError(
            "replication facets schema lacks event/outcome replication fields"
        )


def verify_canonical_claim_contract(core_payloads: dict[str, bytes]) -> None:
    contract_root = "results/ted_bib_companion_evidence_contract_v1/"
    parallel_path = f"{contract_root}parallel_evidence_record_v1.json"
    replication_path = f"{contract_root}replication_facets_v1.json"
    claim_path = f"{contract_root}claim_boundary_v1.json"
    manifest_path = f"{contract_root}manifest.tsv"
    final_path = (
        "results/ted_bnt162b2_flagship/final_evidence_v1/"
        "final_evidence_summary.json"
    )
    audit_path = (
        "results/ted_bnt162b2_flagship/final_evidence_v1/"
        "independent_recalculation_audit.json"
    )

    def load_object(path: str) -> dict[str, object]:
        try:
            value = json.loads(core_payloads[path].decode("utf-8"))
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CompanionVerificationError(
                f"canonical claim asset is missing or invalid: {path}"
            ) from exc
        if not isinstance(value, dict):
            raise CompanionVerificationError(
                f"canonical claim asset is not an object: {path}"
            )
        return value

    parallel = load_object(parallel_path)
    replication = load_object(replication_path)
    claim = load_object(claim_path)
    final = load_object(final_path)
    audit = load_object(audit_path)
    parallel_schema = load_schema(
        core_payloads,
        "schemas/parallel_evidence_record_v1.schema.json",
    )
    replication_schema = load_schema(
        core_payloads,
        "schemas/replication_facets_v1.schema.json",
    )
    for label, instance, schema in (
        ("parallel evidence", parallel, parallel_schema),
        ("replication facets", replication, replication_schema),
    ):
        errors = sorted(
            Draft202012Validator(schema).iter_errors(instance),
            key=lambda error: tuple(
                str(part) for part in error.absolute_path
            ),
        )
        if errors:
            detail = "; ".join(error.message for error in errors[:5])
            raise CompanionVerificationError(
                f"canonical {label} instance violates its schema: {detail}"
            )

    bounded_display = (
        "E0 | protein outcome passed | event replication not_evaluable "
        "(eligibility failed; test not_run) | protein outcome replication "
        "not_tested"
    )
    if (
        parallel.get("status") != "passed"
        or parallel.get("evidence_type") != "orthogonal_outcome"
        or parallel.get("controls_pass") is not True
        or parallel.get("replication_status") != "not_tested"
        or replication.get("event_replication_eligibility_status") != "failed"
        or replication.get("event_replication_test_status") != "not_run"
        or replication.get("event_replication_status") != "not_evaluable"
        or replication.get("outcome_replication_status") != "not_tested"
        or claim.get("event_support_code") != "E0"
        or claim.get("parallel_evidence_record") != parallel
        or claim.get("replication_facets") != replication
        or claim.get("bounded_display") != bounded_display
        or claim.get("legacy_fields_are_source_provenance_only") is not True
        or final.get("event_support_code") != "E0"
        or final.get("within_study_protein_outcome_status") != "passed"
        or final.get("event_replication_eligibility_status") != "failed"
        or final.get("event_replication_test_status") != "not_run"
        or final.get("event_replication_status") != "not_evaluable"
        or final.get("protein_outcome_replication_status") != "not_tested"
        or final.get("bounded_display") != bounded_display
        or audit.get("all_checks_pass") is not True
    ):
        raise CompanionVerificationError(
            "canonical BNT/GSE claim contract does not preserve "
            "E0 / protein-passed / failed-not_run-not_evaluable-not_tested"
        )

    try:
        rows = read_tsv_bytes(
            core_payloads[manifest_path],
            context=manifest_path,
        )
    except KeyError as exc:
        raise CompanionVerificationError(
            "canonical claim manifest is missing"
        ) from exc
    expected_members = {
        "parallel_evidence_record_v1.json",
        "replication_facets_v1.json",
        "claim_boundary_v1.json",
    }
    if {row.get("file", "") for row in rows} != expected_members:
        raise CompanionVerificationError(
            "canonical claim manifest has unexpected members"
        )
    for row in rows:
        member = f"{contract_root}{row['file']}"
        payload = core_payloads.get(member)
        if payload is None:
            raise CompanionVerificationError(
                f"canonical claim manifest member is missing: {member}"
            )
        if (
            row.get("bytes") != str(len(payload))
            or row.get("sha256") != hashlib.sha256(payload).hexdigest()
        ):
            raise CompanionVerificationError(
                f"canonical claim manifest disagrees for {member}"
            )


def verify_figure_chain(
    core_entries: dict[str, dict[str, object]],
    core_payloads: dict[str, bytes],
) -> None:
    path = "reproduction/FIGURE_SOURCE_CHAIN.tsv"
    try:
        rows = read_tsv_bytes(core_payloads[path], context=path)
    except KeyError as exc:
        raise CompanionVerificationError("figure source chain is missing") from exc
    grouped: dict[str, dict[str, set[str]]] = {
        "figure3": {"source": set(), "output": set()},
        "figure5": {"source": set(), "output": set()},
    }
    for row in rows:
        figure_id = row.get("figure_id", "")
        relation = row.get("relation", "")
        if figure_id in grouped and relation in grouped[figure_id]:
            grouped[figure_id][relation].add(row.get("path", ""))
    expected_sources = {"figure3": 4, "figure5": 7}
    for figure_id, expected_count in expected_sources.items():
        source_count = len(grouped[figure_id]["source"])
        output_count = len(grouped[figure_id]["output"])
        if source_count != expected_count or output_count < 2:
            raise CompanionVerificationError(
                f"{figure_id} chain requires {expected_count} sources and "
                f"at least PDF+PNG outputs; found {source_count}, {output_count}"
            )
        for member in (
            grouped[figure_id]["source"] | grouped[figure_id]["output"]
        ):
            if member not in core_entries:
                raise CompanionVerificationError(
                    f"{figure_id} chain member is absent: {member}"
                )


def verify_case_study_assets(core_entries: dict[str, dict[str, object]]) -> None:
    required = {
        "data_external/bnt162b2_cite_asap_2023/download_manifest.tsv",
        "data_external/bnt162b2_cite_asap_2023/download_manifest.json",
        "results/ted_bnt162b2_flagship/protocol_freeze_v1/protocol_freeze.json",
        "results/ted_bnt162b2_flagship/protocol_freeze_v1/protocol_manifest.tsv",
        "results/ted_bnt162b2_flagship/rna_event_freeze_v1/"
        "rna_event_status.json",
        "results/ted_bnt162b2_flagship/rna_event_freeze_v1/"
        "rna_event_gate_table.tsv",
        "results/ted_bnt162b2_flagship/orthogonal_outcome_v1/"
        "protein_outcome_status.json",
        "results/ted_bnt162b2_flagship/orthogonal_outcome_v1/"
        "protein_outcome_gate_table.tsv",
        "data_external/GSE171964_BNT162b2_replication/download_manifest.tsv",
        "data_external/GSE171964_BNT162b2_replication/download_manifest.json",
        "results/ted_gse171964_replication/protocol_freeze_v1/"
        "protocol_freeze.json",
        "results/ted_gse171964_replication/protocol_freeze_v1/"
        "protocol_manifest.tsv",
        "results/ted_gse171964_replication/analysis_v1/"
        "replication_status.json",
        "results/ted_gse171964_replication/analysis_v1/"
        "replication_gate_table.tsv",
        "results/ted_gse171964_replication/analysis_v1/"
        "analysis_manifest.tsv",
    }
    missing = sorted(required - set(core_entries))
    if missing:
        raise CompanionVerificationError(
            f"case-study protocol/output assets are missing: {missing}"
        )


def verify_renderer_contract(
    core_entries: dict[str, dict[str, object]],
    core_payloads: dict[str, bytes],
) -> None:
    required = {
        "reproduce/verify_and_reproduce_figures.py",
        "reproduce/render_bib_figures.py",
        "reproduction/FIGURE_RENDERERS.json",
        "requirements-reproduction-py311.txt",
    }
    missing = sorted(required - set(core_entries))
    if missing:
        raise CompanionVerificationError(
            f"figure renderer or dependency-lock assets are missing: {missing}"
        )
    try:
        contract = json.loads(
            core_payloads["reproduction/FIGURE_RENDERERS.json"].decode("utf-8")
        )
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompanionVerificationError(
            "FIGURE_RENDERERS.json is missing or invalid"
        ) from exc
    if not isinstance(contract, dict):
        raise CompanionVerificationError(
            "FIGURE_RENDERERS.json must be an object"
        )
    for figure_id in ("figure3", "figure5"):
        item = contract.get(figure_id)
        if not isinstance(item, dict):
            raise CompanionVerificationError(
                f"renderer contract lacks {figure_id}"
            )
        command = item.get("command")
        outputs = item.get("outputs")
        semantic_report = item.get("semantic_report")
        if (
            not isinstance(command, list)
            or len(command) < 2
            or command[1] != "reproduce/render_bib_figures.py"
            or not isinstance(outputs, list)
            or len(outputs) != 2
            or not isinstance(semantic_report, str)
            or not semantic_report
        ):
            raise CompanionVerificationError(
                f"renderer contract for {figure_id} is incomplete"
            )
    try:
        lock = core_payloads["requirements-reproduction-py311.txt"].decode(
            "utf-8"
        )
    except (KeyError, UnicodeDecodeError) as exc:
        raise CompanionVerificationError(
            "Python 3.11 reproduction dependency lock is missing or invalid"
        ) from exc
    for package in ("matplotlib", "numpy", "pandas", "pillow"):
        if re.search(rf"(?m)^{package}==[^\s]+$", lock) is None:
            raise CompanionVerificationError(
                f"reproduction dependency lock does not pin {package}"
            )


def verify_nested_manifests(
    core_entries: dict[str, dict[str, object]],
    core_payloads: dict[str, bytes],
    native_entries: dict[str, dict[str, object]],
    native_payloads: dict[str, bytes],
    stability_entries: dict[str, dict[str, object]],
    stability_payloads: dict[str, bytes],
) -> None:
    native_manifest = (
        "results/ted_manuscript_machine_readable_v2/"
        "method_native_outputs/manifest.tsv"
    )
    try:
        core_manifest_payload = core_payloads[native_manifest]
        native_manifest_payload = native_payloads[native_manifest]
    except KeyError as exc:
        raise CompanionVerificationError(
            "native-output manifest is not present in both core and native archives"
        ) from exc
    if core_manifest_payload != native_manifest_payload:
        raise CompanionVerificationError(
            "core and native archives contain different native-output manifests"
        )
    rows = read_tsv_bytes(
        native_manifest_payload,
        context=native_manifest,
    )
    if len(rows) != 2400:
        raise CompanionVerificationError(
            f"native-output manifest has {len(rows)} rows, expected 2400"
        )
    for row_number, row in enumerate(rows, start=2):
        member = normalize_member(
            row.get("file", ""),
            context=f"native manifest row {row_number}",
        )
        entry = native_entries.get(member)
        if entry is None:
            raise CompanionVerificationError(
                f"native-output archive lacks manifest member: {member}"
            )
        if (
            str(entry["bytes"]) != row.get("bytes", "").strip()
            or entry["sha256"] != row.get("sha256", "").strip().lower()
        ):
            raise CompanionVerificationError(
                f"native nested manifest disagrees with package manifest: {member}"
            )
    stability_manifest = (
        "results/ted_submission_supplement/"
        "zscape_repeated_holdout_stability/manifest.tsv"
    )
    try:
        stability_rows = read_tsv_bytes(
            stability_payloads[stability_manifest],
            context=stability_manifest,
        )
    except KeyError as exc:
        raise CompanionVerificationError(
            "stability shard nested manifest is missing"
        ) from exc
    prefix = str(PurePosixPath(stability_manifest).parent)
    for row_number, row in enumerate(stability_rows, start=2):
        source_path = row.get("source_path", "").strip()
        relative = normalize_member(
            source_path or row.get("file", ""),
            context=f"stability manifest row {row_number}",
        )
        member = (
            relative
            if source_path
            else f"{prefix}/{relative}"
        )
        entry = stability_entries.get(member)
        if entry is None:
            raise CompanionVerificationError(
                f"stability archive lacks manifest member: {member}"
            )
        if (
            str(entry["bytes"]) != row.get("bytes", "").strip()
            or entry["sha256"] != row.get("sha256", "").strip().lower()
        ):
            raise CompanionVerificationError(
                f"stability nested manifest disagrees for {member}"
            )
    if native_manifest not in core_entries:
        raise CompanionVerificationError(
            "core package manifest omits the nested native-output manifest"
        )


def verify_package_metadata(
    package_payloads: dict[str, dict[str, bytes]],
    index: dict[str, object],
) -> str:
    commit = index.get("analysis_lock_commit")
    spec_sha256 = index.get("asset_spec_sha256")
    if not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit):
        raise CompanionVerificationError(
            "BUILD_INDEX.json lacks a full analysis-lock commit"
        )
    if not isinstance(spec_sha256, str) or not SHA256_RE.fullmatch(spec_sha256):
        raise CompanionVerificationError(
            "BUILD_INDEX.json lacks the asset-spec SHA-256"
        )
    for package in PACKAGES:
        try:
            metadata = json.loads(
                package_payloads[package]["PACKAGE_METADATA.json"].decode(
                    "utf-8"
                )
            )
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CompanionVerificationError(
                f"{package}: PACKAGE_METADATA.json is missing or invalid"
            ) from exc
        if (
            not isinstance(metadata, dict)
            or metadata.get("release") != f"ted-v{RELEASE_VERSION}"
            or metadata.get("package") != package
            or metadata.get("analysis_lock_commit") != commit
            or metadata.get("asset_spec_sha256") != spec_sha256
        ):
            raise CompanionVerificationError(
                f"{package}: package metadata disagrees with BUILD_INDEX.json"
            )
    return commit


def verify_bundle(bundle_dir: Path) -> dict[str, object]:
    bundle_dir = bundle_dir.resolve()
    if not bundle_dir.is_dir():
        raise CompanionVerificationError(
            f"bundle directory is missing: {bundle_dir}"
        )
    outer = verify_outer_manifest(bundle_dir)
    package_entries: dict[str, dict[str, dict[str, object]]] = {}
    package_payloads: dict[str, dict[str, bytes]] = {}
    for package in PACKAGES:
        entries, payloads = verify_package_archive(
            bundle_dir / archive_name(package),
            package,
        )
        package_entries[package] = entries
        package_payloads[package] = payloads
    try:
        index = json.loads(
            (bundle_dir / "BUILD_INDEX.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompanionVerificationError("BUILD_INDEX.json is invalid") from exc
    if (
        index.get("release") != f"ted-v{RELEASE_VERSION}"
        or index.get("zenodo_version_doi") is not None
    ):
        raise CompanionVerificationError(
            "build index has the wrong release or invents DOI metadata"
        )
    analysis_lock_commit = verify_package_metadata(package_payloads, index)
    core_entries = package_entries["core"]
    core_payloads = package_payloads["core"]
    verify_common_task(core_payloads)
    verify_focused_81(core_payloads, analysis_lock_commit)
    verify_schema(core_payloads)
    verify_canonical_claim_contract(core_payloads)
    verify_figure_chain(core_entries, core_payloads)
    verify_case_study_assets(core_entries)
    verify_renderer_contract(core_entries, core_payloads)
    verify_nested_manifests(
        core_entries,
        core_payloads,
        package_entries["native_outputs"],
        package_payloads["native_outputs"],
        package_entries["stability_shards"],
        package_payloads["stability_shards"],
    )
    return {
        "status": "verified",
        "release": f"ted-v{RELEASE_VERSION}",
        "bundle": str(bundle_dir),
        "outer_members": len(outer),
        "package_members": {
            package: len(package_entries[package]) for package in PACKAGES
        },
        "common_tasks": 480,
        "native_method_task_outputs": 2400,
        "focused_tests": 81,
        "figure3_source_tables": 4,
        "figure5_source_tables": 7,
        "analysis_lock_commit": analysis_lock_commit,
        "zenodo_version_doi": None,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a built TED BIB v1.1.0 companion bundle"
    )
    parser.add_argument("bundle_dir", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = verify_bundle(args.bundle_dir)
    except (
        CompanionVerificationError,
        OSError,
        ValueError,
        csv.Error,
        zipfile.BadZipFile,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
