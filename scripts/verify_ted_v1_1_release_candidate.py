#!/usr/bin/env python3
"""Verify the post-lock TED v1.1.0 release candidate and optional assets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

RELEASE_DIR = Path("release/ted-v1.1.0")
ANALYSIS_LOCK_COMMIT = "32e099c780bf0103bbcfadb2993e59254c6d9e12"
ANALYSIS_LOCK_TREE = "8cf20d4d4cddbf0056880d7daaa8caac62b68a2e"
BASELINE_TAG_OBJECT = "992e7569e2a7cb47fc06557302f7480496f7eeab"
BASELINE_COMMIT = "5cb7b25458b41437b54623488d37b4872e79f474"
ASSET_SPEC_SHA256 = (
    "6074cac638f1b95963d87a876cdf6274151ec1ebbd103a64a2546fa8d2ae83e2"
)
ALLOWED_POST_LOCK_PATHS = {
    ".github/workflows/ci.yml",
    "release/ted-v1.1.0/ANALYSIS_LOCK.json",
    "release/ted-v1.1.0/ARCHIVE_MANIFEST.tsv",
    "release/ted-v1.1.0/BUILD_INDEX.json",
    "release/ted-v1.1.0/EXTERNAL_ARCHIVE_ASSETS.tsv",
    "release/ted-v1.1.0/EXTERNAL_ARCHIVE_ASSETS.template.tsv",
    "release/ted-v1.1.0/FOCUSED_81_EVIDENCE.tsv",
    "release/ted-v1.1.0/GITHUB_RELEASE_BODY.md",
    "release/ted-v1.1.0/README.md",
    "release/ted-v1.1.0/RELEASE_METADATA.json",
    "scripts/verify_ted_v1_1_release_candidate.py",
}
TRACKED_ATTESTATION_PATHS = ALLOWED_POST_LOCK_PATHS - {
    "release/ted-v1.1.0/RELEASE_METADATA.json"
}
ARCHIVE_NAMES = {
    "ted-bib-companion-v1.1.0-core.zip",
    "ted-bib-companion-v1.1.0-native-outputs.zip",
    "ted-bib-companion-v1.1.0-stability-shards.zip",
}
EXTERNAL_ROLES = {
    "ted-bib-companion-v1.1.0-core.zip": "companion_core",
    "ted-bib-companion-v1.1.0-native-outputs.zip": "native_outputs",
    "ted-bib-companion-v1.1.0-stability-shards.zip": "stability_shards",
    "BUILD_INDEX.json": "build_index",
    "ARCHIVE_MANIFEST.tsv": "outer_attestation",
    "pyfgsea-0.2.0-cp38-abi3-win_amd64.whl": "python_wheel",
    "pyfgsea-ted-v1.1.0-figure-report.json": (
        "figure_reproduction_attestation"
    ),
}
ARCHIVE_ROLES = {
    "ted-bib-companion-v1.1.0-core.zip": "core",
    "ted-bib-companion-v1.1.0-native-outputs.zip": "native_outputs",
    "ted-bib-companion-v1.1.0-stability-shards.zip": "stability_shards",
    "BUILD_INDEX.json": "build_index",
}
FOCUSED_NAMES = {
    "focused_81_collection.txt",
    "focused_81_command.json",
    "focused_81_junit.xml",
    "focused_81_terminal_summary.txt",
}
EXTERNAL_COLUMNS = (
    "asset_name",
    "asset_role",
    "archive_location",
    "source_ref_or_manifest",
    "bytes",
    "sha256",
    "verification_status",
    "notes",
)
ARCHIVE_COLUMNS = ("file", "bytes", "sha256", "role")
FOCUSED_COLUMNS = (
    "artifact",
    "role",
    "path",
    "bytes",
    "sha256",
    "verification_status",
    "analysis_lock_commit",
    "result",
)
FOCUSED_ROLES = {
    "focused_81_collection.txt": "collection",
    "focused_81_command.json": "command_record",
    "focused_81_junit.xml": "junit_xml",
    "focused_81_terminal_summary.txt": "terminal_summary",
}
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
EXPECTED_CLAIM_BOUNDARY = {
    "event_support_code": "E0",
    "parallel_protein_outcome_status": "passed",
    "event_replication_eligibility_status": "failed",
    "event_replication_test_status": "not_run",
    "event_replication_status": "not_evaluable",
    "protein_outcome_replication_status": "not_tested",
    "parallel_evidence_never_upgrades_event_e": True,
}
EXPECTED_EXTERNAL_EVIDENCE = {
    "focused_tests": {
        "tests": 81,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    },
    "common_tasks": 480,
    "native_method_task_outputs": 2400,
    "figure3_source_tables": 4,
    "figure5_source_tables": 7,
    "canonical_claim_contract_manifest_sha256": (
        "ad28ab712a089e2eff809d62b1d3243e5fff06fa13d1616463ab6c380e4f16be"
    ),
}
EXPECTED_CONTRACT_PATHS = {
    "release/ted-v1.1.0/CLAIM_BOUNDARY.md",
    "release/ted-v1.1.0/COMPANION_ASSET_RULES.template.tsv",
    "config/ted_nearest_method_common_task_v1.yml",
    "config/ted_bnt162b2_flagship_v1.yaml",
    "config/ted_gse171964_replication_v1.yaml",
    "scripts/run_nearest_method_locked_grid.py",
    "scripts/run_bnt162b2_flagship_rna.py",
    "scripts/run_bnt162b2_flagship_adt.py",
    "scripts/run_gse171964_replication.py",
    "scripts/build_bib_companion_evidence_contracts.py",
    "scripts/run_ted_bib_focused_tests.py",
    "scripts/build_ted_bib_companion.py",
    "scripts/verify_ted_bib_companion.py",
    "reproduce/verify_and_reproduce_figures.py",
    "schemas/parallel_evidence_record_v1.schema.json",
    "schemas/replication_facets_v1.schema.json",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
DOI_RE = re.compile(r"^10\.5281/zenodo\.[0-9]+$")


class ReleaseCandidateError(RuntimeError):
    """The post-lock release candidate or supplied asset failed verification."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseCandidateError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ReleaseCandidateError(f"JSON is not an object: {path}")
    return value


def load_json_bytes(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseCandidateError(f"invalid JSON Git blob: {label}") from exc
    if not isinstance(value, dict):
        raise ReleaseCandidateError(f"JSON Git blob is not an object: {label}")
    return value


def read_tsv_bytes(
    payload: bytes,
    *,
    label: str,
    expected_columns: tuple[str, ...],
) -> list[dict[str, str]]:
    try:
        text = payload.decode("utf-8-sig")
        reader = csv.DictReader(text.splitlines(), delimiter="\t")
        if tuple(reader.fieldnames or ()) != expected_columns:
            raise ReleaseCandidateError(
                f"TSV Git blob has an invalid header: {label}"
            )
        return list(reader)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ReleaseCandidateError(f"invalid TSV Git blob: {label}") from exc


def git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ReleaseCandidateError(
            f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def git_blob(repository: Path, ref: str, relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repository), "show", f"{ref}:{relative}"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        error = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ReleaseCandidateError(
            f"git blob read failed for {ref}:{relative}: {error}"
        )
    return completed.stdout


def verify_file_record(
    path: Path,
    *,
    expected_bytes: str,
    expected_sha256: str,
    label: str,
) -> None:
    if not path.is_file():
        raise ReleaseCandidateError(f"{label} is missing: {path}")
    try:
        size = int(expected_bytes)
    except ValueError as exc:
        raise ReleaseCandidateError(
            f"{label} has invalid expected bytes: {expected_bytes!r}"
        ) from exc
    checksum = expected_sha256.strip().lower()
    if size < 0 or not SHA256_RE.fullmatch(checksum):
        raise ReleaseCandidateError(f"{label} has invalid integrity metadata")
    if path.stat().st_size != size or sha256_path(path) != checksum:
        raise ReleaseCandidateError(f"{label} size or SHA-256 mismatch")


def verify_blob_record(
    payload: bytes,
    *,
    expected_bytes: str,
    expected_sha256: str,
    label: str,
) -> None:
    try:
        size = int(expected_bytes)
    except ValueError as exc:
        raise ReleaseCandidateError(
            f"{label} has invalid expected bytes: {expected_bytes!r}"
        ) from exc
    checksum = expected_sha256.strip().lower()
    if (
        size < 0
        or not SHA256_RE.fullmatch(checksum)
        or len(payload) != size
        or sha256_bytes(payload) != checksum
    ):
        raise ReleaseCandidateError(
            f"{label} Git blob size or SHA-256 mismatch"
        )


def keyed_rows(
    rows: list[dict[str, str]],
    *,
    key: str,
    expected: set[str],
    label: str,
) -> dict[str, dict[str, str]]:
    keyed = {row.get(key, ""): row for row in rows}
    if len(keyed) != len(rows):
        raise ReleaseCandidateError(f"{label} contains duplicate or empty keys")
    if set(keyed) != expected:
        raise ReleaseCandidateError(
            f"{label} has missing or unexpected members"
        )
    return keyed


def normalized_recorded_path(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.replace("\\", "/").rstrip("/").lower()


def verify_focused_archive_evidence(
    *,
    core_archive: Path,
    focused: dict[str, dict[str, str]],
    analysis_lock: str,
) -> None:
    prefix = "results/ted_bib_focused_81/"
    try:
        with zipfile.ZipFile(core_archive) as handle:
            payloads = {
                name: handle.read(prefix + name)
                for name in FOCUSED_NAMES
            }
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise ReleaseCandidateError(
            "core archive lacks valid focused-81 evidence"
        ) from exc
    for name, payload in payloads.items():
        row = focused[name]
        if (
            row.get("path") != prefix + name
            or row.get("role") != FOCUSED_ROLES[name]
            or len(payload) != int(row["bytes"])
            or hashlib.sha256(payload).hexdigest() != row["sha256"]
        ):
            raise ReleaseCandidateError(
                f"packaged focused evidence disagrees with manifest: {name}"
            )

    try:
        collection = payloads["focused_81_collection.txt"].decode("utf-8")
        terminal = payloads[
            "focused_81_terminal_summary.txt"
        ].decode("utf-8")
        command = json.loads(
            payloads["focused_81_command.json"].decode("utf-8")
        )
        junit_root = ElementTree.fromstring(
            payloads["focused_81_junit.xml"]
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ElementTree.ParseError,
    ) as exc:
        raise ReleaseCandidateError(
            "packaged focused evidence is not parseable"
        ) from exc
    if not isinstance(command, dict):
        raise ReleaseCandidateError(
            "packaged focused command record is not an object"
        )
    node_ids = [
        line
        for line in collection.splitlines()
        if "::test_" in line and not line.startswith((" ", "="))
    ]
    if len(node_ids) != 81 or len(set(node_ids)) != 81:
        raise ReleaseCandidateError(
            "focused collection must contain 81 unique node IDs"
        )
    selection_payload = ("\n".join(node_ids) + "\n").encode("utf-8")
    selection = command.get("selection")
    runtime = command.get("runtime")
    environment = command.get("environment_controls")
    evidence_hashes = command.get("evidence_sha256")
    argv = command.get("argv")
    expected_result = {
        "tests": 81,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }
    if not isinstance(selection, dict) or not isinstance(runtime, dict):
        raise ReleaseCandidateError(
            "focused command lacks selection or runtime"
        )
    pyfgsea_runtime = runtime.get("pyfgsea")
    if (
        command.get("evidence_kind") != "ted_v1.1.0_focused_81"
        or command.get("analysis_lock_commit") != analysis_lock
        or command.get("repository_dirty") is not False
        or command.get("result") != expected_result
        or selection.get("test_files") != FOCUSED_TEST_FILES
        or selection.get("excluded_v1_1_extension_tests")
        != FOCUSED_EXCLUDED_TESTS
        or selection.get("collected") != 81
        or selection.get("node_ids") != node_ids
        or selection.get("selection_manifest_sha256")
        != hashlib.sha256(selection_payload).hexdigest()
        or not isinstance(pyfgsea_runtime, dict)
        or pyfgsea_runtime.get("version") != "0.2.0"
        or not isinstance(environment, dict)
        or environment.get("isolated_child_python") is not True
        or environment.get("PYTHONPATH") != "removed"
        or not isinstance(evidence_hashes, dict)
        or not isinstance(argv, list)
        or not all(isinstance(item, str) for item in argv)
        or len(argv) < 12
        or argv[1:3] != ["-I", "-c"]
        or "-p" not in argv
        or "no:cacheprovider" not in argv
        or "--import-mode=importlib" not in argv
        or not any(item.startswith("--junitxml=") for item in argv)
    ):
        raise ReleaseCandidateError(
            "focused command disagrees with the fixed isolated selection"
        )
    repository_record = normalized_recorded_path(argv[4])
    distribution_record = normalized_recorded_path(argv[5])
    import_root = normalized_recorded_path(pyfgsea_runtime.get("import_root"))
    import_origin = normalized_recorded_path(
        pyfgsea_runtime.get("import_origin")
    )
    installed_guard = normalized_recorded_path(
        environment.get("installed_distribution_root_inserted_by_guard")
    )
    support_guard = normalized_recorded_path(
        environment.get("support_path_inserted_by_guard")
    )
    if (
        not repository_record
        or not distribution_record
        or repository_record == distribution_record
        or import_root != distribution_record
        or installed_guard != distribution_record
        or not import_origin.startswith(distribution_record + "/")
        or import_origin.startswith(repository_record + "/")
        or support_guard != repository_record + "/scripts"
    ):
        raise ReleaseCandidateError(
            "focused runtime was not isolated from the source checkout"
        )
    expected_hashes = {
        "collection": hashlib.sha256(
            payloads["focused_81_collection.txt"]
        ).hexdigest(),
        "junit": hashlib.sha256(
            payloads["focused_81_junit.xml"]
        ).hexdigest(),
        "terminal_summary": hashlib.sha256(
            payloads["focused_81_terminal_summary.txt"]
        ).hexdigest(),
    }
    if evidence_hashes != expected_hashes:
        raise ReleaseCandidateError(
            "focused command evidence hashes do not match packaged bytes"
        )
    if re.search(r"\b81 passed,\s+7 deselected\b", terminal) is None:
        raise ReleaseCandidateError(
            "focused terminal summary lacks exact pass/deselection counts"
        )

    suites = (
        [junit_root]
        if junit_root.tag == "testsuite"
        else list(junit_root.findall("testsuite"))
    )
    testcases = list(junit_root.iter("testcase"))
    testcase_ids = {
        (case.get("classname", ""), case.get("name", ""))
        for case in testcases
    }
    try:
        counts = {
            field: sum(int(suite.get(field, "0")) for suite in suites)
            for field in ("tests", "failures", "errors", "skipped")
        }
    except ValueError as exc:
        raise ReleaseCandidateError(
            "focused JUnit aggregate counts are invalid"
        ) from exc
    if (
        counts != expected_result
        or len(testcases) != 81
        or len(testcase_ids) != 81
        or any(
            case.find(kind) is not None
            for case in testcases
            for kind in ("failure", "error", "skipped")
        )
    ):
        raise ReleaseCandidateError(
            "focused JUnit lacks 81 unique passing testcase records"
        )


def verify_candidate(
    *,
    repository: Path,
    release_ref: str,
    bundle_dir: Path | None,
    wheel: Path | None,
    figure_report: Path | None,
    require_publication_identifiers: bool,
) -> dict[str, object]:
    repository = repository.resolve()
    top = Path(git(repository, "rev-parse", "--show-toplevel")).resolve()
    if top != repository:
        raise ReleaseCandidateError(
            f"repository must be the Git top level: {top}"
        )
    if git(repository, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ReleaseCandidateError("repository is not tracked-clean")

    release_commit = git(
        repository,
        "rev-parse",
        f"{release_ref}^{{commit}}",
    ).lower()
    head_commit = git(repository, "rev-parse", "HEAD^{commit}").lower()
    if release_commit != head_commit:
        raise ReleaseCandidateError(
            "--release-ref must resolve to the clean checked-out HEAD"
        )
    metadata_path = (RELEASE_DIR / "RELEASE_METADATA.json").as_posix()
    metadata = load_json_bytes(
        git_blob(repository, release_commit, metadata_path),
        label=metadata_path,
    )
    analysis_lock = str(metadata.get("analysis_lock_commit", "")).lower()
    if analysis_lock != ANALYSIS_LOCK_COMMIT:
        raise ReleaseCandidateError(
            "release metadata does not use the independently pinned analysis lock"
        )
    resolved_lock = git(
        repository,
        "rev-parse",
        f"{analysis_lock}^{{commit}}",
    ).lower()
    if resolved_lock != analysis_lock:
        raise ReleaseCandidateError("analysis-lock commit does not resolve")
    ancestry = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "merge-base",
            "--is-ancestor",
            analysis_lock,
            release_commit,
        ],
        check=False,
    )
    if ancestry.returncode != 0:
        raise ReleaseCandidateError(
            "analysis lock is not an ancestor of the release candidate"
        )
    analysis_tree = git(
        repository,
        "show",
        "-s",
        "--format=%T",
        analysis_lock,
    ).lower()
    if (
        analysis_tree != ANALYSIS_LOCK_TREE
        or metadata.get("analysis_lock_tree") != ANALYSIS_LOCK_TREE
    ):
        raise ReleaseCandidateError(
            "release metadata analysis-lock tree does not match Git"
        )
    if (
        git(repository, "rev-parse", "ted-v1.0.0").lower()
        != BASELINE_TAG_OBJECT
        or git(repository, "cat-file", "-t", "ted-v1.0.0") != "tag"
        or git(repository, "rev-parse", "ted-v1.0.0^{commit}").lower()
        != BASELINE_COMMIT
    ):
        raise ReleaseCandidateError(
            "immutable ted-v1.0.0 tag object or commit has changed"
        )
    changed = {
        line
        for line in git(
            repository,
            "diff",
            "--name-only",
            f"{analysis_lock}..{release_commit}",
        ).splitlines()
        if line
    }
    unexpected = sorted(changed - ALLOWED_POST_LOCK_PATHS)
    if unexpected:
        raise ReleaseCandidateError(
            f"post-lock paths exceed the release-only allowlist: {unexpected}"
        )
    if git(repository, "rev-list", "--merges", f"{analysis_lock}..{release_commit}"):
        raise ReleaseCandidateError("post-lock release history must not contain merges")
    for commit in git(
        repository,
        "rev-list",
        "--reverse",
        f"{analysis_lock}..{release_commit}",
    ).splitlines():
        commit_paths = {
            line
            for line in git(
                repository,
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                f"{commit}^",
                commit,
            ).splitlines()
            if line
        }
        commit_unexpected = sorted(commit_paths - ALLOWED_POST_LOCK_PATHS)
        if commit_unexpected:
            raise ReleaseCandidateError(
                "post-lock commit changes paths outside the release-only "
                f"allowlist: {commit}: {commit_unexpected}"
            )
    declared_allowed = metadata.get("post_lock_allowed_paths")
    if declared_allowed != sorted(ALLOWED_POST_LOCK_PATHS):
        raise ReleaseCandidateError(
            "release metadata post-lock allowlist disagrees with verifier"
        )
    if (
        metadata.get("schema_version") != "ted_v1.1.0_release_metadata_v1"
        or metadata.get("release_tag") != "ted-v1.1.0"
        or metadata.get("python_distribution") != "pyfgsea==0.2.0"
        or metadata.get("baseline_tag") != "ted-v1.0.0"
        or metadata.get("baseline_doi") != "10.5281/zenodo.21403133"
        or metadata.get("release_candidate_status")
        != "verified_not_published"
    ):
        raise ReleaseCandidateError(
            "release identity or immutable baseline metadata is invalid"
        )

    analysis_path = (RELEASE_DIR / "ANALYSIS_LOCK.json").as_posix()
    analysis_record = load_json_bytes(
        git_blob(repository, release_commit, analysis_path),
        label=analysis_path,
    )
    if (
        analysis_record.get("analysis_lock_commit") != analysis_lock
        or analysis_record.get("analysis_lock_tree") != analysis_tree
        or analysis_record.get("repository_dirty") is not False
        or analysis_record.get("claim_boundary") != EXPECTED_CLAIM_BOUNDARY
        or analysis_record.get("external_evidence")
        != EXPECTED_EXTERNAL_EVIDENCE
    ):
        raise ReleaseCandidateError(
            "ANALYSIS_LOCK.json disagrees with Git, claim boundary, or evidence"
        )
    contract_hashes = analysis_record.get("contract_sha256")
    if (
        not isinstance(contract_hashes, dict)
        or set(contract_hashes) != EXPECTED_CONTRACT_PATHS
    ):
        raise ReleaseCandidateError(
            "ANALYSIS_LOCK.json has missing or unexpected contract hashes"
        )
    for relative, expected in contract_hashes.items():
        if (
            not isinstance(relative, str)
            or not isinstance(expected, str)
            or relative.startswith(("/", "\\"))
            or ".." in Path(relative).parts
            or not SHA256_RE.fullmatch(expected)
        ):
            raise ReleaseCandidateError("analysis contract hash map is invalid")
        payload = git_blob(repository, analysis_lock, relative)
        if sha256_bytes(payload) != expected:
            raise ReleaseCandidateError(
                f"analysis contract hash mismatch: {relative}"
            )

    build_index_relative = (RELEASE_DIR / "BUILD_INDEX.json").as_posix()
    build_index_payload = git_blob(
        repository,
        release_commit,
        build_index_relative,
    )
    build_index = load_json_bytes(
        build_index_payload,
        label=build_index_relative,
    )
    expected_archives = [
        {
            "file": "ted-bib-companion-v1.1.0-core.zip",
            "package": "core",
        },
        {
            "file": "ted-bib-companion-v1.1.0-native-outputs.zip",
            "package": "native_outputs",
        },
        {
            "file": "ted-bib-companion-v1.1.0-stability-shards.zip",
            "package": "stability_shards",
        },
    ]
    if (
        build_index.get("analysis_lock_commit") != analysis_lock
        or build_index.get("release") != "ted-v1.1.0"
        or build_index.get("status") != "built_not_published"
        or build_index.get("zenodo_version_doi") is not None
        or build_index.get("archives") != expected_archives
        or build_index.get("asset_spec_sha256") != ASSET_SPEC_SHA256
    ):
        raise ReleaseCandidateError(
            "tracked BUILD_INDEX.json has invalid release bindings"
        )

    tracked_records = metadata.get("tracked_attestation_sha256")
    if not isinstance(tracked_records, dict):
        raise ReleaseCandidateError(
            "release metadata lacks tracked attestation hashes"
        )
    if set(tracked_records) != TRACKED_ATTESTATION_PATHS:
        raise ReleaseCandidateError(
            "tracked attestation hash map has missing or unexpected paths"
        )
    for relative, expected in tracked_records.items():
        if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected):
            raise ReleaseCandidateError(
                "tracked attestation hash map is invalid"
            )
        payload = git_blob(repository, release_commit, relative)
        if sha256_bytes(payload) != expected:
            raise ReleaseCandidateError(
                f"tracked attestation hash mismatch: {relative}"
            )

    external_relative = (
        RELEASE_DIR / "EXTERNAL_ARCHIVE_ASSETS.tsv"
    ).as_posix()
    external_rows = read_tsv_bytes(
        git_blob(repository, release_commit, external_relative),
        label=external_relative,
        expected_columns=EXTERNAL_COLUMNS,
    )
    required_external = ARCHIVE_NAMES | {
        "BUILD_INDEX.json",
        "ARCHIVE_MANIFEST.tsv",
        "pyfgsea-0.2.0-cp38-abi3-win_amd64.whl",
        "pyfgsea-ted-v1.1.0-figure-report.json",
    }
    external = keyed_rows(
        external_rows,
        key="asset_name",
        expected=required_external,
        label="external asset manifest",
    )
    for name, row in external.items():
        if (
            row.get("verification_status") != "verified"
            or row.get("asset_role") != EXTERNAL_ROLES[name]
            or not row.get("bytes", "").isdigit()
            or not SHA256_RE.fullmatch(row.get("sha256", "").lower())
        ):
            raise ReleaseCandidateError(
                f"external asset is not locally verified: {name}"
            )

    archive_relative = (RELEASE_DIR / "ARCHIVE_MANIFEST.tsv").as_posix()
    archive_manifest_payload = git_blob(
        repository,
        release_commit,
        archive_relative,
    )
    archive_rows = read_tsv_bytes(
        archive_manifest_payload,
        label=archive_relative,
        expected_columns=ARCHIVE_COLUMNS,
    )
    archive = keyed_rows(
        archive_rows,
        key="file",
        expected=ARCHIVE_NAMES | {"BUILD_INDEX.json"},
        label="tracked archive manifest",
    )
    for name, row in archive.items():
        external_row = external[name]
        if (
            row.get("role") != ARCHIVE_ROLES[name]
            or
            row.get("bytes") != external_row.get("bytes")
            or row.get("sha256") != external_row.get("sha256")
        ):
            raise ReleaseCandidateError(
                f"archive and external manifests disagree for {name}"
            )
    verify_blob_record(
        archive_manifest_payload,
        expected_bytes=external["ARCHIVE_MANIFEST.tsv"]["bytes"],
        expected_sha256=external["ARCHIVE_MANIFEST.tsv"]["sha256"],
        label="tracked ARCHIVE_MANIFEST.tsv",
    )
    verify_blob_record(
        build_index_payload,
        expected_bytes=external["BUILD_INDEX.json"]["bytes"],
        expected_sha256=external["BUILD_INDEX.json"]["sha256"],
        label="tracked BUILD_INDEX.json",
    )

    focused_relative = (RELEASE_DIR / "FOCUSED_81_EVIDENCE.tsv").as_posix()
    focused_rows = read_tsv_bytes(
        git_blob(repository, release_commit, focused_relative),
        label=focused_relative,
        expected_columns=FOCUSED_COLUMNS,
    )
    focused = keyed_rows(
        focused_rows,
        key="artifact",
        expected=FOCUSED_NAMES,
        label="focused evidence manifest",
    )
    for row in focused.values():
        if (
            row.get("analysis_lock_commit") != analysis_lock
            or row.get("verification_status") != "verified"
            or not row.get("bytes", "").isdigit()
            or not SHA256_RE.fullmatch(row.get("sha256", "").lower())
        ):
            raise ReleaseCandidateError(
                "focused evidence manifest is incomplete or unbound"
            )

    if bundle_dir is not None:
        bundle_dir = bundle_dir.resolve()
        for name in ARCHIVE_NAMES | {
            "BUILD_INDEX.json",
            "ARCHIVE_MANIFEST.tsv",
        }:
            row = external[name]
            verify_file_record(
                bundle_dir / name,
                expected_bytes=row["bytes"],
                expected_sha256=row["sha256"],
                label=name,
            )
        verify_focused_archive_evidence(
            core_archive=(
                bundle_dir / "ted-bib-companion-v1.1.0-core.zip"
            ),
            focused=focused,
            analysis_lock=analysis_lock,
        )
        scripts = repository / "scripts"
        sys.path.insert(0, str(scripts))
        from verify_ted_bib_companion import verify_bundle

        bundle_report = verify_bundle(bundle_dir)
        if (
            bundle_report.get("analysis_lock_commit") != analysis_lock
            or bundle_report.get("common_tasks") != 480
            or bundle_report.get("native_method_task_outputs") != 2400
            or bundle_report.get("focused_tests") != 81
        ):
            raise ReleaseCandidateError(
                "bundle verifier returned invalid analysis bindings or counts"
            )
    else:
        bundle_report = None

    for path, name in (
        (wheel, "pyfgsea-0.2.0-cp38-abi3-win_amd64.whl"),
        (figure_report, "pyfgsea-ted-v1.1.0-figure-report.json"),
    ):
        if path is not None:
            row = external[name]
            verify_file_record(
                path.resolve(),
                expected_bytes=row["bytes"],
                expected_sha256=row["sha256"],
                label=name,
            )
    if figure_report is not None:
        figure = load_json(figure_report.resolve())
        bundle_verification = figure.get("bundle_verification")
        redraw_results = figure.get("redraw_results")
        figure3 = (
            redraw_results.get("figure3")
            if isinstance(redraw_results, dict)
            else None
        )
        figure5 = (
            redraw_results.get("figure5")
            if isinstance(redraw_results, dict)
            else None
        )
        if (
            figure.get("status") != "verified"
            or figure.get("redraw_status")
            != "redrawn_and_semantically_verified"
            or figure.get("chain_members") != 15
            or figure.get("source_to_figure_chain_status")
            != "verified_packaged_sources_and_outputs"
            or not isinstance(bundle_verification, dict)
            or bundle_verification.get("analysis_lock_commit") != analysis_lock
            or bundle_verification.get("release") != "ted-v1.1.0"
            or bundle_verification.get("status") != "verified"
            or bundle_verification.get("common_tasks") != 480
            or bundle_verification.get("native_method_task_outputs") != 2400
            or bundle_verification.get("figure3_source_tables") != 4
            or bundle_verification.get("figure5_source_tables") != 7
            or bundle_verification.get("focused_tests") != 81
            or not isinstance(figure3, dict)
            or not isinstance(figure5, dict)
            or figure3.get("status")
            != "redrawn_and_semantically_verified"
            or figure5.get("status")
            != "redrawn_and_semantically_verified"
            or figure3.get("semantic_checks") != 15
            or figure5.get("semantic_checks") != 13
            or not isinstance(figure3.get("pixel_qa"), dict)
            or not isinstance(figure5.get("pixel_qa"), dict)
            or figure3["pixel_qa"].get("passed") is not True
            or figure5["pixel_qa"].get("passed") is not True
        ):
            raise ReleaseCandidateError(
                "figure reproduction report is not fully verified or bound"
            )

    doi = metadata.get("zenodo_version_doi")
    if doi is not None and (
        not isinstance(doi, str)
        or not DOI_RE.fullmatch(doi)
        or doi == "10.5281/zenodo.21403133"
    ):
        raise ReleaseCandidateError(
            "Zenodo DOI is invalid or reuses the immutable baseline DOI"
        )
    if require_publication_identifiers:
        if doi is None:
            raise ReleaseCandidateError(
                "exact-tag publication requires a reserved v1.1.0 Zenodo DOI"
            )
        if git(repository, "cat-file", "-t", "refs/tags/ted-v1.1.0") != "tag":
            raise ReleaseCandidateError(
                "exact-tag publication requires an annotated ted-v1.1.0 tag"
            )
        tag_commit = git(
            repository,
            "rev-parse",
            "refs/tags/ted-v1.1.0^{commit}",
        ).lower()
        if tag_commit != release_commit:
            raise ReleaseCandidateError(
                "ted-v1.1.0 does not resolve to the candidate commit"
            )

    return {
        "status": "verified",
        "release_commit": release_commit,
        "analysis_lock_commit": analysis_lock,
        "post_lock_changed_paths": sorted(changed),
        "external_assets": len(external),
        "bundle_verified": bundle_report is not None,
        "wheel_verified": wheel is not None,
        "figure_report_verified": figure_report is not None,
        "zenodo_version_doi": doi,
        "publication_identifiers_required": require_publication_identifiers,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--release-ref", default="HEAD")
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--figure-report", type=Path)
    parser.add_argument(
        "--require-publication-identifiers",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = verify_candidate(
            repository=args.repository,
            release_ref=args.release_ref,
            bundle_dir=args.bundle_dir,
            wheel=args.wheel,
            figure_report=args.figure_report,
            require_publication_identifiers=args.require_publication_identifiers,
        )
    except (
        ReleaseCandidateError,
        OSError,
        ValueError,
        ImportError,
    ) as exc:
        print(f"release-candidate verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
