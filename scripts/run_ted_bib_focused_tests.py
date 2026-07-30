#!/usr/bin/env python3
"""Run and attest the exact 81-test TED BIB contract from an installed wheel."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

TEST_FILES = (
    "tests/test_ted_evidence.py",
    "tests/test_ted_schema.py",
    "tests/test_ted_flagship.py",
    "tests/test_gse171964_replication_freeze.py",
    "tests/test_nearest_method_benchmark.py",
)
EXCLUDED_V1_1_EXTENSION_TESTS = (
    "test_parallel_evidence_types_never_upgrade_event_e_code",
    "test_canonical_parallel_evidence_schema_matches_embedded_v2_contract",
    "test_canonical_parallel_evidence_schema_enforces_controls_and_replication_id",
    "test_canonical_replication_facets_schema_matches_embedded_v2_contract",
    "test_canonical_replication_facets_keep_event_and_outcome_results_separate",
)
EXPECTED_TESTS = 81
ISOLATED_PYTEST_RUNNER = """
import importlib.metadata
import pathlib
import sys

repository = pathlib.Path(sys.argv[1]).resolve()
distribution_root = pathlib.Path(sys.argv[2]).resolve()
sys.path.insert(0, str(distribution_root))
sys.path.insert(0, str(repository / "scripts"))
import pyfgsea

origin = pathlib.Path(pyfgsea.__file__).resolve()
try:
    origin.relative_to(repository)
except ValueError:
    pass
else:
    raise SystemExit(
        "focused-test guard rejected pyfgsea from the source checkout: "
        + str(origin)
    )
version = importlib.metadata.version("pyfgsea")
if version != "0.2.0":
    raise SystemExit(
        "focused-test guard requires pyfgsea 0.2.0, found " + version
    )

import pytest

raise SystemExit(pytest.main(sys.argv[3:]))
""".strip()


class FocusedTestError(RuntimeError):
    """The focused-test selection or execution contract was not satisfied."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise FocusedTestError(
            f"git {' '.join(args)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def assert_clean_lock(repository: Path, expected_commit: str | None) -> str:
    commit = git(repository, "rev-parse", "HEAD")
    if expected_commit is not None:
        resolved = git(repository, "rev-parse", f"{expected_commit}^{{commit}}")
        if commit != resolved:
            raise FocusedTestError(
                f"HEAD {commit} differs from requested analysis lock {resolved}"
            )
    working_status = git(
        repository,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    if working_status:
        raise FocusedTestError(
            "working-tree changes are present; commit the analysis lock and "
            "remove untracked build inputs before generating focused evidence"
        )
    return commit


def assert_installed_distribution(repository: Path) -> dict[str, str]:
    spec = importlib.util.find_spec("pyfgsea")
    if spec is None or spec.origin is None:
        raise FocusedTestError("the pyfgsea distribution is not importable")
    origin = Path(spec.origin).resolve()
    try:
        origin.relative_to(repository.resolve())
    except ValueError:
        pass
    else:
        raise FocusedTestError(
            "pyfgsea resolves inside the source checkout; run this entry from "
            "outside the repository against an installed wheel"
        )
    version = importlib.metadata.version("pyfgsea")
    if version != "0.2.0":
        raise FocusedTestError(
            f"focused v1.1.0 evidence requires pyfgsea 0.2.0, found {version}"
        )
    return {
        "version": version,
        "import_origin": str(origin),
        "import_root": str(origin.parent.parent),
    }


def selection_args(repository: Path) -> list[str]:
    expression = " and ".join(f"not {name}" for name in EXCLUDED_V1_1_EXTENSION_TESTS)
    return [
        *(str(repository / path) for path in TEST_FILES),
        "-k",
        expression,
        "--import-mode=importlib",
    ]


def run_pytest(
    args: list[str],
    *,
    repository: Path,
    distribution_root: Path,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        pytest_command(
            args,
            repository=repository,
            distribution_root=distribution_root,
        ),
        cwd=repository.parent,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def pytest_command(
    args: list[str],
    *,
    repository: Path,
    distribution_root: Path,
) -> list[str]:
    return [
        sys.executable,
        "-I",
        "-c",
        ISOLATED_PYTEST_RUNNER,
        str(repository),
        str(distribution_root),
        *args,
    ]


def junit_counts(path: Path) -> dict[str, int]:
    root = ElementTree.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        raise FocusedTestError("JUnit evidence contains no test suites")
    return {
        field: sum(int(suite.get(field, "0")) for suite in suites)
        for field in ("tests", "failures", "errors", "skipped")
    }


def build_evidence(
    *,
    repository: Path,
    output_dir: Path,
    expected_commit: str | None,
) -> dict[str, object]:
    repository = repository.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FocusedTestError(
            f"output directory already exists; refusing to overwrite: {output_dir}"
        )
    commit = assert_clean_lock(repository, expected_commit)
    distribution = assert_installed_distribution(repository)
    distribution_root = Path(distribution["import_root"])
    output_dir.mkdir(parents=True)

    environment = os.environ.copy()
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("PYTHONPATH", None)
    selected = selection_args(repository)

    collection = run_pytest(
        ["--collect-only", "-q", "-p", "no:cacheprovider", *selected],
        repository=repository,
        distribution_root=distribution_root,
        environment=environment,
    )
    (output_dir / "focused_81_collection.txt").write_text(
        collection.stdout,
        encoding="utf-8",
    )
    node_ids = [
        line
        for line in collection.stdout.splitlines()
        if "::test_" in line and not line.startswith((" ", "="))
    ]
    if collection.returncode != 0 or len(node_ids) != EXPECTED_TESTS:
        raise FocusedTestError(
            "focused collection must succeed with exactly "
            f"{EXPECTED_TESTS} node IDs; found {len(node_ids)}"
        )
    selection_payload = ("\n".join(node_ids) + "\n").encode("utf-8")

    junit_path = output_dir / "focused_81_junit.xml"
    execution = run_pytest(
        [
            "-q",
            "-p",
            "no:cacheprovider",
            *selected,
            f"--junitxml={junit_path}",
        ],
        repository=repository,
        distribution_root=distribution_root,
        environment=environment,
    )
    terminal_path = output_dir / "focused_81_terminal_summary.txt"
    terminal_path.write_text(execution.stdout, encoding="utf-8")
    if execution.returncode != 0 or not junit_path.is_file():
        raise FocusedTestError(
            "focused execution failed; inspect focused_81_terminal_summary.txt"
        )
    counts = junit_counts(junit_path)
    expected_counts = {
        "tests": EXPECTED_TESTS,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }
    if counts != expected_counts:
        raise FocusedTestError(
            f"focused JUnit counts differ from the contract: {counts}"
        )

    command_record: dict[str, object] = {
        "evidence_kind": "ted_v1.1.0_focused_81",
        "analysis_lock_commit": commit,
        "repository_dirty": False,
        "argv": pytest_command(
            [
                "-q",
                "-p",
                "no:cacheprovider",
                *selected,
                f"--junitxml={junit_path}",
            ],
            repository=repository,
            distribution_root=distribution_root,
        ),
        "environment_controls": {
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": "removed",
            "isolated_child_python": True,
            "installed_distribution_root_inserted_by_guard": str(
                distribution_root
            ),
            "support_path_inserted_by_guard": str(repository / "scripts"),
        },
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "pyfgsea": distribution,
            "pytest": importlib.metadata.version("pytest"),
        },
        "selection": {
            "test_files": list(TEST_FILES),
            "excluded_v1_1_extension_tests": list(EXCLUDED_V1_1_EXTENSION_TESTS),
            "collected": len(node_ids),
            "node_ids": node_ids,
            "selection_manifest_sha256": hashlib.sha256(
                selection_payload
            ).hexdigest(),
        },
        "result": counts,
        "evidence_sha256": {
            "collection": sha256_path(output_dir / "focused_81_collection.txt"),
            "junit": sha256_path(junit_path),
            "terminal_summary": sha256_path(terminal_path),
        },
    }
    command_path = output_dir / "focused_81_command.json"
    command_path.write_text(
        json.dumps(command_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return command_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--analysis-lock-commit",
        help="optional ref that must resolve to the clean repository HEAD",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_evidence(
            repository=args.repository,
            output_dir=args.output_dir,
            expected_commit=args.analysis_lock_commit,
        )
    except (FocusedTestError, OSError, ValueError, ElementTree.ParseError) as exc:
        print(f"focused-test evidence failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report["result"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
