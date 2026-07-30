from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_release_manifests.py"


def _git(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _commit(repository: Path, message: str) -> str:
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", message)
    return _git(repository, "rev-parse", "HEAD")


def _make_repository(tmp_path: Path) -> tuple[Path, str, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.email", "release-test@example.invalid")
    _git(repository, "config", "user.name", "Release Test")
    _write(
        repository / "results/ted_v1_submission/figure_source_data/figure3.tsv",
        "metric\tvalue\nauprc\t0.8\n",
    )
    _write(repository / "results/ted_v1_submission/figures/figure3.pdf", "%PDF-test\n")
    _write(repository / "results/ted_v1_submission/build_record.json", '{"seed": 7}\n')
    _write(repository / "results/ted_v1_submission/figure_manifest.tsv", "file\tsha256\n")
    _write(repository / "README.md", "baseline\n")
    baseline = _commit(repository, "baseline")
    _git(repository, "tag", "ted-v1.0.0")
    _write(repository / "README.md", "documentation patch\n")
    release = _commit(repository, "release patch")
    return repository, baseline, release


def _run_builder(
    repository: Path,
    outdir: Path,
    *,
    release_ref: str = "HEAD",
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repository",
            str(repository),
            "--baseline-ref",
            "ted-v1.0.0",
            "--release-ref",
            release_ref,
            "--outdir",
            str(outdir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_manifests_use_committed_git_objects_and_ignore_untracked_files(tmp_path: Path):
    repository, baseline, release = _make_repository(tmp_path)
    _write(
        repository / "results/ted_v1_submission/figure_source_data/untracked.tsv",
        "must\tnot\tappear\n",
    )
    outdir = tmp_path / "audit"
    completed = _run_builder(repository, outdir)
    assert completed.returncode == 0, completed.stderr

    summary = json.loads((outdir / "manifest_summary.json").read_text(encoding="utf-8"))
    assert summary["baseline_commit"] == baseline
    assert summary["release_commit"] == release
    assert summary["all_frozen_artifacts_byte_identical"] is True
    assert summary["frozen_scientific_result_count"] == 2
    assert summary["frozen_provenance_count"] == 2

    with (outdir / "RELEASE_TREE_MANIFEST.tsv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    paths = {row["path"] for row in rows}
    assert "results/ted_v1_submission/figure_source_data/untracked.tsv" not in paths
    assert paths == set(_git(repository, "ls-tree", "-r", "--name-only", release).splitlines())

    figure_row = next(row for row in rows if row["path"].endswith("figure3.tsv"))
    raw_blob = subprocess.run(
        ["git", "-C", str(repository), "cat-file", "blob", figure_row["git_blob"]],
        check=True,
        capture_output=True,
    ).stdout
    import hashlib

    assert figure_row["sha256"] == hashlib.sha256(raw_blob).hexdigest()


def test_builder_fails_closed_when_a_frozen_result_changes(tmp_path: Path):
    repository, _baseline, _release = _make_repository(tmp_path)
    _write(
        repository / "results/ted_v1_submission/figure_source_data/figure3.tsv",
        "metric\tvalue\nauprc\t0.9\n",
    )
    changed = _commit(repository, "change frozen result")
    outdir = tmp_path / "audit"
    completed = _run_builder(repository, outdir, release_ref=changed)
    assert completed.returncode == 1
    summary = json.loads((outdir / "manifest_summary.json").read_text(encoding="utf-8"))
    assert summary["all_frozen_artifacts_byte_identical"] is False
    assert summary["frozen_differences"] == [
        "results/ted_v1_submission/figure_source_data/figure3.tsv"
    ]


def test_builder_fails_closed_when_an_unselected_v1_result_is_added(tmp_path: Path):
    repository, _baseline, _release = _make_repository(tmp_path)
    _write(
        repository / "results/ted_v1_submission/unselected_output.txt",
        "a patch release must not add scientific outputs\n",
    )
    changed = _commit(repository, "add unselected result")
    outdir = tmp_path / "audit"
    completed = _run_builder(repository, outdir, release_ref=changed)
    assert completed.returncode == 1
    summary = json.loads((outdir / "manifest_summary.json").read_text(encoding="utf-8"))
    assert summary["all_frozen_artifacts_byte_identical"] is True
    assert summary["all_v1_submission_tree_byte_identical"] is False
    assert summary["v1_submission_tree_differences"] == [
        "results/ted_v1_submission/unselected_output.txt"
    ]
