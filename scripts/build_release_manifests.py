"""Build fail-closed, Git-object-backed TED release manifests.

The three manifest classes deliberately have different scopes:

* ``FROZEN_SCIENTIFIC_ARTIFACTS.tsv`` compares the declared publication
  artifacts in a baseline ref with a patch-release ref.
* ``RELEASE_TREE_MANIFEST.tsv`` enumerates a committed Git tree. Paths come
  from ``git ls-tree`` and bytes and SHA-256 values come from raw Git blobs,
  never a checkout.
* ``EXTERNAL_ARCHIVE_ASSETS.tsv`` records release assets that are not members
  of the Git tree.  Pending rows stay explicitly pending.

This module uses only the Python standard library so the release audit can run
before the package or its scientific dependencies are installed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

FROZEN_EXACT_PATHS = frozenset(
    {
        "results/ted_v1_submission/build_record.json",
        "results/ted_v1_submission/evidence_axis_legacy_crosswalk.tsv",
        "results/ted_v1_submission/figure_manifest.tsv",
        "results/ted_v1_submission/graphical_abstract_alt_text.txt",
        "results/ted_v1_submission/gse271399_design_stratum_audit.tsv",
        "results/ted_v1_submission/gse93735_ev_boundary.tsv",
    }
)
FROZEN_PREFIX_RULES = (
    ("results/ted_v1_submission/figure_source_data/", (".tsv",)),
    ("results/ted_v1_submission/figures/", (".pdf", ".png")),
    ("results/ted_v1_submission/supplementary_figures/", (".pdf", ".png")),
    ("results/ted_v1_submission/gse153056_block_aware/", (".tsv",)),
    ("results/ted_v1_submission/packet_bootstrap/", None),
)
PROVENANCE_PATHS = frozenset(
    {
        "results/ted_v1_submission/build_record.json",
        "results/ted_v1_submission/figure_manifest.tsv",
    }
)
EXTERNAL_ASSET_COLUMNS = (
    "asset_name",
    "asset_role",
    "archive_location",
    "source_ref_or_manifest",
    "bytes",
    "sha256",
    "verification_status",
    "notes",
)
V1_SUBMISSION_PREFIX = "results/ted_v1_submission/"


@dataclass(frozen=True)
class GitEntry:
    path: str
    mode: str
    oid: str
    size: int
    sha256: str


class GitAuditError(RuntimeError):
    """Raised when a Git-backed release audit cannot be completed."""


def _git(
    repository: Path,
    args: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    text: bool = False,
) -> bytes | str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=False,
        capture_output=True,
        env=env,
        text=text,
    )
    if completed.returncode != 0:
        stderr = completed.stderr if text else completed.stderr.decode("utf-8", errors="replace")
        raise GitAuditError(f"git {' '.join(args)} failed: {stderr.strip()}")
    return completed.stdout


def resolve_commit(repository: Path, ref: str) -> str:
    """Resolve ``ref`` to a commit without accepting a working-tree alias."""

    resolved = _git(repository, ["rev-parse", "--verify", f"{ref}^{{commit}}"], text=True)
    return str(resolved).strip()


def resolve_tree(repository: Path, ref: str) -> str:
    """Resolve ``ref`` to its root tree object."""

    resolved = _git(repository, ["rev-parse", "--verify", f"{ref}^{{tree}}"], text=True)
    return str(resolved).strip()


def _tree_entries(repository: Path, ref: str) -> list[tuple[str, str, str]]:
    """Enumerate one committed tree without consulting the index or checkout."""

    raw = _git(repository, ["ls-tree", "-r", "-z", "--full-tree", ref])
    assert isinstance(raw, bytes)

    entries: list[tuple[str, str, str]] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, encoded_path = record.split(b"\t", 1)
        mode, object_type, oid = metadata.decode("ascii").split(" ")
        if object_type != "blob":
            raise GitAuditError(
                f"{encoded_path!r} points to Git object type {object_type!r}; "
                "submodules are unsupported"
            )
        path = encoded_path.decode("utf-8", errors="surrogateescape").replace("\\", "/")
        entries.append((path, mode, oid))
    return sorted(entries, key=lambda item: item[0])


def git_entries(repository: Path, ref: str) -> dict[str, GitEntry]:
    """Return committed paths with hashes of raw Git-object bytes."""

    cache: dict[str, tuple[int, str]] = {}
    result: dict[str, GitEntry] = {}
    for path, mode, oid in _tree_entries(repository, ref):
        if oid not in cache:
            payload = _git(repository, ["cat-file", "blob", oid])
            assert isinstance(payload, bytes)
            cache[oid] = (len(payload), hashlib.sha256(payload).hexdigest())
        size, digest = cache[oid]
        result[path] = GitEntry(path=path, mode=mode, oid=oid, size=size, sha256=digest)
    return result


def aggregate_digest(entries: Iterable[GitEntry]) -> str:
    """Hash canonical ``path<TAB>bytes<TAB>sha256<LF>`` records."""

    canonical = "".join(
        f"{entry.path}\t{entry.size}\t{entry.sha256}\n"
        for entry in sorted(entries, key=lambda item: item.path)
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def is_frozen_artifact(path: str) -> bool:
    if path in FROZEN_EXACT_PATHS:
        return True
    for prefix, suffixes in FROZEN_PREFIX_RULES:
        if path.startswith(prefix) and (suffixes is None or path.endswith(suffixes)):
            return True
    return False


def _write_tsv(path: Path, columns: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _external_assets(
    template: Path | None,
    output: Path,
) -> int:
    if template is None:
        rows: list[dict[str, object]] = []
    else:
        with template.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != EXTERNAL_ASSET_COLUMNS:
                raise GitAuditError(
                    f"{template} must have columns: {', '.join(EXTERNAL_ASSET_COLUMNS)}"
                )
            rows = list(reader)
        for row_number, row in enumerate(rows, start=2):
            status = str(row["verification_status"]).strip()
            size = str(row["bytes"]).strip()
            digest = str(row["sha256"]).strip().lower()
            if status == "verified":
                if not size.isdigit() or len(digest) != 64:
                    raise GitAuditError(
                        f"{template}:{row_number}: verified assets require bytes and SHA-256"
                    )
            elif status not in {"pending_not_built", "pending_external_upload", "not_applicable"}:
                raise GitAuditError(
                    f"{template}:{row_number}: unsupported verification_status={status!r}"
                )
    _write_tsv(output, EXTERNAL_ASSET_COLUMNS, rows)
    return len(rows)


def build_manifests(
    repository: Path,
    baseline_ref: str,
    release_ref: str,
    output_dir: Path,
    external_assets_template: Path | None = None,
) -> dict[str, object]:
    repository = repository.resolve()
    output_dir = output_dir.resolve()
    baseline_commit = resolve_commit(repository, baseline_ref)
    release_commit = resolve_commit(repository, release_ref)
    baseline_tree = resolve_tree(repository, baseline_commit)
    release_tree = resolve_tree(repository, release_commit)
    baseline = git_entries(repository, baseline_commit)
    release = git_entries(repository, release_commit)

    frozen_paths = sorted(path for path in baseline if is_frozen_artifact(path))
    if not frozen_paths:
        raise GitAuditError(f"no frozen artifacts matched baseline ref {baseline_ref!r}")

    frozen_rows: list[dict[str, object]] = []
    differences: list[str] = []
    for path in frozen_paths:
        old = baseline[path]
        new = release.get(path)
        identical = new is not None and old.sha256 == new.sha256 and old.size == new.size
        if not identical:
            differences.append(path)
        frozen_rows.append(
            {
                "artifact_class": "provenance" if path in PROVENANCE_PATHS else "scientific_result",
                "path": path,
                "baseline_ref": baseline_ref,
                "baseline_commit": baseline_commit,
                "baseline_tree": baseline_tree,
                "baseline_git_blob": old.oid,
                "baseline_bytes": old.size,
                "baseline_sha256": old.sha256,
                "release_ref": release_ref,
                "release_commit": release_commit,
                "release_tree": release_tree,
                "release_git_blob": "" if new is None else new.oid,
                "release_bytes": "" if new is None else new.size,
                "release_sha256": "" if new is None else new.sha256,
                "byte_identical": str(identical).lower(),
            }
        )

    v1_submission_paths = sorted(
        {
            path
            for path in baseline.keys() | release.keys()
            if path.startswith(V1_SUBMISSION_PREFIX)
        }
    )
    v1_submission_differences = [
        path
        for path in v1_submission_paths
        if baseline.get(path) != release.get(path)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_tsv(
        output_dir / "FROZEN_SCIENTIFIC_ARTIFACTS.tsv",
        (
            "artifact_class",
            "path",
            "baseline_ref",
            "baseline_commit",
            "baseline_tree",
            "baseline_git_blob",
            "baseline_bytes",
            "baseline_sha256",
            "release_ref",
            "release_commit",
            "release_tree",
            "release_git_blob",
            "release_bytes",
            "release_sha256",
            "byte_identical",
        ),
        frozen_rows,
    )

    tree_rows = [
        {
            "path": entry.path,
            "release_ref": release_ref,
            "release_commit": release_commit,
            "release_tree": release_tree,
            "git_mode": entry.mode,
            "git_blob": entry.oid,
            "bytes": entry.size,
            "sha256": entry.sha256,
        }
        for entry in release.values()
    ]
    _write_tsv(
        output_dir / "RELEASE_TREE_MANIFEST.tsv",
        (
            "path",
            "release_ref",
            "release_commit",
            "release_tree",
            "git_mode",
            "git_blob",
            "bytes",
            "sha256",
        ),
        tree_rows,
    )

    external_count = _external_assets(
        external_assets_template,
        output_dir / "EXTERNAL_ARCHIVE_ASSETS.tsv",
    )
    summary: dict[str, object] = {
        "manifest_schema_version": "1.0",
        "baseline_ref": baseline_ref,
        "baseline_commit": baseline_commit,
        "baseline_tree": baseline_tree,
        "release_ref": release_ref,
        "release_commit": release_commit,
        "release_tree": release_tree,
        "frozen_artifact_count": len(frozen_rows),
        "frozen_scientific_result_count": sum(
            row["artifact_class"] == "scientific_result" for row in frozen_rows
        ),
        "frozen_provenance_count": sum(
            row["artifact_class"] == "provenance" for row in frozen_rows
        ),
        "frozen_difference_count": len(differences),
        "frozen_differences": differences,
        "all_frozen_artifacts_byte_identical": not differences,
        "v1_submission_tree_difference_count": len(v1_submission_differences),
        "v1_submission_tree_differences": v1_submission_differences,
        "all_v1_submission_tree_byte_identical": not v1_submission_differences,
        "frozen_scientific_aggregate_sha256": aggregate_digest(
            baseline[path]
            for path in frozen_paths
            if path not in PROVENANCE_PATHS
        ),
        "frozen_provenance_aggregate_sha256": aggregate_digest(
            baseline[path]
            for path in frozen_paths
            if path in PROVENANCE_PATHS
        ),
        "release_tree_file_count": len(tree_rows),
        "external_asset_row_count": external_count,
        "enumeration": "git ls-tree -r -z --full-tree",
        "hash_source": "raw git cat-file blob bytes",
    }
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--baseline-ref", default="ted-v1.0.0")
    parser.add_argument("--release-ref", default="HEAD")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--external-assets-template", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = build_manifests(
            args.repository,
            args.baseline_ref,
            args.release_ref,
            args.outdir,
            args.external_assets_template,
        )
    except GitAuditError as exc:
        print(f"release manifest audit failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    passed = bool(
        summary["all_frozen_artifacts_byte_identical"]
        and summary["all_v1_submission_tree_byte_identical"]
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
