#!/usr/bin/env python3
"""Build the TED BIB v1.1.0 companion from an explicit asset allowlist.

The builder never discovers result files recursively.  Every source is either
listed as a ``file`` rule or is named by a checksummed, row-counted
``manifest_members`` rule.  The latter is the only supported way to collect
large result families such as the 2,400 native method-task outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


RELEASE_VERSION = "1.1.0"
DEFAULT_SPEC = (
    Path(__file__).resolve().parents[1]
    / "release"
    / "ted-v1.1.0"
    / "COMPANION_ASSET_RULES.template.tsv"
)
DEFAULT_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("core", "native_outputs", "stability_shards")
SPEC_COLUMNS = {
    "rule_id",
    "package",
    "mode",
    "source",
    "destination",
    "required",
    "role",
    "member_base",
    "member_destination_prefix",
    "path_column",
    "bytes_column",
    "sha256_column",
    "expected_rows",
    "expected_bytes",
    "expected_sha256",
    "root",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


class CompanionBuildError(RuntimeError):
    """Fail-closed asset or package validation error."""


@dataclass(frozen=True)
class Rule:
    rule_id: str
    package: str
    mode: str
    source: str
    destination: str
    required: bool
    role: str
    member_base: str
    member_destination_prefix: str
    path_column: str
    bytes_column: str
    sha256_column: str
    expected_rows: int | None
    expected_bytes: int | None
    expected_sha256: str | None
    root: str


@dataclass(frozen=True)
class Asset:
    package: str
    source: Path
    destination: str
    rule_id: str
    role: str
    expected_bytes: int | None
    expected_sha256: str | None


@dataclass(frozen=True)
class WrittenAsset:
    destination: str
    size: int
    sha256: str
    rule_id: str
    role: str


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_bool(value: str, *, field: str, rule_id: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise CompanionBuildError(
        f"{rule_id}: {field} must be true or false, got {value!r}"
    )


def parse_optional_int(value: str, *, field: str, rule_id: str) -> int | None:
    if not value.strip():
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise CompanionBuildError(
            f"{rule_id}: {field} must be an integer, got {value!r}"
        ) from exc
    if parsed < 0:
        raise CompanionBuildError(f"{rule_id}: {field} must be non-negative")
    return parsed


def normalize_archive_path(value: str, *, context: str) -> str:
    raw = value.strip().replace("\\", "/")
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ":" in raw:
        raise CompanionBuildError(f"{context}: unsafe archive path {value!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise CompanionBuildError(f"{context}: unsafe archive path {value!r}")
    return path.as_posix()


def resolve_under(root: Path, relative: str, *, context: str) -> Path:
    normalized = normalize_archive_path(relative, context=context)
    candidate = (root / Path(*PurePosixPath(normalized).parts)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise CompanionBuildError(
            f"{context}: source escapes the configured root: {relative!r}"
        ) from exc
    return candidate


def read_rules(spec_path: Path) -> list[Rule]:
    if not spec_path.is_file():
        raise CompanionBuildError(f"asset specification not found: {spec_path}")
    with spec_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or [])
        missing = SPEC_COLUMNS - fields
        if missing:
            raise CompanionBuildError(
                f"asset specification is missing columns: {sorted(missing)}"
            )
        rules: list[Rule] = []
        seen: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            rule_id = (row.get("rule_id") or "").strip()
            if not rule_id or rule_id.startswith("#"):
                continue
            if rule_id in seen:
                raise CompanionBuildError(f"duplicate rule_id: {rule_id}")
            seen.add(rule_id)
            package = (row.get("package") or "").strip()
            mode = (row.get("mode") or "").strip()
            if package not in PACKAGES:
                raise CompanionBuildError(
                    f"{rule_id}: package must be one of {PACKAGES}, got {package!r}"
                )
            if mode not in {"file", "manifest_members"}:
                raise CompanionBuildError(
                    f"{rule_id}: unsupported mode {mode!r} on line {line_number}"
                )
            source = (row.get("source") or "").strip()
            destination = (row.get("destination") or "").strip()
            if not source or not destination:
                raise CompanionBuildError(
                    f"{rule_id}: source and destination are required"
                )
            expected_sha256 = (row.get("expected_sha256") or "").strip().lower()
            if expected_sha256 and not SHA256_RE.fullmatch(expected_sha256):
                raise CompanionBuildError(
                    f"{rule_id}: expected_sha256 is not a lowercase SHA-256"
                )
            root = (row.get("root") or "data").strip().lower()
            if root not in {"repository", "data"}:
                raise CompanionBuildError(
                    f"{rule_id}: root must be repository or data, got {root!r}"
                )
            rules.append(
                Rule(
                    rule_id=rule_id,
                    package=package,
                    mode=mode,
                    source=source,
                    destination=destination,
                    required=parse_bool(
                        row.get("required") or "",
                        field="required",
                        rule_id=rule_id,
                    ),
                    role=(row.get("role") or "asset").strip(),
                    member_base=(row.get("member_base") or "").strip(),
                    member_destination_prefix=(
                        row.get("member_destination_prefix") or ""
                    ).strip(),
                    path_column=(row.get("path_column") or "path").strip(),
                    bytes_column=(row.get("bytes_column") or "bytes").strip(),
                    sha256_column=(row.get("sha256_column") or "sha256").strip(),
                    expected_rows=parse_optional_int(
                        row.get("expected_rows") or "",
                        field="expected_rows",
                        rule_id=rule_id,
                    ),
                    expected_bytes=parse_optional_int(
                        row.get("expected_bytes") or "",
                        field="expected_bytes",
                        rule_id=rule_id,
                    ),
                    expected_sha256=expected_sha256 or None,
                    root=root,
                )
            )
    if not rules:
        raise CompanionBuildError("asset specification contains no active rules")
    return rules


def assert_expected_file(
    path: Path,
    *,
    expected_bytes: int | None,
    expected_sha256: str | None,
    context: str,
    verify_hash: bool,
) -> None:
    if not path.is_file():
        raise CompanionBuildError(f"{context}: required file is missing: {path}")
    actual_bytes = path.stat().st_size
    if expected_bytes is not None and actual_bytes != expected_bytes:
        raise CompanionBuildError(
            f"{context}: size mismatch for {path}: "
            f"expected {expected_bytes}, found {actual_bytes}"
        )
    if expected_sha256 is not None and verify_hash:
        actual_sha256 = sha256_path(path)
        if actual_sha256 != expected_sha256:
            raise CompanionBuildError(
                f"{context}: SHA-256 mismatch for {path}: "
                f"expected {expected_sha256}, found {actual_sha256}"
            )


def join_destination(prefix: str, member: str, *, context: str) -> str:
    member_path = normalize_archive_path(member, context=context)
    if not prefix.strip():
        return member_path
    prefix_path = normalize_archive_path(prefix, context=context)
    return normalize_archive_path(
        f"{prefix_path}/{member_path}",
        context=context,
    )


def expand_manifest_rule(
    rule: Rule,
    manifest_path: Path,
    rule_root: Path,
    roots: dict[str, Path],
    *,
    verify_hashes: bool,
) -> list[Asset]:
    if rule.member_base not in {
        "rule_root",
        "source_root",
        "repository_root",
        "data_root",
        "manifest_dir",
    }:
        raise CompanionBuildError(
            f"{rule.rule_id}: member_base must be rule_root, repository_root, "
            "data_root, or manifest_dir"
        )
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or [])
        needed = {rule.path_column, rule.bytes_column, rule.sha256_column}
        missing = needed - fields
        if missing:
            raise CompanionBuildError(
                f"{rule.rule_id}: nested manifest is missing columns "
                f"{sorted(missing)}"
            )
        rows = list(reader)
    if rule.expected_rows is not None and len(rows) != rule.expected_rows:
        raise CompanionBuildError(
            f"{rule.rule_id}: expected {rule.expected_rows} manifest rows, "
            f"found {len(rows)}"
        )
    if rule.member_base in {"rule_root", "source_root"}:
        base = rule_root
    elif rule.member_base == "repository_root":
        base = roots["repository"]
    elif rule.member_base == "data_root":
        base = roots["data"]
    else:
        base = manifest_path.parent
    assets: list[Asset] = []
    for row_number, row in enumerate(rows, start=2):
        member = (row.get(rule.path_column) or "").strip()
        if not member:
            raise CompanionBuildError(
                f"{rule.rule_id}: empty path on nested manifest row {row_number}"
            )
        member_context = f"{rule.rule_id} nested row {row_number}"
        source_path = resolve_under(base, member, context=member_context)
        expected_bytes = parse_optional_int(
            row.get(rule.bytes_column) or "",
            field=rule.bytes_column,
            rule_id=member_context,
        )
        expected_sha256 = (row.get(rule.sha256_column) or "").strip().lower()
        if not SHA256_RE.fullmatch(expected_sha256):
            raise CompanionBuildError(
                f"{member_context}: invalid nested SHA-256 {expected_sha256!r}"
            )
        assert_expected_file(
            source_path,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
            context=member_context,
            verify_hash=verify_hashes,
        )
        assets.append(
            Asset(
                package=rule.package,
                source=source_path,
                destination=join_destination(
                    rule.member_destination_prefix,
                    member,
                    context=member_context,
                ),
                rule_id=rule.rule_id,
                role=rule.role,
                expected_bytes=expected_bytes,
                expected_sha256=expected_sha256,
            )
        )
    return assets


def resolve_assets(
    rules: Iterable[Rule],
    roots: dict[str, Path],
    *,
    verify_hashes: bool,
) -> list[Asset]:
    resolved_roots = {
        name: path.resolve() for name, path in roots.items()
    }
    if set(resolved_roots) != {"repository", "data"}:
        raise CompanionBuildError(
            "exactly repository and data roots must be configured"
        )
    for name, root in resolved_roots.items():
        if not root.is_dir():
            raise CompanionBuildError(
                f"{name} root is not a directory: {root}"
            )
    assets: list[Asset] = []
    for rule in rules:
        rule_root = resolved_roots[rule.root]
        source = resolve_under(rule_root, rule.source, context=rule.rule_id)
        if not source.is_file():
            if rule.required:
                raise CompanionBuildError(
                    f"{rule.rule_id}: required source is missing: {source}"
                )
            continue
        destination = normalize_archive_path(
            rule.destination,
            context=rule.rule_id,
        )
        assert_expected_file(
            source,
            expected_bytes=rule.expected_bytes,
            expected_sha256=rule.expected_sha256,
            context=rule.rule_id,
            verify_hash=verify_hashes,
        )
        assets.append(
            Asset(
                package=rule.package,
                source=source,
                destination=destination,
                rule_id=rule.rule_id,
                role=(
                    f"{rule.role}_manifest"
                    if rule.mode == "manifest_members"
                    else rule.role
                ),
                expected_bytes=rule.expected_bytes,
                expected_sha256=rule.expected_sha256,
            )
        )
        if rule.mode == "manifest_members":
            assets.extend(
                expand_manifest_rule(
                    rule,
                    source,
                    rule_root,
                    resolved_roots,
                    verify_hashes=verify_hashes,
                )
            )
    destinations: dict[tuple[str, str], Asset] = {}
    for asset in assets:
        key = (asset.package, asset.destination)
        previous = destinations.get(key)
        if previous is not None:
            raise CompanionBuildError(
                f"archive destination collision in {asset.package}: "
                f"{asset.destination} ({previous.rule_id}, {asset.rule_id})"
            )
        destinations[key] = asset
    for package in PACKAGES:
        if not any(asset.package == package for asset in assets):
            raise CompanionBuildError(f"package {package!r} has no resolved assets")
    return assets


def validate_companion_contract(assets: Iterable[Asset]) -> None:
    destinations = {(asset.package, asset.destination) for asset in assets}
    required_core = {
        "results/ted_manuscript_machine_readable_v2/"
        "common_task_scenario_registry.tsv",
        "results/ted_manuscript_machine_readable_v2/common_task_status.json",
        "results/ted_manuscript_machine_readable_v2/common_task_truth_masked.tsv",
        "results/ted_manuscript_machine_readable_v2/"
        "method_harmonized_event_outputs.tsv",
        "results/ted_manuscript_machine_readable_v2/"
        "method_native_outputs/manifest.tsv",
        "results/ted_bnt162b2_flagship/final_evidence_v1/"
        "final_evidence_summary.json",
        "results/ted_bnt162b2_flagship/final_evidence_v1/"
        "independent_recalculation_audit.json",
        "results/ted_bib_companion_evidence_contract_v1/"
        "parallel_evidence_record_v1.json",
        "results/ted_bib_companion_evidence_contract_v1/"
        "replication_facets_v1.json",
        "results/ted_bib_companion_evidence_contract_v1/"
        "claim_boundary_v1.json",
        "results/ted_bib_companion_evidence_contract_v1/manifest.tsv",
        "schemas/ted_event_report_v2.schema.json",
        "schemas/parallel_evidence_record_v1.schema.json",
        "schemas/replication_facets_v1.schema.json",
        "scripts/build_bib_companion_evidence_contracts.py",
        "release/ted-v1.1.0/CLAIM_BOUNDARY.md",
        "reproduce/verify_and_reproduce_figures.py",
        "reproduce/render_bib_figures.py",
        "reproduction/FIGURE_RENDERERS.json",
        "requirements-reproduction-py311.txt",
        "results/ted_bib_focused_81/focused_81_collection.txt",
        "results/ted_bib_focused_81/focused_81_junit.xml",
        "results/ted_bib_focused_81/focused_81_terminal_summary.txt",
        "results/ted_bib_focused_81/focused_81_command.json",
    }
    figure3 = {
        "figure3_clean_common_task_metrics.tsv",
        "figure3_type_specific_clean_metrics.tsv",
        "figure3_low_signal_noisy_coordinate_metrics.tsv",
        "figure3_artifact_common_task_metrics.tsv",
    }
    figure5 = {
        "figure5_primary_rna_trajectory.tsv",
        "figure5_primary_protein_trajectory.tsv",
        "figure5_rna_protein_donor_contrasts.tsv",
        "figure5_gse171964_blind_qc.tsv",
        "figure5_flagship_design.tsv",
        "figure5_rna_gate_audit.tsv",
        "figure5_evidence_status.tsv",
    }
    required_core.update(
        f"results/ted_v1_submission/figure_source_data/{name}"
        for name in figure3 | figure5
    )
    missing = sorted(
        path for path in required_core if ("core", path) not in destinations
    )
    if missing:
        raise CompanionBuildError(
            "asset allowlist does not satisfy the companion contract; "
            f"missing core destinations: {missing}"
        )
    native_manifest = (
        "results/ted_manuscript_machine_readable_v2/"
        "method_native_outputs/manifest.tsv"
    )
    if ("native_outputs", native_manifest) not in destinations:
        raise CompanionBuildError(
            "native_outputs archive must include its nested 2,400-row manifest"
        )
    stability_manifest = (
        "results/ted_submission_supplement/"
        "zscape_repeated_holdout_stability/manifest.tsv"
    )
    if ("stability_shards", stability_manifest) not in destinations:
        raise CompanionBuildError(
            "stability_shards archive must include its nested manifest"
        )


def verify_repository_lock(
    repository_root: Path,
    analysis_lock_commit: str,
    spec_path: Path,
) -> str:
    commit = analysis_lock_commit.strip().lower()
    if not COMMIT_RE.fullmatch(commit):
        raise CompanionBuildError(
            "analysis-lock commit must be a full 40-character hexadecimal ID"
        )
    root = repository_root.resolve()

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise CompanionBuildError(
                f"repository lock check failed: git {' '.join(arguments)}: "
                f"{completed.stderr.strip()}"
            )
        return completed.stdout.strip()

    top_level = Path(git("rev-parse", "--show-toplevel")).resolve()
    if top_level != root:
        raise CompanionBuildError(
            f"repository root must be the Git top level: {top_level}"
        )
    head = git("rev-parse", "--verify", "HEAD^{commit}").lower()
    if head != commit:
        raise CompanionBuildError(
            f"analysis-lock commit {commit} does not equal repository HEAD {head}"
        )
    dirty = git("status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        first_lines = "\n".join(dirty.splitlines()[:10])
        raise CompanionBuildError(
            "repository root is not tracked-clean at the analysis lock:\n"
            f"{first_lines}"
        )
    resolved_spec = spec_path.resolve()
    try:
        relative_spec = resolved_spec.relative_to(root).as_posix()
    except ValueError as exc:
        raise CompanionBuildError(
            "asset specification must be inside the analysis-lock repository"
        ) from exc
    git("ls-files", "--error-unmatch", "--", relative_spec)
    return commit


def zip_info(path: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(path, date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def write_bytes_member(
    archive: zipfile.ZipFile,
    destination: str,
    payload: bytes,
    *,
    rule_id: str,
    role: str,
) -> WrittenAsset:
    archive.writestr(zip_info(destination), payload)
    return WrittenAsset(
        destination=destination,
        size=len(payload),
        sha256=sha256_bytes(payload),
        rule_id=rule_id,
        role=role,
    )


def write_file_member(
    archive: zipfile.ZipFile,
    asset: Asset,
) -> WrittenAsset:
    digest = hashlib.sha256()
    size = 0
    with asset.source.open("rb") as source_handle:
        with archive.open(zip_info(asset.destination), "w", force_zip64=True) as dest:
            for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
                dest.write(chunk)
                digest.update(chunk)
                size += len(chunk)
    actual_sha256 = digest.hexdigest()
    if asset.expected_bytes is not None and size != asset.expected_bytes:
        raise CompanionBuildError(
            f"{asset.rule_id}: source changed while packaging; size for "
            f"{asset.source} is {size}, expected {asset.expected_bytes}"
        )
    if (
        asset.expected_sha256 is not None
        and actual_sha256 != asset.expected_sha256
    ):
        raise CompanionBuildError(
            f"{asset.rule_id}: source changed or failed integrity; SHA-256 for "
            f"{asset.source} is {actual_sha256}, expected {asset.expected_sha256}"
        )
    return WrittenAsset(
        destination=asset.destination,
        size=size,
        sha256=actual_sha256,
        rule_id=asset.rule_id,
        role=asset.role,
    )


def package_manifest_bytes(entries: Iterable[WrittenAsset]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, delimiter="\t", lineterminator="\n")
    writer.writerow(["path", "bytes", "sha256", "source_rule", "role"])
    for entry in sorted(entries, key=lambda item: item.destination):
        writer.writerow(
            [
                entry.destination,
                entry.size,
                entry.sha256,
                entry.rule_id,
                entry.role,
            ]
        )
    return output.getvalue().encode("utf-8")


def figure_chain_bytes(assets: Iterable[Asset]) -> bytes:
    rows: list[tuple[str, str, str, str]] = []
    for asset in assets:
        if asset.package != "core":
            continue
        if asset.role in {
            "figure3_source",
            "figure3_output",
            "figure5_source",
            "figure5_output",
        }:
            figure_id, relation = asset.role.split("_", maxsplit=1)
            rows.append(
                (figure_id, relation, asset.destination, asset.rule_id)
            )
    output = io.StringIO(newline="")
    writer = csv.writer(output, delimiter="\t", lineterminator="\n")
    writer.writerow(["figure_id", "relation", "path", "source_rule"])
    writer.writerows(sorted(rows))
    return output.getvalue().encode("utf-8")


def package_metadata_bytes(
    package: str,
    assets: list[Asset],
    spec_sha256: str,
    analysis_lock_commit: str,
) -> bytes:
    payload = {
        "release": f"ted-v{RELEASE_VERSION}",
        "package": package,
        "asset_spec_sha256": spec_sha256,
        "analysis_lock_commit": analysis_lock_commit,
        "allowlisted_source_count": len(assets),
        "manifest_policy": (
            "PACKAGE_MANIFEST.tsv lists every payload member, including nested "
            "input manifests; it excludes itself to avoid a recursive hash."
        ),
        "compression": "stored",
    }
    return (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def archive_name(package: str) -> str:
    suffix = package.replace("_", "-")
    return f"ted-bib-companion-v{RELEASE_VERSION}-{suffix}.zip"


def build_archive(
    output_path: Path,
    package: str,
    assets: list[Asset],
    spec_sha256: str,
    analysis_lock_commit: str,
) -> None:
    written: list[WrittenAsset] = []
    with zipfile.ZipFile(
        output_path,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for asset in sorted(assets, key=lambda item: item.destination):
            written.append(write_file_member(archive, asset))
        if package == "core":
            chain = figure_chain_bytes(assets)
            written.append(
                write_bytes_member(
                    archive,
                    "reproduction/FIGURE_SOURCE_CHAIN.tsv",
                    chain,
                    rule_id="generated.figure_source_chain",
                    role="reproduction_chain",
                )
            )
        metadata = package_metadata_bytes(
            package,
            assets,
            spec_sha256,
            analysis_lock_commit,
        )
        written.append(
            write_bytes_member(
                archive,
                "PACKAGE_METADATA.json",
                metadata,
                rule_id="generated.package_metadata",
                role="package_metadata",
            )
        )
        manifest = package_manifest_bytes(written)
        archive.writestr(zip_info("PACKAGE_MANIFEST.tsv"), manifest)


def outer_manifest_bytes(entries: Iterable[tuple[str, int, str, str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, delimiter="\t", lineterminator="\n")
    writer.writerow(["file", "bytes", "sha256", "role"])
    for name, size, checksum, role in sorted(entries):
        writer.writerow([name, size, checksum, role])
    return output.getvalue().encode("utf-8")


def build_bundle(
    *,
    repository_root: Path,
    data_root: Path,
    analysis_lock_commit: str,
    output_dir: Path,
    spec_path: Path,
) -> Path:
    verified_commit = verify_repository_lock(
        repository_root,
        analysis_lock_commit,
        spec_path,
    )
    rules = read_rules(spec_path)
    assets = resolve_assets(
        rules,
        {"repository": repository_root, "data": data_root},
        verify_hashes=False,
    )
    validate_companion_contract(assets)
    if output_dir.exists():
        raise CompanionBuildError(
            f"output directory already exists; refusing to overwrite: {output_dir}"
        )
    output_parent = output_dir.resolve().parent
    output_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.building-",
            dir=output_parent,
        )
    )
    spec_sha256 = sha256_path(spec_path)
    try:
        archive_entries: list[tuple[str, int, str, str]] = []
        for package in PACKAGES:
            package_assets = [
                asset for asset in assets if asset.package == package
            ]
            name = archive_name(package)
            path = temporary / name
            build_archive(
                path,
                package,
                package_assets,
                spec_sha256,
                verified_commit,
            )
            archive_entries.append(
                (name, path.stat().st_size, sha256_path(path), package)
            )
        index = {
            "release": f"ted-v{RELEASE_VERSION}",
            "asset_spec_sha256": spec_sha256,
            "archives": [
                {"file": name, "package": role}
                for name, _, _, role in sorted(archive_entries)
            ],
            "status": "built_not_published",
            "analysis_lock_commit": verified_commit,
            "zenodo_version_doi": None,
        }
        index_path = temporary / "BUILD_INDEX.json"
        index_path.write_bytes(
            (json.dumps(index, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            )
        )
        archive_entries.append(
            (
                index_path.name,
                index_path.stat().st_size,
                sha256_path(index_path),
                "build_index",
            )
        )
        (temporary / "ARCHIVE_MANIFEST.tsv").write_bytes(
            outer_manifest_bytes(archive_entries)
        )
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir


def dry_run(
    *,
    repository_root: Path,
    data_root: Path,
    analysis_lock_commit: str,
    spec_path: Path,
) -> dict[str, object]:
    verified_commit = verify_repository_lock(
        repository_root,
        analysis_lock_commit,
        spec_path,
    )
    rules = read_rules(spec_path)
    assets = resolve_assets(
        rules,
        {"repository": repository_root, "data": data_root},
        verify_hashes=True,
    )
    validate_companion_contract(assets)
    packages: dict[str, dict[str, int]] = {}
    for package in PACKAGES:
        package_assets = [
            asset for asset in assets if asset.package == package
        ]
        packages[package] = {
            "files": len(package_assets),
            "bytes": sum(asset.source.stat().st_size for asset in package_assets),
        }
    return {
        "status": "dry_run_verified",
        "release": f"ted-v{RELEASE_VERSION}",
        "asset_spec": str(spec_path.resolve()),
        "asset_spec_sha256": sha256_path(spec_path),
        "analysis_lock_commit": verified_commit,
        "packages": packages,
        "writes_performed": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the TED BIB v1.1.0 companion from an explicit checksummed "
            "allowlist"
        )
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=DEFAULT_REPOSITORY_ROOT,
        help="clean analysis-lock checkout containing code, schemas, and evidence",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"G:\pyfgsea"),
        help="workspace containing ignored result and public-data artifacts",
    )
    parser.add_argument(
        "--asset-spec",
        type=Path,
        default=DEFAULT_SPEC,
        help="configurable TSV asset allowlist",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dist") / f"ted-bib-companion-v{RELEASE_VERSION}",
    )
    parser.add_argument(
        "--analysis-lock-commit",
        required=False,
        help=(
            "required 40-character commit; must equal tracked-clean "
            "--repository-root HEAD"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="verify every source size/SHA and print the plan without writing",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify an already-built --output-dir without reading sources",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run and args.verify_only:
        print(
            "ERROR: --dry-run and --verify-only are mutually exclusive",
            file=sys.stderr,
        )
        return 2
    if not args.verify_only and not args.analysis_lock_commit:
        print(
            "ERROR: --analysis-lock-commit is required for build and --dry-run",
            file=sys.stderr,
        )
        return 2
    try:
        if args.verify_only:
            from verify_ted_bib_companion import verify_bundle

            report = verify_bundle(args.output_dir)
        elif args.dry_run:
            report = dry_run(
                repository_root=args.repository_root,
                data_root=args.data_root,
                analysis_lock_commit=args.analysis_lock_commit,
                spec_path=args.asset_spec,
            )
        else:
            built = build_bundle(
                repository_root=args.repository_root,
                data_root=args.data_root,
                analysis_lock_commit=args.analysis_lock_commit,
                output_dir=args.output_dir,
                spec_path=args.asset_spec,
            )
            from verify_ted_bib_companion import verify_bundle

            report = verify_bundle(built)
        print(json.dumps(report, indent=2, sort_keys=True))
    except (CompanionBuildError, OSError, ValueError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
