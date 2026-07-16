"""Create-once closure for formal T21 discovery after outcome-blind preflight."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping

from .t21_timing_no_go import validate_timing_no_go_file


SCHEMA_NAME = "t21_formal_discovery_no_go_decision"
SCHEMA_VERSION = "1.0.0"
DECISION = "no_go"
EVIDENCE_VERSION = "v2"
REAL_PATHWAY_OUTCOMES_READ = False

ATLAS_RELATIVE_PATH = (
    "data_external/t21_data_product_v1/audit/terminal_amplitude_design_atlas_v2/"
    "t21_sampling_frame_design_atlas_v2.tsv"
)
DECISION_RELATIVE_PATH = (
    "data_external/t21_data_product_v1/audit/terminal_amplitude_design_atlas_v2/"
    "t21_terminal_amplitude_preflight_decision_v2.json"
)
MANIFEST_RELATIVE_PATH = (
    "data_external/t21_data_product_v1/audit/terminal_amplitude_design_atlas_v2/"
    "t21_sampling_frame_design_atlas_build_record_v2.json"
)
TIMING_NO_GO_RELATIVE_PATH = (
    "data_external/t21_data_product_v1/audit/T21_TIMING_NO_GO_2026-07-14.md"
)

ATLAS_SHA256 = "8e35b87ee393af8e1bf1f72e260f3d5a2c0d10a7e1fc7e5008a3bf19b7853064"
DECISION_SHA256 = "fe6bda1c2346bc3915160eed191f7d3f7fb0a4a1a675a8199d5e126eed7f8128"
MANIFEST_SHA256 = "b465228f4d1bfcdab6de8d51d3e57ddae2870d23daec5f8519d7d7302b33e487"
CONTRACT_PAYLOAD_SHA256 = (
    "fe206ece9529c41342065eccbfc01f0f87f9b9117153803b42e4f8d0923aa007"
)
TIMING_NO_GO_SHA256 = (
    "641aad73dde4e02bf8d03621bf78cf1abfc177ebf4823d6f35218b34f00a8306"
)

REOPEN_FORMULA = (
    "(new_independent_donors OR genuinely_new_full_trajectory_coverage_data) "
    "AND new_predeclared_design_contract"
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _load_exact_json(path: Path, expected_sha256: str, label: str) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing or not a regular file")
    if _sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} SHA256 differs from the frozen v2 evidence")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    return _require_mapping(value, label)


def validate_atlas_v2_closure_evidence(repository_root: str | Path) -> dict[str, Any]:
    """Verify the exact outcome-blind atlas v2 evidence that forces closure."""
    root = Path(repository_root).resolve()
    atlas_path = root / ATLAS_RELATIVE_PATH
    decision_path = root / DECISION_RELATIVE_PATH
    manifest_path = root / MANIFEST_RELATIVE_PATH
    timing_path = root / TIMING_NO_GO_RELATIVE_PATH

    if not atlas_path.is_file() or atlas_path.is_symlink():
        raise ValueError("T21 sampling-frame atlas v2 is missing or not a regular file")
    if _sha256_file(atlas_path) != ATLAS_SHA256:
        raise ValueError("T21 sampling-frame atlas v2 SHA256 differs")
    with atlas_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 4:
        raise ValueError("T21 sampling-frame atlas v2 must contain four frozen frames")
    if any(row.get("design_preflight_pass") != "False" for row in rows):
        raise ValueError("At least one T21 sampling frame now passes preflight")
    if any(row.get("selection_uses_pathway_outcomes") != "False" for row in rows):
        raise ValueError("T21 sampling-frame selection is not outcome-blind")
    if any(row.get("timing_authorized") != "False" for row in rows):
        raise ValueError("T21 timing was unexpectedly authorized")

    decision = _load_exact_json(decision_path, DECISION_SHA256, "atlas v2 decision")
    expected_decision = {
        "evidence_version": EVIDENCE_VERSION,
        "evidence_revision_only": True,
        "method_amendment_changed": False,
        "outcome_blind": True,
        "selection_uses_pathway_outcomes": False,
        "real_pathway_outcomes_read": False,
        "formal_discovery_allowed": False,
        "selected_estimand": None,
        "selected_terminal_frame_id": None,
        "smoke500_allowed": False,
        "smoke500_started": False,
        "screen2000_allowed": False,
        "final10000_allowed": False,
        "timing_authorized": False,
        "timing_decision": "no_go",
        "timing_no_go_validation": "PASS",
        "timing_no_go_sha256": TIMING_NO_GO_SHA256,
        "contract_payload_sha256": CONTRACT_PAYLOAD_SHA256,
    }
    if any(decision.get(key) != value for key, value in expected_decision.items()):
        raise ValueError("Atlas v2 decision no longer forces formal-discovery closure")
    if decision.get("reason_codes") != ["NO_FRAME_PASSED_PREFLIGHT"]:
        raise ValueError("Atlas v2 decision has another closure reason")

    manifest = _load_exact_json(
        manifest_path, MANIFEST_SHA256, "atlas v2 build-record manifest"
    )
    expected_manifest = {
        "schema_name": "t21_sampling_frame_design_atlas_build_record",
        "schema_version": "1.0.0",
        "evidence_version": EVIDENCE_VERSION,
        "evidence_revision_only": True,
        "method_amendment_changed": False,
        "atlas_sha256": ATLAS_SHA256,
        "decision_sha256": DECISION_SHA256,
        "contract_payload_sha256": CONTRACT_PAYLOAD_SHA256,
        "smoke500_allowed": False,
        "smoke500_started": False,
    }
    if any(manifest.get(key) != value for key, value in expected_manifest.items()):
        raise ValueError("Atlas v2 build-record manifest no longer binds closure evidence")

    timing_summary = validate_timing_no_go_file(timing_path)
    if timing_summary.get("sha256") != TIMING_NO_GO_SHA256:
        raise ValueError("Canonical timing no-go decision changed")
    return {
        "status": "pass_atlas_v2_forces_formal_discovery_no_go",
        "frames_evaluated": len(rows),
        "frames_passing_preflight": 0,
        "atlas_sha256": ATLAS_SHA256,
        "decision_sha256": DECISION_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "timing_no_go_sha256": TIMING_NO_GO_SHA256,
        "real_pathway_outcomes_read": False,
    }


def canonical_formal_discovery_no_go_markdown() -> str:
    """Return the only valid byte-canonical formal T21 discovery closure."""
    lines = [
        "# T21 Formal Discovery No-Go Decision",
        "",
        "## Decision",
        "",
        f"schema_name: {SCHEMA_NAME}",
        f"schema_version: {SCHEMA_VERSION}",
        "decision_date: 2026-07-14",
        f"decision: {DECISION}",
        "formal_discovery_decision: no_go",
        "formal_discovery_allowed: false",
        "no_sampling_frame_passed_preflight: true",
        "selected_estimand: null",
        "real_pathway_outcomes_read: false",
        "timing_decision: no_go",
        "timing_remains_no_go: true",
        "",
        "## Execution state",
        "",
        "smoke500_allowed: false",
        "smoke500_started: false",
        "screen2000_allowed: false",
        "screen2000_started: false",
        "final10000_allowed: false",
        "final10000_started: false",
        "",
        "## Frozen atlas v2 evidence",
        "",
        f"atlas_v2_path: {ATLAS_RELATIVE_PATH}",
        f"atlas_v2_sha256: {ATLAS_SHA256}",
        f"decision_v2_path: {DECISION_RELATIVE_PATH}",
        f"decision_v2_sha256: {DECISION_SHA256}",
        f"manifest_v2_path: {MANIFEST_RELATIVE_PATH}",
        f"manifest_v2_sha256: {MANIFEST_SHA256}",
        f"contract_payload_sha256: {CONTRACT_PAYLOAD_SHA256}",
        f"timing_no_go_path: {TIMING_NO_GO_RELATIVE_PATH}",
        f"timing_no_go_sha256: {TIMING_NO_GO_SHA256}",
        "frames_evaluated: 4",
        "frames_passing_preflight: 0",
        "closure_reason: NO_FRAME_PASSED_PREFLIGHT",
        "evidence_revision_only: true",
        "method_amendment_changed: false",
        "selection_uses_pathway_outcomes: false",
        "",
        "## Reopen policy",
        "",
        f"reopen_formula: {REOPEN_FORMULA}",
        "current_thresholds_may_be_post_hoc_relaxed: false",
        "current_estimands_may_be_post_hoc_relaxed: false",
        "",
        "Reopening requires new evidence and a new predeclared design contract.",
        "The current thresholds and estimands cannot be relaxed post hoc.",
    ]
    return "\n".join(lines) + "\n"


def canonical_formal_discovery_no_go_bytes() -> bytes:
    return canonical_formal_discovery_no_go_markdown().encode("utf-8")


def canonical_formal_discovery_no_go_sha256() -> str:
    return hashlib.sha256(canonical_formal_discovery_no_go_bytes()).hexdigest()


def formal_discovery_no_go_sidecar_path(path: str | Path) -> Path:
    target = Path(path)
    return target.with_name(target.name + ".sha256")


def _is_read_only(path: Path) -> bool:
    file_stat = path.stat()
    if os.name == "nt":
        return bool(file_stat.st_file_attributes & stat.FILE_ATTRIBUTE_READONLY)
    return not bool(file_stat.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _set_read_only(path: Path) -> None:
    if os.name == "nt":
        path.chmod(stat.S_IREAD)
    else:
        path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _set_writable_for_cleanup(path: Path) -> None:
    if path.exists():
        path.chmod(stat.S_IREAD | stat.S_IWRITE)


def _summary(
    *, path: Path | None = None, write_action: str | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "pass",
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "decision": DECISION,
        "formal_discovery_allowed": False,
        "selected_estimand": None,
        "real_pathway_outcomes_read": REAL_PATHWAY_OUTCOMES_READ,
        "timing_decision": "no_go",
        "smoke500_allowed": False,
        "smoke500_started": False,
        "screen2000_allowed": False,
        "screen2000_started": False,
        "final10000_allowed": False,
        "final10000_started": False,
        "frames_passing_preflight": 0,
        "atlas_sha256": ATLAS_SHA256,
        "decision_sha256": DECISION_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "reopen_formula": REOPEN_FORMULA,
        "sha256": canonical_formal_discovery_no_go_sha256(),
    }
    if path is not None:
        result["path"] = str(path)
        result["sidecar"] = str(formal_discovery_no_go_sidecar_path(path))
        result["read_only"] = True
    if write_action is not None:
        result["write_action"] = write_action
    return result


def validate_formal_discovery_no_go_markdown(value: str | bytes) -> dict[str, Any]:
    """Reject every semantic or byte-level variant of the canonical closure."""
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Formal discovery no-go must be canonical UTF-8") from exc
    elif isinstance(value, str):
        text = value
    else:
        raise TypeError("Formal discovery no-go must be str or bytes")
    if text != canonical_formal_discovery_no_go_markdown():
        raise ValueError("Formal discovery no-go is not byte-canonical")
    return _summary()


def _validate_sidecar(sidecar: Path, expected_digest: str) -> None:
    if not sidecar.is_file() or sidecar.is_symlink():
        raise ValueError("Formal discovery no-go SHA256 sidecar is missing")
    try:
        declared = sidecar.read_bytes().decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("Formal discovery no-go SHA256 sidecar must be ASCII") from exc
    if (
        declared != f"{expected_digest}\n"
        or not _SHA256_PATTERN.fullmatch(expected_digest)
    ):
        raise ValueError("Formal discovery no-go SHA256 sidecar is mismatched")


def validate_formal_discovery_no_go_file(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    """Validate evidence, canonical bytes, sidecar, and filesystem immutability."""
    validate_atlas_v2_closure_evidence(repository_root)
    target = Path(path)
    sidecar = formal_discovery_no_go_sidecar_path(target)
    if not target.is_file() or target.is_symlink():
        raise ValueError("Formal discovery no-go is missing or not a regular file")
    content = target.read_bytes()
    validate_formal_discovery_no_go_markdown(content)
    observed_digest = hashlib.sha256(content).hexdigest()
    if observed_digest != canonical_formal_discovery_no_go_sha256():
        raise ValueError("Formal discovery no-go SHA256 differs from canonical bytes")
    _validate_sidecar(sidecar, observed_digest)
    if not _is_read_only(target) or not _is_read_only(sidecar):
        raise ValueError("Formal discovery no-go and sidecar must remain read-only")
    return _summary(path=target.resolve())


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def create_or_validate_formal_discovery_no_go(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    """Create the canonical closure once; never overwrite or repair existing files."""
    validate_atlas_v2_closure_evidence(repository_root)
    target = Path(path)
    sidecar = formal_discovery_no_go_sidecar_path(target)
    if target.exists() or sidecar.exists():
        result = validate_formal_discovery_no_go_file(
            target, repository_root=repository_root
        )
        result["write_action"] = "validated_byte_identical_read_only_existing"
        return result

    target.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_formal_discovery_no_go_bytes()
    digest = hashlib.sha256(content).hexdigest()
    target_created = False
    sidecar_created = False
    try:
        _write_exclusive(target, content)
        target_created = True
        _write_exclusive(sidecar, f"{digest}\n".encode("ascii"))
        sidecar_created = True
        _set_read_only(target)
        _set_read_only(sidecar)
    except BaseException:
        for created, created_path in (
            (sidecar_created, sidecar),
            (target_created, target),
        ):
            if created:
                _set_writable_for_cleanup(created_path)
                created_path.unlink(missing_ok=True)
        raise
    result = validate_formal_discovery_no_go_file(
        target, repository_root=repository_root
    )
    result["write_action"] = "created"
    return result


def format_validation_summary(summary: Mapping[str, Any]) -> str:
    return json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True)
