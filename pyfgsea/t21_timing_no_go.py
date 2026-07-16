from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


SCHEMA_NAME = "t21_timing_no_go_decision"
SCHEMA_VERSION = "1.0.0"
DECISION = "no_go"
REAL_PATHWAY_OUTCOMES_READ = False

TIMING_SCOPE = (
    "onset",
    "duration",
    "phase_shift",
    "early_late",
    "transient_sustained",
    "heterochrony",
)

NO_GO_REASONS = (
    "maximum_supported_contiguous_bins_below_5",
    "supported_pseudotime_span_below_0.25",
    "timing_false_positive_rate_above_policy",
    "timing_power_not_evaluable",
)

PROHIBITED_RECOVERY_METHODS = (
    ("relax_to_3_or_4_bins", "放宽至3/4 bins"),
    ("rewrite_onset_definition", "改写onset定义"),
    ("merge_discontinuous_support_intervals", "合并不连续支持区间"),
    ("select_only_best_trajectory_draw", "只选最好trajectory draw"),
    (
        "use_posterior_smoothness_to_fill_unsupported_regions",
        "用posterior smoothness补不存在区域",
    ),
)

REOPEN_CONDITION = "加入新供体 OR 新的真正完整轨迹覆盖数据"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def canonical_timing_no_go_markdown() -> str:
    """Return the only valid byte-canonical T21 timing no-go decision."""

    lines = [
        "# T21 Timing No-Go Decision",
        "",
        "## Decision",
        "",
        f"schema_name: {SCHEMA_NAME}",
        f"schema_version: {SCHEMA_VERSION}",
        f"decision: {DECISION}",
        "real_pathway_outcomes_read: false",
        "",
        "## Scope",
        "",
        *(f"- {value}" for value in TIMING_SCOPE),
        "",
        "## Reasons",
        "",
        *(f"- {value}" for value in NO_GO_REASONS),
        "",
        "## Prohibited recovery methods",
        "",
        *(
            f"- {machine_name}: {original_wording}"
            for machine_name, original_wording in PROHIBITED_RECOVERY_METHODS
        ),
        "",
        "## Reopen condition",
        "",
        f"reopen_condition: {REOPEN_CONDITION}",
        "",
        "No other condition may reopen timing analysis.",
    ]
    return "\n".join(lines) + "\n"


def canonical_timing_no_go_bytes() -> bytes:
    return canonical_timing_no_go_markdown().encode("utf-8")


def canonical_timing_no_go_sha256() -> str:
    return hashlib.sha256(canonical_timing_no_go_bytes()).hexdigest()


def timing_no_go_sidecar_path(path: str | Path) -> Path:
    target = Path(path)
    return target.with_name(target.name + ".sha256")


def _summary(*, path: Path | None = None, write_action: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "pass",
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "decision": DECISION,
        "real_pathway_outcomes_read": REAL_PATHWAY_OUTCOMES_READ,
        "scope": list(TIMING_SCOPE),
        "reasons": list(NO_GO_REASONS),
        "prohibited_recovery_methods": [
            machine_name for machine_name, _ in PROHIBITED_RECOVERY_METHODS
        ],
        "reopen_condition": REOPEN_CONDITION,
        "sha256": canonical_timing_no_go_sha256(),
    }
    if path is not None:
        result["path"] = str(path)
        result["sidecar"] = str(timing_no_go_sidecar_path(path))
    if write_action is not None:
        result["write_action"] = write_action
    return result


def validate_timing_no_go_markdown(value: str | bytes) -> dict[str, Any]:
    """Reject every semantic or byte-level variant of the canonical decision."""

    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Timing no-go decision must be canonical UTF-8") from exc
    elif isinstance(value, str):
        text = value
    else:
        raise TypeError("Timing no-go decision must be str or bytes")
    expected = canonical_timing_no_go_markdown()
    if text != expected:
        raise ValueError("Timing no-go decision is not byte-canonical")
    return _summary()


def _validate_sidecar(sidecar: Path, expected_digest: str) -> None:
    if not sidecar.is_file() or sidecar.is_symlink():
        raise ValueError("Timing no-go SHA256 sidecar is missing or not a regular file")
    try:
        declared_bytes = sidecar.read_bytes()
        declared = declared_bytes.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("Timing no-go SHA256 sidecar must be ASCII") from exc
    if declared != f"{expected_digest}\n" or not _SHA256_PATTERN.fullmatch(
        expected_digest
    ):
        raise ValueError("Timing no-go SHA256 sidecar is malformed or mismatched")


def validate_timing_no_go_file(path: str | Path) -> dict[str, Any]:
    """Validate canonical bytes and the strict external SHA256 sidecar."""

    target = Path(path)
    if not target.is_file() or target.is_symlink():
        raise ValueError("Timing no-go decision is missing or not a regular file")
    content = target.read_bytes()
    validate_timing_no_go_markdown(content)
    observed_digest = hashlib.sha256(content).hexdigest()
    expected_digest = canonical_timing_no_go_sha256()
    if observed_digest != expected_digest:
        raise ValueError("Timing no-go decision SHA256 differs from canonical bytes")
    _validate_sidecar(timing_no_go_sidecar_path(target), observed_digest)
    return _summary(path=target.resolve())


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def create_or_validate_timing_no_go(path: str | Path) -> dict[str, Any]:
    """Create the canonical decision once; existing files are read-only inputs."""

    target = Path(path)
    sidecar = timing_no_go_sidecar_path(target)
    if target.exists() or sidecar.exists():
        result = validate_timing_no_go_file(target)
        result["write_action"] = "validated_byte_identical_existing"
        return result

    target.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_timing_no_go_bytes()
    digest = hashlib.sha256(content).hexdigest()
    target_created = False
    sidecar_created = False
    try:
        _write_exclusive(target, content)
        target_created = True
        _write_exclusive(sidecar, f"{digest}\n".encode("ascii"))
        sidecar_created = True
    except BaseException:
        if sidecar_created:
            sidecar.unlink(missing_ok=True)
        if target_created:
            target.unlink(missing_ok=True)
        raise
    result = validate_timing_no_go_file(target)
    result["write_action"] = "created"
    return result


def format_validation_summary(summary: dict[str, Any]) -> str:
    """Serialize a CLI summary without changing the canonical decision bytes."""

    return json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True)
