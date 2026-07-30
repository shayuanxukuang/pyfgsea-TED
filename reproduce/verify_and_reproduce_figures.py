#!/usr/bin/env python3
"""Verify the companion and audit or redraw the Figure 3/5 chain.

Without ``--redraw`` this entry point verifies that the packaged source tables,
final PDF/PNG files, nested manifests, and their SHA-256 values form a complete
Figure 3/5 evidence chain.  That is an integrity check, not a claim that the
figures were recomputed.

With ``--redraw`` it runs only a packaged
``reproduction/FIGURE_RENDERERS.json`` contract.  If that contract or its
scripts are absent, the command returns an explicit ``not_available`` status
and a non-zero exit code rather than reporting success.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from verify_ted_bib_companion import (  # noqa: E402
    CompanionVerificationError,
    archive_name,
    verify_bundle,
)


class ReproductionError(RuntimeError):
    """The packaged figure-chain or renderer contract is invalid."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_member(value: str) -> str:
    raw = value.strip().replace("\\", "/")
    path = PurePosixPath(raw)
    if (
        not raw
        or path.is_absolute()
        or ":" in raw
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ReproductionError(f"unsafe archive member: {value!r}")
    return path.as_posix()


def read_chain(archive: zipfile.ZipFile) -> list[dict[str, str]]:
    path = "reproduction/FIGURE_SOURCE_CHAIN.tsv"
    try:
        payload = archive.read(path).decode("utf-8-sig")
    except (KeyError, UnicodeDecodeError) as exc:
        raise ReproductionError(f"missing or invalid {path}") from exc
    reader = csv.DictReader(io.StringIO(payload), delimiter="\t")
    rows = list(reader)
    if not rows:
        raise ReproductionError(f"{path} contains no rows")
    return rows


def extract_members(
    archive: zipfile.ZipFile,
    members: set[str],
    destination: Path,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    for member in sorted(members):
        normalized = safe_member(member)
        target = (
            root / Path(*PurePosixPath(normalized).parts)
        ).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ReproductionError(
                f"archive member escapes extraction root: {member}"
            ) from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(normalized, "r") as source:
            with target.open("wb") as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(chunk)


def renderer_status(
    archive: zipfile.ZipFile,
) -> tuple[str, dict[str, object] | None]:
    path = "reproduction/FIGURE_RENDERERS.json"
    try:
        payload = archive.read(path)
    except KeyError:
        return "not_available_renderer_contract_not_packaged", None
    try:
        contract = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReproductionError(f"invalid {path}") from exc
    if not isinstance(contract, dict):
        raise ReproductionError(f"{path} must contain a JSON object")
    return "available", contract


def validate_command(command: object, *, figure_id: str) -> list[str]:
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(item, str) and item for item in command)
    ):
        raise ReproductionError(
            f"{figure_id}: renderer command must be a non-empty string list"
        )
    if command[0] in {"python", "python3"}:
        command = [sys.executable, *command[1:]]
    return list(command)


def redraw(
    archive: zipfile.ZipFile,
    contract: dict[str, object],
    workdir: Path,
) -> dict[str, object]:
    extract_members(archive, set(archive.namelist()), workdir)
    results: dict[str, object] = {}
    for figure_id in ("figure3", "figure5"):
        raw = contract.get(figure_id)
        if not isinstance(raw, dict):
            raise ReproductionError(
                f"renderer contract lacks object for {figure_id}"
            )
        command = validate_command(raw.get("command"), figure_id=figure_id)
        raw_outputs = raw.get("outputs")
        if not isinstance(raw_outputs, list) or not raw_outputs:
            raise ReproductionError(
                f"{figure_id}: renderer contract has no outputs"
            )
        completed = subprocess.run(
            command,
            cwd=workdir,
            check=False,
            text=True,
            capture_output=True,
            timeout=3600,
        )
        if completed.returncode != 0:
            raise ReproductionError(
                f"{figure_id}: renderer failed with exit "
                f"{completed.returncode}\n{completed.stderr[-4000:]}"
            )
        semantic_rel = safe_member(str(raw.get("semantic_report", "")))
        semantic_path = workdir / Path(*PurePosixPath(semantic_rel).parts)
        if not semantic_path.is_file():
            raise ReproductionError(
                f"{figure_id}: semantic renderer report is missing"
            )
        try:
            semantic_report = json.loads(
                semantic_path.read_text(encoding="utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReproductionError(
                f"{figure_id}: semantic renderer report is invalid"
            ) from exc
        if not isinstance(semantic_report, dict):
            raise ReproductionError(
                f"{figure_id}: semantic renderer report is not an object"
            )
        checks = semantic_report.get("checks")
        expected_sources = 4 if figure_id == "figure3" else 7
        pixel_qa = semantic_report.get("pixel_qa")
        if (
            semantic_report.get("status") != "passed"
            or semantic_report.get("source_count") != expected_sources
            or not isinstance(checks, list)
            or not checks
            or not all(
                isinstance(check, dict) and check.get("passed") is True
                for check in checks
            )
            or not isinstance(pixel_qa, dict)
            or pixel_qa.get("passed") is not True
        ):
            raise ReproductionError(
                f"{figure_id}: semantic or pixel QA did not pass"
            )
        output_results: list[dict[str, object]] = []
        for item in raw_outputs:
            if not isinstance(item, dict):
                raise ReproductionError(
                    f"{figure_id}: output contract entry must be an object"
                )
            generated_rel = safe_member(str(item.get("generated", "")))
            reference_rel = safe_member(str(item.get("reference", "")))
            generated = workdir / Path(
                *PurePosixPath(generated_rel).parts
            )
            reference = workdir / Path(
                *PurePosixPath(reference_rel).parts
            )
            if not generated.is_file() or not reference.is_file():
                raise ReproductionError(
                    f"{figure_id}: generated/reference output is missing"
                )
            validation = item.get("validation")
            if validation not in {
                "semantic_and_nonblank_pixel_qa",
                "semantic_and_nonempty_pdf",
            }:
                raise ReproductionError(
                    f"{figure_id}: unsupported output validation mode "
                    f"{validation!r}"
                )
            if generated.stat().st_size == 0:
                raise ReproductionError(
                    f"{figure_id}: generated output is empty: {generated_rel}"
                )
            generated_sha = sha256_path(generated)
            reference_sha = sha256_path(reference)
            output_results.append(
                {
                    "generated": generated_rel,
                    "reference": reference_rel,
                    "generated_sha256": generated_sha,
                    "reference_sha256": reference_sha,
                    "validation": validation,
                    "status": "semantically_verified",
                    "byte_identity_required": False,
                }
            )
        results[figure_id] = {
            "status": "redrawn_and_semantically_verified",
            "command": command,
            "outputs": output_results,
            "semantic_report": semantic_rel,
            "semantic_checks": len(checks),
            "pixel_qa": pixel_qa,
            "scope": semantic_report.get("scope"),
        }
    return results


def run(
    bundle_dir: Path,
    *,
    redraw_requested: bool,
    extract_dir: Path | None,
) -> tuple[dict[str, object], int]:
    package_report = verify_bundle(bundle_dir)
    core_archive = bundle_dir / archive_name("core")
    with zipfile.ZipFile(core_archive, "r") as archive:
        chain = read_chain(archive)
        chain_members = {
            safe_member(row["path"])
            for row in chain
            if row.get("figure_id") in {"figure3", "figure5"}
        }
        if extract_dir is not None:
            if extract_dir.exists() and any(extract_dir.iterdir()):
                raise ReproductionError(
                    f"extraction directory is not empty: {extract_dir}"
                )
            extract_members(archive, chain_members, extract_dir)
        status, contract = renderer_status(archive)
        report: dict[str, object] = {
            "status": "verified",
            "bundle_verification": package_report,
            "source_to_figure_chain_status": (
                "verified_packaged_sources_and_outputs"
            ),
            "source_to_figure_chain_scope": (
                "integrity/provenance only; no redraw was inferred"
            ),
            "chain_members": len(chain_members),
            "redraw_status": "not_requested",
        }
        if redraw_requested:
            if contract is None:
                report["status"] = "incomplete"
                report["redraw_status"] = status
                report["redraw_detail"] = (
                    "No redraw command was executed. Package or allowlist the "
                    "exact renderer scripts and FIGURE_RENDERERS.json first."
                )
                return report, 3
            with tempfile.TemporaryDirectory(
                prefix="ted-bib-figure-redraw-"
            ) as temporary:
                workdir = Path(temporary)
                report["redraw_results"] = redraw(
                    archive,
                    contract,
                    workdir,
                )
            report["redraw_status"] = "redrawn_and_semantically_verified"
        return report, 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the built TED companion and audit or redraw the exact "
            "Figure 3/5 source chain"
        )
    )
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument(
        "--redraw",
        action="store_true",
        help=(
            "run only a packaged FIGURE_RENDERERS.json contract; return an "
            "explicit non-zero not_available status when absent"
        ),
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        help="optionally extract only Figure 3/5 source and output members",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="optional JSON report destination",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report, return_code = run(
            args.bundle_dir.resolve(),
            redraw_requested=args.redraw,
            extract_dir=(
                args.extract_dir.resolve()
                if args.extract_dir is not None
                else None
            ),
        )
    except (
        CompanionVerificationError,
        ReproductionError,
        OSError,
        ValueError,
        csv.Error,
        zipfile.BadZipFile,
        subprocess.SubprocessError,
    ) as exc:
        report = {"status": "failed", "error": str(exc)}
        return_code = 1
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
