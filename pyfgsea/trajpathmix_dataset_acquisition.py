from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, BinaryIO, Callable, Mapping
from urllib.request import Request, urlopen

from pyfgsea.trajpathmix_dataset_portfolio import (
    FROZEN_CONFIG_PAYLOAD_SHA256,
    load_dataset_portfolio_config,
)


RECEIPT_FILE = "acquisition_receipt_v1.json"
CHUNK_SIZE = 8 * 1024 * 1024


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _find_dataset(config: Mapping[str, Any], candidate_id: str) -> Mapping[str, Any]:
    for dataset in config["datasets"]:
        if dataset["candidate_id"] == candidate_id:
            return dataset
    raise ValueError(
        f"Candidate is not present in the frozen portfolio: {candidate_id}"
    )


def authorized_acquisition_plan(
    *, config_path: str | Path, candidate_id: str
) -> dict[str, Any]:
    """Return only explicitly authorized files from the frozen portfolio."""

    config = load_dataset_portfolio_config(config_path)
    dataset = _find_dataset(config, candidate_id)
    authorized_ids = tuple(dataset["acquisition"].get("authorized_file_ids", []))
    files_by_id: dict[str, dict[str, Any]] = {}
    for source in dataset["sources"]:
        for file_info in source.get("files", []):
            file_id = str(file_info["file_id"])
            files_by_id[file_id] = {
                **file_info,
                "source_id": source["source_id"],
                "source_url": source["source_url"],
                "accession": source["accession"],
            }
    missing = set(authorized_ids) - set(files_by_id)
    if missing:
        raise ValueError(
            f"Authorized file IDs are missing from sources: {sorted(missing)}"
        )
    files = [files_by_id[file_id] for file_id in authorized_ids]
    if not files:
        raise ValueError(f"No file downloads are authorized for {candidate_id}")
    return {
        "schema_name": "trajpathmix_authorized_acquisition_plan",
        "schema_version": "1.0.0",
        "portfolio_id": config["portfolio_id"],
        "config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
        "candidate_id": candidate_id,
        "acquisition_decision": dataset["acquisition"]["decision"],
        "authorized_file_ids": list(authorized_ids),
        "files": files,
        "pathway_scoring_authorized": False,
        "pathway_outcomes_read": False,
        "expression_values_read": False,
        "archives_opened": False,
    }


def _verify_download(path: Path, file_info: Mapping[str, Any]) -> dict[str, Any]:
    expected_size = int(file_info["size_bytes"])
    observed_size = path.stat().st_size
    if observed_size != expected_size:
        raise ValueError(
            f"Size mismatch for {path.name}: expected {expected_size}, "
            f"observed {observed_size}"
        )
    algorithm = str(file_info["publisher_checksum_algorithm"])
    expected_checksum = str(file_info["publisher_checksum"])
    if algorithm == "not_reported":
        publisher_checksum_status = "not_reported_by_publisher"
        observed_publisher_checksum = "not_applicable"
    else:
        observed_publisher_checksum = _hash_file(path, algorithm)
        if observed_publisher_checksum != expected_checksum:
            raise ValueError(
                f"Publisher checksum mismatch for {path.name}: expected "
                f"{expected_checksum}, observed {observed_publisher_checksum}"
            )
        publisher_checksum_status = "verified"
    local_sha256 = _hash_file(path, "sha256")
    source_audit_sha256 = str(file_info.get("source_audit_sha256", "not_frozen"))
    if source_audit_sha256 == "not_frozen":
        source_audit_sha256_status = "not_frozen"
    else:
        if local_sha256 != source_audit_sha256:
            raise ValueError(
                f"Frozen source-audit SHA256 mismatch for {path.name}: expected "
                f"{source_audit_sha256}, observed {local_sha256}"
            )
        source_audit_sha256_status = "verified"
    return {
        "file_id": file_info["file_id"],
        "file_name": file_info["file_name"],
        "file_url": file_info["file_url"],
        "source_id": file_info["source_id"],
        "accession": file_info["accession"],
        "local_relative_path": f"source/{file_info['file_name']}",
        "size_bytes": observed_size,
        "publisher_checksum_algorithm": algorithm,
        "publisher_checksum_expected": expected_checksum,
        "publisher_checksum_observed": observed_publisher_checksum,
        "publisher_checksum_status": publisher_checksum_status,
        "source_audit_sha256_expected": source_audit_sha256,
        "source_audit_sha256_status": source_audit_sha256_status,
        "local_sha256": local_sha256,
        "content_contract": file_info["content_contract"],
        "expression_values_read": False,
        "archive_opened": False,
    }


def _open_remote(url: str, offset: int) -> BinaryIO:
    headers = {"User-Agent": "PyFgsea-TrajPathMix/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    response = urlopen(Request(url, headers=headers), timeout=120)
    status = getattr(response, "status", response.getcode())
    if offset and status != 206:
        response.close()
        raise RuntimeError(
            f"Remote server did not honor the resume Range request for {url}"
        )
    return response


def _download_one(
    file_info: Mapping[str, Any],
    *,
    source_dir: Path,
    opener: Callable[[str, int], BinaryIO] = _open_remote,
) -> dict[str, Any]:
    final_path = source_dir / str(file_info["file_name"])
    part_path = final_path.with_name(final_path.name + ".part")
    aria2_state_path = part_path.with_name(part_path.name + ".aria2")
    if final_path.exists():
        return _verify_download(final_path, file_info)
    if aria2_state_path.exists():
        raise RuntimeError(
            f"An aria2 segmented state exists for {part_path.name}; resume with "
            "the aria2 transport instead of the single-stream transport"
        )

    expected_size = int(file_info["size_bytes"])
    offset = part_path.stat().st_size if part_path.exists() else 0
    if offset > expected_size:
        raise ValueError(f"Partial file exceeds expected size: {part_path}")
    if offset < expected_size:
        with opener(str(file_info["file_url"]), offset) as response:
            with part_path.open("ab" if offset else "wb") as output:
                while chunk := response.read(CHUNK_SIZE):
                    output.write(chunk)
                    output.flush()
    if part_path.stat().st_size != expected_size:
        raise ValueError(
            f"Incomplete download for {part_path.name}: expected {expected_size}, "
            f"observed {part_path.stat().st_size}"
        )
    verification = _verify_download(part_path, file_info)
    os.replace(part_path, final_path)
    return verification


def _download_one_aria2(
    file_info: Mapping[str, Any],
    *,
    source_dir: Path,
    max_connections: int,
) -> dict[str, Any]:
    final_path = source_dir / str(file_info["file_name"])
    part_path = final_path.with_name(final_path.name + ".part")
    if final_path.exists():
        return _verify_download(final_path, file_info)
    executable = shutil.which("aria2c") or shutil.which("aria2c.exe")
    if executable is None:
        raise RuntimeError("aria2c is required for the aria2 acquisition transport")
    if not 1 <= max_connections <= 16:
        raise ValueError("aria2 max_connections must be between 1 and 16")
    command = [
        executable,
        "--continue=true",
        f"--max-connection-per-server={max_connections}",
        f"--split={max_connections}",
        "--min-split-size=1M",
        "--file-allocation=none",
        "--summary-interval=30",
        "--console-log-level=notice",
        f"--dir={source_dir}",
        f"--out={part_path.name}",
        str(file_info["file_url"]),
    ]
    subprocess.run(command, check=True)
    verification = _verify_download(part_path, file_info)
    os.replace(part_path, final_path)
    return verification


def validate_acquisition_receipt(
    *, config_path: str | Path, candidate_id: str, output_dir: str | Path
) -> dict[str, Any]:
    plan = authorized_acquisition_plan(
        config_path=config_path, candidate_id=candidate_id
    )
    output = Path(output_dir).resolve()
    receipt_path = output / RECEIPT_FILE
    if not receipt_path.is_file():
        raise FileNotFoundError(f"Acquisition receipt is missing: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for key, expected in {
        "schema_name": "trajpathmix_acquisition_receipt",
        "schema_version": "1.0.0",
        "portfolio_id": plan["portfolio_id"],
        "config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
        "candidate_id": candidate_id,
        "authorized_file_ids": plan["authorized_file_ids"],
        "pathway_scoring_authorized": False,
        "pathway_outcomes_read": False,
        "expression_values_read": False,
        "archives_opened": False,
    }.items():
        if receipt.get(key) != expected:
            raise ValueError(f"Acquisition receipt mismatch for {key}")

    expected_files = {file_info["file_id"]: file_info for file_info in plan["files"]}
    observed_ids = [item["file_id"] for item in receipt.get("files", [])]
    if observed_ids != plan["authorized_file_ids"]:
        raise ValueError("Acquisition receipt file order or set mismatch")
    recomputed: list[dict[str, Any]] = []
    for item in receipt["files"]:
        file_info = expected_files[item["file_id"]]
        local_path = output / str(item["local_relative_path"])
        verification = _verify_download(local_path, file_info)
        if verification != item:
            raise ValueError(
                f"Acquisition receipt evidence mismatch: {item['file_id']}"
            )
        recomputed.append(verification)
    result = dict(receipt)
    result["validation_status"] = "pass_authorized_files_byte_verified"
    result["receipt_sha256"] = _hash_file(receipt_path, "sha256")
    return result


def execute_authorized_acquisition(
    *,
    config_path: str | Path,
    candidate_id: str,
    output_dir: str | Path,
    transport: str = "stdlib",
    max_connections: int = 16,
) -> dict[str, Any]:
    """Download bytes only; never open an archive or inspect expression values."""

    plan = authorized_acquisition_plan(
        config_path=config_path, candidate_id=candidate_id
    )
    output = Path(output_dir).resolve()
    source_dir = output / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output / RECEIPT_FILE
    if receipt_path.exists():
        return validate_acquisition_receipt(
            config_path=config_path,
            candidate_id=candidate_id,
            output_dir=output,
        )

    allowed_names = {str(item["file_name"]) for item in plan["files"]}
    allowed_names.update(
        str(item["file_name"]) + suffix
        for item in plan["files"]
        for suffix in (".part", ".part.aria2")
    )
    unexpected = {
        path.name
        for path in source_dir.iterdir()
        if path.is_file() and path.name not in allowed_names
    }
    if unexpected:
        raise ValueError(
            f"Unexpected files in acquisition source directory: {sorted(unexpected)}"
        )

    if transport == "stdlib":
        files = [
            _download_one(file_info, source_dir=source_dir)
            for file_info in plan["files"]
        ]
    elif transport == "aria2":
        files = [
            _download_one_aria2(
                file_info,
                source_dir=source_dir,
                max_connections=max_connections,
            )
            for file_info in plan["files"]
        ]
    else:
        raise ValueError("transport must be 'stdlib' or 'aria2'")
    receipt = {
        "schema_name": "trajpathmix_acquisition_receipt",
        "schema_version": "1.0.0",
        "portfolio_id": plan["portfolio_id"],
        "config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
        "candidate_id": candidate_id,
        "acquisition_decision": plan["acquisition_decision"],
        "created_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "authorized_file_ids": plan["authorized_file_ids"],
        "transfer_transport": transport,
        "files": files,
        "total_downloaded_bytes": sum(item["size_bytes"] for item in files),
        "pathway_scoring_authorized": False,
        "pathway_outcomes_read": False,
        "expression_values_read": False,
        "archives_opened": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{RECEIPT_FILE}.", suffix=".tmp", dir=output
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
        os.replace(temporary_name, receipt_path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return validate_acquisition_receipt(
        config_path=config_path,
        candidate_id=candidate_id,
        output_dir=output,
    )


__all__ = [
    "RECEIPT_FILE",
    "authorized_acquisition_plan",
    "execute_authorized_acquisition",
    "validate_acquisition_receipt",
]
