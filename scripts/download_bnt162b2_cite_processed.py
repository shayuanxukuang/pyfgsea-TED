from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "ted_bnt162b2_flagship_v1.yaml"
DEFAULT_OUT = ROOT / "data_external" / "bnt162b2_cite_asap_2023"
FREEZE = ROOT / "results" / "ted_bnt162b2_flagship" / "protocol_freeze_v1" / "protocol_freeze.json"


def digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate(path: Path, expected_size: int, expected_md5: str) -> dict[str, object]:
    if path.stat().st_size != expected_size:
        raise ValueError(f"Size mismatch for {path}: {path.stat().st_size} != {expected_size}")
    observed_md5 = digest(path, "md5")
    if observed_md5 != expected_md5:
        raise ValueError(f"MD5 mismatch for {path}: {observed_md5} != {expected_md5}")
    return {
        "size_bytes": path.stat().st_size,
        "md5": observed_md5,
        "sha256": digest(path, "sha256"),
    }


def download_resumable(url: str, destination: Path, expected_size: int) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise ValueError(f"Partial file is larger than expected: {partial}")
    headers = {"User-Agent": "PyFgsea-TED/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=300) as response:
        if offset and response.status != 206:
            raise RuntimeError("Server did not honor Range; refusing to overwrite the partial download")
        mode = "ab" if offset else "wb"
        with partial.open(mode) as handle:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                if chunk:
                    handle.write(chunk)
    if partial.stat().st_size != expected_size:
        raise RuntimeError(f"Incomplete download: {partial.stat().st_size}/{expected_size} bytes")
    partial.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the frozen processed BNT162b2 CITE-seq object")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not FREEZE.is_file():
        raise SystemExit("Protocol freeze is missing; freeze before downloading expression data")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = config["source"]
    args.outdir.mkdir(parents=True, exist_ok=True)
    source_dir = args.outdir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    destination = source_dir / source["processed_file"]
    if not destination.exists():
        download_resumable(
            str(source["processed_file_url"]),
            destination,
            int(source["processed_file_size_bytes"]),
        )
    checks = validate(
        destination,
        int(source["processed_file_size_bytes"]),
        str(source["processed_file_md5"]),
    )
    row = {
        "dataset": source["study_id"],
        "record_doi": source["processed_record_doi"],
        "record_id": source["processed_record_id"],
        "url": source["processed_file_url"],
        "local_path": destination.relative_to(ROOT).as_posix(),
        **checks,
        "download_verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_accession_skipped": source["raw_accession"],
        "raw_skip_reason": "Open processed CITE object is sufficient for frozen RNA/ADT endpoints",
    }
    pd.DataFrame([row]).to_csv(args.outdir / "download_manifest.tsv", sep="\t", index=False)
    (args.outdir / "download_manifest.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(f"Verified processed CITE object: {destination}")


if __name__ == "__main__":
    main()
