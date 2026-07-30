from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "ted_gse171964_replication_v1.yaml"
FREEZE = ROOT / "results" / "ted_gse171964_replication" / "protocol_freeze_v1" / "protocol_freeze.json"
SOURCE_DIR = ROOT / "data_external" / "GSE171964_BNT162b2_replication" / "source"
OUT_DIR = SOURCE_DIR.parent


def digest(path: Path, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    if not FREEZE.is_file():
        raise SystemExit("Replication protocol freeze is missing")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if digest(CONFIG) != freeze["config_sha256"]:
        raise SystemExit("Replication config changed after its create-only freeze")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    rows: list[dict[str, object]] = []
    for role, spec in config["source"]["files"].items():
        path = SOURCE_DIR / spec["name"]
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_size = path.stat().st_size
        expected_size = int(spec["size_bytes"])
        if observed_size != expected_size:
            raise ValueError(f"Size mismatch for {path.name}: {observed_size} != {expected_size}")
        observed_sha256 = digest(path)
        expected_sha256 = spec.get("sha256")
        if expected_sha256 and observed_sha256 != expected_sha256:
            raise ValueError(f"SHA-256 mismatch for {path.name}")
        rows.append(
            {
                "dataset": "GSE171964",
                "corrected_release": config["source"]["corrected_release"],
                "role": role,
                "url": spec.get("url", config["source"]["accession_url"]),
                "local_path": path.relative_to(ROOT).as_posix(),
                "size_bytes": observed_size,
                "sha256": observed_sha256,
                "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUT_DIR / "download_manifest.tsv", sep="\t", index=False)
    (OUT_DIR / "download_manifest.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(f"Verified corrected GSE171964 v2 inputs: {len(rows)} files")


if __name__ == "__main__":
    main()
