"""Build compact, deterministic public-data fixtures for release CI."""

from __future__ import annotations

import hashlib
import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "external_data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_fixture(source: Path, destination: Path, *, nrows: int | None = None) -> dict[str, object]:
    frame = pd.read_csv(source, sep="\t", nrows=nrows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, sep="\t", index=False, lineterminator="\n")
    return {
        "source_path": source.as_posix(),
        "source_bytes": source.stat().st_size,
        "source_sha256": sha256(source),
        "fixture_path": destination.relative_to(ROOT).as_posix(),
        "fixture_bytes": destination.stat().st_size,
        "fixture_sha256": sha256(destination),
        "fixture_rows": len(frame),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=ROOT)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    processed = source_root / "data" / "processed" / "ted_known_source"
    specs = [
        (
            processed / "SCP1064" / "ted_inputs" / "ted_protein_outcome_table.tsv.gz",
            OUT / "scp1064_protein_outcome_100.tsv",
            100,
        ),
        (
            processed / "SCP1064" / "ted_inputs" / "ted_cell_metadata.tsv",
            OUT / "scp1064_cell_metadata_100.tsv",
            100,
        ),
        (
            processed / "GSE153056" / "cell_metadata.tsv.gz",
            OUT / "gse153056_cell_metadata_100.tsv",
            100,
        ),
        (
            processed / "GSE153056" / "qc_summary.tsv",
            OUT / "gse153056_qc_summary.tsv",
            None,
        ),
        (
            processed / "GSE153056" / "processing_manifest.tsv",
            OUT / "gse153056_processing_manifest.tsv",
            None,
        ),
        (
            processed / "GSE93735" / "sample_metadata.tsv",
            OUT / "gse93735_sample_metadata.tsv",
            None,
        ),
        (
            processed / "GSE93735" / "qc_summary.tsv",
            OUT / "gse93735_qc_summary.tsv",
            None,
        ),
        (
            processed / "GSE90546" / "qc_summary.tsv",
            OUT / "gse90546_qc_summary.tsv",
            None,
        ),
    ]
    rows = []
    for source, destination, nrows in specs:
        row = write_fixture(source, destination, nrows=nrows)
        row["source_path"] = source.relative_to(source_root).as_posix()
        rows.append(row)
    manifest = pd.DataFrame(rows).sort_values("fixture_path")
    manifest.to_csv(OUT / "fixture_manifest.tsv", sep="\t", index=False, lineterminator="\n")
    print(f"Wrote {len(rows)} fixtures to {OUT}")


if __name__ == "__main__":
    main()
