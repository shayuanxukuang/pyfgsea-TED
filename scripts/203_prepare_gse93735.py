from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data_external" / "matched_validation_positive_stratum" / "gse93735_lps_dex_late"
DEFAULT_OUT = ROOT / "data" / "processed" / "ted_known_source" / "GSE93735"


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def read_fpkm(path: Path) -> pd.Series:
    df = pd.read_csv(path, sep="\t", compression="gzip", usecols=["gene_short_name", "FPKM"])
    values = pd.to_numeric(df["FPKM"], errors="coerce").fillna(0.0)
    series = pd.Series(values.to_numpy(dtype=float), index=df["gene_short_name"].astype(str), name=path.stem)
    return series.groupby(level=0).mean()


def build_matrix(meta: pd.DataFrame) -> pd.DataFrame:
    series = []
    for _, row in meta.iterrows():
        path = ROOT / str(row["file"])
        s = read_fpkm(path)
        s.name = str(row["sample"])
        series.append(s)
    matrix = pd.concat(series, axis=1).fillna(0.0)
    matrix.index.name = "gene_symbol"
    return np.log2(matrix + 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(args.input / "gse93735_sample_metadata.tsv", sep="\t")
    condition_map = {
        "baseline_0h": "Control",
        "LPS_10h": "LPS",
        "DexLate_LPS_10h": "Dex_LPS",
    }
    meta["condition"] = meta["group"].map(condition_map).fillna(meta["group"])
    meta["timepoint"] = meta["group"].astype(str).str.extract(r"([0-9]+h)", expand=False).fillna("0h")
    meta["intervention"] = meta["condition"].map(
        {"Control": "none", "LPS": "LPS", "Dex_LPS": "Dex_after_LPS"}
    ).fillna("unknown")

    matrix = build_matrix(meta)
    matrix.to_csv(args.out / "expression_matrix.tsv.gz", sep="\t", compression="gzip")
    write_tsv(meta, args.out / "sample_metadata.tsv")
    write_tsv(
        pd.DataFrame({"gene_symbol": matrix.index, "feature_id": matrix.index, "row_index": range(1, len(matrix) + 1)}),
        args.out / "gene_mapping.tsv",
    )

    for name in [
        "gse93735_matched_intervention_effects.tsv",
        "gse93735_matched_intervention_positive_stratum_summary.tsv",
        "gse93735_preregistered_axes.tsv",
    ]:
        src = args.input / name
        if src.exists():
            (args.out / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    qc = pd.DataFrame(
        [
            {"metric": "processing_status", "value": "pass"},
            {"metric": "n_samples", "value": len(meta)},
            {"metric": "n_genes", "value": len(matrix)},
            {"metric": "conditions", "value": ";".join(sorted(meta["condition"].unique()))},
            {"metric": "dex_only_available", "value": False},
            {"metric": "analysis_caveat", "value": "Dex-only samples are not required by the locked late-Dex reversal gate"},
        ]
    )
    write_tsv(qc, args.out / "qc_summary.tsv")
    print(f"Wrote GSE93735 processed inputs to {args.out}")


if __name__ == "__main__":
    main()
