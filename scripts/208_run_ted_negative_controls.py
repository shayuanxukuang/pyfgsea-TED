from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLES = ROOT / "results" / "ted_known_source_validation" / "tables"


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, default=ROOT / "data" / "processed" / "ted_known_source")
    parser.add_argument("--axes", type=Path, default=ROOT / "config" / "ted_negative_control_axes.yml")
    parser.add_argument("--out", type=Path, default=DEFAULT_TABLES)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.out.glob("gse*_negative_control_results.tsv")):
        df = read_tsv(path)
        if df.empty:
            continue
        rows.append(df)
    if (args.out / "ted_negative_control_summary.tsv").exists():
        rows.append(read_tsv(args.out / "ted_negative_control_summary.tsv"))
    negative = pd.concat(rows, ignore_index=True, sort=False).drop_duplicates() if rows else pd.DataFrame()
    if negative.empty:
        negative = pd.DataFrame(
            [{"dataset": "none", "negative_control_pass": False, "status": "missing", "reason": "no negative-control tables found"}]
        )
    write_tsv(negative, args.out / "ted_negative_control_summary.tsv")

    spec_rows = []
    for dataset, sub in negative.groupby("dataset", dropna=False):
        pass_col = sub.get("negative_control_pass", pd.Series([False]))
        pass_rate = pass_col.astype(str).str.lower().isin(["true", "pass"]).mean()
        spec_rows.append(
            {
                "dataset": dataset,
                "specificity_vs_random": "not_computed_in_lightweight_run",
                "specificity_vs_stress": "available" if sub.astype(str).apply(lambda x: x.str.contains("stress", case=False)).any().any() else "not_available",
                "specificity_vs_ribosome": "available" if sub.astype(str).apply(lambda x: x.str.contains("ribosome", case=False)).any().any() else "not_available",
                "specificity_vs_mitochondrial": "available" if sub.astype(str).apply(lambda x: x.str.contains("mito", case=False)).any().any() else "not_available",
                "negative_control_pass": bool(pass_rate == 1.0),
                "negative_control_pass_rate": pass_rate,
            }
        )
    write_tsv(pd.DataFrame(spec_rows), args.out / "ted_specificity_summary.tsv")
    print(f"Wrote negative-control summaries to {args.out}")


if __name__ == "__main__":
    main()
