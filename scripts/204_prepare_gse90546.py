from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data_external" / "ted_upgrade_candidate_audit" / "downloads" / "GSE90546"
DEFAULT_OUT = ROOT / "data" / "processed" / "ted_known_source" / "GSE90546"

UPR_BRANCHES = {
    "PERK_integrated_stress_response": ["EIF2AK3", "ATF4", "DDIT3", "PPP1R15A"],
    "IRE1_XBP1": ["ERN1", "XBP1", "DNAJB9", "HERPUD1"],
    "ATF6": ["ATF6", "HSPA5", "HSP90B1", "CALR"],
}


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def tar_inventory(raw: Path) -> pd.DataFrame:
    if not raw.exists():
        return pd.DataFrame()
    rows = []
    with tarfile.open(raw) as archive:
        for member in archive.getmembers():
            rows.append({"member": member.name, "bytes": member.size})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.input / "GSE90546_RAW.tar"
    inventory = tar_inventory(raw)
    if not inventory.empty:
        write_tsv(inventory, args.out / "raw_inventory.tsv")

    branch_rows = []
    for branch, genes in UPR_BRANCHES.items():
        for gene in genes:
            branch_rows.append({"perturbed_gene": gene, "upr_branch": branch, "expected_ted_axis": branch})
    write_tsv(pd.DataFrame(branch_rows), args.out / "upr_branch_annotation.tsv")

    series_files = sorted(args.input.glob("*series_matrix*.txt.gz"))
    manifest = pd.DataFrame(
        [
            {"file": str(path.resolve()), "role": "series_matrix_metadata", "bytes": path.stat().st_size}
            for path in series_files
        ]
    )
    write_tsv(manifest, args.out / "processing_manifest.tsv")

    status = "pass_raw_available" if raw.exists() else "metadata_only_raw_required_not_present"
    qc = pd.DataFrame(
        [
            {"metric": "processing_status", "value": status},
            {"metric": "raw_tar_present", "value": raw.exists()},
            {"metric": "n_series_matrix_files", "value": len(series_files)},
            {"metric": "n_upr_branch_rows", "value": len(branch_rows)},
            {
                "metric": "analysis_caveat",
                "value": "Expression matrix is not emitted until GSE90546_RAW.tar is downloaded and unpacked",
            },
        ]
    )
    write_tsv(qc, args.out / "qc_summary.tsv")
    print(f"Wrote GSE90546 metadata/prep outputs to {args.out}")


if __name__ == "__main__":
    main()
