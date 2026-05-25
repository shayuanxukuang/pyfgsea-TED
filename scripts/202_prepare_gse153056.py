from __future__ import annotations

import argparse
import gzip
import shutil
import tarfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data_external" / "ted_upgrade_candidate_audit" / "downloads" / "GSE153056"
DEFAULT_OUT = ROOT / "data" / "processed" / "ted_known_source" / "GSE153056"

TAR_MEMBERS = {
    "expression_matrix.tsv.gz": "GSM4633614_ECCITE_cDNA_counts.tsv.gz",
    "protein_matrix.tsv.gz": "GSM4633615_ECCITE_ADT_counts.tsv.gz",
    "guide_matrix.tsv.gz": "GSM4633618_ECCITE_GDO_counts.tsv.gz",
    "arrayed_expression_matrix.tsv.gz": "GSM4633608_ECCITE_Arrayed_cDNA_counts.tsv.gz",
    "arrayed_protein_matrix.tsv.gz": "GSM4633609_ECCITE_Arrayed_ADT_counts.tsv.gz",
    "arrayed_guide_matrix.tsv.gz": "GSM4633612_ECCITE_Arrayed_GO_CITE03_counts.tsv.gz",
}


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def extract_member(tar_path: Path, member: str, dest: Path) -> int:
    with tarfile.open(tar_path) as archive:
        source = archive.extractfile(member)
        if source is None:
            raise FileNotFoundError(member)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as handle:
            shutil.copyfileobj(source, handle)
    return dest.stat().st_size


def standardize_pooled_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    out = df.copy()
    out["cell_id"] = out["Unnamed: 0"].astype(str)
    out["source_condition_raw"] = out.get("con", "unknown").astype(str)
    out["condition"] = out["source_condition_raw"].map({"tx": "IFNg", "ctrl": "control"}).fillna(out["source_condition_raw"])
    out["perturbed_gene"] = out.get("gene", "unknown").astype(str)
    out["guide_id"] = out.get("guide_ID", "unknown").astype(str)
    out["is_non_targeting"] = out["perturbed_gene"].eq("NT") | out["guide_id"].str.startswith("NT", na=False)
    out["replicate"] = out.get("replicate", "unknown").astype(str)
    keep = [
        "cell_id",
        "condition",
        "source_condition_raw",
        "perturbed_gene",
        "guide_id",
        "is_non_targeting",
        "replicate",
        "nCount_RNA",
        "nFeature_RNA",
        "nCount_ADT",
        "nFeature_ADT",
        "percent.mito",
    ]
    return out[[col for col in keep if col in out.columns]]


def standardize_arrayed_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    out = df.copy()
    out["cell_id"] = out["Unnamed: 0"].astype(str)
    raw = out.get("MULTI_ID", "").astype(str)
    out["source_condition_raw"] = raw
    out["condition"] = "other"
    out.loc[raw.str.contains("IFNG|_tx|-tx", case=False, regex=True, na=False), "condition"] = "IFNg"
    out.loc[raw.str.contains("CT|ctrl|control", case=False, regex=True, na=False), "condition"] = "control"
    guide = out.get("GO_cite_classification", "unknown").astype(str)
    out["guide_id"] = guide
    out["perturbed_gene"] = guide.str.replace(r"g[0-9]+.*$", "", regex=True)
    out["is_non_targeting"] = guide.str.startswith("NT", na=False) | guide.eq("Negative")
    keep = [
        "cell_id",
        "condition",
        "source_condition_raw",
        "perturbed_gene",
        "guide_id",
        "is_non_targeting",
        "nCount_RNA",
        "nFeature_RNA",
        "nCount_ADT",
        "nFeature_ADT",
        "percent.mito",
    ]
    return out[[col for col in keep if col in out.columns]]


def matrix_feature_mapping(matrix_path: Path, out_path: Path) -> int:
    rows = []
    with gzip.open(matrix_path, "rt", encoding="utf-8", errors="replace") as handle:
        header = handle.readline()
        for idx, line in enumerate(handle, start=1):
            feature = line.split("\t", 1)[0].strip().strip('"')
            rows.append({"feature_id": feature, "gene_symbol": feature, "row_index": idx})
    write_tsv(pd.DataFrame(rows), out_path)
    return len(rows)


def protein_mapping(barcode_member: str, tar_path: Path, out_path: Path) -> int:
    rows = []
    with tarfile.open(tar_path) as archive:
        source = archive.extractfile(barcode_member)
        if source is None:
            raise FileNotFoundError(barcode_member)
        with gzip.open(source, "rt", encoding="utf-8", errors="replace") as handle:
            for idx, line in enumerate(handle, start=1):
                parts = [part.strip() for part in line.rstrip("\n").split(",")]
                if len(parts) >= 2:
                    rows.append({"feature_id": parts[0], "protein_symbol": parts[1], "row_index": idx})
    write_tsv(pd.DataFrame(rows), out_path)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tar_path = args.input / "GSE153056_RAW.tar"
    if not tar_path.exists():
        raise FileNotFoundError(f"Missing {tar_path}")

    manifest_rows = []
    for dest_name, member in TAR_MEMBERS.items():
        dest = args.out / dest_name
        size = extract_member(tar_path, member, dest)
        manifest_rows.append({"output": dest_name, "source": member, "bytes": size})

    pooled_meta = standardize_pooled_metadata(args.input / "GSE153056_ECCITE_metadata.tsv.gz")
    arrayed_meta = standardize_arrayed_metadata(args.input / "GSE153056_ECCITE_Arrayed_metadata.tsv.gz")
    pooled_meta.to_csv(args.out / "cell_metadata.tsv.gz", sep="\t", index=False, compression="gzip")
    arrayed_meta.to_csv(args.out / "arrayed_cell_metadata.tsv.gz", sep="\t", index=False, compression="gzip")

    n_genes = matrix_feature_mapping(args.out / "expression_matrix.tsv.gz", args.out / "gene_mapping.tsv")
    n_proteins = protein_mapping("GSM4633615_ECCITE_ADT_Barcodes.csv.gz", tar_path, args.out / "protein_mapping.tsv")

    qc = pd.DataFrame(
        [
            {"metric": "processing_status", "value": "pass"},
            {"metric": "pooled_cells", "value": len(pooled_meta)},
            {"metric": "arrayed_cells", "value": len(arrayed_meta)},
            {"metric": "pooled_conditions", "value": ";".join(sorted(pooled_meta["condition"].dropna().unique()))},
            {"metric": "arrayed_conditions", "value": ";".join(sorted(arrayed_meta["condition"].dropna().unique()))},
            {"metric": "genes_in_expression_matrix", "value": n_genes},
            {"metric": "proteins_in_adt_matrix", "value": n_proteins},
            {
                "metric": "analysis_caveat",
                "value": "pooled ECCITE metadata is IFNg-only; IFNg-vs-control baseline is available through arrayed metadata",
            },
        ]
    )
    write_tsv(pd.DataFrame(manifest_rows), args.out / "processing_manifest.tsv")
    write_tsv(qc, args.out / "qc_summary.tsv")
    print(f"Wrote GSE153056 processed inputs to {args.out}")


if __name__ == "__main__":
    main()
