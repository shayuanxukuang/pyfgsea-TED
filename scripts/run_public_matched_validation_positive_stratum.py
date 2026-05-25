from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_external" / "matched_validation_positive_stratum" / "gse137412_lps_dex"
RAW = OUT / "raw"

GSE = "GSE137412"
GEO_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE137412"
SUPPL_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE137nnn/GSE137412/suppl/"
    "GSE137412_macro_Wt_DexLPSoverLPS_expressed_lengthscaledTPMs.txt.gz"
)

AXES = {
    "lps_proinflammatory_axis": {
        "role": "primary_positive_reversal_axis",
        "expected": "LPS_up_and_DexLPS_down_vs_LPS",
        "genes": [
            "Il1b",
            "Il6",
            "Tnf",
            "Cxcl1",
            "Cxcl2",
            "Ccl2",
            "Ccl3",
            "Ccl4",
            "Ptgs2",
            "Nos2",
            "Il12b",
            "Csf2",
            "Saa3",
            "Icam1",
            "Socs3",
        ],
    },
    "chemokine_cytokine_axis": {
        "role": "secondary_positive_reversal_axis",
        "expected": "LPS_up_and_DexLPS_down_vs_LPS",
        "genes": [
            "Cxcl1",
            "Cxcl2",
            "Cxcl3",
            "Cxcl10",
            "Ccl2",
            "Ccl3",
            "Ccl4",
            "Ccl5",
            "Il1a",
            "Il1b",
            "Il6",
            "Il12b",
            "Tnf",
        ],
    },
    "glucocorticoid_engagement_axis": {
        "role": "intervention_engagement_axis",
        "expected": "DexLPS_up_vs_LPS",
        "genes": [
            "Tsc22d3",
            "Dusp1",
            "Fkbp5",
            "Per1",
            "Klf2",
            "Klf4",
            "Zfp36",
            "Nfkbia",
            "Tnfaip3",
        ],
    },
    "housekeeping_negative_control": {
        "role": "negative_control_axis",
        "expected": "no_large_reversal",
        "genes": ["Actb", "Gapdh", "Hprt", "Ppia", "Rplp0", "Tbp", "B2m", "Gusb"],
    },
    "ribosome_negative_control": {
        "role": "negative_control_axis",
        "expected": "no_large_reversal",
        "genes": [
            "Rpl3",
            "Rpl4",
            "Rpl5",
            "Rpl7",
            "Rpl8",
            "Rpl10",
            "Rpl13a",
            "Rps3",
            "Rps6",
            "Rps8",
            "Rps12",
            "Rps18",
        ],
    },
}

PREREG = OUT / "public_matched_validation_positive_stratum_preregistration.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_tsv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    return path


def write_preregistration() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "dataset": GSE,
            "organism": "Mus musculus",
            "system": "wild-type bone marrow-derived macrophages",
            "assay": "bulk RNA-seq processed length-scaled TPM",
            "metadata_only_selection_reason": (
                "GEO metadata specifies vehicle, LPS and Dex+LPS groups with three biological "
                "replicates per treatment and public processed data."
            ),
            "perturbation": "6h 100 ng/ml LPS",
            "matched_intervention": "16h 1uM dexamethasone plus 6h LPS",
            "control": "vehicle",
            "selected_before_expression_analysis": True,
            "geo_url": GEO_URL,
            "supplementary_file_url": SUPPL_URL,
        }
    ]
    write_tsv(pd.DataFrame(rows), OUT / "public_matched_validation_dataset_selection.tsv")

    axis_rows = []
    for axis, spec in AXES.items():
        axis_rows.append(
            {
                "axis": axis,
                "role": spec["role"],
                "expected": spec["expected"],
                "n_preregistered_genes": len(spec["genes"]),
                "genes": ";".join(spec["genes"]),
            }
        )
    write_tsv(pd.DataFrame(axis_rows), OUT / "public_matched_validation_preregistered_axes.tsv")

    text = f"""# Pre-registered public matched-validation positive stratum

## Dataset selected from metadata only

Dataset: `{GSE}`  
GEO: {GEO_URL}  
Processed file: {SUPPL_URL}

GEO metadata describes wild-type mouse bone marrow-derived macrophages treated with vehicle, LPS, or dexamethasone plus LPS, with three biological replicates per treatment. The dataset is selected before expression-level download or analysis because it has a matched perturbation/intervention design:

- control: vehicle
- perturbation: 6 h LPS
- matched intervention: 16 h dexamethasone plus 6 h LPS

This stratum is not used to validate the GSE271399 biology. It is used only to test whether the TED claim-boundary machinery can recognize a public matched intervention case when a perturbation-associated pathway event is reversed by a matched intervention.

## Locked event families and expected directions

The primary event is LPS inflammatory activation followed by dexamethasone-associated reversal. Gene sets are frozen in `public_matched_validation_preregistered_axes.tsv` before expression analysis.

Primary positive gate:

1. `lps_proinflammatory_axis` has at least 6 detected genes.
2. LPS versus vehicle axis delta is >= 0.25 on mean log2(TPM + 1) activity.
3. Dex+LPS versus LPS axis delta is <= -0.15.
4. Recovery fraction = (LPS - DexLPS) / (LPS - vehicle) is >= 0.30.

Secondary support:

1. `chemokine_cytokine_axis` follows the same direction as the primary positive gate.
2. `glucocorticoid_engagement_axis` is higher in Dex+LPS than LPS by >= 0.20 on mean log2(TPM + 1) activity.

Negative-control gate:

1. Housekeeping and ribosome negative-control axes must not show stronger apparent recovery than the primary inflammatory axis.
2. If a negative-control axis has equal or stronger recovery than the primary axis, the case is downgraded to artifact-sensitive rather than counted as a positive stratum.

## Locked decision rule

`matched_validation_positive_stratum = pass` only if:

- the primary positive gate passes;
- the glucocorticoid engagement gate passes;
- the negative-control gate passes.

`partial_positive = true` if the primary positive gate passes but either engagement or negative-control specificity fails.

No threshold or gene set may be changed after expression data are downloaded. Any missing genes are recorded in the gene-detection table rather than replaced.

Registration timestamp: {datetime.now().isoformat(timespec="seconds")}
"""
    PREREG.write_text(text, encoding="utf-8")
    audit = {
        "preregistration_file": str(PREREG.relative_to(ROOT)),
        "preregistration_sha256": sha256(PREREG),
        "status": "rules_frozen_before_expression_download_or_analysis",
    }
    (OUT / "public_matched_validation_preregistration_hash.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )


def download_file() -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    target = RAW / Path(SUPPL_URL).name
    if not target.exists():
        urllib.request.urlretrieve(SUPPL_URL, target)
    manifest = pd.DataFrame(
        [
            {
                "dataset": GSE,
                "url": SUPPL_URL,
                "local_file": str(target.relative_to(ROOT)),
                "size_bytes": target.stat().st_size,
                "sha256": sha256(target),
                "download_or_reuse_time": datetime.now().isoformat(timespec="seconds"),
            }
        ]
    )
    write_tsv(manifest, OUT / "gse137412_download_manifest.tsv")
    return target


def read_expression(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        first = handle.readline()
        sep = "," if first.count(",") > first.count("\t") else "\t"
        handle.seek(0)
        df = pd.read_csv(handle, sep=sep)
    return df


def infer_gene_column(df: pd.DataFrame) -> str:
    candidates = [
        c
        for c in df.columns
        if c.lower() in {"gene", "gene_name", "genesymbol", "symbol", "external_gene_name", "mgi_symbol"}
    ]
    if candidates:
        return candidates[0]
    object_cols = [c for c in df.columns[:5] if df[c].dtype == object]
    if object_cols:
        return object_cols[-1]
    return df.columns[0]


def infer_sample_group(col: str) -> str | None:
    low = col.lower()
    gar_map = {
        "gar1198": "vehicle",
        "gar1199": "vehicle",
        "gar1200": "vehicle",
        "gar1201": "LPS",
        "gar1202": "LPS",
        "gar1203": "LPS",
        "gar1204": "DexLPS",
        "gar1205": "DexLPS",
        "gar1206": "DexLPS",
    }
    for token, group in gar_map.items():
        if token in low:
            return group
    if "dex" in low and "lps" in low:
        return "DexLPS"
    if "lps" in low:
        return "LPS"
    if "veh" in low or "vehicle" in low or "ctrl" in low or "control" in low:
        return "vehicle"
    return None


def build_expression_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gene_col = infer_gene_column(df)
    sample_cols = [c for c in df.columns if c.startswith("TPMs_") and infer_sample_group(c)]
    if not sample_cols:
        numeric = [c for c in df.columns if c != gene_col and pd.api.types.is_numeric_dtype(df[c])]
        sample_cols = numeric
    meta = pd.DataFrame(
        [
            {
                "sample": col,
                "group": infer_sample_group(col) or "unknown",
                "replicate": re.sub(r".*rep(?:licate)?[_-]?([0-9]+).*", r"\1", col, flags=re.I),
            }
            for col in sample_cols
        ]
    )
    expr = df[[gene_col, *sample_cols]].copy()
    expr[gene_col] = expr[gene_col].astype(str).str.replace(r"\.\d+$", "", regex=True)
    expr = expr.dropna(subset=[gene_col])
    expr = expr.groupby(gene_col, as_index=False)[sample_cols].mean(numeric_only=True)
    expr = expr.set_index(gene_col)
    expr = expr.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return expr, meta


def axis_score(expr: pd.DataFrame, genes: list[str]) -> tuple[pd.Series, list[str], list[str]]:
    upper_to_gene = {g.upper(): g for g in expr.index.astype(str)}
    present = [upper_to_gene[g.upper()] for g in genes if g.upper() in upper_to_gene]
    missing = [g for g in genes if g.upper() not in upper_to_gene]
    if present:
        log_expr = (expr.loc[present] + 1.0).applymap(lambda x: math.log2(float(x)))
        return log_expr.mean(axis=0), present, missing
    return pd.Series(0.0, index=expr.columns), present, missing


def mean_by_group(scores: pd.Series, meta: pd.DataFrame) -> dict[str, float]:
    joined = meta.copy()
    joined["score"] = joined["sample"].map(scores.to_dict())
    return joined.groupby("group")["score"].mean().to_dict()


def run_analysis() -> None:
    if not PREREG.exists():
        raise SystemExit("Preregistration is missing. Run with --write-prereg first.")
    prereg_hash = sha256(PREREG)
    data_path = download_file()
    raw = read_expression(data_path)
    expr, meta = build_expression_matrix(raw)
    write_tsv(meta, OUT / "gse137412_sample_metadata.tsv")

    gene_rows = []
    score_rows = []
    effect_rows = []
    for axis, spec in AXES.items():
        scores, present, missing = axis_score(expr, spec["genes"])
        gene_rows.append(
            {
                "axis": axis,
                "role": spec["role"],
                "expected": spec["expected"],
                "n_preregistered_genes": len(spec["genes"]),
                "n_detected_genes": len(present),
                "detected_genes": ";".join(present),
                "missing_genes": ";".join(missing),
            }
        )
        for sample, score in scores.items():
            group = meta.loc[meta["sample"] == sample, "group"].iloc[0]
            score_rows.append({"axis": axis, "sample": sample, "group": group, "log2_tpm_axis_score": score})
        means = mean_by_group(scores, meta)
        veh = means.get("vehicle", float("nan"))
        lps = means.get("LPS", float("nan"))
        dex = means.get("DexLPS", float("nan"))
        lps_delta = lps - veh
        intervention_delta = dex - lps
        recovery = (lps - dex) / lps_delta if pd.notna(lps_delta) and abs(lps_delta) > 1e-9 else float("nan")
        if spec["expected"] == "LPS_up_and_DexLPS_down_vs_LPS":
            direction_pass = lps_delta >= 0.25 and intervention_delta <= -0.15 and recovery >= 0.30
        elif spec["expected"] == "DexLPS_up_vs_LPS":
            direction_pass = intervention_delta >= 0.20
        else:
            direction_pass = abs(lps_delta) < 0.25 and abs(intervention_delta) < 0.25
        effect_rows.append(
            {
                "axis": axis,
                "role": spec["role"],
                "expected": spec["expected"],
                "n_detected_genes": len(present),
                "mean_vehicle": veh,
                "mean_LPS": lps,
                "mean_DexLPS": dex,
                "lps_vs_vehicle_delta": lps_delta,
                "DexLPS_vs_LPS_delta": intervention_delta,
                "recovery_fraction": recovery,
                "direction_pass_preregistered": bool(direction_pass and len(present) >= (6 if "primary" in spec["role"] else 3)),
            }
        )
    gene_df = pd.DataFrame(gene_rows)
    score_df = pd.DataFrame(score_rows)
    effect_df = pd.DataFrame(effect_rows)
    write_tsv(gene_df, OUT / "gse137412_axis_gene_detection.tsv")
    write_tsv(score_df, OUT / "gse137412_axis_sample_scores.tsv")
    write_tsv(effect_df, OUT / "gse137412_matched_validation_effects.tsv")

    primary = effect_df.loc[effect_df["axis"] == "lps_proinflammatory_axis"].iloc[0]
    engagement = effect_df.loc[effect_df["axis"] == "glucocorticoid_engagement_axis"].iloc[0]
    neg = effect_df[effect_df["role"] == "negative_control_axis"].copy()
    primary_recovery = float(primary["recovery_fraction"])
    neg_max_recovery = float(neg["recovery_fraction"].fillna(0).clip(lower=0).max()) if not neg.empty else 0.0
    primary_gate = bool(primary["direction_pass_preregistered"])
    engagement_gate = bool(engagement["direction_pass_preregistered"])
    negative_control_gate = primary_recovery > (neg_max_recovery + 0.10)
    final_pass = primary_gate and engagement_gate and negative_control_gate
    partial = primary_gate and not final_pass
    summary = pd.DataFrame(
        [
            {
                "dataset": GSE,
                "preregistration_sha256": prereg_hash,
                "primary_gate_pass": primary_gate,
                "engagement_gate_pass": engagement_gate,
                "negative_control_gate_pass": negative_control_gate,
                "primary_recovery_fraction": primary_recovery,
                "max_negative_control_recovery_fraction": neg_max_recovery,
                "matched_validation_positive_stratum": "pass" if final_pass else "fail",
                "partial_positive": partial,
                "allowed_claim": (
                    "public matched intervention positive stratum for claim-boundary audit"
                    if final_pass
                    else "public intervention dataset analyzed under preregistered rules but not counted as positive stratum"
                ),
                "forbidden_claim": "does not validate GSE271399 biology or substitute for matched full-length GATA1 rescue",
            }
        ]
    )
    write_tsv(summary, OUT / "public_matched_validation_positive_stratum_summary.tsv")
    write_tsv(summary, OUT / "gse137412_claim_boundary_positive_stratum.tsv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-prereg", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.write_prereg:
        write_preregistration()
        print(PREREG)
    if args.run:
        run_analysis()
        print(OUT)
    if not args.write_prereg and not args.run:
        parser.error("Use --write-prereg and then --run.")


if __name__ == "__main__":
    main()
