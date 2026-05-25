from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import tarfile
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_external" / "matched_validation_positive_stratum" / "gse93735_lps_dex_late"
RAW = OUT / "raw"
EXTRACTED = RAW / "extracted"

GSE = "GSE93735"
GEO_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE93735"
SUPPL_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE93nnn/GSE93735/suppl/GSE93735_RAW.tar"

AXES = {
    "lps_proinflammatory_axis": {
        "role": "primary_positive_reversal_axis",
        "expected": "LPS_10h_up_and_DexLate_10h_down_vs_LPS_10h",
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
        "expected": "LPS_10h_up_and_DexLate_10h_down_vs_LPS_10h",
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
        "expected": "DexLate_10h_up_vs_LPS_10h",
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

PREREG = OUT / "gse93735_matched_intervention_preregistration.md"


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
    write_tsv(
        pd.DataFrame(
            [
                {
                    "dataset": GSE,
                    "organism": "Mus musculus",
                    "system": "mouse bone marrow-derived macrophages",
                    "assay": "RNA-seq processed FPKM tracking",
                    "metadata_only_selection_reason": (
                        "GEO metadata describes an RNA-seq time course with LPS alone and Dex late after LPS, "
                        "and states that Dex following LPS has an anti-inflammatory profile."
                    ),
                    "control": "0h LPS baseline",
                    "perturbation": "10h LPS",
                    "matched_intervention": "Dex late, 10h LPS",
                    "selected_before_expression_analysis": True,
                    "geo_url": GEO_URL,
                    "supplementary_file_url": SUPPL_URL,
                }
            ]
        ),
        OUT / "gse93735_dataset_selection.tsv",
    )
    write_tsv(
        pd.DataFrame(
            [
                {
                    "axis": axis,
                    "role": spec["role"],
                    "expected": spec["expected"],
                    "n_preregistered_genes": len(spec["genes"]),
                    "genes": ";".join(spec["genes"]),
                }
                for axis, spec in AXES.items()
            ]
        ),
        OUT / "gse93735_preregistered_axes.tsv",
    )
    text = f"""# GSE93735 pre-registered matched intervention positive stratum

## Dataset selected from metadata only

Dataset: `{GSE}`  
GEO: {GEO_URL}  
Processed file: {SUPPL_URL}

GEO metadata describes mouse macrophage RNA-seq with LPS alone and dexamethasone treatment either before or after LPS. This preregistration uses only metadata-level information. Expression values are not inspected before these rules are written.

Locked design:

- control: 0h LPS baseline
- perturbation: 10h LPS
- matched intervention: Dex late, 10h LPS

This positive stratum is not used to validate GSE271399 biology. It is used to test whether TED can recognize a public matched intervention case when perturbation-associated inflammatory activation is reversed by a matched intervention.

## Locked axes and thresholds

Gene sets are frozen in `gse93735_preregistered_axes.tsv`.

Primary positive gate:

1. `lps_proinflammatory_axis` has at least 6 detected genes.
2. 10h LPS versus 0h baseline axis delta is >= 0.25 on mean log2(FPKM + 1) activity.
3. Dex late 10h versus 10h LPS axis delta is <= -0.15.
4. Recovery fraction = (LPS10h - DexLate10h) / (LPS10h - baseline0h) is >= 0.30.

Secondary support:

1. `chemokine_cytokine_axis` follows the same direction.
2. `glucocorticoid_engagement_axis` is higher in Dex late 10h than 10h LPS by >= 0.20.

Negative-control gate:

Housekeeping and ribosome axes must not show stronger apparent recovery than the primary inflammatory axis. If either negative-control axis is equal or stronger, the case is not counted as a positive stratum.

Locked final decision:

`matched_validation_positive_stratum = pass` only if the primary positive gate, glucocorticoid engagement gate and negative-control gate all pass.

No threshold or gene set may be changed after processed data are downloaded. Missing genes are recorded and not replaced.

Registration timestamp: {datetime.now().isoformat(timespec="seconds")}
"""
    PREREG.write_text(text, encoding="utf-8")
    (OUT / "gse93735_preregistration_hash.json").write_text(
        json.dumps(
            {
                "preregistration_file": str(PREREG.relative_to(ROOT)),
                "preregistration_sha256": sha256(PREREG),
                "status": "rules_frozen_before_expression_download_or_analysis",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def download_and_extract() -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    tar_path = RAW / "GSE93735_RAW.tar"
    if not tar_path.exists():
        urllib.request.urlretrieve(SUPPL_URL, tar_path)
    EXTRACTED.mkdir(parents=True, exist_ok=True)
    if not any(EXTRACTED.iterdir()):
        with tarfile.open(tar_path) as tar:
            tar.extractall(EXTRACTED)
    write_tsv(
        pd.DataFrame(
            [
                {
                    "dataset": GSE,
                    "url": SUPPL_URL,
                    "local_file": str(tar_path.relative_to(ROOT)),
                    "size_bytes": tar_path.stat().st_size,
                    "sha256": sha256(tar_path),
                    "download_or_reuse_time": datetime.now().isoformat(timespec="seconds"),
                }
            ]
        ),
        OUT / "gse93735_download_manifest.tsv",
    )
    return tar_path


def group_from_name(name: str) -> str | None:
    low = name.lower()
    gsm_map = {
        "gsm2461310": "baseline_0h",
        "gsm2461313": "baseline_0h",
        "gsm2461312": "LPS_10h",
        "gsm2461315": "LPS_10h",
        "gsm2461319": "DexLate_LPS_10h",
        "gsm2461323": "DexLate_LPS_10h",
    }
    for token, group in gsm_map.items():
        if token in low:
            return group
    if "dex_late" in low and "10h" in low:
        return "DexLate_LPS_10h"
    if "dex" in low:
        return "other_dex"
    if "10h" in low and "lps" in low:
        return "LPS_10h"
    if "0h" in low and "lps" in low:
        return "baseline_0h"
    return None


def read_fpkm_file(path: Path) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        df = pd.read_csv(handle, sep="\t")
    gene_col = "gene_short_name" if "gene_short_name" in df.columns else "tracking_id"
    fpkm_col = "FPKM" if "FPKM" in df.columns else next(c for c in df.columns if c.lower() == "fpkm")
    out = df[[gene_col, fpkm_col]].copy()
    out.columns = ["gene", "fpkm"]
    out["gene"] = out["gene"].astype(str).str.replace(r"\.\d+$", "", regex=True)
    out["fpkm"] = pd.to_numeric(out["fpkm"], errors="coerce").fillna(0.0)
    return out


def build_matrix() -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted([p for p in EXTRACTED.rglob("*") if "fpkm_tracking" in p.name.lower()])
    sample_rows = []
    series = []
    for path in files:
        group = group_from_name(path.name)
        if group not in {"baseline_0h", "LPS_10h", "DexLate_LPS_10h"}:
            continue
        df = read_fpkm_file(path)
        sample = path.name.replace(".gz", "").replace(".fpkm_tracking", "")
        sample_rows.append({"sample": sample, "group": group, "file": str(path.relative_to(ROOT))})
        s = df.groupby("gene")["fpkm"].mean()
        s.name = sample
        series.append(s)
    if not series:
        raise RuntimeError("No matching FPKM tracking files were found for baseline, LPS_10h and DexLate_LPS_10h.")
    expr = pd.concat(series, axis=1).fillna(0.0)
    meta = pd.DataFrame(sample_rows)
    return expr, meta


def axis_score(expr: pd.DataFrame, genes: list[str]) -> tuple[pd.Series, list[str], list[str]]:
    upper_to_gene = {g.upper(): g for g in expr.index.astype(str)}
    present = [upper_to_gene[g.upper()] for g in genes if g.upper() in upper_to_gene]
    missing = [g for g in genes if g.upper() not in upper_to_gene]
    if present:
        return (expr.loc[present] + 1.0).applymap(lambda x: math.log2(float(x))).mean(axis=0), present, missing
    return pd.Series(0.0, index=expr.columns), present, missing


def mean_by_group(scores: pd.Series, meta: pd.DataFrame) -> dict[str, float]:
    joined = meta.copy()
    joined["score"] = joined["sample"].map(scores.to_dict())
    return joined.groupby("group")["score"].mean().to_dict()


def run_analysis() -> None:
    if not PREREG.exists():
        raise SystemExit("Preregistration is missing. Run --write-prereg first.")
    prereg_hash = sha256(PREREG)
    download_and_extract()
    expr, meta = build_matrix()
    write_tsv(meta, OUT / "gse93735_sample_metadata.tsv")

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
            score_rows.append({"axis": axis, "sample": sample, "group": group, "log2_fpkm_axis_score": score})
        means = mean_by_group(scores, meta)
        base = means.get("baseline_0h", float("nan"))
        lps = means.get("LPS_10h", float("nan"))
        dex = means.get("DexLate_LPS_10h", float("nan"))
        lps_delta = lps - base
        intervention_delta = dex - lps
        recovery = (lps - dex) / lps_delta if pd.notna(lps_delta) and abs(lps_delta) > 1e-9 else float("nan")
        if "up_and_DexLate" in spec["expected"]:
            direction_pass = lps_delta >= 0.25 and intervention_delta <= -0.15 and recovery >= 0.30
        elif spec["expected"] == "DexLate_10h_up_vs_LPS_10h":
            direction_pass = intervention_delta >= 0.20
        else:
            direction_pass = abs(lps_delta) < 0.25 and abs(intervention_delta) < 0.25
        effect_rows.append(
            {
                "axis": axis,
                "role": spec["role"],
                "expected": spec["expected"],
                "n_detected_genes": len(present),
                "mean_baseline_0h": base,
                "mean_LPS_10h": lps,
                "mean_DexLate_LPS_10h": dex,
                "LPS10h_vs_baseline_delta": lps_delta,
                "DexLate10h_vs_LPS10h_delta": intervention_delta,
                "recovery_fraction": recovery,
                "direction_pass_preregistered": bool(direction_pass and len(present) >= (6 if "primary" in spec["role"] else 3)),
            }
        )
    gene_df = pd.DataFrame(gene_rows)
    score_df = pd.DataFrame(score_rows)
    effect_df = pd.DataFrame(effect_rows)
    write_tsv(gene_df, OUT / "gse93735_axis_gene_detection.tsv")
    write_tsv(score_df, OUT / "gse93735_axis_sample_scores.tsv")
    write_tsv(effect_df, OUT / "gse93735_matched_intervention_effects.tsv")

    primary = effect_df.loc[effect_df["axis"] == "lps_proinflammatory_axis"].iloc[0]
    engagement = effect_df.loc[effect_df["axis"] == "glucocorticoid_engagement_axis"].iloc[0]
    neg = effect_df[effect_df["role"] == "negative_control_axis"].copy()
    primary_recovery = float(primary["recovery_fraction"])
    neg_max_recovery = float(neg["recovery_fraction"].fillna(0).clip(lower=0).max()) if not neg.empty else 0.0
    primary_gate = bool(primary["direction_pass_preregistered"])
    engagement_gate = bool(engagement["direction_pass_preregistered"])
    negative_control_gate = primary_recovery > (neg_max_recovery + 0.10)
    final_pass = primary_gate and engagement_gate and negative_control_gate
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
                "partial_positive": primary_gate and not final_pass,
                "allowed_claim": (
                    "public matched intervention positive stratum for claim-boundary audit"
                    if final_pass
                    else "public intervention dataset analyzed under preregistered rules but not counted as positive stratum"
                ),
                "forbidden_claim": "does not validate GSE271399 biology or substitute for matched full-length GATA1 rescue",
            }
        ]
    )
    write_tsv(summary, OUT / "gse93735_matched_intervention_positive_stratum_summary.tsv")
    write_tsv(summary, OUT / "gse93735_claim_boundary_positive_stratum.tsv")


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
