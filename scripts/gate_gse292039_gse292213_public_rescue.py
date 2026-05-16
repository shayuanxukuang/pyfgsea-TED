#!/usr/bin/env python
"""Gate GSE292039/GSE292213 as public functional-style validation candidates.

The gate is intentionally strict: these datasets may support Level 4B
public functional-style alignment, but they do not replace an own full-length
GATA1 rescue experiment and cannot make GSE271399 strict Level 4.
"""

from __future__ import annotations

import gzip
import hashlib
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "data_external" / "ted_generalization_panel" / "public_rescue_gate"
DOWNLOAD_DIR = OUTDIR / "downloads"


ACCESSIONS = ["GSE292039", "GSE292213"]
SERIES_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc}&targ=self&form=text&view=quick"
SAMPLE_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc}&targ=self&form=text&view=quick"
RAW_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}nnn/{acc}/suppl/{acc}_RAW.tar"


ENTREZ_TO_SYMBOL = {
    "2623": "GATA1",
    "10661": "KLF1",
    "6886": "TAL1",
    "161882": "ZFPM1",
    "2057": "EPOR",
    "4778": "NFE2",
    "4005": "LMO2",
    "8328": "GFI1B",
    "286": "ANK1",
    "2993": "GYPA",
    "2994": "GYPB",
    "2995": "GYPC",
    "6005": "RHAG",
    "114625": "ERMAP",
    "212": "ALAS2",
    "2235": "FECH",
    "7037": "TFRC",
    "51312": "SLC25A37",
    "23456": "ABCB10",
    "54977": "SLC25A38",
    "55240": "STEAP3",
    "3039": "HBA1",
    "3040": "HBA2",
    "3043": "HBB",
    "3047": "HBG1",
    "3048": "HBG2",
    "51327": "AHSP",
    "6521": "SLC4A1",
    "60": "ACTB",
    "2597": "GAPDH",
    "6175": "RPLP0",
    "567": "B2M",
    "3251": "HPRT1",
}

AXES = {
    "regulatory_axis": ["GATA1", "KLF1", "TAL1", "ZFPM1", "EPOR", "NFE2", "LMO2", "GFI1B"],
    "maturation_membrane_axis": ["ANK1", "GYPA", "GYPB", "GYPC", "RHAG", "ERMAP"],
    "heme_iron_axis": ["ALAS2", "FECH", "TFRC", "SLC25A37", "ABCB10", "SLC25A38", "STEAP3"],
    "erythroid_output_axis": ["HBA1", "HBA2", "HBB", "HBG1", "HBG2", "AHSP", "SLC4A1"],
    "housekeeping_control": ["ACTB", "GAPDH", "RPLP0", "B2M", "HPRT1"],
}


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def write_tsv(df: pd.DataFrame, name: str) -> Path:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / name
    df.to_csv(path, sep="\t", index=False, na_rep="")
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_text(url: str) -> str:
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    return r.text


def parse_series(acc: str) -> dict[str, object]:
    text = get_text(SERIES_URL.format(acc=acc))
    out: dict[str, object] = {"accession": acc, "series_url": SERIES_URL.format(acc=acc), "soft_bytes": len(text)}
    sample_ids = []
    design = []
    summary = []
    for line in text.splitlines():
        if line.startswith("!Series_title"):
            out["series_title"] = line.split("=", 1)[1].strip()
        elif line.startswith("!Series_status"):
            out["series_status"] = line.split("=", 1)[1].strip()
        elif line.startswith("!Series_sample_id"):
            sample_ids.append(line.split("=", 1)[1].strip())
        elif line.startswith("!Series_overall_design"):
            design.append(line.split("=", 1)[1].strip())
        elif line.startswith("!Series_summary"):
            summary.append(line.split("=", 1)[1].strip())
    out["sample_ids"] = sample_ids
    out["n_samples"] = len(sample_ids)
    out["overall_design"] = " ".join(design)
    out["summary_head"] = " ".join(summary)[:600]
    return out


def parse_sample(acc: str) -> dict[str, str]:
    text = get_text(SAMPLE_URL.format(acc=acc))
    row = {"sample_accession": acc, "sample_url": SAMPLE_URL.format(acc=acc)}
    characteristics = []
    for line in text.splitlines():
        if line.startswith("!Sample_title"):
            row["title"] = line.split("=", 1)[1].strip()
        elif line.startswith("!Sample_source_name"):
            row["source_name"] = line.split("=", 1)[1].strip()
        elif line.startswith("!Sample_characteristics"):
            value = line.split("=", 1)[1].strip()
            characteristics.append(value)
            if ":" in value:
                key, val = value.split(":", 1)
                row[key.strip().lower().replace(" ", "_")] = val.strip()
        elif line.startswith("!Sample_description"):
            row["description"] = line.split("=", 1)[1].strip()
    row["characteristics"] = ";".join(characteristics)
    row["condition_group"] = infer_condition(row)
    return row


def infer_condition(row: dict[str, str]) -> str:
    text = " ".join(str(row.get(k, "")) for k in ["title", "source_name", "treatment", "description"]).lower()
    if "non-responder" in text or "non responder" in text:
        return "luspatercept_non_responder"
    if "responder" in text:
        return "luspatercept_responder"
    if "gdf11" in text:
        return "GDF11"
    if "tgfb" in text or "tgf" in text:
        return "TGFb1"
    if "vehicle" in text or "control" in text:
        return "vehicle_or_control"
    return "unknown"


def download_raw(acc: str) -> dict[str, object]:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    url = RAW_URL.format(prefix=acc[:6], acc=acc)
    dest = DOWNLOAD_DIR / f"{acc}_RAW.tar"
    if not dest.exists():
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        dest.write_bytes(r.content)
    with tarfile.open(dest, "r") as archive:
        members = archive.getmembers()
    return {
        "accession": acc,
        "download_url": url,
        "local_path": str(dest),
        "downloaded": dest.exists(),
        "local_size_bytes": dest.stat().st_size,
        "sha256": sha256_file(dest),
        "tar_member_count": len(members),
        "tar_members": ";".join(m.name for m in members[:40]),
        "tar_suffixes": ";".join(f"{k}:{v}" for k, v in pd.Series([Path(m.name).suffix for m in members]).value_counts().items()),
    }


def parse_counts_tar(path: Path, sample_meta: pd.DataFrame) -> pd.DataFrame:
    sample_by_gsm = {row["sample_accession"]: row.to_dict() for _, row in sample_meta.iterrows()}
    target_ids = set(ENTREZ_TO_SYMBOL)
    rows = []
    with tarfile.open(path, "r") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith(".counts.gz"):
                continue
            gsm = member.name.split("_", 1)[0]
            f = archive.extractfile(member)
            if f is None:
                continue
            target_counts: dict[str, float] = {}
            total_counts = 0.0
            detected_ids = set()
            with gzip.GzipFile(fileobj=f) as gz:
                for raw in gz:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    gene_id, count_text = parts[0], parts[1]
                    try:
                        count = float(count_text)
                    except ValueError:
                        continue
                    total_counts += count
                    if gene_id in target_ids:
                        detected_ids.add(gene_id)
                        target_counts[ENTREZ_TO_SYMBOL[gene_id]] += count
            meta = sample_by_gsm.get(gsm, {})
            row = {
                "sample_accession": gsm,
                "file_name": member.name,
                "condition_group": meta.get("condition_group", "unknown") if isinstance(meta, dict) else "unknown",
                "source_name": meta.get("source_name", "") if isinstance(meta, dict) else "",
                "title": meta.get("title", "") if isinstance(meta, dict) else "",
                "total_counts": total_counts,
                "target_gene_ids_detected": ";".join(sorted(detected_ids)),
                "n_target_gene_ids_detected": len(detected_ids),
            }
            for symbol, count in target_counts.items():
                row[symbol] = np.log2((count / max(total_counts, 1.0)) * 1_000_000 + 1.0)
            rows.append(row)
    return pd.DataFrame(rows)


def axis_scores(counts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in counts.iterrows():
        for axis, genes in AXES.items():
            present = [g for g in genes if g in counts.columns]
            score = float(pd.to_numeric(row[present], errors="coerce").mean()) if present else np.nan
            rows.append(
                {
                    "sample_accession": row["sample_accession"],
                    "condition_group": row["condition_group"],
                    "source_name": row["source_name"],
                    "axis": axis,
                    "n_axis_genes_scored": len(present),
                    "logCPM_axis_score": score,
                }
            )
    return pd.DataFrame(rows)


def contrast_axis_scores(scores: pd.DataFrame) -> pd.DataFrame:
    contrasts = [
        ("GDF11_vs_vehicle", "GDF11", "vehicle_or_control"),
        ("luspatercept_responder_vs_non_responder", "luspatercept_responder", "luspatercept_non_responder"),
    ]
    rows = []
    for contrast, case, control in contrasts:
        for axis, sub in scores.groupby("axis"):
            case_vals = pd.to_numeric(sub[sub["condition_group"].eq(case)]["logCPM_axis_score"], errors="coerce")
            ctrl_vals = pd.to_numeric(sub[sub["condition_group"].eq(control)]["logCPM_axis_score"], errors="coerce")
            case_vals = case_vals.dropna()
            ctrl_vals = ctrl_vals.dropna()
            if case_vals.empty or ctrl_vals.empty:
                rows.append(
                    {
                        "contrast": contrast,
                        "axis": axis,
                        "case_group": case,
                        "control_group": control,
                        "n_case": len(case_vals),
                        "n_control": len(ctrl_vals),
                        "case_mean": "",
                        "control_mean": "",
                        "delta_case_minus_control": "",
                        "direction": "not_scored",
                        "ted_rescue_alignment_interpretation": "count-level data available but TED axis scoring requires gene-ID mapping",
                    }
                )
                continue
            delta = float(case_vals.mean() - ctrl_vals.mean())
            rows.append(
                {
                    "contrast": contrast,
                    "axis": axis,
                    "case_group": case,
                    "control_group": control,
                    "n_case": len(case_vals),
                    "n_control": len(ctrl_vals),
                    "case_mean": float(case_vals.mean()),
                    "control_mean": float(ctrl_vals.mean()),
                    "delta_case_minus_control": delta,
                    "direction": "up" if delta > 0 else "down" if delta < 0 else "flat",
                    "ted_rescue_alignment_interpretation": interpret_contrast(contrast, axis, delta),
                }
            )
    return pd.DataFrame(rows)


def interpret_contrast(contrast: str, axis: str, delta: float) -> str:
    if contrast == "GDF11_vs_vehicle":
        if axis in {"regulatory_axis", "maturation_membrane_axis", "heme_iron_axis", "erythroid_output_axis"} and delta < 0:
            return "GDF11 perturbation directionally matches TED-predicted erythroid/heme loss, but is not rescue"
        return "not a direct TED rescue alignment"
    if contrast == "luspatercept_responder_vs_non_responder":
        return "public response-stratified association only; not paired treatment rescue"
    return ""


def build_gate(series_rows: list[dict[str, object]], downloads: pd.DataFrame, alignment: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in series_rows:
        acc = str(s["accession"])
        title = str(s.get("series_title", ""))
        is_rna = acc == "GSE292039"
        is_chip = acc == "GSE292213"
        member_count = int(downloads[downloads["accession"].eq(acc)]["tar_member_count"].iloc[0])
        count_level = is_rna and member_count >= 2
        chip_context = is_chip and member_count >= 2
        has_public_response = "Luspatercept" in str(s.get("overall_design", "")) or "Luspatercept" in str(s.get("summary_head", ""))
        has_treatment = "GDF11" in str(s.get("overall_design", "")) or "TGF" in str(s.get("overall_design", ""))
        if count_level:
            gate_status = "pass_for_Level4B_public_functional_style_gate_not_strict_Level4"
        elif chip_context:
            gate_status = "pass_for_ChIP_context_only_not_count_level_rescue"
        else:
            gate_status = "metadata_only"
        rows.append(
            {
                "accession": acc,
                "series_title": title,
                "n_samples": s.get("n_samples", ""),
                "raw_tar_member_count": member_count,
                "count_level_RNA_available": count_level,
                "chip_peak_context_available": chip_context,
                "rescue_or_response_contrast_detected": has_public_response,
                "perturbation_treatment_contrast_detected": has_treatment,
                "can_measure_TED_axes": count_level,
                "negative_controls_possible": count_level,
                "gate_status": gate_status,
                "max_allowed_claim": "Level_4B_public_functional_style_alignment_candidate" if count_level else "chromatin_context_or_metadata_support_only",
                "forbidden_claim": "not own wet-lab Level 4; not T21_GATA1s full-length GATA1 rescue",
            }
        )
    return pd.DataFrame(rows)


def build_claim_ceiling(gate: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "evidence_layer": "GSE292039_RNAseq",
                "status": gate.loc[gate["accession"].eq("GSE292039"), "gate_status"].iloc[0] if "GSE292039" in set(gate["accession"]) else "missing",
                "allowed_claim": "public functional-style erythroid/TGF-GDF11/luspatercept RNA-seq gate can test TED rescue-axis alignment",
                "forbidden_claim": "direct full-length GATA1 rescue in T21_GATA1s; strict Level 4",
                "claim_ceiling": "Level_4B_candidate_if_alignment_passes;not_Level_4",
            },
            {
                "evidence_layer": "GSE292213_ChIPseq",
                "status": gate.loc[gate["accession"].eq("GSE292213"), "gate_status"].iloc[0] if "GSE292213" in set(gate["accession"]) else "missing",
                "allowed_claim": "public ChIP peak context for GDF11-related erythroid response",
                "forbidden_claim": "RNA rescue or causal chromatin-first mechanism",
                "claim_ceiling": "context_only_below_Level_4B_unless_linked_to_RNA_axis",
            },
            {
                "evidence_layer": "strict_Level_4",
                "status": "not_met",
                "allowed_claim": "none",
                "forbidden_claim": "own functional rescue completed",
                "claim_ceiling": "not_Level_4",
            },
        ]
    )


def build_manifest(paths: list[Path]) -> pd.DataFrame:
    purposes = {
        "gse292039_gse292213_gate.tsv": "pass/fail gates for public functional-style validation",
        "gse292039_download_decision.tsv": "downloaded raw tar paths, sizes, members, and hashes",
        "gse292039_public_rescue_contrast_map.tsv": "sample metadata and planned contrasts",
        "gse292039_ted_rescue_alignment_if_available.tsv": "axis-level alignment for count-level contrasts",
        "gse292039_claim_ceiling.tsv": "strict Level 4 and Level 4B guardrails",
    }
    return pd.DataFrame(
        [
            {
                "output_file": rel(path),
                "n_rows": safe_n_rows(path),
                "purpose": purposes.get(path.name, ""),
            }
            for path in paths
        ]
    )


def safe_n_rows(path: Path) -> int:
    try:
        return len(pd.read_csv(path, sep="\t"))
    except pd.errors.EmptyDataError:
        return 0


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    series_rows = [parse_series(acc) for acc in ACCESSIONS]
    sample_rows = []
    for s in series_rows:
        for sample_id in s["sample_ids"]:
            sample_rows.append(parse_sample(sample_id))
            time.sleep(0.05)
    sample_meta = pd.DataFrame(sample_rows)
    download_rows = [download_raw(acc) for acc in ACCESSIONS]
    downloads = pd.DataFrame(download_rows)
    rna_tar = Path(downloads[downloads["accession"].eq("GSE292039")]["local_path"].iloc[0])
    counts = parse_counts_tar(rna_tar, sample_meta)
    scores = axis_scores(counts)
    alignment = contrast_axis_scores(scores)
    gate = build_gate(series_rows, downloads, alignment)
    claim = build_claim_ceiling(gate)

    contrast_map = sample_meta.merge(
        counts[["sample_accession", "file_name", "total_counts"]], on="sample_accession", how="left"
    )
    paths = [
        write_tsv(gate, "gse292039_gse292213_gate.tsv"),
        write_tsv(downloads, "gse292039_download_decision.tsv"),
        write_tsv(contrast_map, "gse292039_public_rescue_contrast_map.tsv"),
        write_tsv(alignment, "gse292039_ted_rescue_alignment_if_available.tsv"),
        write_tsv(claim, "gse292039_claim_ceiling.tsv"),
    ]
    manifest = build_manifest(paths)
    manifest_path = write_tsv(manifest, "gse292039_gate_output_manifest.tsv")
    print(f"Wrote {len(paths) + 1} GSE292039/GSE292213 gate files to {rel(OUTDIR)}")
    print(rel(manifest_path))


if __name__ == "__main__":
    main()
