"""Preregistered GSE298761 single-cell validation workflow.

This script follows the frozen analysis plan in
data_external/t21_gata1s_external_validation/outputs/gse298761_preregistered_analysis_plan.md.
It builds an h5ad from GEO CellRanger MTX triplets, runs QC sensitivity,
audits erythroid marker recovery, then performs conservative block-aware
pseudobulk and TED-Lite trajectory summaries only if the erythroid continuum is
detectable.
"""

from __future__ import annotations

import gzip
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats
from scipy.io import mmread


ROOT = Path("data_external/t21_gata1s_external_validation")
OUT = ROOT / "outputs"
MTX_DIR = ROOT / "GSE298761" / "processed" / "cellranger_mtx"
H5AD = ROOT / "GSE298761" / "processed" / "gse298761_merged_raw_counts.h5ad"
SOURCE = "GSE298761"

AXES = {
    "erythroid_output_axis": ["Gata1", "Klf1", "Tal1", "Zfpm1", "Epor", "Nfe2", "Gypa", "Alas2", "Hba-a1", "Hba-a2", "Hbb-bt", "Hbb-bs"],
    "heme_iron_axis": ["Alas2", "Fech", "Tfrc", "Slc25a37", "Abcb10"],
    "maturation_membrane_axis": ["Gypa", "Ank1", "Slc4a1", "Rhag", "Epb42"],
    "glycolysis_axis": ["Pkm", "Ldha", "Aldoa", "Gapdh", "Pgk1", "Eno1"],
    "immature_progenitor_axis": ["Kit", "Myb", "Gata2", "Runx1", "Erg"],
}

MARKER_GROUPS = {
    "early_progenitor": ["Kit", "Myb", "Gata2", "Runx1", "Erg"],
    "erythroid_regulatory": ["Gata1", "Klf1", "Tal1", "Zfpm1", "Epor", "Nfe2"],
    "maturation_membrane": ["Gypa", "Ank1", "Slc4a1", "Rhag", "Epb42"],
    "heme_iron": ["Alas2", "Fech", "Tfrc", "Slc25a37", "Abcb10"],
    "globin": ["Hba-a1", "Hba-a2", "Hbb-bt", "Hbb-bs"],
    "glycolysis": ["Pkm", "Ldha", "Aldoa", "Gapdh", "Pgk1", "Eno1"],
}

NEGATIVE_CONTROLS = {
    "housekeeping": ["Actb", "B2m", "Tbp", "Hprt", "Ppia", "Rplp0", "Ywhaz", "Eef1a1"],
    "proliferation": ["Mki67", "Top2a", "Pcna", "Ccnb1", "Ccnb2", "Cdk1", "Ube2c", "Mcm2", "Mcm3", "Mcm4", "Mcm5", "Mcm6", "Mcm7"],
}

NONERYTHROID_MARKERS = {
    "myeloid": ["Lyz2", "S100a8", "S100a9", "Csf1r", "Itgam"],
    "lymphoid": ["Cd3d", "Cd3e", "Cd79a", "Ms4a1"],
    "endothelial": ["Pecam1", "Kdr", "Cdh5"],
    "stromal": ["Col1a1", "Col1a2", "Dcn"],
    "megakaryocyte": ["Pf4", "Ppbp", "Itga2b", "Gp9"],
}

QC_TIERS = {
    "lenient": {"min_genes": 100, "min_counts": 200, "max_percent_mito": 25.0},
    "standard": {"min_genes": 200, "min_counts": 500, "max_percent_mito": 15.0},
    "strict": {"min_genes": 500, "min_counts": 1000, "max_percent_mito": 10.0},
}

EXPECTED = {
    "erythroid_output_axis": "down_or_delayed",
    "heme_iron_axis": "down_or_delayed",
    "maturation_membrane_axis": "down_or_delayed",
    "glycolysis_axis": "up",
    "immature_progenitor_axis": "retained_or_up",
}


@dataclass
class MatrixFiles:
    sample_id: str
    condition: str
    embryo_block: str
    barcode_path: Path
    feature_path: Path
    matrix_path: Path


def log(message: str) -> None:
    print(f"[gse298761] {message}", flush=True)


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sample_metadata() -> pd.DataFrame:
    meta = pd.read_csv(OUT / "gse298761_sample_metadata.tsv", sep="\t")
    meta = meta.rename(columns={"sample": "sample_id"})
    meta["condition"] = meta["condition"].replace({"GATA1s": "GATA1s", "WT": "WT"})
    meta["genotype"] = meta["condition"]
    meta["embryo_block"] = meta["sample_id"]
    meta["source_accession"] = SOURCE
    keep = ["sample_id", "title", "source", "tissue", "cell_type", "genotype", "condition", "embryo_block", "source_accession", "description"]
    return meta[[c for c in keep if c in meta.columns]].copy()


def resolve_matrix_files(meta: pd.DataFrame) -> list[MatrixFiles]:
    items: list[MatrixFiles] = []
    for row in meta.itertuples(index=False):
        sid = row.sample_id
        matches = {
            "barcodes": sorted(MTX_DIR.glob(f"{sid}_*_barcodes.tsv.gz")),
            "features": sorted(MTX_DIR.glob(f"{sid}_*_features.tsv.gz")),
            "matrix": sorted(MTX_DIR.glob(f"{sid}_*_matrix.mtx.gz")),
        }
        if any(len(v) != 1 for v in matches.values()):
            raise FileNotFoundError(f"Expected one 10x triplet for {sid}; found {matches}")
        items.append(
            MatrixFiles(
                sample_id=sid,
                condition=row.condition,
                embryo_block=row.embryo_block,
                barcode_path=matches["barcodes"][0],
                feature_path=matches["features"][0],
                matrix_path=matches["matrix"][0],
            )
        )
    return items


def read_gzip_lines(path: Path) -> list[str]:
    with gzip.open(path, "rt", errors="replace") as fh:
        return [line.rstrip("\n") for line in fh]


def make_unique(values: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for value in values:
        if value not in seen:
            seen[value] = 0
            out.append(value)
        else:
            seen[value] += 1
            out.append(f"{value}-{seen[value]}")
    return out


def read_one_sample(mf: MatrixFiles) -> ad.AnnData:
    barcodes = [f"{mf.sample_id}:{bc}" for bc in read_gzip_lines(mf.barcode_path)]
    features = pd.read_csv(mf.feature_path, sep="\t", header=None, compression="gzip")
    gene_ids = features.iloc[:, 0].astype(str).tolist()
    gene_symbols = features.iloc[:, 1].astype(str).tolist()
    feature_types = features.iloc[:, 2].astype(str).tolist() if features.shape[1] > 2 else ["Gene Expression"] * len(gene_symbols)
    matrix = mmread(str(mf.matrix_path)).tocsr().T
    var_names = make_unique(gene_symbols)
    obs = pd.DataFrame(index=barcodes)
    obs["barcode"] = barcodes
    obs["sample_id"] = mf.sample_id
    obs["condition"] = mf.condition
    obs["genotype"] = mf.condition
    obs["embryo_block"] = mf.embryo_block
    obs["source_accession"] = SOURCE
    var = pd.DataFrame(index=var_names)
    var["gene_symbol"] = gene_symbols
    var["gene_id"] = gene_ids
    var["feature_type"] = feature_types
    return ad.AnnData(X=matrix, obs=obs, var=var)


def build_h5ad() -> ad.AnnData:
    meta = load_sample_metadata()
    items = resolve_matrix_files(meta)
    write_tsv(meta, OUT / "gse298761_sample_metadata_resolved.tsv")
    reports = []
    adatas = []
    for mf in items:
        log(f"reading {mf.sample_id}")
        a = read_one_sample(mf)
        adatas.append(a)
        reports.append(
            {
                "sample_id": mf.sample_id,
                "condition": mf.condition,
                "barcodes_file": str(mf.barcode_path.resolve()),
                "features_file": str(mf.feature_path.resolve()),
                "matrix_file": str(mf.matrix_path.resolve()),
                "n_cells": a.n_obs,
                "n_features": a.n_vars,
                "matrix_nnz": int(a.X.nnz),
                "matrix_bytes": mf.matrix_path.stat().st_size,
                "matrix_sha256": sha256(mf.matrix_path),
            }
        )
    log("concatenating samples")
    adata = ad.concat(adatas, join="outer", label="sample_concat", keys=[mf.sample_id for mf in items], index_unique=None)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    if "gene_symbol" not in adata.var.columns:
        adata.var = adatas[0].var.reindex(adata.var_names).copy()
        adata.var["gene_symbol"] = adata.var["gene_symbol"].fillna(pd.Series(adata.var_names, index=adata.var_names))
    log(f"writing {H5AD}")
    H5AD.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(H5AD, compression="gzip")
    report_df = pd.DataFrame(reports)
    report_df["merged_h5ad"] = str(H5AD.resolve())
    report_df["merged_n_cells"] = adata.n_obs
    report_df["merged_n_genes"] = adata.n_vars
    write_tsv(report_df, OUT / "gse298761_h5ad_build_report.tsv")
    write_tsv(
        adata.obs.groupby(["sample_id", "condition"], observed=True).size().reset_index(name="n_cells"),
        OUT / "gse298761_cell_count_by_sample.tsv",
    )
    return adata


def add_qc_metrics(adata: ad.AnnData) -> None:
    x = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    adata.obs["n_counts"] = np.asarray(x.sum(axis=1)).ravel()
    adata.obs["n_genes"] = np.diff(x.indptr)
    gene_upper = pd.Index(adata.var["gene_symbol"].astype(str)).str.upper()
    mito = gene_upper.str.startswith("MT-")
    ribo = gene_upper.str.match(r"^RP[SL]")
    mito = np.asarray(mito)
    ribo = np.asarray(ribo)
    adata.var["is_mito"] = mito
    adata.var["is_ribo"] = ribo
    mito_counts = np.asarray(x[:, mito].sum(axis=1)).ravel() if mito.any() else np.zeros(adata.n_obs)
    ribo_counts = np.asarray(x[:, ribo].sum(axis=1)).ravel() if ribo.any() else np.zeros(adata.n_obs)
    total = np.asarray(adata.obs["n_counts"], dtype=float)
    total_safe = np.where(total > 0, total, np.nan)
    adata.obs["percent_mito"] = np.nan_to_num(mito_counts / total_safe * 100.0)
    adata.obs["percent_ribo"] = np.nan_to_num(ribo_counts / total_safe * 100.0)


def qc_mask(adata: ad.AnnData, tier: str) -> np.ndarray:
    cfg = QC_TIERS[tier]
    return (
        (adata.obs["n_genes"].to_numpy() >= cfg["min_genes"])
        & (adata.obs["n_counts"].to_numpy() >= cfg["min_counts"])
        & (adata.obs["percent_mito"].to_numpy() <= cfg["max_percent_mito"])
    )


def write_qc_outputs(adata: ad.AnnData) -> None:
    rows = []
    for keys, sub in adata.obs.groupby(["sample_id", "condition"], observed=True):
        sid, cond = keys
        rows.append(
            {
                "sample_id": sid,
                "condition": cond,
                "n_cells": len(sub),
                "median_n_counts": float(sub["n_counts"].median()),
                "median_n_genes": float(sub["n_genes"].median()),
                "median_percent_mito": float(sub["percent_mito"].median()),
                "median_percent_ribo": float(sub["percent_ribo"].median()),
            }
        )
    write_tsv(pd.DataFrame(rows), OUT / "gse298761_qc_summary.tsv")

    sens = []
    retained = []
    for tier in QC_TIERS:
        mask = qc_mask(adata, tier)
        obs = adata.obs.copy()
        obs["retained"] = mask
        sens.append(
            {
                "tier": tier,
                **QC_TIERS[tier],
                "n_cells_total": adata.n_obs,
                "n_cells_retained": int(mask.sum()),
                "fraction_retained": float(mask.mean()),
                "n_GATA1s_retained": int(((obs["condition"] == "GATA1s") & obs["retained"]).sum()),
                "n_WT_retained": int(((obs["condition"] == "WT") & obs["retained"]).sum()),
            }
        )
        grouped = obs.groupby(["sample_id", "condition"], observed=True)["retained"].agg(["sum", "count"]).reset_index()
        grouped["tier"] = tier
        grouped = grouped.rename(columns={"sum": "n_retained", "count": "n_input"})
        grouped["fraction_retained"] = grouped["n_retained"] / grouped["n_input"]
        retained.append(grouped)
    write_tsv(pd.DataFrame(sens), OUT / "gse298761_qc_threshold_sensitivity.tsv")
    write_tsv(pd.concat(retained, ignore_index=True), OUT / "gse298761_cells_retained_by_sample_condition.tsv")


def log_normalized_matrix(adata: ad.AnnData) -> sparse.csr_matrix:
    x = adata.X.tocsr().astype(np.float32)
    counts = np.asarray(x.sum(axis=1)).ravel()
    scale = np.divide(1e4, counts, out=np.zeros_like(counts, dtype=np.float32), where=counts > 0)
    x = sparse.diags(scale).dot(x).tocsr()
    x.data = np.log1p(x.data)
    return x


def gene_indices(adata: ad.AnnData, genes: list[str]) -> list[int]:
    symbol_to_idx: dict[str, int] = {}
    for i, sym in enumerate(adata.var["gene_symbol"].astype(str)):
        symbol_to_idx.setdefault(sym.upper(), i)
    return [symbol_to_idx[g.upper()] for g in genes if g.upper() in symbol_to_idx]


def module_score(logx: sparse.csr_matrix, indices: list[int]) -> np.ndarray:
    if not indices:
        return np.full(logx.shape[0], np.nan)
    return np.asarray(logx[:, indices].mean(axis=1)).ravel()


def add_module_scores(adata: ad.AnnData, logx: sparse.csr_matrix) -> None:
    for name, genes in {**AXES, **MARKER_GROUPS, **NEGATIVE_CONTROLS}.items():
        adata.obs[f"score_{name}"] = module_score(logx, gene_indices(adata, genes))
    nonery_genes = sorted({g for genes in NONERYTHROID_MARKERS.values() for g in genes})
    adata.obs["score_nonerythroid"] = module_score(logx, gene_indices(adata, nonery_genes))
    adata.obs["score_globin_terminal"] = module_score(logx, gene_indices(adata, MARKER_GROUPS["globin"] + MARKER_GROUPS["maturation_membrane"]))
    adata.obs["score_erythroid_core"] = module_score(
        logx,
        gene_indices(
            adata,
            sorted(
                set(
                    MARKER_GROUPS["erythroid_regulatory"]
                    + MARKER_GROUPS["maturation_membrane"]
                    + MARKER_GROUPS["heme_iron"]
                    + MARKER_GROUPS["globin"]
                )
            ),
        ),
    )


def write_marker_detection(adata: ad.AnnData, logx: sparse.csr_matrix) -> None:
    markers = []
    for group, genes in MARKER_GROUPS.items():
        for gene in genes:
            idx = gene_indices(adata, [gene])
            if not idx:
                markers.append(
                    {
                        "marker_group": group,
                        "gene": gene,
                        "present": False,
                        "n_cells_expressing": 0,
                        "fraction_cells_expressing": 0.0,
                        "mean_log1p_normalized": math.nan,
                    }
                )
                continue
            col = logx[:, idx[0]]
            markers.append(
                {
                    "marker_group": group,
                    "gene": gene,
                    "present": True,
                    "n_cells_expressing": int(col.count_nonzero()),
                    "fraction_cells_expressing": float(col.count_nonzero() / adata.n_obs),
                    "mean_log1p_normalized": float(col.mean()),
                }
            )
    write_tsv(pd.DataFrame(markers), OUT / "gse298761_marker_detection.tsv")


def annotate_cells(adata: ad.AnnData) -> None:
    obs = adata.obs
    ery = obs["score_erythroid_core"].fillna(0)
    nonery = obs["score_nonerythroid"].fillna(0)
    immature = obs["score_immature_progenitor_axis"].fillna(0)
    terminal = obs["score_globin_terminal"].fillna(0)
    glycolysis = obs["score_glycolysis_axis"].fillna(0)
    ery_threshold = max(0.05, float(np.nanquantile(ery, 0.20)))
    terminal_hi = float(np.nanquantile(terminal, 0.70))
    immature_hi = float(np.nanquantile(immature, 0.70))
    is_ery = (ery >= ery_threshold) & (ery >= nonery * 0.8)
    label = np.full(adata.n_obs, "nonerythroid_or_low_signal", dtype=object)
    label[is_ery & (immature >= immature_hi) & (terminal < terminal_hi)] = "erythroid_progenitor_like"
    label[is_ery & (terminal >= terminal_hi)] = "terminal_erythroid_like"
    label[is_ery & (label == "nonerythroid_or_low_signal")] = "erythroid_intermediate_like"
    label[(~is_ery) & (glycolysis > np.nanquantile(glycolysis, 0.90))] = "nonerythroid_or_metabolic_high"
    adata.obs["erythroid_candidate"] = is_ery.to_numpy()
    adata.obs["annotation_label"] = label
    pt_raw = (
        obs["score_erythroid_regulatory"].fillna(0)
        + obs["score_maturation_membrane"].fillna(0)
        + obs["score_heme_iron"].fillna(0)
        + obs["score_globin"].fillna(0)
        - obs["score_early_progenitor"].fillna(0)
    )
    rank = pd.Series(pt_raw).rank(method="average").to_numpy()
    adata.obs["marker_pseudotime"] = (rank - np.nanmin(rank)) / (np.nanmax(rank) - np.nanmin(rank))


def write_annotation_outputs(adata: ad.AnnData) -> None:
    audit = (
        adata.obs.groupby(["condition", "annotation_label"], observed=True)
        .size()
        .reset_index(name="n_cells")
    )
    totals = adata.obs.groupby("condition", observed=True).size().rename("condition_total")
    audit = audit.merge(totals.reset_index(), on="condition", how="left")
    audit["fraction_of_condition"] = audit["n_cells"] / audit["condition_total"]
    write_tsv(audit, OUT / "gse298761_celltype_annotation_audit.tsv")

    ery = adata.obs[adata.obs["erythroid_candidate"]].copy()
    rows = []
    for keys, sub in ery.groupby(["sample_id", "condition"], observed=True):
        sid, cond = keys
        rows.append(
            {
                "sample_id": sid,
                "condition": cond,
                "n_erythroid_candidate_cells": len(sub),
                "median_marker_pseudotime": float(sub["marker_pseudotime"].median()),
                "mean_erythroid_output_axis": float(sub["score_erythroid_output_axis"].mean()),
                "mean_heme_iron_axis": float(sub["score_heme_iron_axis"].mean()),
                "mean_maturation_membrane_axis": float(sub["score_maturation_membrane_axis"].mean()),
                "mean_glycolysis_axis": float(sub["score_glycolysis_axis"].mean()),
                "mean_immature_progenitor_axis": float(sub["score_immature_progenitor_axis"].mean()),
            }
        )
    write_tsv(pd.DataFrame(rows), OUT / "gse298761_erythroid_subset_summary.tsv")

    cont = []
    for keys, sub in adata.obs.groupby(["sample_id", "condition"], observed=True):
        sid, cond = keys
        cont.append(
            {
                "sample_id": sid,
                "condition": cond,
                "n_cells": len(sub),
                "n_erythroid_candidate": int(sub["erythroid_candidate"].sum()),
                "fraction_erythroid_candidate": float(sub["erythroid_candidate"].mean()),
                "fraction_nonerythroid_or_low_signal": float((sub["annotation_label"] == "nonerythroid_or_low_signal").mean()),
                "mean_nonerythroid_score": float(sub["score_nonerythroid"].mean()),
            }
        )
    write_tsv(pd.DataFrame(cont), OUT / "gse298761_nonerythroid_contamination.tsv")


def summarize_effect(values: pd.DataFrame, module: str) -> dict[str, object]:
    g = values[values["condition"] == "GATA1s"][module].dropna()
    w = values[values["condition"] == "WT"][module].dropna()
    effect = float(g.mean() - w.mean()) if len(g) and len(w) else math.nan
    p = float(stats.ttest_ind(g, w, equal_var=False).pvalue) if len(g) >= 2 and len(w) >= 2 else math.nan
    return {
        "mean_GATA1s": float(g.mean()) if len(g) else math.nan,
        "mean_WT": float(w.mean()) if len(w) else math.nan,
        "delta_GATA1s_minus_WT": effect,
        "welch_p_block_level": p,
        "n_GATA1s_blocks": len(g),
        "n_WT_blocks": len(w),
    }


def expected_call(axis: str, effect: float) -> str:
    exp = EXPECTED.get(axis, "not_prespecified")
    if not np.isfinite(effect):
        return "not_scored"
    if exp == "up" or exp == "retained_or_up":
        return "pass" if effect > 0 else "fail"
    if exp == "down_or_delayed":
        return "pass" if effect < 0 else "fail"
    return "not_prespecified"


def pseudobulk_and_controls(adata: ad.AnnData, logx: sparse.csr_matrix) -> None:
    rng = np.random.default_rng(298761)
    rows = []
    block_scores_all = []
    for tier in QC_TIERS:
        mask = qc_mask(adata, tier) & adata.obs["erythroid_candidate"].to_numpy()
        obs = adata.obs.loc[mask].copy()
        for keys, sub in obs.groupby(["sample_id", "condition"], observed=True):
            sid, cond = keys
            row = {"tier": tier, "sample_id": sid, "condition": cond, "n_cells": len(sub)}
            for axis in AXES:
                row[axis] = float(sub[f"score_{axis}"].mean())
            for ctrl in NEGATIVE_CONTROLS:
                row[ctrl] = float(sub[f"score_{ctrl}"].mean())
            row["ribosome_score"] = float(sub["percent_ribo"].mean())
            row["mitochondrial_percent"] = float(sub["percent_mito"].mean())
            block_scores_all.append(row)
        bdf = pd.DataFrame([r for r in block_scores_all if r["tier"] == tier])
        for axis in list(AXES) + ["housekeeping", "proliferation", "ribosome_score", "mitochondrial_percent"]:
            eff = summarize_effect(bdf, axis)
            rows.append(
                {
                    "tier": tier,
                    "module": axis,
                    "expected_direction": EXPECTED.get(axis, "negative_or_sensitivity_control"),
                    **eff,
                    "success": expected_call(axis, eff["delta_GATA1s_minus_WT"]) if axis in AXES else "control",
                }
            )
    block_scores = pd.DataFrame(block_scores_all)
    write_tsv(pd.DataFrame(rows), OUT / "gse298761_pseudobulk_block_module_effects.tsv")

    boot_rows = []
    for tier, bdf in block_scores.groupby("tier", observed=True):
        for axis in AXES:
            g = bdf[bdf["condition"] == "GATA1s"][axis].dropna().to_numpy()
            w = bdf[bdf["condition"] == "WT"][axis].dropna().to_numpy()
            if len(g) < 2 or len(w) < 2:
                continue
            boots = []
            for _ in range(1000):
                boots.append(float(rng.choice(g, len(g), replace=True).mean() - rng.choice(w, len(w), replace=True).mean()))
            boot_rows.append(
                {
                    "tier": tier,
                    "module": axis,
                    "bootstrap_mean_delta": float(np.mean(boots)),
                    "bootstrap_ci025": float(np.quantile(boots, 0.025)),
                    "bootstrap_ci975": float(np.quantile(boots, 0.975)),
                    "fraction_expected_sign": float(np.mean(np.asarray(boots) < 0)) if EXPECTED[axis] == "down_or_delayed" else float(np.mean(np.asarray(boots) > 0)),
                }
            )
    write_tsv(pd.DataFrame(boot_rows), OUT / "gse298761_block_bootstrap_family_effects.tsv")

    # Random matched controls: match each axis by average expression quintiles.
    mean_expr = np.asarray(adata.X.mean(axis=0)).ravel()
    bins = pd.qcut(pd.Series(mean_expr).rank(method="first"), q=10, labels=False, duplicates="drop").to_numpy()
    gene_names = adata.var["gene_symbol"].astype(str).to_numpy()
    std_mask = qc_mask(adata, "standard") & adata.obs["erythroid_candidate"].to_numpy()
    obs = adata.obs.loc[std_mask].copy()
    control_rows = []
    for axis, genes in AXES.items():
        idx = gene_indices(adata, genes)
        if not idx:
            continue
        random_effects = []
        for _ in range(100):
            sampled = []
            for i in idx:
                pool = np.where(bins == bins[i])[0]
                pool = np.setdiff1d(pool, np.asarray(idx), assume_unique=False)
                if len(pool):
                    sampled.append(int(rng.choice(pool)))
            if not sampled:
                continue
            # Approximate random module at block level using existing raw count mean as a stable negative margin proxy.
            random_gene_symbols = gene_names[sampled].tolist()
            random_score = module_score(logx, gene_indices(adata, random_gene_symbols))
            tmp = obs[["sample_id", "condition"]].copy()
            tmp["score"] = random_score[std_mask]
            btmp = tmp.groupby(["sample_id", "condition"], observed=True)["score"].mean().reset_index()
            eff = summarize_effect(btmp.rename(columns={"score": axis}), axis)["delta_GATA1s_minus_WT"]
            random_effects.append(eff)
        observed = summarize_effect(
            obs.groupby(["sample_id", "condition"], observed=True)[f"score_{axis}"].mean().reset_index().rename(columns={f"score_{axis}": axis}),
            axis,
        )["delta_GATA1s_minus_WT"]
        control_rows.append(
            {
                "tier": "standard",
                "module": axis,
                "observed_delta": observed,
                "random_matched_mean_delta": float(np.nanmean(random_effects)) if random_effects else math.nan,
                "random_matched_ci025": float(np.nanquantile(random_effects, 0.025)) if random_effects else math.nan,
                "random_matched_ci975": float(np.nanquantile(random_effects, 0.975)) if random_effects else math.nan,
                "housekeeping_delta": summarize_effect(
                    obs.groupby(["sample_id", "condition"], observed=True)["score_housekeeping"].mean().reset_index().rename(columns={"score_housekeeping": "housekeeping"}),
                    "housekeeping",
                )["delta_GATA1s_minus_WT"],
            }
        )
    write_tsv(pd.DataFrame(control_rows), OUT / "gse298761_negative_control_margin.tsv")


def run_ted_lite(adata: ad.AnnData) -> None:
    mask = qc_mask(adata, "standard") & adata.obs["erythroid_candidate"].to_numpy()
    n_ery = int(mask.sum())
    sample_counts = adata.obs.loc[mask].groupby(["sample_id", "condition"], observed=True).size()
    enough = n_ery >= 1000 and (sample_counts >= 50).all() and sample_counts.index.get_level_values("sample_id").nunique() >= 5
    diag_rows = [
        {
            "tier": "standard",
            "n_erythroid_candidate_cells": n_ery,
            "min_erythroid_cells_per_sample": int(sample_counts.min()) if len(sample_counts) else 0,
            "n_samples_with_erythroid_cells": int(sample_counts.index.get_level_values("sample_id").nunique()) if len(sample_counts) else 0,
            "pseudotime_run": bool(enough),
            "pseudotime_method": "marker_rank_primary;graph_dpt_not_run_in_this_pass",
        }
    ]
    if not enough:
        write_tsv(pd.DataFrame(diag_rows), OUT / "gse298761_pseudotime_diagnostics.tsv")
        empty = pd.DataFrame(columns=["event", "module", "mode", "evidence"])
        write_tsv(empty, OUT / "gse298761_ted_lite_event_table.tsv")
        write_tsv(empty, OUT / "gse298761_delay_vs_loss_classifier.tsv")
        write_tsv(empty, OUT / "gse298761_event_mode_summary.tsv")
        return

    obs = adata.obs.loc[mask].copy()
    obs["pt_bin"] = pd.qcut(obs["marker_pseudotime"], q=4, labels=["early", "mid_early", "mid_late", "terminal"])
    diag_rows[0]["median_pseudotime_GATA1s"] = float(obs.loc[obs["condition"] == "GATA1s", "marker_pseudotime"].median())
    diag_rows[0]["median_pseudotime_WT"] = float(obs.loc[obs["condition"] == "WT", "marker_pseudotime"].median())
    diag_rows[0]["terminal_fraction_GATA1s"] = float(((obs["condition"] == "GATA1s") & (obs["pt_bin"] == "terminal")).sum() / max((obs["condition"] == "GATA1s").sum(), 1))
    diag_rows[0]["terminal_fraction_WT"] = float(((obs["condition"] == "WT") & (obs["pt_bin"] == "terminal")).sum() / max((obs["condition"] == "WT").sum(), 1))
    write_tsv(pd.DataFrame(diag_rows), OUT / "gse298761_pseudotime_diagnostics.tsv")

    event_rows = []
    classifier_rows = []
    for axis in AXES:
        terminal = obs[obs["pt_bin"] == "terminal"]
        early = obs[obs["pt_bin"] == "early"]
        term_delta = float(terminal.loc[terminal["condition"] == "GATA1s", f"score_{axis}"].mean() - terminal.loc[terminal["condition"] == "WT", f"score_{axis}"].mean())
        early_delta = float(early.loc[early["condition"] == "GATA1s", f"score_{axis}"].mean() - early.loc[early["condition"] == "WT", f"score_{axis}"].mean())
        overall_delta = float(obs.loc[obs["condition"] == "GATA1s", f"score_{axis}"].mean() - obs.loc[obs["condition"] == "WT", f"score_{axis}"].mean())
        if axis in {"erythroid_output_axis", "heme_iron_axis", "maturation_membrane_axis"}:
            mode = "true_loss" if term_delta < -0.05 else "developmental_delay" if diag_rows[0]["terminal_fraction_GATA1s"] < diag_rows[0]["terminal_fraction_WT"] else "not_supported"
        elif axis == "glycolysis_axis":
            mode = "metabolic_shift" if overall_delta > 0 else "not_supported"
        elif axis == "immature_progenitor_axis":
            mode = "state_accumulation" if overall_delta > 0 or diag_rows[0]["median_pseudotime_GATA1s"] < diag_rows[0]["median_pseudotime_WT"] else "not_supported"
        else:
            mode = "not_supported"
        event_rows.append(
            {
                "module": axis,
                "expected_direction": EXPECTED[axis],
                "overall_delta_GATA1s_minus_WT": overall_delta,
                "early_bin_delta": early_delta,
                "terminal_bin_delta": term_delta,
                "event_mode": mode,
            }
        )
        classifier_rows.append(
            {
                "module": axis,
                "true_loss_score": -term_delta if axis in {"erythroid_output_axis", "heme_iron_axis", "maturation_membrane_axis"} else math.nan,
                "developmental_delay_score": diag_rows[0]["terminal_fraction_WT"] - diag_rows[0]["terminal_fraction_GATA1s"],
                "state_accumulation_score": overall_delta if axis == "immature_progenitor_axis" else math.nan,
                "metabolic_shift_score": overall_delta if axis == "glycolysis_axis" else math.nan,
                "assigned_mode": mode,
            }
        )
    event_df = pd.DataFrame(event_rows)
    write_tsv(event_df, OUT / "gse298761_ted_lite_event_table.tsv")
    write_tsv(pd.DataFrame(classifier_rows), OUT / "gse298761_delay_vs_loss_classifier.tsv")
    summary = event_df.groupby("event_mode", observed=True).size().reset_index(name="n_modules")
    write_tsv(summary, OUT / "gse298761_event_mode_summary.tsv")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if H5AD.exists():
        log(f"loading existing {H5AD}")
        adata = sc.read_h5ad(H5AD)
        if "gene_symbol" not in adata.var.columns:
            log("existing h5ad is missing gene_symbol metadata; rebuilding")
            H5AD.unlink()
            adata = build_h5ad()
    else:
        adata = build_h5ad()
    log("computing QC metrics")
    add_qc_metrics(adata)
    write_qc_outputs(adata)
    log("computing log-normalized marker/module scores")
    logx = log_normalized_matrix(adata)
    add_module_scores(adata, logx)
    write_marker_detection(adata, logx)
    annotate_cells(adata)
    write_annotation_outputs(adata)
    log("writing updated h5ad with QC and annotation")
    adata.write_h5ad(H5AD, compression="gzip")
    log("running block-aware pseudobulk and controls")
    pseudobulk_and_controls(adata, logx)
    log("running conditional TED-Lite")
    run_ted_lite(adata)
    log("done")


if __name__ == "__main__":
    main()
