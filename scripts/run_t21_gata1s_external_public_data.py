"""Build public-data external validation tables for T21/GATA1s TED claims.

The workflow is intentionally conservative. It uses only downloaded public
processed data or small raw/processed bundles, records skipped large archives,
and labels every output with a claim ceiling so that directional support is not
misread as rescue-level validation.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import math
import re
import tarfile
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"scipy\.stats")

try:
    from scipy import stats
except Exception:  # pragma: no cover - scipy is present in the project env.
    stats = None


ROOT = Path("data_external/t21_gata1s_external_validation")
OUT = ROOT / "outputs"
MANIFESTS = ROOT / "manifests"
ANNOTATION = ROOT / "annotation"


HUMAN_ERYTHROID = [
    "GATA1",
    "KLF1",
    "TAL1",
    "ZFPM1",
    "EPOR",
    "GYPA",
    "GYPB",
    "GYPC",
    "HBB",
    "HBA1",
    "HBA2",
    "AHSP",
    "SLC4A1",
    "ANK1",
    "SPTA1",
    "SPTB",
    "EPB42",
    "RHAG",
    "TMOD1",
    "TFRC",
]

HUMAN_HEME = [
    "ALAS2",
    "ALAD",
    "HMBS",
    "UROS",
    "UROD",
    "CPOX",
    "PPOX",
    "FECH",
    "BLVRB",
    "SLC25A37",
    "ABCB10",
    "STEAP3",
]

HUMAN_APOPTOSIS = [
    "BAX",
    "BAK1",
    "BCL2",
    "BCL2L1",
    "MCL1",
    "CASP3",
    "CASP7",
    "CASP8",
    "CASP9",
    "PMAIP1",
    "BBC3",
    "FAS",
    "TNFRSF10B",
]

HUMAN_IMMATURE = [
    "KIT",
    "CD34",
    "PROM1",
    "GATA2",
    "RUNX1",
    "MYB",
    "SPI1",
    "LMO2",
    "MECOM",
]

HUMAN_CHR21 = [
    "RUNX1",
    "DYRK1A",
    "ERG",
    "ETS2",
    "HMGN1",
    "CHAF1B",
    "SON",
    "U2AF1",
    "RCAN1",
    "IFNAR1",
    "IFNAR2",
    "IFNGR2",
    "IL10RB",
    "CBS",
]

HUMAN_GLYCOLYSIS = [
    "PKM",
    "HK1",
    "HK2",
    "PFKP",
    "PFKL",
    "ALDOA",
    "GAPDH",
    "PGK1",
    "ENO1",
    "ENO2",
    "LDHA",
    "SLC2A1",
]

ORTHOLOGS = {
    "GATA1": "Gata1",
    "KLF1": "Klf1",
    "TAL1": "Tal1",
    "ZFPM1": "Zfpm1",
    "EPOR": "Epor",
    "GYPA": "Gypa",
    "GYPB": "Gypb",
    "GYPC": "Gypc",
    "HBB": "Hbb-bt",
    "HBA1": "Hba-a1",
    "HBA2": "Hba-a2",
    "AHSP": "Ahsp",
    "SLC4A1": "Slc4a1",
    "ANK1": "Ank1",
    "SPTA1": "Spta1",
    "SPTB": "Sptb",
    "EPB42": "Epb42",
    "RHAG": "Rhag",
    "TMOD1": "Tmod1",
    "TFRC": "Tfrc",
    "ALAS2": "Alas2",
    "ALAD": "Alad",
    "HMBS": "Hmbs",
    "UROS": "Uros",
    "UROD": "Urod",
    "CPOX": "Cpox",
    "PPOX": "Ppox",
    "FECH": "Fech",
    "BLVRB": "Blvrb",
    "SLC25A37": "Slc25a37",
    "ABCB10": "Abcb10",
    "STEAP3": "Steap3",
    "KIT": "Kit",
    "CD34": "Cd34",
    "PROM1": "Prom1",
    "GATA2": "Gata2",
    "RUNX1": "Runx1",
    "MYB": "Myb",
    "SPI1": "Spi1",
    "LMO2": "Lmo2",
    "MECOM": "Mecom",
    "PKM": "Pkm",
    "HK1": "Hk1",
    "HK2": "Hk2",
    "PFKP": "Pfkp",
    "PFKL": "Pfkl",
    "ALDOA": "Aldoa",
    "GAPDH": "Gapdh",
    "PGK1": "Pgk1",
    "ENO1": "Eno1",
    "LDHA": "Ldha",
    "SLC2A1": "Slc2a1",
}

MOUSE_ERYTHROID = [ORTHOLOGS[g] for g in HUMAN_ERYTHROID if g in ORTHOLOGS]
MOUSE_HEME = [ORTHOLOGS[g] for g in HUMAN_HEME if g in ORTHOLOGS]
MOUSE_IMMATURE = [ORTHOLOGS[g] for g in HUMAN_IMMATURE if g in ORTHOLOGS]
MOUSE_GLYCOLYSIS = [ORTHOLOGS[g] for g in HUMAN_GLYCOLYSIS if g in ORTHOLOGS]

MODULES_HUMAN = {
    "heme_metabolism": HUMAN_HEME,
    "erythroid_output": HUMAN_ERYTHROID,
    "heme_erythroid_combined": sorted(set(HUMAN_HEME + HUMAN_ERYTHROID)),
    "apoptosis": HUMAN_APOPTOSIS,
    "immature_regulatory": HUMAN_IMMATURE,
    "chr21_context": HUMAN_CHR21,
    "glycolysis": HUMAN_GLYCOLYSIS,
}

MODULES_MOUSE = {
    "heme_metabolism": MOUSE_HEME,
    "erythroid_output": MOUSE_ERYTHROID,
    "heme_erythroid_combined": sorted(set(MOUSE_HEME + MOUSE_ERYTHROID)),
    "immature_regulatory": MOUSE_IMMATURE,
    "glycolysis": MOUSE_GLYCOLYSIS,
}


def log(message: str) -> None:
    print(f"[t21-gata1s-public] {message}", flush=True)


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def write_file_inventory() -> None:
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and "outputs" not in path.parts:
            rows.append(
                {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_tsv(pd.DataFrame(rows), MANIFESTS / "downloaded_input_file_inventory.tsv")


def clean_symbol(value: object) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    symbol = str(value).strip()
    if not symbol or symbol in {"---", "NA", "nan"}:
        return None
    return symbol


def load_gene_info(path: Path) -> pd.DataFrame:
    cols = ["tax_id", "GeneID", "Symbol", "LocusTag", "Synonyms", "dbXrefs", "chromosome"]
    df = pd.read_csv(path, sep="\t", dtype=str)
    df = df.rename(columns={"#tax_id": "tax_id"})
    df = df[[c for c in cols if c in df.columns]]
    df["GeneID"] = df["GeneID"].astype(str)
    return df


def parse_geo_series_matrix(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_rows: dict[str, list[str]] = defaultdict(list)
    expr_lines: list[str] = []
    in_table = False
    with gzip.open(path, "rt", errors="replace", newline="") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if in_table:
                if line:
                    expr_lines.append(line)
                continue
            if line.startswith("!Sample_"):
                parts = next(csv.reader([line], delimiter="\t"))
                metadata_rows[parts[0]].append(parts[1:])

    expr = pd.DataFrame()
    if expr_lines:
        expr = pd.read_csv(io.StringIO("\n".join(expr_lines)), sep="\t", dtype=str)
        expr.columns = [str(c).strip().strip('"') for c in expr.columns]
        expr["ID_REF"] = expr["ID_REF"].astype(str).str.strip().str.strip('"')
        for col in expr.columns[1:]:
            expr[col] = pd.to_numeric(expr[col], errors="coerce")
        expr = expr.set_index("ID_REF")

    sample_ids = []
    if "!Sample_geo_accession" in metadata_rows:
        sample_ids = metadata_rows["!Sample_geo_accession"][0]
    elif not expr.empty:
        sample_ids = list(expr.columns)

    sample_records = {sid: {"sample": sid} for sid in sample_ids}
    repeated_counts: dict[str, int] = defaultdict(int)
    for key, value_lists in metadata_rows.items():
        short = key.replace("!Sample_", "")
        for values in value_lists:
            field = short
            if field in repeated_counts:
                repeated_counts[field] += 1
                field = f"{field}_{repeated_counts[short]}"
            else:
                repeated_counts[field] = 0
            for sid, value in zip(sample_ids, values):
                sample_records.setdefault(sid, {"sample": sid})[field] = value
    meta = pd.DataFrame(sample_records.values())
    return expr, meta


def parse_platform_symbols(soft_path: Path) -> dict[str, list[str]]:
    in_table = False
    header: list[str] | None = None
    mapping: dict[str, set[str]] = defaultdict(set)
    with gzip.open(soft_path, "rt", errors="replace", newline="") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("!platform_table_begin"):
                in_table = True
                header = None
                continue
            if line.startswith("!platform_table_end"):
                break
            if not in_table:
                continue
            parts = next(csv.reader([line], delimiter="\t"))
            if header is None:
                header = parts
                continue
            if not header or len(parts) < 1:
                continue
            row = dict(zip(header, parts))
            probe = str(row.get("ID", "")).strip()
            symbols: set[str] = set()
            for col in ["Gene Symbol", "GENE_SYMBOL", "gene_symbol", "Gene symbol"]:
                symbol = clean_symbol(row.get(col))
                if symbol:
                    for token in re.split(r"\s*///\s*|\s*//\s*|,\s*|;\s*", symbol):
                        token = clean_symbol(token)
                        if token:
                            symbols.add(token)
            assignment = row.get("gene_assignment")
            if assignment:
                for entry in str(assignment).split(" /// "):
                    chunks = [x.strip() for x in entry.split(" // ")]
                    if len(chunks) >= 2:
                        symbol = clean_symbol(chunks[1])
                        if symbol:
                            symbols.add(symbol)
            for symbol in symbols:
                if re.search(r"[A-Za-z]", symbol):
                    mapping[probe].add(symbol)
    return {probe: sorted(symbols) for probe, symbols in mapping.items()}


def collapse_probe_matrix(expr: pd.DataFrame, probe_to_symbols: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    map_rows = []
    for probe, values in expr.iterrows():
        symbols = probe_to_symbols.get(str(probe), [])
        if not symbols:
            continue
        for symbol in symbols:
            row = values.copy()
            row.name = symbol
            records.append(row)
            map_rows.append(
                {
                    "probe_id": probe,
                    "symbol": symbol,
                    "probe_mean_expression": float(pd.to_numeric(values, errors="coerce").mean()),
                }
            )
    if not records:
        return pd.DataFrame(), pd.DataFrame(map_rows)
    stacked = pd.DataFrame(records)
    stacked.index.name = "symbol"
    stacked["__mean"] = stacked.mean(axis=1, numeric_only=True)
    selected = stacked.reset_index().sort_values(["symbol", "__mean"], ascending=[True, False])
    selected = selected.drop_duplicates("symbol").set_index("symbol").drop(columns=["__mean"])
    selected.index = selected.index.astype(str)
    return selected, pd.DataFrame(map_rows)


def normalize_gse238115_group(value: str) -> str:
    value = value.lower()
    if "stag2" in value or "sa2" in value:
        return "T21_GATA1s_STAG2null"
    if "trisomy 21-gata1s" in value:
        return "T21_GATA1s"
    if "euploid-gata1s" in value:
        return "Euploid_GATA1s"
    if "trisomy 21" in value:
        return "T21"
    if "euploid" in value:
        return "Euploid"
    return "unknown"


def normalize_gse36787_group(value: str) -> str:
    v = value.lower()
    if "t21/gata1s" in v:
        return "T21_GATA1s"
    if "t21/wtgata1" in v:
        return "T21_wtGATA1"
    if "euploid/gata1s" in v:
        return "Euploid_GATA1s"
    if "euploid/wtgata1" in v:
        return "Euploid_wtGATA1"
    return "unknown"


def safe_ttest(case: pd.Series, control: pd.Series) -> float:
    # This workflow is a directional dry-lab screen, not a differential
    # expression reanalysis. Keep p-values blank to avoid implying an
    # edgeR/limma-grade model from these lightweight matrices.
    return math.nan
    a = pd.to_numeric(case, errors="coerce").dropna()
    b = pd.to_numeric(control, errors="coerce").dropna()
    if stats is None or len(a) < 2 or len(b) < 2:
        return math.nan
    _, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return float(p) if np.isfinite(p) else math.nan


def gene_contrast_table(expr: pd.DataFrame, sample_groups: dict[str, str], contrasts: list[tuple[str, str, str]]) -> pd.DataFrame:
    rows = []
    for contrast, case_group, control_group in contrasts:
        case_samples = [s for s, g in sample_groups.items() if g == case_group and s in expr.columns]
        control_samples = [s for s, g in sample_groups.items() if g == control_group and s in expr.columns]
        for gene, values in expr.iterrows():
            case = values[case_samples]
            control = values[control_samples]
            rows.append(
                {
                    "contrast": contrast,
                    "gene": gene,
                    "case_group": case_group,
                    "control_group": control_group,
                    "n_case": len(case_samples),
                    "n_control": len(control_samples),
                    "mean_case": float(pd.to_numeric(case, errors="coerce").mean()),
                    "mean_control": float(pd.to_numeric(control, errors="coerce").mean()),
                    "logFC_or_delta": float(pd.to_numeric(case, errors="coerce").mean() - pd.to_numeric(control, errors="coerce").mean()),
                    "welch_p": safe_ttest(case, control),
                }
            )
    return pd.DataFrame(rows)


def module_scorecard(gene_contrasts: pd.DataFrame, modules: dict[str, list[str]], expected: dict[tuple[str, str], str] | None = None) -> pd.DataFrame:
    expected = expected or {}
    rows = []
    available = set(gene_contrasts["gene"].astype(str))
    for contrast, cdf in gene_contrasts.groupby("contrast"):
        for module, genes in modules.items():
            present = [g for g in genes if g in available]
            sub = cdf[cdf["gene"].isin(present)]
            if sub.empty:
                rows.append(
                    {
                        "contrast": contrast,
                        "module": module,
                        "n_genes_scored": 0,
                        "genes_scored": "",
                        "mean_delta": math.nan,
                        "median_delta": math.nan,
                        "fraction_up": math.nan,
                        "fraction_down": math.nan,
                        "expected_direction": expected.get((contrast, module), "not_prespecified"),
                        "direction_call": "not_scored",
                        "success": "not_scored",
                    }
                )
                continue
            vals = pd.to_numeric(sub["logFC_or_delta"], errors="coerce").dropna()
            mean_delta = float(vals.mean())
            direction = "up" if mean_delta > 0 else "down" if mean_delta < 0 else "flat"
            exp = expected.get((contrast, module), "not_prespecified")
            success = "not_prespecified"
            if exp in {"up", "down"}:
                success = "pass" if direction == exp else "fail"
            rows.append(
                {
                    "contrast": contrast,
                    "module": module,
                    "n_genes_scored": len(sub),
                    "genes_scored": ",".join(sorted(sub["gene"].astype(str))),
                    "mean_delta": mean_delta,
                    "median_delta": float(vals.median()),
                    "fraction_up": float((vals > 0).mean()),
                    "fraction_down": float((vals < 0).mean()),
                    "expected_direction": exp,
                    "direction_call": direction,
                    "success": success,
                }
            )
    return pd.DataFrame(rows)


def read_gse238115_counts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tar_path = ROOT / "GSE238115/processed/GSE238115_RAW.tar"
    series_path = ROOT / "GSE238115/metadata/GSE238115_series_matrix.txt.gz"
    _, meta = parse_geo_series_matrix(series_path)
    records = []
    with tarfile.open(tar_path) as tf:
        for member in tf.getmembers():
            if "_counts_" not in member.name or not member.name.endswith(".txt.gz"):
                continue
            sample = member.name.split("_")[0]
            payload = tf.extractfile(member).read()
            df = pd.read_csv(io.BytesIO(gzip.decompress(payload)), sep="\t", dtype={"GeneID": str})
            value_col = df.columns[-1]
            records.append(df[["GeneID", value_col]].rename(columns={value_col: sample}).set_index("GeneID"))
    counts = pd.concat(records, axis=1).fillna(0)
    counts = counts.loc[:, sorted(counts.columns)]
    gene_info = load_gene_info(ANNOTATION / "Homo_sapiens.gene_info.gz")
    gene_info = gene_info[["GeneID", "Symbol", "chromosome"]].drop_duplicates("GeneID")
    counts = counts.reset_index().merge(gene_info, on="GeneID", how="left")
    counts["symbol"] = counts["Symbol"].fillna(counts["GeneID"])
    sample_cols = [c for c in counts.columns if c.startswith("GSM")]
    numeric = counts[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    library = numeric.sum(axis=0)
    logcpm = np.log2(numeric.div(library, axis=1) * 1_000_000 + 1)
    logcpm["symbol"] = counts["symbol"].values
    gene_logcpm = logcpm.groupby("symbol")[sample_cols].mean()
    counts_out = counts[["GeneID", "Symbol", "chromosome"] + sample_cols]
    meta["group"] = "unknown"
    char_cols = [c for c in meta.columns if c.startswith("characteristics")]
    for idx, row in meta.iterrows():
        joined = " ".join(str(row.get(c, "")) for c in char_cols)
        meta.loc[idx, "group"] = normalize_gse238115_group(joined)
    return counts_out, gene_logcpm, meta


def run_gse238115() -> None:
    log("scoring GSE238115")
    counts, expr, meta = read_gse238115_counts()
    processed_dir = ROOT / "GSE238115/processed"
    counts.to_csv(processed_dir / "GSE238115_combined_raw_counts_by_entrez.tsv.gz", sep="\t", index=False, compression="gzip")
    expr.reset_index().to_csv(processed_dir / "GSE238115_logCPM_by_gene_symbol.tsv.gz", sep="\t", index=False, compression="gzip")
    write_tsv(meta, ROOT / "GSE238115/metadata/GSE238115_sample_metadata_parsed.tsv")

    sample_groups = dict(zip(meta["sample"], meta["group"]))
    contrasts = [
        ("T21_GATA1s_vs_T21", "T21_GATA1s", "T21"),
        ("Euploid_GATA1s_vs_Euploid", "Euploid_GATA1s", "Euploid"),
        ("T21_vs_Euploid", "T21", "Euploid"),
        ("T21_GATA1s_STAG2null_vs_T21_GATA1s", "T21_GATA1s_STAG2null", "T21_GATA1s"),
    ]
    gene_contrasts = gene_contrast_table(expr, sample_groups, contrasts)
    gene_contrasts.to_csv(processed_dir / "GSE238115_gene_level_logCPM_contrasts.tsv.gz", sep="\t", index=False, compression="gzip")

    expected = {
        ("T21_GATA1s_vs_T21", "heme_metabolism"): "down",
        ("T21_GATA1s_vs_T21", "erythroid_output"): "down",
        ("T21_GATA1s_vs_T21", "heme_erythroid_combined"): "down",
        ("T21_GATA1s_vs_T21", "apoptosis"): "down",
        ("T21_vs_Euploid", "chr21_context"): "up",
    }
    score = module_scorecard(gene_contrasts, MODULES_HUMAN, expected)
    score["dataset"] = "GSE238115"
    score["claim_ceiling"] = "external_count_level_T21_GATA1s_directional_support;not_rescue;below_Level_4"
    write_tsv(score, OUT / "gse238115_external_t21_gata1s_scorecard.tsv")

    axis_genes = sorted(set(HUMAN_HEME + HUMAN_ERYTHROID))
    write_tsv(gene_contrasts[gene_contrasts["gene"].isin(axis_genes)], OUT / "gse238115_heme_erythroid_axis_direction.tsv")
    write_tsv(gene_contrasts[gene_contrasts["gene"].isin(HUMAN_CHR21)], OUT / "gse238115_chr21_context_axis.tsv")
    write_tsv(
        pd.DataFrame(
            [
                {
                    "dataset": "GSE238115",
                    "claim": "external_count_level_T21_GATA1s_directional_support",
                    "allowed_claim_ceiling": "below_Level_4",
                    "not_allowed_claims": "rescue;causal_rescue;single_cell_state_transition_proof",
                    "basis": "human iPSC-derived HPC bulk RNA-seq counts; T21/GATA1s versus T21 direction scored prospectively",
                }
            ]
        ),
        OUT / "gse238115_claim_ceiling.tsv",
    )


def run_gse36787() -> None:
    log("scoring GSE36787")
    expr, meta = parse_geo_series_matrix(ROOT / "GSE36787/metadata/GSE36787_series_matrix.txt.gz")
    probe_map = parse_platform_symbols(ROOT / "GSE36787/metadata/GSE36787_family.soft.gz")
    gene_expr, map_df = collapse_probe_matrix(expr, probe_map)
    processed_dir = ROOT / "GSE36787/processed"
    gene_expr.reset_index().to_csv(processed_dir / "GSE36787_gene_symbol_matrix.tsv.gz", sep="\t", index=False, compression="gzip")
    write_tsv(map_df, processed_dir / "GSE36787_probe_to_symbol_mapping.tsv")
    char_cols = [c for c in meta.columns if c.startswith("characteristics")] + ["title"]
    meta["group"] = [
        normalize_gse36787_group(" ".join(str(row.get(c, "")) for c in char_cols)) for _, row in meta.iterrows()
    ]
    write_tsv(meta, ROOT / "GSE36787/metadata/GSE36787_sample_metadata_parsed.tsv")
    sample_groups = dict(zip(meta["sample"], meta["group"]))
    contrasts = [
        ("T21_GATA1s_vs_T21_wtGATA1", "T21_GATA1s", "T21_wtGATA1"),
        ("Euploid_GATA1s_vs_Euploid_wtGATA1", "Euploid_GATA1s", "Euploid_wtGATA1"),
        ("T21_wtGATA1_vs_Euploid_wtGATA1", "T21_wtGATA1", "Euploid_wtGATA1"),
    ]
    gene_contrasts = gene_contrast_table(gene_expr, sample_groups, contrasts)

    pivot = gene_contrasts.pivot(index="gene", columns="contrast", values="logFC_or_delta")
    if {"T21_GATA1s_vs_T21_wtGATA1", "Euploid_GATA1s_vs_Euploid_wtGATA1"}.issubset(pivot.columns):
        interaction = (
            pivot["T21_GATA1s_vs_T21_wtGATA1"] - pivot["Euploid_GATA1s_vs_Euploid_wtGATA1"]
        ).dropna()
        interaction_df = pd.DataFrame(
            {
                "contrast": "factorial_interaction_T21_context_x_GATA1s",
                "gene": interaction.index,
                "case_group": "interaction",
                "control_group": "interaction",
                "n_case": math.nan,
                "n_control": math.nan,
                "mean_case": math.nan,
                "mean_control": math.nan,
                "logFC_or_delta": interaction.values,
                "welch_p": math.nan,
            }
        )
        gene_contrasts = pd.concat([gene_contrasts, interaction_df], ignore_index=True)
    gene_contrasts.to_csv(processed_dir / "GSE36787_gene_level_contrasts.tsv.gz", sep="\t", index=False, compression="gzip")

    expected = {
        ("T21_GATA1s_vs_T21_wtGATA1", "heme_metabolism"): "down",
        ("T21_GATA1s_vs_T21_wtGATA1", "erythroid_output"): "down",
        ("T21_GATA1s_vs_T21_wtGATA1", "heme_erythroid_combined"): "down",
        ("T21_wtGATA1_vs_Euploid_wtGATA1", "chr21_context"): "up",
    }
    score = module_scorecard(gene_contrasts, MODULES_HUMAN, expected)
    score["dataset"] = "GSE36787"
    score["claim_ceiling"] = "direct_human_iPSC_T21_GATA1s_proxy;not_independent_rescue;Level_3_directional_support_if_passes"
    write_tsv(score, OUT / "gse36787_t21_gata1s_factorial_proxy.tsv")
    write_tsv(gene_contrasts[gene_contrasts["gene"].isin(sorted(set(HUMAN_HEME + HUMAN_ERYTHROID)))], OUT / "gse36787_erythroid_event_family_direction.tsv")
    write_tsv(
        score[score["contrast"].isin(["T21_GATA1s_vs_T21_wtGATA1", "T21_wtGATA1_vs_Euploid_wtGATA1"])],
        OUT / "gse36787_gata1s_main_vs_t21_context.tsv",
    )
    write_tsv(
        pd.DataFrame(
            [
                {
                    "dataset": "GSE36787",
                    "claim": "direct_human_iPSC_T21_GATA1s_proxy",
                    "allowed_claim_ceiling": "Level_3_directional_support_if_passes",
                    "not_allowed_claims": "independent_rescue;Level_4_rescue;single_cell_transition_claim",
                    "basis": "human iPSC-derived CD43+/41+/235+ progenitor expression array; T21/wtGATA1 and T21/GATA1s plus euploid controls",
                }
            ]
        ),
        OUT / "gse36787_claim_ceiling.tsv",
    )


def run_gse62879() -> None:
    log("scoring GSE62879")
    expr = pd.read_csv(ROOT / "GSE62879/processed/GSE62879_Normalized_data_new_and_reanalyzed_samples.txt.gz", sep="\t", dtype={"ID_REF": str})
    expr = expr.set_index("ID_REF")
    for col in expr.columns:
        expr[col] = pd.to_numeric(expr[col], errors="coerce")
    probe_map = parse_platform_symbols(ROOT / "GSE62879/metadata/GSE62879_family.soft.gz")
    gene_expr, map_df = collapse_probe_matrix(expr, probe_map)
    processed_dir = ROOT / "GSE62879/processed"
    gene_expr.reset_index().to_csv(processed_dir / "GSE62879_gene_symbol_matrix.tsv.gz", sep="\t", index=False, compression="gzip")
    write_tsv(map_df, processed_dir / "GSE62879_probe_to_symbol_mapping.tsv")
    groups = {sample: ("GATA1s" if sample.startswith("GSM153") else "GATA1fl") for sample in gene_expr.columns}
    gene_contrasts = gene_contrast_table(gene_expr, groups, [("GATA1s_vs_GATA1fl", "GATA1s", "GATA1fl")])
    gene_contrasts.to_csv(processed_dir / "GSE62879_gene_level_contrasts.tsv.gz", sep="\t", index=False, compression="gzip")
    expected = {
        ("GATA1s_vs_GATA1fl", "heme_metabolism"): "down",
        ("GATA1s_vs_GATA1fl", "erythroid_output"): "down",
        ("GATA1s_vs_GATA1fl", "heme_erythroid_combined"): "down",
    }
    score = module_scorecard(gene_contrasts, MODULES_MOUSE, expected)
    score["dataset"] = "GSE62879"
    score["claim_ceiling"] = "isoform-level erythroid-output support;not_T21_specific;not_Level_4"
    write_tsv(score, OUT / "gse62879_gata1fl_vs_gata1s_isoform_scorecard.tsv")
    write_tsv(gene_contrasts[gene_contrasts["gene"].isin(sorted(set(MOUSE_HEME + MOUSE_ERYTHROID)))], OUT / "gse62879_erythroid_output_direction.tsv")
    mapping_rows = [
        {
            "human_symbol": human,
            "mouse_symbol": mouse,
            "mapping_basis": "manual_symbol_ortholog_seed_for_targeted_axis",
            "present_in_gse62879_matrix": mouse in gene_expr.index,
        }
        for human, mouse in ORTHOLOGS.items()
    ]
    write_tsv(pd.DataFrame(mapping_rows), OUT / "gse62879_mouse_to_human_ortholog_mapping.tsv")
    write_tsv(
        pd.DataFrame(
            [
                {
                    "dataset": "GSE62879_plus_reanalyzed_GSE14980",
                    "claim": "isoform-level erythroid-output support",
                    "allowed_claim_ceiling": "mechanistic_directional_support_not_T21_specific",
                    "not_allowed_claims": "T21_specific_validation;rescue;Level_4",
                    "basis": "mouse G1ME GATA1fl versus GATA1s reanalyzed same-platform microarray",
                }
            ]
        ),
        OUT / "gse62879_claim_ceiling.tsv",
    )


def parse_gse130156_columns(columns: list[str]) -> dict[str, str]:
    groups = {}
    for col in columns:
        if col == "symbol":
            continue
        genotype = "Gata1s_KO" if "_KO_" in col else "WT" if "_WT_" in col else "unknown"
        population = "R2" if "_R2_" in col else "R3" if "_R3_" in col else "unknown"
        stage = "12.5d" if "_12.5d" in col else "14.5d" if "_14.5d" in col else "unknown"
        groups[col] = f"{genotype}_{population}_{stage}"
    return groups


def run_gse130156() -> None:
    log("scoring GSE130156")
    df = pd.read_excel(ROOT / "GSE130156/processed/GSE130156_Normalized_count.xlsx", sheet_name=0)
    df = df.rename(columns={df.columns[0]: "symbol"})
    sample_cols = [c for c in df.columns if c != "symbol"]
    expr = df.set_index("symbol")[sample_cols].apply(pd.to_numeric, errors="coerce")
    expr = np.log2(expr + 1)
    groups = parse_gse130156_columns(sample_cols)
    aggregate_groups = {s: ("Gata1s_KO" if g.startswith("Gata1s_KO") else "WT") for s, g in groups.items()}
    gene_contrast_parts = [
        gene_contrast_table(expr, aggregate_groups, [("Gata1s_KO_vs_WT_all", "Gata1s_KO", "WT")])
    ]
    contrasts = []
    for population in ["R2", "R3"]:
        for stage in ["12.5d", "14.5d"]:
            case = f"Gata1s_KO_{population}_{stage}"
            control = f"WT_{population}_{stage}"
            if case in set(groups.values()) and control in set(groups.values()):
                contrasts.append((f"Gata1s_KO_vs_WT_{population}_{stage}", case, control))
    gene_contrast_parts.append(gene_contrast_table(expr, groups, contrasts))
    gene_contrasts = pd.concat(gene_contrast_parts, ignore_index=True)
    processed_dir = ROOT / "GSE130156/processed"
    gene_contrasts.to_csv(processed_dir / "GSE130156_gene_level_log_normalized_contrasts.tsv.gz", sep="\t", index=False, compression="gzip")
    expected = {
        ("Gata1s_KO_vs_WT_all", "heme_metabolism"): "down",
        ("Gata1s_KO_vs_WT_all", "erythroid_output"): "down",
        ("Gata1s_KO_vs_WT_all", "heme_erythroid_combined"): "down",
        ("Gata1s_KO_vs_WT_all", "immature_regulatory"): "up",
    }
    score = module_scorecard(gene_contrasts, MODULES_MOUSE, expected)
    score["dataset"] = "GSE130156"
    score["claim_ceiling"] = "mouse_N_terminus_mechanism_support;not_human_T21_rescue"
    write_tsv(score, OUT / "gse130156_gata1s_mouse_erythroid_scorecard.tsv")
    write_tsv(score[score["module"].isin(["immature_regulatory", "heme_erythroid_combined", "erythroid_output"])], OUT / "gse130156_immature_axis_vs_output_axis.tsv")
    axis = ["Gata2", "Runx1", "Kit", "Myb", "Spi1", "Lmo2", "Tal1", "Klf1", "Alas2", "Slc4a1"]
    write_tsv(gene_contrasts[gene_contrasts["gene"].isin(axis)], OUT / "gse130156_gata2_runx1_axis.tsv")
    write_tsv(
        pd.DataFrame(
            [
                {
                    "dataset": "GSE130156",
                    "claim": "mouse_N_terminus_mechanism_support",
                    "allowed_claim_ceiling": "mechanistic_support_for_erythroid_maturation_and_immature_axis",
                    "not_allowed_claims": "human_T21_specific_validation;rescue;Level_4",
                    "basis": "mouse Gata1s fetal liver erythroid RNA-seq normalized count table",
                }
            ]
        ),
        OUT / "gse130156_claim_ceiling.tsv",
    )


def run_gse315981() -> None:
    log("scoring GSE315981")
    df = pd.read_csv(ROOT / "GSE315981/processed/GSE315981_TPM_combined.tsv.gz", sep="\t")
    sample_cols = [c for c in df.columns if c not in {"geneID", "geneSymbol"}]
    expr = df.rename(columns={"geneSymbol": "symbol"}).set_index("symbol")[sample_cols]
    expr = np.log2(expr.apply(pd.to_numeric, errors="coerce") + 1)
    groups = {}
    for sample in sample_cols:
        lower = sample.lower()
        state = "differentiation" if lower.startswith("267078") or lower.startswith("diff") or "_diff" in lower else "expansion"
        iso = "GATA1FL" if "fl1" in lower else "GATA1s"
        groups[sample] = f"{state}_{iso}"
    contrasts = [
        ("differentiation_GATA1s_vs_GATA1FL", "differentiation_GATA1s", "differentiation_GATA1FL"),
        ("expansion_GATA1s_vs_GATA1FL", "expansion_GATA1s", "expansion_GATA1FL"),
    ]
    gene_contrasts = gene_contrast_table(expr, groups, contrasts)
    processed_dir = ROOT / "GSE315981/processed"
    gene_contrasts.to_csv(processed_dir / "GSE315981_gene_level_TPM_contrasts.tsv.gz", sep="\t", index=False, compression="gzip")
    expected = {
        ("differentiation_GATA1s_vs_GATA1FL", "heme_metabolism"): "down",
        ("differentiation_GATA1s_vs_GATA1FL", "erythroid_output"): "down",
        ("differentiation_GATA1s_vs_GATA1FL", "glycolysis"): "up",
    }
    score = module_scorecard(gene_contrasts, MODULES_HUMAN, expected)
    score["dataset"] = "GSE315981"
    score["claim_ceiling"] = "human_HUDEP1_GATA1_N_terminus_bulk_support;not_T21_specific;not_rescue"
    write_tsv(score, OUT / "gse315981_gata1s_bulk_direction_scorecard.tsv")
    probe = pd.DataFrame(
        [
            {
                "dataset": "GSE315981",
                "file": str((ROOT / "GSE315981/processed/GSE315981_TPM_combined.tsv.gz").resolve()),
                "data_type": "bulk RNA-seq TPM",
                "n_genes": df.shape[0],
                "n_samples": len(sample_cols),
                "groups_detected": ",".join(sorted(set(groups.values()))),
                "analysis_status": "scored_directionally",
            }
        ]
    )
    write_tsv(probe, OUT / "gse315981_bulk_rnaseq_file_probe.tsv")


def file_probe_from_filelist(acc: str, data_type: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            [{"dataset": acc, "data_type": data_type, "analysis_status": "filelist_missing"}]
        )
    df = pd.read_csv(path, sep="\t")
    df["dataset"] = acc
    df["data_type"] = data_type
    df["analysis_status"] = "file_probe_only_large_archive_not_downloaded"
    return df


def run_family_audits() -> None:
    log("writing family/file-probe audits")
    decisions = pd.read_csv(MANIFESTS / "download_decisions.tsv", sep="\t")
    accessions = ["GSE316151", "GSE298761", "GSE315981", "GSE315985", "GSE315986"]
    rows = []
    for acc in accessions:
        subset = decisions[decisions["accession"] == acc]
        rows.append(
            {
                "accession": acc,
                "downloaded_files": int((subset["status"] == "downloaded").sum()),
                "reused_existing_files": int((subset["status"] == "reused_existing_local_file").sum()),
                "skipped_files": int((subset["status"] == "skipped").sum()),
                "available_processed_or_matrix": ",".join(
                    subset[
                        subset["name"].str.contains("TPM|count|matrix|series_matrix", case=False, regex=True)
                    ]["name"].astype(str)
                ),
                "interpretation": {
                    "GSE316151": "PRO-seq/bigWig branch; mechanism support only unless bigWig-level analysis is added",
                    "GSE298761": "scRNA-seq branch; filelist confirms 10x matrix files but 940MB archive was not downloaded in this pass",
                    "GSE315981": "bulk RNA-seq branch; TPM table downloaded and scored",
                    "GSE315985": "mouse CUT&RUN branch; filelist confirms bigWig files, archive skipped",
                    "GSE315986": "human CUT&RUN branch; metadata/filelist reused from previous local download, archive skipped",
                }[acc],
            }
        )
    write_tsv(pd.DataFrame(rows), OUT / "gse316151_family_accession_audit.tsv")
    write_tsv(file_probe_from_filelist("GSE298761", "mouse fetal liver scRNA-seq 10x matrix", ROOT / "GSE298761/metadata/GSE298761_filelist.txt"), OUT / "gse298761_scrna_file_probe.tsv")
    write_tsv(file_probe_from_filelist("GSE315985", "mouse CUT&RUN bigWig", ROOT / "GSE315985/metadata/GSE315985_filelist.txt"), OUT / "gse315985_cutrun_file_probe.tsv")
    gse315986_filelist = Path("data_external/prospective_validation/GSE315986/metadata/GSE315986_filelist.txt")
    if gse315986_filelist.exists():
        write_tsv(file_probe_from_filelist("GSE315986", "human CUT&RUN bigWig", gse315986_filelist), OUT / "gse315986_cutrun_file_probe.tsv")
    write_tsv(
        pd.DataFrame(
            [
                {
                    "dataset_family": "GSE316151/GSE298761/GSE315981/GSE315985/GSE315986",
                    "claim": "GATA1_N_terminus_metabolic_and_erythroid_mechanism_support",
                    "allowed_claim_ceiling": "mechanistic_support;not_human_T21_rescue",
                    "not_allowed_claims": "Level_4_rescue;T21_GATA1s_causal_rescue",
                    "basis": "family audit plus GSE315981 bulk TPM score; scRNA/CUT&RUN branches require optional large-file analysis",
                }
            ]
        ),
        OUT / "gse316151_claim_ceiling.tsv",
    )


def run_restoration_probe() -> None:
    log("writing restoration-series probes")
    decisions = pd.read_csv(MANIFESTS / "download_decisions.tsv", sep="\t")
    accessions = ["GSE40522", "GSE51338", "GSE36029", "GSE49847"]
    rows = []
    for acc in accessions:
        subset = decisions[decisions["accession"] == acc]
        rows.append(
            {
                "dataset": acc,
                "downloaded_or_reused_small_files": int(subset["status"].isin(["downloaded", "reused_existing_local_file"]).sum()),
                "raw_archive_status": ";".join(subset[subset["name"].str.endswith("RAW.tar")]["decision"].astype(str)),
                "processed_matrix_files": ",".join(subset[subset["subdir"] == "matrix"]["name"].astype(str)),
                "analysis_status": "metadata_or_series_matrix_probe_only",
                "directional_claim": "not_scored_as_rescue_in_this_pass",
                "reason": "restoration evidence requires targeted reconstruction of time/induction contrasts from large raw archives or multi-platform matrices",
            }
        )
    probe = pd.DataFrame(rows)
    write_tsv(probe, OUT / "gse40522_gata1_restoration_direction.tsv")
    write_tsv(probe, OUT / "gse40522_early_late_gata1_targets.tsv")
    write_tsv(probe, OUT / "gse40522_ted_predicted_rescue_direction.tsv")
    write_tsv(
        pd.DataFrame(
            [
                {
                    "dataset_group": "GSE40522/GSE51338/GSE36029/GSE49847",
                    "claim": "positive_restoration_direction_reference_pending_reconstruction",
                    "allowed_claim_ceiling": "reference_only_until_processed_contrast_is_rebuilt",
                    "not_allowed_claims": "external_rescue_support;Level_4",
                    "basis": "metadata and series matrix/filelist downloaded or reused; very large raw archives skipped",
                }
            ]
        ),
        OUT / "gse40522_claim_ceiling.tsv",
    )


def run_gse32388_context() -> None:
    log("scoring GSE32388 context probe")
    expr, meta = parse_geo_series_matrix(ROOT / "GSE32388/metadata/GSE32388_series_matrix.txt.gz")
    probe_map = parse_platform_symbols(ROOT / "GSE32388/metadata/GSE32388_family.soft.gz")
    gene_expr, map_df = collapse_probe_matrix(expr, probe_map)
    processed_dir = ROOT / "GSE32388/processed"
    if not gene_expr.empty:
        gene_expr.reset_index().to_csv(processed_dir / "GSE32388_gene_symbol_matrix.tsv.gz", sep="\t", index=False, compression="gzip")
        write_tsv(map_df, processed_dir / "GSE32388_probe_to_symbol_mapping.tsv")
    # The four arrays are a narrow leukemia-context series; preserve it as a
    # context-only claim rather than erythroid validation.
    write_tsv(
        pd.DataFrame(
            [
                {
                    "dataset": "GSE32388",
                    "analysis_status": "context_only_not_erythroid_validation",
                    "n_samples": expr.shape[1],
                    "n_gene_symbols_mapped": int(gene_expr.shape[0]) if not gene_expr.empty else 0,
                    "allowed_claim": "contextual_DS_AMKL_GATA1s_biology",
                    "not_allowed_claims": "erythroid_event_loss_validation;T21_iPSC_proxy;rescue",
                }
            ]
        ),
        OUT / "gse32388_ds_amkl_context_scorecard.tsv",
    )
    write_tsv(
        pd.DataFrame(
            [
                {
                    "dataset": "GSE32388",
                    "claim": "contextual_DS_AMKL_GATA1s_biology",
                    "allowed_claim_ceiling": "context_only",
                    "not_allowed_claims": "erythroid_validation;rescue;Level_4",
                    "basis": "DS-AMKL CMK GATA1/GATA1s knockdown microarray, not an erythroid differentiation model",
                }
            ]
        ),
        OUT / "gse32388_claim_ceiling.tsv",
    )


def main() -> None:
    ensure_dirs()
    write_file_inventory()
    run_gse238115()
    run_gse36787()
    run_gse62879()
    run_gse130156()
    run_gse315981()
    run_family_audits()
    run_restoration_probe()
    run_gse32388_context()
    log(f"done: outputs in {OUT.resolve()}")


if __name__ == "__main__":
    main()
