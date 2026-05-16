#!/usr/bin/env python
"""Stratified streamed TED-lite pass for GSE199308.

This improves on the earlier 50k prefix pass by selecting a balanced sample
across Mutant, embryo_id, Background, and RT_group within a larger streaming
window. It remains an embryo-block/pseudobulk result, not a full atlas or
cell-type-specific fate analysis.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data_external" / "ted_first_candidate_downloads" / "GSE199308"
MTX = DATA_ROOT / "GSE199308_gene_count.mtx.gz"
CELL_META = DATA_ROOT / "GSE199308_cell_annotate.csv.gz"
GENE_META = DATA_ROOT / "GSE199308_gene_annotate.csv.gz"
OUTDIR = ROOT / "data_external" / "ted_generalization_panel" / "GSE199308"
PREFIX_DIR = OUTDIR
BASE_SCRIPT = ROOT / "scripts" / "run_gse199308_mutant_pleiotropy_ted_lite.py"


def load_base_module():
    spec = importlib.util.spec_from_file_location("gse199308_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()


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


def select_stratified_cells(max_scan_columns: int, min_cells_per_mutant: int, seed: int) -> pd.DataFrame:
    meta = pd.read_csv(CELL_META)
    meta["matrix_col"] = np.arange(1, len(meta) + 1)
    meta = meta[meta["Mutant"].ne("poorly_assigning") & meta["matrix_col"].le(max_scan_columns)].copy()
    selected = []
    strata_cols = ["embryo_id", "Background", "RT_group"]
    for mutant, group in meta.groupby("Mutant", sort=True):
        target = min(len(group), min_cells_per_mutant)
        shuffled = group.sample(frac=1.0, random_state=seed + stable_int(mutant)).copy()
        shuffled["_stratum"] = shuffled[strata_cols].astype(str).agg("|".join, axis=1)
        shuffled["_rank_in_stratum"] = shuffled.groupby("_stratum").cumcount()
        take = shuffled.sort_values(["_rank_in_stratum", "_stratum", "matrix_col"]).head(target)
        selected.append(take.drop(columns=["_stratum", "_rank_in_stratum"]))
    out = pd.concat(selected, ignore_index=True)
    return out.sort_values("matrix_col").reset_index(drop=True)


def stable_int(text: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(text))) % 100000


def stream_selected_pseudobulk(selected_meta: pd.DataFrame, gene_meta: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    embryos = list(dict.fromkeys(selected_meta["embryo_id"].astype(str).tolist()))
    embryo_to_idx = {emb: i for i, emb in enumerate(embryos)}
    col_to_embryo = dict(
        zip(
            selected_meta["matrix_col"].astype(int).to_numpy(),
            selected_meta["embryo_id"].astype(str).map(embryo_to_idx).astype(int).to_numpy(),
        )
    )
    selected_cols = set(col_to_embryo)
    max_selected_col = max(selected_cols)
    n_genes = len(gene_meta)
    sums = np.zeros((n_genes, len(embryos)), dtype=np.float64)
    triplets_seen = 0
    triplets_used = 0
    max_col_seen = 0
    stopped_at_col = 0
    with gzip.open(MTX, "rb") as handle:
        for raw in handle:
            if raw.startswith(b"%"):
                continue
            dims = raw.split()
            if len(dims) >= 3:
                break
        for raw in handle:
            triplets_seen += 1
            parts = raw.split()
            if len(parts) < 3:
                continue
            col_i = int(parts[1])
            if col_i > max_selected_col:
                stopped_at_col = col_i
                break
            max_col_seen = col_i
            emb_i = col_to_embryo.get(col_i)
            if emb_i is None:
                continue
            gene_i = int(parts[0]) - 1
            sums[gene_i, emb_i] += float(parts[2])
            triplets_used += 1

    embryo_meta = (
        selected_meta.groupby("embryo_id", as_index=False)
        .agg(
            n_cells=("cell_id", "count"),
            mutant=("Mutant", "first"),
            background=("Background", "first"),
            mutant_type=("Mutant_type", "first"),
            rt_group=("RT_group", "first"),
            expr_id=("expr_id", "first"),
            min_matrix_col=("matrix_col", "min"),
            max_matrix_col=("matrix_col", "max"),
        )
        .rename(columns={"embryo_id": "embryo_id"})
    )
    embryo_meta["embryo_index"] = embryo_meta["embryo_id"].map(embryo_to_idx)
    embryo_meta = embryo_meta.sort_values("embryo_index").reset_index(drop=True)
    stats = {
        "selected_cells": len(selected_meta),
        "selected_mutants": selected_meta["Mutant"].nunique(),
        "selected_embryo_blocks": selected_meta["embryo_id"].nunique(),
        "selected_backgrounds": selected_meta["Background"].nunique(),
        "selected_RT_groups": selected_meta["RT_group"].nunique(),
        "min_matrix_col_selected": int(selected_meta["matrix_col"].min()),
        "max_matrix_col_selected": int(max_selected_col),
        "triplets_seen_until_stop": triplets_seen,
        "triplets_used": triplets_used,
        "max_col_seen": max_col_seen,
        "stopped_at_col": stopped_at_col,
    }
    return sums, embryo_meta, stats


def build_streaming_summary(selected_meta: pd.DataFrame, embryo_meta: pd.DataFrame, gene_meta: pd.DataFrame, sums: np.ndarray, stats: dict[str, object], args: argparse.Namespace) -> pd.DataFrame:
    rows = [
        {"metric": "sampling_strategy", "value": "stratified_by_Mutant_embryo_Background_RT_group", "interpretation": "avoids simple 50k prefix order bias"},
        {"metric": "max_scan_columns", "value": args.max_scan_columns, "interpretation": "streaming window, not full atlas"},
        {"metric": "min_cells_per_mutant_target", "value": args.min_cells_per_mutant, "interpretation": "mutants below target are fully sampled within scan window"},
        {"metric": "random_seed", "value": args.seed, "interpretation": "deterministic stratified selection"},
    ]
    rows.extend({"metric": key, "value": value, "interpretation": "stratified streamed MatrixMarket TED-lite"} for key, value in stats.items())
    rows.extend(
        [
            {"metric": "n_genes", "value": len(gene_meta), "interpretation": "gene annotation rows"},
            {"metric": "n_nonzero_embryo_gene_entries", "value": int((sums > 0).sum()), "interpretation": "dense embryo pseudobulk matrix nonzero entries"},
        ]
    )
    mutant_counts = selected_meta.groupby("Mutant").size().sort_values(ascending=False)
    for mutant, n in mutant_counts.items():
        rows.append({"metric": f"selected_cells::{mutant}", "value": int(n), "interpretation": "stratified cells per mutant"})
    embryo_counts = embryo_meta[["embryo_id", "mutant", "n_cells"]].sort_values("n_cells", ascending=False)
    for _, row in embryo_counts.iterrows():
        rows.append({"metric": f"embryo_cells::{row['embryo_id']}", "value": int(row["n_cells"]), "interpretation": row["mutant"]})
    return pd.DataFrame(rows)


def compare_prefix_vs_stratified(strat_pleio: pd.DataFrame, strat_grammar: pd.DataFrame, strat_negative: pd.DataFrame) -> pd.DataFrame:
    prefix_pleio_path = PREFIX_DIR / "gse199308_pleiotropy_score.tsv"
    prefix_grammar_path = PREFIX_DIR / "gse199308_mutant_event_grammar.tsv"
    if not prefix_pleio_path.exists():
        return pd.DataFrame()
    prefix = pd.read_csv(prefix_pleio_path, sep="\t")
    prefix_grammar = pd.read_csv(prefix_grammar_path, sep="\t") if prefix_grammar_path.exists() else pd.DataFrame()
    p = prefix[["mutant", "pleiotropy_score", "event_grammar_class"]].rename(
        columns={"pleiotropy_score": "prefix_pleiotropy_score", "event_grammar_class": "prefix_event_grammar_class"}
    )
    s = strat_pleio[["mutant", "pleiotropy_score", "pleiotropy_score_sampling_adjusted", "sampling_confidence", "event_grammar_class"]].rename(
        columns={
            "pleiotropy_score": "stratified_pleiotropy_score",
            "pleiotropy_score_sampling_adjusted": "stratified_pleiotropy_score_sampling_adjusted",
            "event_grammar_class": "stratified_event_grammar_class",
        }
    )
    out = p.merge(s, on="mutant", how="outer")
    out["prefix_rank"] = out["prefix_pleiotropy_score"].rank(ascending=False, method="min")
    out["stratified_rank"] = out["stratified_pleiotropy_score_sampling_adjusted"].rank(ascending=False, method="min")
    out["rank_delta_stratified_minus_prefix"] = out["stratified_rank"] - out["prefix_rank"]
    out["event_class_same"] = out["prefix_event_grammar_class"].eq(out["stratified_event_grammar_class"])
    out["top10_in_both"] = out["prefix_rank"].le(10) & out["stratified_rank"].le(10)
    neg = strat_negative[["mutant", "claim_adjustment", "negative_control_margin"]].rename(
        columns={"claim_adjustment": "stratified_control_claim_adjustment"}
    )
    out = out.merge(neg, on="mutant", how="left")
    common = out[["prefix_pleiotropy_score", "stratified_pleiotropy_score_sampling_adjusted"]].dropna()
    rho = common["prefix_pleiotropy_score"].rank().corr(common["stratified_pleiotropy_score_sampling_adjusted"].rank()) if len(common) >= 3 else np.nan
    out["global_spearman_pleiotropy"] = rho
    out["top10_overlap_count"] = int(out["top10_in_both"].sum())
    out["comparison_interpretation"] = np.where(
        out["top10_in_both"],
        "top10_stable_between_prefix_and_stratified",
        "rank_or_class_changed_review_before_maintext",
    )
    return out.sort_values("stratified_rank", na_position="last")


def add_sampling_confidence(pleiotropy: pd.DataFrame, min_cells_per_mutant: int) -> pd.DataFrame:
    out = pleiotropy.copy()
    out["cell_sampling_confidence"] = np.minimum(1.0, pd.to_numeric(out["n_cells_in_streamed_prefix"], errors="coerce") / float(min_cells_per_mutant))
    out["block_sampling_confidence"] = np.minimum(1.0, pd.to_numeric(out["n_embryo_blocks"], errors="coerce") / 2.0)
    out["sampling_confidence"] = np.sqrt(out["cell_sampling_confidence"] * out["block_sampling_confidence"])
    out["pleiotropy_score_sampling_adjusted"] = out["pleiotropy_score"] * out["sampling_confidence"]
    out["sampling_confidence_interpretation"] = np.where(
        out["sampling_confidence"] >= 1.0,
        "meets_min_cells_and_block_confidence",
        "shrunk_for_low_cell_or_single_block_coverage",
    )
    return out.sort_values("pleiotropy_score_sampling_adjusted", ascending=False)


def build_control_sensitive_mutants(negative: pd.DataFrame, grammar: pd.DataFrame) -> pd.DataFrame:
    out = negative.merge(grammar[["mutant", "event_grammar_class", "max_abs_axis_free_delta"]], on="mutant", how="left")
    out["is_control_sensitive"] = out["claim_adjustment"].eq("downgrade_or_flag_control_sensitive")
    out["recommended_action"] = np.where(
        out["is_control_sensitive"],
        "downgrade event claim or require stronger null/block analysis",
        "retain as stratified TED-lite candidate",
    )
    return out.sort_values(["is_control_sensitive", "negative_control_margin"], ascending=[False, True])


def build_claim_update(summary: pd.DataFrame, comparison: pd.DataFrame, control_sensitive: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    top_overlap = int(comparison["top10_overlap_count"].dropna().iloc[0]) if not comparison.empty and "top10_overlap_count" in comparison else 0
    rho = float(comparison["global_spearman_pleiotropy"].dropna().iloc[0]) if not comparison.empty and comparison["global_spearman_pleiotropy"].notna().any() else np.nan
    n_control_sensitive = int(control_sensitive["is_control_sensitive"].sum()) if not control_sensitive.empty else 0
    if top_overlap >= 5 and (pd.isna(rho) or rho >= 0.35):
        stability = "prefix_vs_stratified_partially_stable"
        ceiling = "Level_3_stratified_streamed_embryo_pseudobulk_event_grammar_candidate"
    else:
        stability = "prefix_vs_stratified_requires_review"
        ceiling = "Level_2.5_stratified_streamed_screen_not_maintext_ready"
    return pd.DataFrame(
        [
            {
                "dataset": "GSE199308",
                "previous_claim_ceiling": "Level_2.5_to_3_prefix_streamed_embryo_pseudobulk_event_grammar",
                "updated_claim_ceiling": ceiling,
                "stability_status": stability,
                "top10_overlap_prefix_vs_stratified": top_overlap,
                "spearman_pleiotropy_prefix_vs_stratified": rho,
                "n_control_sensitive_mutants": n_control_sensitive,
                "streaming_scope": f"stratified sample within first {args.max_scan_columns} matrix columns; {args.min_cells_per_mutant} cells/mutant target",
                "allowed_claim": "GSE199308 supports TED generalization to mouse whole-embryo mutant phenotyping at stratified embryo-block/pseudobulk event-grammar level.",
                "forbidden_claim": "do not claim full-atlas result, cell-type-specific fate loss, developmental delay vs true loss, or functional rescue",
                "missing_evidence": "full matrix or multi-window sampler, cell-state annotation, mutant x cell-type event grammar, block permutation FDR",
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-scan-columns", type=int, default=400000)
    parser.add_argument("--min-cells-per-mutant", type=int, default=2000)
    parser.add_argument("--top-n-genes-per-module", type=int, default=250)
    parser.add_argument("--seed", type=int, default=271399)
    args = parser.parse_args()

    gene_meta = pd.read_csv(GENE_META)
    selected_meta = select_stratified_cells(args.max_scan_columns, args.min_cells_per_mutant, args.seed)
    sums, embryo_meta, stats = stream_selected_pseudobulk(selected_meta, gene_meta)
    logcpm = BASE.normalize_logcpm(sums)
    summary = build_streaming_summary(selected_meta, embryo_meta, gene_meta, sums, stats, args)

    curated_scores = {module: BASE.score_genes(logcpm, gene_meta, genes) for module, genes in BASE.CURATED_GENE_SETS.items()}
    axis_modules, axis_scores = BASE.discover_axis_free_modules(logcpm, gene_meta, embryo_meta, top_n=args.top_n_genes_per_module)
    all_scores = {**curated_scores, **axis_scores}
    effects = BASE.module_effects_from_scores(all_scores, embryo_meta)
    grammar = BASE.build_event_grammar(effects)
    pleiotropy = add_sampling_confidence(BASE.build_pleiotropy(grammar, effects, embryo_meta), args.min_cells_per_mutant)
    negative = BASE.build_negative_control(effects)
    concordance = BASE.build_concordance(effects)
    marker_dep = BASE.build_marker_dependency(axis_modules)
    comparison = compare_prefix_vs_stratified(pleiotropy, grammar, negative)
    control_sensitive = build_control_sensitive_mutants(negative, grammar)
    claim = build_claim_update(summary, comparison, control_sensitive, args)

    paths = [
        write_tsv(summary, "gse199308_stratified_streaming_summary.tsv"),
        write_tsv(comparison, "gse199308_prefix_vs_stratified_concordance.tsv"),
        write_tsv(effects, "gse199308_mutant_module_effects_stratified.tsv"),
        write_tsv(axis_modules, "gse199308_axis_free_modules_stratified.tsv"),
        write_tsv(grammar, "gse199308_mutant_event_grammar_stratified.tsv"),
        write_tsv(pleiotropy, "gse199308_pleiotropy_score_stratified.tsv"),
        write_tsv(negative, "gse199308_negative_control_audit_stratified.tsv"),
        write_tsv(control_sensitive, "gse199308_control_sensitive_mutants.tsv"),
        write_tsv(concordance, "gse199308_curated_vs_axisfree_concordance_stratified.tsv"),
        write_tsv(marker_dep, "gse199308_marker_dependency_audit_stratified.tsv"),
        write_tsv(claim, "gse199308_claim_ceiling_update.tsv"),
    ]
    manifest = pd.DataFrame({"output_file": [rel(p) for p in paths], "n_rows": [len(pd.read_csv(p, sep="\t")) for p in paths]})
    write_tsv(manifest, "gse199308_stratified_output_manifest.tsv")
    print(f"Wrote {len(paths) + 1} GSE199308 stratified streamed TED-lite files to {rel(OUTDIR)}")
    print(f"Selected cells={len(selected_meta)}; max selected col={stats['max_matrix_col_selected']}; triplets used={stats['triplets_used']}")


if __name__ == "__main__":
    main()
