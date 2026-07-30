from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import io, sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyfgsea.nearest_method_benchmark import (
    ARTIFACTS,
    RawCountDesign,
    evaluate_common_task,
    pathway_scores,
    score_then_smooth_common_task,
    simulate_raw_count_dataset,
)


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    values = p[finite]
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    out[finite] = restored
    return out


def profile(name: str) -> dict[str, int]:
    if name == "smoke":
        return {"n_cells": 240, "n_genes": 800, "n_pathways": 30, "size_min": 20, "size_max": 45}
    if name == "development":
        return {"n_cells": 600, "n_genes": 2_000, "n_pathways": 30, "size_min": 30, "size_max": 70}
    return {"n_cells": 2_000, "n_genes": 5_000, "n_pathways": 30, "size_min": 30, "size_max": 80}


def generic_characteristics(counts, cells, pathways, method: str) -> pd.DataFrame:
    out = score_then_smooth_common_task(counts, cells, pathways).copy()
    out["method"] = method
    return out


def expression_matched_adapter(
    native: pd.DataFrame,
    counts: sparse.csr_matrix,
    cells: pd.DataFrame,
    pathways: dict[str, np.ndarray],
    method: str,
    *,
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    gene_names = np.array([f"gene_{i + 1:05d}" for i in range(counts.shape[1])])
    native = native.set_index("gene").reindex(gene_names)
    if method == "tradeSeq":
        raw = -np.log10(np.clip(pd.to_numeric(native["p_value"], errors="coerce").to_numpy(), 1e-300, 1.0))
    else:
        raw = pd.to_numeric(native["native_score"], errors="coerce").to_numpy()
    raw[~np.isfinite(raw)] = np.nanmin(raw[np.isfinite(raw)]) if np.isfinite(raw).any() else 0.0
    percentile = pd.Series(raw).rank(method="average", pct=True).to_numpy(dtype=float)

    mean_expression = np.asarray(counts.mean(axis=0)).ravel()
    bins = pd.qcut(pd.Series(mean_expression).rank(method="first"), q=10, labels=False).to_numpy()
    pools = {int(b): np.flatnonzero(bins == b) for b in np.unique(bins)}
    rng = np.random.default_rng(seed)
    rows = []
    for pathway, genes in pathways.items():
        genes = np.asarray(genes, dtype=int)
        observed = float(np.mean(percentile[genes]))
        null_sum = np.zeros(permutations, dtype=float)
        for b in bins[genes]:
            pool = pools[int(b)]
            null_sum += percentile[rng.choice(pool, size=permutations, replace=True)]
        null = null_sum / len(genes)
        p_value = float((1 + np.sum(null >= observed)) / (permutations + 1))
        rows.append({"pathway": pathway, "ranking_score": observed, "p_value": p_value})
    adapted = pd.DataFrame(rows)
    adapted["q_value"] = bh_adjust(adapted["p_value"].to_numpy())
    curve = generic_characteristics(counts, cells, pathways, method)
    keep = ["pathway", "direction", "event_mode", "event_center", "event_width", "status"]
    adapted = adapted.merge(curve[keep], on="pathway", how="left", validate="one_to_one")
    adapted["method"] = method
    adapted["event_detected"] = adapted["q_value"] <= 0.10
    adapted["formal_p_value_available"] = True
    adapted["adapter"] = "expression_matched_rank_auc"
    adapted["adapter_permutations"] = permutations
    return adapted


def tips_predictions(native: pd.DataFrame, counts, cells, pathways) -> pd.DataFrame:
    curve = generic_characteristics(counts, cells, pathways, "TIPS")
    keep = ["pathway", "direction", "event_mode", "event_center", "event_width"]
    out = native.merge(curve[keep], on="pathway", how="left", validate="one_to_one")
    out = out.rename(columns={"native_score": "ranking_score"})
    out["method"] = "TIPS"
    out["event_detected"] = out["q_value"] <= 0.10
    out["formal_p_value_available"] = True
    out["adapter"] = "native_pathway_score_common_curve_characterization"
    return out


def ted_predictions(counts, cells, pathways, out_dir: Path, seed: int) -> pd.DataFrame:
    import anndata as ad
    from pyfgsea.trajectory import run_trajectory_gsea

    totals = np.asarray(counts.sum(axis=1)).ravel()
    scale = np.divide(10_000.0, totals, out=np.zeros_like(totals, dtype=float), where=totals > 0)
    normalized = counts.multiply(scale[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    adata = ad.AnnData(normalized)
    adata.var_names = [f"gene_{i + 1:05d}" for i in range(counts.shape[1])]
    adata.obs_names = cells["cell_id"].astype(str).tolist()
    adata.obs["ordered_coordinate"] = cells["ordered_coordinate"].to_numpy(dtype=float)
    gmt = out_dir / "pathways.gmt"
    with gmt.open("w", encoding="utf-8") as handle:
        for name, genes in pathways.items():
            members = [adata.var_names[int(g)] for g in genes]
            handle.write("\t".join([name, "common_task"] + members) + "\n")
    window_size = max(40, counts.shape[0] // 5)
    step = max(20, window_size // 2)
    windows = run_trajectory_gsea(
        adata,
        str(gmt),
        pseudotime_key="ordered_coordinate",
        window_size=window_size,
        step=step,
        min_size=15,
        max_size=500,
        sample_size=101,
        seed=seed,
        nperm_nes=100,
        ranker="mean_diff",
        calculate_nes=True,
        use_nes_cache=False,
    )
    write_tsv(windows, out_dir / "ted_native_windows.tsv")
    curve = generic_characteristics(counts, cells, pathways, "TED")
    rows = []
    for pathway, group in windows.groupby("Pathway", sort=False):
        score = float(np.nanmax(np.abs(pd.to_numeric(group["NES"], errors="coerce"))))
        p_col = "padj" if "padj" in group else ("pval" if "pval" in group else None)
        p_value = float(np.nanmin(pd.to_numeric(group[p_col], errors="coerce"))) if p_col else np.nan
        rows.append({"pathway": pathway, "ranking_score": score, "p_value": p_value})
    out = pd.DataFrame(rows)
    out["q_value"] = np.nan
    keep = ["pathway", "direction", "event_mode", "event_center", "event_width", "status"]
    out = out.merge(curve[keep], on="pathway", how="right", validate="one_to_one")
    out["method"] = "TED"
    out["event_detected"] = out["p_value"] <= 0.05
    out["formal_p_value_available"] = False
    out["adapter"] = "native_ted_window_score_common_task_summary"
    return out


def run_command(command: list[str], cwd: Path, timeout: int) -> float:
    started = time.perf_counter()
    subprocess.run(command, cwd=cwd, check=True, timeout=timeout)
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "development", "locked"], default="smoke")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "ted_nearest_method_five_method")
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--artifact", choices=list(ARTIFACTS), default="composition")
    parser.add_argument("--coordinate", choices=["true", "noisy"], default="noisy")
    parser.add_argument("--signal", choices=["low", "high"], default="low")
    parser.add_argument("--blocks", choices=[4, 6, 10], type=int, default=4)
    parser.add_argument("--adapter-permutations", type=int, default=10_000)
    parser.add_argument("--r-image", default="ted-nearest-methods:20260717")
    parser.add_argument("--sctransient-python", type=Path, default=ROOT / ".venv-sctransient" / "Scripts" / "python.exe")
    parser.add_argument("--method-timeout", type=int, default=7_200)
    args = parser.parse_args()

    dims = profile(args.profile)
    scenario_name = f"{args.blocks}b_{args.signal}_{args.coordinate}_{args.artifact}_seed{args.seed}"
    out = (args.output_dir / args.profile / scenario_name).resolve()
    out.mkdir(parents=True, exist_ok=True)
    design = RawCountDesign(
        n_blocks=args.blocks,
        n_cells=dims["n_cells"],
        n_genes=dims["n_genes"],
        n_pathways=dims["n_pathways"],
        pathway_size_min=dims["size_min"],
        pathway_size_max=dims["size_max"],
        signal_strength=args.signal,
        coordinate_quality=args.coordinate,
        artifact=args.artifact,
        seed=args.seed,
    )
    dataset = simulate_raw_count_dataset(design)
    public_cells = dataset.cells.drop(columns=["true_time_private"])
    sparse.save_npz(out / "raw_counts.npz", dataset.counts)
    io.mmwrite(out / "raw_counts.mtx", dataset.counts)
    write_tsv(public_cells, out / "cell_metadata.tsv")
    membership = pd.DataFrame(
        [{"pathway": name, "gene_index": int(gene)} for name, genes in dataset.pathways.items() for gene in genes]
    )
    write_tsv(membership, out / "pathway_membership.tsv")

    runtime_rows = []
    started = time.perf_counter()
    score_predictions = score_then_smooth_common_task(dataset.counts, public_cells, dataset.pathways)
    runtime_rows.append({"method": "score_then_smooth", "elapsed_seconds": time.perf_counter() - started})

    started = time.perf_counter()
    ted = ted_predictions(dataset.counts, public_cells, dataset.pathways, out, args.seed)
    runtime_rows.append({"method": "TED", "elapsed_seconds": time.perf_counter() - started})

    sc_elapsed = run_command(
        [str(args.sctransient_python), str(ROOT / "scripts" / "run_sctransient_common_task.py"), str(out)],
        ROOT,
        args.method_timeout,
    )
    runtime_rows.append({"method": "scTransient_total", "elapsed_seconds": sc_elapsed})

    relative_out = out.relative_to(ROOT).as_posix()
    r_elapsed = run_command(
        [
            "docker", "run", "--rm", "-v", f"{ROOT.as_posix()}:/workspace", "-w", "/workspace",
            args.r_image, "Rscript", "scripts/run_nearest_method_r_methods.R", f"/workspace/{relative_out}",
        ],
        ROOT,
        args.method_timeout,
    )
    runtime_rows.append({"method": "R_methods_total", "elapsed_seconds": r_elapsed})

    sc_native = pd.read_csv(out / "sctransient_native_gene.tsv", sep="\t")
    trade_native = pd.read_csv(out / "tradeseq_native_gene.tsv", sep="\t")
    tips_native = pd.read_csv(out / "tips_native.tsv", sep="\t")
    sc = expression_matched_adapter(
        sc_native, dataset.counts, public_cells, dataset.pathways, "scTransient",
        permutations=args.adapter_permutations, seed=args.seed + 11,
    )
    trade = expression_matched_adapter(
        trade_native, dataset.counts, public_cells, dataset.pathways, "tradeSeq",
        permutations=args.adapter_permutations, seed=args.seed + 17,
    )
    tips = tips_predictions(tips_native, dataset.counts, public_cells, dataset.pathways)
    predictions = pd.concat([tips, sc, trade, score_predictions, ted], ignore_index=True, sort=False)
    write_tsv(predictions, out / "five_method_predictions.tsv")
    write_tsv(dataset.truth, out / "truth.tsv")

    metrics = pd.concat(
        [evaluate_common_task(group.copy(), dataset.truth) for _, group in predictions.groupby("method", sort=False)],
        ignore_index=True,
    )
    write_tsv(metrics, out / "five_method_metrics.tsv")
    runtimes = pd.DataFrame(runtime_rows)
    for extra in ["r_method_runtime.tsv", "sctransient_runtime.tsv"]:
        path = out / extra
        if path.exists():
            runtimes = pd.concat([runtimes, pd.read_csv(path, sep="\t")], ignore_index=True)
    write_tsv(runtimes, out / "five_method_runtime.tsv")

    scenario = dict(dataset.scenario)
    scenario.update(
        {
            "profile": args.profile,
            "adapter_permutations": args.adapter_permutations,
            "r_image": args.r_image,
            "python": platform.python_version(),
            "methods_completed": sorted(predictions["method"].unique().tolist()),
        }
    )
    (out / "scenario.json").write_text(json.dumps(scenario, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(f"outputs={out}")


if __name__ == "__main__":
    main()
