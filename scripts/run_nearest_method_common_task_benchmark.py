from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyfgsea.nearest_method_benchmark import (
    ARTIFACTS,
    RawCountDesign,
    evaluate_common_task,
    realized_pairwise_overlap,
    score_then_smooth_common_task,
    simulate_raw_count_dataset,
)


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile(name: str) -> dict[str, int]:
    if name == "smoke":
        return {"n_cells": 240, "n_genes": 800, "n_pathways": 30, "size_min": 20, "size_max": 45}
    if name == "development":
        return {"n_cells": 600, "n_genes": 2_000, "n_pathways": 30, "size_min": 30, "size_max": 70}
    return {"n_cells": 2_000, "n_genes": 5_000, "n_pathways": 30, "size_min": 30, "size_max": 80}


def availability() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"method": "TIPS", "native_level": "pathway", "status": "not_run", "reason": "native package not present in locked baseline container"},
            {"method": "scTransient", "native_level": "gene/protein feature", "status": "not_run", "reason": "native implementation not present in locked baseline container"},
            {"method": "tradeSeq", "native_level": "gene", "status": "available", "reason": "tradeSeq 1.24.0 present in ted-baselines:1.0.0; shared-task adapter not yet frozen"},
            {"method": "score_then_smooth", "native_level": "pathway score", "status": "executed", "reason": "internal minimal common-task baseline"},
            {"method": "TED", "native_level": "downstream pathway event", "status": "not_run", "reason": "raw-count common-task adapter is not yet frozen; E/V fields excluded from comparison"},
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "development", "locked"], default="smoke")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "ted_nearest_method_common_task")
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--artifact", choices=list(ARTIFACTS) + ["complete_batch_time"], default="composition")
    parser.add_argument("--coordinate", choices=["true", "noisy"], default="noisy")
    parser.add_argument("--signal", choices=["low", "high"], default="low")
    parser.add_argument("--blocks", choices=[4, 6, 10], type=int, default=4)
    args = parser.parse_args()

    dims = profile(args.profile)
    scenario_name = (
        f"{args.blocks}b_{args.signal}_{args.coordinate}_{args.artifact}_seed{args.seed}"
    )
    out = args.output_dir / args.profile / scenario_name
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

    started = time.perf_counter()
    dataset = simulate_raw_count_dataset(design)
    # Method-facing files omit private true time and all truth labels.
    sparse.save_npz(out / "raw_counts.npz", dataset.counts)
    public_cells = dataset.cells.drop(columns=["true_time_private"])
    write_tsv(public_cells, out / "cell_metadata.tsv")
    pathway_rows = [
        {"pathway": name, "gene_index": int(gene)}
        for name, genes in dataset.pathways.items()
        for gene in genes
    ]
    write_tsv(pd.DataFrame(pathway_rows), out / "pathway_membership.tsv")

    predictions = score_then_smooth_common_task(dataset.counts, public_cells, dataset.pathways)
    write_tsv(predictions, out / "common_task_predictions.tsv")
    # Development profiles are unmasked only after the method output exists.
    if args.profile == "locked":
        metrics = pd.DataFrame(
            [{"method": "score_then_smooth", "status": "locked_truth_masked", "pathway_level_auprc": np.nan}]
        )
        truth_status = "masked; final test not run"
    else:
        write_tsv(dataset.truth, out / "development_truth.tsv")
        metrics = evaluate_common_task(predictions, dataset.truth)
        truth_status = "development truth unmasked after predictions were written"
    write_tsv(metrics, out / "common_task_metrics.tsv")
    write_tsv(availability(), out / "method_availability.tsv")

    scenario = dict(dataset.scenario)
    scenario.update(
        {
            "profile": args.profile,
            "realized_max_pairwise_pathway_jaccard": realized_pairwise_overlap(dataset.pathways),
            "truth_status": truth_status,
            "elapsed_seconds": time.perf_counter() - started,
            "python": platform.python_version(),
        }
    )
    (out / "scenario.json").write_text(json.dumps(scenario, indent=2), encoding="utf-8")
    files = [p for p in out.iterdir() if p.is_file()]
    manifest = pd.DataFrame(
        [{"file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(files)]
    )
    write_tsv(manifest, out / "manifest.tsv")
    print(json.dumps(scenario, indent=2))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
