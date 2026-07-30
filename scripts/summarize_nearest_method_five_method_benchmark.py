from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def markdown_table(frame: pd.DataFrame) -> str:
    values = frame.copy()
    for column in values.columns:
        if pd.api.types.is_numeric_dtype(values[column]):
            values[column] = values[column].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    header = "| " + " | ".join(map(str, values.columns)) + " |"
    divider = "| " + " | ".join(["---"] * len(values.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in values.itertuples(index=False, name=None)]
    return "\n".join([header, divider] + rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/ted_nearest_method_five_method/development"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ted_nearest_method_five_method/pilot_summary"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metric_frames = []
    runtime_frames = []
    failure_rows = []
    for metric_path in sorted(args.root.glob("*/five_method_metrics.tsv")):
        scenario_dir = metric_path.parent
        scenario = json.loads((scenario_dir / "scenario.json").read_text(encoding="utf-8"))
        scenario_id = scenario_dir.name
        common = {
            "scenario_id": scenario_id,
            "n_blocks": scenario["n_blocks"],
            "signal_strength": scenario["signal_strength"],
            "coordinate_quality": scenario["coordinate_quality"],
            "artifact": scenario["artifact"],
            "seed": scenario["seed"],
        }

        metrics = pd.read_csv(metric_path, sep="\t")
        for key, value in common.items():
            metrics[key] = value
        metric_frames.append(metrics)

        runtime = pd.read_csv(scenario_dir / "five_method_runtime.tsv", sep="\t")
        runtime = runtime[runtime["method"].isin(["TIPS", "tradeSeq", "scTransient", "TED", "score_then_smooth"])]
        for key, value in common.items():
            runtime[key] = value
        runtime_frames.append(runtime)

        predictions = pd.read_csv(scenario_dir / "five_method_predictions.tsv", sep="\t")
        for method, group in predictions.groupby("method", sort=False):
            failure_rows.append(
                {
                    **common,
                    "method": method,
                    "pathways": len(group),
                    "failed_pathways": int((group["status"] != "ok").sum()),
                    "pathway_failure_rate": float((group["status"] != "ok").mean()),
                }
            )

    if not metric_frames:
        raise SystemExit(f"no completed scenarios found below {args.root}")

    scenario_metrics = pd.concat(metric_frames, ignore_index=True)
    scenario_runtime = pd.concat(runtime_frames, ignore_index=True)
    failures = pd.DataFrame(failure_rows)
    scenario_metrics.to_csv(args.output_dir / "scenario_metrics.tsv", sep="\t", index=False)
    scenario_runtime.to_csv(args.output_dir / "scenario_runtime.tsv", sep="\t", index=False)
    failures.to_csv(args.output_dir / "pathway_failures.tsv", sep="\t", index=False)

    numeric_metrics = [
        "pathway_level_auprc",
        "top_k_precision",
        "top_k_recall",
        "ndcg",
        "artifact_only_false_call_rate",
    ]
    method_summary = scenario_metrics.groupby("method", sort=False)[numeric_metrics].agg(
        ["mean", "std", "min", "max"]
    )
    method_summary.columns = [f"{metric}_{stat}" for metric, stat in method_summary.columns]
    method_summary = method_summary.reset_index()
    method_summary.insert(1, "scenario_count", scenario_metrics.groupby("method").size().reindex(method_summary["method"]).to_numpy())
    method_summary.to_csv(args.output_dir / "method_summary.tsv", sep="\t", index=False)

    runtime_summary = scenario_runtime.groupby("method", sort=False)["elapsed_seconds"].agg(
        ["mean", "std", "min", "max"]
    ).reset_index()
    runtime_summary.to_csv(args.output_dir / "runtime_summary.tsv", sep="\t", index=False)

    lines = [
        "# Five-method common-task development pilot",
        "",
        f"Completed scenarios: {scenario_metrics['scenario_id'].nunique()}.",
        "Methods: TIPS, scTransient, tradeSeq, score-then-smooth, and TED common-task output.",
        "",
        "This is a development pilot, not the prespecified 48-cell by 10-seed locked benchmark.",
        "TIPS uses its pathway-specific Monocle2/DDRTree score with exact igraph argument-name compatibility patches; dispersion estimation is skipped because current dplyr removed group_by_().",
        "scTransient and tradeSeq use a frozen expression-matched pathway adapter with 10,000 random sets.",
        "Direction and event-centre fields use the same common curve characterizer for all methods and are therefore not method-discriminating metrics in this pilot.",
        "The TED row summarizes native TED rolling-window output but does not execute the full E/V evidence or artifact-gating contract; artifact false-call rates must not be interpreted as E-level false promotion.",
        "",
        "## Mean performance across the four pilot scenarios",
        "",
        markdown_table(method_summary[["method", "pathway_level_auprc_mean", "top_k_precision_mean", "ndcg_mean", "artifact_only_false_call_rate_mean"]]),
        "",
        "## Mean native method runtime (seconds)",
        "",
        markdown_table(runtime_summary[["method", "mean", "min", "max"]]),
        "",
    ]
    (args.output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")

    manifest_rows = []
    for path in sorted(args.output_dir.iterdir()):
        if not path.is_file() or path.name == "manifest.tsv":
            continue
        manifest_rows.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(args.output_dir / "manifest.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
