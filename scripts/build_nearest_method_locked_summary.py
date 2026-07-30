from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOCKED_ROOT = ROOT / "results" / "ted_nearest_method_five_method" / "locked"
CONTROL_ROOT = ROOT / "results" / "ted_nearest_method_five_method" / "locked_grid_control"
SUMMARY_ROOT = ROOT / "results" / "ted_nearest_method_five_method" / "locked_summary"
MACHINE_ROOT = ROOT / "results" / "ted_manuscript_machine_readable_v2"
FIGURE_SOURCE_ROOT = ROOT / "results" / "ted_v1_submission" / "figure_source_data"

METHODS = ["TIPS", "scTransient", "tradeSeq", "score_then_smooth", "TED"]
METHOD_LABELS = ["TIPS", "scTransient", "tradeSeq", "Score + smooth", "TED"]
COLORS = ["#7A8B99", "#5B8FF9", "#61B15A", "#C28A2C", "#2F5D8A"]
NATIVE_FILES = {
    "TIPS": "tips_native.tsv",
    "scTransient": "sctransient_native_gene.tsv",
    "tradeSeq": "tradeseq_native_gene.tsv",
    "TED": "ted_native_windows.tsv",
}

FIGURE_TARGETS = [
    ROOT / "results" / "ted_v1_submission" / "figures",
    ROOT / "results" / "bib_manuscript_revision" / "figures",
    ROOT / "GenomeBiology_known_source_submission_package" / "01_main_manuscript" / "figures",
    ROOT / "GenomeBiology_known_source_submission_package" / "03_figures_final_upload",
    ROOT / "GenomeBiology_known_source_submission_package" / "03_figures",
    ROOT
    / "GenomeBiology_known_source_submission_package"
    / "06_latex_source"
    / "TED_GenomeBiology_Main_Manuscript_Only"
    / "figures",
    ROOT / "latex_submission_package" / "TED_GenomeBiology_LaTeX_submission" / "figures",
]
SOURCE_TARGETS = [
    ROOT / "GenomeBiology_known_source_submission_package" / "05_source_data_and_audits" / "figure_source_data",
    ROOT
    / "GenomeBiology_known_source_submission_package"
    / "06_latex_source"
    / "TED_GenomeBiology_Main_Manuscript_Only"
    / "tables"
    / "figure_source_data",
    ROOT / "latex_submission_package" / "TED_GenomeBiology_LaTeX_submission" / "tables" / "figure_source_data",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(score, dtype=float)
    finite = np.isfinite(s)
    y = y[finite]
    s = s[finite]
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ranked = y[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / positives)


def require_complete_scenarios(allow_partial: bool) -> list[Path]:
    registry = pd.read_csv(CONTROL_ROOT / "locked_task_registry.tsv", sep="\t")
    scenario_dirs: list[Path] = []
    missing: list[str] = []
    required = [
        "scenario.json",
        "truth.tsv",
        "five_method_predictions.tsv",
        "five_method_metrics.tsv",
        "five_method_runtime.tsv",
    ]
    for row in registry.itertuples(index=False):
        directory = LOCKED_ROOT / str(row.scenario_name)
        if all((directory / name).is_file() and (directory / name).stat().st_size > 0 for name in required):
            scenario_dirs.append(directory)
        else:
            missing.append(str(row.scenario_name))
    if missing and not allow_partial:
        raise SystemExit(
            f"locked benchmark incomplete: {len(scenario_dirs)}/480 tasks have validated output; "
            "wait for run_nearest_method_locked_grid.py before publishing"
        )
    if not scenario_dirs:
        raise SystemExit("no complete locked common-task scenarios found")
    return scenario_dirs


def add_context(frame: pd.DataFrame, scenario: dict[str, object], scenario_id: str) -> pd.DataFrame:
    out = frame.copy()
    out.insert(0, "scenario_id", scenario_id)
    out.insert(1, "n_blocks", int(scenario["n_blocks"]))
    out.insert(2, "signal_strength", str(scenario["signal_strength"]))
    out.insert(3, "coordinate_quality", str(scenario["coordinate_quality"]))
    out.insert(4, "artifact", str(scenario["artifact"]))
    out.insert(5, "seed", int(scenario["seed"]))
    return out


def event_type_metrics(predictions: pd.DataFrame, truth: pd.DataFrame, context: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    merged = predictions.merge(truth, on="pathway", suffixes=("_pred", "_truth"), validate="many_to_one")
    for method, group in merged.groupby("method", sort=False):
        k = int(group["is_dynamic"].astype(bool).sum())
        ranked = group[pd.to_numeric(group["ranking_score"], errors="coerce").notna()].sort_values(
            "ranking_score", ascending=False
        )
        selected_pathways = set(ranked.head(k)["pathway"].astype(str))
        artifact_targets = group["artifact_target"].astype(str).str.lower().eq("true") & ~group[
            "is_dynamic"
        ].astype(bool)
        target_names = group.loc[artifact_targets, "pathway"].astype(str)
        matched_top_k_artifact_rate = (
            float(target_names.isin(selected_pathways).mean()) if len(target_names) else float("nan")
        )
        for event_type in ["activation", "suppression", "transient"]:
            positive = group["event_mode_truth"].eq(event_type).to_numpy()
            scores = pd.to_numeric(group["ranking_score"], errors="coerce").to_numpy(dtype=float)
            detected = group["event_detected"].astype(str).str.lower().eq("true").to_numpy()
            selected = group.loc[positive]
            direction_accuracy = float(
                selected["direction_pred"].astype(str).eq(selected["direction_truth"].astype(str)).mean()
            )
            center_mae = float("nan")
            if event_type == "transient":
                center_mae = float(
                    np.nanmean(
                        np.abs(
                            pd.to_numeric(selected["event_center"], errors="coerce").to_numpy(dtype=float)
                            - pd.to_numeric(selected["true_center"], errors="coerce").to_numpy(dtype=float)
                        )
                    )
                )
            rows.append(
                {
                    **context,
                    "method": method,
                    "event_type": event_type,
                    "n_truth_pathways": int(positive.sum()),
                    "pathway_auprc": average_precision(positive, scores),
                    "truth_pathway_recall": float(detected[positive].mean()),
                    "direction_accuracy": direction_accuracy,
                    "transient_center_mae": center_mae,
                    "artifact_target_false_call_rate": float(
                        detected[group["artifact_target"].astype(str).str.lower().eq("true").to_numpy()].mean()
                    )
                    if group["artifact_target"].astype(str).str.lower().eq("true").any()
                    else float("nan"),
                    "matched_top_k_artifact_false_promotion_rate": matched_top_k_artifact_rate,
                }
            )
    return rows


def collect(scenario_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    truth_frames: list[pd.DataFrame] = []
    event_rows: list[dict[str, object]] = []
    native_rows: list[dict[str, object]] = []
    for directory in sorted(scenario_dirs):
        scenario = json.loads((directory / "scenario.json").read_text(encoding="utf-8"))
        scenario_id = directory.name
        context = {
            "scenario_id": scenario_id,
            "n_blocks": int(scenario["n_blocks"]),
            "signal_strength": str(scenario["signal_strength"]),
            "coordinate_quality": str(scenario["coordinate_quality"]),
            "artifact": str(scenario["artifact"]),
            "seed": int(scenario["seed"]),
        }
        metrics = pd.read_csv(directory / "five_method_metrics.tsv", sep="\t")
        predictions = pd.read_csv(directory / "five_method_predictions.tsv", sep="\t")
        truth = pd.read_csv(directory / "truth.tsv", sep="\t")
        fair_rates: dict[str, float] = {}
        merged_for_fairness = predictions.merge(
            truth[["pathway", "is_dynamic", "artifact_target"]],
            on="pathway",
            validate="many_to_one",
        )
        for method, group in merged_for_fairness.groupby("method", sort=False):
            k = int(group["is_dynamic"].astype(bool).sum())
            ranked = group[pd.to_numeric(group["ranking_score"], errors="coerce").notna()].sort_values(
                "ranking_score", ascending=False
            )
            selected = set(ranked.head(k)["pathway"].astype(str))
            target_names = group.loc[
                group["artifact_target"].astype(bool) & ~group["is_dynamic"].astype(bool), "pathway"
            ].astype(str)
            fair_rates[str(method)] = (
                float(target_names.isin(selected).mean()) if len(target_names) else float("nan")
            )
        metrics["matched_top_k_artifact_false_promotion_rate"] = metrics["method"].map(fair_rates)
        metrics = add_context(metrics, scenario, scenario_id)
        metric_frames.append(metrics)
        prediction_frames.append(add_context(predictions, scenario, scenario_id))
        exposed_truth = add_context(truth, scenario, scenario_id)
        exposed_truth["truth_access_policy"] = "masked_from_method_adapters_until_native_outputs_serialized"
        exposed_truth["truth_revealed_for_scoring"] = True
        truth_frames.append(exposed_truth)
        event_rows.extend(event_type_metrics(predictions, truth, context))

        for method in METHODS:
            if method == "score_then_smooth":
                path = directory / "five_method_predictions.tsv"
                selector = "method=score_then_smooth"
                native_kind = "harmonized direct score output"
            else:
                path = directory / NATIVE_FILES[method]
                selector = ""
                native_kind = "native method output"
            native_rows.append(
                {
                    **context,
                    "method": method,
                    "native_kind": native_kind,
                    "file": path.relative_to(ROOT).as_posix(),
                    "row_selector": selector,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )

    metrics_all = pd.concat(metric_frames, ignore_index=True)
    predictions_all = pd.concat(prediction_frames, ignore_index=True)
    truth_all = pd.concat(truth_frames, ignore_index=True)
    event_all = pd.DataFrame(event_rows)
    native_manifest = pd.DataFrame(native_rows)
    return metrics_all, predictions_all, truth_all, event_all, native_manifest


def scenario_registry() -> pd.DataFrame:
    registry = pd.read_csv(CONTROL_ROOT / "locked_task_registry.tsv", sep="\t")
    scenarios = (
        registry[["scenario_index", "blocks", "signal", "coordinate", "artifact"]]
        .drop_duplicates()
        .sort_values("scenario_index")
        .rename(
            columns={
                "blocks": "n_blocks",
                "signal": "signal_strength",
                "coordinate": "coordinate_quality",
            }
        )
    )
    scenarios["replicates"] = 10
    scenarios["seeds"] = "20262001-20262010"
    scenarios["n_cells"] = 2000
    scenarios["n_genes"] = 5000
    scenarios["n_pathways"] = 30
    scenarios["truth_masking"] = "private true time removed; truth written only after native outputs"
    return scenarios


def aggregate_event_metrics(event_all: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "pathway_auprc",
        "truth_pathway_recall",
        "direction_accuracy",
        "transient_center_mae",
        "artifact_target_false_call_rate",
        "matched_top_k_artifact_false_promotion_rate",
    ]
    grouped = event_all.groupby(["method", "event_type", "artifact"], sort=False)[numeric].agg(
        ["mean", "std", "count"]
    )
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def headline_figure_metrics(metrics: pd.DataFrame, event_metrics: pd.DataFrame) -> pd.DataFrame:
    """Return the exact aggregates plotted in Figure 3.

    This table keeps the visual summary independently auditable without asking a
    reader to reverse-engineer panel filters from the scenario-level file.
    """
    rows: list[dict[str, object]] = []
    clean = metrics[metrics["artifact"].eq("none") & metrics["coordinate_quality"].eq("true")]
    clean_by_event = event_metrics[event_metrics["artifact"].eq("none")]
    hard = metrics[
        metrics["signal_strength"].eq("low") & metrics["coordinate_quality"].eq("noisy")
    ]
    artifact = metrics[metrics["artifact"].isin(["composition", "stress", "partial_batch_time"])]
    for method in METHODS:
        method_clean = clean[clean["method"].eq(method)]
        for event_type in ["activation", "suppression", "transient"]:
            values = (
                pd.to_numeric(
                    clean_by_event.loc[
                        clean_by_event["method"].eq(method)
                        & clean_by_event["event_type"].eq(event_type),
                        "pathway_auprc",
                    ],
                    errors="coerce",
                )
                .dropna()
                .to_numpy(dtype=float)
            )
            rows.append(
                {
                    "panel": "A",
                    "method": method,
                    "stratum": f"artifact=none; event_type={event_type}",
                    "metric": "pathway_auprc",
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "n_tasks": int(len(values)),
                }
            )
        hard_values = (
            pd.to_numeric(
                hard.loc[hard["method"].eq(method), "pathway_level_auprc"], errors="coerce"
            )
            .dropna()
            .to_numpy(dtype=float)
        )
        rows.append(
            {
                "panel": "B",
                "method": method,
                "stratum": "signal=low; coordinate=noisy; artifacts=all",
                "metric": "pathway_level_auprc",
                "mean": float(np.mean(hard_values)),
                "sd": float(np.std(hard_values, ddof=1)),
                "n_tasks": int(len(hard_values)),
            }
        )
        method_artifact = artifact[artifact["method"].eq(method)]
        for artifact_name in ["composition", "stress", "partial_batch_time"]:
            selected = method_artifact[method_artifact["artifact"].eq(artifact_name)]
            values = (
                pd.to_numeric(selected["matched_top_k_artifact_false_promotion_rate"], errors="coerce")
                .dropna()
                .to_numpy(dtype=float)
            )
            rows.append(
                {
                    "panel": "C",
                    "method": method,
                    "stratum": artifact_name,
                    "metric": "matched_top_k_artifact_false_promotion_rate",
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "n_tasks": int(len(values)),
                }
            )
        for panel, selected, metric, stratum in (
            (
                "D_accuracy",
                method_clean,
                "pathway_level_auprc",
                "artifact=none; coordinate=true",
            ),
            (
                "D_risk",
                method_artifact,
                "matched_top_k_artifact_false_promotion_rate",
                "all_artifacts",
            ),
        ):
            values = pd.to_numeric(selected[metric], errors="coerce").dropna().to_numpy(dtype=float)
            rows.append(
                {
                    "panel": panel,
                    "method": method,
                    "stratum": stratum,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "n_tasks": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def s13_latex_rows(event_summary: pd.DataFrame) -> str:
    """Render the complete method x event-type x artifact table body."""
    method_labels = dict(zip(METHODS, METHOD_LABELS, strict=True))
    artifact_labels = {
        "none": "none",
        "composition": "composition",
        "stress": "stress",
        "partial_batch_time": "partial batch--time",
    }
    rows: list[str] = []
    for method_index, method in enumerate(METHODS):
        if method_index:
            rows.append(r"\midrule")
        for event_type in ["activation", "suppression", "transient"]:
            for artifact in ["none", "composition", "stress", "partial_batch_time"]:
                selected = event_summary[
                    event_summary["method"].eq(method)
                    & event_summary["event_type"].eq(event_type)
                    & event_summary["artifact"].eq(artifact)
                ]
                if len(selected) != 1:
                    raise ValueError(
                        f"expected one S13 row for {method}/{event_type}/{artifact}, got {len(selected)}"
                    )
                row = selected.iloc[0]

                def summary(metric: str) -> str:
                    mean = float(row[f"{metric}_mean"])
                    sd = float(row[f"{metric}_std"])
                    count = int(row[f"{metric}_count"])
                    if not np.isfinite(mean):
                        return "--"
                    return f"{mean:.3f} ({sd:.3f}); n={count}"

                rows.append(
                    " & ".join(
                        [
                            method_labels[method],
                            event_type,
                            artifact_labels[artifact],
                            summary("pathway_auprc"),
                            summary("truth_pathway_recall"),
                            summary("matched_top_k_artifact_false_promotion_rate"),
                        ]
                    )
                    + r" \\"
                )
    return "\n".join(rows) + "\n" + r"\bottomrule" + "\n"


def build_figure(
    metrics: pd.DataFrame, event_metrics: pd.DataFrame, completed_tasks: int, publish: bool
) -> None:
    order = {method: index for index, method in enumerate(METHODS)}
    metrics = metrics.copy()
    metrics["method_order"] = metrics["method"].map(order)
    clean = metrics[
        metrics["artifact"].eq("none") & metrics["coordinate_quality"].eq("true")
    ].copy()
    clean_by_event = event_metrics[event_metrics["artifact"].eq("none")].copy()
    hard = metrics[
        metrics["signal_strength"].eq("low") & metrics["coordinate_quality"].eq("noisy")
    ].copy()
    artifact = metrics[metrics["artifact"].isin(["composition", "stress", "partial_batch_time"])].copy()

    FIGURE_SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    clean.to_csv(FIGURE_SOURCE_ROOT / "figure3_clean_common_task_metrics.tsv", sep="\t", index=False)
    clean_by_event.to_csv(
        FIGURE_SOURCE_ROOT / "figure3_type_specific_clean_metrics.tsv", sep="\t", index=False
    )
    hard.to_csv(
        FIGURE_SOURCE_ROOT / "figure3_low_signal_noisy_coordinate_metrics.tsv",
        sep="\t",
        index=False,
    )
    artifact.to_csv(FIGURE_SOURCE_ROOT / "figure3_artifact_common_task_metrics.tsv", sep="\t", index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.8,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4))
    fig.suptitle(
        "Controlled raw-count nearest-method common task",
        fontsize=15,
        fontweight="bold",
        x=0.04,
        ha="left",
    )

    ax = axes[0, 0]
    event_types = ["activation", "suppression", "transient"]
    event_labels = ["Activation", "Suppression", "Transient"]
    x = np.arange(len(event_types))
    offsets = np.linspace(-0.28, 0.28, len(METHODS))
    for offset, method, label, color in zip(offsets, METHODS, METHOD_LABELS, COLORS, strict=True):
        means = []
        sems = []
        for event_type in event_types:
            values = clean_by_event.loc[
                clean_by_event["method"].eq(method)
                & clean_by_event["event_type"].eq(event_type),
                "pathway_auprc",
            ].to_numpy(dtype=float)
            means.append(float(np.mean(values)))
            sems.append(float(np.std(values, ddof=1) / np.sqrt(len(values))))
        ax.errorbar(
            x + offset,
            means,
            yerr=sems,
            fmt="o",
            color=color,
            markeredgecolor="#24313D",
            markeredgewidth=0.45,
            markersize=6.2,
            capsize=2.4,
            linewidth=1.1,
            label=label,
        )
    ax.set_xticks(x, event_labels)
    ax.set_ylim(0, 1.02)
    ax.set_title("A. Type-specific pathway AUPRC", loc="left")
    ax.set_ylabel("Pathway AUPRC")
    ax.text(
        0.02,
        0.04,
        "No-artifact tasks; n=120 per method x event type",
        transform=ax.transAxes,
        fontsize=7.2,
        color="#65727E",
    )
    ax.legend(frameon=False, fontsize=6.8, ncol=2, loc="lower right")
    ax.grid(axis="y", color="#D8DEE4", linewidth=0.6)

    ax = axes[0, 1]
    hard_values = [
        hard.loc[hard["method"].eq(method), "pathway_level_auprc"].to_numpy(dtype=float)
        for method in METHODS
    ]
    bp = ax.boxplot(hard_values, tick_labels=METHOD_LABELS, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], COLORS, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_ylim(0, 1.02)
    ax.set_title("B. Low-signal, noisy-coordinate AUPRC", loc="left")
    ax.set_ylabel("Pathway AUPRC")
    ax.tick_params(axis="x", rotation=22)
    ax.text(
        0.02,
        0.04,
        "All artifact strata; n=120 tasks per method",
        transform=ax.transAxes,
        fontsize=7.2,
        color="#65727E",
    )
    ax.grid(axis="y", color="#D8DEE4", linewidth=0.6)

    ax = axes[1, 0]
    artifacts = ["composition", "stress", "partial_batch_time"]
    artifact_labels = ["Composition", "Stress", "Batch/time"]
    x = np.arange(len(artifacts))
    width = 0.15
    for index, (method, label, color) in enumerate(zip(METHODS, METHOD_LABELS, COLORS, strict=True)):
        means = []
        sems = []
        for artifact_name in artifacts:
            values = artifact.loc[
                artifact["method"].eq(method) & artifact["artifact"].eq(artifact_name),
                "matched_top_k_artifact_false_promotion_rate",
            ].to_numpy(dtype=float)
            means.append(float(np.mean(values)))
            sems.append(float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0)
        ax.bar(x + (index - 2) * width, means, width, yerr=sems, label=label, color=color, capsize=2)
    ax.set_xticks(x, artifact_labels)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Matched top-k artifact false-promotion rate")
    ax.set_title("C. Artifact false-promotion at matched top-k", loc="left")
    ax.legend(frameon=False, fontsize=6.8, ncol=2)
    ax.grid(axis="y", color="#D8DEE4", linewidth=0.6)

    ax = axes[1, 1]
    label_offsets = {
        "TIPS": (6, -13),
        "scTransient": (-72, 9),
        "tradeSeq": (-58, -16),
        "score_then_smooth": (-94, -17),
        "TED": (7, 11),
    }
    for method, label, color in zip(METHODS, METHOD_LABELS, COLORS, strict=True):
        x_value = float(
            artifact.loc[
                artifact["method"].eq(method),
                "matched_top_k_artifact_false_promotion_rate",
            ].mean()
        )
        y_value = float(clean.loc[clean["method"].eq(method), "pathway_level_auprc"].mean())
        ax.scatter(x_value, y_value, s=90, color=color, edgecolor="#24313D", linewidth=0.6)
        ax.annotate(
            label,
            (x_value, y_value),
            xytext=label_offsets[method],
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.6},
        )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Matched top-k artifact false-promotion rate (lower is better)")
    ax.set_ylabel("Clean-scenario pathway AUPRC")
    ax.set_title("D. Accuracy-risk Pareto", loc="left")
    ax.grid(color="#D8DEE4", linewidth=0.6)
    ax.text(0.02, 0.04, "Detection-level common task; not an E-level false-promotion claim", transform=ax.transAxes, fontsize=7.2, color="#65727E")

    fig.text(0.99, 0.01, f"Validated locked tasks: {completed_tasks}/480", ha="right", fontsize=7.5, color="#65727E")
    fig.tight_layout(rect=(0, 0.025, 1, 0.95))
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    png = SUMMARY_ROOT / "figure3_primary_heldout_performance.png"
    pdf = SUMMARY_ROOT / "figure3_primary_heldout_performance.pdf"
    fig.savefig(png, dpi=320, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    if publish:
        for target in FIGURE_TARGETS:
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(png, target / png.name)
            shutil.copy2(pdf, target / pdf.name)
        for target in SOURCE_TARGETS:
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(FIGURE_SOURCE_ROOT / "figure3_clean_common_task_metrics.tsv", target / "figure3_clean_common_task_metrics.tsv")
            shutil.copy2(
                FIGURE_SOURCE_ROOT / "figure3_type_specific_clean_metrics.tsv",
                target / "figure3_type_specific_clean_metrics.tsv",
            )
            shutil.copy2(
                FIGURE_SOURCE_ROOT / "figure3_low_signal_noisy_coordinate_metrics.tsv",
                target / "figure3_low_signal_noisy_coordinate_metrics.tsv",
            )
            shutil.copy2(FIGURE_SOURCE_ROOT / "figure3_artifact_common_task_metrics.tsv", target / "figure3_artifact_common_task_metrics.tsv")


def write_outputs(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    event_metrics: pd.DataFrame,
    native_manifest: pd.DataFrame,
    completed_tasks: int,
) -> None:
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    MACHINE_ROOT.mkdir(parents=True, exist_ok=True)
    native_root = MACHINE_ROOT / "method_native_outputs"
    native_root.mkdir(parents=True, exist_ok=True)

    scenarios = scenario_registry()
    event_summary = aggregate_event_metrics(event_metrics)
    headline_summary = headline_figure_metrics(metrics, event_metrics)
    scenarios.to_csv(MACHINE_ROOT / "common_task_scenario_registry.tsv", sep="\t", index=False)
    truth.to_csv(MACHINE_ROOT / "common_task_truth_masked.tsv", sep="\t", index=False)
    predictions.to_csv(MACHINE_ROOT / "method_harmonized_event_outputs.tsv", sep="\t", index=False)
    predictions.to_csv(
        MACHINE_ROOT / "method_harmonized_event_outputs.tsv.gz",
        sep="\t",
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    event_summary.to_csv(MACHINE_ROOT / "metrics_by_method_event_type_artifact.tsv", sep="\t", index=False)
    headline_summary.to_csv(MACHINE_ROOT / "common_task_headline_summary.tsv", sep="\t", index=False)
    (SUMMARY_ROOT / "table_s13_rows.tex").write_text(s13_latex_rows(event_summary), encoding="utf-8")
    metrics.to_csv(SUMMARY_ROOT / "scenario_method_metrics.tsv", sep="\t", index=False)
    event_metrics.to_csv(SUMMARY_ROOT / "scenario_method_event_type_metrics.tsv", sep="\t", index=False)
    headline_summary.to_csv(SUMMARY_ROOT / "common_task_headline_summary.tsv", sep="\t", index=False)
    native_manifest.to_csv(native_root / "manifest.tsv", sep="\t", index=False)
    (native_root / "README.md").write_text(
        "# Native method outputs\n\n"
        "The manifest indexes the immutable per-scenario native files used by the locked common task. "
        "Files remain in their scenario directories to avoid duplicating several thousand artifacts. "
        "The score-then-smooth row points to the directly harmonized method slice because it has no separate adapter.\n",
        encoding="utf-8",
    )
    status = {
        "expected_tasks": 480,
        "completed_tasks": completed_tasks,
        "complete": completed_tasks == 480,
        "truth_access_policy": "private true time removed; truth.tsv written only after all native outputs were serialized",
        "fair_operating_point": (
            "artifact-target false-promotion rate at matched top-k, "
            "where k is the number of true dynamic pathways"
        ),
        "fair_metric_derivation": (
            "recomputed uniformly from serialized ranking outputs and post-output truth; "
            "native-threshold false-call rates retained separately"
        ),
        "methods": METHODS,
    }
    (SUMMARY_ROOT / "summary_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (MACHINE_ROOT / "common_task_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    manifest_rows = []
    for path in sorted(MACHINE_ROOT.rglob("*")):
        if path.is_file() and path.name != "manifest.tsv":
            manifest_rows.append(
                {
                    "file": path.relative_to(MACHINE_ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    pd.DataFrame(manifest_rows).to_csv(MACHINE_ROOT / "manifest.tsv", sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true", help="development-only; never publishes manuscript Figure 3")
    parser.add_argument("--no-publish", action="store_true")
    args = parser.parse_args()
    scenario_dirs = require_complete_scenarios(args.allow_partial)
    metrics, predictions, truth, event_metrics, native_manifest = collect(scenario_dirs)
    completed = len(scenario_dirs)
    write_outputs(metrics, predictions, truth, event_metrics, native_manifest, completed)
    publish = completed == 480 and not args.no_publish
    build_figure(metrics, event_metrics, completed, publish)
    print(f"common-task summary complete={completed}/480 publish={publish} outputs={SUMMARY_ROOT}")


if __name__ == "__main__":
    main()
