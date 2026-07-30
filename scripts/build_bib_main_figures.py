from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "ted_v1_submission"
SOURCE = RESULTS / "figure_source_data"
FIGURE_DESTINATIONS = [
    ROOT / "results" / "bib_manuscript_revision" / "figures",
    ROOT / "GenomeBiology_known_source_submission_package" / "03_figures_final_upload",
    ROOT / "GenomeBiology_known_source_submission_package" / "03_figures",
    ROOT / "GenomeBiology_known_source_submission_package" / "01_main_manuscript" / "figures",
    ROOT
    / "GenomeBiology_known_source_submission_package"
    / "06_latex_source"
    / "TED_GenomeBiology_Main_Manuscript_Only"
    / "figures",
    ROOT
    / "GenomeBiology_known_source_submission_package"
    / "06_latex_source"
    / "TED_GenomeBiology_LaTeX_submission"
    / "figures",
]
SUPPLEMENT_FIGURE_DESTINATIONS = [
    ROOT
    / "GenomeBiology_known_source_submission_package"
    / "06_latex_source"
    / "TED_GenomeBiology_LaTeX_submission"
    / "supplementary_files"
    / "figures",
]
CURRENT_FIGURE_STEMS = [
    "figure1_problem_definition_ted_algorithm",
    "figure2_frozen_benchmark_design",
    "figure3_primary_heldout_performance",
    "figure4_robustness_portability_scalability",
    "figure5_independent_real_data_validation",
    "graphical_abstract",
]
LEGACY_FIGURE_STEMS = [
    "figure1_ted_event_object_claim_boundary",
    "figure2_known_source_validation",
    "figure3_benchmark_hardening",
    "figure4_gse271399_gata1_cross_dataset_support",
    "figure5_claim_upgrade_block_audit",
]
LEGACY_SOURCE_FILES = [
    "figure3_event_mode_confusion_matrix.tsv",
    "figure3_ambiguity_summary.tsv",
    "figure3_confounded_null_max.tsv",
    "figure3_packet_bootstrap_summary.tsv",
]
LEGACY_PACKAGE_SOURCE_FILES = [f"figure{index}_source_data.tsv" for index in range(1, 6)]

BLUE = "#2F5D8A"
BLUE_MID = "#6F9BC3"
BLUE_LIGHT = "#DCE9F4"
GOLD = "#C28A2C"
GOLD_LIGHT = "#F5E5BF"
INK = "#24313D"
MID = "#65727E"
GRID = "#D8DEE4"
OPEN = "#F7F9FB"
FAIL = "#555B61"


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.8,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axis(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)


def panel(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_title(f"{letter}. {title}", loc="left", pad=8)


def box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    face: str,
    edge: str = INK,
    title_size: float = 9,
    body_size: float = 7.5,
    linestyle: str = "-",
    wrap_width: int | None = None,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.1,
        linestyle=linestyle,
    )
    ax.add_patch(patch)
    x, y = xy
    ax.text(x + 0.03 * width, y + height - 0.10 * height, title, ha="left", va="top", fontsize=title_size, fontweight="bold")
    ax.text(
        x + 0.03 * width,
        y + height - 0.30 * height,
        textwrap.fill(body, wrap_width or max(14, int(width * 60))),
        ha="left",
        va="top",
        fontsize=body_size,
        linespacing=1.25,
    )
    return patch


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = MID, linestyle: str = "-") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color=color,
            linestyle=linestyle,
        )
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    staging = RESULTS / "figures"
    staging.mkdir(parents=True, exist_ok=True)
    png = staging / f"{stem}.png"
    pdf = staging / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    for destination in FIGURE_DESTINATIONS:
        destination.mkdir(parents=True, exist_ok=True)
        for source_file in (pdf, png):
            try:
                shutil.copy2(source_file, destination / source_file.name)
            except OSError as exc:
                # A PDF viewer or antivirus scanner can briefly retain a Windows
                # image mapping.  Keep generating all other publication outputs;
                # the locked raster is retried by the release-sync step.
                print(f"Warning: could not refresh {destination / source_file.name}: {exc}")
    plt.close(fig)


def save_supplement_figure(fig: plt.Figure, stem: str) -> None:
    staging = RESULTS / "supplementary_figures"
    staging.mkdir(parents=True, exist_ok=True)
    png = staging / f"{stem}.png"
    pdf = staging / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    for destination in SUPPLEMENT_FIGURE_DESTINATIONS:
        destination.mkdir(parents=True, exist_ok=True)
        for source_file in (pdf, png):
            try:
                shutil.copy2(source_file, destination / source_file.name)
            except OSError as exc:
                print(f"Warning: could not refresh {destination / source_file.name}: {exc}")
    plt.close(fig)


def clean_legacy_outputs() -> None:
    for directory in [RESULTS / "figures", *FIGURE_DESTINATIONS]:
        for stem in LEGACY_FIGURE_STEMS:
            for suffix in (".pdf", ".png"):
                path = directory / f"{stem}{suffix}"
                if path.exists():
                    path.unlink()
    for name in LEGACY_SOURCE_FILES:
        path = SOURCE / name
        if path.exists():
            path.unlink()
    for directory in FIGURE_DESTINATIONS[:2]:
        for name in LEGACY_PACKAGE_SOURCE_FILES:
            (directory / name).unlink(missing_ok=True)


def figure1() -> None:
    figure1_source = pd.DataFrame(
        [
            ["A", "input", "upstream_signal", "trajectory, time, perturbation or pathway activity"],
            ["A", "inference", "event_inference", "declared family, contrast, unit, null, effect and mode"],
            ["A", "gate", "competing_explanations", "composition, stress, state overlap and negative controls"],
            ["B", "event_field", "identity", "dataset, contrast, family and biological unit"],
            ["B", "event_field", "estimate", "effect, direction, timing and test status"],
            ["B", "event_field", "support", "block support, LODO, overlap and control margin"],
            ["B", "event_field", "interpretation", "mode, ambiguity set and claim boundary"],
            ["C", "within_study_support", "E0", "unsupported, non-estimable, non-identifiable or artifact dominated"],
            ["C", "within_study_support", "E1", "current-design statistical support"],
            ["C", "within_study_support", "E2", "independent-block support and identifiable mode"],
            ["D", "parallel_evidence", "outcome", "orthogonal outcome qualification and controls"],
            ["D", "parallel_evidence", "reversal", "prespecified intervention reversal"],
            ["D", "parallel_evidence", "rescue", "matched same-system rescue and recovery"],
            ["E", "independence_context", "event_replication_eligibility_status", "pending, passed or failed under frozen prerequisites"],
            ["E", "independence_context", "event_replication_test_status", "not_run, run_not_supported or run_supported"],
            ["E", "independence_context", "event_replication_status", "pending, not_evaluable, failed or passed"],
            ["E", "independence_context", "outcome_replication_status", "pending, not_tested, failed or passed"],
        ],
        columns=["panel", "element_type", "element", "definition"],
    )
    figure1_source.to_csv(SOURCE / "figure1_problem_definition.tsv", sep="\t", index=False)

    fig = plt.figure(figsize=(13.2, 8.4))
    gs = fig.add_gridspec(2, 6, hspace=0.48, wspace=0.48)
    fig.suptitle("TED evidence structure", fontsize=15, fontweight="bold", x=0.04, ha="left")

    ax = fig.add_subplot(gs[0, :3])
    panel(ax, "A", "Signal to event inference")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, (0.03, 0.29), 0.25, 0.48, "Upstream signal", "Trajectory, time, perturbation or pathway activity", BLUE_LIGHT, title_size=8.5)
    box(ax, (0.37, 0.29), 0.25, 0.48, "Event inference", "Frozen family, contrast, biological unit, null, effect and mode", OPEN, title_size=8.5)
    box(ax, (0.71, 0.29), 0.25, 0.48, "Gate audit", "State overlap, composition, stress, negative controls and robustness", GOLD_LIGHT, title_size=8.5)
    arrow(ax, (0.28, 0.53), (0.37, 0.53), BLUE)
    arrow(ax, (0.62, 0.53), (0.71, 0.53), GOLD)
    ax.text(0.50, 0.11, "Signal detection does not itself establish event support or biological validation.", ha="center", fontsize=8.2, color=MID)

    ax = fig.add_subplot(gs[0, 3:])
    panel(ax, "B", "Event object")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fields = [
        ("Identity", "dataset; contrast; event family; biological unit"),
        ("Estimate", "effect; direction; timing; test status; q/missing reason"),
        ("Support", "blocks; direction; LODO; overlap; control margin"),
        ("Interpretation", "mode; ambiguity set; supported and unsupported claim"),
    ]
    y = 0.81
    for i, (name, value) in enumerate(fields):
        face = BLUE_LIGHT if i < 2 else OPEN
        ax.add_patch(Rectangle((0.05, y - 0.11), 0.90, 0.14, facecolor=face, edgecolor=GRID, linewidth=0.8))
        ax.text(0.08, y - 0.04, name, fontweight="bold", va="center", fontsize=8.4)
        ax.text(0.33, y - 0.04, value, va="center", fontsize=6.9)
        y -= 0.18
    ax.text(0.05, 0.05, "Every failed mandatory gate is serialized with its observed value and reason.", fontsize=7.6, color=MID)

    ax = fig.add_subplot(gs[1, :2])
    panel(ax, "C", "Within-study event support")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    levels = [
        (0.69, "E2", "independent-block support\n+ identifiable mode", BLUE),
        (0.42, "E1", "current-design statistical support", BLUE_MID),
        (0.15, "E0", "unsupported / non-estimable /\nnon-identifiable /\nartifact dominated", OPEN),
    ]
    for y, code, body, face in levels:
        ax.add_patch(FancyBboxPatch((0.10, y), 0.80, 0.18, boxstyle="round,pad=0.012", facecolor=face, edgecolor=INK, linewidth=0.9))
        text_color = "white" if code == "E2" else INK
        ax.text(0.17, y + 0.09, code, va="center", fontweight="bold", fontsize=11, color=text_color)
        ax.text(0.38, y + 0.09, body, va="center", fontsize=7.5, color=text_color)
    ax.text(0.50, 0.03, "Selection stability remains a separate field.", ha="center", fontsize=7.4, color=MID)

    ax = fig.add_subplot(gs[1, 2:4])
    panel(ax, "D", "Parallel evidence records")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    records = [
        (0.04, "Outcome", "orthogonal endpoint\n+ prespecified controls", BLUE_LIGHT),
        (0.36, "Reversal", "intervention reversal\n+ matched controls", GOLD_LIGHT),
        (0.68, "Rescue", "same-system rescue\n+ molecular recovery", "#E7F2E7"),
    ]
    for x, title, body, face in records:
        box(ax, (x, 0.34), 0.27, 0.42, title, body, face, title_size=8.8, body_size=7.2, wrap_width=16)
    ax.text(0.50, 0.22, "Each record stores status, basis and controls.", ha="center", fontsize=7.2, color=MID)
    ax.text(0.50, 0.08, "Parallel evidence does not repair a failed E gate.", ha="center", fontsize=7.8, color=FAIL, fontweight="bold")

    ax = fig.add_subplot(gs[1, 4:])
    panel(ax, "E", "Independence context and replication")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, (0.04, 0.56), 0.34, 0.25, "Primary study", "event + evidence records", BLUE_LIGHT, title_size=8.0, body_size=6.8)
    box(ax, (0.62, 0.56), 0.34, 0.25, "Independent cohort", "same frozen family and gates", OPEN, title_size=7.6, body_size=6.6, linestyle="--")
    arrow(ax, (0.38, 0.68), (0.62, 0.68), MID, linestyle="--")
    replication_rows = [
        (0.39, "Event eligibility", "pending | passed | failed", BLUE_LIGHT),
        (0.29, "Event test", "not_run | run_not_supported | run_supported", OPEN),
        (0.19, "Event replication", "pending | not_evaluable | failed | passed", BLUE_LIGHT),
        (0.09, "Outcome replication", "pending | not_tested | failed | passed", GOLD_LIGHT),
    ]
    for y, label, values, face in replication_rows:
        ax.add_patch(Rectangle((0.04, y), 0.92, 0.075, facecolor=face, edgecolor=GRID))
        ax.text(0.065, y + 0.0375, label, va="center", fontweight="bold", fontsize=5.8)
        ax.text(0.94, y + 0.0375, values, ha="right", va="center", fontsize=4.7)
    ax.text(0.50, 0.015, "Eligibility, test execution and result are distinct facets.", ha="center", fontsize=6.3, color=MID)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, "figure1_problem_definition_ted_algorithm")


def figure2() -> None:
    split_audit = read_tsv(ROOT / "results" / "ted_current_task_benchmark" / "split_and_leakage_audit.tsv")
    split_audit.to_csv(SOURCE / "figure2_split_and_leakage_audit.tsv", sep="\t", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.3))
    fig.suptitle("Leakage control and locked common-task design", fontsize=15, fontweight="bold", x=0.04, ha="left")

    ax = axes[0, 0]
    panel(ax, "A", "Lock before outcome access")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, (0.03, 0.29), 0.27, 0.45, "1  Define task", "Inputs, biological units, pathway family, contrasts and estimand", OPEN, title_size=8.2)
    box(ax, (0.37, 0.22), 0.27, 0.59, "2  Freeze manifest", "Adapters, thresholds, seeds, QC, gates, hashes and failure rules", GOLD_LIGHT, edge=GOLD, title_size=8.2)
    box(ax, (0.71, 0.29), 0.26, 0.45, "3  Reveal outcome", "Truth, masked ADT or cohort values opened only after the lock", BLUE_LIGHT, title_size=8.2)
    arrow(ax, (0.30, 0.51), (0.38, 0.51), MID)
    arrow(ax, (0.64, 0.51), (0.72, 0.51), MID)
    ax.text(0.50, 0.075, "Future test: a direction-unknown cohort locked before any result access", ha="center", color=MID, fontsize=7.7)

    ax = axes[0, 1]
    panel(ax, "B", "Locked components and withheld information")
    ax.axis("off")
    split_rows = [
        ["Raw-count\ncommon task", "48-cell design\n+ five adapters", "Pathway +\nartifact truth", "480 held-out\nsets"],
        ["BNT162b2\nboundary audit", "Pathway, cells,\ncontrast + gates", "CD64/CD169\nADT", "RNA frozen\nfirst"],
        ["GSE171964\nreplication", "Corrected v2,\ntimes, QC + gates", "Expression +\neligibility", "No retuning"],
    ]
    table = ax.table(
        cellText=split_rows,
        colLabels=["Analysis", "Frozen before access", "Withheld", "Audit point"],
        cellLoc="left",
        colLoc="left",
        colWidths=[0.22, 0.34, 0.25, 0.19],
        bbox=[0.01, 0.18, 0.98, 0.69],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.15)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_facecolor(BLUE_LIGHT)
            cell.set_text_props(weight="bold")
    ax.text(0.02, 0.09, "A lock is reproducibility evidence; only the withheld field is protected from result-driven revision.", fontsize=7.8, color=MID)

    ax = axes[1, 0]
    panel(ax, "C", "Evidence strata and inferential units")
    ax.axis("off")
    rows = [
        ["Adaptive-window simulation", "null/event truth", "block profile"],
        ["Raw-count common task", "pathway + artifact truth", "dataset"],
        ["ZSCAPE resampling", "full-fit event object", "embryo"],
        ["BNT162b2 outcome", "masked same-cell protein", "donor"],
        ["GSE171964", "replication eligibility", "donor"],
    ]
    table = ax.table(cellText=rows, colLabels=["Stratum", "Truth/readout", "Unit"], cellLoc="left", colLoc="left", bbox=[0.02, 0.08, 0.96, 0.82])
    table.auto_set_font_size(False)
    table.set_fontsize(7.1)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_facecolor(BLUE_LIGHT)
            cell.set_text_props(weight="bold")

    ax = axes[1, 1]
    panel(ax, "D", "Locked nearest-method common task")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.50, 0.91, "Controlled raw counts; no shared real-data comparison is claimed", ha="center", fontsize=8.0, color=MID, fontweight="bold")
    box(ax, (0.07, 0.67), 0.86, 0.16, "Shared input", "Counts, metadata, ordered coordinate and pathway memberships", BLUE_LIGHT, body_size=7.7)
    box(ax, (0.07, 0.39), 0.86, 0.19, "Frozen adapters", "TIPS  •  scTransient  •  tradeSeq  •  score-then-smooth  •  TED", GOLD_LIGHT, edge=GOLD, body_size=8.0)
    box(ax, (0.07, 0.09), 0.86, 0.20, "Common evaluation after serialization", "Native outputs → harmonized pathway ranks → truth reveal → AUPRC and artifact risk", OPEN, body_size=7.7)
    arrow(ax, (0.50, 0.67), (0.50, 0.58), MID)
    arrow(ax, (0.50, 0.39), (0.50, 0.29), MID)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, "figure2_frozen_benchmark_design")


def figure3() -> None:
    calibration = ROOT / "results" / "ted_submission_calibration"
    benchmark = ROOT / "results" / "ted_current_task_benchmark"
    adaptive = ROOT / "results" / "ted_adaptive_window_multiplicity"
    predictions = read_tsv(benchmark / "audit_predictions.tsv")
    metrics = read_tsv(benchmark / "current_task_metrics.tsv")
    deltas = read_tsv(benchmark / "paired_deltas.tsv")
    fdr = read_tsv(calibration / "event_fdr_calibration.tsv")
    confounded_signal = read_tsv(calibration / "confounded_signal_calibration.tsv")
    adaptive_summary = read_tsv(adaptive / "method_summary_by_stratum.tsv")
    fdr_rows: list[dict[str, object]] = []
    for target_q, group in fdr.groupby("target_q", sort=True):
        worst = group.loc[group["empirical_fdp"].idxmax()]
        fdr_rows.append(
            {
                "target_q": target_q,
                "mean_empirical_fdp": float(group["empirical_fdp"].mean()),
                "mean_power": float(group["power"].mean()),
                "worst_empirical_fdp": float(worst["empirical_fdp"]),
                "worst_ci95_low": float(worst["empirical_fdp_ci95_low"]),
                "worst_ci95_high": float(worst["empirical_fdp_ci95_high"]),
                "worst_n_pathways": int(worst["n_pathways"]),
                "worst_pathway_overlap_rho": float(worst["pathway_overlap_rho"]),
                "relative_criterion_passed": int(group["calibration_pass"].sum()),
                "n_configurations": int(len(group)),
                "replicates_per_configuration": int(group["n_replicates"].iloc[0]),
            }
        )
    fdr_summary = pd.DataFrame(fdr_rows)

    predictions.to_csv(SOURCE / "figure3_current_task_audit_predictions.tsv", sep="\t", index=False)
    metrics.to_csv(SOURCE / "figure3_current_task_metrics.tsv", sep="\t", index=False)
    deltas.to_csv(SOURCE / "figure3_paired_deltas.tsv", sep="\t", index=False)
    fdr.to_csv(SOURCE / "figure3_fdr_configurations.tsv", sep="\t", index=False)
    fdr_summary.to_csv(SOURCE / "figure3_fdr_summary.tsv", sep="\t", index=False)
    confounded_signal.to_csv(SOURCE / "figure3_confounded_signal_calibration.tsv", sep="\t", index=False)
    adaptive_summary.to_csv(SOURCE / "figure3_adaptive_window_multiplicity.tsv", sep="\t", index=False)

    def matrix_for(task: str, labels: list[object]) -> np.ndarray:
        frame = predictions[(predictions["task"] == task) & (predictions["method"] == "TED_fixed_rules")]
        truth = pd.Categorical(frame["truth"], categories=labels, ordered=True)
        predicted = pd.Categorical(frame["prediction"], categories=labels, ordered=True)
        return pd.crosstab(truth, predicted, dropna=False).to_numpy(dtype=float)

    packet_class_labels = [
        "activation",
        "suppression",
        "developmental_delay",
        "true_loss",
        "fate_redirection",
        "composition_artifact",
        "stress_dominated",
        "not_identifiable",
        "outcome_supported",
        "reversal_supported",
    ]
    artifact_labels = ["False", "True"]
    evidence_labels = ["E0", "E1", "E2"]

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.7))
    fig.suptitle("Current-task performance and adaptive-window multiplicity", fontsize=15, fontweight="bold", x=0.04, ha="left")

    ax = axes[0, 0]
    panel(ax, "A", "Controlled packet-class confusion")
    matrix = matrix_for("packet_class", packet_class_labels)
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    short = [x.replace("developmental_", "dev_").replace("composition_", "comp_").replace("not_identifiable", "not_id").replace("outcome_supported", "outcome").replace("reversal_supported", "reversal") for x in packet_class_labels]
    ax.set_xticks(range(len(short)), short, rotation=55, ha="right", fontsize=7.0)
    ax.set_yticks(range(len(short)), short, fontsize=7.0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] >= 0.30:
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if matrix[i, j] > 0.55 else INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    ax.set_xlabel("Predicted packet class")
    ax.set_ylabel("Controlled truth")

    ax = axes[0, 1]
    panel(ax, "B", "Artifact confusion")
    artifact = matrix_for("artifact", artifact_labels)
    artifact_fraction = artifact / artifact.sum(axis=1, keepdims=True)
    im = ax.imshow(artifact_fraction, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1], ["No artifact", "Artifact"])
    ax.set_yticks([0, 1], ["No artifact", "Artifact"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(artifact[i, j])}\n({artifact_fraction[i, j]:.2f})", ha="center", va="center", fontsize=10, color="white" if artifact_fraction[i, j] > 0.55 else INK)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Controlled truth")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    ax = axes[0, 2]
    panel(ax, "C", "Current E-level confusion")
    evidence = matrix_for("evidence", evidence_labels)
    evidence_fraction = evidence / evidence.sum(axis=1, keepdims=True)
    im = ax.imshow(evidence_fraction, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3), evidence_labels)
    ax.set_yticks(range(3), evidence_labels)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{int(evidence[i, j])}\n({evidence_fraction[i, j]:.2f})", ha="center", va="center", fontsize=9, color="white" if evidence_fraction[i, j] > 0.55 else INK)
    ax.set_xlabel("Assigned E level")
    ax.set_ylabel("Controlled truth")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    ax = axes[1, 0]
    panel(ax, "D", "Executable baseline comparison")
    method_order = ["TED_fixed_rules", "logistic", "random_forest", "hist_gradient_boosting", "dummy_most_frequent"]
    display = ["TED", "Logistic", "RF", "HistGB", "Dummy"]
    task_metrics = [("packet_class", "macro_f1", "Packet-class F1"), ("artifact", "primary_metric", "Artifact F1"), ("evidence", "macro_f1", "E macro-F1")]
    x = np.arange(len(method_order))
    width = 0.24
    colors = [BLUE, GOLD, BLUE_MID]
    for offset, (task, field, label) in enumerate(task_metrics):
        values = [float(metrics[(metrics["task"] == task) & (metrics["method"] == method)][field].iloc[0]) for method in method_order]
        ax.bar(x + (offset - 1) * width, values, width, label=label, color=colors[offset], edgecolor=INK, linewidth=0.5)
    ax.set_xticks(x, display, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Shifted-audit metric")
    ax.legend(frameon=False, fontsize=7.5, ncol=1)
    clean_axis(ax)

    ax = axes[1, 1]
    panel(ax, "E", "Accuracy-risk Pareto summary")
    e_metrics = metrics[metrics["task"] == "evidence"].set_index("method")
    table_methods = ["TED_fixed_rules", "logistic", "random_forest"]
    table_labels = ["TED", "Logistic", "Random forest"]
    table_rows = []
    for method in table_methods:
        packet_f1 = float(metrics[(metrics["task"] == "packet_class") & (metrics["method"] == method)]["macro_f1"].iloc[0])
        artifact_f1 = float(metrics[(metrics["task"] == "artifact") & (metrics["method"] == method)]["primary_metric"].iloc[0])
        table_rows.append(
            [
                f"{packet_f1:.3f}",
                f"{artifact_f1:.3f}",
                f"{float(e_metrics.loc[method, 'false_e_promotion']):.3f}",
                f"{float(e_metrics.loc[method, 'false_e_demotion']):.3f}",
                f"{float(e_metrics.loc[method, 'non_e0_call_fraction']):.3f}",
                "Yes" if method == "TED_fixed_rules" else "No",
            ]
        )
    pd.DataFrame(
        table_rows,
        index=table_labels,
        columns=["packet_class_f1", "artifact_f1", "false_e_promotion", "false_e_demotion", "non_e0_call_fraction", "reason_codes"],
    ).rename_axis("method").reset_index().to_csv(SOURCE / "figure3_accuracy_risk_pareto.tsv", sep="\t", index=False)
    ax.axis("off")
    pareto = ax.table(
        cellText=table_rows,
        rowLabels=table_labels,
        colLabels=["Packet\nF1", "Artifact\nF1", "False E\nprom.", "False E\ndem.", "Non-E0\ncall frac.", "Reason\ncodes"],
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    pareto.auto_set_font_size(False)
    pareto.set_fontsize(6.6)
    pareto.scale(1.02, 1.55)
    for (row, _), cell in pareto.get_celld().items():
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.set_facecolor(BLUE_LIGHT)
            cell.set_text_props(weight="bold")
    ax.text(0.5, 0.04, "Packet = 10-class aggregate; non-E0 call fraction uses all audit packets as denominator.", ha="center", va="bottom", fontsize=6.8, color=MID, transform=ax.transAxes)

    ax = axes[1, 2]
    panel(ax, "F", "Full adaptive-window benchmark")
    method_order = ["naive_selected_window_bh", "per_event_max_window_bh", "family_wide_maxT_fwer"]
    method_labels = ["Naive\nselected", "Per-event\nmax + BH", "Family\nmaxT"]
    clean_signal = adaptive_summary[adaptive_summary["analysis_stratum"].eq("clean_signal")].set_index("method")
    clean_null = adaptive_summary[adaptive_summary["analysis_stratum"].eq("clean_null")].set_index("method")
    values = np.array(
        [
            [float(clean_signal.loc[m, "mean_fdp"]) for m in method_order],
            [float(clean_null.loc[m, "mean_family_wise_false_positive"]) for m in method_order],
            [float(clean_signal.loc[m, "mean_power"]) for m in method_order],
        ]
    )
    x = np.arange(len(method_order))
    width = 0.24
    for offset, (label, color) in enumerate(
        zip(["Clean-signal FDP", "Clean-null FWER", "Clean-signal power"], [FAIL, GOLD, BLUE], strict=True)
    ):
        ax.bar(x + (offset - 1) * width, values[offset], width, label=label, color=color, edgecolor=INK, linewidth=0.5)
    ax.axhline(0.10, color=MID, linestyle="--", linewidth=1, label="Target 0.10")
    ax.set_xticks(x, method_labels)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Scenario-replicate mean")
    ax.text(0.02, 0.94, "36 scenarios; 12 repeats (clean null: 100)\nFull scan rerun; artifacts reported separately", transform=ax.transAxes, va="top", fontsize=6.8, color=MID)
    ax.legend(frameon=False, fontsize=6.8, loc="upper right", bbox_to_anchor=(1.0, 0.82))
    clean_axis(ax, "y")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, "figure3_primary_heldout_performance")

def figure3_factorized() -> None:
    base = ROOT / "results" / "ted_factorized_ablation"
    ood_base = ROOT / "results" / "ted_factorized_ood_challenge"
    truth = read_tsv(base / "factorized_packet_truth.tsv")
    predictions = read_tsv(base / "factorized_predictions.tsv")
    ablation = read_tsv(base / "ablation_metrics.tsv")
    reason = read_tsv(base / "reason_code_confusion.tsv")
    ood = read_tsv(ood_base / "ood_metrics.tsv")
    full = predictions[predictions["variant"].eq("full_ted")].merge(
        truth, on="packet_id", validate="one_to_one"
    )

    # The complete factor grid is a rule-consistency/software test suite. Keep
    # its calls and confusion inputs as supplementary source data, not as the
    # main performance display.
    truth.to_csv(SOURCE / "supplementary_rule_consistency_truth.tsv", sep="\t", index=False)
    full.to_csv(SOURCE / "supplementary_rule_consistency_calls.tsv", sep="\t", index=False)
    reason.to_csv(SOURCE / "supplementary_rule_consistency_reason_confusion.tsv", sep="\t", index=False)
    ablation.to_csv(SOURCE / "figure3_gate_ablation.tsv", sep="\t", index=False)
    overall = ood[(ood["stratum"] == "all") & (ood["level"] == "all")].iloc[0]
    threshold = ood[ood["stratum"] == "borderline_case"].copy()
    overlap = ood[ood["stratum"] == "gene_set_overlap"].copy()
    pd.DataFrame(
        {
            "target": ["Biological mode", "Artifact", "Identifiability", "E assignment"],
            "macro_f1": [overall["biological_mode_macro_f1"], overall["artifact_macro_f1"], overall["identifiability_macro_f1"], overall["event_support_macro_f1"]],
        }
    ).to_csv(SOURCE / "figure3_ood_overall_metrics.tsv", sep="\t", index=False)
    threshold.to_csv(SOURCE / "figure3_ood_threshold_sensitivity.tsv", sep="\t", index=False)
    overlap.to_csv(SOURCE / "figure3_ood_overlap_sensitivity.tsv", sep="\t", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.4))
    fig.suptitle("Randomized OOD performance and TED gate sensitivity", fontsize=15, fontweight="bold", x=0.04, ha="left")

    ax = axes[0, 0]
    panel(ax, "A", "Randomized OOD stress test")
    target_labels = ["Mode", "Artifact", "Identifiability", "E assignment"]
    target_values = [float(overall["biological_mode_macro_f1"]), float(overall["artifact_macro_f1"]), float(overall["identifiability_macro_f1"]), float(overall["event_support_macro_f1"])]
    bars = ax.bar(np.arange(4), target_values, color=[GOLD, BLUE, BLUE_MID, BLUE_LIGHT], edgecolor=INK, linewidth=0.6)
    ax.set_xticks(np.arange(4), target_labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Macro-F1")
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in target_values], padding=3, fontsize=8)
    ax.text(0.02, 0.94, f"n={int(overall['n_packets'])}; false E promotion={float(overall['false_e_promotion']):.3f}\nfalse E demotion={float(overall['false_e_demotion']):.3f}", transform=ax.transAxes, va="top", fontsize=7.2, color=MID)
    clean_axis(ax, "y")

    ax = axes[0, 1]
    panel(ax, "B", "Borderline threshold sensitivity")
    threshold = threshold.set_index("level").loc[["False", "True"]]
    x = np.arange(2)
    width = 0.34
    ax.bar(x - width / 2, threshold["event_support_macro_f1"], width, label="E macro-F1", color=BLUE, edgecolor=INK, linewidth=0.5)
    ax.bar(x + width / 2, threshold["false_e_promotion"], width, label="False E promotion", color=FAIL, edgecolor=INK, linewidth=0.5)
    ax.set_xticks(x, [f"Away from boundary\n(n={int(threshold.loc['False', 'n_packets'])})", f"Borderline\n(n={int(threshold.loc['True', 'n_packets'])})"])
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Fraction")
    ax.legend(frameon=False, fontsize=7.5)
    clean_axis(ax, "y")

    ax = axes[1, 0]
    panel(ax, "C", "Gene-set-overlap sensitivity")
    overlap = overlap.set_index("level").loc[["low", "moderate", "high"]]
    x = np.arange(3)
    width = 0.34
    ax.bar(x - width / 2, overlap["biological_mode_macro_f1"], width, label="Mode macro-F1", color=GOLD, edgecolor=INK, linewidth=0.5)
    ax.bar(x + width / 2, overlap["event_support_macro_f1"], width, label="E macro-F1", color=BLUE, edgecolor=INK, linewidth=0.5)
    ax.set_xticks(x, [f"Low\n(n={int(overlap.loc['low', 'n_packets'])})", f"Moderate\n(n={int(overlap.loc['moderate', 'n_packets'])})", f"High\n(n={int(overlap.loc['high', 'n_packets'])})"])
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Macro-F1")
    ax.legend(frameon=False, fontsize=7.5)
    clean_axis(ax, "y")

    ax = axes[1, 1]
    panel(ax, "D", "Gate ablation on the rule-consistency suite")
    order = ["full_ted", "without_block_gate", "without_matched_state_adjustment", "without_negative_controls", "without_identifiability_gate", "without_ambiguity_set", "ev_collapsed_ladder"]
    labels = ["Full", "No block", "No matched", "No neg. ctrl", "No ident.", "No ambiguity", "E/V collapsed"]
    local = ablation.set_index("variant").loc[order]
    x = np.arange(len(order))
    width = 0.25
    for offset, (column, label, color) in enumerate(
        [("false_e_promotion", "False E promotion", FAIL), ("artifact_recall", "Artifact recall", BLUE), ("incorrect_definite_mode_rate", "Wrong definite mode", GOLD)]
    ):
        ax.bar(x + (offset - 1) * width, local[column], width, label=label, color=color, edgecolor=INK, linewidth=0.4)
    ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=7.0)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Fraction")
    ax.legend(frameon=False, fontsize=6.8)
    clean_axis(ax, "y")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, "figure3_primary_heldout_performance")

    def confusion(frame: pd.DataFrame, truth_col: str, pred_col: str, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
        observed = pd.crosstab(
            pd.Categorical(frame[truth_col], categories=labels, ordered=True),
            pd.Categorical(frame[pred_col], categories=labels, ordered=True),
            dropna=False,
        ).to_numpy(dtype=float)
        fractions = np.divide(observed, observed.sum(axis=1, keepdims=True), out=np.zeros_like(observed), where=observed.sum(axis=1, keepdims=True) > 0)
        return observed, fractions

    def draw_confusion(ax, observed: np.ndarray, fractions: np.ndarray, labels: list[str]) -> None:
        ax.imshow(fractions, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        short = [label.replace("persistent_", "persist.").replace("not_identifiable", "not ident.") for label in labels]
        ax.set_xticks(range(len(labels)), short, rotation=35, ha="right", fontsize=7.0)
        ax.set_yticks(range(len(labels)), short, fontsize=7.0)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if observed[i, j] > 0:
                    ax.text(j, i, f"{int(observed[i, j])}\n({fractions[i, j]:.2f})", ha="center", va="center", fontsize=6.5, color="white" if fractions[i, j] > 0.55 else INK)
        ax.set_xlabel("Assigned")
        ax.set_ylabel("Controlled truth")

    supp_fig, supp_axes = plt.subplots(2, 3, figsize=(13.2, 8.7))
    supp_fig.suptitle("Rule-consistency suite: contract confusion matrices", fontsize=15, fontweight="bold", x=0.04, ha="left")
    evaluable = full[full.truth_artifact_class.eq("none") & full.truth_identifiability.eq("identifiable")]
    confusion_specs = [
        (supp_axes[0, 0], "A", "Biological mode (evaluable subset)", evaluable, "truth_biological_mode", "predicted_top_mode", ["activation", "suppression", "delay", "persistent_loss", "redirection"]),
        (supp_axes[0, 1], "B", "Artifact class", full, "truth_artifact_class", "predicted_artifact_class", ["none", "composition", "stress"]),
        (supp_axes[0, 2], "C", "Identifiability", full, "truth_identifiability", "predicted_identifiability", ["identifiable", "ambiguous", "not_identifiable"]),
        (supp_axes[1, 0], "D", "E assignment", full, "truth_event_support_code", "predicted_event_support_code", ["E0", "E1", "E2"]),
    ]
    for ax, letter, title, frame, truth_col, pred_col, labels in confusion_specs:
        panel(ax, letter, title)
        observed, fractions = confusion(frame, truth_col, pred_col, labels)
        draw_confusion(ax, observed, fractions, labels)
        pd.DataFrame(observed.astype(int), index=labels, columns=labels).rename_axis("truth").to_csv(SOURCE / f"supplementary_rule_consistency_{letter.lower()}_confusion.tsv", sep="\t")

    ax = supp_axes[1, 1]
    panel(ax, "E", "E0 reason code")
    reason_labels = ["not supported", "not estimable", "not identifiable", "artifact", "missing design"]
    values = reason.drop(columns=["truth_reason"]).to_numpy(float)
    fractions = values / values.sum(axis=1, keepdims=True)
    draw_confusion(ax, values, fractions, reason_labels)

    ax = supp_axes[1, 2]
    ax.axis("off")
    ax.text(0.02, 0.92, "Interpretation boundary", fontsize=11, fontweight="bold", color=INK, transform=ax.transAxes)
    ax.text(0.02, 0.78, "One packet represents each factor combination.\nThese matrices test software and rule consistency;\nthey do not estimate performance on new data.", fontsize=9.5, color=MID, va="top", linespacing=1.5, transform=ax.transAxes)
    ax.text(0.02, 0.45, "Randomized OOD performance is reported in Figure 3.", fontsize=9.5, color=BLUE, va="top", transform=ax.transAxes)
    supp_fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(supp_fig, "supplementary_figure_s6_rule_consistency_confusions")


def figure4() -> None:
    base = ROOT / "results" / "ted_submission_supplement"
    zscape = read_tsv(base / "zscape_repeated_holdout_stability" / "summary.tsv").iloc[0]
    threshold = read_tsv(base / "zscape_repeated_holdout_stability" / "threshold_sensitivity.tsv")
    subsampling = read_tsv(base / "zscape_repeated_holdout_stability" / "subsampling_curve.tsv")
    sensitivity = read_tsv(base / "upstream_sensitivity" / "upstream_sensitivity_summary.tsv")
    real_upstream = read_tsv(ROOT / "results" / "ted_real_data_upstream_sensitivity" / "real_data_upstream_metrics.tsv")
    scaling = read_tsv(base / "event_layer_scaling" / "ted_event_layer_scaling_summary.tsv")
    environment = json.loads((base / "event_layer_scaling" / "environment.json").read_text(encoding="utf-8"))

    holdout_source = pd.DataFrame(
        {
            "metric": ["Event Jaccard", "Direction", "Event mode"],
            "median": [zscape["event_jaccard_median"], zscape["direction_agreement_median"], zscape["event_mode_agreement_median"]],
            "iqr_low": [zscape["event_jaccard_iqr_low"], zscape["direction_agreement_iqr_low"], zscape["event_mode_agreement_iqr_low"]],
            "iqr_high": [zscape["event_jaccard_iqr_high"], zscape["direction_agreement_iqr_high"], zscape["event_mode_agreement_iqr_high"]],
            "minimum": [zscape["event_jaccard_minimum"], zscape["direction_agreement_minimum"], zscape["event_mode_agreement_minimum"]],
        }
    )
    split_source = pd.DataFrame(
        {
            "metric": ["Event Jaccard", "Direction", "Effect Spearman"],
            "median": [zscape["split_half_event_jaccard_median"], zscape["split_half_direction_agreement_on_common_calls_median"], zscape["split_half_effect_spearman_all_estimable_median"]],
            "iqr_low": [zscape["split_half_event_jaccard_iqr_low"], zscape["split_half_direction_agreement_on_common_calls_iqr_low"], zscape["split_half_effect_spearman_all_estimable_iqr_low"]],
            "iqr_high": [zscape["split_half_event_jaccard_iqr_high"], zscape["split_half_direction_agreement_on_common_calls_iqr_high"], zscape["split_half_effect_spearman_all_estimable_iqr_high"]],
            "minimum": [zscape["split_half_event_jaccard_minimum"], zscape["split_half_direction_agreement_on_common_calls_minimum"], zscape["split_half_effect_spearman_all_estimable_minimum"]],
        }
    )
    holdout_source.to_csv(SOURCE / "figure4_zscape_repeated_20pct_holdout.tsv", sep="\t", index=False)
    split_source.to_csv(SOURCE / "figure4_zscape_split_half.tsv", sep="\t", index=False)
    threshold.to_csv(SOURCE / "figure4_zscape_threshold_sensitivity.tsv", sep="\t", index=False)
    subsampling.to_csv(SOURCE / "figure4_zscape_subsampling_curve.tsv", sep="\t", index=False)
    sensitivity.to_csv(SOURCE / "figure4_upstream_sensitivity.tsv", sep="\t", index=False)
    real_upstream.to_csv(SOURCE / "figure4_real_data_upstream_sensitivity.tsv", sep="\t", index=False)
    scaling.to_csv(SOURCE / "figure4_scaling.tsv", sep="\t", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.4))
    fig.suptitle("Discovery stability and upstream sensitivity", fontsize=15, fontweight="bold", x=0.04, ha="left")

    ax = axes[0, 0]
    panel(ax, "A", "Repeated 20% embryo holdout")
    y = np.arange(len(holdout_source))
    xerr = np.vstack([holdout_source["median"] - holdout_source["iqr_low"], holdout_source["iqr_high"] - holdout_source["median"]])
    ax.errorbar(holdout_source["median"], y, xerr=xerr, fmt="o", color=BLUE, ecolor=BLUE_MID, capsize=4, markersize=7, label="Median and IQR")
    ax.scatter(holdout_source["minimum"], y, s=45, facecolors="white", edgecolors=GOLD, linewidths=1.5, label="Minimum")
    ax.set_yticks(y, holdout_source["metric"])
    ax.set_xlim(0, 1.03)
    ax.set_xlabel("Agreement with full fit (100 repeats)")
    ax.legend(frameon=False, fontsize=7.5, loc="center left")
    clean_axis(ax, "x")

    ax = axes[0, 1]
    panel(ax, "B", "Balanced split-half")
    y = np.arange(len(split_source))
    xerr = np.vstack([split_source["median"] - split_source["iqr_low"], split_source["iqr_high"] - split_source["median"]])
    ax.errorbar(split_source["median"], y, xerr=xerr, fmt="o", color=BLUE, ecolor=BLUE_MID, capsize=4, markersize=7, label="Median and IQR")
    ax.scatter(split_source["minimum"], y, s=45, facecolors="white", edgecolors=GOLD, linewidths=1.5, label="Minimum")
    ax.set_yticks(y, split_source["metric"])
    ax.set_xlim(0, 1.03)
    ax.set_xlabel("Between halves (50 repeats; 402 embryos/half)")
    ax.text(0.03, 0.07, "Discovery-set overlap falls sharply\nat half sample size", transform=ax.transAxes, fontsize=7.8, color=FAIL)
    clean_axis(ax, "x")

    ax = axes[1, 0]
    panel(ax, "C", "Embryo subsampling curve")
    ax.plot(subsampling["median_actual_retained_fraction"], subsampling["median_event_jaccard"], marker="o", color=BLUE, linewidth=1.8, label="Event-set Jaccard")
    ax.fill_between(subsampling["median_actual_retained_fraction"], subsampling["iqr_low_event_jaccard"], subsampling["iqr_high_event_jaccard"], color=BLUE_LIGHT, alpha=0.65)
    ax.plot(subsampling["median_actual_retained_fraction"], subsampling["median_effect_spearman"], marker="s", markerfacecolor="white", color=GOLD, linewidth=1.8, label="Effect Spearman")
    for _, row in subsampling.iterrows():
        ax.annotate(f"E2:{int(row['median_e2_eligible_calls'])}", (row["median_actual_retained_fraction"], row["median_event_jaccard"]), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=6.5, color=MID)
    ax.set_xlim(0.15, 0.95)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Retained embryo fraction (20 repeats/point)")
    ax.set_ylabel("Agreement with full fit")
    ax.legend(frameon=False, fontsize=7.0, loc="lower right")
    clean_axis(ax, "both")

    ax = axes[1, 1]
    panel(ax, "D", "Real-data upstream agreement")
    local = (
        real_upstream.groupby(["dataset_id", "scoring_method"], as_index=False)
        .agg(
            direction=("direction_agreement", "median"),
            mode=("event_mode_agreement", "median"),
            e_code=("e_code_agreement", "median"),
        )
    )
    dataset_short = {"GSE126085_real_time": "GSE126085", "Nestorowa_trajectory": "Nestorowa"}
    method_short = {"pyfgsea_rolling": "PyFgsea", "rank_auc": "rank-AUC", "mean_zscore": "mean-z", "ssgsea_bridge": "ssGSEA bridge"}
    local["row_label"] = local.apply(lambda row: f"{dataset_short[row['dataset_id']]} | {method_short[row['scoring_method']]}", axis=1)
    values = local[["direction", "mode", "e_code"]].to_numpy(float)
    im = ax.imshow(values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_yticks(range(len(local)), local["row_label"], fontsize=8.0)
    ax.set_xticks(range(3), ["Direction", "Mode", "E code"], rotation=25, ha="right")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8.0, color="white" if values[i, j] > 0.65 else INK)
    ax.text(0.02, -0.20, "Agreement <0.75 triggers an E2-blocking disagreement flag; alternatives are internal implementations.", transform=ax.transAxes, fontsize=7.5, color=MID)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.95), h_pad=2.0, w_pad=2.2)
    save_figure(fig, "figure4_robustness_portability_scalability")


def supplementary_scaling_figure() -> None:
    scaling = read_tsv(
        ROOT
        / "results"
        / "ted_submission_supplement"
        / "event_layer_scaling"
        / "ted_event_layer_scaling_summary.tsv"
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.94))
    fig.suptitle("Module-level scaling benchmark", fontsize=14, fontweight="bold", x=0.04, ha="left")

    ax = axes[0]
    panel(ax, "A", "Runtime")
    ax.plot(scaling["cells"], scaling["median_upstream_seconds"], marker="o", color=BLUE, linewidth=2.0, label="Upstream module scoring")
    ax.fill_between(scaling["cells"], scaling["iqr_low_upstream_seconds"], scaling["iqr_high_upstream_seconds"], color=BLUE_LIGHT, alpha=0.65)
    ax.plot(scaling["cells"], scaling["median_event_layer_seconds"], marker="s", markerfacecolor="white", color=GOLD, linewidth=2.0, label="TED event layer")
    ax.fill_between(scaling["cells"], scaling["iqr_low_event_layer_seconds"], scaling["iqr_high_event_layer_seconds"], color=GOLD_LIGHT, alpha=0.55)
    ax.set_xscale("log")
    ax.set_xlabel("Synthetic cells")
    ax.set_ylabel("Seconds")
    ax.legend(frameon=False, fontsize=8.5)
    clean_axis(ax, "both")

    ax = axes[1]
    panel(ax, "B", "Peak resident memory")
    ax.plot(scaling["cells"], scaling["median_peak_rss_mb"], marker="o", color=BLUE, linewidth=2.0)
    ax.fill_between(scaling["cells"], scaling["iqr_low_peak_rss_mb"], scaling["iqr_high_peak_rss_mb"], color=BLUE_LIGHT, alpha=0.65)
    ax.set_xscale("log")
    ax.set_xlabel("Synthetic cells")
    ax.set_ylabel("Peak RSS (MB)")
    last = scaling.iloc[-1]
    ax.annotate(f"{last['median_peak_rss_mb']:.1f} MB", (last["cells"], last["median_peak_rss_mb"]), xytext=(-70, -20), textcoords="offset points", fontsize=9.0, fontweight="bold")
    clean_axis(ax, "both")

    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=2.5)
    save_supplement_figure(fig, "supplementary_figure_s1_scaling")


def supplementary_event_grammar_figure() -> None:
    event_types = ["onset", "peak", "shutdown", "branch-specific", "spatial-localized", "failed-rescue\nprediction-only"]
    event_counts = [8, 14, 10, 14, 6, 4]
    datasets = ["GSE123013\nArabidopsis root", "GSE271399\nerythroid", "Paul15\nbranch", "Bombyx\nStereo-seq"]
    coverage = np.array([[0, 0, 0, 4, 0, 0], [0, 0, 3, 0, 0, 4], [8, 8, 7, 10, 0, 0], [0, 6, 0, 0, 6, 0]])
    required_fields = [
        "pathway",
        "event_type",
        "event_FDR",
        "event_FDR_available",
        "robustness_score",
        "evidence_boundary",
        "supported_interpretation",
        "unsupported_interpretation",
    ]
    pd.DataFrame({"event_type": event_types, "event_rows": event_counts}).to_csv(
        SOURCE / "supplementary_figure_s4_event_type_counts.tsv", sep="\t", index=False
    )
    pd.DataFrame(coverage, index=datasets, columns=event_types).rename_axis("dataset").reset_index().to_csv(
        SOURCE / "supplementary_figure_s4_dataset_coverage.tsv", sep="\t", index=False
    )
    pd.DataFrame({"required_field": required_fields}).to_csv(
        SOURCE / "supplementary_figure_s4_required_fields.tsv", sep="\t", index=False
    )

    fig = plt.figure(figsize=(13.2, 7.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.20, 0.95], hspace=0.48, wspace=0.42)
    fig.suptitle("Dynamic pathway event grammar standardization", fontsize=15, fontweight="bold", x=0.04, ha="left")

    ax = fig.add_subplot(gs[:, 0])
    panel(ax, "A", "From pathway curves to event grammar")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = [
        (0.77, "Pathway/module activity curve", BLUE_LIGHT),
        (0.54, "Detect ordered features\nonset, peak, shutdown", "#E7F2E7"),
        (0.31, "Attach context\nbranch, spatial, rescue", "#FFF3D8"),
        (0.08, "Write event table row\nFDR, robustness, boundary", "#F2E5F2"),
    ]
    for y, text, face in steps:
        patch = FancyBboxPatch((0.07, y), 0.86, 0.14, boxstyle="round,pad=0.015,rounding_size=0.02", facecolor=face, edgecolor=MID, linewidth=1.0)
        ax.add_patch(patch)
        ax.text(0.50, y + 0.07, text, ha="center", va="center", fontsize=9.0)
    for y1, y2 in [(0.77, 0.68), (0.54, 0.45), (0.31, 0.22)]:
        arrow(ax, (0.50, y1), (0.50, y2), INK)
    ax.text(0.09, 0.025, "Output: dynamic_pathway_event_table.tsv", fontsize=8.5, fontweight="bold")

    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "B", "Event-type coverage")
    colors = [BLUE, "#59A14F", "#E15759", "#F28E2B", "#B07AA1", "#9C755F"]
    y = np.arange(len(event_types))
    ax.barh(y, event_counts, color=colors)
    ax.set_yticks(y, event_types, fontsize=8.2)
    ax.invert_yaxis()
    ax.set_xlabel("Event rows")
    ax.text(0.98, 0.05, "N = 56 standardized rows", transform=ax.transAxes, ha="right", fontsize=8.0)
    clean_axis(ax, "x")

    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "C", "Dataset x event-type coverage")
    im = ax.imshow(coverage, cmap="Blues", vmin=0, vmax=10, aspect="auto")
    ax.set_yticks(range(len(datasets)), datasets, fontsize=8.0)
    ax.set_xticks(range(len(event_types)), event_types, rotation=40, ha="right", fontsize=7.8)
    for i in range(coverage.shape[0]):
        for j in range(coverage.shape[1]):
            ax.text(j, i, str(int(coverage[i, j])), ha="center", va="center", fontsize=8.0, color="white" if coverage[i, j] >= 4 else INK)

    ax = fig.add_subplot(gs[:, 2])
    panel(ax, "D", "Required event-row fields")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    top = 0.86
    height = 0.085
    for idx, field in enumerate(required_fields):
        y0 = top - idx * 0.095
        patch = FancyBboxPatch((0.06, y0), 0.88, height, boxstyle="round,pad=0.012,rounding_size=0.015", facecolor="#F4F4F4", edgecolor="#AAB4BE", linewidth=0.9)
        ax.add_patch(patch)
        ax.text(0.10, y0 + height / 2, field, va="center", fontsize=8.5, fontweight="bold")
    ax.text(0.07, 0.035, "Scores describe activity changes; TED rows add event type, robustness and an auditable evidence boundary.", fontsize=8.2, wrap=True)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_supplement_figure(fig, "supplementary_figure_s4_dynamic_pathway_event_grammar")


def figure5() -> None:
    known = ROOT / "results" / "ted_known_source_validation" / "tables"
    scp_root = ROOT / "data" / "processed" / "ted_known_source" / "SCP1064" / "results"
    gse153 = read_tsv(known / "gse153056_pdl1_outcome_alignment.tsv")
    scp = read_tsv(scp_root / "scp1064_heavy_shuffle_summary.tsv")
    gse153_block = read_tsv(RESULTS / "gse153056_block_aware" / "gse153056_block_event_support.tsv")
    gse271_day = read_tsv(ROOT / "data_external" / "deliverables_all_ted_rounds" / "GSE271399_T21_GATA1s" / "gse271399_day_stratified_event_consistency.tsv")
    erythroid_family = {
        "ERYTHROID_MATURATION",
        "HEME_GLOBIN",
        "GATA_KLF_TAL1_REGULON",
        "IRON_TRANSPORT",
        "HALLMARK_HEME_METABOLISM",
    }
    gse271_family_day = (
        gse271_day[
            (gse271_day["trajectory"] == "erythroid")
            & (gse271_day["contrast"] == "T21_GATA1s_vs_T21_wtGATA1")
            & gse271_day["pathway"].isin(erythroid_family)
        ]
        .groupby("day", as_index=False)
        .agg(
            n_family_members=("pathway", "nunique"),
            mean_delta_auc_day=("delta_auc_day", "mean"),
            all_negative=("delta_auc_day", lambda values: bool((values < 0).all())),
        )
    )
    gse271_family_day["day_order"] = gse271_family_day["day"].map({"D7": 7, "D9": 9, "D11": 11})
    gse271_family_day = gse271_family_day.sort_values("day_order").drop(columns="day_order")

    gse153.to_csv(SOURCE / "figure5_gse153056_rna_protein.tsv", sep="\t", index=False)
    scp.to_csv(SOURCE / "figure5_scp1064_heavy_shuffle.tsv", sep="\t", index=False)
    gse271_day.to_csv(SOURCE / "figure5_gse271399_day_consistency.tsv", sep="\t", index=False)
    gse271_family_day.to_csv(SOURCE / "figure5_gse271399_family_day_summary.tsv", sep="\t", index=False)
    evidence = pd.DataFrame(
        [
            ["GSE153056 (STAT1)", 1, "outcome_passed", "E1 | protein outcome passed", "independent-block q=0.0556 passes E1; E2 sensitivity post hoc"],
            ["SCP1064", 1, "outcome_failed", "E1 | protein outcome failed", "mandatory guide-label shuffle passed 1/4 axes"],
            ["GSE271399", 1, "none_passed", "E1 | no qualified parallel record", "three-day direction consistency; no independent biological replicate"],
        ],
        columns=["dataset", "event_support", "parallel_record_status", "descriptor", "basis_or_limit"],
    )
    evidence.to_csv(SOURCE / "figure5_evidence_descriptors.tsv", sep="\t", index=False)

    fig = plt.figure(figsize=(12.6, 8.2))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34, height_ratios=[1.25, 0.75])
    fig.suptitle("Orthogonal public-data readouts with bounded interpretation", fontsize=15, fontweight="bold", x=0.04, ha="left")

    ax = fig.add_subplot(gs[0, 0])
    panel(ax, "A", "GSE153056 RNA event vs PD-L1 protein")
    ax.scatter(gse153["event_effect_size"], gse153["pdl1_protein_effect_size"], s=38, color=BLUE_MID, edgecolor=INK, linewidth=0.5, alpha=0.85)
    for label in ["IFNGR1", "IFNGR2", "JAK2", "STAT1", "CMTM6"]:
        row = gse153[gse153["perturbation"] == label]
        if not row.empty:
            x = float(row.iloc[0]["event_effect_size"])
            y = float(row.iloc[0]["pdl1_protein_effect_size"])
            ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8.8)
    ax.axhline(0, color=GRID, linewidth=0.8)
    ax.axvline(0, color=GRID, linewidth=0.8)
    ax.set_xlabel("RNA-event effect")
    ax.set_ylabel("PD-L1 protein effect")
    stat1 = gse153_block[gse153_block["perturbation"].eq("STAT1")].iloc[0]
    ax.text(
        0.04,
        0.94,
        f"Across perturbations: Spearman 0.785\nDirection match 0.769\nSTAT1: event q={stat1['block_q_value']:.4f}; E1\nprotein outcome passed; E2 sensitivity post hoc",
        transform=ax.transAxes,
        va="top",
        fontsize=9.0,
        bbox=dict(facecolor="white", edgecolor=GRID, pad=3),
    )
    clean_axis(ax, None)

    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "B", "SCP1064 mandatory-control failure")
    scp = scp.copy()
    scp["observed_to_q95"] = scp["observed_spearman"].abs() / scp["null_q95_abs"].replace(0, np.nan)
    group_order = [
        "rna_protein_cell_pairing_within_block",
        "guide_labels_within_block",
        "guide_to_target_assignment",
        "expression_matched_random_gene_set",
        "protein_outcome_column_mapping",
    ]
    short_names = ["Cell pairing", "Guide labels", "Guide-target", "Random sets", "Protein columns"]
    positions = []
    labels_y = []
    y_offset = 0
    for group, label in zip(group_order, short_names):
        subset = scp[scp["shuffle_type"] == group]
        ys = np.arange(len(subset)) + y_offset
        passed = subset["gate_pass"].astype(str).str.lower().eq("true").to_numpy()
        ax.scatter(subset.loc[passed, "observed_to_q95"], ys[passed], s=40, color=BLUE, edgecolor=INK, linewidth=0.5)
        ax.scatter(subset.loc[~passed, "observed_to_q95"], ys[~passed], s=58, color=FAIL, marker="x", linewidths=1.6)
        positions.append(float(ys.mean()))
        labels_y.append(f"{label} ({passed.sum()}/{len(passed)})")
        y_offset += len(subset) + 1
    ax.axvline(1, color=GOLD, linestyle="--", linewidth=1.2, label="Observed = null q95")
    ax.set_yticks(positions, labels_y, fontsize=8.8)
    ax.set_xlabel("|observed correlation| / null q95")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.text(0.03, 0.97, "Event remains E1; guide-label shuffling passed 1/4 axes;\nprotein-outcome qualification failed", transform=ax.transAxes, va="top", fontsize=8.8, color=FAIL, bbox=dict(facecolor="white", edgecolor=GRID, pad=2.5))
    clean_axis(ax, "x")

    ax = fig.add_subplot(gs[1, :])
    panel(ax, "C", "GATA1/T21 evidence boundary")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, (0.02, 0.17), 0.28, 0.56, "Observed: E1 event", "All five erythroid-family members were negative at D7, D9 and D11.", BLUE_LIGHT, title_size=10.0, body_size=8.6, wrap_width=25)
    box(ax, (0.36, 0.17), 0.28, 0.56, "Design sensitivity", "Twenty-one day-by-pseudotime-bin-by-state strata; sign-flip q=0.0008. No biological-replicate inference.", OPEN, title_size=10.0, body_size=8.2, wrap_width=27)
    box(ax, (0.70, 0.17), 0.28, 0.56, "Missing matched rescue", "Same-system full-length GATA1 rescue with molecular and phenotypic recovery.", GOLD_LIGHT, edge=FAIL, title_size=10.0, body_size=8.5, linestyle="--", wrap_width=25)
    arrow(ax, (0.30, 0.48), (0.36, 0.48), MID)
    arrow(ax, (0.64, 0.48), (0.70, 0.48), FAIL, linestyle="--")

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, "figure5_independent_real_data_validation")


def graphical_abstract() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.1))
    fig.suptitle(
        "TED turns dynamic pathway signals into artifact-aware event records",
        fontsize=15,
        fontweight="bold",
        x=0.04,
        ha="left",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(
        ax,
        (0.03, 0.34),
        0.20,
        0.42,
        "Dynamic single-cell input",
        "Trajectory, perturbation, time course or pathway score with biological blocks",
        BLUE_LIGHT,
        title_size=9.5,
        body_size=8.0,
    )
    box(
        ax,
        (0.29, 0.34),
        0.20,
        0.42,
        "TED event inference",
        "Detect event family, timing and direction; audit matched state, controls and ambiguity",
        OPEN,
        title_size=9.5,
        body_size=8.0,
    )
    box(
        ax,
        (0.55, 0.34),
        0.18,
        0.42,
        "Evidence records",
        "E0-E2 event support plus parallel outcome, reversal and rescue records",
        GOLD_LIGHT,
        title_size=9.5,
        body_size=8.0,
    )
    box(
        ax,
        (0.79, 0.34),
        0.18,
        0.42,
        "Bounded interpretation",
        "State supported meaning, the evidence boundary and next validation",
        OPEN,
        title_size=9.5,
        body_size=7.5,
    )
    arrow(ax, (0.23, 0.55), (0.29, 0.55), BLUE)
    arrow(ax, (0.49, 0.55), (0.55, 0.55), BLUE)
    arrow(ax, (0.73, 0.55), (0.79, 0.55), GOLD)
    ax.text(
        0.50,
        0.13,
        "Detection, artifact control and biological validation remain separate, auditable targets.",
        ha="center",
        fontsize=9,
        color=MID,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(fig, "graphical_abstract")


def write_manifest() -> None:
    rows = []
    allowed_names = {f"{stem}{suffix}" for stem in CURRENT_FIGURE_STEMS for suffix in (".pdf", ".png")}
    for path in sorted((RESULTS / "figures").glob("*.*")):
        if path.name not in allowed_names:
            continue
        rows.append(
            {
                "file": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(RESULTS / "figure_manifest.tsv", sep="\t", index=False)

    upload_rows = []
    for row in rows:
        name = Path(row["file"]).name
        stem = Path(name).stem
        if stem == "graphical_abstract":
            label = "Graphical abstract"
        else:
            label = f"Figure {CURRENT_FIGURE_STEMS.index(stem) + 1}"
        upload_rows.append(
            {
                "figure": label,
                "file": name,
                "format": Path(name).suffix.lstrip(".").upper(),
                "bytes": row["bytes"],
                "sha256": row["sha256"],
                "upload_note": "current July 2026 composite; title and legend are in the manuscript",
            }
        )
    upload_manifest = pd.DataFrame(upload_rows)
    for destination in FIGURE_DESTINATIONS[:2]:
        upload_manifest.to_csv(
            destination / "final_main_figure_upload_manifest.tsv", sep="\t", index=False
        )


def main() -> None:
    setup_style()
    SOURCE.mkdir(parents=True, exist_ok=True)
    clean_legacy_outputs()
    figure1()
    figure2()
    # Retain the factorized contract matrices as Supplementary Figure S6, then
    # overwrite its historical main-figure rendering with the locked common task.
    figure3_factorized()
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_nearest_method_locked_summary.py")],
        cwd=ROOT,
        check=True,
    )
    figure4()
    supplementary_scaling_figure()
    supplementary_event_grammar_figure()
    # The historical public-case composite still emits supporting source tables;
    # the frozen BNT162b2 builder owns the current manuscript Figure 5.
    figure5()
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_bnt162b2_flagship_figure.py")],
        cwd=ROOT,
        check=True,
    )
    graphical_abstract()
    write_manifest()
    print(f"Wrote BIB figures and source tables under {RESULTS}")


if __name__ == "__main__":
    main()
