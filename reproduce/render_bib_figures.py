#!/usr/bin/env python3
"""Redraw BIB Figures 3 and 5 from packaged source tables only.

This renderer intentionally starts at the frozen figure-source layer. It does
not rerun the 480 raw-count tasks or the BNT162b2/GSE171964 analyses. Its
semantic report binds every plotted table and status JSON by SHA-256 and checks
the manuscript headline values before an image is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402


METHODS = ["TIPS", "scTransient", "tradeSeq", "score_then_smooth", "TED"]
METHOD_LABELS = ["TIPS", "scTransient", "tradeSeq", "Score + smooth", "TED"]
COLORS = ["#7A8B99", "#5B8FF9", "#61B15A", "#C28A2C", "#2F5D8A"]
INK = "#24313D"
MID = "#65727E"
GRID = "#D8DEE4"
FAIL = "#B84A4A"
PASS = "#3C8C62"
SOURCE_BASE = Path("results/ted_v1_submission/figure_source_data")


class FigureRenderError(RuntimeError):
    """A required source or semantic contract failed."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(root: Path, relative: Path) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise FigureRenderError(f"source escapes companion root: {relative}") from exc
    if not path.is_file():
        raise FigureRenderError(f"required packaged source is missing: {relative}")
    return path


def read_tsv(root: Path, relative: Path) -> pd.DataFrame:
    return pd.read_csv(require_file(root, relative), sep="\t")


def read_json(root: Path, relative: Path) -> dict[str, Any]:
    payload = json.loads(require_file(root, relative).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FigureRenderError(f"JSON source is not an object: {relative}")
    return payload


def close_enough(actual: float, expected: float, tolerance: float = 0.002) -> bool:
    return bool(np.isfinite(actual) and abs(actual - expected) <= tolerance)


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    observed: Any,
    expected: Any,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
        }
    )


def source_records(root: Path, relatives: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": relative.as_posix(),
            "bytes": require_file(root, relative).stat().st_size,
            "sha256": sha256_path(require_file(root, relative)),
        }
        for relative in relatives
    ]


def setup_style() -> None:
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


def save_and_check(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    with Image.open(png) as image:
        grayscale = np.asarray(image.convert("L"), dtype=float)
        width, height = image.size
    standard_deviation = float(grayscale.std())
    pixel_pass = width >= 1000 and height >= 600 and standard_deviation >= 5.0
    outputs = [
        {
            "path": png.name,
            "bytes": png.stat().st_size,
            "sha256": sha256_path(png),
        },
        {
            "path": pdf.name,
            "bytes": pdf.stat().st_size,
            "sha256": sha256_path(pdf),
        },
    ]
    pixel_qa = {
        "passed": pixel_pass,
        "png_width": width,
        "png_height": height,
        "grayscale_standard_deviation": standard_deviation,
        "minimum_width": 1000,
        "minimum_height": 600,
        "minimum_grayscale_standard_deviation": 5.0,
    }
    return outputs, pixel_qa


def render_figure3(root: Path, output_dir: Path) -> dict[str, Any]:
    names = [
        "figure3_clean_common_task_metrics.tsv",
        "figure3_type_specific_clean_metrics.tsv",
        "figure3_low_signal_noisy_coordinate_metrics.tsv",
        "figure3_artifact_common_task_metrics.tsv",
    ]
    relatives = [SOURCE_BASE / name for name in names]
    clean, typed, hard, artifact = [
        read_tsv(root, relative) for relative in relatives
    ]
    checks: list[dict[str, Any]] = []
    for name, frame, expected_rows in zip(
        names,
        [clean, typed, hard, artifact],
        [300, 1800, 600, 1800],
        strict=True,
    ):
        add_check(
            checks,
            f"{name}:row_count",
            len(frame) == expected_rows,
            len(frame),
            expected_rows,
        )
        observed_methods = sorted(frame["method"].dropna().astype(str).unique())
        add_check(
            checks,
            f"{name}:five_methods",
            set(observed_methods) == set(METHODS),
            observed_methods,
            sorted(METHODS),
        )

    event_types = ["activation", "suppression", "transient"]
    expected_typed = {
        "activation": [0.855, 0.631, 0.599, 0.682, 0.580],
        "suppression": [0.447, 0.239, 0.287, 0.375, 0.447],
        "transient": [0.102, 0.716, 0.681, 0.499, 0.422],
    }
    typed_means: dict[str, list[float]] = {}
    for event_type in event_types:
        values = [
            float(
                typed.loc[
                    typed["method"].eq(method)
                    & typed["event_type"].eq(event_type),
                    "pathway_auprc",
                ].mean()
            )
            for method in METHODS
        ]
        typed_means[event_type] = values
        add_check(
            checks,
            f"type_specific_mean:{event_type}",
            all(
                close_enough(actual, expected)
                for actual, expected in zip(
                    values,
                    expected_typed[event_type],
                    strict=True,
                )
            ),
            values,
            expected_typed[event_type],
        )

    expected_hard = [0.674, 0.671, 0.751, 0.846, 0.764]
    hard_means = [
        float(hard.loc[hard["method"].eq(method), "pathway_level_auprc"].mean())
        for method in METHODS
    ]
    add_check(
        checks,
        "low_signal_noisy_coordinate_means",
        all(
            close_enough(actual, expected)
            for actual, expected in zip(hard_means, expected_hard, strict=True)
        ),
        hard_means,
        expected_hard,
    )
    expected_clean = [0.803, 0.994, 1.000, 1.000, 1.000]
    clean_means = [
        float(
            clean.loc[
                clean["method"].eq(method),
                "pathway_level_auprc",
            ].mean()
        )
        for method in METHODS
    ]
    add_check(
        checks,
        "clean_true_coordinate_means",
        all(
            close_enough(actual, expected)
            for actual, expected in zip(clean_means, expected_clean, strict=True)
        ),
        clean_means,
        expected_clean,
    )
    risk_column = "matched_top_k_artifact_false_promotion_rate"
    expected_risk = [0.890, 0.881, 0.745, 0.614, 0.811]
    risk_means = [
        float(artifact.loc[artifact["method"].eq(method), risk_column].mean())
        for method in METHODS
    ]
    add_check(
        checks,
        "pooled_artifact_risk_means",
        all(
            close_enough(actual, expected)
            for actual, expected in zip(risk_means, expected_risk, strict=True)
        ),
        risk_means,
        expected_risk,
    )
    if not all(check["passed"] for check in checks):
        raise FigureRenderError("Figure 3 semantic checks failed")

    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4))
    fig.suptitle(
        "Controlled raw-count nearest-method common task",
        fontsize=15,
        fontweight="bold",
        x=0.04,
        ha="left",
    )

    ax = axes[0, 0]
    x = np.arange(len(event_types))
    offsets = np.linspace(-0.28, 0.28, len(METHODS))
    for offset, method, label, color in zip(
        offsets,
        METHODS,
        METHOD_LABELS,
        COLORS,
        strict=True,
    ):
        means: list[float] = []
        sems: list[float] = []
        for event_type in event_types:
            values = typed.loc[
                typed["method"].eq(method)
                & typed["event_type"].eq(event_type),
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
            markeredgecolor=INK,
            markeredgewidth=0.45,
            markersize=6.2,
            capsize=2.4,
            linewidth=1.1,
            label=label,
        )
    ax.set_xticks(x, ["Activation", "Suppression", "Transient"])
    ax.set_ylim(0, 1.02)
    ax.set_title("A. Type-specific pathway AUPRC", loc="left")
    ax.set_ylabel("Pathway AUPRC")
    ax.legend(frameon=False, fontsize=6.8, ncol=2, loc="lower right")
    ax.grid(axis="y", color=GRID, linewidth=0.6)

    ax = axes[0, 1]
    hard_values = [
        hard.loc[hard["method"].eq(method), "pathway_level_auprc"].to_numpy(
            dtype=float
        )
        for method in METHODS
    ]
    boxes = ax.boxplot(
        hard_values,
        tick_labels=METHOD_LABELS,
        patch_artist=True,
        showfliers=False,
    )
    for patch, color in zip(boxes["boxes"], COLORS, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_ylim(0, 1.02)
    ax.set_title("B. Low-signal, noisy-coordinate AUPRC", loc="left")
    ax.set_ylabel("Pathway AUPRC")
    ax.tick_params(axis="x", rotation=22)
    ax.grid(axis="y", color=GRID, linewidth=0.6)

    ax = axes[1, 0]
    artifacts = ["composition", "stress", "partial_batch_time"]
    x = np.arange(len(artifacts))
    width = 0.15
    for index, (method, label, color) in enumerate(
        zip(METHODS, METHOD_LABELS, COLORS, strict=True)
    ):
        means = [
            float(
                artifact.loc[
                    artifact["method"].eq(method)
                    & artifact["artifact"].eq(artifact_name),
                    risk_column,
                ].mean()
            )
            for artifact_name in artifacts
        ]
        ax.bar(
            x + (index - 2) * width,
            means,
            width,
            label=label,
            color=color,
        )
    ax.set_xticks(x, ["Composition", "Stress", "Batch/time"])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Matched top-k artifact false-promotion rate")
    ax.set_title("C. Artifact false-promotion at matched top-k", loc="left")
    ax.legend(frameon=False, fontsize=6.8, ncol=2)
    ax.grid(axis="y", color=GRID, linewidth=0.6)

    ax = axes[1, 1]
    for method, label, color, x_value, y_value in zip(
        METHODS,
        METHOD_LABELS,
        COLORS,
        risk_means,
        clean_means,
        strict=True,
    ):
        ax.scatter(
            x_value,
            y_value,
            s=90,
            color=color,
            edgecolor=INK,
            linewidth=0.6,
        )
        ax.annotate(
            label,
            (x_value, y_value),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Matched top-k artifact false-promotion rate")
    ax.set_ylabel("Clean-scenario pathway AUPRC")
    ax.set_title("D. Accuracy-risk plane", loc="left")
    ax.grid(color=GRID, linewidth=0.6)
    ax.text(
        0.02,
        0.04,
        "Detection-level comparison; not an E-level claim",
        transform=ax.transAxes,
        fontsize=7.2,
        color=MID,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.95))
    outputs, pixel_qa = save_and_check(
        fig,
        output_dir,
        "figure3_reproduced",
    )
    add_check(
        checks,
        "generated_png_pixel_qa",
        bool(pixel_qa["passed"]),
        pixel_qa,
        "nonblank PNG >=1000x600",
    )
    report = {
        "figure": "figure3",
        "status": "passed" if all(check["passed"] for check in checks) else "failed",
        "scope": (
            "source-data redraw from four packaged tables; raw-count tasks were "
            "not rerun"
        ),
        "source_count": len(relatives),
        "sources": source_records(root, relatives),
        "checks": checks,
        "outputs": outputs,
        "pixel_qa": pixel_qa,
    }
    return report


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def render_figure5(root: Path, output_dir: Path) -> dict[str, Any]:
    names = [
        "figure5_primary_rna_trajectory.tsv",
        "figure5_primary_protein_trajectory.tsv",
        "figure5_rna_protein_donor_contrasts.tsv",
        "figure5_gse171964_blind_qc.tsv",
        "figure5_flagship_design.tsv",
        "figure5_rna_gate_audit.tsv",
        "figure5_evidence_status.tsv",
    ]
    relatives = [SOURCE_BASE / name for name in names]
    rna, protein, aligned, qc, design, gates, evidence = [
        read_tsv(root, relative) for relative in relatives
    ]
    status_relatives = [
        Path(
            "results/ted_bnt162b2_flagship/rna_event_freeze_v1/"
            "rna_event_status.json"
        ),
        Path(
            "results/ted_bnt162b2_flagship/orthogonal_outcome_v1/"
            "protein_outcome_status.json"
        ),
        Path(
            "results/ted_gse171964_replication/analysis_v1/"
            "replication_status.json"
        ),
    ]
    rna_status, protein_status, replication_status = [
        read_json(root, relative) for relative in status_relatives
    ]
    checks: list[dict[str, Any]] = []
    for name, frame, expected_rows in zip(
        names,
        [rna, protein, aligned, qc, design, gates, evidence],
        [24, 24, 6, 24, 72, 5, 1],
        strict=True,
    ):
        add_check(
            checks,
            f"{name}:row_count",
            len(frame) == expected_rows,
            len(frame),
            expected_rows,
        )
    design_shape = {
        "donors": int(design["donor_id"].nunique()),
        "days": int(design["day"].nunique()),
        "modalities": sorted(design["modality"].astype(str).unique()),
    }
    add_check(
        checks,
        "locked_design_shape",
        design_shape
        == {
            "donors": 6,
            "days": 4,
            "modalities": ["ATAC", "RNA", "protein_ADT"],
        },
        design_shape,
        "6 donors x 4 days x RNA/protein_ADT/ATAC",
    )
    gate_pass = bool_series(gates["passed"])
    add_check(
        checks,
        "rna_gate_pattern",
        int(gate_pass.sum()) == 2 and int((~gate_pass).sum()) == 3,
        {"passed": int(gate_pass.sum()), "failed": int((~gate_pass).sum())},
        {"passed": 2, "failed": 3},
    )
    evidence_row = evidence.iloc[0].astype(str).to_dict()
    expected_evidence = {
        "primary_event_support": "E0",
        "primary_protein_outcome_status": "passed",
        "event_replication_eligibility_status": "failed",
        "event_replication_test_status": "not_run",
        "event_replication_status": "not_evaluable",
        "protein_outcome_replication_status": "not_tested",
    }
    observed_evidence = {
        key: evidence_row.get(key) for key in expected_evidence
    }
    add_check(
        checks,
        "bounded_evidence_states",
        observed_evidence == expected_evidence,
        observed_evidence,
        expected_evidence,
    )
    observed_status = {
        "primary_event_support": (
            rna_status.get("event_support", {}).get("code")
            if isinstance(rna_status.get("event_support"), dict)
            else None
        ),
        "primary_protein_outcome_status": protein_status.get(
            "protein_outcome_status"
        ),
        "event_replication_eligibility_status": replication_status.get(
            "event_replication_eligibility_status"
        ),
        "event_replication_test_status": replication_status.get(
            "event_replication_test_status"
        ),
        "event_replication_status": replication_status.get(
            "event_replication_status"
        ),
        "protein_outcome_replication_status": (
            replication_status.get("protein_outcome", {}).get(
                "replication_status"
            )
            if isinstance(replication_status.get("protein_outcome"), dict)
            else replication_status.get("outcome_replication_status")
        ),
    }
    add_check(
        checks,
        "status_json_matches_figure_table",
        observed_status == expected_evidence,
        observed_status,
        expected_evidence,
    )
    qc_pass = bool_series(qc["blind_qc_pass"])
    evaluable_donors = int(qc.loc[qc_pass].groupby("pt_id")["day"].nunique().eq(4).sum())
    add_check(
        checks,
        "replication_evaluable_donors",
        evaluable_donors == 4
        and replication_status.get("n_evaluable_donors") == 4,
        {
            "from_figure_source": evaluable_donors,
            "from_status_json": replication_status.get("n_evaluable_donors"),
        },
        4,
    )
    if not all(check["passed"] for check in checks):
        raise FigureRenderError("Figure 5 semantic checks failed")

    setup_style()
    fig = plt.figure(figsize=(14.2, 9.1), constrained_layout=True)
    grid = fig.add_gridspec(2, 6, height_ratios=[0.92, 1.08])
    axes = [
        fig.add_subplot(grid[0, :2]),
        fig.add_subplot(grid[0, 2:]),
        fig.add_subplot(grid[1, :2]),
        fig.add_subplot(grid[1, 2:4]),
        fig.add_subplot(grid[1, 4:]),
    ]
    fig.suptitle(
        "Masked-outcome boundary audit and fail-closed replication attempt",
        fontsize=14.2,
        fontweight="bold",
        x=0.03,
        ha="left",
    )

    ax = axes[0]
    donors = sorted(design["donor_id"].astype(str).unique())
    days = sorted(design["day"].unique())
    for index, donor in enumerate(donors):
        ax.plot(range(len(days)), [index] * len(days), color=GRID, lw=0.8)
        ax.scatter(range(len(days)), [index] * len(days), color=COLORS[-1], s=35)
    ax.set_xticks(range(len(days)), [f"day {day}" for day in days])
    ax.set_yticks(range(len(donors)), donors)
    ax.invert_yaxis()
    ax.set_title("A. Locked design and masking", loc="left")
    ax.text(
        0.02,
        0.04,
        "RNA first | protein masked | ATAC absent",
        transform=ax.transAxes,
        fontsize=8,
        color=MID,
    )

    ax = axes[1]
    for donor, part in rna.groupby("donor_id"):
        ordered = part.sort_values("day")
        ax.plot(
            ordered["day"],
            ordered["IFN_alpha_score"],
            marker="o",
            alpha=0.55,
            lw=1,
        )
    summary = rna.groupby("day")["IFN_alpha_score"].agg(["mean", "sem"]).reset_index()
    ax.errorbar(
        summary["day"],
        summary["mean"],
        yerr=summary["sem"],
        color=INK,
        marker="o",
        linewidth=2.2,
        capsize=3,
        label="mean ± SE",
    )
    ax.set_title("B. Donor IFN RNA trajectory", loc="left")
    ax.set_xlabel("Day")
    ax.set_ylabel("IFN-alpha pathway score")
    ax.legend(frameon=False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)

    ax = axes[2]
    ax.axhline(0, color=GRID, lw=0.8)
    ax.axvline(0, color=GRID, lw=0.8)
    ax.scatter(
        aligned["RNA_transient"],
        aligned["protein_transient"],
        color=[
            PASS if value > 0 else FAIL
            for value in aligned["RNA_transient"]
        ],
        s=55,
    )
    for row in aligned.itertuples(index=False):
        ax.annotate(
            str(row.donor_id),
            (row.RNA_transient, row.protein_transient),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    gate_text = " | ".join(
        f"{'PASS' if passed else 'FAIL'} {gate}"
        for gate, passed in zip(gates["gate"], gate_pass, strict=True)
    )
    ax.set_title("C. RNA gates and protein alignment", loc="left")
    ax.set_xlabel("RNA transient contrast")
    ax.set_ylabel("CD64/CD169 transient contrast")
    ax.text(
        0.01,
        0.02,
        gate_text,
        transform=ax.transAxes,
        fontsize=6.5,
        wrap=True,
    )

    ax = axes[3]
    for donor, part in protein.groupby("donor_id"):
        ordered = part.sort_values("day")
        ax.plot(
            ordered["day"],
            ordered["CD64_CD169_index"],
            marker="o",
            alpha=0.55,
            lw=1,
        )
    protein_summary = (
        protein.groupby("day")["CD64_CD169_index"]
        .agg(["mean", "sem"])
        .reset_index()
    )
    ax.errorbar(
        protein_summary["day"],
        protein_summary["mean"],
        yerr=protein_summary["sem"],
        color=PASS,
        marker="o",
        linewidth=2.2,
        capsize=3,
    )
    ax.set_title("D. Unmasked protein outcome", loc="left")
    ax.set_xlabel("Day")
    ax.set_ylabel("CD64/CD169 index")
    ax.text(
        0.02,
        0.96,
        "protein outcome passed; RNA event remains E0",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    ax.grid(axis="y", color=GRID, linewidth=0.6)

    ax = axes[4]
    participant_order = sorted(qc["pt_id"].astype(str).unique())
    x_lookup = {participant: index for index, participant in enumerate(participant_order)}
    day_order = sorted(qc["day"].unique())
    offsets = dict(zip(day_order, np.linspace(-0.24, 0.24, len(day_order)), strict=True))
    for day in day_order:
        part = qc.loc[qc["day"].eq(day)]
        xs = [
            x_lookup[str(participant)] + offsets[day]
            for participant in part["pt_id"]
        ]
        colors = [PASS if passed else FAIL for passed in bool_series(part["blind_qc_pass"])]
        ax.scatter(xs, part["max_abs_mad_z"], c=colors, s=38, label=f"day {day}")
    ax.axhline(3.0, color=FAIL, lw=1.2, ls="--")
    ax.set_xticks(range(len(participant_order)), participant_order, rotation=25)
    ax.set_title("E. Corrected-v2 replication eligibility", loc="left")
    ax.set_ylabel("Maximum absolute MAD z")
    ax.text(
        0.98,
        0.98,
        (
            "eligibility failed\n"
            "event test not_run\n"
            "event not_evaluable\n"
            "protein not_tested"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": GRID,
        },
    )
    outputs, pixel_qa = save_and_check(
        fig,
        output_dir,
        "figure5_reproduced",
    )
    add_check(
        checks,
        "generated_png_pixel_qa",
        bool(pixel_qa["passed"]),
        pixel_qa,
        "nonblank PNG >=1000x600",
    )
    all_relatives = relatives + status_relatives
    report = {
        "figure": "figure5",
        "status": "passed" if all(check["passed"] for check in checks) else "failed",
        "scope": (
            "source-data redraw from seven packaged tables plus three frozen "
            "status JSON records; upstream case-study analyses were not rerun"
        ),
        "source_count": len(relatives),
        "status_json_count": len(status_relatives),
        "sources": source_records(root, all_relatives),
        "checks": checks,
        "outputs": outputs,
        "pixel_qa": pixel_qa,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redraw one BIB figure from packaged source tables"
    )
    parser.add_argument("--companion-root", type=Path, required=True)
    parser.add_argument(
        "--figure",
        choices=["figure3", "figure5"],
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.companion_root.resolve()
    if not root.is_dir():
        raise FigureRenderError(f"companion root is missing: {root}")
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else root / args.output_dir
    ).resolve()
    try:
        output_dir.relative_to(root)
    except ValueError as exc:
        raise FigureRenderError(
            "output directory must stay inside the extracted companion"
        ) from exc
    report = (
        render_figure3(root, output_dir)
        if args.figure == "figure3"
        else render_figure5(root, output_dir)
    )
    report_path = output_dir / f"{args.figure}_semantic_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
