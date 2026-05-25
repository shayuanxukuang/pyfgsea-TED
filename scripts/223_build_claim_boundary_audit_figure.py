from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "GenomeBiology_known_source_submission_package"
FIG_DIR = ACTIVE / "03_figures"
LATEX_FIG_DIR = ROOT / "latex_submission_package" / "TED_GenomeBiology_LaTeX_submission" / "figures"
GLOBAL_TABLES = ROOT / "results" / "ted_known_source_validation" / "tables"


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def clean(value: object) -> str:
    labels = {
        "outcome_supported_event": "outcome-supported event",
        "reversal_supported_event": "reversal-supported event",
        "cross_dataset_supported_event": "cross-dataset-supported event",
    }
    value = str(value)
    return labels.get(value, value.replace("_", " "))


def card(ax, x, y, w, h, title, subtitle, face, edge="#4b6578", title_color="#111827", wrap=36):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + 0.04 * w, y + h - 0.025, title, fontsize=11, fontweight="bold", color=title_color, va="top")
    ax.text(x + 0.04 * w, y + h - 0.08, fill(subtitle, wrap), fontsize=9.0, color="#374151", va="top")


def main() -> None:
    claim = read_tsv(GLOBAL_TABLES / "ted_dataset_level_claim_boundary.tsv")
    if claim.empty:
        claim = pd.DataFrame(
            [
                {"dataset": "GSE153056", "claim_boundary": "outcome_supported_event", "status": "pass"},
                {"dataset": "GSE93735", "claim_boundary": "reversal_supported_event", "status": "pass"},
                {"dataset": "SCP1064", "claim_boundary": "outcome_supported_event", "status": "pass"},
            ]
        )

    source_data = pd.DataFrame(
        [
            {
                "case": "GSE153056",
                "gate_class": "upgraded",
                "evidence": "known source + PD-L1 protein outcome",
                "boundary": "outcome_supported_event",
                "unsupported_escalation": "universal PD-L1 causal regulator proof",
            },
            {
                "case": "GSE93735",
                "gate_class": "upgraded",
                "evidence": "LPS event + dexamethasone reversal above controls",
                "boundary": "reversal_supported_event",
                "unsupported_escalation": "validation of GSE271399 biology",
            },
            {
                "case": "SCP1064",
                "gate_class": "upgraded",
                "evidence": "CRISPR guide + RNA event + protein readout",
                "boundary": "outcome_supported_event",
                "unsupported_escalation": "level4_causal_rescue",
            },
            {
                "case": "GSE271399",
                "gate_class": "blocked",
                "evidence": "robust event + cross-dataset support",
                "boundary": "cross_dataset_supported_event",
                "unsupported_escalation": "same-system full-length GATA1 rescue",
            },
            {
                "case": "SCP1064",
                "gate_class": "blocked",
                "evidence": "source-to-protein support without matched rescue",
                "boundary": "outcome_supported_event",
                "unsupported_escalation": "matched rescue or exhaustive perturbation-causal validation",
            },
            {
                "case": "GSE90546/GSE90063/GSE133344",
                "gate_class": "pending",
                "evidence": "incomplete adapters or raw matrix",
                "boundary": "not_evaluable / pending",
                "unsupported_escalation": "methodology support claim",
            },
        ]
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    source_data.to_csv(FIG_DIR / "figure5_source_data.tsv", sep="\t", index=False)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig = plt.figure(figsize=(13.5, 10.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1], width_ratios=[0.9, 1.1], hspace=0.34, wspace=0.25)

    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("A. Boundary assignment is gate-based", loc="left", fontweight="bold")
    steps = [
        ("source", "known perturbation/intervention"),
        ("event", "pre-specified event family"),
        ("readout", "outcome or reversal metric"),
        ("controls", "negative controls and shuffles"),
        ("boundary", "highest supported boundary"),
    ]
    y = 0.82
    for i, (title, body) in enumerate(steps):
        card(ax, 0.08, y, 0.78, 0.12, title, body, "#e8f1fb", wrap=34)
        if i < len(steps) - 1:
            ax.annotate("", xy=(0.47, y - 0.015), xytext=(0.47, y - 0.05), arrowprops=dict(arrowstyle="-|>", color="#64748b", lw=1.2))
        y -= 0.17
    ax.text(0.09, 0.03, "Registry entries are frozen before expression-level interpretation.", fontsize=8.5, color="#64748b")

    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("B. Evidence-supported upgrades", loc="left", fontweight="bold")
    upgraded = source_data[source_data["gate_class"].eq("upgraded")]
    colors = {"GSE153056": "#e8f1fb", "GSE93735": "#e8f5e5", "SCP1064": "#fff0df"}
    y = 0.66
    for _, row in upgraded.iterrows():
        card(
            ax,
            0.04,
            y,
            0.88,
            0.21,
            row["case"],
            f"{clean(row['boundary'])}\n{row['evidence']}",
            colors.get(row["case"], "#f3f4f6"),
            wrap=62,
        )
        y -= 0.25

    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("C. Blocked or pending escalations", loc="left", fontweight="bold")
    blocked = source_data[~source_data["gate_class"].eq("upgraded")]
    y = 0.68
    for _, row in blocked.iterrows():
        card(
            ax,
            0.06,
            y,
            0.86,
            0.22,
            row["case"],
            f"boundary: {clean(row['boundary'])}\nblocked: {row['unsupported_escalation']}",
            "#f5ebe7" if row["gate_class"] == "blocked" else "#f1f5f9",
            edge="#b7796d" if row["gate_class"] == "blocked" else "#64748b",
            wrap=52,
        )
        y -= 0.29

    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("D. Boundary summary", loc="left", fontweight="bold")
    rows = [
        ("Case", "TED boundary", "Unsupported escalation"),
        ("GSE153056", "outcome-supported event", "universal PD-L1 causal proof"),
        ("GSE93735", "reversal-supported event", "GSE271399 rescue validation"),
        ("SCP1064", "outcome-supported event", "Level 4 causal rescue"),
        ("GSE271399", "cross-dataset-supported candidate", "same-system GATA1 rescue"),
        ("Pending datasets", "not evaluable / pending", "methodology support claim"),
    ]
    col_x = [0.02, 0.30, 0.62]
    col_w = [0.25, 0.29, 0.34]
    row_h = 0.135
    y0 = 0.82
    for r, row in enumerate(rows):
        y = y0 - r * row_h
        face = "#dbeafe" if r == 0 else ("#f8fafc" if r % 2 == 1 else "white")
        for c, text in enumerate(row):
            rect = FancyBboxPatch(
                (col_x[c], y),
                col_w[c],
                row_h * 0.86,
                boxstyle="round,pad=0.005,rounding_size=0.006",
                linewidth=0.7,
                edgecolor="#cbd5e1",
                facecolor=face,
            )
            ax.add_patch(rect)
            ax.text(
                col_x[c] + 0.012,
                y + row_h * 0.58,
                fill(text, 24 if c == 2 else 20),
                fontsize=8.6 if r else 9.2,
                fontweight="bold" if r == 0 else "normal",
                va="center",
                color="#111827",
            )
    ax.text(0.02, 0.02, "Upgrades require outcome/reversal gates; missing rescue gates remain explicit.", fontsize=8.5, color="#64748b")

    fig.suptitle("TED upgrades supported event claims while blocking unsupported causal escalation", x=0.02, ha="left", fontsize=16, fontweight="bold")
    fig.savefig(FIG_DIR / "figure5_claim_upgrade_block_audit.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure5_claim_upgrade_block_audit.pdf", bbox_inches="tight")
    LATEX_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(LATEX_FIG_DIR / "figure5_claim_upgrade_block_audit.png", dpi=220, bbox_inches="tight")
    fig.savefig(LATEX_FIG_DIR / "figure5_claim_upgrade_block_audit.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
