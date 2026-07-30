"""Build the frozen Figure 5 assets, including labelled legacy fields.

The v1.1 companion's canonical claim boundary is emitted separately by
``build_bib_companion_evidence_contracts.py`` and verified against its schemas.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RNA_DIR = ROOT / "results" / "ted_bnt162b2_flagship" / "rna_event_freeze_v1"
ADT_DIR = ROOT / "results" / "ted_bnt162b2_flagship" / "orthogonal_outcome_v1"
REP_DIR = ROOT / "results" / "ted_gse171964_replication" / "analysis_v1"
OUT_DIR = ROOT / "results" / "ted_bnt162b2_flagship" / "figure5"

FIGURE_TARGETS = [
    ROOT / "results" / "ted_v1_submission" / "figures",
    ROOT / "results" / "bib_manuscript_revision" / "figures",
    ROOT / "GenomeBiology_known_source_submission_package" / "01_main_manuscript" / "figures",
    ROOT
    / "GenomeBiology_known_source_submission_package"
    / "06_latex_source"
    / "TED_GenomeBiology_Main_Manuscript_Only"
    / "figures",
    ROOT / "latex_submission_package" / "TED_GenomeBiology_LaTeX_submission" / "figures",
    ROOT / "GenomeBiology_known_source_submission_package" / "03_figures",
    ROOT / "GenomeBiology_known_source_submission_package" / "03_figures_final_upload",
]

SOURCE_TARGETS = [
    ROOT / "results" / "ted_v1_submission" / "figure_source_data",
    ROOT
    / "GenomeBiology_known_source_submission_package"
    / "05_source_data_and_audits"
    / "figure_source_data",
    ROOT
    / "GenomeBiology_known_source_submission_package"
    / "06_latex_source"
    / "TED_GenomeBiology_Main_Manuscript_Only"
    / "tables"
    / "figure_source_data",
    ROOT
    / "latex_submission_package"
    / "TED_GenomeBiology_LaTeX_submission"
    / "tables"
    / "figure_source_data",
]

PRIMARY = "HALLMARK_INTERFERON_ALPHA_RESPONSE"
PRIMARY_DAYS = [0, 2, 10, 28]
REP_DAYS = [21, 22, 28, 42]
BLUE = "#1769AA"
ORANGE = "#D97706"
GREEN = "#138A72"
RED = "#C23B22"
GREY = "#6B7280"


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
    )


def donor_trajectory(
    ax: plt.Axes,
    table: pd.DataFrame,
    *,
    days: list[int],
    value: str,
    ylabel: str,
    color: str,
) -> None:
    for _, donor in table.groupby("donor_id", sort=True):
        donor = donor.set_index("day").reindex(days)
        ax.plot(days, donor[value], color=color, alpha=0.28, lw=1.1, marker="o", ms=3)
    summary = table.groupby("day")[value].agg(["mean", "sem"]).reindex(days)
    ax.errorbar(
        days,
        summary["mean"],
        yerr=summary["sem"],
        color=color,
        lw=2.7,
        marker="o",
        ms=6,
        capsize=3,
        label="donor mean +/- SEM",
    )
    ax.axhline(0, color="#9CA3AF", lw=0.8, zorder=0)
    ax.set_xticks(days)
    ax.set_xlabel("Day")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8, loc="best")


def save_source(name: str, table: pd.DataFrame) -> None:
    path = OUT_DIR / name
    table.to_csv(path, sep="\t", index=False)
    for target in SOURCE_TARGETS:
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target / name)


def main() -> None:
    rna_status = json.loads(require(RNA_DIR / "rna_event_status.json").read_text(encoding="utf-8"))
    adt_status = json.loads(require(ADT_DIR / "protein_outcome_status.json").read_text(encoding="utf-8"))
    rep_status = json.loads(require(REP_DIR / "replication_status.json").read_text(encoding="utf-8"))
    rna_scores = pd.read_csv(require(RNA_DIR / "pathway_donor_time_scores.tsv"), sep="\t")
    protein_scores = pd.read_csv(require(ADT_DIR / "protein_donor_time_scores.tsv"), sep="\t")
    rna_contrasts = pd.read_csv(require(RNA_DIR / "pathway_donor_contrasts.tsv"), sep="\t")
    protein_contrasts = pd.read_csv(require(ADT_DIR / "protein_donor_contrasts.tsv"), sep="\t")
    rep_qc = pd.read_csv(require(REP_DIR / "sample_blind_qc.tsv"), sep="\t", dtype={"pt_id": str})

    primary_scores = rna_scores[["donor_id", "day", "selected_cells", PRIMARY]].copy()
    primary_scores = primary_scores.rename(columns={PRIMARY: "IFN_alpha_score"})
    rna_primary = rna_contrasts.loc[
        rna_contrasts["pathway"].eq(PRIMARY), ["donor_id", "activation", "recovery", "transient"]
    ].rename(
        columns={
            "activation": "RNA_activation",
            "recovery": "RNA_recovery",
            "transient": "RNA_transient",
        }
    )
    protein_primary = protein_contrasts[["donor_id", "activation", "recovery", "transient"]].rename(
        columns={
            "activation": "protein_activation",
            "recovery": "protein_recovery",
            "transient": "protein_transient",
        }
    )
    aligned = rna_primary.merge(protein_primary, on="donor_id", validate="one_to_one")
    rep_qc = rep_qc.loc[rep_qc["day"].isin(REP_DAYS)].copy()
    rep_qc["max_abs_mad_z"] = rep_qc[
        ["median_rna_umi_abs_mad_z", "median_detected_genes_abs_mad_z"]
    ].max(axis=1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_source("figure5_primary_rna_trajectory.tsv", primary_scores)
    save_source("figure5_primary_protein_trajectory.tsv", protein_scores)
    save_source("figure5_rna_protein_donor_contrasts.tsv", aligned)
    save_source("figure5_gse171964_blind_qc.tsv", rep_qc)
    design_rows = []
    for donor in sorted(primary_scores["donor_id"].unique()):
        for day in PRIMARY_DAYS:
            for modality, availability, masking in [
                ("RNA", "available", "used for event freeze"),
                ("protein_ADT", "available", "masked until RNA status serialized"),
                ("ATAC", "absent", "not tested in the analysed object"),
            ]:
                design_rows.append(
                    {
                        "donor_id": donor,
                        "day": day,
                        "modality": modality,
                        "availability": availability,
                        "masking_or_use": masking,
                    }
                )
    save_source("figure5_flagship_design.tsv", pd.DataFrame(design_rows))
    rna_gate_audit = pd.DataFrame(
        [
            ["family maxT p", rna_status["primary_family_adjusted_p"], "<=0.10", True],
            ["donor direction", rna_status["direction_fraction"], ">=0.80", True],
            ["LODO retention", rna_status["lodo_retention_fraction"], ">=0.80", False],
            ["matched-state attenuation", rna_status["state_match_attenuation"], "<=0.50 and all fits converge", False],
            ["negative-control margin", rna_status["negative_control_margin"], ">0", False],
        ],
        columns=["gate", "observed", "frozen_requirement", "passed"],
    )
    save_source("figure5_rna_gate_audit.tsv", rna_gate_audit)
    status_table = pd.DataFrame(
        [
            {
                "primary_event_support": rna_status["event_support"]["code"],
                "primary_validation_provenance": adt_status["validation_provenance_code"],
                "primary_evidence_boundary": adt_status["evidence_boundary"],
                "primary_protein_outcome_status": adt_status["protein_outcome_status"],
                "event_replication_eligibility_status": rep_status[
                    "event_replication_eligibility_status"
                ],
                "event_replication_test_status": rep_status["event_replication_test_status"],
                "event_replication_status": rep_status["event_replication_status"],
                "protein_outcome_replication_status": rep_status["protein_outcome"]["replication_status"],
                "event_replication_attempt_status": rep_status.get("event_replication_attempt_status"),
                "event_replication_reason": rep_status.get("event_replication_reason"),
            }
        ]
    )
    save_source("figure5_evidence_status.tsv", status_table)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(14.2, 9.1), constrained_layout=True)
    gs = fig.add_gridspec(2, 6, height_ratios=[0.92, 1.08])
    ax_a = fig.add_subplot(gs[0, :2])
    ax_b = fig.add_subplot(gs[0, 2:])
    ax_c = fig.add_subplot(gs[1, :2])
    ax_d = fig.add_subplot(gs[1, 2:4])
    ax_e = fig.add_subplot(gs[1, 4:])
    fig.suptitle(
        "Masked-outcome boundary audit with a fail-closed independent-cohort replication attempt",
        fontsize=14.2,
        fontweight="bold",
        x=0.03,
        ha="left",
    )

    donors = sorted(primary_scores["donor_id"].unique())
    for yi, donor in enumerate(donors):
        ax_a.plot(range(len(PRIMARY_DAYS)), [yi] * len(PRIMARY_DAYS), color="#D1D5DB", lw=0.8, zorder=0)
        ax_a.scatter(range(len(PRIMARY_DAYS)), [yi] * len(PRIMARY_DAYS), s=38, color=BLUE, edgecolor="white", linewidth=0.5)
    ax_a.set_xticks(range(len(PRIMARY_DAYS)), [f"day {day}" for day in PRIMARY_DAYS])
    ax_a.set_yticks(range(len(donors)), donors)
    ax_a.set_ylim(-0.8, len(donors) + 2.4)
    ax_a.invert_yaxis()
    ax_a.set_title("Locked design and outcome masking")
    ax_a.text(0.02, 0.26, "RNA: event inference first", color=BLUE, fontweight="bold", transform=ax_a.transAxes, fontsize=8.5)
    ax_a.text(0.02, 0.17, "Protein ADT: masked until RNA freeze", color=ORANGE, fontweight="bold", transform=ax_a.transAxes, fontsize=8.5)
    ax_a.text(0.02, 0.08, "ATAC: absent from analysed object; not tested", color=GREY, fontweight="bold", transform=ax_a.transAxes, fontsize=8.2)
    ax_a.text(0.98, 0.02, "6 donors x 4 timepoints", ha="right", transform=ax_a.transAxes, fontsize=8.0, color=GREY)
    panel_label(ax_a, "A")

    donor_trajectory(
        ax_b,
        primary_scores,
        days=PRIMARY_DAYS,
        value="IFN_alpha_score",
        ylabel="IFN-alpha pathway score\n(mean gene z)",
        color=BLUE,
    )
    ax_b.set_title("Donor-level IFN RNA pathway time course")
    ax_b.text(
        0.02,
        0.98,
        f"maxT p={rna_status['primary_family_adjusted_p']:.4g}; "
        f"direction={rna_status['direction_fraction']:.2f}",
        transform=ax_b.transAxes,
        va="top",
        fontsize=8.5,
    )
    panel_label(ax_b, "B")

    rna_ordered = rna_primary.sort_values("RNA_transient")
    bar_colors = np.where(rna_ordered["RNA_transient"] > 0, BLUE, RED)
    ax_c.barh(rna_ordered["donor_id"], rna_ordered["RNA_transient"], color=bar_colors, edgecolor="white")
    ax_c.axvline(0, color="#9CA3AF", lw=0.8)
    ax_c.set_xlabel("RNA transient contrast")
    ax_c.set_title("Donor effects and frozen event gates")
    gate_lines = [
        f"PASS  maxT p={rna_status['primary_family_adjusted_p']:.4g}",
        f"PASS  direction={rna_status['direction_fraction']:.2f}",
        f"FAIL  LODO={rna_status['lodo_retention_fraction']:.2f}",
        f"FAIL  matched attenuation={rna_status['state_match_attenuation']:.2f}",
        f"FAIL  control margin={rna_status['negative_control_margin']:.3f}",
        "FINAL  event support E0",
    ]
    ax_c.text(0.98, 0.04, "\n".join(gate_lines), transform=ax_c.transAxes, ha="right", va="bottom", fontsize=7.6,
              bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#D1D5DB"})
    panel_label(ax_c, "C")

    ax_d.axhline(0, color="#9CA3AF", lw=0.8)
    ax_d.axvline(0, color="#9CA3AF", lw=0.8)
    ax_d.scatter(aligned["RNA_transient"], aligned["protein_transient"], s=58, color=GREEN)
    for row in aligned.itertuples(index=False):
        ax_d.annotate(str(row.donor_id), (row.RNA_transient, row.protein_transient), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax_d.set_xlabel("Donor RNA transient contrast")
    ax_d.set_ylabel("Donor CD64/CD169 transient contrast")
    ax_d.set_title("RNA event and same-cell protein outcome")
    ax_d.text(
        0.02,
        0.98,
        f"protein exact p={adt_status['protein_exact_p']:.4g}; direction={adt_status['protein_direction_stability']:.2f}\n"
        f"agreement={adt_status['rna_protein_donor_direction_agreement']:.2f}; protein outcome {adt_status['protein_outcome_status']}\n"
        "event remains E0",
        transform=ax_d.transAxes,
        va="top",
        fontsize=8.0,
    )
    panel_label(ax_d, "D")

    donor_order = sorted(rep_qc["pt_id"].unique())
    x_lookup = {donor: idx for idx, donor in enumerate(donor_order)}
    offsets = dict(zip(REP_DAYS, np.linspace(-0.24, 0.24, len(REP_DAYS))))
    markers = {21: "o", 22: "s", 28: "^", 42: "D"}
    for day in REP_DAYS:
        part = rep_qc.loc[rep_qc["day"].eq(day)]
        xs = [x_lookup[d] + offsets[day] for d in part["pt_id"]]
        colors = np.where(part["blind_qc_pass"], GREY, RED)
        ax_e.scatter(xs, part["max_abs_mad_z"], c=colors, marker=markers[day], s=42, label=f"day {day}")
    ax_e.axhline(3.0, color=RED, lw=1.3, ls="--", label="frozen QC threshold")
    ax_e.set_xticks(range(len(donor_order)), donor_order, rotation=25)
    ax_e.set_xlabel("GSE171964 participant")
    ax_e.set_ylabel("Maximum absolute MAD z")
    ax_e.set_title("Independent-cohort event replication")
    ax_e.legend(frameon=False, fontsize=6.7, ncol=2, loc="upper left")
    ax_e.text(
        0.98,
        0.98,
        f"eligibility: {rep_status['event_replication_eligibility_status']}\n"
        f"event test: {rep_status['event_replication_test_status']}\n"
        f"event replication: {rep_status['event_replication_status']}\n"
        f"{rep_status['n_evaluable_donors']}/6 donors after frozen QC\n"
        f"protein replication: {rep_status['protein_outcome']['replication_status']}\n"
        "CD64/CD169 ADT absent",
        transform=ax_e.transAxes,
        ha="right",
        va="top",
        fontsize=7.4,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#D1D5DB"},
    )
    panel_label(ax_e, "E")

    pdf_path = OUT_DIR / "figure5_independent_real_data_validation.pdf"
    png_path = OUT_DIR / "figure5_independent_real_data_validation.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    for target in FIGURE_TARGETS:
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf_path, target / pdf_path.name)
        shutil.copy2(png_path, target / png_path.name)
    print(f"Wrote Figure 5 and source data to {OUT_DIR}")


if __name__ == "__main__":
    main()
