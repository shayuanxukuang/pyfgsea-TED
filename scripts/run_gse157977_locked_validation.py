from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data_external" / "ted_generalization_panel" / "GSE157977"
OUTDIR = ROOT / "data_external" / "ted_locked_validation" / "GSE157977"


EVENT_FAMILIES = [
    {
        "event_family": "neural_axis_shift_candidate",
        "predefined_axes": "neural_marker_axis;neural_progenitor_axis;excitatory_neuron_axis;inhibitory_neuron_axis",
        "positive_rule": "max_abs_neural_delta>=0.15 and negative_control_margin>=0.05",
        "allowed_claim": "guide-level neural-axis adapter candidate",
        "forbidden_claim": "cell-type-specific fate loss, developmental delay/loss/redirection, target-gene mechanism",
    },
    {
        "event_family": "glial_axis_shift_candidate",
        "predefined_axes": "astro_glia_axis;oligodendrocyte_axis;microglia_axis",
        "positive_rule": "max_abs_neural_delta>=0.15 and negative_control_margin>=0.05",
        "allowed_claim": "guide-level glial-axis adapter candidate",
        "forbidden_claim": "cell-type-specific glial fate mechanism or causal target-gene effect",
    },
    {
        "event_family": "control_sensitive_or_mixed",
        "predefined_axes": "any neural/glial axis",
        "positive_rule": "negative_control_margin<0.05 or control axis dominates",
        "allowed_claim": "control-sensitive or mixed signal",
        "forbidden_claim": "neurodevelopmental mechanism candidate",
    },
    {
        "event_family": "low_support_guide",
        "predefined_axes": "any neural/glial axis",
        "positive_rule": "n_samples<8 or total_cells<100",
        "allowed_claim": "low-support guide, not interpretable as event",
        "forbidden_claim": "validated perturbation event",
    },
]


NEGATIVE_CONTROLS = [
    {
        "control_axis": "housekeeping_control",
        "purpose": "detect generic library-size or stable-expression dominance",
        "failure_rule": "max_abs_control_delta >= max_abs_neural_delta or negative_control_margin < 0.05",
    },
    {
        "control_axis": "stress_axis",
        "purpose": "detect stress-dominated perturbation signatures",
        "failure_rule": "stress_axis is the max control and negative_control_margin < 0.05",
    },
    {
        "control_axis": "proliferation_axis",
        "purpose": "detect cell-cycle/proliferation confounding",
        "failure_rule": "proliferation_axis is the max control and negative_control_margin < 0.05",
    },
]


CLAIM_GATES = [
    {
        "gate": "G1_data_independence",
        "pass_rule": "dataset not used for TED parameter selection or GSE271399 mechanism construction",
        "effect_on_failure": "cannot be called independent locked validation",
    },
    {
        "gate": "G2_locked_axes",
        "pass_rule": "event families and control axes defined before this locked validation summary is interpreted",
        "effect_on_failure": "downgrade to exploratory adapter analysis",
    },
    {
        "gate": "G3_guide_support",
        "pass_rule": "n_samples>=8 and total_cells>=100",
        "effect_on_failure": "downgrade_low_support",
    },
    {
        "gate": "G4_event_strength",
        "pass_rule": "max_abs_neural_delta>=0.15",
        "effect_on_failure": "downgrade_low_signal",
    },
    {
        "gate": "G5_negative_control_margin",
        "pass_rule": "negative_control_margin>=0.05",
        "effect_on_failure": "downgrade_ambiguous_control_margin or reject_control_sensitive",
    },
    {
        "gate": "G6_missing_target_map_ceiling",
        "pass_rule": "target_gene is known and guide-target map restored",
        "effect_on_failure": "cap at Level_3_adapter_candidate; forbid target-gene mechanism",
    },
    {
        "gate": "G7_missing_cell_state_ceiling",
        "pass_rule": "cell-state annotation sufficient for delay/loss/redirection",
        "effect_on_failure": "forbid cell-type-specific fate loss and delay/loss/redirection",
    },
]


def ensure_outdir() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)


def classify(row: pd.Series) -> tuple[str, str, str]:
    if float(row["n_samples"]) < 8 or float(row["total_cells"]) < 100:
        return (
            "downgrade_low_support",
            "Level_2.5_low_support_adapter_signal",
            "Guide is retained in audit but not interpreted as an event because support is below the locked gate.",
        )
    if float(row["negative_control_margin"]) < 0:
        return (
            "reject_control_sensitive",
            "rejected_or_Level_1_control_sensitive",
            "Control axis exceeds neural/glial axis; TED rejects mechanism interpretation.",
        )
    if float(row["negative_control_margin"]) < 0.05:
        return (
            "downgrade_ambiguous_control_margin",
            "Level_2.5_control_sensitive_candidate",
            "Neural/glial signal is close to control signal; TED downgrades rather than promotes.",
        )
    if float(row["max_abs_neural_delta"]) < 0.15:
        return (
            "downgrade_low_signal",
            "Level_2.5_low_signal_adapter_candidate",
            "Guide passes control margin but event strength is below locked signal threshold.",
        )
    if row["event_family"] in {"neural_axis_shift_candidate", "glial_axis_shift_candidate"}:
        return (
            "identify_adapter_candidate",
            "Level_3_in_vivo_guide_pseudobulk_adapter_candidate",
            "Guide passes locked support, signal and negative-control gates, but target/cell-state gaps cap the claim.",
        )
    return (
        "downgrade_mixed_or_uninterpretable",
        "Level_2.5_mixed_adapter_candidate",
        "Signal passes some gates but locked event family remains mixed or uninterpretable.",
    )


def write_preregistration() -> None:
    text = f"""# GSE157977 Locked Validation Preregistration

Generated UTC: {datetime.now(timezone.utc).isoformat()}

## Rationale

GSE157977 is used as an independent locked validation of TED's event-and-claim behavior. It is not used to tune TED parameters, choose the GSE271399 mechanism, select algorithm variants, or define the benchmark scoring rules. The validation question is not whether TED proves a neurodevelopmental mechanism. The locked question is whether TED can identify plausible guide-level adapter candidates while downgrading or rejecting guides whose neural/glial signal is weak, low-support, or dominated by negative controls.

## Locked Event Families

- neural_axis_shift_candidate
- glial_axis_shift_candidate
- control_sensitive_or_mixed
- low_support_guide

## Locked Negative Controls

- housekeeping_control
- stress_axis
- proliferation_axis

## Locked Claim Gates

- guide support: n_samples >= 8 and total_cells >= 100
- event strength: max_abs_neural_delta >= 0.15
- negative-control margin: negative_control_margin >= 0.05
- missing guide-target map caps claims at adapter level
- missing cell-state annotation forbids delay/loss/redirection or cell-type-specific fate claims

## Allowed Interpretation

GSE157977 can support independent guide-level in vivo Perturb-seq adapter validation of TED's claim discipline. It can show that TED identifies a small number of neural/glial-axis adapter candidates while downgrading or rejecting control-sensitive guides.

## Forbidden Interpretation

Do not claim validated ASD/NDD gene mechanisms, cell-type-specific fate loss, developmental delay, fate redirection, or functional causality from this locked validation.
"""
    (OUTDIR / "gse157977_locked_validation_preregistration.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_outdir()
    family = pd.read_csv(INPUT / "gse157977_in_vivo_perturbation_event_family.tsv", sep="\t")

    classified = family.copy()
    values = classified.apply(classify, axis=1, result_type="expand")
    classified["locked_validation_status"] = values[0]
    classified["locked_claim_ceiling"] = values[1]
    classified["locked_interpretation"] = values[2]
    classified["locked_allowed_claim"] = classified["locked_claim_ceiling"].map(
        {
            "Level_3_in_vivo_guide_pseudobulk_adapter_candidate": "guide-level neural/glial adapter candidate",
            "Level_2.5_low_support_adapter_signal": "low-support adapter signal only",
            "Level_2.5_control_sensitive_candidate": "control-sensitive candidate only",
            "Level_2.5_low_signal_adapter_candidate": "low-signal adapter candidate only",
            "Level_2.5_mixed_adapter_candidate": "mixed adapter candidate only",
            "rejected_or_Level_1_control_sensitive": "rejected or descriptive control-sensitive signal",
        }
    )
    classified["locked_forbidden_claim"] = (
        "target-gene-specific perturbation mechanism; cell-type-specific fate loss; developmental delay/loss/redirection; functional causality"
    )

    summary = (
        classified.groupby("locked_validation_status", dropna=False)
        .agg(
            n_guides=("guide_barcode", "count"),
            median_negative_control_margin=("negative_control_margin", "median"),
            median_neural_delta=("max_abs_neural_delta", "median"),
            median_control_delta=("max_abs_control_delta", "median"),
            median_total_cells=("total_cells", "median"),
        )
        .reset_index()
    )
    summary["fraction_guides"] = summary["n_guides"] / len(classified)

    claim = pd.DataFrame(
        [
            {
                "dataset": "GSE157977",
                "validation_type": "independent_locked_adapter_validation",
                "n_guides": len(classified),
                "n_identified_adapter_candidates": int((classified["locked_validation_status"] == "identify_adapter_candidate").sum()),
                "n_downgraded": int(classified["locked_validation_status"].str.startswith("downgrade").sum()),
                "n_rejected": int((classified["locked_validation_status"] == "reject_control_sensitive").sum()),
                "max_claim_ceiling": "Level_3_in_vivo_guide_pseudobulk_adapter_candidate",
                "allowed_claim": "TED identifies a small set of guide-level neural/glial adapter candidates and conservatively downgrades or rejects most control-sensitive guides in an independent in vivo Perturb-seq dataset.",
                "forbidden_claim": "Validated ASD/NDD gene mechanism, cell-type-specific fate loss, developmental delay/loss/redirection, or functional causality.",
                "missing_evidence": "guide-to-target map; high-resolution cell-state annotation; raw leave-one-cell-state/guide refit; functional validation",
            }
        ]
    )

    pd.DataFrame(EVENT_FAMILIES).to_csv(OUTDIR / "gse157977_locked_event_families.tsv", sep="\t", index=False)
    pd.DataFrame(NEGATIVE_CONTROLS).to_csv(OUTDIR / "gse157977_locked_negative_controls.tsv", sep="\t", index=False)
    pd.DataFrame(CLAIM_GATES).to_csv(OUTDIR / "gse157977_locked_claim_gates.tsv", sep="\t", index=False)
    classified.to_csv(OUTDIR / "gse157977_locked_validation_results.tsv", sep="\t", index=False)
    summary.to_csv(OUTDIR / "gse157977_locked_validation_summary.tsv", sep="\t", index=False)
    claim.to_csv(OUTDIR / "gse157977_locked_validation_claim_ceiling.tsv", sep="\t", index=False)
    write_preregistration()

    report = [
        "# GSE157977 Locked Validation Report",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Claim Ceiling",
        "",
        claim.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "This locked validation supports TED's computational claim discipline rather than a new neurodevelopmental mechanism. Most guides are downgraded or rejected because negative controls, low support or weak signal prevent stronger interpretation. A small subset passes locked adapter gates, but the missing guide-target map and missing cell-state annotation cap the result at Level 3 adapter evidence.",
        "",
    ]
    (OUTDIR / "gse157977_locked_validation_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"wrote GSE157977 locked validation outputs to {OUTDIR}")


if __name__ == "__main__":
    main()
