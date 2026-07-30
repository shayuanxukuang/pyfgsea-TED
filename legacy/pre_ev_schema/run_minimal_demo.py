from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUTDIR = Path("data_external") / "ted_development_reproducibility" / "minimal_demo"


LEVELS = [
    ("Level_1", 1.0, ["has_signal"]),
    ("Level_2", 2.0, ["has_signal", "event_fdr_pass"]),
    ("Level_2.5", 2.5, ["has_signal", "event_fdr_pass", "schema_or_adapter_valid"]),
    ("Level_3", 3.0, ["has_signal", "event_fdr_pass", "block_robust", "negative_control_pass"]),
    ("Level_3.5", 3.5, ["has_signal", "event_fdr_pass", "block_robust", "negative_control_pass", "perturbation_or_lineage_support"]),
    ("Level_4", 4.0, ["has_signal", "event_fdr_pass", "block_robust", "negative_control_pass", "perturbation_or_lineage_support", "functional_validation"]),
]


def assign_claim_ceiling(row: pd.Series) -> tuple[str, float]:
    best_name = "Level_1"
    best_value = 1.0
    for name, value, gates in LEVELS:
        if all(bool(row.get(g, False)) for g in gates):
            best_name = name
            best_value = value
    return best_name, best_value


def make_demo_events() -> pd.DataFrame:
    np.random.seed(271399)
    x = np.linspace(0, 1, 60)
    reference = 0.2 + 0.7 / (1 + np.exp(-10 * (x - 0.45)))
    perturbed_loss = 0.2 + 0.35 / (1 + np.exp(-10 * (x - 0.62)))
    control_like = reference + np.random.normal(0, 0.01, size=len(x))

    loss_effect = float(np.trapz(perturbed_loss - reference, x))
    control_effect = float(np.trapz(control_like - reference, x))

    events = pd.DataFrame(
        [
            {
                "dataset": "minimal_demo",
                "contrast": "perturbed_vs_reference",
                "event_family": "demo_developmental_output_loss",
                "module_or_axis": "demo_output_axis",
                "effect_size": loss_effect,
                "effect_direction": "loss" if loss_effect < 0 else "gain",
                "event_fdr": 0.01,
                "block_robustness": 0.94,
                "negative_control_margin": 0.42,
                "event_mode": "developmental_delay_or_loss",
                "has_signal": True,
                "event_fdr_pass": True,
                "schema_or_adapter_valid": True,
                "block_robust": True,
                "negative_control_pass": True,
                "perturbation_or_lineage_support": True,
                "functional_validation": False,
            },
            {
                "dataset": "minimal_demo",
                "contrast": "placebo_vs_reference",
                "event_family": "demo_negative_control",
                "module_or_axis": "housekeeping_control",
                "effect_size": control_effect,
                "effect_direction": "near_zero",
                "event_fdr": 0.62,
                "block_robustness": 0.12,
                "negative_control_margin": -0.05,
                "event_mode": "not_supported",
                "has_signal": False,
                "event_fdr_pass": False,
                "schema_or_adapter_valid": True,
                "block_robust": False,
                "negative_control_pass": False,
                "perturbation_or_lineage_support": False,
                "functional_validation": False,
            },
        ]
    )
    ceilings = events.apply(assign_claim_ceiling, axis=1, result_type="expand")
    events["claim_ceiling"] = ceilings[0]
    events["claim_ceiling_numeric"] = ceilings[1]
    events["allowed_claim"] = np.where(
        events["claim_ceiling"].eq("Level_3.5"),
        "computational mechanism candidate; not functional validation",
        "descriptive or rejected control signal",
    )
    events["forbidden_claim"] = np.where(
        events["functional_validation"],
        "",
        "strict Level 4 causal or rescue claim",
    )
    return events


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    events = make_demo_events()
    event_path = OUTDIR / "demo_output_event_objects.tsv"
    claim_path = OUTDIR / "demo_claim_ceiling.tsv"
    report_path = OUTDIR / "demo_report.md"
    events.to_csv(event_path, sep="\t", index=False)
    events[
        [
            "event_family",
            "effect_size",
            "event_fdr",
            "block_robustness",
            "negative_control_margin",
            "event_mode",
            "claim_ceiling",
            "allowed_claim",
            "forbidden_claim",
        ]
    ].to_csv(claim_path, sep="\t", index=False)
    report = f"""# TED Minimal Demo Report

The minimal demo generated {len(events)} TED-style event objects from deterministic toy trajectories.

## Expected behavior

- `demo_developmental_output_loss` passes event-FDR, block robustness and negative-control gates and is capped at Level 3.5 because no functional validation gate is present.
- `demo_negative_control` does not pass signal or event-FDR gates and remains descriptive/rejected.

## Outputs

- `{event_path.as_posix()}`
- `{claim_path.as_posix()}`
"""
    report_path.write_text(report, encoding="utf-8")
    print(f"wrote minimal demo outputs to {OUTDIR}")


if __name__ == "__main__":
    main()
