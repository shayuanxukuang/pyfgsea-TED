from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_ted_development_phase4_hardening import (  # noqa: E402
    DEFAULT_ROOT,
    METRICS,
    PHASE4_DIRNAME,
    infer_failure_regions,
    run_hard_synthetic_sweeps,
    safe_float,
    write_tsv,
)


OUTDIR_NAME = "adversarial_benchmark"


SWEEP_OUTPUTS = {
    "phase4_5_noise_sweep.tsv": "signal_strength",
    "phase4_5_dropout_sweep.tsv": "dropout_rate",
    "phase4_5_block_imbalance_sweep.tsv": "block_imbalance",
    "phase4_5_missing_timepoint_sweep.tsv": "timepoint_missingness",
    "phase4_5_batch_confounding_sweep.tsv": "batch_time_confounding",
    "phase4_5_rare_lineage_sweep.tsv": "rare_lineage_fraction",
}


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def add_adversarial_interpretation(curve: pd.DataFrame) -> pd.DataFrame:
    out = curve.copy()
    out["recognizable_region"] = (
        (pd.to_numeric(out["event_recovery_mean"], errors="coerce") >= 0.60)
        & (pd.to_numeric(out["event_type_accuracy_mean"], errors="coerce") >= 0.55)
    )
    out["low_overclaim"] = pd.to_numeric(out["overclaim_rate_mean"], errors="coerce") <= 0.10
    out["claim_ceiling_downgraded"] = pd.to_numeric(out["claim_ceiling_downgrade_rate_mean"], errors="coerce") >= 0.45
    out["TED_adversarial_behavior"] = np.where(
        out["method"].eq("TED-Development") & ~out["recognizable_region"] & out["claim_ceiling_downgraded"],
        "unidentifiable_region_downgraded",
        np.where(
            out["method"].eq("TED-Development") & out["recognizable_region"] & out["low_overclaim"],
            "recognizable_region_supported",
            "baseline_or_standard_behavior",
        ),
    )
    return out


def build_performance_ci(curve: pd.DataFrame) -> pd.DataFrame:
    cols = ["sweep_factor", "sweep_value", "method", "n_replicates"]
    for metric in METRICS:
        cols.extend([f"{metric}_mean", f"{metric}_ci95_low", f"{metric}_ci95_high"])
    cols.extend(["recognizable_region", "low_overclaim", "claim_ceiling_downgraded", "TED_adversarial_behavior"])
    return curve[[col for col in cols if col in curve.columns]].copy()


def build_failure_modes(curve: pd.DataFrame) -> pd.DataFrame:
    failure = infer_failure_regions(curve)
    ted = curve[curve["method"].eq("TED-Development")].copy()
    rows = []
    for _, row in ted.iterrows():
        recognizable = safe_float(row.get("event_recovery_mean")) >= 0.60 and safe_float(row.get("event_type_accuracy_mean")) >= 0.55
        downgrade = safe_float(row.get("claim_ceiling_downgrade_rate_mean")) >= 0.45
        rows.append(
            {
                "sweep_factor": row["sweep_factor"],
                "sweep_value": row["sweep_value"],
                "method": "TED-Development",
                "TED_recognizable_region": recognizable,
                "TED_claim_ceiling_downgraded": downgrade,
                "TED_overclaim_rate_mean": row.get("overclaim_rate_mean"),
                "adversarial_claim": (
                    "passes_with_low_overclaim"
                    if recognizable and safe_float(row.get("overclaim_rate_mean")) <= 0.10
                    else "fails_but_downgrades_claim_ceiling"
                    if downgrade
                    else "ambiguous_requires_manual_review"
                ),
            }
        )
    ted_summary = pd.DataFrame(rows)
    merged = failure.merge(
        ted_summary,
        on=["sweep_factor", "sweep_value", "method"],
        how="left",
    )
    return merged


def write_report(outdir: Path, performance: pd.DataFrame, failure: pd.DataFrame) -> None:
    ted_signal = performance[
        performance["method"].eq("TED-Development") & performance["sweep_factor"].eq("signal_strength")
    ][
        [
            "sweep_value",
            "event_recovery_mean",
            "event_type_accuracy_mean",
            "overclaim_rate_mean",
            "claim_ceiling_downgrade_rate_mean",
            "TED_adversarial_behavior",
        ]
    ]
    report = [
        "# Phase 4.5 Adversarial Benchmark",
        "",
        f"Generated: {date.today().isoformat()}",
        "",
        "Purpose: replace perfect synthetic separation with adversarial sweeps that test signal weakness, dropout, block imbalance, missing timepoints, batch-time confounding, and rare lineages.",
        "",
        "## TED Signal Sweep",
        "",
        ted_signal.to_markdown(index=False),
        "",
        "## Failure Interpretation",
        "",
        "TED is expected to fail in genuinely unidentifiable regimes. The target behavior is not all-1.000 performance; it is low overclaim and a downgraded claim ceiling when the event cannot be identified.",
        "",
        "## Required Outputs",
        "",
        "\n".join(f"- {name}" for name in list(SWEEP_OUTPUTS) + ["phase4_5_performance_ci.tsv", "phase4_5_failure_modes.tsv"]),
        "",
    ]
    (outdir / "phase4_5_adversarial_report.md").write_text("\n".join(report), encoding="utf-8")


def run(root: Path, n_replicates: int) -> Path:
    outdir = root / PHASE4_DIRNAME / OUTDIR_NAME
    ensure_outdir(outdir)

    sweeps = run_hard_synthetic_sweeps(n_replicates=n_replicates)
    curve = add_adversarial_interpretation(sweeps["curve"])

    for filename, factor in SWEEP_OUTPUTS.items():
        write_tsv(curve[curve["sweep_factor"].eq(factor)].copy(), outdir / filename)

    performance = build_performance_ci(curve)
    failure = build_failure_modes(curve)
    write_tsv(performance, outdir / "phase4_5_performance_ci.tsv")
    write_tsv(failure, outdir / "phase4_5_failure_modes.tsv")
    write_report(outdir, performance, failure)
    return outdir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TED-Development Phase 4.5 adversarial synthetic benchmark.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--replicates", type=int, default=18)
    args = parser.parse_args()
    outdir = run(args.root, args.replicates)
    print(f"wrote Phase 4.5 adversarial benchmark outputs to {outdir}")


if __name__ == "__main__":
    main()
