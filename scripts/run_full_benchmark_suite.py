from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "data_external"
REPRO_DIR = DEFAULT_ROOT / "ted_development_reproducibility"
LOG_DIR = REPRO_DIR / "benchmark_logs"


@dataclass(frozen=True)
class BenchmarkStep:
    step: str
    command: list[str]
    expected_outputs: list[Path]
    quick_command: list[str] | None = None


def script(name: str) -> str:
    return str(ROOT / "scripts" / name)


def steps(data_root: Path, quick: bool) -> list[BenchmarkStep]:
    phase4 = data_root / "ted_development_phase4_benchmark"
    adversarial_reps = "4" if quick else "18"
    baseline_reps = "6" if quick else "24"
    return [
        BenchmarkStep(
            "direct_external_baseline_suite",
            [
                sys.executable,
                script("run_direct_external_baseline_suite.py"),
                "--outdir",
                str(phase4 / "direct_external_baseline"),
                *(["--quick"] if quick else []),
            ],
            [
                phase4 / "direct_external_baseline" / "direct_external_baseline_registry.tsv",
                phase4 / "direct_external_baseline" / "direct_external_baseline_metric_table.tsv",
                phase4 / "direct_external_baseline" / "direct_external_baseline_docker_report.md",
            ],
        ),
        BenchmarkStep(
            "phase4_5_adversarial_benchmark",
            [
                sys.executable,
                script("run_ted_development_phase4_5_adversarial_benchmark.py"),
                "--root",
                str(data_root),
                "--replicates",
                adversarial_reps,
            ],
            [
                phase4 / "adversarial_benchmark" / "phase4_5_performance_ci.tsv",
                phase4 / "adversarial_benchmark" / "phase4_5_failure_modes.tsv",
            ],
        ),
        BenchmarkStep(
            "phase4_6_serious_baseline_suite",
            [
                sys.executable,
                script("run_ted_development_phase4_6_baseline_suite.py"),
                "--root",
                str(data_root),
                "--replicates",
                baseline_reps,
            ],
            [
                phase4 / "serious_baseline_suite" / "phase4_6_baseline_metric_table.tsv",
                phase4 / "serious_baseline_suite" / "phase4_6_method_capability_coverage.tsv",
            ],
        ),
        BenchmarkStep(
            "phase4_2_4_3_baseline_ablation",
            [
                sys.executable,
                script("run_ted_development_phase4_baseline_ablation.py"),
                "--root",
                str(data_root),
            ],
            [
                phase4 / "baseline_comparison" / "phase4_baseline_comparison.tsv",
                phase4 / "ablation" / "phase4_ablation_summary.tsv",
            ],
        ),
        BenchmarkStep(
            "phase4_8_algorithm_sensitivity",
            [
                sys.executable,
                script("run_ted_development_phase4_8_algorithm_sensitivity.py"),
            ],
            [
                phase4 / "algorithm_sensitivity" / "variant_selection_decision.tsv",
                phase4 / "algorithm_sensitivity" / "ted_vnext_recommendation.md",
            ],
        ),
    ]


def run_step(step: BenchmarkStep, log_dir: Path) -> dict[str, object]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{step.step}.log"
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            step.command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.perf_counter() - start
    existing = [str(path.relative_to(ROOT)) for path in step.expected_outputs if path.exists()]
    missing = [str(path.relative_to(ROOT)) for path in step.expected_outputs if not path.exists()]
    status = "pass" if proc.returncode == 0 and not missing else "fail"
    return {
        "step": step.step,
        "status": status,
        "exit_code": proc.returncode,
        "runtime_seconds": round(elapsed, 3),
        "command": " ".join(step.command),
        "log": str(log_path.relative_to(ROOT)),
        "expected_outputs_present": ";".join(existing),
        "expected_outputs_missing": ";".join(missing),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_report(manifest: pd.DataFrame, output: Path, quick: bool) -> None:
    passed = int((manifest["status"] == "pass").sum())
    total = int(len(manifest))
    report = [
        "# TED Full Benchmark Suite Report",
        "",
        f"Mode: {'quick smoke test' if quick else 'full benchmark'}",
        f"Steps passed: {passed}/{total}",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Step Summary",
        "",
        manifest[["step", "status", "exit_code", "runtime_seconds", "log"]].to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "This entry point is intended for reviewer-visible benchmark execution. The quick mode uses fewer synthetic replicates and is appropriate for smoke testing. The full mode uses the release default replicate counts and regenerates adversarial benchmark, serious baseline, ablation and algorithm-sensitivity outputs.",
        "",
    ]
    output.write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TED full benchmark suite entry point.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--quick", action="store_true", help="Use reduced replicate counts for reviewer smoke testing.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed step and record all statuses.")
    args = parser.parse_args()

    data_root = args.root if args.root.is_absolute() else ROOT / args.root
    REPRO_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for step in steps(data_root, args.quick):
        row = run_step(step, LOG_DIR)
        rows.append(row)
        print(f"{row['step']}: {row['status']} ({row['runtime_seconds']}s)")
        if row["status"] != "pass" and not args.keep_going:
            break

    manifest = pd.DataFrame(rows)
    suffix = "quick" if args.quick else "full"
    manifest_path = REPRO_DIR / f"{suffix}_benchmark_run_manifest.tsv"
    report_path = REPRO_DIR / f"{suffix}_benchmark_run_report.md"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    write_report(manifest, report_path, quick=args.quick)
    if not manifest.empty and (manifest["status"] != "pass").any():
        raise SystemExit(1)
    print(f"wrote benchmark manifest to {manifest_path}")
    print(f"wrote benchmark report to {report_path}")


if __name__ == "__main__":
    main()
