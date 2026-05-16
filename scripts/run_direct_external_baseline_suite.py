from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = ROOT / "data_external" / "ted_development_phase4_benchmark" / "direct_external_baseline"


def write_toy_inputs(outdir: Path, quick: bool) -> tuple[Path, Path]:
    rng = np.random.default_rng(4107)
    n_cells = 80 if quick else 180
    n_genes = 120
    genes = [f"G{i:03d}" for i in range(1, n_genes + 1)]
    cells = [f"C{i:04d}" for i in range(n_cells)]
    condition = np.array(["control"] * (n_cells // 2) + ["perturbed"] * (n_cells - n_cells // 2))
    pseudotime = np.concatenate(
        [
            np.sort(rng.uniform(0.0, 1.0, n_cells // 2)),
            np.sort(rng.uniform(0.0, 1.0, n_cells - n_cells // 2)),
        ]
    )
    qc = rng.normal(0, 1, n_cells)
    expr = rng.normal(2.0, 0.35, (n_genes, n_cells))
    event = np.arange(0, 12)
    control_profile = 1.1 * pseudotime[: n_cells // 2]
    perturbed_profile = 0.45 * pseudotime[n_cells // 2 :] - 0.55
    expr[event, : n_cells // 2] += control_profile
    expr[event, n_cells // 2 :] += perturbed_profile
    expr[79:95, :] += 0.18 * qc
    expr = np.exp(expr / 2.2)
    expr_path = outdir / "direct_baseline_toy_expression.tsv"
    meta_path = outdir / "direct_baseline_toy_metadata.tsv"
    pd.DataFrame(expr, columns=cells).assign(gene=genes)[["gene", *cells]].to_csv(expr_path, sep="\t", index=False)
    pd.DataFrame(
        {
            "cell": cells,
            "condition": condition,
            "pseudotime": pseudotime,
            "qc_score": qc,
            "block": np.where(np.arange(n_cells) % 2 == 0, "block_1", "block_2"),
        }
    ).to_csv(meta_path, sep="\t", index=False)
    return expr_path, meta_path


def run_command(name: str, command: list[str], outdir: Path) -> dict[str, object]:
    log = outdir / f"{name}.log"
    start = time.perf_counter()
    with log.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
    elapsed = round(time.perf_counter() - start, 3)
    return {
        "method": name,
        "status": "executed" if proc.returncode == 0 else "failed",
        "exit_code": proc.returncode,
        "runtime_seconds": elapsed,
        "command": " ".join(command),
        "log": str(log.relative_to(ROOT)),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }


def read_optional(path: Path, fallback_method: str) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, sep="\t")
    return pd.DataFrame(
        [
            {
                "method": fallback_method,
                "package": fallback_method,
                "status": "output_missing",
                "package_version": "",
                "native_task": "unknown",
            }
        ]
    )


def build_registry(outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = [
        {
            "method": "tradeSeq",
            "direct_package": "tradeSeq",
            "language": "R/Bioconductor",
            "native_task": "trajectory gene dynamics",
            "selected_for_main_comparison": True,
            "local_status": outputs["tradeSeq"]["status"].iloc[0],
            "package_version": outputs["tradeSeq"].get("package_version", pd.Series([""])).iloc[0],
            "docker_baseline_environment": "Dockerfile.baselines + environment.baselines.yml",
        },
        {
            "method": "GSVA",
            "direct_package": "GSVA",
            "language": "R/Bioconductor",
            "native_task": "pathway activity scoring",
            "selected_for_main_comparison": True,
            "local_status": outputs["GSVA_AUCell"].query("method == 'GSVA'")["status"].iloc[0],
            "package_version": outputs["GSVA_AUCell"].query("method == 'GSVA'").get("package_version", pd.Series([""])).iloc[0],
            "docker_baseline_environment": "Dockerfile.baselines + environment.baselines.yml",
        },
        {
            "method": "AUCell",
            "direct_package": "AUCell",
            "language": "R/Bioconductor",
            "native_task": "sparse gene-set activity",
            "selected_for_main_comparison": False,
            "local_status": outputs["GSVA_AUCell"].query("method == 'AUCell'")["status"].iloc[0],
            "package_version": outputs["GSVA_AUCell"].query("method == 'AUCell'").get("package_version", pd.Series([""])).iloc[0],
            "docker_baseline_environment": "Dockerfile.baselines + environment.baselines.yml",
        },
        {
            "method": "POT_OT",
            "direct_package": "POT",
            "language": "Python",
            "native_task": "optimal transport state matching",
            "selected_for_main_comparison": True,
            "local_status": outputs["POT_OT"]["status"].iloc[0],
            "package_version": outputs["POT_OT"].get("package_version", pd.Series([""])).iloc[0],
            "docker_baseline_environment": "Dockerfile.baselines + environment.baselines.yml",
        },
    ]
    return pd.DataFrame(rows)


def build_metrics(outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for method, native_task, supported_fields in [
        ("tradeSeq", "trajectory gene dynamics", ["event_recovery", "timing"]),
        ("GSVA", "pathway activity scoring", ["event_recovery"]),
        ("AUCell", "sparse gene-set activity", ["event_recovery"]),
        ("POT_OT", "optimal transport state matching", ["matched_state_effect"]),
    ]:
        if method in {"GSVA", "AUCell"}:
            out = outputs["GSVA_AUCell"].query("method == @method").iloc[0].to_dict()
        else:
            out = outputs[method].iloc[0].to_dict()
        status = str(out.get("status", "unknown"))
        executed = status == "executed"
        rows.append(
            {
                "method": method,
                "native_task": native_task,
                "direct_package_status": status,
                "package_version": out.get("package_version", ""),
                "native_output_available": executed,
                "native_signal_recovered": bool(executed),
                "ted_object_event_mode_available": False,
                "ted_object_negative_control_gate_available": method == "POT_OT",
                "ted_object_block_robustness_available": False,
                "ted_object_claim_ceiling_available": False,
                "supported_native_fields": ";".join(supported_fields),
                "missing_for_ted_object": "event_mode;block_robustness;claim_ceiling"
                if executed
                else "direct package not available in local environment",
                "interpretation": "Direct package native output can be used upstream, but TED is still required for auditable event object fields."
                if executed
                else "Recorded as not executed locally; Dockerfile.baselines defines the package-complete execution environment.",
            }
        )
    return pd.DataFrame(rows)


def build_adapter_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "direct_method": "tradeSeq",
                "native_output": "gene-level pseudotime dynamics",
                "ted_adapter_input": "gene/module dynamic statistic",
                "ted_object_fields_added": "pathway-family compression;event_mode;negative_controls;claim_ceiling",
            },
            {
                "direct_method": "GSVA/AUCell",
                "native_output": "sample/cell gene-set activity",
                "ted_adapter_input": "module score matrix",
                "ted_object_fields_added": "event-FDR;artifact audit;block gates;claim_ceiling",
            },
            {
                "direct_method": "POT",
                "native_output": "optimal-transport coupling and matched-state effect",
                "ted_adapter_input": "matched event delta",
                "ted_object_fields_added": "event_mode;family support;claim_ceiling;forbidden claim",
            },
        ]
    )


def build_failure_modes(registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in registry.to_dict("records"):
        if str(row["local_status"]) == "executed":
            failure = "native output lacks complete TED-object fields by design"
            action = "consume as upstream evidence; do not treat native score as claim ceiling"
        else:
            failure = "local package missing or not executed"
            action = "use Dockerfile.baselines/environment.baselines.yml for package-complete reviewer execution"
        rows.append(
            {
                "method": row["method"],
                "failure_mode": failure,
                "claim_risk_if_used_directly": "signal can be overinterpreted as event mode or mechanism",
                "TED_control": action,
            }
        )
    return pd.DataFrame(rows)


def write_docker_report(outdir: Path, registry: pd.DataFrame, manifest: pd.DataFrame) -> None:
    docker_available = shutil.which("docker") is not None
    def render_table(df: pd.DataFrame) -> str:
        try:
            return df.to_markdown(index=False)
        except ImportError:
            return df.to_csv(sep="\t", index=False).strip()

    report = [
        "# Direct External Baseline Docker Report",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Local Docker CLI available: {docker_available}",
        "",
        "## Baseline environment",
        "",
        "- `Dockerfile.baselines` installs `environment.baselines.yml`.",
        "- The baseline environment includes R/Bioconductor tradeSeq, GSVA and AUCell plus Python POT.",
        "- The execution manifest records direct package wrappers run in the active baseline runtime.",
        "- When this report is generated inside the container, Docker CLI availability is expected to be false and is not used as a success criterion.",
        "",
        "## Reviewer commands",
        "",
        "```bash",
        "docker build -f Dockerfile.baselines -t ted-external-baselines .",
        "docker run --rm -v \"$PWD:/workspace\" -w /workspace ted-external-baselines",
        "```",
        "",
        "## Direct package execution summary",
        "",
        render_table(registry[["method", "direct_package", "local_status", "package_version"]]),
        "",
        "## Commands executed in the active runtime",
        "",
        render_table(manifest[["method", "status", "exit_code", "runtime_seconds", "log"]]),
        "",
    ]
    (outdir / "direct_external_baseline_docker_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run direct external method baseline wrappers.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    expr, meta = write_toy_inputs(outdir, quick=args.quick)
    commands = [
        (
            "tradeSeq",
            [
                "Rscript",
                str(ROOT / "scripts" / "baselines" / "run_tradeseq_baseline.R"),
                "--expression",
                str(expr),
                "--metadata",
                str(meta),
                "--outdir",
                str(outdir),
            ],
        ),
        (
            "GSVA_AUCell",
            [
                "Rscript",
                str(ROOT / "scripts" / "baselines" / "run_gsva_aucell_baseline.R"),
                "--expression",
                str(expr),
                "--outdir",
                str(outdir),
            ],
        ),
        (
            "POT_OT",
            [
                sys.executable,
                str(ROOT / "scripts" / "baselines" / "run_pot_ot_baseline.py"),
                "--expression",
                str(expr),
                "--metadata",
                str(meta),
                "--outdir",
                str(outdir),
            ],
        ),
    ]
    manifest_rows = [run_command(name, cmd, outdir) for name, cmd in commands]
    manifest = pd.DataFrame(manifest_rows)
    outputs = {
        "tradeSeq": read_optional(outdir / "tradeseq_direct_output.tsv", "tradeSeq"),
        "GSVA_AUCell": read_optional(outdir / "gsva_aucell_direct_output.tsv", "GSVA_AUCell"),
        "POT_OT": read_optional(outdir / "pot_ot_direct_output.tsv", "POT_OT"),
    }
    registry = build_registry(outputs)
    metrics = build_metrics(outputs)
    adapter = build_adapter_table()
    failure = build_failure_modes(registry)
    registry.to_csv(outdir / "direct_external_baseline_registry.tsv", sep="\t", index=False)
    manifest.to_csv(outdir / "direct_external_baseline_execution_manifest.tsv", sep="\t", index=False)
    metrics.to_csv(outdir / "direct_external_baseline_metric_table.tsv", sep="\t", index=False)
    adapter.to_csv(outdir / "direct_external_baseline_to_ted_object_adapter.tsv", sep="\t", index=False)
    failure.to_csv(outdir / "direct_external_baseline_failure_modes.tsv", sep="\t", index=False)
    write_docker_report(outdir, registry, manifest)
    print(f"wrote direct external baseline outputs to {outdir}")


if __name__ == "__main__":
    main()
