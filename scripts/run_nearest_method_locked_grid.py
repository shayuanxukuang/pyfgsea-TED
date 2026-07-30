from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Task:
    task_id: str
    scenario_index: int
    replicate: int
    seed: int
    blocks: int
    signal: str
    coordinate: str
    artifact: str
    scenario_name: str


def build_tasks() -> list[Task]:
    tasks: list[Task] = []
    scenario_index = 0
    for blocks in (4, 6, 10):
        for signal in ("low", "high"):
            for coordinate in ("true", "noisy"):
                for artifact in ("none", "composition", "stress", "partial_batch_time"):
                    scenario_index += 1
                    for replicate, seed in enumerate(range(20262001, 20262011), start=1):
                        scenario_name = f"{blocks}b_{signal}_{coordinate}_{artifact}_seed{seed}"
                        tasks.append(
                            Task(
                                task_id=f"S{scenario_index:02d}_R{replicate:02d}",
                                scenario_index=scenario_index,
                                replicate=replicate,
                                seed=seed,
                                blocks=blocks,
                                signal=signal,
                                coordinate=coordinate,
                                artifact=artifact,
                                scenario_name=scenario_name,
                            )
                        )
    return tasks


def registry_hash(tasks: list[Task]) -> str:
    payload = json.dumps([asdict(task) for task in tasks], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def complete(output_root: Path, task: Task) -> bool:
    out = output_root / "locked" / task.scenario_name
    required = [
        out / "scenario.json",
        out / "five_method_predictions.tsv",
        out / "five_method_metrics.tsv",
        out / "five_method_runtime.tsv",
        out / "truth.tsv",
    ]
    if not all(path.exists() and path.stat().st_size > 0 for path in required):
        return False
    try:
        predictions = pd.read_csv(required[1], sep="\t")
        metrics = pd.read_csv(required[2], sep="\t")
        scenario = json.loads(required[0].read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        len(predictions) == 150
        and predictions["method"].nunique() == 5
        and predictions.groupby("method").size().eq(30).all()
        and len(metrics) == 5
        and metrics["method"].nunique() == 5
        and sorted(scenario.get("methods_completed", []))
        == sorted(["TIPS", "TED", "scTransient", "score_then_smooth", "tradeSeq"])
    )


def run_one(
    task: Task,
    *,
    output_root: Path,
    log_dir: Path,
    timeout: int,
    adapter_permutations: int,
) -> dict[str, object]:
    stdout_path = log_dir / f"{task.task_id}.stdout.log"
    stderr_path = log_dir / f"{task.task_id}.stderr.log"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_nearest_method_five_method_benchmark.py"),
        "--profile", "locked",
        "--output-dir", str(output_root),
        "--blocks", str(task.blocks),
        "--signal", task.signal,
        "--coordinate", task.coordinate,
        "--artifact", task.artifact,
        "--seed", str(task.seed),
        "--adapter-permutations", str(adapter_permutations),
        "--method-timeout", str(timeout),
    ]
    started_at = time.time()
    status = "failed"
    exit_code: int | None = None
    error = ""
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout + 600,
                check=False,
            )
        exit_code = result.returncode
        status = "complete" if result.returncode == 0 and complete(output_root, task) else "failed"
        if status != "complete":
            error = "nonzero_exit_or_output_validation_failed"
    except subprocess.TimeoutExpired:
        status = "timeout"
        error = f"task_timeout_after_{timeout + 600}_seconds"
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    return {
        **asdict(task),
        "status": status,
        "exit_code": exit_code,
        "started_at_epoch": started_at,
        "finished_at_epoch": time.time(),
        "elapsed_seconds": time.time() - started_at,
        "attempt": 1,
        "error": error,
        "stdout_log": str(stdout_path.relative_to(ROOT)),
        "stderr_log": str(stderr_path.relative_to(ROOT)),
    }


def write_status(rows: list[dict[str, object]], path: Path) -> None:
    frame = pd.DataFrame(rows).sort_values(["scenario_index", "replicate"])
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=7_200)
    parser.add_argument("--adapter-permutations", type=int, default=10_000)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results" / "ted_nearest_method_five_method",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("workers must be positive")

    output_root = args.output_root.resolve()
    control_dir = output_root / "locked_grid_control"
    log_dir = control_dir / "logs"
    control_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks()
    registry = pd.DataFrame([asdict(task) for task in tasks])
    registry_path = control_dir / "locked_task_registry.tsv"
    expected_hash = registry_hash(tasks)
    if registry_path.exists():
        existing = pd.read_csv(registry_path, sep="\t")
        existing_tasks = [Task(**row) for row in existing.to_dict(orient="records")]
        if registry_hash(existing_tasks) != expected_hash:
            raise SystemExit("existing locked registry does not match the frozen task grid")
    else:
        registry.to_csv(registry_path, sep="\t", index=False)
    (control_dir / "locked_registry.sha256").write_text(
        f"{expected_hash}  locked_task_registry.tsv\n", encoding="utf-8"
    )
    (control_dir / "run_config.json").write_text(
        json.dumps(
            {
                "task_count": len(tasks),
                "scenario_count": 48,
                "seeds_per_scenario": 10,
                "seeds": list(range(20262001, 20262011)),
                "workers": args.workers,
                "method_timeout_seconds": args.timeout,
                "adapter_permutations": args.adapter_permutations,
                "registry_sha256": expected_hash,
                "automatic_retry": False,
                "python": sys.executable,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    prior_rows: list[dict[str, object]] = []
    pending: list[Task] = []
    for task in tasks:
        if complete(output_root, task):
            prior_rows.append({
                **asdict(task),
                "status": "complete",
                "exit_code": 0,
                "started_at_epoch": pd.NA,
                "finished_at_epoch": pd.NA,
                "elapsed_seconds": pd.NA,
                "attempt": 0,
                "error": "",
                "stdout_log": "",
                "stderr_log": "",
            })
        else:
            pending.append(task)
    status_path = control_dir / "task_status.tsv"
    write_status(prior_rows + [{**asdict(t), "status": "pending"} for t in pending], status_path)
    print(f"locked_registry={expected_hash} total={len(tasks)} complete={len(prior_rows)} pending={len(pending)} workers={args.workers}", flush=True)

    completed_rows = list(prior_rows)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_one,
                task,
                output_root=output_root,
                log_dir=log_dir,
                timeout=args.timeout,
                adapter_permutations=args.adapter_permutations,
            ): task
            for task in pending
        }
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            completed_rows.append(row)
            remaining = [task for task in pending if task.task_id not in {str(r["task_id"]) for r in completed_rows}]
            write_status(
                completed_rows + [{**asdict(t), "status": "pending"} for t in remaining],
                status_path,
            )
            counts = pd.Series([str(r["status"]) for r in completed_rows]).value_counts().to_dict()
            print(
                f"progress={len(completed_rows)}/{len(tasks)} task={row['task_id']} status={row['status']} elapsed={row['elapsed_seconds']:.1f}s counts={counts}",
                flush=True,
            )

    final = pd.DataFrame(completed_rows)
    if len(final) != len(tasks) or not final["status"].eq("complete").all():
        failures = final[~final["status"].eq("complete")]
        print(f"locked grid incomplete: {len(failures)} failed/timeout tasks", flush=True)
        raise SystemExit(2)
    print("locked grid complete: 480/480 validated", flush=True)


if __name__ == "__main__":
    main()
