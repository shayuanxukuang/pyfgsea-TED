from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run direct POT optimal-transport baseline on TED toy data.")
    parser.add_argument("--expression", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "pot_ot_direct_output.tsv"

    try:
        import ot  # type: ignore
    except Exception as exc:
        pd.DataFrame(
            [
                {
                    "method": "POT_OT",
                    "package": "POT",
                    "status": "not_run_missing_python_package",
                    "package_version": "",
                    "native_task": "optimal_transport_state_matching",
                    "ot_cost": np.nan,
                    "matched_event_delta": np.nan,
                    "output": "missing",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            ]
        ).to_csv(out, sep="\t", index=False)
        return

    expr = pd.read_csv(args.expression, sep="\t").set_index("gene")
    meta = pd.read_csv(args.metadata, sep="\t")
    values = expr.T
    event_genes = [f"G{i:03d}" for i in range(1, 13)]
    score = values[event_genes].mean(axis=1).to_numpy()
    cov = meta[["pseudotime", "qc_score"]].to_numpy(float)
    control = meta["condition"].to_numpy() == "control"
    perturbed = meta["condition"].to_numpy() == "perturbed"
    x = cov[control]
    y = cov[perturbed]
    a = np.ones(x.shape[0]) / x.shape[0]
    b = np.ones(y.shape[0]) / y.shape[0]
    cost = ot.dist(x, y, metric="sqeuclidean")
    plan = ot.emd(a, b, cost)
    control_score = score[control]
    perturbed_score = score[perturbed]
    counterfactual_control_for_perturbed = plan.T @ control_score / np.maximum(plan.T.sum(axis=1), 1e-12)
    delta = float(np.mean(perturbed_score - counterfactual_control_for_perturbed))
    pd.DataFrame(
        [
            {
                "method": "POT_OT",
                "package": "POT",
                "status": "executed",
                "package_version": getattr(ot, "__version__", "unknown"),
                "native_task": "optimal_transport_state_matching",
                "ot_cost": float((plan * cost).sum()),
                "matched_event_delta": delta,
                "output": str(out),
                "error": "",
            }
        ]
    ).to_csv(out, sep="\t", index=False)


if __name__ == "__main__":
    main()
