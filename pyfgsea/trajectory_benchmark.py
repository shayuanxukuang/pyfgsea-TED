from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .trajectory import run_trajectory_gsea
from .trajectory_compare import compare_trajectory_gsea, run_branch_gsea
from .trajectory_events import summarize_events


TRUTH_TYPES = [
    "monotonic_up",
    "monotonic_down",
    "narrow_transient_pulse",
    "broad_transient_plateau",
    "biphasic_early_late",
    "sparse_dropout_burst",
    "branch_specific_activation",
    "condition_delayed_activation",
]


def _dynamic_profile(pseudotime: np.ndarray, truth_type: str) -> np.ndarray:
    pt = np.asarray(pseudotime, dtype=float)
    if truth_type == "monotonic_up":
        return 2.0 * pt
    if truth_type == "monotonic_down":
        return 2.0 * (1.0 - pt)
    if truth_type == "narrow_transient_pulse":
        return 3.0 * np.exp(-((pt - 0.5) ** 2) / (2 * 0.035**2))
    if truth_type == "broad_transient_plateau":
        return 2.0 * ((pt >= 0.35) & (pt <= 0.65)).astype(float)
    if truth_type == "biphasic_early_late":
        early = np.exp(-((pt - 0.22) ** 2) / (2 * 0.05**2))
        late = np.exp(-((pt - 0.78) ** 2) / (2 * 0.05**2))
        return 2.2 * (early + late)
    if truth_type == "sparse_dropout_burst":
        return 3.5 * ((pt >= 0.45) & (pt <= 0.50)).astype(float)
    raise ValueError(f"Unsupported profile truth_type '{truth_type}'")


def _expected_peak(truth_type: str) -> float:
    return {
        "monotonic_up": 1.0,
        "monotonic_down": 0.0,
        "narrow_transient_pulse": 0.5,
        "broad_transient_plateau": 0.5,
        "biphasic_early_late": 0.22,
        "sparse_dropout_burst": 0.475,
        "branch_specific_activation": 0.75,
        "condition_delayed_activation": 0.7,
    }[truth_type]


def make_synthetic_trajectory_truth(
    truth_type: str,
    n_cells: int = 400,
    n_genes: int = 120,
    pathway_size: int = 12,
    effect_size: float = 1.0,
    noise_sd: float = 0.25,
    seed: int = 0,
):
    """
    Build a compact AnnData object with an embedded pathway dynamic.

    Returns ``(adata, gene_sets, truth)``. ``gene_sets`` is a dict accepted by
    ``run_trajectory_gsea``.
    """
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("make_synthetic_trajectory_truth requires anndata") from exc

    if truth_type not in TRUTH_TYPES:
        raise ValueError(f"Unsupported truth_type '{truth_type}'")
    if pathway_size >= n_genes:
        raise ValueError("pathway_size must be smaller than n_genes")

    rng = np.random.default_rng(seed)
    pseudotime = np.linspace(0.0, 1.0, n_cells)
    X = rng.normal(loc=0.0, scale=noise_sd, size=(n_cells, n_genes))
    X = X - X.min() + 0.1

    signal_genes = [f"Gene_{idx}" for idx in range(pathway_size)]
    noise_genes = [f"Gene_{idx}" for idx in range(pathway_size, 2 * pathway_size)]
    genes = [f"Gene_{idx}" for idx in range(n_genes)]

    obs = pd.DataFrame({"dpt_pseudotime": pseudotime})
    if truth_type == "branch_specific_activation":
        branch = np.where(np.arange(n_cells) % 2 == 0, "branch_a", "branch_b")
        obs["branch"] = pd.Categorical(branch)
        profile = 2.0 * np.clip((pseudotime - 0.5) / 0.5, 0, 1)
        X[branch == "branch_b", :pathway_size] += effect_size * profile[branch == "branch_b", None]
    elif truth_type == "condition_delayed_activation":
        condition = np.where(np.arange(n_cells) % 2 == 0, "control", "case")
        obs["condition"] = pd.Categorical(condition)
        control_profile = 2.0 * np.clip((pseudotime - 0.35) / 0.4, 0, 1)
        case_profile = 2.0 * np.clip((pseudotime - 0.55) / 0.4, 0, 1)
        X[condition == "control", :pathway_size] += effect_size * control_profile[condition == "control", None]
        X[condition == "case", :pathway_size] += effect_size * case_profile[condition == "case", None]
    else:
        profile = _dynamic_profile(pseudotime, truth_type)
        X[:, :pathway_size] += effect_size * profile[:, None]
        if truth_type == "sparse_dropout_burst":
            burst_cells = np.where(profile > 0)[0]
            keep = rng.choice(burst_cells, size=max(1, len(burst_cells) // 5), replace=False)
            X[burst_cells, :pathway_size] = 0.0
            X[keep, :pathway_size] = effect_size * 8.0

    adata = ad.AnnData(
        X=X,
        obs=obs,
        var=pd.DataFrame(index=genes),
    )
    gene_sets = {
        "TRUE_SIGNAL": signal_genes,
        "RANDOM_BACKGROUND": noise_genes,
    }
    truth = pd.DataFrame(
        [
            {
                "truth_type": truth_type,
                "pathway": "TRUE_SIGNAL",
                "expected_peak_time": _expected_peak(truth_type),
                "expected_label": truth_type,
            }
        ]
    )
    return adata, gene_sets, truth


def score_synthetic_events(
    events: pd.DataFrame,
    truth: pd.DataFrame,
    pathway_col: str = "Pathway",
) -> dict[str, float | str]:
    if events is None or events.empty:
        return {
            "detected": 0.0,
            "peak_time_error": np.nan,
            "event_label_accuracy": 0.0,
        }

    truth_row = truth.iloc[0]
    signal = events[events[pathway_col] == truth_row["pathway"]]
    if signal.empty:
        return {
            "detected": 0.0,
            "peak_time_error": np.nan,
            "event_label_accuracy": 0.0,
        }

    row = signal.iloc[0]
    peak_error = abs(float(row["peak_time"]) - float(truth_row["expected_peak_time"]))
    label = str(row.get("event_label", ""))
    expected = str(truth_row["expected_label"])
    if "monotonic" in expected:
        label_ok = "sustained" in label or "activation" in label or "suppression" in label
    elif "transient" in expected:
        label_ok = "transient" in label
    elif "biphasic" in expected:
        label_ok = "biphasic" in label or "recurrent" in label
    else:
        label_ok = "no clear event" not in label

    background = events[events[pathway_col] == "RANDOM_BACKGROUND"]
    background_sig = 0.0
    if not background.empty and "window_fdr_min" in background:
        background_sig = float((background["window_fdr_min"] <= 0.05).mean())

    return {
        "detected": 1.0,
        "peak_time_error": float(peak_error),
        "event_label_accuracy": float(label_ok),
        "random_background_significant": background_sig,
    }


def run_synthetic_truth_benchmark(
    truth_types: Optional[Iterable[str]] = None,
    rankers: Optional[Iterable[str]] = None,
    n_cells: int = 400,
    n_genes: int = 120,
    window_size: int = 80,
    step: int = 40,
    seed: int = 0,
    **kwargs,
) -> pd.DataFrame:
    """
    Run a compact ranker benchmark over semi-synthetic pathway truth patterns.
    """
    truth_types = list(truth_types or TRUTH_TYPES[:6])
    rankers = list(rankers or ["mean_diff", "detection_weighted", "local_slope", "neighbor_contrast"])
    rows = []
    for truth_idx, truth_type in enumerate(truth_types):
        adata, gene_sets, truth = make_synthetic_trajectory_truth(
            truth_type,
            n_cells=n_cells,
            n_genes=n_genes,
            seed=seed + truth_idx,
        )
        for ranker in rankers:
            run_kwargs = dict(kwargs)
            nperm_nes = run_kwargs.pop("nperm_nes", 50)
            sample_size = run_kwargs.pop("sample_size", 51)
            if truth_type == "branch_specific_activation":
                branch = run_branch_gsea(
                    adata,
                    gene_sets,
                    branch_key="branch",
                    pseudotime_key="dpt_pseudotime",
                    window_size=window_size,
                    step=step,
                    min_size=5,
                    max_size=500,
                    nperm_nes=nperm_nes,
                    sample_size=sample_size,
                    seed=seed,
                    ranker=ranker,
                    event_kwargs={"min_consecutive": 1},
                    **run_kwargs,
                )
                cmp_df = branch["comparisons"]
                signal = cmp_df[cmp_df["Pathway"] == "TRUE_SIGNAL"] if not cmp_df.empty else pd.DataFrame()
                detected = float(
                    not signal.empty
                    and signal.iloc[0]["program_type"]
                    in {"divergence_program", "specific_program"}
                )
                rows.append(
                    {
                        "truth_type": truth_type,
                        "ranker": ranker,
                        "detected": detected,
                        "peak_time_error": np.nan,
                        "event_label_accuracy": detected,
                        "random_background_significant": float(
                            not cmp_df.empty
                            and (
                                cmp_df.loc[
                                    cmp_df["Pathway"] == "RANDOM_BACKGROUND",
                                    "program_type",
                                ]
                                .isin(["divergence_program", "specific_program"])
                                .any()
                            )
                        ),
                    }
                )
                continue

            if truth_type == "condition_delayed_activation":
                cmp_df = compare_trajectory_gsea(
                    adata,
                    gene_sets,
                    condition_key="condition",
                    control="control",
                    case="case",
                    pseudotime_key="dpt_pseudotime",
                    window_size=window_size,
                    step=step,
                    min_size=5,
                    max_size=500,
                    nperm_nes=nperm_nes,
                    sample_size=sample_size,
                    seed=seed,
                    ranker=ranker,
                    event_kwargs={"min_consecutive": 1},
                    **run_kwargs,
                )
                signal = cmp_df[cmp_df["Pathway"] == "TRUE_SIGNAL"] if not cmp_df.empty else pd.DataFrame()
                if signal.empty:
                    delta_peak = np.nan
                    detected = 0.0
                else:
                    delta_peak = float(signal.iloc[0]["delta_peak_time"])
                    detected = float(delta_peak > 0)
                rows.append(
                    {
                        "truth_type": truth_type,
                        "ranker": ranker,
                        "detected": detected,
                        "peak_time_error": abs(delta_peak - 0.2)
                        if np.isfinite(delta_peak)
                        else np.nan,
                        "event_label_accuracy": detected,
                        "random_background_significant": float(
                            not cmp_df.empty
                            and (
                                cmp_df.loc[
                                    cmp_df["Pathway"] == "RANDOM_BACKGROUND",
                                    "program_type",
                                ]
                                .isin(["divergence_program", "specific_program"])
                                .any()
                            )
                        ),
                    }
                )
                continue

            res = run_trajectory_gsea(
                adata,
                gene_sets,
                pseudotime_key="dpt_pseudotime",
                window_size=window_size,
                step=step,
                min_size=5,
                max_size=500,
                nperm_nes=nperm_nes,
                sample_size=sample_size,
                seed=seed,
                ranker=ranker,
                **run_kwargs,
            )
            events = summarize_events(res, min_consecutive=1)
            score = score_synthetic_events(events, truth)
            rows.append(
                {
                    "truth_type": truth_type,
                    "ranker": ranker,
                    **score,
                }
            )
    return pd.DataFrame(rows)
