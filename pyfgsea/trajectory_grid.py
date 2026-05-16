from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

from .trajectory import run_trajectory_gsea
from .trajectory_events import summarize_events


def _as_count(value: Union[int, float], n_obs: int) -> int:
    if isinstance(value, float) and 0 < value < 1:
        return max(1, int(round(value * n_obs)))
    return int(value)


def _classify_consensus(
    peak_time_sd: float,
    sign_consistency: float,
    significant_overlap: float,
    median_duration: float,
    ranker_support: float = np.nan,
    event_label_consistency: float = np.nan,
) -> str:
    if np.isfinite(ranker_support) and ranker_support < 0.5:
        return "unstable"
    if np.isfinite(event_label_consistency) and event_label_consistency < 0.5:
        return "unstable"
    if significant_overlap < 0.25 or sign_consistency < 0.6:
        return "unstable"
    if peak_time_sd <= 0.05 and sign_consistency >= 0.9 and significant_overlap >= 0.6:
        if median_duration <= 0.12:
            return "robust transient"
        return "robust"
    if significant_overlap >= 0.5 and sign_consistency >= 0.75:
        return "moderate"
    return "unstable"


def summarize_event_consensus(
    events: pd.DataFrame,
    pathway_col: str = "Pathway",
    run_col: str = "grid_run",
    fdr_threshold: float = 0.05,
    total_runs: Optional[int] = None,
) -> pd.DataFrame:
    """Summarize event stability across parameter-grid runs."""
    if events is None or events.empty:
        return pd.DataFrame()
    if pathway_col not in events.columns:
        raise ValueError(f"Missing pathway column '{pathway_col}'")
    if run_col not in events.columns:
        raise ValueError(f"Missing run column '{run_col}'")

    rows = []
    total_runs = max(events[run_col].nunique(), 1) if total_runs is None else max(int(total_runs), 1)
    total_rankers = (
        max(events["grid_ranker"].nunique(), 1) if "grid_ranker" in events.columns else np.nan
    )
    total_seeds = max(events["grid_seed"].nunique(), 1) if "grid_seed" in events.columns else np.nan
    for pathway, group in events.groupby(pathway_col, sort=False):
        peak_times = group["peak_time"].to_numpy(dtype=np.float64)
        peak_nes = group["peak_NES"].to_numpy(dtype=np.float64)
        fdr = group["window_fdr_min"].to_numpy(dtype=np.float64)

        finite_peak = np.isfinite(peak_times)
        significant = np.isfinite(fdr) & (fdr <= fdr_threshold)
        significant_overlap = float(significant.sum() / total_runs)

        signs = np.sign(peak_nes[np.isfinite(peak_nes)])
        if len(signs):
            pos = np.mean(signs > 0)
            neg = np.mean(signs < 0)
            sign_consistency = float(max(pos, neg))
        else:
            sign_consistency = np.nan

        if finite_peak.any():
            consensus_peak = float(np.nanmedian(peak_times[finite_peak]))
            peak_time_sd = float(np.nanstd(peak_times[finite_peak], ddof=0))
        else:
            consensus_peak = np.nan
            peak_time_sd = np.nan

        median_duration = float(np.nanmedian(group["duration"])) if "duration" in group else np.nan
        rankers_observed = (
            sorted(map(str, group["grid_ranker"].dropna().unique()))
            if "grid_ranker" in group.columns
            else []
        )
        significant_rankers = (
            sorted(map(str, group.loc[significant, "grid_ranker"].dropna().unique()))
            if "grid_ranker" in group.columns
            else []
        )
        ranker_support = (
            float(len(significant_rankers) / total_rankers)
            if np.isfinite(total_rankers)
            else np.nan
        )
        seeds_observed = (
            sorted(map(str, group["grid_seed"].dropna().unique()))
            if "grid_seed" in group.columns
            else []
        )
        seed_support = (
            float(group.loc[significant, "grid_seed"].nunique() / total_seeds)
            if "grid_seed" in group.columns and np.isfinite(total_seeds)
            else np.nan
        )
        if "event_label" in group.columns and group["event_label"].notna().any():
            label_counts = group["event_label"].fillna("missing").value_counts()
            dominant_label = str(label_counts.index[0])
            event_label_consistency = float(label_counts.iloc[0] / max(len(group), 1))
        else:
            dominant_label = ""
            event_label_consistency = np.nan

        rows.append(
            {
                pathway_col: pathway,
                "n_runs": total_runs,
                "observed_runs": int(group[run_col].nunique()),
                "observed_run_fraction": float(group[run_col].nunique() / total_runs),
                "consensus_peak_time": consensus_peak,
                "peak_time_sd": peak_time_sd,
                "sign_consistency": sign_consistency,
                "significant_overlap": significant_overlap,
                "ranker_support": ranker_support,
                "ranker_agreement": ranker_support,
                "seed_support": seed_support,
                "dominant_event_label": dominant_label,
                "consensus_label": dominant_label,
                "event_label_consistency": event_label_consistency,
                "rankers_observed": ";".join(rankers_observed),
                "significant_rankers": ";".join(significant_rankers),
                "seeds_observed": ";".join(seeds_observed),
                "median_duration": median_duration,
                "median_peak_NES": float(np.nanmedian(peak_nes)),
                "min_window_fdr": float(np.nanmin(fdr)) if len(fdr) else np.nan,
                "recommendation": _classify_consensus(
                    peak_time_sd,
                    sign_consistency,
                    significant_overlap,
                    median_duration,
                    ranker_support,
                    event_label_consistency,
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["recommendation", "significant_overlap", "sign_consistency"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def run_trajectory_gsea_grid(
    adata,
    gmt_path: str,
    window_sizes: Iterable[Union[int, float]],
    step_sizes: Iterable[Union[int, float]],
    rankers: Optional[Iterable[str]] = None,
    seeds: Optional[Iterable[int]] = None,
    return_consensus: bool = True,
    event_kwargs: Optional[dict] = None,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """
    Run trajectory GSEA across window/step parameters and summarize stability.

    Fractions in ``window_sizes`` or ``step_sizes`` are interpreted as fractions
    of cells, e.g. ``0.05`` means 5% of ``adata.n_obs``.
    """
    n_obs = adata.n_obs if hasattr(adata, "n_obs") else len(adata.obs)
    event_kwargs = {} if event_kwargs is None else dict(event_kwargs)

    base_kwargs = dict(kwargs)
    default_ranker = base_kwargs.pop("ranker", "mean_diff")
    default_seed = base_kwargs.pop("seed", 42)
    ranker_values = list(rankers) if rankers is not None else [default_ranker]
    seed_values = list(seeds) if seeds is not None else [default_seed]
    if not ranker_values:
        raise ValueError("rankers must contain at least one ranker")
    if not seed_values:
        raise ValueError("seeds must contain at least one seed")

    result_frames = []
    event_frames = []
    grid_run = 0
    gene_set_index = base_kwargs.pop("gene_set_index", None)
    window_index_cache = {}
    for window_size in window_sizes:
        for step_size in step_sizes:
            window_count = _as_count(window_size, n_obs)
            step_count = _as_count(step_size, n_obs)
            window_key = (
                window_count,
                step_count,
                base_kwargs.get("window_mode", "cell_count"),
                base_kwargs.get("min_cells"),
                base_kwargs.get("max_cells"),
                base_kwargs.get("target_span"),
                base_kwargs.get("span_step"),
                base_kwargs.get("graph_key", "connectivities"),
                base_kwargs.get("graph_radius", 2),
                base_kwargs.get("branch_key"),
                base_kwargs.get("cell_weight_key"),
            )
            for ranker in ranker_values:
                for seed in seed_values:
                    grid_run += 1
                    res = run_trajectory_gsea(
                        adata,
                        gmt_path=gmt_path,
                        window_size=window_count,
                        step=step_count,
                        ranker=ranker,
                        seed=int(seed),
                        gene_set_index=gene_set_index,
                        window_index=window_index_cache.get(window_key),
                        **base_kwargs,
                    )
                    if res is None or res.empty:
                        continue
                    gene_set_index = gene_set_index or res.attrs.get("gene_set_index")
                    if window_key not in window_index_cache:
                        window_index_cache[window_key] = res.attrs.get("window_index")

                    res = res.copy()
                    res["grid_run"] = grid_run
                    res["grid_window_size"] = window_count
                    res["grid_step_size"] = step_count
                    res["grid_window_size_input"] = window_size
                    res["grid_step_size_input"] = step_size
                    res["grid_ranker"] = ranker
                    res["grid_seed"] = int(seed)
                    result_frames.append(res)

                    events = summarize_events(res, **event_kwargs)
                    if not events.empty:
                        events["grid_run"] = grid_run
                        events["grid_window_size"] = window_count
                        events["grid_step_size"] = step_count
                        events["grid_window_size_input"] = window_size
                        events["grid_step_size_input"] = step_size
                        events["grid_ranker"] = ranker
                        events["grid_seed"] = int(seed)
                        event_frames.append(events)

    results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    consensus = (
        summarize_event_consensus(events, total_runs=grid_run)
        if return_consensus
        else pd.DataFrame()
    )

    return {"results": results, "events": events, "consensus": consensus}


def run_ranker_consensus(
    adata,
    gmt_path: str,
    pseudotime_key: str = "dpt_pseudotime",
    rankers: Optional[Iterable[str]] = None,
    window_mode: str = "cell_count",
    window_sizes: Optional[Iterable[Union[int, float]]] = None,
    step_sizes: Optional[Iterable[Union[int, float]]] = None,
    window_size: Union[int, float] = 500,
    step: Union[int, float] = 100,
    seeds: Optional[Iterable[int]] = None,
    return_details: bool = False,
    event_kwargs: Optional[dict] = None,
    **kwargs,
):
    """
    Run several trajectory rankers and summarize pathway-event agreement.

    This is a convenience wrapper around ``run_trajectory_gsea_grid`` where the
    grid axis is primarily the ranker. It returns a consensus table by default.
    """
    rankers = list(
        rankers
        or ["mean_diff", "detection_weighted", "local_slope", "neighbor_contrast"]
    )
    window_sizes = list(window_sizes) if window_sizes is not None else [window_size]
    step_sizes = list(step_sizes) if step_sizes is not None else [step]

    out = run_trajectory_gsea_grid(
        adata,
        gmt_path=gmt_path,
        pseudotime_key=pseudotime_key,
        window_sizes=window_sizes,
        step_sizes=step_sizes,
        rankers=rankers,
        seeds=seeds,
        return_consensus=True,
        event_kwargs=event_kwargs,
        window_mode=window_mode,
        **kwargs,
    )
    consensus = out["consensus"].copy()
    consensus.attrs["results"] = out["results"]
    consensus.attrs["events"] = out["events"]
    consensus.attrs["ranker_consensus"] = {
        "rankers": rankers,
        "window_mode": window_mode,
        "window_sizes": window_sizes,
        "step_sizes": step_sizes,
    }
    if return_details:
        out["consensus"] = consensus
        return out
    return consensus
