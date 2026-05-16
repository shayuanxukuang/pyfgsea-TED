from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .trajectory import run_trajectory_gsea
from .trajectory_events import summarize_events


def _bootstrap_cell_indices(n_obs: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n_obs, size=n_obs)


def _bootstrap_sample_indices(
    obs: pd.DataFrame,
    sample_key: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if sample_key not in obs:
        raise ValueError(f"sample_key '{sample_key}' not found in adata.obs")
    samples = pd.Series(obs[sample_key]).dropna().astype(str).unique()
    if len(samples) == 0:
        raise ValueError(f"sample_key '{sample_key}' has no non-null samples")

    sampled = rng.choice(samples, size=len(samples), replace=True)
    indices = []
    sample_values = pd.Series(obs[sample_key]).astype(str).to_numpy()
    for sample in sampled:
        indices.extend(np.where(sample_values == str(sample))[0].tolist())
    if not indices:
        raise ValueError("sample bootstrap produced no cells")
    return np.asarray(indices, dtype=int)


def _subset_with_unique_obs_names(adata, indices: np.ndarray, boot_id: int):
    boot = adata[indices].copy()
    boot.obs_names = [
        f"{name}__boot{boot_id}_{idx}"
        for idx, name in enumerate(map(str, boot.obs_names))
    ]
    return boot


def _quantile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(np.quantile(values, q))


def _make_bands(
    boot_results: pd.DataFrame,
    lower_q: float,
    upper_q: float,
    pathway_col: str = "Pathway",
) -> pd.DataFrame:
    if boot_results is None or boot_results.empty:
        return pd.DataFrame()
    required = {pathway_col, "window_id", "NES"}
    missing = required - set(boot_results.columns)
    if missing:
        raise ValueError(f"boot_results is missing required columns: {sorted(missing)}")

    rows = []
    for keys, group in boot_results.groupby([pathway_col, "window_id"], sort=False):
        pathway, window_id = keys
        row = {
            pathway_col: pathway,
            "window_id": window_id,
            "n_boot": int(group["boot_id"].nunique()) if "boot_id" in group else len(group),
            "NES_mean": float(pd.to_numeric(group["NES"], errors="coerce").mean()),
            "NES_lower": _quantile(group["NES"], lower_q),
            "NES_upper": _quantile(group["NES"], upper_q),
        }
        if "ES" in group:
            row["ES_mean"] = float(pd.to_numeric(group["ES"], errors="coerce").mean())
            row["ES_lower"] = _quantile(group["ES"], lower_q)
            row["ES_upper"] = _quantile(group["ES"], upper_q)
        if "padj" in group:
            row["padj_median"] = _quantile(group["padj"], 0.5)
        if "pt_mid" in group:
            row["pt_mid_median"] = _quantile(group["pt_mid"], 0.5)
            row["pt_mid_lower"] = _quantile(group["pt_mid"], lower_q)
            row["pt_mid_upper"] = _quantile(group["pt_mid"], upper_q)
        if "n_cells" in group:
            row["n_cells_median"] = _quantile(group["n_cells"], 0.5)
        rows.append(row)

    return pd.DataFrame(rows).sort_values([pathway_col, "window_id"]).reset_index(drop=True)


def bootstrap_trajectory_gsea(
    adata,
    gmt_path,
    pseudotime_key: str = "dpt_pseudotime",
    n_boot: int = 100,
    resample: str = "cells_within_windows",
    sample_key: Optional[str] = None,
    seed: int = 42,
    ci: tuple[float, float] = (0.025, 0.975),
    event_kwargs: Optional[dict] = None,
    **trajectory_kwargs,
) -> pd.DataFrame:
    """
    Bootstrap rolling-window trajectory GSEA and return NES confidence bands.

    ``resample="samples"`` resamples biological samples using ``sample_key``.
    Cell-level modes are useful for curve stability diagnostics, while
    sample-level resampling is preferred when biological replicates exist.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    if len(ci) != 2 or not (0 <= ci[0] < ci[1] <= 1):
        raise ValueError("ci must be a two-value tuple inside [0, 1]")
    if pseudotime_key not in adata.obs:
        raise ValueError(f"pseudotime_key '{pseudotime_key}' not found in adata.obs")

    resample = resample.lower().replace("-", "_")
    if resample not in {"cells", "cells_within_windows", "samples"}:
        raise ValueError("resample must be 'cells', 'cells_within_windows', or 'samples'")
    if resample == "samples" and sample_key is None:
        raise ValueError("sample_key is required when resample='samples'")

    event_kwargs = {} if event_kwargs is None else dict(event_kwargs)
    rng = np.random.default_rng(seed)
    result_frames = []
    event_frames = []
    gene_set_index = trajectory_kwargs.pop("gene_set_index", None)
    for boot_id in range(n_boot):
        if resample == "samples":
            indices = _bootstrap_sample_indices(adata.obs, sample_key, rng)
        else:
            indices = _bootstrap_cell_indices(adata.n_obs, rng)

        boot = _subset_with_unique_obs_names(adata, indices, boot_id)
        res = run_trajectory_gsea(
            boot,
            gmt_path=gmt_path,
            pseudotime_key=pseudotime_key,
            seed=seed + boot_id,
            gene_set_index=gene_set_index,
            **trajectory_kwargs,
        )
        if res is None or res.empty:
            continue
        gene_set_index = gene_set_index or res.attrs.get("gene_set_index")
        res = res.copy()
        res["boot_id"] = boot_id
        res["bootstrap_resample"] = resample
        result_frames.append(res)

        events = summarize_events(res, **event_kwargs)
        if not events.empty:
            events = events.copy()
            events["boot_id"] = boot_id
            events["bootstrap_resample"] = resample
            event_frames.append(events)

    boot_results = (
        pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    )
    boot_events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    bands = _make_bands(boot_results, ci[0], ci[1])
    bands.attrs["boot_results"] = boot_results
    bands.attrs["boot_events"] = boot_events
    bands.attrs["bootstrap"] = {
        "n_boot": int(n_boot),
        "resample": resample,
        "sample_key": sample_key,
        "seed": int(seed),
        "ci": tuple(ci),
    }
    return bands
