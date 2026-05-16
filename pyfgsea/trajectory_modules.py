from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .trajectory import _make_windows
from .trajectory_events import summarize_events
from .validation import _expression_matrix
from .wrapper import load_gmt


_MODULE_OUTPUT_FILENAMES = {
    "dynamic_gene_modules": "dynamic_gene_modules.tsv",
    "module_time_profiles": "module_time_profiles.tsv",
    "module_event_table": "module_event_table.tsv",
    "module_event_fdr": "module_event_fdr.tsv",
    "module_pathway_annotation": "module_pathway_annotation.tsv",
    "module_driver_score": "module_driver_score.tsv",
    "module_leading_edge_drivers": "module_leading_edge_drivers.tsv",
}


def _axis_mean(X, indices: np.ndarray) -> np.ndarray:
    sub = X[indices]
    if hasattr(sub, "mean"):
        return np.asarray(sub.mean(axis=0)).ravel().astype(float)
    return np.asarray(sub, dtype=float).mean(axis=0)


def _axis_var(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.nanvar(values, axis=0)


def _zscore_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    center = np.nanmean(matrix, axis=1, keepdims=True)
    scale = np.nanstd(matrix, axis=1, keepdims=True)
    scale[~np.isfinite(scale) | (scale <= 0)] = 1.0
    z = (matrix - center) / scale
    z[~np.isfinite(z)] = 0.0
    return z


def _zscore_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    center = float(np.nanmean(values))
    scale = float(np.nanstd(values))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    out = (values - center) / scale
    out[~np.isfinite(out)] = 0.0
    return out


def _smooth_profile(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    width = int(width)
    if width <= 1 or len(values) < 3:
        return values
    width = min(width, len(values))
    if width % 2 == 0:
        width += 1
    pad = width // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(padded, kernel, mode="valid")


def _bh_adjust(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(list(values), dtype=float)
    out = np.full(len(p), np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    pf = p[finite]
    order = np.argsort(pf)
    ranked = pf[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[finite] = restored
    return out


def _window_expression_matrix(
    adata,
    *,
    pseudotime_key: str,
    layer: Optional[str],
    use_raw: bool,
    window_size: int,
    step: int,
    min_cells: Optional[int],
    max_cells: Optional[int],
    target_span: Optional[float],
    span_step: Optional[float],
    window_mode: str,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    if pseudotime_key not in adata.obs:
        raise ValueError(f"pseudotime_key '{pseudotime_key}' not found in adata.obs")
    pt = pd.to_numeric(adata.obs[pseudotime_key], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(pt)
    if int(keep.sum()) < 2:
        raise ValueError("Fewer than two finite pseudotime values")
    work = adata[keep].copy()
    pt = pt[keep]
    X, genes, _source = _expression_matrix(work, layer=layer, use_raw=use_raw)
    order = np.argsort(pt)
    windows = _make_windows(
        order,
        window_size=window_size,
        step=step,
        pt=pt,
        window_mode=window_mode,
        min_cells=min_cells,
        max_cells=max_cells,
        target_span=target_span,
        span_step=span_step,
    )
    if not windows:
        raise ValueError("No trajectory windows could be constructed")

    rows = []
    means = []
    for window_id, (_start, _end, indices) in enumerate(windows):
        indices = np.asarray(indices, dtype=int)
        pt_values = pt[indices]
        means.append(_axis_mean(X, indices))
        rows.append(
            {
                "window_id": int(window_id),
                "pt_start": float(np.nanmin(pt_values)),
                "pt_end": float(np.nanmax(pt_values)),
                "pt_mid": float((np.nanmin(pt_values) + np.nanmax(pt_values)) / 2.0),
                "n_cells": int(len(indices)),
            }
        )
    return np.vstack(means), pd.DataFrame(rows), np.asarray(genes, dtype=str)


def _select_variable_genes(
    window_means: np.ndarray,
    genes: np.ndarray,
    *,
    top_variable_genes: Optional[int],
    min_gene_variance: float,
) -> tuple[np.ndarray, np.ndarray]:
    var = _axis_var(window_means)
    keep = np.isfinite(var) & (var >= float(min_gene_variance))
    if top_variable_genes is not None and int(top_variable_genes) > 0:
        eligible = np.where(keep)[0]
        if len(eligible) > int(top_variable_genes):
            order = eligible[np.argsort(var[eligible])[::-1]]
            keep = np.zeros(len(genes), dtype=bool)
            keep[order[: int(top_variable_genes)]] = True
    if not keep.any():
        raise ValueError("No variable genes passed module discovery filters")
    return window_means[:, keep], genes[keep]


def _module_gene_table(
    weights: np.ndarray,
    genes: np.ndarray,
    *,
    top_genes: int,
    min_weight_fraction: float,
) -> pd.DataFrame:
    rows = []
    for module_idx in range(weights.shape[1]):
        module = f"module_{module_idx + 1:02d}"
        w = weights[:, module_idx]
        max_w = float(np.nanmax(w)) if len(w) else 0.0
        threshold = max_w * float(min_weight_fraction)
        order = np.argsort(w)[::-1]
        selected = []
        for idx in order:
            if not np.isfinite(w[idx]) or w[idx] <= 0:
                continue
            if w[idx] >= threshold or len(selected) < int(top_genes):
                selected.append(idx)
        selected = selected[: int(top_genes)]
        total = float(np.nansum(w[w > 0]))
        for rank, gene_idx in enumerate(selected, start=1):
            rows.append(
                {
                    "module": module,
                    "gene": str(genes[gene_idx]),
                    "gene_rank": int(rank),
                    "module_weight": float(w[gene_idx]),
                    "relative_weight": float(w[gene_idx] / max_w) if max_w > 0 else np.nan,
                    "weight_fraction": float(w[gene_idx] / total) if total > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _module_profile_table(
    components: np.ndarray,
    windows: pd.DataFrame,
    *,
    smooth_width: int,
) -> pd.DataFrame:
    rows = []
    for module_idx in range(components.shape[0]):
        module = f"module_{module_idx + 1:02d}"
        profile = _zscore_vector(_smooth_profile(components[module_idx], smooth_width))
        for idx, row in windows.iterrows():
            rows.append(
                {
                    "module": module,
                    "Pathway": module,
                    "window_id": int(row["window_id"]),
                    "pt_start": row["pt_start"],
                    "pt_end": row["pt_end"],
                    "pt_mid": row["pt_mid"],
                    "module_score": float(profile[idx]),
                    "NES": float(profile[idx]),
                    "n_cells": int(row["n_cells"]),
                }
            )
    return pd.DataFrame(rows)


def _calibrate_module_events(
    profiles: pd.DataFrame,
    events: pd.DataFrame,
    *,
    event_threshold: float,
    min_consecutive: int,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame()
    out = events.copy()
    if int(n_permutations) <= 0:
        out["event_p"] = np.nan
        out["event_q"] = np.nan
        out["event_fdr"] = np.nan
        out["n_perm"] = 0
        out["null_model"] = "none"
        out["calibration_status"] = "exploratory_no_null"
        return out

    rng = np.random.default_rng(seed)
    null_rows = []
    for perm_id in range(int(n_permutations)):
        perm = profiles.copy()
        values = []
        for _, group in profiles.groupby("module", sort=False):
            shuffled = group["NES"].to_numpy(dtype=float).copy()
            rng.shuffle(shuffled)
            values.extend(shuffled.tolist())
        perm["NES"] = values
        perm_events = summarize_events(
            perm,
            pathway_col="module",
            time_col="pt_mid",
            nes_col="NES",
            fdr_col=None,
            nes_threshold=event_threshold,
            min_consecutive=min_consecutive,
        )
        if not perm_events.empty:
            perm_events["perm_id"] = int(perm_id)
            null_rows.append(perm_events[["module", "AUC_abs", "perm_id"]])
    null = pd.concat(null_rows, ignore_index=True) if null_rows else pd.DataFrame()
    p_values = []
    for row in out.itertuples():
        observed = float(row.AUC_abs)
        if null.empty or not np.isfinite(observed):
            p_values.append(np.nan)
            continue
        values = pd.to_numeric(null["AUC_abs"], errors="coerce").dropna().to_numpy(dtype=float)
        p_values.append((1.0 + float(np.sum(values >= observed))) / (1.0 + float(len(values))))
    out["event_p"] = p_values
    out["event_q"] = _bh_adjust(p_values)
    out["event_fdr"] = out["event_q"]
    out["n_perm"] = int(n_permutations)
    out["null_model"] = "module_time_permutation"
    out["calibration_status"] = "screening_time_permutation"
    return out


def _annotate_modules(
    modules: pd.DataFrame,
    genes: np.ndarray,
    gene_sets,
    *,
    top_genes: int,
) -> pd.DataFrame:
    if gene_sets is None or modules is None or modules.empty:
        return pd.DataFrame()
    raw_sets = load_gmt(str(gene_sets)) if isinstance(gene_sets, (str, Path)) else gene_sets
    universe = set(map(str, genes))
    rows = []
    for module, group in modules.groupby("module", sort=False):
        module_genes = set(group.sort_values("gene_rank")["gene"].astype(str).head(int(top_genes)))
        if not module_genes:
            continue
        for pathway, pathway_genes_raw in raw_sets.items():
            pathway_genes = set(map(str, pathway_genes_raw)) & universe
            if not pathway_genes:
                continue
            overlap = sorted(module_genes & pathway_genes)
            if not overlap:
                continue
            try:
                from scipy.stats import hypergeom

                p = float(
                    hypergeom.sf(
                        len(overlap) - 1,
                        len(universe),
                        len(pathway_genes),
                        len(module_genes),
                    )
                )
            except Exception:
                p = np.nan
            rows.append(
                {
                    "module": module,
                    "pathway": str(pathway),
                    "overlap_genes": ";".join(overlap),
                    "overlap_count": int(len(overlap)),
                    "module_gene_count": int(len(module_genes)),
                    "pathway_gene_count": int(len(pathway_genes)),
                    "enrichment_p": p,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["enrichment_q"] = _bh_adjust(out["enrichment_p"].to_numpy(dtype=float))
    return out.sort_values(["module", "enrichment_q", "overlap_count"], ascending=[True, True, False]).reset_index(drop=True)


def _module_drivers(modules: pd.DataFrame, events: pd.DataFrame, top_genes: int) -> pd.DataFrame:
    if modules is None or modules.empty or events is None or events.empty:
        return pd.DataFrame()
    rows = []
    for event in events.itertuples():
        group = modules[modules["module"].astype(str) == str(event.module)].copy()
        if group.empty:
            continue
        group["driver_score"] = pd.to_numeric(group["module_weight"], errors="coerce") * abs(float(event.peak_NES))
        group = group.sort_values("driver_score", ascending=False).head(int(top_genes))
        for row in group.itertuples():
            rows.append(
                {
                    "module": event.module,
                    "event_id": event.event_id,
                    "gene": row.gene,
                    "gene_rank": int(row.gene_rank),
                    "module_weight": float(row.module_weight),
                    "event_peak_time": float(event.peak_time),
                    "event_peak_score": float(event.peak_NES),
                    "driver_score": float(row.driver_score),
                }
            )
    return pd.DataFrame(rows)


def discover_dynamic_gene_modules(
    adata,
    *,
    gmt_path=None,
    pseudotime_key: str = "dpt_pseudotime",
    n_modules: int = 8,
    layer: Optional[str] = None,
    use_raw: bool = False,
    window_size: int = 200,
    step: int = 50,
    window_mode: str = "cell_count",
    min_cells: Optional[int] = None,
    max_cells: Optional[int] = None,
    target_span: Optional[float] = None,
    span_step: Optional[float] = None,
    top_variable_genes: Optional[int] = 2000,
    min_gene_variance: float = 1e-8,
    top_genes: int = 50,
    min_weight_fraction: float = 0.05,
    l1_weight: float = 0.1,
    smooth_width: int = 3,
    event_threshold: float = 0.75,
    min_consecutive: int = 2,
    n_permutations: int = 0,
    seed: int = 42,
    max_iter: int = 1000,
    nmf_tol: float = 1e-3,
) -> dict[str, pd.DataFrame]:
    """
    Discover de novo dynamic gene modules along pseudotime using NMF.

    The implementation builds a gene-by-window z-score matrix, factorizes a
    nonnegative shifted version with sparse NMF, smooths module time profiles,
    scans module events, and optionally annotates modules against pathway GMTs.
    """
    try:
        from sklearn.decomposition import NMF
    except Exception as exc:
        raise ImportError("discover_dynamic_gene_modules requires scikit-learn") from exc

    window_means, windows, genes = _window_expression_matrix(
        adata,
        pseudotime_key=pseudotime_key,
        layer=layer,
        use_raw=use_raw,
        window_size=window_size,
        step=step,
        min_cells=min_cells,
        max_cells=max_cells,
        target_span=target_span,
        span_step=span_step,
        window_mode=window_mode,
    )
    selected_means, selected_genes = _select_variable_genes(
        window_means,
        genes,
        top_variable_genes=top_variable_genes,
        min_gene_variance=min_gene_variance,
    )
    z_gene_time = _zscore_rows(selected_means.T)
    nmf_input = z_gene_time - np.nanmin(z_gene_time, axis=1, keepdims=True)
    nmf_input[~np.isfinite(nmf_input)] = 0.0
    nmf_input = np.maximum(nmf_input, 0.0)
    model = NMF(
        n_components=int(n_modules),
        init="nndsvda",
        random_state=int(seed),
        max_iter=int(max_iter),
        tol=float(nmf_tol),
        alpha_W=float(l1_weight),
        alpha_H=0.0,
        l1_ratio=1.0,
    )
    weights = model.fit_transform(nmf_input)
    components = model.components_
    modules = _module_gene_table(
        weights,
        selected_genes,
        top_genes=top_genes,
        min_weight_fraction=min_weight_fraction,
    )
    profiles = _module_profile_table(
        components,
        windows,
        smooth_width=smooth_width,
    )
    events = summarize_events(
        profiles,
        pathway_col="module",
        time_col="pt_mid",
        nes_col="NES",
        fdr_col=None,
        nes_threshold=event_threshold,
        min_consecutive=min_consecutive,
    )
    if not events.empty:
        events = events.rename(columns={"module": "module"})
        events["event_id"] = [
            f"{row.module}|event_{idx + 1:03d}" for idx, row in enumerate(events.itertuples())
        ]
        events["event_statistic_used"] = "module_AUC_abs"
        events["observed_event_statistic"] = pd.to_numeric(events["AUC_abs"], errors="coerce")
    event_fdr = _calibrate_module_events(
        profiles,
        events,
        event_threshold=event_threshold,
        min_consecutive=min_consecutive,
        n_permutations=n_permutations,
        seed=seed,
    )
    annotation = _annotate_modules(
        modules,
        selected_genes,
        gmt_path,
        top_genes=top_genes,
    )
    drivers = _module_drivers(modules, event_fdr, top_genes=min(15, int(top_genes)))
    for table in (modules, profiles, events, event_fdr, annotation, drivers):
        table.attrs["dynamic_module_discovery"] = {
            "n_modules": int(n_modules),
            "top_variable_genes": int(top_variable_genes) if top_variable_genes else None,
            "window_mode": window_mode,
            "event_threshold": float(event_threshold),
            "n_permutations": int(n_permutations),
        }
    return {
        "dynamic_gene_modules": modules,
        "module_time_profiles": profiles,
        "module_event_table": event_fdr,
        "module_event_fdr": event_fdr,
        "module_pathway_annotation": annotation,
        "module_driver_score": drivers,
        "module_leading_edge_drivers": drivers,
    }


def write_dynamic_gene_modules(
    tables: dict[str, pd.DataFrame],
    outdir: str | Path,
    *,
    sep: str = "\t",
) -> dict[str, Path]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, filename in _MODULE_OUTPUT_FILENAMES.items():
        table = tables.get(key, pd.DataFrame())
        path = out / filename
        table.to_csv(path, sep=sep, index=False, na_rep="NA")
        paths[key] = path
    return paths
