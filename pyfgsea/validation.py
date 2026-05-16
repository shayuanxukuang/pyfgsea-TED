from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .wrapper import load_gmt, prepare_pathways


@dataclass
class ValidationIssue:
    level: str
    check: str
    message: str
    detail: str = ""


def _report(issues: list[ValidationIssue]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "level": issue.level,
                "check": issue.check,
                "message": issue.message,
                "detail": issue.detail,
            }
            for issue in issues
        ],
        columns=["level", "check", "message", "detail"],
    )


def _add(issues: list[ValidationIssue], level: str, check: str, message: str, detail: str = ""):
    issues.append(ValidationIssue(level, check, message, detail))


def _expression_matrix(adata, layer: Optional[str] = None, use_raw: bool = False):
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers")
        return adata.layers[layer], np.asarray(adata.var_names), f"layer:{layer}"
    if use_raw:
        if adata.raw is None:
            raise ValueError("use_raw=True but adata.raw is not available")
        return adata.raw.X, np.asarray(adata.raw.var_names), "raw.X"
    return adata.X, np.asarray(adata.var_names), "X"


def _axis_sum(X):
    return np.asarray(X.sum(axis=0)).ravel()


def _axis_sum_squares(X):
    if hasattr(X, "power"):
        return np.asarray(X.power(2).sum(axis=0)).ravel()
    return np.asarray(np.square(X).sum(axis=0)).ravel()


def _finite_matrix_check(X) -> tuple[bool, int]:
    if hasattr(X, "data"):
        data = X.data
        bad = int((~np.isfinite(data)).sum())
        return bad == 0, bad
    arr = np.asarray(X)
    bad = int((~np.isfinite(arr)).sum())
    return bad == 0, bad


def _load_gene_sets(gmt):
    return load_gmt(gmt) if isinstance(gmt, str) else gmt


def _pathway_filter_counts(genes: np.ndarray, gmt, min_size: int, max_size: int) -> dict[str, int]:
    gene_set = set(map(str, genes))
    counts = {
        "raw_pathways": 0,
        "valid_pathways": 0,
        "no_gene_overlap": 0,
        "too_small": 0,
        "too_large": 0,
    }
    for _, pathway_genes in _load_gene_sets(gmt).items():
        counts["raw_pathways"] += 1
        overlap = len(set(map(str, pathway_genes)) & gene_set)
        if overlap == 0:
            counts["no_gene_overlap"] += 1
        elif overlap < min_size:
            counts["too_small"] += 1
        elif overlap > max_size:
            counts["too_large"] += 1
        else:
            counts["valid_pathways"] += 1
    return counts


def validate_inputs(
    adata,
    gmt_path,
    pseudotime_key: str = "dpt_pseudotime",
    condition_key: Optional[str] = None,
    branch_key: Optional[str] = None,
    layer: Optional[str] = None,
    use_raw: bool = False,
    min_size: int = 15,
    max_size: int = 500,
    window_size: Optional[int] = None,
    min_cells: Optional[int] = None,
    min_group_cells: Optional[int] = None,
    cell_weight_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Validate inputs for PyFgsea-TED without running GSEA.

    Returns a structured report with ``level`` in ``ok/info/warning/error``.
    """
    issues: list[ValidationIssue] = []
    n_obs = int(adata.n_obs)

    try:
        X, genes, source = _expression_matrix(adata, layer=layer, use_raw=use_raw)
        _add(issues, "info", "expression_source", f"Using expression source {source}")
    except Exception as exc:
        _add(issues, "error", "expression_source", str(exc))
        return _report(issues)

    duplicated = pd.Index(genes).duplicated()
    if duplicated.any():
        examples = ", ".join(map(str, pd.Index(genes)[duplicated][:5]))
        _add(
            issues,
            "error",
            "gene_names_unique",
            "Expression gene names are duplicated; pass make_var_names_unique=True or fix adata.var_names.",
            examples,
        )
    else:
        _add(issues, "ok", "gene_names_unique", "Gene names are unique")

    if pseudotime_key not in adata.obs:
        _add(
            issues,
            "error",
            "pseudotime_key",
            f"pseudotime_key '{pseudotime_key}' is not present in adata.obs",
        )
    else:
        pt = pd.to_numeric(adata.obs[pseudotime_key], errors="coerce").to_numpy()
        missing = int((~np.isfinite(pt)).sum())
        if missing:
            _add(
                issues,
                "warning",
                "pseudotime_finite",
                f"{missing} cells have missing or non-finite pseudotime",
                "run_trajectory_gsea drops these by default; pass dropna=False to fail fast.",
            )
        else:
            _add(issues, "ok", "pseudotime_finite", "Pseudotime is finite for all cells")
        if np.isfinite(pt).sum() < 2:
            _add(issues, "error", "pseudotime_range", "Fewer than two finite pseudotime values")
        elif float(np.nanmax(pt) - np.nanmin(pt)) <= 0:
            _add(issues, "error", "pseudotime_range", "Pseudotime has zero range")

    if cell_weight_key is not None:
        if cell_weight_key not in adata.obs:
            _add(
                issues,
                "error",
                "cell_weight_key",
                f"cell_weight_key '{cell_weight_key}' is not present in adata.obs",
            )
        else:
            weights = pd.to_numeric(adata.obs[cell_weight_key], errors="coerce")
            nonfinite = int((~np.isfinite(weights.to_numpy(dtype=float))).sum())
            negative = int((weights < 0).sum())
            total = float(weights.sum(skipna=True))
            if nonfinite:
                _add(
                    issues,
                    "error",
                    "cell_weight_finite",
                    f"{nonfinite} cells have missing or non-finite weights",
                )
            elif negative:
                _add(
                    issues,
                    "error",
                    "cell_weight_nonnegative",
                    f"{negative} cells have negative weights",
                )
            elif total <= 0:
                _add(issues, "error", "cell_weight_sum", "Cell weights sum to zero")
            else:
                _add(
                    issues,
                    "ok",
                    "cell_weight_key",
                    f"Cell weights in '{cell_weight_key}' look usable",
                    {
                        "min": float(weights.min()),
                        "max": float(weights.max()),
                        "mean": float(weights.mean()),
                        "sum": total,
                    }.__repr__(),
                )

    finite_ok, n_bad = _finite_matrix_check(X)
    if finite_ok:
        _add(issues, "ok", "expression_finite", "Expression matrix contains no non-finite stored values")
    else:
        _add(issues, "error", "expression_finite", f"Expression matrix has {n_bad} non-finite stored values")

    try:
        sums = _axis_sum(X)
        sums_sq = _axis_sum_squares(X)
        n = max(int(X.shape[0]), 1)
        variances = np.maximum((sums_sq - (sums * sums / n)) / max(n - 1, 1), 0.0)
        constant = int((variances <= 1e-12).sum())
        if constant == len(genes):
            _add(issues, "warning", "constant_genes", "All genes are constant across cells")
        elif constant:
            _add(issues, "info", "constant_genes", f"{constant} genes are constant across cells")
        else:
            _add(issues, "ok", "constant_genes", "No constant genes detected")
    except Exception as exc:
        _add(issues, "warning", "constant_genes", f"Could not check constant genes: {exc}")

    try:
        counts = _pathway_filter_counts(genes, gmt_path, min_size=min_size, max_size=max_size)
        if counts["valid_pathways"] == 0:
            _add(
                issues,
                "warning",
                "pathway_coverage",
                "No pathways pass gene overlap and size filters",
                str(counts),
            )
        else:
            _add(
                issues,
                "ok",
                "pathway_coverage",
                f"{counts['valid_pathways']} pathways pass filters",
                str(counts),
            )
    except Exception as exc:
        _add(issues, "error", "pathway_coverage", f"Could not read or filter GMT: {exc}")

    min_required = min_group_cells or min_cells or window_size
    for key_name, key in (("condition_key", condition_key), ("branch_key", branch_key)):
        if key is None:
            continue
        if key not in adata.obs:
            _add(issues, "error", key_name, f"{key_name} '{key}' is not present in adata.obs")
            continue
        counts = adata.obs[key].value_counts(dropna=True)
        if len(counts) < 2:
            _add(issues, "warning", key_name, f"{key} has fewer than two non-empty groups")
        if min_required is not None and not counts.empty and counts.min() < min_required:
            _add(
                issues,
                "warning",
                f"{key_name}_balance",
                f"Smallest {key} group has {int(counts.min())} cells, below {int(min_required)}",
                counts.to_dict().__repr__(),
            )
        else:
            _add(issues, "ok", f"{key_name}_balance", f"{key} group sizes look usable", counts.to_dict().__repr__())

    if n_obs == 0:
        _add(issues, "error", "n_cells", "AnnData has zero cells")
    elif window_size is not None and window_size > n_obs:
        _add(issues, "warning", "window_size", "window_size exceeds number of cells; no windows will be generated")
    else:
        _add(issues, "ok", "n_cells", f"AnnData has {n_obs} cells")

    return _report(issues)


def _contains_issue(report: pd.DataFrame, level: str) -> bool:
    return not report.empty and (report["level"] == level).any()


def validate_trajectory_result(
    result: pd.DataFrame,
    events: Optional[pd.DataFrame] = None,
    leading_edge: Optional[pd.DataFrame] = None,
    gmt_path=None,
    pathway_col: str = "Pathway",
    time_col: str = "pt_mid",
    group_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Validate internal consistency of TED window, event, and leading-edge tables.
    """
    issues: list[ValidationIssue] = []
    if result is None or result.empty:
        _add(issues, "error", "result_nonempty", "Trajectory result is empty")
        return _report(issues)

    required = {pathway_col, "ES", "NES", "P-value", "padj", time_col}
    missing = sorted(required - set(result.columns))
    if missing:
        _add(issues, "error", "required_columns", "Missing required result columns", ", ".join(missing))
        return _report(issues)
    _add(issues, "ok", "required_columns", "Required result columns are present")

    for col in ("ES", "NES", "P-value", "padj", time_col):
        values = pd.to_numeric(result[col], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(values).all():
            _add(issues, "ok", f"{col}_finite", f"{col} is finite")
        else:
            _add(issues, "error", f"{col}_finite", f"{col} contains non-finite values")

    for col in ("P-value", "padj"):
        values = pd.to_numeric(result[col], errors="coerce")
        if ((values >= 0) & (values <= 1)).all():
            _add(issues, "ok", f"{col}_range", f"{col} is within [0, 1]")
        else:
            _add(issues, "error", f"{col}_range", f"{col} has values outside [0, 1]")

    if group_cols is None:
        group_cols = [
            col
            for col in ("condition", "branch", "grid_run", "ranker", "window_mode")
            if col in result.columns
        ]

    grouping = [pathway_col] + group_cols
    bad_pathways = []
    for group_key, group in result.groupby(grouping, sort=False):
        times = pd.to_numeric(group[time_col], errors="coerce").to_numpy(dtype=float)
        if np.any(np.diff(times) < -1e-12):
            bad_pathways.append(str(group_key))
    if bad_pathways:
        _add(issues, "error", "time_monotonic", "Window midpoint is not monotonic within pathways", ", ".join(bad_pathways[:5]))
    else:
        _add(issues, "ok", "time_monotonic", "Window midpoint is monotonic within pathways")

    span_start_col = "pt_start" if "pt_start" in result.columns else time_col
    span_end_col = "pt_end" if "pt_end" in result.columns else time_col
    pt_min = float(pd.to_numeric(result[span_start_col], errors="coerce").min())
    pt_max = float(pd.to_numeric(result[span_end_col], errors="coerce").max())
    pt_span = max(pt_max - pt_min, 0.0)

    if events is not None and not events.empty:
        event_missing = sorted({pathway_col, "peak_time", "duration", "event_label"} - set(events.columns))
        if event_missing:
            _add(issues, "error", "event_columns", "Missing event columns", ", ".join(event_missing))
        else:
            _add(issues, "ok", "event_columns", "Event columns are present")
            result_times = {
                pathway: set(np.round(group[time_col].to_numpy(dtype=float), 12))
                for pathway, group in result.groupby(pathway_col)
            }
            missing_peaks = []
            bad_duration = []
            bad_onset = []
            bad_labels = []
            for _, row in events.iterrows():
                pathway = row[pathway_col]
                peak_time = row.get("peak_time", np.nan)
                if np.isfinite(peak_time) and np.round(float(peak_time), 12) not in result_times.get(pathway, set()):
                    missing_peaks.append(str(pathway))
                duration = row.get("duration", np.nan)
                if np.isfinite(duration) and duration > pt_span + 1e-12:
                    bad_duration.append(str(pathway))
                onset = row.get("activation_onset", np.nan)
                label = str(row.get("event_label", ""))
                if (
                    "activation" in label
                    and "suppression" not in label
                    and np.isfinite(onset)
                    and np.isfinite(peak_time)
                    and onset > peak_time + 1e-12
                ):
                    bad_onset.append(str(pathway))
                peak = row.get("peak_NES", np.nan)
                trough = row.get("trough_NES", np.nan)
                if (
                    "activation" in label
                    and "suppression" not in label
                    and "biphasic" not in label
                    and "recurrent" not in label
                    and np.isfinite(peak)
                    and np.isfinite(trough)
                    and peak < abs(trough)
                ):
                    bad_labels.append(str(pathway))
                if (
                    "suppression" in label
                    and "activation" not in label
                    and "recovery" not in label
                    and "biphasic" not in label
                    and "recurrent" not in label
                    and np.isfinite(peak)
                    and np.isfinite(trough)
                    and abs(trough) < peak
                ):
                    bad_labels.append(str(pathway))

            for check, bad, msg in (
                ("event_peak_time", missing_peaks, "Event peak_time is not present in source window table"),
                ("event_duration", bad_duration, "Event duration exceeds pseudotime span"),
                ("event_onset_order", bad_onset, "activation_onset occurs after peak_time"),
                ("event_label_direction", bad_labels, "event_label direction does not match NES extrema"),
            ):
                if bad:
                    _add(issues, "error", check, msg, ", ".join(bad[:5]))
                else:
                    _add(issues, "ok", check, f"{check} passed")

    if gmt_path is not None and "leading_edge" in result.columns:
        gmt = _load_gene_sets(gmt_path)
        bad = []
        for _, row in result.iterrows():
            pathway = row[pathway_col]
            allowed = set(map(str, gmt.get(pathway, [])))
            genes = [gene for gene in str(row.get("leading_edge", "")).split(";") if gene]
            if allowed and any(gene not in allowed for gene in genes):
                bad.append(str(pathway))
        if bad:
            _add(issues, "error", "leading_edge_membership", "Leading-edge genes outside their pathway", ", ".join(bad[:5]))
        else:
            _add(issues, "ok", "leading_edge_membership", "Leading-edge genes are pathway members")

    if leading_edge is not None and not leading_edge.empty and "core_genes" in leading_edge.columns:
        bad_core = []
        for pathway, group in leading_edge.groupby(pathway_col, sort=False):
            edge_sets = [
                set(str(value).split(";")) - {""}
                for value in group.get("leading_edge_genes", pd.Series(dtype=str))
            ]
            for core_text in group["core_genes"].dropna().astype(str).unique():
                core = set(core_text.split(";")) - {""}
                for gene in core:
                    if sum(gene in edge for edge in edge_sets) < 1:
                        bad_core.append(str(pathway))
                        break
        if bad_core:
            _add(issues, "error", "core_leading_edge", "Core genes were not found in leading-edge windows", ", ".join(bad_core[:5]))
        else:
            _add(issues, "ok", "core_leading_edge", "Core leading-edge genes appear in leading-edge windows")

    if not _contains_issue(_report(issues), "error"):
        _add(issues, "ok", "trajectory_result", "Trajectory result passed consistency checks")
    return _report(issues)
