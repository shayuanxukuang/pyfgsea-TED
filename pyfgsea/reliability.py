from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .calibration import estimate_event_fdr
from .trajectory import run_trajectory_gsea
from .trajectory_benchmark import (
    TRUTH_TYPES,
    make_synthetic_trajectory_truth,
    score_synthetic_events,
)
from .trajectory_compare import compare_trajectory_gsea, run_branch_gsea
from .trajectory_events import summarize_events


RELIABILITY_TRUTH_TYPES = [
    "monotonic_up",
    "monotonic_down",
    "narrow_transient_pulse",
    "biphasic_early_late",
    "sparse_dropout_burst",
    "branch_specific_activation",
    "condition_delayed_activation",
    "condition_amplitude_loss",
    "replicate_imbalance",
    "graph_branch_mixing",
]


def _expected_peak(truth_type: str) -> float:
    return {
        "monotonic_up": 1.0,
        "monotonic_down": 0.0,
        "narrow_transient_pulse": 0.5,
        "broad_transient_plateau": 0.5,
        "biphasic_early_late": 0.22,
        "sparse_dropout_burst": 0.475,
        "branch_specific_activation": 0.75,
        "condition_delayed_activation": 0.2,
        "condition_amplitude_loss": 0.0,
        "replicate_imbalance": 0.2,
        "graph_branch_mixing": np.nan,
    }.get(truth_type, np.nan)


def _expected_duration(truth_type: str) -> float:
    return {
        "narrow_transient_pulse": 0.1,
        "biphasic_early_late": 0.2,
        "sparse_dropout_burst": 0.05,
    }.get(truth_type, np.nan)


def _condition_amplitude_loss(
    n_cells: int,
    n_genes: int,
    pathway_size: int,
    effect_size: float,
    noise_sd: float,
    seed: int,
):
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("condition amplitude loss benchmark requires anndata") from exc

    rng = np.random.default_rng(seed)
    pt = np.linspace(0.0, 1.0, n_cells)
    condition = np.where(np.arange(n_cells) % 2 == 0, "control", "case")
    X = rng.normal(loc=0.0, scale=noise_sd, size=(n_cells, n_genes))
    X = X - X.min() + 0.1
    profile = 2.0 * np.clip((pt - 0.3) / 0.5, 0, 1)
    X[condition == "control", :pathway_size] += effect_size * profile[condition == "control", None]
    X[condition == "case", :pathway_size] += 0.35 * effect_size * profile[condition == "case", None]
    obs = pd.DataFrame({"dpt_pseudotime": pt, "condition": pd.Categorical(condition)})
    genes = [f"Gene_{idx}" for idx in range(n_genes)]
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
    return adata, {
        "TRUE_SIGNAL": genes[:pathway_size],
        "RANDOM_BACKGROUND": genes[pathway_size : 2 * pathway_size],
    }


def _add_replicates(
    adata,
    condition_key: str = "condition",
    replicate_key: str = "donor",
    n_reps: int = 3,
    imbalanced: bool = False,
):
    labels = adata.obs[condition_key].astype(str).to_numpy()
    out = []
    counters = {condition: 0 for condition in np.unique(labels)}
    for label in labels:
        idx = counters[label]
        counters[label] += 1
        if imbalanced and label == "case":
            donor = 1 if idx < int(0.75 * max(1, (labels == label).sum())) else (idx % n_reps) + 1
        else:
            donor = (idx % n_reps) + 1
        out.append(f"{label}_{donor}")
    adata = adata.copy()
    adata.obs[replicate_key] = out
    return adata


def _graph_branch_mixing(
    n_cells: int,
    n_genes: int,
    pathway_size: int,
    effect_size: float,
    noise_sd: float,
    seed: int,
):
    try:
        import anndata as ad
        from scipy import sparse
    except ImportError as exc:
        raise ImportError("graph branch-mixing benchmark requires anndata and scipy") from exc

    rng = np.random.default_rng(seed)
    n_branch = n_cells // 2
    n_cells = n_branch * 2
    pt = np.tile(np.linspace(0.0, 1.0, n_branch), 2)
    branch = np.array(["branch_a"] * n_branch + ["branch_b"] * n_branch)
    X = rng.normal(loc=0.0, scale=noise_sd, size=(n_cells, n_genes))
    X = X - X.min() + 0.1
    profile = np.clip((pt - 0.4) / 0.6, 0, 1)
    X[branch == "branch_a", :pathway_size] += effect_size * profile[branch == "branch_a", None]
    rows = []
    cols = []

    def add_edge(i, j):
        rows.extend([i, j])
        cols.extend([j, i])

    for offset in range(n_branch - 1):
        add_edge(offset, offset + 1)
        add_edge(n_branch + offset, n_branch + offset + 1)
        add_edge(offset, n_branch + offset)
        if offset + 1 < n_branch:
            add_edge(offset, n_branch + offset + 1)
    graph = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_cells, n_cells))
    genes = [f"Gene_{idx}" for idx in range(n_genes)]
    obs = pd.DataFrame(
        {
            "dpt_pseudotime": pt,
            "branch": pd.Categorical(branch),
            "fate_prob": np.where(branch == "branch_a", 0.9, 0.1),
        }
    )
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
    adata.obsp["connectivities"] = graph
    return adata, {
        "TRUE_SIGNAL": genes[:pathway_size],
        "RANDOM_BACKGROUND": genes[pathway_size : 2 * pathway_size],
    }


def _event_q_for_pathway(event_fdr: pd.DataFrame, pathway: str, stat: str = "max_abs_NES") -> float:
    return _event_value_for_pathway(event_fdr, pathway, "event_q", stat=stat)


def _event_p_for_pathway(event_fdr: pd.DataFrame, pathway: str, stat: str = "max_abs_NES") -> float:
    return _event_value_for_pathway(event_fdr, pathway, "event_p", stat=stat)


def _event_value_for_pathway(
    event_fdr: pd.DataFrame,
    pathway: str,
    column: str,
    stat: Optional[str] = None,
) -> float:
    if event_fdr is None or event_fdr.empty:
        return np.nan
    table = event_fdr[event_fdr["pathway"].astype(str) == pathway]
    if stat is not None and "event_stat" in table:
        table = table[table["event_stat"].astype(str) == stat]
    if table.empty or column not in table:
        return np.nan
    return float(pd.to_numeric(table[column], errors="coerce").min())


def _event_row(events: pd.DataFrame, pathway: str) -> pd.Series:
    if events is None or events.empty:
        return pd.Series(dtype=object)
    hit = events[events["Pathway"].astype(str) == pathway]
    return hit.iloc[0] if not hit.empty else pd.Series(dtype=object)


def _as_float(value, default: float = np.nan) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(out) if np.isfinite(out) else default


def _finite_max(*values: float) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return max(finite) if finite else np.nan


def _event_windows(row: pd.Series) -> float:
    if row is None or row.empty:
        return 0.0
    if "event_window_count" in row:
        return _as_float(row.get("event_window_count"), 0.0)
    if "significant_window_count" in row:
        return _as_float(row.get("significant_window_count"), 0.0)
    return 0.0


def _event_class(row: pd.Series, fallback: str = "missing") -> str:
    if row is None or row.empty:
        return fallback
    return str(row.get("event_confidence_class", fallback))


def _event_fields(row: pd.Series, prefix: str) -> dict:
    if row is None or row.empty:
        return {
            f"{prefix}_event_confidence_class": "missing",
            f"{prefix}_event_windows": 0.0,
            f"{prefix}_duration": np.nan,
        }
    return {
        f"{prefix}_event_confidence_class": _event_class(row),
        f"{prefix}_event_windows": _event_windows(row),
        f"{prefix}_duration": _as_float(row.get("duration"), np.nan),
    }


_GATE_PRESETS = {
    "quick": {
        "event_p_threshold": 0.10,
        "q_threshold": None,
        "min_event_windows": 2,
        "min_ranker_support": None,
        "min_seed_support": None,
        "max_technical_confound_score": 0.5,
        "min_balance_pass_rate": 0.33,
        "min_median_balance_score": 0.6,
        "balance_logic": "or",
        "min_event_balance_coverage": 0.5,
        "require_bootstrap_support": False,
    },
    "screening": {
        "event_p_threshold": 0.10,
        "q_threshold": None,
        "min_event_windows": 2,
        "min_ranker_support": None,
        "min_seed_support": None,
        "max_technical_confound_score": 0.5,
        "min_balance_pass_rate": 0.33,
        "min_median_balance_score": 0.6,
        "balance_logic": "or",
        "min_event_balance_coverage": 0.5,
        "require_bootstrap_support": False,
    },
    "candidate": {
        "event_p_threshold": None,
        "q_threshold": 0.10,
        "min_event_windows": 2,
        "min_ranker_support": 0.33,
        "min_seed_support": 0.5,
        "max_technical_confound_score": 0.5,
        "min_balance_pass_rate": 0.5,
        "min_median_balance_score": 0.7,
        "balance_logic": "and",
        "min_event_balance_coverage": 0.7,
        "require_bootstrap_support": False,
    },
    "discovery": {
        "event_p_threshold": None,
        "q_threshold": 0.05,
        "min_event_windows": 2,
        "min_ranker_support": 0.5,
        "min_seed_support": 0.8,
        "max_technical_confound_score": 0.5,
        "min_balance_pass_rate": 0.8,
        "min_median_balance_score": 0.8,
        "balance_logic": "and",
        "min_event_balance_coverage": 0.8,
        "require_bootstrap_support": True,
    },
}


def _gate_preset(mode: str) -> dict:
    key = str(mode).lower().replace("-", "_")
    if key == "raw":
        key = "screening"
    if key not in _GATE_PRESETS:
        supported = ", ".join(sorted(_GATE_PRESETS))
        raise ValueError(f"Unsupported synthetic gate mode '{mode}'. Supported modes: {supported}")
    return dict(_GATE_PRESETS[key])


def _is_transition_ranker(ranker: str) -> bool:
    return str(ranker) in {"neighbor_contrast", "local_slope"}


def _eligible_for_discovery(row: pd.Series) -> tuple[bool, str]:
    null_model = str(row.get("null_model", ""))
    if "uncalibrated" in null_model:
        return False, "uncalibrated_null_model"
    return True, "eligible"


def _gate_event(
    *,
    event_p: float,
    event_q: float,
    event_windows: float,
    duration: float,
    event_class: str,
    null_model: str,
    ranker: str,
    ranker_support: float,
    seed_support: float,
    technical_confound_score: float,
    bootstrap_support: float,
    balance_pass_rate: float,
    median_balance_score: float,
    event_balance_coverage: float,
    event_p_threshold: Optional[float],
    q_threshold: Optional[float],
    min_event_windows: int,
    min_duration: float,
    min_ranker_support: Optional[float],
    min_seed_support: Optional[float],
    max_technical_confound_score: float,
    min_balance_pass_rate: Optional[float],
    min_median_balance_score: Optional[float],
    balance_logic: str,
    min_event_balance_coverage: Optional[float],
    require_bootstrap_support: bool,
) -> tuple[bool, str]:
    reasons = []
    if "uncalibrated" in str(null_model):
        reasons.append("uncalibrated_null_model")
    if event_p_threshold is not None:
        if not np.isfinite(event_p):
            reasons.append("missing_event_p")
        elif event_p > event_p_threshold:
            reasons.append("event_p_gt_threshold")
    if q_threshold is not None:
        if not np.isfinite(event_q):
            reasons.append("missing_event_q")
        elif event_q > q_threshold:
            reasons.append("event_q_gt_threshold")

    if _is_transition_ranker(ranker):
        min_event_windows = max(int(min_event_windows), 3)
    if not np.isfinite(event_windows) or event_windows < min_event_windows:
        reasons.append("min_event_windows")
    if np.isfinite(duration) and duration <= min_duration:
        reasons.append("min_duration")
    if str(event_class) == "single_window_pulse":
        reasons.append("single_window_pulse")
    if min_ranker_support is not None and (
        not np.isfinite(ranker_support) or ranker_support < min_ranker_support
    ):
        reasons.append("ranker_support")
    if min_seed_support is not None and (
        not np.isfinite(seed_support) or seed_support < min_seed_support
    ):
        reasons.append("seed_support")
    if require_bootstrap_support and not (
        np.isfinite(bootstrap_support) and bootstrap_support > 0
    ):
        reasons.append("bootstrap_support")
    balance_checks = []
    if min_balance_pass_rate is not None and np.isfinite(balance_pass_rate):
        balance_checks.append(
            ("balance_pass_rate", balance_pass_rate >= min_balance_pass_rate)
        )
    if min_median_balance_score is not None and np.isfinite(median_balance_score):
        balance_checks.append(
            ("median_balance_score", median_balance_score >= min_median_balance_score)
        )
    if balance_checks:
        logic = str(balance_logic or "and").lower()
        if logic == "or":
            if not any(ok for _name, ok in balance_checks):
                reasons.append("balance_pass_rate_or_score")
        else:
            reasons.extend(name for name, ok in balance_checks if not ok)
    if (
        min_event_balance_coverage is not None
        and np.isfinite(event_balance_coverage)
        and event_balance_coverage < min_event_balance_coverage
    ):
        reasons.append("event_balance_coverage")
    if (
        np.isfinite(technical_confound_score)
        and technical_confound_score > max_technical_confound_score
    ):
        reasons.append("technical_confound")
    return (len(reasons) == 0, "pass" if not reasons else ";".join(reasons))


def apply_synthetic_discovery_gate(
    table: pd.DataFrame,
    mode: str = "candidate",
    q_threshold: Optional[float] = None,
    event_p_threshold: Optional[float] = None,
    min_event_windows: Optional[int] = None,
    min_duration: float = 0.0,
    min_ranker_support: Optional[float] = None,
    min_seed_support: Optional[float] = None,
    max_technical_confound_score: Optional[float] = None,
    min_balance_pass_rate: Optional[float] = None,
    min_median_balance_score: Optional[float] = None,
    min_event_balance_coverage: Optional[float] = None,
    balance_logic: Optional[str] = None,
    require_bootstrap_support: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Add TED gate calls to a synthetic truth benchmark table.

    Raw ``power`` and ``false_positive_rate`` remain unchanged. The gated
    columns answer a stricter evidence question. ``mode="screening"`` is
    suitable for low-permutation CI/debug runs, ``mode="candidate"`` is the
    default robust-candidate filter, and ``mode="discovery"`` is conservative
    enough for formal claims.
    """
    if table is None or table.empty:
        return pd.DataFrame() if table is None else table.copy()

    mode_key = str(mode).lower().replace("-", "_")
    if mode_key == "quick":
        mode_key = "screening"
    params = _gate_preset(mode_key)
    if q_threshold is not None:
        params["q_threshold"] = q_threshold
    if event_p_threshold is not None:
        params["event_p_threshold"] = event_p_threshold
    if min_event_windows is not None:
        params["min_event_windows"] = int(min_event_windows)
    if min_ranker_support is not None:
        params["min_ranker_support"] = min_ranker_support
    if min_seed_support is not None:
        params["min_seed_support"] = min_seed_support
    if max_technical_confound_score is not None:
        params["max_technical_confound_score"] = max_technical_confound_score
    if min_balance_pass_rate is not None:
        params["min_balance_pass_rate"] = min_balance_pass_rate
    if min_median_balance_score is not None:
        params["min_median_balance_score"] = min_median_balance_score
    if min_event_balance_coverage is not None:
        params["min_event_balance_coverage"] = min_event_balance_coverage
    if balance_logic is not None:
        params["balance_logic"] = balance_logic
    if require_bootstrap_support is not None:
        params["require_bootstrap_support"] = bool(require_bootstrap_support)

    out = table.copy()
    if "seed_support" not in out:
        out["seed_support"] = 1.0
    if "technical_confound_score" not in out:
        out["technical_confound_score"] = 0.0
    if "ranker_support" not in out:
        out["ranker_support"] = np.nan
    if "bootstrap_support" not in out:
        out["bootstrap_support"] = np.nan
    if "raw_power" not in out and "power" in out:
        out["raw_power"] = pd.to_numeric(out["power"], errors="coerce")
    if "raw_false_positive_rate" not in out and "false_positive_rate" in out:
        out["raw_false_positive_rate"] = pd.to_numeric(
            out["false_positive_rate"], errors="coerce"
        )
    eligibility = out.apply(_eligible_for_discovery, axis=1, result_type="expand")
    out["eligible_for_discovery"] = eligibility[0].astype(bool)
    out["discovery_status"] = np.where(
        out["eligible_for_discovery"], "eligible", eligibility[1].astype(str)
    )

    target_pass = []
    target_reason = []
    background_pass = []
    background_reason = []
    for _, row in out.iterrows():
        ranker_support = _as_float(row.get("ranker_support"))
        seed_support = _as_float(row.get("seed_support"), 1.0)
        technical_score = _as_float(row.get("technical_confound_score"), 0.0)
        bootstrap_support = _as_float(row.get("bootstrap_support"), np.nan)
        balance_pass_rate = _as_float(row.get("balance_pass_rate"), np.nan)
        median_balance_score = _as_float(row.get("median_balance_score"), np.nan)
        if mode_key == "screening":
            median_balance_score = _finite_max(
                median_balance_score,
                _as_float(row.get("median_core_balance_score"), np.nan),
            )
        target_event_balance_coverage = _as_float(
            row.get(
                "target_event_balance_coverage",
                row.get("event_balance_coverage", np.nan),
            ),
            np.nan,
        )
        background_event_balance_coverage = _as_float(
            row.get(
                "background_event_balance_coverage",
                row.get("event_balance_coverage", np.nan),
            ),
            np.nan,
        )
        if mode_key == "screening":
            target_event_balance_coverage = _finite_max(
                target_event_balance_coverage,
                _as_float(row.get("target_event_core_balance_coverage"), np.nan),
            )
            background_event_balance_coverage = _finite_max(
                background_event_balance_coverage,
                _as_float(row.get("background_event_core_balance_coverage"), np.nan),
            )
        ok, reason = _gate_event(
            event_p=_as_float(row.get("target_event_p")),
            event_q=_as_float(row.get("target_event_q")),
            event_windows=_as_float(row.get("target_event_windows"), 0.0),
            duration=_as_float(row.get("target_duration"), np.nan),
            event_class=str(row.get("target_event_confidence_class", "missing")),
            null_model=str(row.get("null_model", "")),
            ranker=str(row.get("ranker", "")),
            ranker_support=ranker_support,
            seed_support=seed_support,
            technical_confound_score=technical_score,
            bootstrap_support=bootstrap_support,
            balance_pass_rate=balance_pass_rate,
            median_balance_score=median_balance_score,
            event_balance_coverage=target_event_balance_coverage,
            event_p_threshold=params["event_p_threshold"],
            q_threshold=params["q_threshold"],
            min_event_windows=int(params["min_event_windows"]),
            min_duration=min_duration,
            min_ranker_support=params["min_ranker_support"],
            min_seed_support=params["min_seed_support"],
            max_technical_confound_score=float(params["max_technical_confound_score"]),
            min_balance_pass_rate=params["min_balance_pass_rate"],
            min_median_balance_score=params.get("min_median_balance_score"),
            balance_logic=params.get("balance_logic", "and"),
            min_event_balance_coverage=params.get("min_event_balance_coverage"),
            require_bootstrap_support=bool(params["require_bootstrap_support"]),
        )
        target_pass.append(ok)
        target_reason.append(reason)
        ok, reason = _gate_event(
            event_p=_as_float(row.get("background_event_p")),
            event_q=_as_float(row.get("background_event_q")),
            event_windows=_as_float(row.get("background_event_windows"), 0.0),
            duration=_as_float(row.get("background_duration"), np.nan),
            event_class=str(row.get("background_event_confidence_class", "missing")),
            null_model=str(row.get("null_model", "")),
            ranker=str(row.get("ranker", "")),
            ranker_support=ranker_support,
            seed_support=seed_support,
            technical_confound_score=technical_score,
            bootstrap_support=bootstrap_support,
            balance_pass_rate=balance_pass_rate,
            median_balance_score=median_balance_score,
            event_balance_coverage=background_event_balance_coverage,
            event_p_threshold=params["event_p_threshold"],
            q_threshold=params["q_threshold"],
            min_event_windows=int(params["min_event_windows"]),
            min_duration=min_duration,
            min_ranker_support=params["min_ranker_support"],
            min_seed_support=params["min_seed_support"],
            max_technical_confound_score=float(params["max_technical_confound_score"]),
            min_balance_pass_rate=params["min_balance_pass_rate"],
            min_median_balance_score=params.get("min_median_balance_score"),
            balance_logic=params.get("balance_logic", "and"),
            min_event_balance_coverage=params.get("min_event_balance_coverage"),
            require_bootstrap_support=bool(params["require_bootstrap_support"]),
        )
        background_pass.append(ok)
        background_reason.append(reason)

    out[f"target_{mode_key}_pass"] = target_pass
    out[f"background_{mode_key}_pass"] = background_pass
    out[f"{mode_key}_power"] = out[f"target_{mode_key}_pass"].astype(float)
    out[f"{mode_key}_false_positive_rate"] = out[
        f"background_{mode_key}_pass"
    ].astype(float)
    out[f"target_{mode_key}_gate_reason"] = target_reason
    out[f"background_{mode_key}_gate_reason"] = background_reason
    out["target_gate_reason"] = target_reason
    out["background_gate_reason"] = background_reason
    out["gated_power"] = out[f"{mode_key}_power"]
    out["gated_false_positive_rate"] = out[f"{mode_key}_false_positive_rate"]
    if mode_key == "discovery":
        out["target_discovery_pass"] = target_pass
        out["background_discovery_pass"] = background_pass
    elif "target_discovery_pass" not in out:
        out["target_discovery_pass"] = False
        out["background_discovery_pass"] = False
    out.attrs[f"{mode_key}_gate"] = {
        "event_p_threshold": params["event_p_threshold"],
        "event_q_threshold": params["q_threshold"],
        "min_event_windows": int(params["min_event_windows"]),
        "min_duration": float(min_duration),
        "min_ranker_support": params["min_ranker_support"],
        "min_seed_support": params["min_seed_support"],
        "max_technical_confound_score": float(params["max_technical_confound_score"]),
        "min_balance_pass_rate": params["min_balance_pass_rate"],
        "min_median_balance_score": params.get("min_median_balance_score"),
        "balance_logic": params.get("balance_logic", "and"),
        "min_event_balance_coverage": params.get("min_event_balance_coverage"),
        "require_bootstrap_support": bool(params["require_bootstrap_support"]),
    }
    return out


def apply_synthetic_gate_sweep(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    for mode in ("screening", "discovery", "candidate"):
        out = apply_synthetic_discovery_gate(out, mode=mode)
    return out


def summarize_synthetic_fpr_breakdown(
    table: pd.DataFrame,
    dimensions: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Summarize raw and gated synthetic false positives across benchmark axes.
    """
    if table is None or table.empty:
        return pd.DataFrame()
    dimensions = list(
        dimensions
        or [
            "truth_type",
            "ranker",
            "window_mode",
            "background_event_confidence_class",
            "event_stat",
            "null_model",
        ]
    )
    rows = []
    def numeric_col(group: pd.DataFrame, column: str) -> pd.Series:
        if column not in group:
            return pd.Series(dtype=float)
        return pd.to_numeric(group[column], errors="coerce")

    for dim in dimensions:
        if dim not in table.columns:
            continue
        for value, group in table.groupby(dim, dropna=False):
            raw_fp = numeric_col(group, "false_positive_rate")
            gated_fp = numeric_col(group, "gated_false_positive_rate")
            power = numeric_col(group, "power")
            gated_power = numeric_col(group, "gated_power")
            row = {
                    "dimension": dim,
                    "level": str(value),
                    "n_rows": int(len(group)),
                    "raw_false_positive_rate": float(raw_fp.mean())
                    if len(raw_fp)
                    else np.nan,
                    "gated_false_positive_rate": float(gated_fp.mean())
                    if len(gated_fp)
                    else np.nan,
                    "raw_false_positive_count": int((raw_fp > 0).sum())
                    if len(raw_fp)
                    else 0,
                    "gated_false_positive_count": int((gated_fp > 0).sum())
                    if len(gated_fp)
                    else 0,
                    "mean_power": float(power.mean()) if len(power) else np.nan,
                    "mean_gated_power": float(gated_power.mean())
                    if len(gated_power)
                    else np.nan,
            }
            for mode in ("screening", "candidate", "discovery"):
                mode_fp = numeric_col(group, f"{mode}_false_positive_rate")
                mode_power = numeric_col(group, f"{mode}_power")
                row[f"{mode}_false_positive_rate"] = (
                    float(mode_fp.mean()) if len(mode_fp) else np.nan
                )
                row[f"{mode}_false_positive_count"] = (
                    int((mode_fp > 0).sum()) if len(mode_fp) else 0
                )
                row[f"{mode}_power"] = (
                    float(mode_power.mean()) if len(mode_power) else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _trajectory_truth_row(
    truth_type: str,
    ranker: str,
    n_cells: int,
    n_genes: int,
    pathway_size: int,
    seed: int,
    n_perm: int,
    calibrate: bool,
    trajectory_kwargs: dict,
) -> dict:
    adata, gene_sets, truth = make_synthetic_trajectory_truth(
        truth_type,
        n_cells=n_cells,
        n_genes=n_genes,
        pathway_size=pathway_size,
        seed=seed,
    )
    adata = _add_replicates(adata.assign if False else adata, condition_key="condition", replicate_key="donor") if "condition" in adata.obs else adata
    res = run_trajectory_gsea(
        adata,
        gene_sets,
        pseudotime_key="dpt_pseudotime",
        ranker=ranker,
        **trajectory_kwargs,
    )
    events = summarize_events(res, min_consecutive=1)
    score = score_synthetic_events(events, truth)
    row = _event_row(events, "TRUE_SIGNAL")
    background = _event_row(events, "RANDOM_BACKGROUND")
    event_fdr = pd.DataFrame()
    if calibrate:
        if "donor" not in adata.obs:
            adata.obs["donor"] = np.where(np.arange(adata.n_obs) % 2 == 0, "D1", "D2")
        event_fdr = estimate_event_fdr(
            adata=adata,
            gmt_path=gene_sets,
            result=res,
            pseudotime_key="dpt_pseudotime",
            null="pseudotime_within_replicate_permutation",
            replicate_key="donor",
            n_perm=n_perm,
            event_stats=["max_abs_NES"],
            event_kwargs={"min_consecutive": 1},
            **trajectory_kwargs,
        )
    signal_q = _event_q_for_pathway(event_fdr, "TRUE_SIGNAL")
    background_q = _event_q_for_pathway(event_fdr, "RANDOM_BACKGROUND")
    signal_p = _event_p_for_pathway(event_fdr, "TRUE_SIGNAL")
    background_p = _event_p_for_pathway(event_fdr, "RANDOM_BACKGROUND")
    onset = row.get("activation_onset", np.nan)
    if not np.isfinite(pd.to_numeric(pd.Series([onset]), errors="coerce").iloc[0]):
        onset = row.get("suppression_onset", np.nan)
    expected_peak = _expected_peak(truth_type)
    duration_error = np.nan
    expected_duration = _expected_duration(truth_type)
    if np.isfinite(expected_duration) and "duration" in row:
        duration_error = abs(float(row["duration"]) - expected_duration)
    return {
        "truth_type": truth_type,
        "ranker": ranker,
        "window_mode": "matched_window",
        "event_stat": "max_abs_NES",
        "null_model": "pseudotime_within_replicate_permutation"
        if calibrate
        else "none",
        "power": float(np.isfinite(signal_q) and signal_q <= 0.05)
        if calibrate
        else score.get("detected", 0.0),
        "false_positive_rate": float(np.isfinite(background_q) and background_q <= 0.05)
        if calibrate
        else float(background.get("window_fdr_min", 1.0) <= 0.05)
        if not background.empty
        else 0.0,
        "event_label_accuracy": score.get("event_label_accuracy", np.nan),
        "peak_time_error": score.get("peak_time_error", np.nan),
        "onset_error": abs(float(onset) - expected_peak) if np.isfinite(pd.to_numeric(pd.Series([onset]), errors="coerce").iloc[0]) and np.isfinite(expected_peak) else np.nan,
        "AUC_error": np.nan,
        "duration_error": duration_error,
        "branch_assignment_accuracy": np.nan,
        "condition_delta_error": np.nan,
        "ranker_support": np.nan,
        "seed_support": 1.0,
        "technical_confound_score": 0.0,
        "event_q_calibration": float(not calibrate or not np.isfinite(background_q) or background_q > 0.05),
        "target_event_p": signal_p,
        "background_event_p": background_p,
        "target_event_q": signal_q,
        "background_event_q": background_q,
        **_event_fields(row, "target"),
        **_event_fields(background, "background"),
    }


def _condition_truth_row(
    truth_type: str,
    ranker: str,
    n_cells: int,
    n_genes: int,
    pathway_size: int,
    seed: int,
    n_perm: int,
    trajectory_kwargs: dict,
) -> dict:
    if truth_type == "condition_amplitude_loss":
        adata, gene_sets = _condition_amplitude_loss(
            n_cells, n_genes, pathway_size, effect_size=1.0, noise_sd=0.25, seed=seed
        )
        expected_delta_peak = 0.0
        expected_delta_auc_sign = -1
    else:
        adata, gene_sets, _truth = make_synthetic_trajectory_truth(
            "condition_delayed_activation",
            n_cells=n_cells,
            n_genes=n_genes,
            pathway_size=pathway_size,
            seed=seed,
        )
        expected_delta_peak = 0.2
        expected_delta_auc_sign = 0
    adata = _add_replicates(
        adata,
        condition_key="condition",
        replicate_key="donor",
        imbalanced=truth_type == "replicate_imbalance",
    )
    matched_kwargs = dict(trajectory_kwargs)
    matched_kwargs.setdefault("balance", "weights")
    matched_kwargs.setdefault("balance_smd_threshold", 0.25)
    cmp_df = compare_trajectory_gsea(
        adata,
        gene_sets,
        condition_key="condition",
        mode="matched_window",
        control="control",
        case="case",
        pseudotime_key="dpt_pseudotime",
        ranker=ranker,
        n_permutations=0,
        event_kwargs={"min_consecutive": 1},
        **matched_kwargs,
    )
    signal = cmp_df[cmp_df["Pathway"] == "TRUE_SIGNAL"] if not cmp_df.empty else pd.DataFrame()
    background = cmp_df[cmp_df["Pathway"] == "RANDOM_BACKGROUND"] if not cmp_df.empty else pd.DataFrame()
    if signal.empty:
        delta_peak = np.nan
        delta_auc = np.nan
    else:
        delta_peak = float(signal.iloc[0].get("delta_peak_time", np.nan))
        delta_auc = float(signal.iloc[0].get("delta_AUC", np.nan))
    if truth_type == "condition_amplitude_loss":
        detected = float(np.isfinite(delta_auc) and np.sign(delta_auc) == expected_delta_auc_sign)
        condition_error = abs(delta_auc) if np.isfinite(delta_auc) else np.nan
    else:
        detected = float(np.isfinite(delta_peak) and delta_peak > 0)
        condition_error = abs(delta_peak - expected_delta_peak) if np.isfinite(delta_peak) else np.nan
    fp = float(
        not background.empty
        and str(background.iloc[0].get("program_type", "")).endswith("program")
        and abs(float(background.iloc[0].get("delta_AUC", 0.0))) > abs(delta_auc if np.isfinite(delta_auc) else 0.0)
    )
    event_fdr = estimate_event_fdr(
        adata=adata,
        gmt_path=gene_sets,
        pseudotime_key="dpt_pseudotime",
        event_stats=["delta_AUC", "delta_peak_time"],
        null="condition_label_permutation_within_pseudotime_bins",
        condition_key="condition",
        control="control",
        case="case",
        n_perm=n_perm,
        seed=seed,
        ranker=ranker,
        mode="matched_window",
        event_kwargs={"min_consecutive": 1},
        **matched_kwargs,
    )
    target_q = _event_q_for_pathway(event_fdr, "TRUE_SIGNAL", stat="delta_AUC")
    background_q = _event_q_for_pathway(event_fdr, "RANDOM_BACKGROUND", stat="delta_AUC")
    target_p = _event_p_for_pathway(event_fdr, "TRUE_SIGNAL", stat="delta_AUC")
    background_p = _event_p_for_pathway(event_fdr, "RANDOM_BACKGROUND", stat="delta_AUC")
    if np.isfinite(background_q):
        fp = float(background_q <= 0.05)
    target_event_balance_coverage = (
        float(signal.iloc[0].get("event_balance_coverage", np.nan))
        if not signal.empty
        else np.nan
    )
    background_event_balance_coverage = (
        float(background.iloc[0].get("event_balance_coverage", np.nan))
        if not background.empty
        else np.nan
    )
    target_event_median_balance_score = (
        float(signal.iloc[0].get("event_median_balance_score", np.nan))
        if not signal.empty
        else np.nan
    )
    background_event_median_balance_score = (
        float(background.iloc[0].get("event_median_balance_score", np.nan))
        if not background.empty
        else np.nan
    )
    target_event_core_balance_coverage = (
        float(signal.iloc[0].get("event_core_balance_coverage", np.nan))
        if not signal.empty
        else np.nan
    )
    background_event_core_balance_coverage = (
        float(background.iloc[0].get("event_core_balance_coverage", np.nan))
        if not background.empty
        else np.nan
    )
    target_event_median_core_balance_score = (
        float(signal.iloc[0].get("event_median_core_balance_score", np.nan))
        if not signal.empty
        else np.nan
    )
    background_event_median_core_balance_score = (
        float(background.iloc[0].get("event_median_core_balance_score", np.nan))
        if not background.empty
        else np.nan
    )
    target_n_counts_sensitivity_flag = (
        str(signal.iloc[0].get("n_counts_sensitivity_flag", "not_comparable"))
        if not signal.empty
        else "not_comparable"
    )
    background_n_counts_sensitivity_flag = (
        str(background.iloc[0].get("n_counts_sensitivity_flag", "not_comparable"))
        if not background.empty
        else "not_comparable"
    )
    return {
        "truth_type": truth_type,
        "ranker": ranker,
        "window_mode": trajectory_kwargs.get("window_mode", "cell_count"),
        "event_stat": "delta_AUC_abs"
        if truth_type == "condition_amplitude_loss"
        else "delta_peak_time",
        "null_model": "condition_label_permutation_within_pseudotime_bins",
        "power": detected,
        "false_positive_rate": fp,
        "event_label_accuracy": detected,
        "peak_time_error": np.nan,
        "onset_error": np.nan,
        "AUC_error": np.nan,
        "duration_error": np.nan,
        "branch_assignment_accuracy": np.nan,
        "condition_delta_error": condition_error,
        "ranker_support": np.nan,
        "seed_support": 1.0,
        "event_q_calibration": float(not np.isfinite(background_q) or background_q > 0.05),
        "target_event_p": target_p,
        "background_event_p": background_p,
        "target_event_q": target_q,
        "background_event_q": background_q,
        "target_event_confidence_class": "condition_comparison",
        "background_event_confidence_class": "condition_comparison"
        if not background.empty
        else "missing",
        "target_event_windows": float(signal.iloc[0].get("matched_event_windows", 0.0)) if not signal.empty else 0.0,
        "background_event_windows": float(background.iloc[0].get("matched_event_windows", 0.0)) if not background.empty else 0.0,
        "target_duration": np.nan,
        "background_duration": np.nan,
        "balance_pass_rate": float(cmp_df.get("balance_pass_rate", pd.Series([np.nan])).min()) if not cmp_df.empty else np.nan,
        "median_balance_score": float(cmp_df.get("median_balance_score", pd.Series([np.nan])).median()) if not cmp_df.empty else np.nan,
        "core_balance_pass_rate": float(cmp_df.get("core_balance_pass_rate", pd.Series([np.nan])).min()) if not cmp_df.empty else np.nan,
        "median_core_balance_score": float(cmp_df.get("median_core_balance_score", pd.Series([np.nan])).median()) if not cmp_df.empty else np.nan,
        "target_event_balance_coverage": target_event_balance_coverage,
        "background_event_balance_coverage": background_event_balance_coverage,
        "target_event_median_balance_score": target_event_median_balance_score,
        "background_event_median_balance_score": background_event_median_balance_score,
        "target_event_core_balance_coverage": target_event_core_balance_coverage,
        "background_event_core_balance_coverage": background_event_core_balance_coverage,
        "target_event_median_core_balance_score": target_event_median_core_balance_score,
        "background_event_median_core_balance_score": background_event_median_core_balance_score,
        "target_n_counts_sensitivity_flag": target_n_counts_sensitivity_flag,
        "background_n_counts_sensitivity_flag": background_n_counts_sensitivity_flag,
        "technical_confound_score": float(cmp_df.get("max_abs_detection_rate_smd_after", pd.Series([0.0])).max()) if not cmp_df.empty else 0.0,
    }


def _branch_truth_row(
    ranker: str,
    n_cells: int,
    n_genes: int,
    pathway_size: int,
    seed: int,
    n_perm: int,
    trajectory_kwargs: dict,
) -> dict:
    adata, gene_sets, _truth = make_synthetic_trajectory_truth(
        "branch_specific_activation",
        n_cells=n_cells,
        n_genes=n_genes,
        pathway_size=pathway_size,
        seed=seed,
    )
    branch = run_branch_gsea(
        adata,
        gene_sets,
        branch_key="branch",
        mode="branch_contrast",
        pseudotime_key="dpt_pseudotime",
        ranker=ranker,
        event_kwargs={"min_consecutive": 1},
        **trajectory_kwargs,
    )
    cmp_df = branch["comparisons"]
    signal = cmp_df[cmp_df["Pathway"] == "TRUE_SIGNAL"] if not cmp_df.empty else pd.DataFrame()
    background = cmp_df[cmp_df["Pathway"] == "RANDOM_BACKGROUND"] if not cmp_df.empty else pd.DataFrame()
    accuracy = float(
        not signal.empty
        and str(signal.iloc[0].get("program_type", "")) in {"divergence_program", "specific_program"}
    )
    fp = float(
        not background.empty
        and str(background.iloc[0].get("program_type", "")) in {"divergence_program", "specific_program"}
    )
    event_fdr = estimate_event_fdr(
        adata=adata,
        gmt_path=gene_sets,
        pseudotime_key="dpt_pseudotime",
        event_stats=["delta_AUC", "delta_peak_time"],
        null="branch_label_permutation_within_pseudotime_bins",
        branch_key="branch",
        branch_a="branch_a",
        branch_b="branch_b",
        n_perm=n_perm,
        seed=seed,
        ranker=ranker,
        event_kwargs={"min_consecutive": 1},
        **trajectory_kwargs,
    )
    signal_q = _event_q_for_pathway(event_fdr, "TRUE_SIGNAL", stat="delta_AUC")
    background_q = _event_q_for_pathway(event_fdr, "RANDOM_BACKGROUND", stat="delta_AUC")
    signal_p = _event_p_for_pathway(event_fdr, "TRUE_SIGNAL", stat="delta_AUC")
    background_p = _event_p_for_pathway(event_fdr, "RANDOM_BACKGROUND", stat="delta_AUC")
    if np.isfinite(background_q):
        fp = float(background_q <= 0.05)
    return {
        "truth_type": "branch_specific_activation",
        "ranker": ranker,
        "window_mode": trajectory_kwargs.get("window_mode", "cell_count"),
        "event_stat": "delta_AUC_abs",
        "null_model": "branch_label_permutation_within_pseudotime_bins",
        "power": accuracy,
        "false_positive_rate": fp,
        "event_label_accuracy": accuracy,
        "peak_time_error": np.nan,
        "onset_error": np.nan,
        "AUC_error": np.nan,
        "duration_error": np.nan,
        "branch_assignment_accuracy": accuracy,
        "condition_delta_error": np.nan,
        "ranker_support": np.nan,
        "seed_support": 1.0,
        "technical_confound_score": 0.0,
        "event_q_calibration": float(not np.isfinite(background_q) or background_q > 0.05),
        "target_event_p": signal_p,
        "background_event_p": background_p,
        "target_event_q": signal_q,
        "background_event_q": background_q,
        "target_event_confidence_class": "branch_contrast",
        "background_event_confidence_class": "branch_contrast"
        if not background.empty
        else "missing",
        "target_event_windows": 2.0 if not signal.empty else 0.0,
        "background_event_windows": 2.0 if not background.empty else 0.0,
        "target_duration": np.nan,
        "background_duration": np.nan,
    }


def _graph_mixing_row(
    ranker: str,
    n_cells: int,
    n_genes: int,
    pathway_size: int,
    seed: int,
    trajectory_kwargs: dict,
) -> dict:
    adata, gene_sets = _graph_branch_mixing(
        n_cells, n_genes, pathway_size, effect_size=1.0, noise_sd=0.25, seed=seed
    )
    kwargs = dict(trajectory_kwargs)
    kwargs.pop("window_size", None)
    kwargs.pop("step", None)
    res = run_trajectory_gsea(
        adata,
        gene_sets,
        pseudotime_key="dpt_pseudotime",
        ranker=ranker,
        window_mode="graph_adaptive",
        graph_key="connectivities",
        graph_radius=3,
        target_span=0.15,
        span_step=0.15,
        min_cells=5,
        max_cells=max(10, n_cells // 4),
        branch_key="branch",
        min_branch_purity=0.95,
        cell_weight_key="fate_prob",
        experimental=True,
        **kwargs,
    )
    diag = res.attrs.get("graph_window_diagnostics", pd.DataFrame())
    low_purity = float(
        not diag.empty and (diag.get("skip_reason", pd.Series(dtype=str)) == "low_branch_purity").any()
    )
    return {
        "truth_type": "graph_branch_mixing",
        "ranker": ranker,
        "window_mode": "graph_adaptive",
        "event_stat": "branch_purity",
        "null_model": "graph_purity_diagnostic",
        "power": low_purity,
        "false_positive_rate": 0.0,
        "event_label_accuracy": low_purity,
        "peak_time_error": np.nan,
        "onset_error": np.nan,
        "AUC_error": np.nan,
        "duration_error": np.nan,
        "branch_assignment_accuracy": low_purity,
        "condition_delta_error": np.nan,
        "ranker_support": np.nan,
        "seed_support": 1.0,
        "technical_confound_score": 0.0,
        "event_q_calibration": np.nan,
        "target_event_q": np.nan,
        "background_event_q": np.nan,
        "target_event_confidence_class": "graph_purity_diagnostic",
        "background_event_confidence_class": "graph_purity_diagnostic",
        "target_event_windows": 0.0,
        "background_event_windows": 0.0,
        "target_duration": np.nan,
        "background_duration": np.nan,
    }


def run_reliability_synthetic_truth_benchmark(
    truth_types: Optional[Iterable[str]] = None,
    rankers: Optional[Iterable[str]] = None,
    n_cells: int = 160,
    n_genes: int = 80,
    pathway_size: int = 8,
    seed: int = 0,
    n_perm: int = 5,
    calibrate: bool = True,
    **trajectory_kwargs,
) -> pd.DataFrame:
    """
    Evaluate TED on truth-labelled synthetic pathway dynamics.

    The table is designed to answer whether TED avoids false discoveries while
    still detecting known pathway events.
    """
    truth_types = list(truth_types or RELIABILITY_TRUTH_TYPES)
    rankers = list(rankers or ["mean_diff", "detection_weighted"])
    defaults = {
        "window_size": max(20, n_cells // 5),
        "step": max(10, n_cells // 10),
        "min_size": max(5, pathway_size // 2),
        "max_size": 500,
        "nperm_nes": 8,
        "sample_size": min(8, pathway_size),
        "bin_width": None,
    }
    defaults.update(trajectory_kwargs)
    rows = []
    for truth_idx, truth_type in enumerate(truth_types):
        for ranker in rankers:
            run_seed = seed + truth_idx * 101
            if truth_type in TRUTH_TYPES and truth_type not in {
                "branch_specific_activation",
                "condition_delayed_activation",
            }:
                rows.append(
                    _trajectory_truth_row(
                        truth_type,
                        ranker,
                        n_cells,
                        n_genes,
                        pathway_size,
                        run_seed,
                        n_perm,
                        calibrate,
                        defaults,
                    )
                )
            elif truth_type == "branch_specific_activation":
                rows.append(
                    _branch_truth_row(
                        ranker, n_cells, n_genes, pathway_size, run_seed, n_perm, defaults
                    )
                )
            elif truth_type in {
                "condition_delayed_activation",
                "condition_amplitude_loss",
                "replicate_imbalance",
            }:
                rows.append(
                    _condition_truth_row(
                        truth_type,
                        ranker,
                        n_cells,
                        n_genes,
                        pathway_size,
                        run_seed,
                        n_perm,
                        defaults,
                    )
                )
            elif truth_type == "graph_branch_mixing":
                rows.append(
                    _graph_mixing_row(
                        ranker, n_cells, n_genes, pathway_size, run_seed, defaults
                    )
                )
            else:
                raise ValueError(f"Unsupported reliability truth_type '{truth_type}'")
    out = pd.DataFrame(rows)
    if not out.empty:
        out["ranker_support"] = out.groupby("truth_type")["power"].transform("mean")
        out = apply_synthetic_gate_sweep(out)
    return out


def _null_event_rate(table: pd.DataFrame, q_threshold: float) -> dict:
    if table is None or table.empty:
        return {
            "robust_event_count": 0,
            "q_lt_005_rate": 0.0,
            "median_event_p": np.nan,
            "min_event_q": np.nan,
            "event_q_calibration": 1.0,
        }
    q = pd.to_numeric(table["event_q"], errors="coerce") if "event_q" in table else pd.Series(dtype=float)
    p = pd.to_numeric(table["event_p"], errors="coerce") if "event_p" in table else pd.Series(dtype=float)
    rate = float((q <= q_threshold).mean()) if len(q) else 0.0
    return {
        "robust_event_count": int((q <= q_threshold).sum()) if len(q) else 0,
        "q_lt_005_rate": rate,
        "median_event_p": float(np.nanmedian(p)) if len(p) else np.nan,
        "min_event_q": float(np.nanmin(q)) if len(q) and np.isfinite(q).any() else np.nan,
        "event_q_calibration": float(rate <= max(q_threshold * 2.0, 0.1)),
    }


def run_null_calibration_benchmark(
    nulls: Optional[Iterable[str]] = None,
    n_perm_values: Optional[Iterable[int]] = None,
    n_cells: int = 120,
    n_genes: int = 80,
    pathway_size: int = 8,
    seed: int = 0,
    q_threshold: float = 0.05,
    **trajectory_kwargs,
) -> pd.DataFrame:
    """
    Run compact null benchmarks for pseudotime, condition, and branch nulls.
    """
    nulls = list(nulls or ["pseudotime", "condition", "branch"])
    n_perm_values = list(n_perm_values or [50, 100, 200])
    defaults = {
        "window_size": max(20, n_cells // 5),
        "step": max(10, n_cells // 10),
        "min_size": max(5, pathway_size // 2),
        "max_size": 500,
        "nperm_nes": 8,
        "sample_size": min(8, pathway_size),
        "bin_width": None,
    }
    defaults.update(trajectory_kwargs)
    rng = np.random.default_rng(seed)
    rows = []
    for null_name in nulls:
        for n_perm in n_perm_values:
            if null_name == "pseudotime":
                adata, gene_sets, _truth = make_synthetic_trajectory_truth(
                    "monotonic_up",
                    n_cells=n_cells,
                    n_genes=n_genes,
                    pathway_size=pathway_size,
                    seed=seed,
                )
                adata.obs["dpt_pseudotime"] = rng.permutation(adata.obs["dpt_pseudotime"].to_numpy())
                adata.obs["donor"] = np.where(np.arange(adata.n_obs) % 2 == 0, "D1", "D2")
                table = estimate_event_fdr(
                    adata=adata,
                    gmt_path=gene_sets,
                    pseudotime_key="dpt_pseudotime",
                    null="pseudotime_within_replicate_permutation",
                    replicate_key="donor",
                    n_perm=n_perm,
                    event_stats=["max_abs_NES"],
                    event_kwargs={"min_consecutive": 1},
                    **defaults,
                )
                null_model = "pseudotime_within_replicate_permutation"
            elif null_name == "condition":
                adata, gene_sets, _truth = make_synthetic_trajectory_truth(
                    "monotonic_up",
                    n_cells=n_cells,
                    n_genes=n_genes,
                    pathway_size=pathway_size,
                    seed=seed + 1,
                )
                adata.obs["condition"] = np.where(np.arange(adata.n_obs) % 2 == 0, "control", "case")
                adata = _add_replicates(adata, condition_key="condition", replicate_key="donor")
                table = estimate_event_fdr(
                    adata=adata,
                    gmt_path=gene_sets,
                    pseudotime_key="dpt_pseudotime",
                    null="condition_label_permutation_by_replicate",
                    condition_key="condition",
                    replicate_key="donor",
                    control="control",
                    case="case",
                    n_perm=n_perm,
                    event_stats=["delta_AUC", "delta_peak_time"],
                    event_kwargs={"min_consecutive": 1},
                    min_cells_per_replicate=1,
                    min_replicates_per_condition=2,
                    **defaults,
                )
                null_model = "condition_label_permutation_by_replicate"
            elif null_name == "branch":
                adata, gene_sets, _truth = make_synthetic_trajectory_truth(
                    "monotonic_up",
                    n_cells=n_cells,
                    n_genes=n_genes,
                    pathway_size=pathway_size,
                    seed=seed + 2,
                )
                adata.obs["branch"] = np.where(np.arange(adata.n_obs) % 2 == 0, "branch_a", "branch_b")
                table = estimate_event_fdr(
                    adata=adata,
                    gmt_path=gene_sets,
                    pseudotime_key="dpt_pseudotime",
                    null="branch_label_permutation_within_pseudotime_bins",
                    branch_key="branch",
                    branch_a="branch_a",
                    branch_b="branch_b",
                    n_pseudotime_bins=4,
                    n_perm=n_perm,
                    event_stats=["delta_AUC", "delta_peak_time"],
                    event_kwargs={"min_consecutive": 1},
                    **defaults,
                )
                null_model = "branch_label_permutation_within_pseudotime_bins"
            else:
                raise ValueError(f"Unsupported null benchmark '{null_name}'")
            metrics = _null_event_rate(table, q_threshold=q_threshold)
            random_sets = int(
                table["pathway"].astype(str).str.contains("RANDOM", case=False).sum()
            ) if not table.empty and "pathway" in table else 0
            rows.append(
                {
                    "null": null_name,
                    "null_model": null_model,
                    "n_perm": n_perm,
                    "minimum_attainable_p": 1.0 / (n_perm + 1.0),
                    "random_gene_set_event_rows": random_sets,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def run_reliability_ablation_study(
    n_cells: int = 120,
    n_genes: int = 80,
    pathway_size: int = 8,
    seed: int = 0,
    n_perm: int = 5,
    **trajectory_kwargs,
) -> pd.DataFrame:
    """
    Produce a compact ablation table linking modules to reduced failure modes.
    """
    defaults = {
        "window_size": max(20, n_cells // 5),
        "step": max(10, n_cells // 10),
        "min_size": max(5, pathway_size // 2),
        "max_size": 500,
        "nperm_nes": 8,
        "sample_size": min(8, pathway_size),
        "bin_width": None,
    }
    defaults.update(trajectory_kwargs)

    sparse = run_reliability_synthetic_truth_benchmark(
        truth_types=["sparse_dropout_burst"],
        rankers=["mean_diff", "detection_weighted"],
        n_cells=n_cells,
        n_genes=n_genes,
        pathway_size=pathway_size,
        seed=seed,
        n_perm=n_perm,
        calibrate=True,
        **defaults,
    )
    mean_fp = float(sparse.loc[sparse["ranker"] == "mean_diff", "false_positive_rate"].mean())
    det_fp = float(sparse.loc[sparse["ranker"] == "detection_weighted", "false_positive_rate"].mean())

    null = run_null_calibration_benchmark(
        nulls=["pseudotime"],
        n_perm_values=[n_perm],
        n_cells=n_cells,
        n_genes=n_genes,
        pathway_size=pathway_size,
        seed=seed + 10,
        **defaults,
    )

    graph = run_reliability_synthetic_truth_benchmark(
        truth_types=["graph_branch_mixing"],
        rankers=["mean_diff"],
        n_cells=n_cells,
        n_genes=n_genes,
        pathway_size=pathway_size,
        seed=seed + 20,
        calibrate=False,
        **defaults,
    )

    rep = run_reliability_synthetic_truth_benchmark(
        truth_types=["replicate_imbalance"],
        rankers=["detection_weighted"],
        n_cells=n_cells,
        n_genes=n_genes,
        pathway_size=pathway_size,
        seed=seed + 30,
        n_perm=n_perm,
        calibrate=False,
        **defaults,
    )

    rows = [
        {
            "Module added": "detection_weighted",
            "Failure mode reduced": "sparse false peak",
            "Evidence": f"background false-positive rate mean_diff={mean_fp:.3f}, detection_weighted={det_fp:.3f}",
            "metric": det_fp - mean_fp,
        },
        {
            "Module added": "event_fdr",
            "Failure mode reduced": "trajectory-wide overcalling",
            "Evidence": f"null event_q<0.05 rate={float(null['q_lt_005_rate'].iloc[0]):.3f}",
            "metric": float(null["q_lt_005_rate"].iloc[0]),
        },
        {
            "Module added": "graph_adaptive",
            "Failure mode reduced": "branch mixing",
            "Evidence": f"low branch-purity warning detected={float(graph['branch_assignment_accuracy'].iloc[0]):.3f}",
            "metric": float(graph["branch_assignment_accuracy"].iloc[0]),
        },
        {
            "Module added": "replicate-aware",
            "Failure mode reduced": "donor imbalance",
            "Evidence": f"replicate-aware condition delta error={float(rep['condition_delta_error'].iloc[0]):.3f}",
            "metric": float(rep["condition_delta_error"].iloc[0]) if np.isfinite(float(rep["condition_delta_error"].iloc[0])) else np.nan,
        },
        {
            "Module added": "consensus",
            "Failure mode reduced": "parameter instability",
            "Evidence": "ranker_support reports the fraction of rankers recovering each truth event",
            "metric": float(sparse["ranker_support"].mean()),
        },
    ]
    return pd.DataFrame(rows)
