"""Small-block inference utilities for the TED BNT162b2 flagship workflow.

The functions in this module are intentionally data-agnostic.  Dataset adapters
produce donor-by-time score tables; this module applies the frozen transient
contrast, exhaustive sign-flip/maxT inference, leave-one-donor checks, and
entropy-balancing diagnostics without touching the masked ADT outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import minimize


TRANSIENT_WEIGHTS = {0: -0.50, 2: 1.00, 10: -0.25, 28: -0.25}
ACTIVATION_WEIGHTS = {0: -1.00, 2: 1.00, 10: 0.00, 28: 0.00}
RECOVERY_WEIGHTS = {0: 0.00, 2: 1.00, 10: -0.50, 28: -0.50}


def _ordered_time_table(
    table: pd.DataFrame,
    *,
    donor_col: str = "donor_id",
    time_col: str = "day",
    value_col: str = "score",
    required_days: tuple[int, ...] = (0, 2, 10, 28),
) -> pd.DataFrame:
    required = {donor_col, time_col, value_col}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing donor-time columns: {sorted(missing)}")
    work = table[[donor_col, time_col, value_col]].copy()
    work[time_col] = pd.to_numeric(work[time_col], errors="raise").astype(int)
    if work.duplicated([donor_col, time_col]).any():
        raise ValueError("Each donor-time combination must occur exactly once")
    wide = work.pivot(index=donor_col, columns=time_col, values=value_col)
    absent = [day for day in required_days if day not in wide.columns]
    if absent:
        raise ValueError(f"Missing required timepoints: {absent}")
    return wide[list(required_days)].dropna()


def contrast_from_wide(wide: pd.DataFrame, weights: dict[int, float]) -> pd.Series:
    missing = set(weights).difference(wide.columns)
    if missing:
        raise ValueError(f"Missing contrast timepoints: {sorted(missing)}")
    vector = pd.Series(weights, dtype=float)
    return wide.loc[:, vector.index].dot(vector)


def transient_contrasts(
    table: pd.DataFrame,
    *,
    donor_col: str = "donor_id",
    time_col: str = "day",
    value_col: str = "score",
    transient_weights: dict[int, float] | None = None,
    activation_weights: dict[int, float] | None = None,
    recovery_weights: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Return donor-specific activation, recovery, and transient contrasts.

    Optional weights support a separately frozen replication time map without
    changing the primary flagship defaults. All three contrasts must reference
    the same ordered set of timepoints.
    """

    transient_weights = TRANSIENT_WEIGHTS if transient_weights is None else transient_weights
    activation_weights = ACTIVATION_WEIGHTS if activation_weights is None else activation_weights
    recovery_weights = RECOVERY_WEIGHTS if recovery_weights is None else recovery_weights
    required_days = tuple(transient_weights)
    if tuple(activation_weights) != required_days or tuple(recovery_weights) != required_days:
        raise ValueError("All contrast weights must use the same ordered timepoints")

    wide = _ordered_time_table(
        table,
        donor_col=donor_col,
        time_col=time_col,
        value_col=value_col,
        required_days=required_days,
    )
    return pd.DataFrame(
        {
            "activation": contrast_from_wide(wide, activation_weights),
            "recovery": contrast_from_wide(wide, recovery_weights),
            "transient": contrast_from_wide(wide, transient_weights),
        }
    )


def _standardized_sign_stat(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        return np.nan
    scale = float(np.sqrt(np.mean(values**2)))
    if scale <= np.finfo(float).eps:
        return 0.0
    return float(np.mean(values) / scale)


def exhaustive_sign_flip_max_t(effects: pd.DataFrame) -> pd.DataFrame:
    """Compute exact one-sided raw and single-step maxT p-values.

    Rows are exchangeable biological blocks and columns are the frozen pathway
    family.  The test statistic is mean(effect)/RMS(effect); its denominator is
    invariant to sign flips and keeps differently scaled pathway scores on a
    common bounded scale.  All ``2**n`` configurations, including the observed
    one, are enumerated, so no Monte Carlo correction is needed.
    """

    if effects.empty:
        raise ValueError("At least one donor and one pathway are required")
    matrix = effects.to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Effects must be finite")
    n_blocks, n_events = matrix.shape
    signs = np.asarray(list(product((-1.0, 1.0), repeat=n_blocks)), dtype=float)
    scales = np.sqrt(np.mean(matrix**2, axis=0))
    safe_scales = np.where(scales <= np.finfo(float).eps, 1.0, scales)
    null_t = (signs @ matrix) / n_blocks / safe_scales
    null_t[:, scales <= np.finfo(float).eps] = 0.0
    observed = np.asarray([_standardized_sign_stat(matrix[:, j]) for j in range(n_events)])
    max_null = np.max(null_t, axis=1)
    raw_p = np.mean(null_t >= observed[None, :], axis=0)
    adjusted_p = np.mean(max_null[:, None] >= observed[None, :], axis=0)
    return pd.DataFrame(
        {
            "pathway": effects.columns,
            "n_blocks": n_blocks,
            "n_sign_configurations": len(signs),
            "mean_effect": np.mean(matrix, axis=0),
            "standardized_sign_statistic": observed,
            "exact_raw_p": raw_p,
            "exact_maxT_p": adjusted_p,
        }
    )


def leave_one_donor_retention(
    transient: pd.DataFrame,
    activation: pd.DataFrame,
    recovery: pd.DataFrame,
    *,
    alpha: float = 0.10,
    direction_min: float = 0.80,
) -> pd.DataFrame:
    """Re-run family maxT and mode gates after leaving out each donor."""

    if not transient.index.equals(activation.index) or not transient.index.equals(recovery.index):
        raise ValueError("Transient, activation, and recovery tables must share donor rows")
    if list(transient.columns) != list(activation.columns) or list(transient.columns) != list(recovery.columns):
        raise ValueError("All contrast tables must share pathway columns")
    rows: list[dict[str, object]] = []
    for held_out in transient.index:
        keep = transient.index != held_out
        infer = exhaustive_sign_flip_max_t(transient.loc[keep]).set_index("pathway")
        for pathway in transient.columns:
            d = transient.loc[keep, pathway].to_numpy(dtype=float)
            p = activation.loc[keep, pathway].to_numpy(dtype=float)
            r = recovery.loc[keep, pathway].to_numpy(dtype=float)
            direction = float(np.mean(d > 0))
            activation_direction = float(np.mean(p > 0))
            recovery_direction = float(np.mean(r > 0))
            mode_pass = bool(
                np.mean(p) > 0
                and np.mean(r) > 0
                and activation_direction >= direction_min
                and recovery_direction >= direction_min
            )
            rows.append(
                {
                    "held_out_donor": held_out,
                    "pathway": pathway,
                    "n_retained_donors": int(np.sum(keep)),
                    "exact_maxT_p": float(infer.loc[pathway, "exact_maxT_p"]),
                    "direction_stability": direction,
                    "activation_direction_stability": activation_direction,
                    "recovery_direction_stability": recovery_direction,
                    "mode_pass": mode_pass,
                    "selection_retained": bool(
                        infer.loc[pathway, "exact_maxT_p"] <= alpha
                        and direction >= direction_min
                        and mode_pass
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result["retention_fraction"] = result.groupby("pathway")["selection_retained"].transform("mean")
    return result


@dataclass(frozen=True)
class BalanceDiagnostics:
    converged: bool
    max_abs_smd: float
    effective_sample_size: float
    max_weight_ratio_to_uniform: float
    objective: float


def entropy_balance(
    covariates: pd.DataFrame,
    target_mean: pd.Series | np.ndarray,
    *,
    max_iter: int = 1_000,
) -> tuple[pd.Series, BalanceDiagnostics]:
    """Exponential-tilting weights that target a declared covariate mean.

    Covariates are standardized internally using the unweighted group standard
    deviation.  The returned weights sum to one.  No clipping is performed:
    extreme weights remain visible to the effective-sample-size and weight-ratio
    gates instead of being silently repaired.
    """

    if covariates.empty:
        raise ValueError("Entropy balancing requires at least one covariate")
    x_frame = covariates.apply(pd.to_numeric, errors="coerce")
    if x_frame.isna().any().any():
        raise ValueError("Entropy-balancing covariates must be finite")
    x = x_frame.to_numpy(dtype=float)
    target = np.asarray(target_mean, dtype=float)
    if target.shape != (x.shape[1],) or not np.isfinite(target).all():
        raise ValueError("Target mean must contain one finite value per covariate")
    center = np.mean(x, axis=0)
    scale = np.std(x, axis=0, ddof=0)
    keep = scale > np.finfo(float).eps
    if not np.any(keep):
        weights = np.repeat(1.0 / len(x), len(x))
        diagnostics = BalanceDiagnostics(True, 0.0, float(len(x)), 1.0, 0.0)
        return pd.Series(weights, index=covariates.index, name="weight"), diagnostics
    z = (x[:, keep] - center[keep]) / scale[keep]
    target_z = (target[keep] - center[keep]) / scale[keep]

    def weights_for(lam: np.ndarray) -> np.ndarray:
        eta = z @ lam
        eta -= np.max(eta)
        raw = np.exp(eta)
        return raw / np.sum(raw)

    def objective(lam: np.ndarray) -> tuple[float, np.ndarray]:
        eta = z @ lam
        max_eta = float(np.max(eta))
        log_mean_exp = max_eta + np.log(np.mean(np.exp(eta - max_eta)))
        w = weights_for(lam)
        value = float(log_mean_exp - np.dot(lam, target_z))
        gradient = z.T @ w - target_z
        return value, gradient

    fit = minimize(
        lambda lam: objective(lam)[0],
        np.zeros(z.shape[1], dtype=float),
        jac=lambda lam: objective(lam)[1],
        method="BFGS",
        options={"maxiter": max_iter, "gtol": 1e-10},
    )
    weights = weights_for(np.asarray(fit.x, dtype=float))
    weighted_mean = weights @ x
    smd_scale = np.std(x, axis=0, ddof=0)
    smd = np.zeros(x.shape[1], dtype=float)
    variable = smd_scale > np.finfo(float).eps
    smd[variable] = np.abs(weighted_mean[variable] - target[variable]) / smd_scale[variable]
    diagnostics = BalanceDiagnostics(
        converged=bool(fit.success or np.max(smd) <= 1e-6),
        max_abs_smd=float(np.max(smd)),
        effective_sample_size=float(1.0 / np.sum(weights**2)),
        max_weight_ratio_to_uniform=float(np.max(weights) * len(weights)),
        objective=float(fit.fun),
    )
    return pd.Series(weights, index=covariates.index, name="weight"), diagnostics


def peak_day(wide: pd.DataFrame, days: tuple[int, ...] = (0, 2, 10, 28)) -> int:
    """Return the day with the largest aggregate activity."""

    required = list(days)
    if any(day not in wide.columns for day in required):
        raise ValueError(f"Peak-day evaluation requires days {required}")
    return int(wide[required].mean(axis=0).idxmax())
