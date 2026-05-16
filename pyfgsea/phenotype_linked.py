from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


_PHENOTYPE_OUTPUT_FILENAMES = {
    "event_burden_score": "event_burden_score.tsv",
    "phenotype_event_association": "phenotype_event_association.tsv",
    "phenotype_prediction_performance": "phenotype_prediction_performance.tsv",
    "phenotype_linked_event_report": "phenotype_linked_event_report.tsv",
}


def _pick_column(df: pd.DataFrame, *candidates: Optional[str]) -> Optional[str]:
    lower_to_original = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate is None:
            continue
        if candidate in df.columns:
            return candidate
        lower = str(candidate).lower()
        if lower in lower_to_original:
            return lower_to_original[lower]
    return None


def _safe_float(value: object, default: float = np.nan) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(out) if np.isfinite(out) else float(default)


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


def _trapz(values: np.ndarray, x: np.ndarray) -> float:
    if len(values) == 0:
        return np.nan
    if len(values) == 1:
        return float(values[0])
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, x=x))
    return float(np.trapz(values, x=x))


def _event_label(row: pd.Series, label_col: Optional[str]) -> str:
    if label_col is not None and label_col in row.index and pd.notna(row[label_col]):
        return str(row[label_col])
    for col in ("pathway", "Pathway", "module", "event"):
        if col in row.index and pd.notna(row[col]):
            return str(row[col])
    return ""


def _event_interval(row: pd.Series) -> tuple[float, float, float]:
    peak = _safe_float(row.get("peak_time", row.get("event_peak", np.nan)))
    starts = [
        _safe_float(row.get("event_onset", np.nan)),
        _safe_float(row.get("activation_onset", np.nan)),
        _safe_float(row.get("suppression_onset", np.nan)),
        _safe_float(row.get("pt_start", np.nan)),
    ]
    start = next((value for value in starts if np.isfinite(value)), peak)
    duration = abs(_safe_float(row.get("duration", np.nan)))
    end = start + duration if np.isfinite(start) and np.isfinite(duration) and duration > 0 else _safe_float(row.get("pt_end", peak))
    if not np.isfinite(peak):
        peak = start if np.isfinite(start) else end
    if not np.isfinite(start):
        start = peak
    if not np.isfinite(end):
        end = peak
    if end < start:
        start, end = end, start
    return float(start), float(end), float(peak)


def _prepare_events(
    events: pd.DataFrame,
    *,
    event_id_col: Optional[str],
    label_col: Optional[str],
) -> pd.DataFrame:
    if events is None or events.empty:
        raise ValueError("events must be a non-empty event table")
    event_id_col = _pick_column(events, event_id_col, "event_id")
    label_col = label_col or _pick_column(events, "pathway", "Pathway", "module", "event")
    rows = []
    for idx, row in events.reset_index(drop=True).iterrows():
        label = _event_label(row, label_col)
        event_id = str(row[event_id_col]) if event_id_col and pd.notna(row.get(event_id_col)) else f"{label}|event_{idx + 1:03d}"
        start, end, peak = _event_interval(row)
        rows.append(
            {
                "event_id": event_id,
                "event": label,
                "event_start": start,
                "event_end": end,
                "event_peak": peak,
                "event_q": _safe_float(row.get("event_q", row.get("event_fdr", np.nan))),
            }
        )
    return pd.DataFrame(rows)


def _phenotype_frame(
    phenotype: pd.DataFrame | pd.Series | dict,
    *,
    sample_col: str,
    phenotype_col: str,
    survival_time_col: Optional[str],
    survival_event_col: Optional[str],
    covariates: Optional[Sequence[str]],
) -> pd.DataFrame:
    if isinstance(phenotype, pd.Series):
        frame = phenotype.rename(phenotype_col).reset_index()
        frame = frame.rename(columns={frame.columns[0]: sample_col})
    elif isinstance(phenotype, dict):
        frame = pd.DataFrame({sample_col: list(phenotype), phenotype_col: list(phenotype.values())})
    else:
        frame = phenotype.copy()
    required = [sample_col]
    if survival_time_col and survival_event_col:
        required.extend([survival_time_col, survival_event_col])
    else:
        required.append(phenotype_col)
    for col in list(required) + list(covariates or []):
        if col not in frame:
            raise ValueError(f"phenotype table is missing required column '{col}'")
    keep = list(dict.fromkeys(required + list(covariates or [])))
    return frame[keep].drop_duplicates(sample_col).reset_index(drop=True)


def _normalise_scores(
    event_scores: pd.DataFrame,
    *,
    sample_col: str,
    label_col: Optional[str],
    time_col: Optional[str],
    score_col: Optional[str],
) -> tuple[pd.DataFrame, str, str, str]:
    if event_scores is None or event_scores.empty:
        raise ValueError("event_scores must be a non-empty sample-window score table")
    if sample_col not in event_scores:
        raise ValueError(f"event_scores is missing sample column '{sample_col}'")
    label_col = label_col or _pick_column(event_scores, "pathway", "Pathway", "module", "event")
    time_col = _pick_column(event_scores, time_col, "pt_mid", "center_time", "state_time")
    score_col = _pick_column(event_scores, score_col, "NES", "module_score", "score")
    if label_col is None:
        raise ValueError("Could not infer event/pathway label column in event_scores")
    if time_col is None:
        raise ValueError("Could not infer time column in event_scores")
    if score_col is None:
        raise ValueError("Could not infer score column in event_scores")
    work = event_scores.copy()
    work["__sample"] = work[sample_col].astype(str)
    work["__event_label"] = work[label_col].astype(str)
    work["__time"] = pd.to_numeric(work[time_col], errors="coerce")
    work["__score"] = pd.to_numeric(work[score_col], errors="coerce")
    work = work[np.isfinite(work["__time"]) & np.isfinite(work["__score"])].copy()
    return work, label_col, time_col, score_col


def _compute_burdens(
    event_scores: pd.DataFrame,
    events: pd.DataFrame,
    *,
    sample_col: str,
    label_col: Optional[str],
    time_col: Optional[str],
    score_col: Optional[str],
    use_abs_score: bool,
) -> pd.DataFrame:
    scores, _label_col, _time_col, _score_col = _normalise_scores(
        event_scores,
        sample_col=sample_col,
        label_col=label_col,
        time_col=time_col,
        score_col=score_col,
    )
    rows = []
    all_samples = sorted(scores["__sample"].dropna().astype(str).unique(), key=str)
    for event in events.itertuples(index=False):
        group = scores[scores["__event_label"].astype(str) == str(event.event)].copy()
        if group.empty:
            continue
        start = float(event.event_start)
        end = float(event.event_end)
        peak = float(event.event_peak)
        span = max(end - start, 1e-12)
        for sample in all_samples:
            sample_group = group[group["__sample"] == sample].sort_values("__time")
            if sample_group.empty:
                continue
            in_event = sample_group[
                (pd.to_numeric(sample_group["__time"], errors="coerce") >= start)
                & (pd.to_numeric(sample_group["__time"], errors="coerce") <= end)
            ].copy()
            if in_event.empty:
                distances = (pd.to_numeric(sample_group["__time"], errors="coerce") - peak).abs()
                in_event = sample_group.loc[[distances.idxmin()]].copy()
            values = pd.to_numeric(in_event["__score"], errors="coerce").to_numpy(dtype=float)
            if use_abs_score:
                values = np.abs(values)
            times = pd.to_numeric(in_event["__time"], errors="coerce").to_numpy(dtype=float)
            order = np.argsort(times)
            values = values[order]
            times = times[order]
            burden = _trapz(values, times)
            if len(values) == 1 and np.isfinite(burden) and span > 0:
                burden = float(burden * span)
            rows.append(
                {
                    sample_col: sample,
                    "event_id": event.event_id,
                    "event": event.event,
                    "event_start": start,
                    "event_end": end,
                    "event_peak": peak,
                    "event_burden": burden,
                    "event_burden_abs": float(abs(burden)) if np.isfinite(burden) else np.nan,
                    "n_event_windows": int(len(in_event)),
                    "burden_score_type": "absolute_integrated_score" if use_abs_score else "signed_integrated_score",
                }
            )
    return pd.DataFrame(rows)


def _infer_phenotype_type(
    phenotype_type: str,
    y: pd.Series,
    *,
    survival_time_col: Optional[str],
    survival_event_col: Optional[str],
) -> str:
    if survival_time_col and survival_event_col:
        return "survival"
    if phenotype_type != "auto":
        return phenotype_type
    values = y.dropna()
    unique = set(values.astype(str).unique())
    if len(unique) == 2:
        return "binary"
    return "continuous"


def _standardize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = np.asarray(values, dtype=float)
    center = float(np.nanmean(values))
    scale = float(np.nanstd(values))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return (values - center) / scale, center, scale


def _design_matrix(data: pd.DataFrame, burden_col: str, covariates: Optional[Sequence[str]]) -> tuple[np.ndarray, list[str], float]:
    burden = pd.to_numeric(data[burden_col], errors="coerce").to_numpy(dtype=float)
    burden_z, _center, burden_scale = _standardize(burden)
    frames = [pd.Series(burden_z, name=burden_col, index=data.index)]
    for covariate in covariates or []:
        values = data[covariate]
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.notna().all():
            cov_z, _c, _s = _standardize(numeric.to_numpy(dtype=float))
            frames.append(pd.Series(cov_z, name=str(covariate), index=data.index))
        else:
            dummies = pd.get_dummies(values.astype(str), prefix=str(covariate), drop_first=True, dtype=float)
            for col in dummies.columns:
                frames.append(dummies[col])
    X_no_intercept = pd.concat(frames, axis=1).astype(float)
    X = np.column_stack([np.ones(len(X_no_intercept)), X_no_intercept.to_numpy(dtype=float)])
    return X, ["intercept", *list(X_no_intercept.columns)], burden_scale


def _ols_fit(y: np.ndarray, X: np.ndarray, predictor_index: int = 1) -> dict:
    y = np.asarray(y, dtype=float)
    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    resid = y - X @ beta
    dof = max(X.shape[0] - X.shape[1], 1)
    sigma2 = float((resid @ resid) / dof)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    stat = beta[predictor_index] / se[predictor_index] if se[predictor_index] > 0 else np.nan
    try:
        from scipy import stats

        p = float(2.0 * stats.t.sf(abs(stat), dof)) if np.isfinite(stat) else np.nan
    except Exception:
        p = np.nan
    return {
        "beta": float(beta[predictor_index]),
        "standard_error": float(se[predictor_index]),
        "statistic": float(stat) if np.isfinite(stat) else np.nan,
        "p_value": p,
        "model_status": "ok",
    }


def _logistic_fit(y: np.ndarray, X: np.ndarray, predictor_index: int = 1) -> dict:
    y = np.asarray(y, dtype=float)
    try:
        import statsmodels.api as sm

        fit = sm.Logit(y, X).fit(disp=False, maxiter=200)
        beta = float(fit.params[predictor_index])
        se = float(fit.bse[predictor_index])
        stat = float(fit.tvalues[predictor_index])
        p = float(fit.pvalues[predictor_index])
        return {
            "beta": beta,
            "standard_error": se,
            "statistic": stat,
            "p_value": p,
            "model_status": "ok",
        }
    except Exception:
        out = _ols_fit(y, X, predictor_index=predictor_index)
        out["model_status"] = "linear_probability_fallback"
        return out


def _cox_fit(data: pd.DataFrame, X: np.ndarray, names: list[str], time_col: str, event_col: str, predictor_index: int = 1) -> dict:
    try:
        from lifelines import CoxPHFitter

        frame = pd.DataFrame(X[:, 1:], columns=names[1:], index=data.index)
        frame["__duration"] = pd.to_numeric(data[time_col], errors="coerce").to_numpy(dtype=float)
        frame["__event"] = pd.to_numeric(data[event_col], errors="coerce").to_numpy(dtype=float)
        fit = CoxPHFitter(penalizer=0.01)
        fit.fit(frame, duration_col="__duration", event_col="__event")
        predictor = names[predictor_index]
        row = fit.summary.loc[predictor]
        return {
            "beta": float(row["coef"]),
            "standard_error": float(row["se(coef)"]),
            "statistic": float(row["z"]),
            "p_value": float(row["p"]),
            "model_status": "ok",
        }
    except Exception:
        return {
            "beta": np.nan,
            "standard_error": np.nan,
            "statistic": np.nan,
            "p_value": np.nan,
            "model_status": "survival_model_unavailable",
        }


def _fit_event_association(
    merged: pd.DataFrame,
    *,
    burden_col: str,
    phenotype_col: str,
    phenotype_type: str,
    survival_time_col: Optional[str],
    survival_event_col: Optional[str],
    covariates: Optional[Sequence[str]],
) -> dict:
    data = merged.copy()
    required = [burden_col]
    if phenotype_type == "survival":
        required.extend([survival_time_col, survival_event_col])
    else:
        required.append(phenotype_col)
    required.extend(list(covariates or []))
    data = data.dropna(subset=[col for col in required if col is not None])
    if len(data) < max(4, len(covariates or []) + 3):
        return {
            "beta": np.nan,
            "standard_error": np.nan,
            "statistic": np.nan,
            "p_value": np.nan,
            "model_status": "too_few_samples",
            "n_samples": int(len(data)),
        }
    X, names, burden_scale = _design_matrix(data, burden_col, covariates)
    if phenotype_type == "continuous":
        y = pd.to_numeric(data[phenotype_col], errors="coerce").to_numpy(dtype=float)
        y, _center, _scale = _standardize(y)
        out = _ols_fit(y, X)
        out["model_type"] = "linear"
    elif phenotype_type == "binary":
        values = data[phenotype_col]
        if pd.api.types.is_numeric_dtype(values):
            y = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        else:
            categories = sorted(values.astype(str).unique(), key=str)
            if len(categories) != 2:
                return {
                    "beta": np.nan,
                    "standard_error": np.nan,
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "model_status": "binary_outcome_requires_two_classes",
                    "n_samples": int(len(data)),
                }
            y = (values.astype(str) == categories[-1]).astype(float).to_numpy()
        if len(np.unique(y[np.isfinite(y)])) != 2:
            return {
                "beta": np.nan,
                "standard_error": np.nan,
                "statistic": np.nan,
                "p_value": np.nan,
                "model_status": "binary_outcome_requires_two_classes",
                "n_samples": int(len(data)),
            }
        out = _logistic_fit(y, X)
        out["model_type"] = "logistic"
    elif phenotype_type == "survival":
        out = _cox_fit(data, X, names, str(survival_time_col), str(survival_event_col))
        out["model_type"] = "cox"
    else:
        raise ValueError(f"Unsupported phenotype_type '{phenotype_type}'")
    out["n_samples"] = int(len(data))
    out["burden_scale"] = float(burden_scale)
    out["covariates"] = ";".join(map(str, covariates or []))
    return out


def _association_table(
    burdens: pd.DataFrame,
    phenotype: pd.DataFrame,
    *,
    sample_col: str,
    phenotype_col: str,
    phenotype_type: str,
    survival_time_col: Optional[str],
    survival_event_col: Optional[str],
    covariates: Optional[Sequence[str]],
    burden_col: str,
    min_samples: int,
) -> pd.DataFrame:
    rows = []
    for (event_id, event), group in burdens.groupby(["event_id", "event"], sort=False):
        merged = group.merge(phenotype, on=sample_col, how="inner")
        if len(merged) < int(min_samples):
            continue
        if phenotype_type == "auto":
            inferred = _infer_phenotype_type(
                phenotype_type,
                merged[phenotype_col] if phenotype_col in merged else pd.Series(dtype=float),
                survival_time_col=survival_time_col,
                survival_event_col=survival_event_col,
            )
        else:
            inferred = phenotype_type
        fit = _fit_event_association(
            merged,
            burden_col=burden_col,
            phenotype_col=phenotype_col,
            phenotype_type=inferred,
            survival_time_col=survival_time_col,
            survival_event_col=survival_event_col,
            covariates=covariates,
        )
        rows.append(
            {
                "event_id": event_id,
                "event": event,
                "phenotype": phenotype_col if inferred != "survival" else f"{survival_time_col}:{survival_event_col}",
                "phenotype_type": inferred,
                "model_type": fit.get("model_type", inferred),
                "beta": fit.get("beta", np.nan),
                "standard_error": fit.get("standard_error", np.nan),
                "statistic": fit.get("statistic", np.nan),
                "event_p": fit.get("p_value", np.nan),
                "n_samples": fit.get("n_samples", len(merged)),
                "burden_scale": fit.get("burden_scale", np.nan),
                "burden_col": burden_col,
                "covariates": fit.get("covariates", ""),
                "model_status": fit.get("model_status", "failed"),
                "effect_direction": "positive" if _safe_float(fit.get("beta", np.nan)) > 0 else "negative",
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["event_q"] = _bh_adjust(out["event_p"].to_numpy(dtype=float))
    return out.sort_values(["event_q", "event_p", "event"], na_position="last").reset_index(drop=True)


def _prediction_performance(
    burdens: pd.DataFrame,
    phenotype: pd.DataFrame,
    *,
    sample_col: str,
    phenotype_col: str,
    phenotype_type: str,
    survival_time_col: Optional[str],
    survival_event_col: Optional[str],
    cv_splits: int,
    seed: int,
    burden_col: str,
) -> pd.DataFrame:
    matrix = burdens.pivot_table(index=sample_col, columns="event_id", values=burden_col, aggfunc="mean")
    if matrix.empty:
        return pd.DataFrame()
    data = phenotype.merge(matrix.reset_index(), on=sample_col, how="inner")
    event_cols = [col for col in matrix.columns if col in data]
    if len(event_cols) == 0:
        return pd.DataFrame()
    X = data[event_cols].astype(float)
    X = X.fillna(X.mean()).fillna(0.0)
    X = ((X - X.mean()) / X.std(ddof=0).replace(0.0, 1.0)).to_numpy(dtype=float)
    inferred = _infer_phenotype_type(
        phenotype_type,
        data[phenotype_col] if phenotype_col in data else pd.Series(dtype=float),
        survival_time_col=survival_time_col,
        survival_event_col=survival_event_col,
    )
    rows = []
    if inferred == "continuous":
        y = pd.to_numeric(data[phenotype_col], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(y)
        y = y[keep]
        X_use = X[keep]
        if len(y) >= 4:
            try:
                from sklearn.linear_model import Ridge
                from sklearn.metrics import r2_score
                from sklearn.model_selection import KFold, cross_val_predict

                splits = min(int(cv_splits), len(y))
                pred = cross_val_predict(
                    Ridge(alpha=1.0),
                    X_use,
                    y,
                    cv=KFold(n_splits=splits, shuffle=True, random_state=int(seed)),
                )
                corr = float(np.corrcoef(y, pred)[0, 1]) if np.std(pred) > 0 and np.std(y) > 0 else np.nan
                rows.append(
                    {
                        "phenotype": phenotype_col,
                        "phenotype_type": inferred,
                        "model": "ridge_all_event_burdens",
                        "n_samples": int(len(y)),
                        "n_events": int(X_use.shape[1]),
                        "cv_splits": int(splits),
                        "cross_validated_r2": float(r2_score(y, pred)),
                        "cross_validated_correlation": corr,
                        "cross_validated_AUC": np.nan,
                        "concordance_index": np.nan,
                    }
                )
            except Exception:
                rows.append({"phenotype": phenotype_col, "phenotype_type": inferred, "model": "ridge_unavailable", "n_samples": int(len(y)), "n_events": int(X_use.shape[1])})
    elif inferred == "binary":
        y_raw = data[phenotype_col]
        if pd.api.types.is_numeric_dtype(y_raw):
            y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)
        else:
            cats = sorted(y_raw.astype(str).unique(), key=str)
            y = (y_raw.astype(str) == cats[-1]).astype(float).to_numpy()
        keep = np.isfinite(y)
        y = y[keep].astype(int)
        X_use = X[keep]
        class_counts = np.bincount(y, minlength=2)
        splits = min(int(cv_splits), int(class_counts.min())) if len(class_counts) >= 2 else 0
        if splits >= 2:
            try:
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import roc_auc_score
                from sklearn.model_selection import StratifiedKFold, cross_val_predict

                pred = cross_val_predict(
                    LogisticRegression(max_iter=1000, class_weight="balanced"),
                    X_use,
                    y,
                    cv=StratifiedKFold(n_splits=splits, shuffle=True, random_state=int(seed)),
                    method="predict_proba",
                )[:, 1]
                rows.append(
                    {
                        "phenotype": phenotype_col,
                        "phenotype_type": inferred,
                        "model": "logistic_all_event_burdens",
                        "n_samples": int(len(y)),
                        "n_events": int(X_use.shape[1]),
                        "cv_splits": int(splits),
                        "cross_validated_r2": np.nan,
                        "cross_validated_correlation": np.nan,
                        "cross_validated_AUC": float(roc_auc_score(y, pred)),
                        "concordance_index": np.nan,
                    }
                )
            except Exception:
                rows.append({"phenotype": phenotype_col, "phenotype_type": inferred, "model": "logistic_unavailable", "n_samples": int(len(y)), "n_events": int(X_use.shape[1])})
    elif inferred == "survival":
        try:
            from lifelines.utils import concordance_index

            duration = pd.to_numeric(data[str(survival_time_col)], errors="coerce").to_numpy(dtype=float)
            event = pd.to_numeric(data[str(survival_event_col)], errors="coerce").to_numpy(dtype=float)
            risk = np.nanmean(X, axis=1)
            keep = np.isfinite(duration) & np.isfinite(event) & np.isfinite(risk)
            cindex = float(concordance_index(duration[keep], -risk[keep], event[keep])) if keep.sum() >= 4 else np.nan
            rows.append(
                {
                    "phenotype": f"{survival_time_col}:{survival_event_col}",
                    "phenotype_type": inferred,
                    "model": "mean_event_burden_survival_risk",
                    "n_samples": int(keep.sum()),
                    "n_events": int(X.shape[1]),
                    "cv_splits": 0,
                    "cross_validated_r2": np.nan,
                    "cross_validated_correlation": np.nan,
                    "cross_validated_AUC": np.nan,
                    "concordance_index": cindex,
                }
            )
        except Exception:
            rows.append({"phenotype": f"{survival_time_col}:{survival_event_col}", "phenotype_type": inferred, "model": "survival_performance_unavailable", "n_samples": int(len(data)), "n_events": int(X.shape[1])})
    return pd.DataFrame(rows)


def _linked_report(associations: pd.DataFrame, q_threshold: float) -> pd.DataFrame:
    if associations is None or associations.empty:
        return pd.DataFrame()
    out = associations.copy()
    out["evidence_level"] = np.where(
        (pd.to_numeric(out["event_q"], errors="coerce") <= float(q_threshold)) & (out["model_status"].astype(str) == "ok"),
        "phenotype_linked_event",
        np.where(pd.to_numeric(out["event_p"], errors="coerce") <= 0.1, "screening_phenotype_association", "not_phenotype_linked"),
    )
    cols = [
        "event_id",
        "event",
        "phenotype",
        "phenotype_type",
        "model_type",
        "beta",
        "event_p",
        "event_q",
        "effect_direction",
        "n_samples",
        "evidence_level",
        "model_status",
    ]
    return out[[col for col in cols if col in out.columns]].sort_values(["event_q", "event_p", "event"]).reset_index(drop=True)


def associate_phenotype_events(
    event_scores: pd.DataFrame,
    events: pd.DataFrame,
    phenotype: pd.DataFrame | pd.Series | dict,
    *,
    sample_col: str = "sample",
    phenotype_col: str = "phenotype",
    phenotype_type: str = "auto",
    survival_time_col: Optional[str] = None,
    survival_event_col: Optional[str] = None,
    covariates: Optional[Sequence[str]] = None,
    event_id_col: Optional[str] = "event_id",
    label_col: Optional[str] = None,
    time_col: Optional[str] = None,
    score_col: Optional[str] = None,
    use_abs_score: bool = False,
    burden_col: str = "event_burden",
    min_samples: int = 4,
    cv_splits: int = 5,
    q_threshold: float = 0.1,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Link trajectory events to sample-level phenotypes.

    The method integrates a sample-specific pathway/module score over each
    event interval to form an event burden, then fits phenotype association
    models per event. Continuous phenotypes use linear regression, binary
    phenotypes use logistic regression, and survival phenotypes use Cox models
    when ``lifelines`` is available.
    """
    prepared_events = _prepare_events(events, event_id_col=event_id_col, label_col=label_col)
    burdens = _compute_burdens(
        event_scores,
        prepared_events,
        sample_col=sample_col,
        label_col=label_col,
        time_col=time_col,
        score_col=score_col,
        use_abs_score=use_abs_score,
    )
    pheno = _phenotype_frame(
        phenotype,
        sample_col=sample_col,
        phenotype_col=phenotype_col,
        survival_time_col=survival_time_col,
        survival_event_col=survival_event_col,
        covariates=covariates,
    )
    associations = _association_table(
        burdens,
        pheno,
        sample_col=sample_col,
        phenotype_col=phenotype_col,
        phenotype_type=phenotype_type,
        survival_time_col=survival_time_col,
        survival_event_col=survival_event_col,
        covariates=covariates,
        burden_col=burden_col,
        min_samples=min_samples,
    )
    performance = _prediction_performance(
        burdens,
        pheno,
        sample_col=sample_col,
        phenotype_col=phenotype_col,
        phenotype_type=phenotype_type,
        survival_time_col=survival_time_col,
        survival_event_col=survival_event_col,
        cv_splits=cv_splits,
        seed=seed,
        burden_col=burden_col,
    )
    report = _linked_report(associations, q_threshold=q_threshold)
    for table in (burdens, associations, performance, report):
        table.attrs["phenotype_linked_ted"] = {
            "sample_col": sample_col,
            "phenotype_col": phenotype_col,
            "phenotype_type": phenotype_type,
            "burden_col": burden_col,
            "q_threshold": float(q_threshold),
        }
    return {
        "event_burden_score": burdens,
        "phenotype_event_association": associations,
        "phenotype_prediction_performance": performance,
        "phenotype_linked_event_report": report,
    }


def write_phenotype_event_association(
    tables: dict[str, pd.DataFrame],
    outdir: str | Path,
    *,
    sep: str = "\t",
) -> dict[str, Path]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, filename in _PHENOTYPE_OUTPUT_FILENAMES.items():
        table = tables.get(key, pd.DataFrame())
        path = out / filename
        table.to_csv(path, sep=sep, index=False, na_rep="NA")
        paths[key] = path
    return paths
