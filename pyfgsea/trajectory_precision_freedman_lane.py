from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .trajectory_covariate_pseudobulk import _by_adjust
from .trajectory_covariate_simulation import ArrayFreedmanLanePlan


def _studentized(beta: np.ndarray, standard_error: np.ndarray) -> np.ndarray:
    beta_array = np.asarray(beta, dtype=float)
    error_array = np.asarray(standard_error, dtype=float)
    tolerance = np.finfo(float).eps * np.maximum(1.0, np.abs(beta_array))
    result = np.zeros_like(beta_array)
    regular = error_array > tolerance
    result[regular] = beta_array[regular] / error_array[regular]
    deterministic = (~regular) & (np.abs(beta_array) > tolerance)
    result[deterministic] = np.sign(beta_array[deterministic]) * np.inf
    return result


def _precision_batch(
    precision_scale: np.ndarray,
    *,
    n_replicates: int,
    n_donors: int,
    n_bins: int,
) -> np.ndarray:
    scale = np.asarray(precision_scale, dtype=float)
    if scale.ndim == 2:
        if scale.shape != (n_donors, n_bins):
            raise ValueError("Two-dimensional precision_scale must be donor/bin aligned")
        scale = np.broadcast_to(scale[None, :, :], (n_replicates, n_donors, n_bins))
    elif scale.ndim == 3:
        if scale.shape != (n_replicates, n_donors, n_bins):
            raise ValueError(
                "Three-dimensional precision_scale must be replicate/donor/bin aligned"
            )
    else:
        raise ValueError(
            "precision_scale must be two- or three-dimensional with donor/bin axes"
        )
    return scale


def _validate_alpha(alpha: float) -> float:
    value = float(alpha)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    return value


def additive_cell_count_precision_scale(
    cell_count: np.ndarray,
    *,
    available: np.ndarray,
    biological_variance_fraction: float,
    reference_count: float | None = None,
    maximum_scale: float | None = None,
) -> np.ndarray:
    """Build ``sqrt(tau^2 + sigma^2 * n_ref / n)`` precision multipliers.

    The biological and sampling fractions sum to one at ``reference_count``.
    Requiring the fraction explicitly prevents a post-outcome default from being
    silently selected.  This model does not multiply a second total-cell-count
    factor into the donor-by-bin sampling precision.
    """

    counts = np.asarray(cell_count, dtype=float)
    mask = np.asarray(available, dtype=bool)
    if counts.ndim not in {2, 3} or mask.ndim not in {2, 3}:
        raise ValueError("cell_count and available must be aligned 2D or 3D arrays")
    if counts.ndim == 2:
        if mask.shape != counts.shape:
            raise ValueError("cell_count and available must be aligned 2D or 3D arrays")
    elif mask.ndim == 2 and mask.shape == counts.shape[1:]:
        mask = np.broadcast_to(mask[None, :, :], counts.shape)
    elif mask.shape != counts.shape:
        raise ValueError("cell_count and available must be aligned 2D or 3D arrays")
    if np.any(~np.isfinite(counts[mask])) or np.any(counts[mask] <= 0):
        raise ValueError("Available cell counts must be finite and positive")
    fraction = float(biological_variance_fraction)
    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("biological_variance_fraction must be strictly between 0 and 1")
    positive = counts[mask]
    reference = float(np.median(positive) if reference_count is None else reference_count)
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("reference_count must be finite and positive")
    result = np.full_like(counts, np.nan, dtype=float)
    result[mask] = np.sqrt(
        fraction + (1.0 - fraction) * reference / counts[mask]
    )
    if maximum_scale is not None:
        maximum = float(maximum_scale)
        if not np.isfinite(maximum) or maximum < 1.0:
            raise ValueError("maximum_scale must be finite and at least one")
        result[mask] = np.minimum(result[mask], maximum)
    return result


def precision_estimability_diagnostics(
    plan: ArrayFreedmanLanePlan,
    precision_scale: np.ndarray,
    *,
    available: np.ndarray,
    minimum_group_kish_ess: float = 2.0,
    minimum_condition_information: float = 0.10,
    maximum_group_weight_ratio: float = 100.0,
) -> dict[str, Any]:
    """Diagnose whether fixed, outcome-blind precision weights retain information.

    ``precision_scale`` is a residual standard-deviation multiplier, so WLS weights
    are ``1 / precision_scale**2``.  No expression or pathway outcome is used.
    """

    available_array = np.asarray(available, dtype=bool)
    if available_array.ndim != 2 or available_array.shape[0] != len(plan.condition):
        raise ValueError("available must be a donor/bin mask aligned with the plan")
    scale = np.asarray(precision_scale, dtype=float)
    n_replicates = 1 if scale.ndim == 2 else scale.shape[0] if scale.ndim == 3 else 0
    if n_replicates < 1:
        raise ValueError("precision_scale must be two- or three-dimensional")
    scale_batch = _precision_batch(
        scale,
        n_replicates=n_replicates,
        n_donors=available_array.shape[0],
        n_bins=available_array.shape[1],
    )
    valid_positions = np.broadcast_to(available_array[None, :, :], scale_batch.shape)
    if np.any(~np.isfinite(scale_batch[valid_positions])) or np.any(
        scale_batch[valid_positions] <= 0
    ):
        raise ValueError("Available precision scales must be finite and positive")
    thresholds = (
        float(minimum_group_kish_ess),
        float(minimum_condition_information),
        float(maximum_group_weight_ratio),
    )
    if (
        not np.isfinite(thresholds).all()
        or thresholds[0] <= 0
        or thresholds[1] <= 0
        or thresholds[2] < 1
    ):
        raise ValueError("Precision estimability thresholds are invalid")

    condition = np.asarray(plan.condition, dtype=bool)
    reduced = np.asarray(plan.reduced_design, dtype=float)
    n_bins = available_array.shape[1]
    control_ess = np.zeros((n_replicates, n_bins), dtype=float)
    case_ess = np.zeros_like(control_ess)
    condition_information = np.zeros_like(control_ess)
    unit_noise_se = np.full_like(control_ess, np.inf)
    group_weight_ratio = np.full_like(control_ess, np.inf)

    for replicate in range(n_replicates):
        for bin_index in range(n_bins):
            indices = np.flatnonzero(available_array[:, bin_index])
            local_condition = condition[indices]
            weights = 1.0 / scale_batch[replicate, indices, bin_index] ** 2
            group_sums = []
            for group, output in ((False, control_ess), (True, case_ess)):
                local = weights[local_condition == group]
                group_sums.append(float(local.sum()))
                if len(local) and float(np.sum(local**2)) > 0:
                    output[replicate, bin_index] = float(local.sum() ** 2 / np.sum(local**2))
            if min(group_sums) > 0:
                group_weight_ratio[replicate, bin_index] = max(group_sums) / min(
                    group_sums
                )
            square_root_weight = np.sqrt(weights)
            z = reduced[indices] * square_root_weight[:, None]
            c = local_condition.astype(float) * square_root_weight
            residual_condition = c - z @ (np.linalg.pinv(z) @ c)
            information = float(residual_condition @ residual_condition)
            condition_information[replicate, bin_index] = information
            if information > np.finfo(float).eps:
                unit_noise_se[replicate, bin_index] = 1.0 / np.sqrt(information)

    reasons = []
    if float(np.min(control_ess)) < thresholds[0]:
        reasons.append("CONTROL_KISH_ESS_BELOW_MINIMUM")
    if float(np.min(case_ess)) < thresholds[0]:
        reasons.append("CASE_KISH_ESS_BELOW_MINIMUM")
    if float(np.min(condition_information)) < thresholds[1]:
        reasons.append("CONDITION_INFORMATION_BELOW_MINIMUM")
    if float(np.max(group_weight_ratio)) > thresholds[2]:
        reasons.append("GROUP_WEIGHT_RATIO_ABOVE_MAXIMUM")
    return {
        "precision_weighting_outcome_blind": True,
        "n_replicates": int(n_replicates),
        "n_bins": int(n_bins),
        "control_kish_ess": control_ess,
        "case_kish_ess": case_ess,
        "condition_information": condition_information,
        "unit_noise_standard_error": unit_noise_se,
        "group_weight_ratio": group_weight_ratio,
        "minimum_group_kish_ess_observed": float(
            min(np.min(control_ess), np.min(case_ess))
        ),
        "minimum_condition_information_observed": float(
            np.min(condition_information)
        ),
        "maximum_group_weight_ratio_observed": float(np.max(group_weight_ratio)),
        "thresholds": {
            "minimum_group_kish_ess": thresholds[0],
            "minimum_condition_information": thresholds[1],
            "maximum_group_weight_ratio": thresholds[2],
        },
        "estimability_pass": not reasons,
        "reason_codes": reasons,
    }


def run_precision_standardized_array_freedman_lane_batch(
    scores: np.ndarray,
    plan: ArrayFreedmanLanePlan,
    *,
    precision_scale: np.ndarray,
    alpha: float = 0.05,
    family_index: Sequence[int] | None = None,
    mapping_batch_size: int = 32,
) -> dict[str, np.ndarray]:
    """Run WLS Freedman--Lane after whitening whole donor residual curves.

    The standard-deviation scale must be fixed without pathway outcomes.  The
    response, nuisance design, condition contrast, and residual curves are all
    transformed by the same donor-by-bin scale.  One donor mapping is then used
    synchronously across every selected bin and pathway.
    """

    values = np.asarray(scores, dtype=float)
    if values.ndim != 4:
        raise ValueError("scores require replicate/donor/bin/pathway axes")
    n_replicates, n_donors, n_bins, n_pathways = values.shape
    if n_donors != len(plan.condition) or n_bins != len(plan.widths):
        raise ValueError("scores and the Freedman-Lane plan are not aligned")
    alpha_value = _validate_alpha(alpha)
    if not isinstance(mapping_batch_size, (int, np.integer)) or int(mapping_batch_size) < 1:
        raise ValueError("mapping_batch_size must be a positive integer")
    mapping_batch_size = int(mapping_batch_size)
    scale = _precision_batch(
        precision_scale,
        n_replicates=n_replicates,
        n_donors=n_donors,
        n_bins=n_bins,
    )
    reduced = np.asarray(plan.reduced_design, dtype=float)
    condition = np.asarray(plan.condition, dtype=bool)
    mappings = np.asarray(plan.null_mappings, dtype=int)
    if (
        reduced.ndim != 2
        or reduced.shape[0] != n_donors
        or not np.isfinite(reduced).all()
        or mappings.ndim != 2
        or mappings.shape[1] != n_donors
    ):
        raise ValueError("Freedman-Lane design or residual mappings are invalid")

    beta_curve = np.empty((n_replicates, n_bins, n_pathways), dtype=float)
    t_curve = np.empty_like(beta_curve)
    standard_error_curve = np.empty_like(beta_curve)
    adjusted_control_curve = np.empty_like(beta_curve)
    adjusted_case_curve = np.empty_like(beta_curve)
    residual_df_by_bin = np.empty(n_bins, dtype=int)
    bin_payloads: list[dict[str, Any]] = []

    for bin_index in range(n_bins):
        first_available = np.isfinite(values[0, :, bin_index, 0])
        expected_availability = np.broadcast_to(
            first_available, (n_replicates, n_donors)
        )
        if not np.array_equal(
            np.isfinite(values[:, :, bin_index, 0]), expected_availability
        ):
            raise ValueError("Every replicate must share the frozen missingness")
        indices = np.flatnonzero(first_available)
        y = values[:, indices, bin_index, :]
        local_scale = scale[:, indices, bin_index]
        if (
            not np.isfinite(y).all()
            or not np.isfinite(local_scale).all()
            or np.any(local_scale <= 0)
        ):
            raise ValueError("Available scores and precision scales must be finite")
        weight = 1.0 / local_scale
        z_original = reduced[indices]
        c_original = condition[indices].astype(float)
        z = weight[:, :, None] * z_original[None, :, :]
        c = weight * c_original[None, :]
        full = np.concatenate([z, c[:, :, None]], axis=2)
        y_whitened = weight[:, :, None] * y
        pinv_z = np.linalg.pinv(z)
        pinv_full = np.linalg.pinv(full)
        fitted_reduced = np.einsum(
            "rik,rkj,rjp->rip", z, pinv_z, y_whitened, optimize=True
        )
        residual_reduced = y_whitened - fitted_reduced
        beta_weight = pinv_full[:, -1, :]
        beta = np.einsum("ri,rip->rp", beta_weight, y_whitened, optimize=True)
        full_projection = np.einsum("rik,rkj->rij", full, pinv_full, optimize=True)
        residual_projection = np.eye(len(indices))[None, :, :] - full_projection
        residual_full = np.einsum(
            "rij,rjp->rip", residual_projection, y_whitened, optimize=True
        )
        ranks = np.linalg.matrix_rank(full)
        if not np.all(ranks == ranks[0]):
            raise ValueError("Precision weights changed the full-model rank")
        residual_df = int(len(indices) - ranks[0])
        if residual_df < 1:
            raise ValueError("Precision-standardized model has no residual degrees of freedom")
        sigma2 = np.sum(residual_full**2, axis=1) / residual_df
        beta_standard_error_factor = np.sqrt(np.sum(beta_weight**2, axis=1))
        standard_error = (
            np.sqrt(np.maximum(sigma2, 0.0))
            * beta_standard_error_factor[:, None]
        )
        t_value = _studentized(beta, standard_error)
        if not np.isfinite(t_value).all():
            raise ValueError("Precision-standardized observed statistic is undefined")
        baseline_weight = np.einsum(
            "k,rki->ri", np.mean(z_original, axis=0), pinv_full[:, :-1, :]
        )
        baseline = np.einsum(
            "ri,rip->rp", baseline_weight, y_whitened, optimize=True
        )
        beta_curve[:, bin_index, :] = beta
        t_curve[:, bin_index, :] = t_value
        standard_error_curve[:, bin_index, :] = standard_error
        adjusted_control_curve[:, bin_index, :] = baseline
        adjusted_case_curve[:, bin_index, :] = baseline + beta
        residual_df_by_bin[bin_index] = residual_df

        global_to_local = np.full(n_donors, -1, dtype=int)
        global_to_local[indices] = np.arange(len(indices), dtype=int)
        source_local = global_to_local[mappings[:, indices]]
        if np.any(source_local < 0):
            raise RuntimeError("Residual mapping crossed the frozen availability signature")
        bin_payloads.append(
            {
                "fitted_reduced": fitted_reduced,
                "residual_reduced": residual_reduced,
                "source_local": source_local,
                "beta_weight": beta_weight,
                "residual_projection": residual_projection,
                "residual_df": residual_df,
                "standard_error_factor": beta_standard_error_factor,
            }
        )

    observed = np.max(np.abs(t_curve), axis=1)
    effect_statistics = np.max(np.abs(beta_curve), axis=1)
    exceed = np.zeros((n_replicates, n_pathways), dtype=np.int64)
    max_exceed = np.zeros_like(exceed)
    families = None if family_index is None else np.asarray(family_index, dtype=int)
    within_family_exceed = np.zeros_like(exceed)
    family_gate_exceed = np.zeros_like(exceed)
    if families is not None:
        if families.shape != (n_pathways,) or np.any(families < -1):
            raise ValueError("family_index must use -1 or a non-negative family code")
        assigned_families = np.unique(families[families >= 0])
        if not len(assigned_families):
            raise ValueError("family_index must assign at least one pathway")
        observed_family_max = np.stack(
            [
                np.max(observed[:, families == family], axis=1)
                for family in assigned_families
            ],
            axis=1,
        )
    else:
        assigned_families = np.empty(0, dtype=int)
        observed_family_max = np.empty((n_replicates, 0), dtype=float)

    for start in range(0, len(mappings), mapping_batch_size):
        stop = min(start + mapping_batch_size, len(mappings))
        null_stat = np.zeros(
            (stop - start, n_replicates, n_pathways), dtype=float
        )
        for payload in bin_payloads:
            residual = payload["residual_reduced"]
            source = payload["source_local"][start:stop]
            permuted = np.transpose(residual[:, source, :], (1, 0, 2, 3))
            y_star = permuted + payload["fitted_reduced"][None, :, :, :]
            null_beta = np.einsum(
                "ri,mrip->mrp", payload["beta_weight"], y_star, optimize=True
            )
            null_residual = np.einsum(
                "rij,mrjp->mrip",
                payload["residual_projection"],
                y_star,
                optimize=True,
            )
            null_sigma2 = np.sum(null_residual**2, axis=2) / payload["residual_df"]
            null_error = (
                np.sqrt(np.maximum(null_sigma2, 0.0))
                * payload["standard_error_factor"][None, :, None]
            )
            null_t = _studentized(null_beta, null_error)
            if not np.isfinite(null_t).all():
                raise ValueError("Precision-standardized null statistic is undefined")
            null_stat = np.maximum(null_stat, np.abs(null_t))
        exceed += np.sum(null_stat >= observed[None, :, :] - 1e-12, axis=0)
        global_max = np.max(null_stat, axis=2)
        max_exceed += np.sum(
            global_max[:, :, None] >= observed[None, :, :] - 1e-12, axis=0
        )
        if families is not None:
            null_family_max = np.stack(
                [
                    np.max(null_stat[:, :, families == family], axis=2)
                    for family in assigned_families
                ],
                axis=2,
            )
            global_family_max = np.max(null_family_max, axis=2)
            for position, family in enumerate(assigned_families):
                members = families == family
                within_family_exceed[:, members] += np.sum(
                    null_family_max[:, :, position, None]
                    >= observed[:, members][None, :, :] - 1e-12,
                    axis=0,
                )
                family_gate_exceed[:, members] += np.sum(
                    global_family_max[:, :, None]
                    >= observed_family_max[:, position][None, :, None] - 1e-12,
                    axis=0,
                )

    denominator = len(mappings) + 1
    p_raw = (exceed + 1.0) / denominator
    p_max_t = (max_exceed + 1.0) / denominator
    q_by = np.vstack([_by_adjust(row) for row in p_raw])
    if families is None:
        p_within_family = np.full_like(p_raw, np.nan)
        p_family_gate = np.full_like(p_raw, np.nan)
    else:
        p_within_family = (within_family_exceed + 1.0) / denominator
        p_family_gate = (family_gate_exceed + 1.0) / denominator
        p_within_family[:, families < 0] = 1.0
        p_family_gate[:, families < 0] = 1.0
    hierarchical_reject = (
        np.isfinite(p_within_family)
        & (p_within_family <= alpha_value)
        & (p_family_gate <= alpha_value)
    )
    production_reject = (
        np.isfinite(p_family_gate)
        & (p_max_t <= alpha_value)
        & (p_family_gate <= alpha_value)
    )
    return {
        "beta_curve": beta_curve,
        "standard_error_curve": standard_error_curve,
        "adjusted_control_curve": adjusted_control_curve,
        "adjusted_case_curve": adjusted_case_curve,
        "residual_df_by_bin": residual_df_by_bin,
        "calibration_curve": t_curve,
        "effect_statistic": effect_statistics,
        "calibration_statistic": observed,
        "p_raw": p_raw,
        "p_maxT": p_max_t,
        "q_by": q_by,
        "raw_reject": p_raw <= alpha_value,
        "maxT_reject": p_max_t <= alpha_value,
        "by_reject": q_by <= alpha_value,
        "p_pathway_within_family_maxT": p_within_family,
        "p_family_gate": p_family_gate,
        "within_family_maxT_reject": p_within_family <= alpha_value,
        "family_gate_reject": p_family_gate <= alpha_value,
        "family_hierarchical_reject": hierarchical_reject,
        "production_family_hierarchical_reject": production_reject,
        "precision_standardized": np.asarray(True),
    }


def json_ready_precision_diagnostics(value: Mapping[str, Any]) -> dict[str, Any]:
    """Convert precision diagnostics to a strict JSON-ready mapping."""

    result: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, np.ndarray):
            result[key] = item.tolist()
        elif isinstance(item, np.generic):
            result[key] = item.item()
        elif isinstance(item, Mapping):
            result[key] = json_ready_precision_diagnostics(item)
        else:
            result[key] = item
    return result
