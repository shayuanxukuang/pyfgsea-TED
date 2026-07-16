from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .trajpathmix_functional_core_v1 import (
    BinSpecificDesignPlan,
    FunctionalCoreDesignError,
    RANK_RELATIVE_TOLERANCE,
    build_bin_specific_designs,
)


METHOD_ID = "trajpathmix_functional_core_v2_lodo_multiplier"
SCHEMA_VERSION = "2.0.0-v2.0-synthetic"
NUMERICAL_TOLERANCE = 1.0e-12
DEFAULT_DONOR_MULTIPLIER_SEED = 2026071501
DEFAULT_COMPONENT_MULTIPLIER_SEED = 2026071502
WEBB_SIX_POINT_SUPPORT = np.asarray(
    [
        -math.sqrt(1.5),
        -1.0,
        -math.sqrt(0.5),
        math.sqrt(0.5),
        1.0,
        math.sqrt(1.5),
    ],
    dtype=float,
)


class FunctionalCoreV2DesignError(FunctionalCoreDesignError):
    """Fail-closed V2 design or leave-one-donor-out error."""


def _freeze(values: Any, dtype: Any | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _canonical_axis(values: Sequence[Any], label: str) -> tuple[tuple[str, ...], np.ndarray]:
    observed = tuple(str(value).strip() for value in values)
    if not observed or any(not value for value in observed):
        raise ValueError(f"{label} must be nonempty and nonblank")
    if len(observed) != len(set(observed)):
        raise ValueError(f"{label} must be unique")
    order = np.argsort(np.asarray(observed, dtype=object), kind="stable")
    canonical = tuple(observed[int(index)] for index in order)
    return canonical, np.asarray(order, dtype=int)


def _as_binary(values: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    observed = np.asarray(values)
    if observed.shape != shape or not np.isin(observed, [0, 1, False, True]).all():
        raise ValueError(f"{label} must be binary with shape {shape}")
    return observed.astype(bool)


def _condition_weights_and_leverage(
    design: np.ndarray,
    *,
    rank_tolerance: float = RANK_RELATIVE_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.asarray(design, dtype=float)
    if x.ndim != 2 or x.shape[1] == 0 or x.shape[0] <= x.shape[1]:
        raise FunctionalCoreV2DesignError(
            "The condition model must have positive residual degrees of freedom"
        )
    q, r = np.linalg.qr(x, mode="reduced")
    diagonal = np.abs(np.diag(r))
    threshold = float(diagonal.max() * rank_tolerance) if len(diagonal) else 0.0
    if len(diagonal) != x.shape[1] or bool((diagonal <= threshold).any()):
        raise FunctionalCoreV2DesignError("Condition design is rank deficient")
    inverse_r = np.linalg.solve(r, np.eye(r.shape[0], dtype=float))
    condition_row = inverse_r[-1]
    weights = q @ condition_row
    leverage = np.sum(q * q, axis=1)
    return weights, leverage, threshold


def _nuisance_span_condition_weights(
    nuisance_design: np.ndarray,
    condition: np.ndarray,
    *,
    rank_tolerance: float = RANK_RELATIVE_TOLERANCE,
    minimum_residual_df: int = 1,
    minimum_donors_per_condition: int = 1,
) -> tuple[np.ndarray, int, float, int]:
    z = np.asarray(nuisance_design, dtype=float)
    c = np.asarray(condition, dtype=float)
    if z.ndim != 2 or c.shape != (z.shape[0],):
        raise FunctionalCoreV2DesignError(
            "Nuisance design and condition vector do not align"
        )
    n_case = int(np.sum(c == 1.0))
    n_control = int(np.sum(c == 0.0))
    if (
        z.shape[0] < 3
        or len(np.unique(c)) != 2
        or min(n_case, n_control) < int(minimum_donors_per_condition)
    ):
        raise FunctionalCoreV2DesignError(
            "LODO deletion left an invalid condition design"
        )
    u, singular, _ = np.linalg.svd(z, full_matrices=False)
    if len(singular) and singular[0] > 0:
        threshold = float(singular[0] * rank_tolerance)
        rank = int(np.sum(singular > threshold))
    else:
        threshold = 0.0
        rank = 0
    if rank == 0:
        raise FunctionalCoreV2DesignError("LODO nuisance span is empty")
    basis = u[:, :rank]
    residualized_condition = c - basis @ (basis.T @ c)
    information = float(residualized_condition @ residualized_condition)
    information_floor = NUMERICAL_TOLERANCE
    if information <= information_floor:
        raise FunctionalCoreV2DesignError(
            "Condition is not identifiable after LODO nuisance-span projection"
        )
    residual_df = int(len(c) - rank - 1)
    if residual_df < int(minimum_residual_df):
        raise FunctionalCoreV2DesignError(
            "LODO nuisance-span model has no residual degrees of freedom"
        )
    return residualized_condition / information, rank, information, residual_df


def _deletion_condition_weights(
    reduced_design: np.ndarray,
    condition: np.ndarray,
    delete_index: int,
    *,
    minimum_residual_df: int,
    minimum_donors_per_condition: int,
) -> tuple[np.ndarray, int, float, int]:
    z = np.asarray(reduced_design, dtype=float)
    c = np.asarray(condition, dtype=float)
    keep_rows = np.ones(len(c), dtype=bool)
    keep_rows[int(delete_index)] = False
    z_minus = z[keep_rows]
    c_minus = c[keep_rows]
    local_weights, nuisance_rank, information, residual_df = (
        _nuisance_span_condition_weights(
            z_minus,
            c_minus,
            minimum_residual_df=minimum_residual_df,
            minimum_donors_per_condition=minimum_donors_per_condition,
        )
    )
    embedded = np.zeros(len(c), dtype=float)
    embedded[keep_rows] = local_weights
    return embedded, nuisance_rank, information, residual_df


def _support_weights(
    support: np.ndarray, bin_weights: np.ndarray
) -> np.ndarray:
    weighted = support.astype(float) * np.asarray(bin_weights, dtype=float)[:, None]
    denominator = weighted.sum(axis=0)
    if np.any(denominator <= 0):
        raise FunctionalCoreV2DesignError("Every pathway must have positive supported weight")
    return weighted / denominator[None, :]


def _canonical_partition(labels: Sequence[Any]) -> tuple[tuple[int, ...], ...]:
    observed = tuple(str(value) for value in labels)
    blocks: dict[str, list[int]] = {}
    for index, value in enumerate(observed):
        blocks.setdefault(value, []).append(index)
    return tuple(sorted(tuple(indices) for indices in blocks.values()))


def build_experiment_overlap_components(
    *,
    donor_ids: Sequence[Any],
    availability: Any,
    experiment_fractions: Any,
    experiment_ids: Sequence[Any],
) -> tuple[str, ...]:
    """Derive the only admissible donor partition from experiment incidence.

    Donors are joined when they share any experiment with positive frozen
    fraction in an available bin.  Transitive closure therefore produces the
    connected components of the donor--experiment bipartite graph.  Component
    labels are deterministic under donor, bin, and experiment reordering.
    """

    donors, donor_order = _canonical_axis(donor_ids, "donor_ids")
    experiments, experiment_order = _canonical_axis(
        experiment_ids, "experiment_ids"
    )
    observed_availability = np.asarray(availability)
    if observed_availability.ndim != 2 or observed_availability.shape[0] != len(
        donors
    ):
        raise ValueError("availability must be donor x bin")
    available = _as_binary(
        observed_availability,
        (len(donors), observed_availability.shape[1]),
        "availability",
    )[donor_order]
    fractions = np.asarray(experiment_fractions, dtype=float)
    expected = (len(donors), available.shape[1], len(experiments))
    if fractions.shape != expected:
        raise ValueError(f"experiment_fractions must have shape {expected}")
    fractions = fractions[donor_order][:, :, experiment_order]
    if not bool(np.isfinite(fractions[available]).all()):
        raise ValueError("Available experiment fractions must be finite")
    incidence = np.any(
        (fractions > NUMERICAL_TOLERANCE) & available[:, :, None], axis=1
    )
    if not bool(incidence.any(axis=1).all()):
        raise FunctionalCoreV2DesignError(
            "Every donor must be incident to at least one experiment"
        )

    parent = np.arange(len(donors), dtype=int)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            if root_left < root_right:
                parent[root_right] = root_left
            else:
                parent[root_left] = root_right

    for experiment_index in range(len(experiments)):
        members = np.flatnonzero(incidence[:, experiment_index])
        if len(members) > 1:
            anchor = int(members[0])
            for member in members[1:]:
                union(anchor, int(member))

    roots = [find(index) for index in range(len(donors))]
    ordered_roots = tuple(sorted(set(roots)))
    root_to_label = {
        root: f"experiment_overlap_cc_{index:03d}"
        for index, root in enumerate(ordered_roots)
    }
    return tuple(root_to_label[root] for root in roots)


@dataclass(frozen=True)
class LodoInfluenceFit:
    donor_ids: tuple[str, ...]
    bin_ids: tuple[str, ...]
    pathway_ids: tuple[str, ...]
    experiment_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    experiment_component_ids: tuple[str, ...]
    availability: np.ndarray
    support_mask: np.ndarray
    pathway_bin_weights: np.ndarray
    effect: np.ndarray
    standard_error: np.ndarray
    studentized_effect: np.ndarray
    leverage: np.ndarray
    leave_one_out_effect: np.ndarray
    donor_influence: np.ndarray
    signed_auc: np.ndarray
    signed_auc_standard_error: np.ndarray
    signed_auc_studentized: np.ndarray
    signed_auc_donor_influence: np.ndarray
    design_plan: BinSpecificDesignPlan
    design_audit: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "method_id": METHOD_ID,
                "schema_version": SCHEMA_VERSION,
                "donor_ids": self.donor_ids,
                "bin_ids": self.bin_ids,
                "pathway_ids": self.pathway_ids,
                "experiment_ids": self.experiment_ids,
                "family_ids": self.family_ids,
                "experiment_component_ids": self.experiment_component_ids,
                "availability": self.availability,
                "support_mask": self.support_mask,
                "pathway_bin_weights": self.pathway_bin_weights,
                "effect": self.effect,
                "lodo_jackknife_standard_error": self.standard_error,
                "studentized_effect": self.studentized_effect,
                "leverage": self.leverage,
                "signed_auc": self.signed_auc,
                "signed_auc_standard_error": self.signed_auc_standard_error,
                "signed_auc_studentized": self.signed_auc_studentized,
                "design_audit": self.design_audit,
            }
        )


@dataclass(frozen=True)
class MultiplierProcessResult:
    unit_type: str
    unit_ids: tuple[str, ...]
    distribution: str
    seed: int
    multiplier_stream_sha256: str
    finite_sample_scale: float
    multipliers: np.ndarray
    coordinate_standard_error: np.ndarray
    observed_studentized_effect: np.ndarray
    studentized_draws: np.ndarray
    curve_statistic: np.ndarray
    curve_p_value: np.ndarray
    global_curve_maxT_p_value: np.ndarray
    family_ids: tuple[str, ...]
    family_statistic: np.ndarray
    family_maxT_p_value: np.ndarray
    signed_auc_standard_error: np.ndarray
    signed_auc_studentized: np.ndarray
    signed_auc_studentized_draws: np.ndarray
    signed_auc_p_value: np.ndarray
    signed_auc_global_maxT_p_value: np.ndarray
    simultaneous_critical: float
    simultaneous_order_index_1based: int
    simultaneous_lower: np.ndarray
    simultaneous_upper: np.ndarray
    sensitivity_informative: bool

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "unit_type": self.unit_type,
                "unit_ids": self.unit_ids,
                "distribution": self.distribution,
                "seed": self.seed,
                "multiplier_stream_sha256": self.multiplier_stream_sha256,
                "finite_sample_scale": self.finite_sample_scale,
                "n_draws": int(self.multipliers.shape[0]),
                "n_units": int(self.multipliers.shape[1]),
                "coordinate_standard_error": self.coordinate_standard_error,
                "observed_studentized_effect": self.observed_studentized_effect,
                "curve_statistic": self.curve_statistic,
                "experimental_curve_p_value": self.curve_p_value,
                "experimental_global_curve_maxT_p_value": self.global_curve_maxT_p_value,
                "family_ids": self.family_ids,
                "family_statistic": self.family_statistic,
                "experimental_family_maxT_p_value": self.family_maxT_p_value,
                "signed_auc_standard_error": self.signed_auc_standard_error,
                "signed_auc_studentized": self.signed_auc_studentized,
                "experimental_signed_auc_p_value": self.signed_auc_p_value,
                "experimental_signed_auc_global_maxT_p_value": (
                    self.signed_auc_global_maxT_p_value
                ),
                "simultaneous_critical": self.simultaneous_critical,
                "simultaneous_order_index_1based": self.simultaneous_order_index_1based,
                "experimental_simultaneous_lower": self.simultaneous_lower,
                "experimental_simultaneous_upper": self.simultaneous_upper,
                "sensitivity_informative": self.sensitivity_informative,
                "formal_inference_authorized": False,
            }
        )


@dataclass(frozen=True)
class FunctionalCoreV2Result:
    fit: LodoInfluenceFit
    donor_primary: MultiplierProcessResult
    experiment_overlap_sensitivity: MultiplierProcessResult | None
    claim_scope: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": METHOD_ID,
            "schema_version": SCHEMA_VERSION,
            "fit": self.fit.to_dict(),
            "donor_primary": self.donor_primary.to_dict(),
            "experiment_overlap_sensitivity": (
                self.experiment_overlap_sensitivity.to_dict()
                if self.experiment_overlap_sensitivity is not None
                else {
                    "available": False,
                    "sensitivity_informative": False,
                    "reason": "fewer_than_two_experiment_overlap_components",
                    "formal_cluster_inference_authorized": False,
                }
            ),
            "claim_scope": _json_safe(self.claim_scope),
        }


def fit_lodo_donor_influence(
    *,
    outcomes: Any,
    donor_ids: Sequence[Any],
    bin_ids: Sequence[Any],
    pathway_ids: Sequence[Any],
    condition: Any,
    availability: Any,
    experiment_fractions: Any,
    experiment_ids: Sequence[Any],
    experiment_component_ids: Sequence[Any] | None = None,
    family_ids: Sequence[Any] | None = None,
    support_mask: Any = None,
    bin_weights: Any = None,
    rank_tolerance: float = RANK_RELATIVE_TOLERANCE,
    min_donors_per_condition: int = 10,
    min_residual_df: int = 3,
    max_condition_vif: float = 10.0,
) -> LodoInfluenceFit:
    """Fit bin-specific OLS and direct LODO jackknife donor influences.

    The only reused V1 component is the retained design/estimability builder.
    No residual mapping, residual refit, absolute-value endpoint, or V1
    inferential result is called.
    """

    if float(rank_tolerance) != RANK_RELATIVE_TOLERANCE:
        raise ValueError(f"rank_tolerance is frozen at {RANK_RELATIVE_TOLERANCE:.1e}")
    donors, donor_order = _canonical_axis(donor_ids, "donor_ids")
    bins, bin_order = _canonical_axis(bin_ids, "bin_ids")
    pathways, pathway_order = _canonical_axis(pathway_ids, "pathway_ids")
    experiments, experiment_order = _canonical_axis(experiment_ids, "experiment_ids")
    n_donors, n_bins, n_pathways = len(donors), len(bins), len(pathways)

    observed = np.asarray(outcomes, dtype=float)
    expected = (n_donors, n_bins, n_pathways)
    if observed.shape != expected:
        raise ValueError(f"outcomes must have shape {expected}")
    observed = observed[donor_order][:, bin_order][:, :, pathway_order]
    available = _as_binary(availability, (n_donors, n_bins), "availability")
    available = available[donor_order][:, bin_order]
    if not bool(np.isfinite(observed[available]).all()):
        raise ValueError("Available donor-bin outcomes must be finite")
    if not bool(np.isnan(observed[~available]).all()):
        raise ValueError("Unavailable donor-bin outcomes must remain NA")

    c = _as_binary(condition, (n_donors,), "condition")[donor_order]
    fractions = np.asarray(experiment_fractions, dtype=float)
    expected_fractions = (n_donors, n_bins, len(experiments))
    if fractions.shape != expected_fractions:
        raise ValueError(f"experiment_fractions must have shape {expected_fractions}")
    fractions = fractions[donor_order][:, bin_order][:, :, experiment_order]

    derived_components = build_experiment_overlap_components(
        donor_ids=donors,
        availability=available,
        experiment_fractions=fractions,
        experiment_ids=experiments,
    )
    supplied_component_partition_validated = experiment_component_ids is not None
    if experiment_component_ids is not None:
        component_values = np.asarray(
            [str(value).strip() for value in experiment_component_ids], dtype=object
        )
        if component_values.shape != (n_donors,) or any(
            not str(value) for value in component_values
        ):
            raise ValueError(
                "experiment_component_ids must be one nonblank ID per donor"
            )
        component_values = component_values[donor_order]
        if _canonical_partition(component_values) != _canonical_partition(
            derived_components
        ):
            raise FunctionalCoreV2DesignError(
                "Supplied experiment_component_ids do not equal the derived "
                "donor--experiment connected-component partition"
            )
    component_values = np.asarray(derived_components, dtype=object)
    if family_ids is None:
        families = tuple("unassigned" for _ in pathways)
    else:
        if len(family_ids) != n_pathways:
            raise ValueError("family_ids must align one-to-one with pathway_ids")
        family_observed = np.asarray([str(value).strip() for value in family_ids], dtype=object)
        if any(not str(value) for value in family_observed):
            raise ValueError("family_ids must be nonblank")
        families = tuple(str(value) for value in family_observed[pathway_order])

    if support_mask is None:
        support = np.ones((n_bins, n_pathways), dtype=bool)
    else:
        support_observed = np.asarray(support_mask)
        if support_observed.shape == (n_bins,):
            support_observed = np.repeat(support_observed[:, None], n_pathways, axis=1)
        support = _as_binary(
            support_observed, (n_bins, n_pathways), "support_mask"
        )[bin_order][:, pathway_order]
    if not bool(support.any(axis=0).all()):
        raise FunctionalCoreV2DesignError("Every pathway must have supported bins")

    if bin_weights is None:
        base_weights = np.full(n_bins, 1.0 / n_bins, dtype=float)
    else:
        base_weights = np.asarray(bin_weights, dtype=float)
        if (
            base_weights.shape != (n_bins,)
            or not bool(np.isfinite(base_weights).all())
            or bool((base_weights <= 0).any())
        ):
            raise ValueError("bin_weights must be finite, positive, and length n_bins")
        base_weights = base_weights[bin_order]
    pathway_weights = _support_weights(support, base_weights)

    design_plan = build_bin_specific_designs(
        donors,
        c,
        available,
        fractions,
        experiments,
        rank_tolerance=rank_tolerance,
        min_donors_per_condition=min_donors_per_condition,
        min_residual_df=min_residual_df,
        max_condition_vif=max_condition_vif,
    )
    if not design_plan.all_estimable:
        raise FunctionalCoreV2DesignError("One or more observed designs are not estimable")

    effect = np.full((n_bins, n_pathways), np.nan, dtype=float)
    standard_error = np.full_like(effect, np.nan)
    leverage = np.full((n_donors, n_bins), np.nan, dtype=float)
    leave_one_out = np.full((n_donors, n_bins, n_pathways), np.nan, dtype=float)
    influence = np.full_like(leave_one_out, np.nan)
    deletion_nuisance_rank_losses: list[int] = []
    deletion_condition_information: list[float] = []
    deletion_residual_df: list[int] = []
    deletion_failures: list[dict[str, Any]] = []

    for item in design_plan.bins:
        b = int(item.bin_index)
        indices = np.asarray(item.available_donor_indices, dtype=int)
        local_y = observed[indices, b]
        local_condition = c[indices].astype(float)
        observed_weights, local_leverage, _ = _condition_weights_and_leverage(
            item.full_design
        )
        effect[b] = observed_weights @ local_y
        leverage[indices, b] = local_leverage
        local_lodo = np.empty((len(indices), n_pathways), dtype=float)
        for local_index, global_index in enumerate(indices):
            try:
                weights, nuisance_rank, information, residual_df = _deletion_condition_weights(
                    item.reduced_design,
                    local_condition,
                    local_index,
                    minimum_residual_df=int(min_residual_df),
                    minimum_donors_per_condition=1,
                )
            except FunctionalCoreV2DesignError as exc:
                deletion_failures.append(
                    {
                        "bin_index": b,
                        "donor_id": donors[int(global_index)],
                        "reason": str(exc),
                    }
                )
                continue
            deletion_nuisance_rank_losses.append(
                int(item.reduced_rank - nuisance_rank)
            )
            deletion_condition_information.append(float(information))
            deletion_residual_df.append(int(residual_df))
            local_lodo[local_index] = weights @ local_y
        if deletion_failures:
            raise FunctionalCoreV2DesignError(
                "One or more direct LODO refits are not estimable",
                {"deletion_failures": deletion_failures},
            )
        leave_one_out[indices, b] = local_lodo
        lodo_mean = local_lodo.mean(axis=0)
        scale = math.sqrt((len(indices) - 1.0) / len(indices))
        local_influence = scale * (lodo_mean[None, :] - local_lodo)
        influence[indices, b] = local_influence
        standard_error[b] = np.sqrt(np.sum(local_influence**2, axis=0))

    supported_se = standard_error[support]
    if not bool(np.isfinite(supported_se).all()) or bool(
        (supported_se <= NUMERICAL_TOLERANCE).any()
    ):
        raise FunctionalCoreV2DesignError(
            "LODO jackknife standard error is undefined on supported coordinates"
        )
    studentized = effect / standard_error
    effect[~support] = np.nan
    standard_error[~support] = np.nan
    studentized[~support] = np.nan
    influence[:, ~support] = np.nan
    leave_one_out[:, ~support] = np.nan

    safe_influence = np.where(np.isfinite(influence), influence, 0.0)
    signed_influence = np.einsum(
        "dbp,bp->dp", safe_influence, pathway_weights, optimize=True
    )
    signed_auc = np.nansum(effect * pathway_weights, axis=0)
    signed_se = np.sqrt(np.sum(signed_influence**2, axis=0))
    if bool((signed_se <= NUMERICAL_TOLERANCE).any()):
        raise FunctionalCoreV2DesignError("Signed-AUC jackknife SE is undefined")
    signed_t = signed_auc / signed_se

    audit = {
        "influence_correction": "direct_leave_one_donor_out_jackknife",
        "jackknife_centering": "sqrt((n-1)/n)_times_mean_lodo_minus_lodo",
        "lodo_nuisance_rule": "rank_revealing_svd_projection_of_remaining_nuisance_column_span_then_fwl",
        "outcome_or_condition_based_nuisance_column_selection": False,
        "full_model_pseudoinverse_used": False,
        "leverage_clipped": False,
        "maximum_observed_leverage": float(np.nanmax(leverage)),
        "n_observed_leverage_at_or_above_0_99": int(
            np.sum(leverage >= 0.99)
        ),
        "n_lodo_refits": int(np.sum(available)),
        "maximum_lodo_nuisance_rank_loss": int(
            max(deletion_nuisance_rank_losses, default=0)
        ),
        "minimum_lodo_condition_information": float(
            min(deletion_condition_information, default=math.nan)
        ),
        "minimum_lodo_residual_df": int(min(deletion_residual_df, default=-1)),
        "missing_outcome_imputed": False,
        "unavailable_influence_serialized_as_na": True,
        "signed_auc_cross_bin_covariance_included": True,
        "design_builder_reused_for_estimability_only": True,
        "experiment_component_source": (
            "derived_donor_experiment_bipartite_connected_components"
        ),
        "supplied_component_partition_validated": bool(
            supplied_component_partition_validated
        ),
        "experiment_component_count": int(len(set(component_values.tolist()))),
        "experiment_component_donor_sizes_descending": sorted(
            (
                int(np.sum(component_values == component))
                for component in set(component_values.tolist())
            ),
            reverse=True,
        ),
    }
    return LodoInfluenceFit(
        donor_ids=donors,
        bin_ids=bins,
        pathway_ids=pathways,
        experiment_ids=experiments,
        family_ids=families,
        experiment_component_ids=tuple(str(value) for value in component_values),
        availability=_freeze(available, bool),
        support_mask=_freeze(support, bool),
        pathway_bin_weights=_freeze(pathway_weights, float),
        effect=_freeze(effect, float),
        standard_error=_freeze(standard_error, float),
        studentized_effect=_freeze(studentized, float),
        leverage=_freeze(leverage, float),
        leave_one_out_effect=_freeze(leave_one_out, float),
        donor_influence=_freeze(influence, float),
        signed_auc=_freeze(signed_auc, float),
        signed_auc_standard_error=_freeze(signed_se, float),
        signed_auc_studentized=_freeze(signed_t, float),
        signed_auc_donor_influence=_freeze(signed_influence, float),
        design_plan=design_plan,
        design_audit=audit,
    )


def generate_multiplier_stream(
    *, n_draws: int, n_units: int, seed: int, distribution: str
) -> np.ndarray:
    if int(n_draws) < 19 or int(n_units) < 2:
        raise ValueError("Multiplier streams require at least 19 draws and two units")
    rng = np.random.default_rng(int(seed))
    if distribution == "rademacher":
        draws = rng.integers(0, 2, size=(int(n_draws), int(n_units)), dtype=np.int8)
        output = 2.0 * draws.astype(float) - 1.0
    elif distribution == "webb_six_point":
        indices = rng.integers(
            0,
            len(WEBB_SIX_POINT_SUPPORT),
            size=(int(n_draws), int(n_units)),
        )
        output = WEBB_SIX_POINT_SUPPORT[indices]
    else:
        raise ValueError("distribution must be rademacher or webb_six_point")
    return _freeze(output, float)


def _plus_one_p(null: np.ndarray, observed: np.ndarray) -> np.ndarray:
    reference = np.asarray(null, dtype=float)
    target = np.asarray(observed, dtype=float)
    return (
        1.0
        + np.sum(
            reference >= target[None, ...] - NUMERICAL_TOLERANCE,
            axis=0,
        )
    ) / (reference.shape[0] + 1.0)


def _aggregate_influence(
    fit: LodoInfluenceFit, unit_ids: Sequence[str]
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    observed = np.asarray(unit_ids, dtype=object)
    if observed.shape != (len(fit.donor_ids),):
        raise ValueError("unit_ids must align to canonical donors")
    units = tuple(sorted(set(str(value) for value in observed)))
    coordinate = np.zeros(
        (len(units), len(fit.bin_ids), len(fit.pathway_ids)), dtype=float
    )
    signed = np.zeros((len(units), len(fit.pathway_ids)), dtype=float)
    safe_coordinate = np.where(np.isfinite(fit.donor_influence), fit.donor_influence, 0.0)
    for index, unit in enumerate(units):
        members = observed.astype(str) == unit
        coordinate[index] = np.sum(safe_coordinate[members], axis=0)
        signed[index] = np.sum(fit.signed_auc_donor_influence[members], axis=0)
    return units, coordinate, signed


def run_multiplier_process(
    fit: LodoInfluenceFit,
    *,
    unit_type: str,
    unit_ids: Sequence[str],
    distribution: str,
    n_draws: int,
    seed: int,
    alpha: float,
    minimum_informative_units: int,
) -> MultiplierProcessResult:
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must lie strictly between zero and one")
    if unit_type == "donor":
        if distribution != "rademacher" or tuple(unit_ids) != fit.donor_ids:
            raise FunctionalCoreV2DesignError(
                "Donor-primary inference requires one Rademacher unit per canonical donor"
            )
    elif unit_type == "experiment_overlap_connected_component":
        if distribution != "webb_six_point" or _canonical_partition(
            unit_ids
        ) != _canonical_partition(fit.experiment_component_ids):
            raise FunctionalCoreV2DesignError(
                "Experiment sensitivity requires the derived component partition "
                "and Webb six-point multipliers"
            )
    else:
        raise FunctionalCoreV2DesignError("Unknown multiplier independence unit")
    units, coordinate_influence, signed_influence = _aggregate_influence(
        fit, unit_ids
    )
    finite_sample_scale = (
        math.sqrt(len(units) / (len(units) - 1.0))
        if unit_type == "experiment_overlap_connected_component"
        else 1.0
    )
    coordinate_influence = coordinate_influence * finite_sample_scale
    signed_influence = signed_influence * finite_sample_scale
    multipliers = generate_multiplier_stream(
        n_draws=n_draws,
        n_units=len(units),
        seed=seed,
        distribution=distribution,
    )
    coordinate_se = np.sqrt(np.sum(coordinate_influence**2, axis=0))
    signed_se = np.sqrt(np.sum(signed_influence**2, axis=0))
    supported_coordinate_se = coordinate_se[fit.support_mask]
    if not bool(np.isfinite(supported_coordinate_se).all()) or bool(
        (supported_coordinate_se <= NUMERICAL_TOLERANCE).any()
    ):
        raise FunctionalCoreV2DesignError(
            f"{unit_type} multiplier coordinate SE is undefined"
        )
    if not bool(np.isfinite(signed_se).all()) or bool(
        (signed_se <= NUMERICAL_TOLERANCE).any()
    ):
        raise FunctionalCoreV2DesignError(
            f"{unit_type} multiplier signed-AUC SE is undefined"
        )

    coordinate_numerator = np.einsum(
        "ru,ubp->rbp", multipliers, coordinate_influence, optimize=True
    )
    coordinate_draws = np.divide(
        coordinate_numerator,
        coordinate_se[None, :, :],
        out=np.zeros_like(coordinate_numerator),
        where=coordinate_se[None, :, :] > NUMERICAL_TOLERANCE,
    )
    coordinate_draws[:, ~fit.support_mask] = np.nan
    observed_t = fit.effect / coordinate_se
    observed_t[~fit.support_mask] = np.nan

    curve_statistic = np.asarray(
        [
            np.max(np.abs(observed_t[fit.support_mask[:, p], p]))
            for p in range(len(fit.pathway_ids))
        ],
        dtype=float,
    )
    null_curve = np.asarray(
        [
            np.max(
                np.abs(coordinate_draws[:, fit.support_mask[:, p], p]),
                axis=1,
            )
            for p in range(len(fit.pathway_ids))
        ],
        dtype=float,
    ).T
    curve_p = _plus_one_p(null_curve, curve_statistic)
    global_null = np.nanmax(np.abs(coordinate_draws), axis=(1, 2))
    global_p = _plus_one_p(global_null[:, None], curve_statistic)

    family_order = tuple(dict.fromkeys(fit.family_ids))
    family_observed: list[float] = []
    family_null: list[np.ndarray] = []
    for family in family_order:
        members = np.asarray([value == family for value in fit.family_ids])
        family_observed.append(float(np.max(curve_statistic[members])))
        family_null.append(np.max(null_curve[:, members], axis=1))
    family_observed_array = np.asarray(family_observed, dtype=float)
    family_null_array = np.asarray(family_null, dtype=float).T
    family_p = _plus_one_p(family_null_array, family_observed_array)

    signed_numerator = np.einsum(
        "ru,up->rp", multipliers, signed_influence, optimize=True
    )
    signed_draws = signed_numerator / signed_se[None, :]
    signed_t = fit.signed_auc / signed_se
    signed_p = _plus_one_p(np.abs(signed_draws), np.abs(signed_t))
    signed_global_null = np.max(np.abs(signed_draws), axis=1)
    signed_global_p = _plus_one_p(signed_global_null[:, None], np.abs(signed_t))

    order_index = int(
        math.ceil((int(n_draws) + 1) * (1.0 - float(alpha)))
    )
    if order_index > int(n_draws):
        raise FunctionalCoreV2DesignError(
            "Requested simultaneous confidence level is unattainable with "
            "the frozen finite multiplier draw count"
        )
    critical = float(np.sort(global_null)[order_index - 1])
    lower = fit.effect - critical * coordinate_se
    upper = fit.effect + critical * coordinate_se
    lower[~fit.support_mask] = np.nan
    upper[~fit.support_mask] = np.nan
    stream_hash = hashlib.sha256(
        np.ascontiguousarray(multipliers, dtype="<f8").tobytes()
    ).hexdigest()
    return MultiplierProcessResult(
        unit_type=str(unit_type),
        unit_ids=units,
        distribution=str(distribution),
        seed=int(seed),
        multiplier_stream_sha256=stream_hash,
        finite_sample_scale=float(finite_sample_scale),
        multipliers=multipliers,
        coordinate_standard_error=_freeze(coordinate_se, float),
        observed_studentized_effect=_freeze(observed_t, float),
        studentized_draws=_freeze(coordinate_draws, float),
        curve_statistic=_freeze(curve_statistic, float),
        curve_p_value=_freeze(curve_p, float),
        global_curve_maxT_p_value=_freeze(global_p, float),
        family_ids=family_order,
        family_statistic=_freeze(family_observed_array, float),
        family_maxT_p_value=_freeze(family_p, float),
        signed_auc_standard_error=_freeze(signed_se, float),
        signed_auc_studentized=_freeze(signed_t, float),
        signed_auc_studentized_draws=_freeze(signed_draws, float),
        signed_auc_p_value=_freeze(signed_p, float),
        signed_auc_global_maxT_p_value=_freeze(signed_global_p, float),
        simultaneous_critical=critical,
        simultaneous_order_index_1based=order_index,
        simultaneous_lower=_freeze(lower, float),
        simultaneous_upper=_freeze(upper, float),
        sensitivity_informative=len(units) >= int(minimum_informative_units),
    )


def run_functional_core_v2(
    *,
    outcomes: Any,
    donor_ids: Sequence[Any],
    bin_ids: Sequence[Any],
    pathway_ids: Sequence[Any],
    condition: Any,
    availability: Any,
    experiment_fractions: Any,
    experiment_ids: Sequence[Any],
    experiment_component_ids: Sequence[Any] | None = None,
    family_ids: Sequence[Any] | None = None,
    support_mask: Any = None,
    bin_weights: Any = None,
    n_multiplier_draws: int = 999,
    donor_multiplier_seed: int = DEFAULT_DONOR_MULTIPLIER_SEED,
    component_multiplier_seed: int = DEFAULT_COMPONENT_MULTIPLIER_SEED,
    alpha: float = 0.05,
    rank_tolerance: float = RANK_RELATIVE_TOLERANCE,
    min_donors_per_condition: int = 10,
    min_residual_df: int = 3,
    max_condition_vif: float = 10.0,
) -> FunctionalCoreV2Result:
    fit = fit_lodo_donor_influence(
        outcomes=outcomes,
        donor_ids=donor_ids,
        bin_ids=bin_ids,
        pathway_ids=pathway_ids,
        condition=condition,
        availability=availability,
        experiment_fractions=experiment_fractions,
        experiment_ids=experiment_ids,
        experiment_component_ids=experiment_component_ids,
        family_ids=family_ids,
        support_mask=support_mask,
        bin_weights=bin_weights,
        rank_tolerance=rank_tolerance,
        min_donors_per_condition=min_donors_per_condition,
        min_residual_df=min_residual_df,
        max_condition_vif=max_condition_vif,
    )
    donor_primary = run_multiplier_process(
        fit,
        unit_type="donor",
        unit_ids=fit.donor_ids,
        distribution="rademacher",
        n_draws=n_multiplier_draws,
        seed=donor_multiplier_seed,
        alpha=alpha,
        minimum_informative_units=30,
    )
    n_components = len(set(fit.experiment_component_ids))
    overlap = (
        run_multiplier_process(
            fit,
            unit_type="experiment_overlap_connected_component",
            unit_ids=fit.experiment_component_ids,
            distribution="webb_six_point",
            n_draws=n_multiplier_draws,
            seed=component_multiplier_seed,
            alpha=alpha,
            minimum_informative_units=8,
        )
        if n_components >= 2
        else None
    )
    claim_scope = {
        "v2_0_pure_synthetic_implementation_only": True,
        "functional_core_v2_calibrated": False,
        "formal_inference_authorized": False,
        "holdout_500_authorized": False,
        "real_condition_unblinding_authorized": False,
        "injection_recovery_authorized": False,
        "biological_discovery_authorized": False,
        "experiment_overlap_sensitivity_formal_cluster_inference": False,
        "experiment_overlap_sensitivity_available": overlap is not None,
        "experiment_overlap_component_count": n_components,
        "sensitivity_noninformative_if_fewer_than_eight_components": True,
    }
    return FunctionalCoreV2Result(
        fit=fit,
        donor_primary=donor_primary,
        experiment_overlap_sensitivity=overlap,
        claim_scope=claim_scope,
    )


__all__ = [
    "DEFAULT_COMPONENT_MULTIPLIER_SEED",
    "DEFAULT_DONOR_MULTIPLIER_SEED",
    "FunctionalCoreV2DesignError",
    "FunctionalCoreV2Result",
    "LodoInfluenceFit",
    "METHOD_ID",
    "MultiplierProcessResult",
    "NUMERICAL_TOLERANCE",
    "WEBB_SIX_POINT_SUPPORT",
    "build_experiment_overlap_components",
    "fit_lodo_donor_influence",
    "generate_multiplier_stream",
    "run_functional_core_v2",
    "run_multiplier_process",
]
