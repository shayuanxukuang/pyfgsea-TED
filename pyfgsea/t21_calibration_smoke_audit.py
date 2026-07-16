from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from .t21_data_product import sha256_file, stable_json
from .trajectory_precision_freedman_lane import (
    json_ready_precision_diagnostics,
    precision_estimability_diagnostics,
)


AUDIT_SCHEMA_NAME = "t21_calibration_smoke_failure_audit"
AUDIT_SCHEMA_VERSION = "1.0.0"
REQUIRED_ROLES = {
    "report": "t21_pre_unblinding_calibration_report.json",
    "scenario_replicates": "scenario_replicates.tsv",
    "scenario_metrics": "scenario_metrics.tsv",
    "power_replicates": "power_replicates.tsv",
    "power_metrics": "power_metrics.tsv",
    "derived_parameters": "profile_derived_parameters.json",
}


def _mapping(path: Path, *, label: str) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one mapping")
    return value


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("Calibration audit paths must remain inside the repository") from exc


def _input(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "relative_path": _relative(path, root),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.mean())


def recompute_scenario_metrics(replicates: pd.DataFrame) -> pd.DataFrame:
    required = {
        "scenario",
        "any_maxT_false_rejection",
        "by_false_discovery_proportion",
        "onset_false_positive",
        "duration_false_positive",
        "pointwise_curve_coverage_fraction",
        "regulation_false_discovery_proportion",
        "false_timing_shift",
        "trans_false_discovery_proportion",
        "occupancy_signal_detected",
        "fate_signal_detected",
    }
    missing = sorted(required.difference(replicates.columns))
    if missing:
        raise ValueError(f"Scenario replicates are missing columns: {missing}")
    rows = []
    for scenario, group in replicates.groupby("scenario", sort=False):
        rows.append(
            {
                "scenario": str(scenario),
                "n_replicates": int(len(group)),
                "empirical_fwer": _mean(group, "any_maxT_false_rejection"),
                "empirical_fdr": _mean(group, "by_false_discovery_proportion"),
                "onset_false_positive_rate": _mean(group, "onset_false_positive"),
                "duration_false_positive_rate": _mean(
                    group, "duration_false_positive"
                ),
                "pointwise_curve_coverage": _mean(
                    group, "pointwise_curve_coverage_fraction"
                ),
                "regulation_false_discovery_rate": _mean(
                    group, "regulation_false_discovery_proportion"
                ),
                "false_timing_shift_rate": _mean(group, "false_timing_shift"),
                "trans_false_discovery_rate": _mean(
                    group, "trans_false_discovery_proportion"
                ),
                "occupancy_detection_rate": _mean(
                    group, "occupancy_signal_detected"
                ),
                "fate_detection_rate": _mean(group, "fate_signal_detected"),
            }
        )
    return pd.DataFrame(rows)


def _verify_summary(recomputed: pd.DataFrame, declared: pd.DataFrame) -> None:
    if set(recomputed["scenario"]) != set(declared["scenario"]):
        raise ValueError("Scenario summary and raw replicates have different scenarios")
    left = recomputed.set_index("scenario")
    right = declared.set_index("scenario")
    for scenario in left.index:
        for column in left.columns.intersection(right.columns):
            observed = left.loc[scenario, column]
            expected = right.loc[scenario, column]
            if pd.isna(observed) and pd.isna(expected):
                continue
            if column == "n_replicates":
                if int(observed) != int(expected):
                    raise ValueError("Scenario replicate count differs from its summary")
            elif not np.isclose(float(observed), float(expected), atol=1e-12, rtol=0):
                raise ValueError(
                    f"Scenario metric differs from raw rows: {scenario}/{column}"
                )


def _scale_diagnostics(derived: Mapping[str, Any]) -> dict[str, Any]:
    available = np.asarray(derived["primary_draw_available_mask"], dtype=bool)
    counts = np.asarray(derived["primary_draw_cell_count"], dtype=float)
    assignment = np.asarray(derived["observed_assignment"], dtype=bool)
    reduced = np.asarray(derived["support_reduced_design"], dtype=float)
    donor_scale = np.asarray(derived["donor_noise_scale"], dtype=float)
    if (
        available.shape != counts.shape
        or counts.shape[0] != len(assignment)
        or donor_scale.shape != assignment.shape
        or reduced.shape[0] != len(assignment)
        or not np.array_equal(available, counts > 0)
    ):
        raise ValueError("Derived precision arrays are not donor-grid aligned")
    reference = float(derived["primary_draw_median_positive_cell_count"])
    bin_scale = np.full_like(counts, np.nan)
    combined = np.full_like(counts, np.nan)
    bin_scale[available] = np.sqrt(reference / counts[available])
    combined[available] = np.broadcast_to(donor_scale[:, None], counts.shape)[
        available
    ] * bin_scale[available]
    plan = SimpleNamespace(condition=assignment, reduced_design=reduced)
    estimability = precision_estimability_diagnostics(
        plan,
        combined,
        available=available,
        minimum_group_kish_ess=2.0,
        minimum_condition_information=0.10,
        maximum_group_weight_ratio=100.0,
    )

    def group_median(values: np.ndarray, group: bool) -> float:
        selected = np.broadcast_to((assignment == group)[:, None], values.shape)
        return float(np.nanmedian(values[selected & available]))

    return {
        "failed_dgp_variance_rule": (
            "donor_total_cell_precision_multiplied_by_donor_bin_cell_precision"
        ),
        "precision_reference_cell_count": reference,
        "donor_noise_scale_median_control": float(
            np.median(donor_scale[~assignment])
        ),
        "donor_noise_scale_median_case": float(np.median(donor_scale[assignment])),
        "bin_precision_scale_median_control": group_median(bin_scale, False),
        "bin_precision_scale_median_case": group_median(bin_scale, True),
        "combined_scale_median_control": group_median(combined, False),
        "combined_scale_median_case": group_median(combined, True),
        "estimability_diagnostic_threshold_status": (
            "proposed_blind_amendment_diagnostic_not_frozen_publication_policy"
        ),
        "precision_weight_diagnostics": json_ready_precision_diagnostics(
            estimability
        ),
    }


def _timing_diagnostics(
    derived: Mapping[str, Any], runner: Mapping[str, Any], power: pd.DataFrame
) -> dict[str, Any]:
    selected = [int(value) for value in derived["selected_bin_indices"]]
    left = np.asarray(derived["selected_bin_left"], dtype=float)
    right = np.asarray(derived["selected_bin_right"], dtype=float)
    if len(selected) != len(left) or len(left) != len(right) or len(left) < 1:
        raise ValueError("Selected timing support is invalid")
    consecutive = int(runner["inference"]["timing_min_consecutive_windows"])
    minimum_bins = 2 * consecutive + 1
    source_width = float(np.median(right - left))
    minimum_span = minimum_bins * source_width
    observed_span = float(right[-1] - left[0])
    onset = power.loc[power["power_kind"].astype(str).eq("onset")]
    finite_estimates = pd.to_numeric(
        onset["onset_shift_estimate"], errors="coerce"
    ).dropna()
    all_finite_zero = bool(
        len(finite_estimates)
        and np.allclose(finite_estimates, 0.0, atol=1e-12, rtol=0)
    )
    eligible = len(selected) >= minimum_bins and observed_span >= minimum_span
    return {
        "selected_source_bins": selected,
        "selected_bin_count": int(len(selected)),
        "selected_pseudotime_interval": [float(left[0]), float(right[-1])],
        "selected_pseudotime_span": observed_span,
        "minimum_contiguous_bins": minimum_bins,
        "minimum_pseudotime_span": minimum_span,
        "finite_onset_shift_estimates": int(len(finite_estimates)),
        "all_finite_onset_shift_estimates_numerically_zero_at_1e_12": (
            all_finite_zero
        ),
        "timing_claim_eligible": bool(eligible),
        "timing_status": "evaluable" if eligible else "not_evaluable",
        "reason_codes": (
            []
            if eligible
            else [
                "SELECTED_CONTIGUOUS_BIN_COUNT_LT_5",
                "SELECTED_PSEUDOTIME_SPAN_LT_0_25",
                "TWO_POINT_HALF_RISE_ONSET_DEGENERATE",
            ]
        ),
        "allowed_if_all_other_gates_pass": "terminal_window_amplitude_only",
        "forbidden_claims": [
            "onset_shift",
            "duration_change",
            "phase_shift",
            "earlier_or_later_activation",
            "transient_or_sustained_dynamics",
        ],
    }


def _failure_reasons(
    scenarios: pd.DataFrame,
    power: Mapping[str, Any],
    policy: Mapping[str, Any],
    timing: Mapping[str, Any],
    scale: Mapping[str, Any],
) -> list[str]:
    thresholds = policy["acceptance_thresholds"]
    complete = scenarios.set_index("scenario").loc["complete_null"]
    checks = {
        "COMPLETE_NULL_FWER_ABOVE_POLICY": float(complete["empirical_fwer"])
        > float(thresholds["empirical_fwer_max"]),
        "COMPLETE_NULL_FDR_ABOVE_POLICY": float(complete["empirical_fdr"])
        > float(thresholds["empirical_fdr_max"]),
        "COMPLETE_NULL_ONSET_FP_ABOVE_POLICY": float(
            complete["onset_false_positive_rate"]
        )
        > float(thresholds["onset_false_positive_rate_max"]),
        "COMPLETE_NULL_DURATION_FP_ABOVE_POLICY": float(
            complete["duration_false_positive_rate"]
        )
        > float(thresholds["duration_false_positive_rate_max"]),
        "COMPLETE_NULL_COVERAGE_BELOW_POLICY": float(
            complete["pointwise_curve_coverage"]
        )
        < float(thresholds["complete_null_pointwise_curve_coverage_min"]),
        "MDE_RIGHT_CENSORED_ABOVE_FROZEN_GRID": bool(
            power["minimum_detectable_effect_extrapolated"]
        ),
        "TIMING_NOT_EVALUABLE": timing["timing_claim_eligible"] is False,
        "PRECISION_WEIGHTED_DESIGN_NOT_ESTIMABLE": scale[
            "precision_weight_diagnostics"
        ]["estimability_pass"]
        is False,
    }
    return [code for code, failed in checks.items() if failed]


def _payload_sha256(record: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "integrity"}
    return sha256(stable_json(payload).encode("utf-8")).hexdigest()


def build_smoke_failure_audit(
    *,
    smoke_output_dir: str | Path,
    runner_spec_path: str | Path,
    policy_path: str | Path,
    repository_root: str | Path,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(smoke_output_dir).resolve()
    _relative(output, root)
    paths = {role: output / filename for role, filename in REQUIRED_ROLES.items()}
    runner_path = Path(runner_spec_path).resolve()
    policy_file = Path(policy_path).resolve()
    inputs = {role: _input(path, root) for role, path in paths.items()}
    inputs["runner_spec"] = _input(runner_path, root)
    inputs["acceptance_policy"] = _input(policy_file, root)
    module_path = Path(__file__).resolve()
    precision_module = module_path.with_name("trajectory_precision_freedman_lane.py")
    inputs["audit_module"] = _input(module_path, root)
    inputs["precision_kernel"] = _input(precision_module, root)

    report = _mapping(paths["report"], label="smoke report")
    runner = _mapping(runner_path, label="runner spec")
    policy = _mapping(policy_file, label="acceptance policy")
    derived = _mapping(paths["derived_parameters"], label="derived parameters")
    if (
        report.get("calibration_stage") != "smoke"
        or report.get("outcome_blinded") is not True
        or report.get("execution", {}).get("real_pathway_results_read") is not False
        or report.get("publication_minima_satisfied") is not False
    ):
        raise ValueError("Audit requires the intact outcome-blind underpowered smoke report")

    raw = pd.read_csv(paths["scenario_replicates"], sep="\t")
    declared = pd.read_csv(paths["scenario_metrics"], sep="\t")
    power_replicates = pd.read_csv(paths["power_replicates"], sep="\t")
    power_metrics_frame = pd.read_csv(paths["power_metrics"], sep="\t")
    if len(power_metrics_frame) != 1:
        raise ValueError("Power metrics must contain exactly one row")
    recomputed = recompute_scenario_metrics(raw)
    _verify_summary(recomputed, declared)
    power_metrics = power_metrics_frame.iloc[0].to_dict()
    for key, value in report["power_metrics"].items():
        if key in power_metrics and not np.isclose(
            float(power_metrics[key]), float(value), equal_nan=True, atol=1e-12, rtol=0
        ):
            raise ValueError(f"Power metric differs from report: {key}")

    scale = _scale_diagnostics(derived)
    timing = _timing_diagnostics(derived, runner, power_replicates)
    maximum_grid_effect = float(max(runner["power"]["effect_grid_standardized"]))
    mde = {
        "reported_extrapolated_value_for_consistency_only": float(
            power_metrics["minimum_detectable_effect_standardized"]
        ),
        "reported_extrapolated_flag": bool(
            power_metrics["minimum_detectable_effect_extrapolated"]
        ),
        "maximum_frozen_effect_grid": maximum_grid_effect,
        "scientific_status": "right_censored",
        "scientific_report": f">{maximum_grid_effect:g}",
    }
    reasons = _failure_reasons(recomputed, power_metrics, policy, timing, scale)
    record: dict[str, Any] = {
        "schema_name": AUDIT_SCHEMA_NAME,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "created_at_utc": created_at_utc
        or datetime.now(timezone.utc).isoformat(),
        "status": "fail_closed_pre_unblinding_calibration_smoke",
        "outcome_blinded": True,
        "real_pathway_results_read": False,
        "candidate_expression_matrices_read": False,
        "candidate_pathway_artifacts_read": False,
        "screen_authorized": False,
        "final_calibration_authorized": False,
        "unblinding_authorized": False,
        "release_authorized": False,
        "inputs": inputs,
        "profile_binding": {
            "profile_file_sha256": report["design_profile_usage"][
                "profile_file_sha256"
            ],
            "profile_payload_sha256": report["design_profile_usage"][
                "profile_payload_sha256"
            ],
            "derived_parameters_sha256": report["design_profile_usage"][
                "derived_parameters_sha256"
            ],
        },
        "scenario_metrics_recomputed": recomputed.replace({np.nan: None}).to_dict(
            orient="records"
        ),
        "raw_summary_exactly_reproduced": True,
        "precision_failure": scale,
        "timing_eligibility": timing,
        "minimum_detectable_effect": mde,
        "failure_reason_codes": reasons,
        "required_next_action": (
            "blind_method_and_design_amendment_then_new_500_replicate_smoke"
        ),
        "claim_boundary": {
            "maximum_t21_gata1_level": 3.5,
            "terminal_window_amplitude_claim_currently_authorized": False,
            "timing_claim_authorized": False,
        },
    }
    record["integrity"] = {"payload_sha256": _payload_sha256(record)}
    return record


def write_smoke_failure_audit(record: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, target)
    return target


def validate_smoke_failure_audit(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    audit_path = Path(path).resolve()
    record = _mapping(audit_path, label="smoke failure audit")
    if (
        record.get("schema_name") != AUDIT_SCHEMA_NAME
        or record.get("schema_version") != AUDIT_SCHEMA_VERSION
        or record.get("status") != "fail_closed_pre_unblinding_calibration_smoke"
        or record.get("outcome_blinded") is not True
        or record.get("real_pathway_results_read") is not False
        or record.get("screen_authorized") is not False
        or record.get("unblinding_authorized") is not False
        or record.get("integrity", {}).get("payload_sha256") != _payload_sha256(record)
    ):
        raise ValueError("Smoke failure audit has an invalid fail-closed contract")
    for role, binding in record["inputs"].items():
        input_path = (root / binding["relative_path"]).resolve()
        if (
            _relative(input_path, root) != binding["relative_path"]
            or not input_path.is_file()
            or input_path.stat().st_size != int(binding["bytes"])
            or sha256_file(input_path) != binding["sha256"]
        ):
            raise ValueError(f"Smoke failure audit input changed: {role}")
    rebuilt = build_smoke_failure_audit(
        smoke_output_dir=(
            root / record["inputs"]["report"]["relative_path"]
        ).resolve().parent,
        runner_spec_path=root / record["inputs"]["runner_spec"]["relative_path"],
        policy_path=root / record["inputs"]["acceptance_policy"]["relative_path"],
        repository_root=root,
        created_at_utc=str(record["created_at_utc"]),
    )
    if stable_json(rebuilt) != stable_json(record):
        raise ValueError("Smoke failure audit differs from independent recomputation")
    return {
        "status": "pass_fail_closed_smoke_audit_validation",
        "audit_sha256": sha256_file(audit_path),
        "screen_authorized": False,
        "unblinding_authorized": False,
        "failure_reason_codes": list(record["failure_reason_codes"]),
    }
