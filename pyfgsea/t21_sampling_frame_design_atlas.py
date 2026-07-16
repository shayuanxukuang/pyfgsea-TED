from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .t21_covariate_design import build_t21_canonical_donor_design
from .t21_timing_no_go import validate_timing_no_go_file
from .trajectory_precision_freedman_lane import additive_cell_count_precision_scale


SCHEMA_NAME = "t21_terminal_amplitude_sampling_frame_design_atlas_contract"
SCHEMA_VERSION = "1.0.0"
CONTRACT_ID = "t21_terminal_amplitude_sampling_frame_design_atlas_v1"
FROZEN_CONTRACT_PAYLOAD_SHA256 = (
    "fe206ece9529c41342065eccbfc01f0f87f9b9117153803b42e4f8d0923aa007"
)
ATLAS_SCHEMA_NAME = "t21_sampling_frame_design_atlas"
ATLAS_SCHEMA_VERSION = "1.0.0"
EVIDENCE_VERSION_PATTERN = re.compile(r"^v([1-9][0-9]*)$")

H5AD_FILE = "t21_scRNA_analysis_ready_v1.h5ad"
TRAJECTORY_FILE = "t21_trajectory_draws_v1.zarr"
DONOR_DESIGN_FILE = "t21_donor_design_v1.tsv"
BUILD_RECORD_FILE = "t21_trajectory_fate_build_record_v1.json"

EXPECTED_FRAME_IDS = (
    "t21_fetal_liver_cd45_hsc_mpp_to_terminal_v1",
    "t21_fetal_liver_cd235a_neg_erythroid_v1",
    "t21_fetal_liver_cd45_myeloid_v1",
    "t21_fetal_liver_cd45_megakaryocyte_coverage_v1",
)
EXPECTED_FRAME_SPECS = (
    (
        "t21_fetal_liver_cd45_hsc_mpp_to_terminal_v1",
        "cd45",
        1,
        "t21_cd45_dpt_pseudotime_kernel_reference_mapping_v1",
        "lineage_inclusion",
        "is_true",
        (),
    ),
    (
        "t21_fetal_liver_cd235a_neg_erythroid_v1",
        "cd235a",
        2,
        "t21_cd235a_neg_condition_blind_trajectory_v1",
        "analysis_cell_type",
        "in",
        (
            "hsc_mpp",
            "cycling_hsc_mpp",
            "memp",
            "early_erythroid",
            "late_erythroid",
        ),
    ),
    (
        "t21_fetal_liver_cd45_myeloid_v1",
        "cd45",
        3,
        "t21_cd45_dpt_pseudotime_kernel_reference_mapping_v1",
        "analysis_cell_type",
        "in",
        (
            "hsc_mpp",
            "cycling_hsc_mpp",
            "granulocyte_progenitor",
            "monocyte_progenitor",
        ),
    ),
    (
        "t21_fetal_liver_cd45_megakaryocyte_coverage_v1",
        "cd45",
        4,
        "t21_cd45_dpt_pseudotime_kernel_reference_mapping_v1",
        "analysis_cell_type",
        "in",
        ("hsc_mpp", "cycling_hsc_mpp", "memp", "megakaryocyte"),
    ),
)

ATLAS_COLUMNS = (
    "frame_id",
    "display_name",
    "source_key",
    "rank_order",
    "input_status",
    "structural_input_pass",
    "outcome_blind_input_pass",
    "trajectory_plan_id",
    "preflight_draw_id",
    "n_donors",
    "n_controls",
    "n_cases",
    "n_cells_selected",
    "n_trajectory_draws",
    "n_grid_bins",
    "common_supported_bins",
    "terminal_window_start",
    "terminal_window_end",
    "terminal_window_contiguous_bins",
    "terminal_window_pseudotime_span",
    "terminal_support_gate_pass",
    "maximum_timing_contiguous_bins_record_only",
    "maximum_timing_pseudotime_span_record_only",
    "timing_design_gate_record_only",
    "timing_authorized",
    "minimum_control_support_selected",
    "minimum_case_support_selected",
    "minimum_loco_control_support_selected",
    "complete_window_controls",
    "complete_window_cases",
    "unweighted_control_ess_minimum",
    "unweighted_case_ess_minimum",
    "unweighted_condition_information_minimum",
    "precision_reference_count",
    "precision_control_ess_minimum",
    "precision_case_ess_minimum",
    "precision_condition_information_minimum",
    "precision_group_weight_ratio_maximum",
    "precision_sensitivity_gate_pass",
    "analytic_standardized_mde_family_adjusted",
    "analytic_mde_gate_pass",
    "design_preflight_pass",
    "selection_uses_pathway_outcomes",
    "reason_codes",
    "precision_fraction_diagnostics_json",
    "draw_terminal_suffix_bins_json",
)


@dataclass(frozen=True)
class StructuralFrameData:
    source_key: str
    candidate_dir: Path
    obs: pd.DataFrame
    donor_ids: tuple[str, ...]
    draw_ids: tuple[str, ...]
    primary_draw_id: str
    bin_left: np.ndarray
    bin_right: np.ndarray
    pseudotime: np.ndarray
    stored_cell_count: np.ndarray
    donor_metadata: pd.DataFrame
    trajectory_plan_id: str
    build_record: Mapping[str, Any]
    input_hashes: Mapping[str, str]


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def validate_evidence_version(value: str) -> str:
    """Validate an append-only evidence label, independent of the method contract."""

    normalized = str(value).strip()
    if EVIDENCE_VERSION_PATTERN.fullmatch(normalized) is None:
        raise ValueError("Evidence version must use vN with N >= 1")
    return normalized


def _contract_payload_sha256(contract: Mapping[str, Any]) -> str:
    payload = {
        key: value for key, value in contract.items() if not str(key).startswith("_")
    }
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def _require_exact(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise ValueError(f"Frozen contract mismatch for {label}: {value!r}")


def validate_design_atlas_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate all non-negotiable values in the one allowed blind amendment."""

    _require_exact(contract.get("schema_name"), SCHEMA_NAME, "schema_name")
    _require_exact(contract.get("schema_version"), SCHEMA_VERSION, "schema_version")
    _require_exact(contract.get("contract_id"), CONTRACT_ID, "contract_id")
    _require_exact(contract.get("outcome_blinded_at_freeze"), True, "outcome_blinded")
    _require_exact(contract.get("real_pathway_outcomes_read"), False, "outcomes_read")
    _require_exact(
        contract.get("selection_uses_pathway_outcomes"),
        False,
        "selection_uses_pathway_outcomes",
    )

    boundary = contract.get("decision_boundary", {})
    _require_exact(boundary.get("timing_decision"), "no_go", "timing_decision")
    _require_exact(
        boundary.get("timing_reopening_allowed"), False, "timing_reopening_allowed"
    )
    _require_exact(
        boundary.get("timing_minimum_contiguous_bins_record_only"),
        5,
        "timing_minimum_contiguous_bins",
    )
    _require_exact(
        float(boundary.get("timing_minimum_pseudotime_span_record_only", np.nan)),
        0.25,
        "timing_minimum_pseudotime_span",
    )

    amendment = contract.get("method_amendment", {})
    _require_exact(
        amendment.get("maximum_blind_method_amendments"), 1, "amendment_count"
    )
    _require_exact(amendment.get("primary_weighting"), "donor_equal", "weighting")
    _require_exact(
        amendment.get("precision_standardized_role"),
        "sensitivity_only",
        "precision_role",
    )

    support = contract.get("fixed_grid_support", {})
    required_support = {
        "minimum_cells_per_donor_bin": 5,
        "minimum_controls_with_support_per_bin": 3,
        "minimum_cases_with_support_per_bin": 10,
        "minimum_controls_after_leave_one_control_out_per_bin": 2,
        "required_trajectory_draw_policy": "frozen_primary_draw",
        "robustness_draws_reported_without_adaptive_selection": True,
        "terminal_window_selection": "longest_common_supported_contiguous_suffix",
        "terminal_window_required_right_edge": 1.0,
        "terminal_window_minimum_contiguous_bins": 2,
        "terminal_window_minimum_pseudotime_span": 0.10,
        "complete_window_donors_required_for_scalar_estimand": True,
    }
    for key, expected in required_support.items():
        _require_exact(support.get(key), expected, f"fixed_grid_support.{key}")

    precision = contract.get("bounded_additive_precision_sensitivity", {})
    _require_exact(
        [
            float(value)
            for value in precision.get("biological_variance_fraction_grid", [])
        ],
        [0.50, 0.75, 0.90],
        "biological_variance_fraction_grid",
    )
    for key, expected in {
        "reference_count_rule": (
            "median_positive_donor_bin_count_in_selected_terminal_window"
        ),
        "maximum_residual_standard_deviation_scale": 2.0,
        "minimum_control_kish_ess": 2.0,
        "minimum_case_kish_ess": 8.0,
        "minimum_condition_information": 0.10,
        "maximum_group_total_weight_ratio": 10.0,
        "every_fraction_and_required_draw_must_pass": True,
    }.items():
        _require_exact(precision.get(key), expected, f"precision.{key}")

    mde = contract.get("analytic_mde", {})
    for key, expected in {
        "method": "covariate_adjusted_noncentral_t",
        "two_sided_alpha": 0.05,
        "target_power": 0.80,
        "family_adjustment": "bonferroni",
        "number_of_primary_pathway_families": 13,
        "maximum_standardized_mde": 0.80,
    }.items():
        _require_exact(mde.get(key), expected, f"analytic_mde.{key}")

    smoke = contract.get("smoke_500", {})
    _require_exact(smoke.get("maximum_runs_after_this_amendment"), 1, "smoke runs")
    _require_exact(smoke.get("replicates"), 500, "smoke replicates")
    _require_exact(smoke.get("default_allowed"), False, "smoke default")

    firewall = contract.get("input_firewall", {})
    _require_exact(
        set(firewall.get("permitted_artifact_files", [])),
        {DONOR_DESIGN_FILE, H5AD_FILE, TRAJECTORY_FILE, BUILD_RECORD_FILE},
        "permitted artifacts",
    )
    permitted_obs = set(firewall.get("permitted_h5ad_obs_columns", []))
    if permitted_obs != {
        "cell_id",
        "donor_id",
        "analysis_cell_type",
        "lineage_inclusion",
    }:
        raise ValueError("Frozen H5AD obs-column firewall was changed")

    frames = contract.get("sampling_frames", [])
    frame_ids = tuple(str(frame.get("frame_id")) for frame in frames)
    _require_exact(frame_ids, EXPECTED_FRAME_IDS, "sampling frame order")
    if [int(frame.get("rank_order", -1)) for frame in frames] != [1, 2, 3, 4]:
        raise ValueError("Sampling-frame rank order must be exactly 1 through 4")
    for frame in frames:
        cell_filter = frame.get("cell_filter", {})
        if cell_filter.get("column") not in permitted_obs:
            raise ValueError("Sampling-frame filter bypasses the H5AD input firewall")
        if cell_filter.get("operator") not in {"is_true", "in"}:
            raise ValueError("Sampling-frame filter operator is not frozen")
        if not str(frame.get("expected_trajectory_plan_id", "")).strip():
            raise ValueError("Every sampling frame needs a trajectory plan binding")
    observed_specs = tuple(
        (
            str(frame["frame_id"]),
            str(frame["source_key"]),
            int(frame["rank_order"]),
            str(frame["expected_trajectory_plan_id"]),
            str(frame["cell_filter"]["column"]),
            str(frame["cell_filter"]["operator"]),
            tuple(str(value) for value in frame["cell_filter"].get("values", [])),
        )
        for frame in frames
    )
    _require_exact(observed_specs, EXPECTED_FRAME_SPECS, "sampling frame definitions")
    return dict(contract)


def load_design_atlas_contract(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    value = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("Design-atlas contract must be a YAML mapping")
    observed_sha256 = _contract_payload_sha256(value)
    if observed_sha256 != FROZEN_CONTRACT_PAYLOAD_SHA256:
        raise ValueError("Design-atlas contract payload differs from the frozen SHA256")
    result = validate_design_atlas_contract(value)
    result["_contract_payload_sha256"] = observed_sha256
    return result


def _read_structural_obs(path: Path, permitted_columns: Sequence[str]) -> pd.DataFrame:
    """Read only allowlisted cell labels from a backed H5AD; never read X/layers."""

    import anndata as ad

    adata = ad.read_h5ad(path, backed="r")
    try:
        missing = sorted(set(permitted_columns).difference(adata.obs.columns))
        if missing:
            raise ValueError(f"H5AD is missing structural obs columns: {missing}")
        obs = adata.obs.loc[:, list(permitted_columns)].copy()
        obs.index = obs.index.astype(str)
    finally:
        adata.file.close()
    if obs.index.duplicated().any():
        raise ValueError("H5AD cell identifiers are not unique")
    if not obs["cell_id"].astype(str).to_numpy().tolist() == obs.index.tolist():
        raise ValueError("H5AD cell_id does not equal the structural observation index")
    return obs


def _validate_outcome_blind_build_record(record: Mapping[str, Any]) -> None:
    required_false = (
        "read_pathway_result_artifacts",
        "used_candidate_pathway_genes",
        "used_condition_information_for_inference",
    )
    for key in required_false:
        if record.get(key) is not False:
            raise ValueError(f"Structural build record does not prove {key}=false")


def read_structural_frame_source(
    source_key: str,
    candidate_dir: str | Path,
    contract: Mapping[str, Any],
) -> StructuralFrameData:
    """Load only donor metadata, cell labels, and trajectory structural arrays."""

    import zarr

    root = Path(candidate_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Frame source directory is absent: {root}")
    paths = {
        "h5ad": root / H5AD_FILE,
        "trajectory": root / TRAJECTORY_FILE,
        "donor_design": root / DONOR_DESIGN_FILE,
        "build_record": root / BUILD_RECORD_FILE,
    }
    missing = sorted(str(path) for path in paths.values() if not path.exists())
    if missing:
        raise FileNotFoundError(f"Structural frame artifacts are absent: {missing}")
    if not paths["trajectory"].is_dir():
        raise ValueError("Trajectory structural artifact must be a Zarr directory")

    record = json.loads(paths["build_record"].read_text(encoding="utf-8"))
    if not isinstance(record, Mapping):
        raise ValueError("Trajectory/fate build record must be a JSON object")
    _validate_outcome_blind_build_record(record)

    permitted = contract["input_firewall"]["permitted_h5ad_obs_columns"]
    obs = _read_structural_obs(paths["h5ad"], permitted)
    donor_metadata = pd.read_csv(paths["donor_design"], sep="\t", low_memory=False)
    required_donor = {"donor_id", "condition", "pcw"}
    if not required_donor.issubset(donor_metadata.columns):
        raise ValueError("Donor metadata lacks donor_id, condition, or pcw")
    if donor_metadata["donor_id"].astype(str).duplicated().any():
        raise ValueError("Donor metadata contains duplicate donor IDs")

    group = zarr.open_group(paths["trajectory"], mode="r")
    required_arrays = (
        "axes/cell_id",
        "axes/donor_id",
        "axes/trajectory_draw_id",
        "axes/bin_left",
        "axes/bin_right",
        "pseudotime",
        "donor_bin/cell_count",
    )
    if any(name not in group for name in required_arrays):
        raise ValueError("Trajectory Zarr lacks a required structural array")
    cell_ids = np.asarray(group["axes/cell_id"][:], dtype=str)
    if not np.array_equal(cell_ids, obs.index.to_numpy(dtype=str)):
        raise ValueError("Trajectory and H5AD cell order differ")
    donor_ids = tuple(str(value) for value in group["axes/donor_id"][:])
    if len(set(donor_ids)) != len(donor_ids):
        raise ValueError("Trajectory donor axis is not unique")
    draw_ids = tuple(str(value) for value in group["axes/trajectory_draw_id"][:])
    primary_draw_id = str(group.attrs.get("primary_trajectory_draw_id", ""))
    if primary_draw_id not in draw_ids:
        raise ValueError("Trajectory Zarr lacks its frozen primary draw binding")
    left = np.asarray(group["axes/bin_left"][:], dtype=float)
    right = np.asarray(group["axes/bin_right"][:], dtype=float)
    pseudotime = np.asarray(group["pseudotime"][:], dtype=float)
    stored_counts = np.asarray(group["donor_bin/cell_count"][:], dtype=np.int64)
    if pseudotime.shape != (len(obs), len(draw_ids)):
        raise ValueError("Trajectory pseudotime axes are malformed")
    if stored_counts.shape != (len(donor_ids), len(left), len(draw_ids)):
        raise ValueError("Stored donor-bin cell counts are malformed")
    if len(left) < 2 or len(left) != len(right):
        raise ValueError("Trajectory fixed grid is malformed")
    if not (
        np.isclose(left[0], 0.0)
        and np.isclose(right[-1], 1.0)
        and np.allclose(left[1:], right[:-1])
        and np.all(right > left)
    ):
        raise ValueError("Trajectory fixed grid must be contiguous from zero to one")

    observed_donors = set(obs["donor_id"].astype(str))
    if observed_donors != set(donor_ids):
        raise ValueError("H5AD and trajectory donor sets differ")
    design_donors = set(donor_metadata["donor_id"].astype(str))
    if not set(donor_ids).issubset(design_donors):
        raise ValueError("Trajectory donors are absent from donor metadata")

    input_hashes = {
        H5AD_FILE: _sha256_file(paths["h5ad"]),
        DONOR_DESIGN_FILE: _sha256_file(paths["donor_design"]),
        BUILD_RECORD_FILE: _sha256_file(paths["build_record"]),
        TRAJECTORY_FILE: str(
            record.get("contracts", {})
            .get("trajectory", {})
            .get("tree_digest_sha256", "")
        ),
    }
    if len(input_hashes[TRAJECTORY_FILE]) != 64:
        raise ValueError("Build record lacks the trajectory tree digest")
    return StructuralFrameData(
        source_key=str(source_key),
        candidate_dir=root.resolve(),
        obs=obs,
        donor_ids=donor_ids,
        draw_ids=draw_ids,
        primary_draw_id=primary_draw_id,
        bin_left=left,
        bin_right=right,
        pseudotime=pseudotime,
        stored_cell_count=stored_counts,
        donor_metadata=donor_metadata,
        trajectory_plan_id=str(group.attrs.get("plan_id", "")),
        build_record=record,
        input_hashes=input_hashes,
    )


def _cell_filter_mask(obs: pd.DataFrame, spec: Mapping[str, Any]) -> np.ndarray:
    column = str(spec["column"])
    operator = str(spec["operator"])
    if operator == "is_true":
        values = obs[column]
        if pd.api.types.is_bool_dtype(values.dtype):
            return values.fillna(False).to_numpy(dtype=bool)
        normalized = values.astype(str).str.strip().str.lower()
        return normalized.isin({"true", "1", "yes"}).to_numpy(dtype=bool)
    if operator == "in":
        allowed = {str(value) for value in spec.get("values", [])}
        if not allowed:
            raise ValueError("An in-filter must contain predeclared labels")
        return obs[column].astype(str).isin(allowed).to_numpy(dtype=bool)
    raise ValueError("Unsupported frozen cell-filter operator")


def _recompute_donor_bin_counts(
    data: StructuralFrameData, cell_mask: np.ndarray
) -> np.ndarray:
    if cell_mask.shape != (len(data.obs),):
        raise ValueError("Cell-filter mask is not H5AD aligned")
    donor_lookup = {value: index for index, value in enumerate(data.donor_ids)}
    donor_index = np.asarray(
        [donor_lookup[value] for value in data.obs["donor_id"].astype(str)], dtype=int
    )
    edges = np.concatenate(([data.bin_left[0]], data.bin_right))
    counts = np.zeros(
        (len(data.donor_ids), len(data.bin_left), len(data.draw_ids)), dtype=np.int64
    )
    for draw_index in range(len(data.draw_ids)):
        values = data.pseudotime[:, draw_index]
        selected = cell_mask & np.isfinite(values)
        bins = np.searchsorted(edges, values[selected], side="right") - 1
        bins = np.clip(bins, 0, len(data.bin_left) - 1)
        np.add.at(counts[:, :, draw_index], (donor_index[selected], bins), 1)
    return counts


def _aligned_design(
    data: StructuralFrameData,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    indexed = data.donor_metadata.assign(
        donor_id=data.donor_metadata["donor_id"].astype(str)
    ).set_index("donor_id", drop=False)
    rows = indexed.loc[list(data.donor_ids)].copy()
    if rows[["condition", "pcw"]].isna().any().any():
        raise ValueError("Trajectory donor metadata has missing design values")
    technical_batch = (
        rows["technical_batch"].astype(str)
        if "technical_batch" in rows
        else pd.Series("not_resolved", index=rows.index, dtype=object)
    )
    canonical = build_t21_canonical_donor_design(
        donor_ids=rows["donor_id"],
        conditions=rows["condition"],
        pcw=pd.to_numeric(rows["pcw"], errors="raise"),
        technical_batch=technical_batch,
        control="disomy",
        case="T21",
        expected_primary_batch_status=None,
    )
    order = [str(value) for value in canonical.donor_frame["donor"]]
    canonical_index = {donor: index for index, donor in enumerate(order)}
    take = np.asarray([canonical_index[donor] for donor in data.donor_ids], dtype=int)
    reduced = np.asarray(canonical.reduced_design, dtype=float)[take]
    condition = np.asarray(canonical.donor_frame["observed_case"], dtype=bool)[take]
    return condition, reduced, canonical.audit_manifest()


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    best_start = 0
    best_length = 0
    start = 0
    for index, value in enumerate(np.r_[np.asarray(mask, dtype=bool), False]):
        if value:
            continue
        length = index - start
        if length > best_length:
            best_start, best_length = start, length
        start = index + 1
    return best_start, best_length


def _terminal_suffix_length(mask: np.ndarray) -> int:
    values = np.asarray(mask, dtype=bool)
    if not len(values) or not values[-1]:
        return 0
    return int(np.argmax(~values[::-1])) if (~values[::-1]).any() else len(values)


def _weighted_design_metrics(
    condition: np.ndarray,
    reduced_design: np.ndarray,
    available: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    indices = np.flatnonzero(available)
    if not len(indices):
        return {
            "control_ess": 0.0,
            "case_ess": 0.0,
            "condition_information": 0.0,
            "group_weight_ratio": float("inf"),
        }
    local_condition = condition[indices]
    local_weights = np.asarray(weights, dtype=float)[indices]
    ess: dict[bool, float] = {}
    sums: dict[bool, float] = {}
    for group in (False, True):
        local = local_weights[local_condition == group]
        sums[group] = float(local.sum())
        ess[group] = (
            float(local.sum() ** 2 / np.sum(local**2))
            if len(local) and float(np.sum(local**2)) > 0
            else 0.0
        )
    ratio = (
        max(sums.values()) / min(sums.values())
        if min(sums.values()) > 0
        else float("inf")
    )
    square_root_weight = np.sqrt(local_weights)
    z = reduced_design[indices] * square_root_weight[:, None]
    c = local_condition.astype(float) * square_root_weight
    residual = c - z @ (np.linalg.pinv(z) @ c)
    return {
        "control_ess": ess[False],
        "case_ess": ess[True],
        "condition_information": float(residual @ residual),
        "group_weight_ratio": float(ratio),
    }


def _noncentral_t_required_ncp(df: int, alpha: float, power: float) -> float:
    from scipy.stats import nct, t

    if df < 1:
        return float("inf")
    critical = float(t.ppf(1.0 - alpha / 2.0, df))

    def achieved(ncp: float) -> float:
        return float(nct.cdf(-critical, df, ncp) + nct.sf(critical, df, ncp))

    lower, upper = 0.0, 1.0
    while achieved(upper) < power and upper < 4096.0:
        upper *= 2.0
    if achieved(upper) < power:
        return float("inf")
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if achieved(midpoint) >= power:
            upper = midpoint
        else:
            lower = midpoint
    return float(upper)


def _analytic_mde(
    condition: np.ndarray,
    reduced_design: np.ndarray,
    complete_window: np.ndarray,
    contract: Mapping[str, Any],
) -> float:
    indices = np.flatnonzero(complete_window)
    if len(indices) < 3 or len(np.unique(condition[indices])) != 2:
        return float("inf")
    metrics = _weighted_design_metrics(
        condition,
        reduced_design,
        complete_window,
        np.ones(len(condition), dtype=float),
    )
    information = metrics["condition_information"]
    if not np.isfinite(information) or information <= np.finfo(float).eps:
        return float("inf")
    z = reduced_design[indices]
    c = condition[indices].astype(float)[:, None]
    full_rank = int(np.linalg.matrix_rank(np.column_stack((z, c))))
    df = int(len(indices) - full_rank)
    mde = contract["analytic_mde"]
    adjusted_alpha = float(mde["two_sided_alpha"]) / int(
        mde["number_of_primary_pathway_families"]
    )
    ncp = _noncentral_t_required_ncp(df, adjusted_alpha, float(mde["target_power"]))
    return float(ncp / np.sqrt(information))


def _precision_diagnostics(
    counts: np.ndarray,
    selected_bins: np.ndarray,
    draw_indices: Sequence[int],
    condition: np.ndarray,
    reduced_design: np.ndarray,
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, float], bool]:
    precision = contract["bounded_additive_precision_sensitivity"]
    minimum_cells = int(contract["fixed_grid_support"]["minimum_cells_per_donor_bin"])
    if not len(selected_bins):
        empty = {
            "control_ess": 0.0,
            "case_ess": 0.0,
            "condition_information": 0.0,
            "group_weight_ratio": float("inf"),
            "reference_count": float("nan"),
        }
        return {}, empty, False
    selected_counts = counts[:, selected_bins, :]
    positive = selected_counts[selected_counts > 0]
    if not len(positive):
        empty = {
            "control_ess": 0.0,
            "case_ess": 0.0,
            "condition_information": 0.0,
            "group_weight_ratio": float("inf"),
            "reference_count": float("nan"),
        }
        return {}, empty, False
    reference = float(np.median(positive))
    by_fraction: dict[str, Any] = {}
    aggregate = {
        "control_ess": float("inf"),
        "case_ess": float("inf"),
        "condition_information": float("inf"),
        "group_weight_ratio": 0.0,
        "reference_count": reference,
    }
    all_pass = True
    for fraction in precision["biological_variance_fraction_grid"]:
        metrics_list: list[dict[str, float]] = []
        for draw_index in draw_indices:
            local_counts = counts[:, selected_bins, draw_index]
            local_available = local_counts >= minimum_cells
            scales = additive_cell_count_precision_scale(
                local_counts,
                available=local_available,
                biological_variance_fraction=float(fraction),
                reference_count=reference,
                maximum_scale=float(
                    precision["maximum_residual_standard_deviation_scale"]
                ),
            )
            for local_bin in range(len(selected_bins)):
                available = local_available[:, local_bin]
                weights = np.zeros(len(condition), dtype=float)
                weights[available] = 1.0 / scales[available, local_bin] ** 2
                metrics_list.append(
                    _weighted_design_metrics(
                        condition, reduced_design, available, weights
                    )
                )
        summary = {
            "minimum_control_kish_ess": float(
                min(value["control_ess"] for value in metrics_list)
            ),
            "minimum_case_kish_ess": float(
                min(value["case_ess"] for value in metrics_list)
            ),
            "minimum_condition_information": float(
                min(value["condition_information"] for value in metrics_list)
            ),
            "maximum_group_total_weight_ratio": float(
                max(value["group_weight_ratio"] for value in metrics_list)
            ),
        }
        summary["pass"] = bool(
            summary["minimum_control_kish_ess"]
            >= float(precision["minimum_control_kish_ess"])
            and summary["minimum_case_kish_ess"]
            >= float(precision["minimum_case_kish_ess"])
            and summary["minimum_condition_information"]
            >= float(precision["minimum_condition_information"])
            and summary["maximum_group_total_weight_ratio"]
            <= float(precision["maximum_group_total_weight_ratio"])
        )
        by_fraction[f"{float(fraction):.2f}"] = summary
        aggregate["control_ess"] = min(
            aggregate["control_ess"], summary["minimum_control_kish_ess"]
        )
        aggregate["case_ess"] = min(
            aggregate["case_ess"], summary["minimum_case_kish_ess"]
        )
        aggregate["condition_information"] = min(
            aggregate["condition_information"],
            summary["minimum_condition_information"],
        )
        aggregate["group_weight_ratio"] = max(
            aggregate["group_weight_ratio"],
            summary["maximum_group_total_weight_ratio"],
        )
        all_pass = all_pass and bool(summary["pass"])
    return by_fraction, aggregate, bool(all_pass)


def _unavailable_row(frame: Mapping[str, Any], reason: str) -> dict[str, Any]:
    row: dict[str, Any] = {column: "" for column in ATLAS_COLUMNS}
    row.update(
        {
            "frame_id": str(frame["frame_id"]),
            "display_name": str(frame["display_name"]),
            "source_key": str(frame["source_key"]),
            "rank_order": int(frame["rank_order"]),
            "input_status": "unavailable",
            "structural_input_pass": False,
            "outcome_blind_input_pass": False,
            "preflight_draw_id": "",
            "n_donors": 0,
            "n_controls": 0,
            "n_cases": 0,
            "n_cells_selected": 0,
            "n_trajectory_draws": 0,
            "n_grid_bins": 0,
            "common_supported_bins": 0,
            "terminal_window_contiguous_bins": 0,
            "terminal_window_pseudotime_span": 0.0,
            "terminal_support_gate_pass": False,
            "maximum_timing_contiguous_bins_record_only": 0,
            "maximum_timing_pseudotime_span_record_only": 0.0,
            "timing_design_gate_record_only": False,
            "timing_authorized": False,
            "minimum_control_support_selected": 0,
            "minimum_case_support_selected": 0,
            "minimum_loco_control_support_selected": 0,
            "complete_window_controls": 0,
            "complete_window_cases": 0,
            "unweighted_control_ess_minimum": 0.0,
            "unweighted_case_ess_minimum": 0.0,
            "unweighted_condition_information_minimum": 0.0,
            "precision_sensitivity_gate_pass": False,
            "analytic_standardized_mde_family_adjusted": "not_evaluable",
            "analytic_mde_gate_pass": False,
            "design_preflight_pass": False,
            "selection_uses_pathway_outcomes": False,
            "reason_codes": reason,
            "precision_fraction_diagnostics_json": "{}",
            "draw_terminal_suffix_bins_json": "[]",
        }
    )
    return row


def evaluate_sampling_frame(
    frame: Mapping[str, Any],
    data: StructuralFrameData,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    expected_plan = str(frame["expected_trajectory_plan_id"])
    if data.trajectory_plan_id != expected_plan:
        reasons.append("TRAJECTORY_PLAN_BINDING_MISMATCH")
    cell_mask = _cell_filter_mask(data.obs, frame["cell_filter"])
    counts = _recompute_donor_bin_counts(data, cell_mask)
    if frame["cell_filter"]["operator"] == "is_true" and not np.array_equal(
        counts, data.stored_cell_count
    ):
        reasons.append("RECOMPUTED_FULL_FRAME_COUNTS_MISMATCH")

    condition, reduced, _ = _aligned_design(data)
    primary_draw_index = data.draw_ids.index(data.primary_draw_id)
    support = contract["fixed_grid_support"]
    available = counts >= int(support["minimum_cells_per_donor_bin"])
    control_support = np.sum(available[~condition], axis=0)
    case_support = np.sum(available[condition], axis=0)
    loco_control = np.maximum(control_support - 1, 0)
    bin_pass = (
        (control_support >= int(support["minimum_controls_with_support_per_bin"]))
        & (case_support >= int(support["minimum_cases_with_support_per_bin"]))
        & (
            loco_control
            >= int(support["minimum_controls_after_leave_one_control_out_per_bin"])
        )
    )
    # The draw was frozen by the trajectory plan before pathway outcomes existed.
    # Other draws are reported below, but may not be searched to rescue a frame.
    common_pass = bin_pass[:, primary_draw_index]
    common_supported_bins = int(np.sum(common_pass))
    terminal_bins = _terminal_suffix_length(common_pass)
    selected_bins = (
        np.arange(len(common_pass) - terminal_bins, len(common_pass), dtype=int)
        if terminal_bins
        else np.asarray([], dtype=int)
    )
    terminal_span = (
        float(data.bin_right[selected_bins[-1]] - data.bin_left[selected_bins[0]])
        if len(selected_bins)
        else 0.0
    )
    terminal_right = (
        float(data.bin_right[selected_bins[-1]]) if len(selected_bins) else float("nan")
    )
    terminal_gate = bool(
        len(selected_bins) >= int(support["terminal_window_minimum_contiguous_bins"])
        and terminal_span >= float(support["terminal_window_minimum_pseudotime_span"])
        and np.isclose(
            terminal_right,
            float(support["terminal_window_required_right_edge"]),
            atol=1e-12,
            rtol=0,
        )
    )
    if not terminal_gate:
        reasons.append("TERMINAL_WINDOW_SUPPORT_GATE_FAILED")

    timing_start, timing_bins = _longest_true_run(common_pass)
    timing_span = (
        float(
            data.bin_right[timing_start + timing_bins - 1] - data.bin_left[timing_start]
        )
        if timing_bins
        else 0.0
    )
    boundary = contract["decision_boundary"]
    timing_design_gate = bool(
        timing_bins >= int(boundary["timing_minimum_contiguous_bins_record_only"])
        and timing_span >= float(boundary["timing_minimum_pseudotime_span_record_only"])
    )

    draw_suffix = [
        _terminal_suffix_length(bin_pass[:, draw]) for draw in range(bin_pass.shape[1])
    ]
    if len(selected_bins):
        selected_control = control_support[selected_bins, primary_draw_index]
        selected_case = case_support[selected_bins, primary_draw_index]
        selected_loco = loco_control[selected_bins, primary_draw_index]
        min_control = int(np.min(selected_control))
        min_case = int(np.min(selected_case))
        min_loco = int(np.min(selected_loco))
        unweighted_metrics = [
            _weighted_design_metrics(
                condition,
                reduced,
                available[:, bin_index, draw_index],
                np.ones(len(condition), dtype=float),
            )
            for draw_index in (primary_draw_index,)
            for bin_index in selected_bins
        ]
        unweighted_control_ess = min(
            value["control_ess"] for value in unweighted_metrics
        )
        unweighted_case_ess = min(value["case_ess"] for value in unweighted_metrics)
        unweighted_information = min(
            value["condition_information"] for value in unweighted_metrics
        )
        complete_window = np.all(
            available[:, selected_bins, primary_draw_index], axis=1
        )
    else:
        min_control = min_case = min_loco = 0
        unweighted_control_ess = unweighted_case_ess = unweighted_information = 0.0
        complete_window = np.zeros(len(condition), dtype=bool)
    complete_controls = int(np.sum(complete_window & ~condition))
    complete_cases = int(np.sum(complete_window & condition))
    if complete_controls < int(support["minimum_controls_with_support_per_bin"]):
        reasons.append("COMPLETE_WINDOW_CONTROL_DONORS_BELOW_MINIMUM")
    if complete_cases < int(support["minimum_cases_with_support_per_bin"]):
        reasons.append("COMPLETE_WINDOW_CASE_DONORS_BELOW_MINIMUM")

    precision_by_fraction, precision_summary, precision_pass = _precision_diagnostics(
        counts,
        selected_bins,
        (primary_draw_index,),
        condition,
        reduced,
        contract,
    )
    if not precision_pass:
        reasons.append("BOUNDED_PRECISION_ESTIMABILITY_GATE_FAILED")
    mde = _analytic_mde(condition, reduced, complete_window, contract)
    mde_pass = bool(
        np.isfinite(mde)
        and mde <= float(contract["analytic_mde"]["maximum_standardized_mde"])
    )
    if not mde_pass:
        reasons.append("FAMILY_ADJUSTED_ANALYTIC_MDE_ABOVE_MAXIMUM")

    structural_pass = not any(
        value
        in {
            "TRAJECTORY_PLAN_BINDING_MISMATCH",
            "RECOMPUTED_FULL_FRAME_COUNTS_MISMATCH",
        }
        for value in reasons
    )
    design_pass = bool(
        structural_pass
        and terminal_gate
        and complete_controls >= int(support["minimum_controls_with_support_per_bin"])
        and complete_cases >= int(support["minimum_cases_with_support_per_bin"])
        and precision_pass
        and mde_pass
    )
    return {
        "frame_id": str(frame["frame_id"]),
        "display_name": str(frame["display_name"]),
        "source_key": str(frame["source_key"]),
        "rank_order": int(frame["rank_order"]),
        "input_status": "available",
        "structural_input_pass": structural_pass,
        "outcome_blind_input_pass": True,
        "trajectory_plan_id": data.trajectory_plan_id,
        "preflight_draw_id": data.primary_draw_id,
        "n_donors": len(condition),
        "n_controls": int(np.sum(~condition)),
        "n_cases": int(np.sum(condition)),
        "n_cells_selected": int(np.sum(cell_mask)),
        "n_trajectory_draws": len(data.draw_ids),
        "n_grid_bins": len(data.bin_left),
        "common_supported_bins": common_supported_bins,
        "terminal_window_start": (
            float(data.bin_left[selected_bins[0]]) if len(selected_bins) else ""
        ),
        "terminal_window_end": (
            float(data.bin_right[selected_bins[-1]]) if len(selected_bins) else ""
        ),
        "terminal_window_contiguous_bins": int(len(selected_bins)),
        "terminal_window_pseudotime_span": terminal_span,
        "terminal_support_gate_pass": terminal_gate,
        "maximum_timing_contiguous_bins_record_only": int(timing_bins),
        "maximum_timing_pseudotime_span_record_only": timing_span,
        "timing_design_gate_record_only": timing_design_gate,
        "timing_authorized": False,
        "minimum_control_support_selected": min_control,
        "minimum_case_support_selected": min_case,
        "minimum_loco_control_support_selected": min_loco,
        "complete_window_controls": complete_controls,
        "complete_window_cases": complete_cases,
        "unweighted_control_ess_minimum": float(unweighted_control_ess),
        "unweighted_case_ess_minimum": float(unweighted_case_ess),
        "unweighted_condition_information_minimum": float(unweighted_information),
        "precision_reference_count": precision_summary["reference_count"],
        "precision_control_ess_minimum": precision_summary["control_ess"],
        "precision_case_ess_minimum": precision_summary["case_ess"],
        "precision_condition_information_minimum": precision_summary[
            "condition_information"
        ],
        "precision_group_weight_ratio_maximum": precision_summary["group_weight_ratio"],
        "precision_sensitivity_gate_pass": precision_pass,
        "analytic_standardized_mde_family_adjusted": (
            float(mde) if np.isfinite(mde) else "not_evaluable"
        ),
        "analytic_mde_gate_pass": mde_pass,
        "design_preflight_pass": design_pass,
        "selection_uses_pathway_outcomes": False,
        "reason_codes": ";".join(sorted(set(reasons))) if reasons else "PASS",
        "precision_fraction_diagnostics_json": _canonical_json(precision_by_fraction),
        "draw_terminal_suffix_bins_json": _canonical_json(draw_suffix),
    }


def _safe_frame_evaluation(
    frame: Mapping[str, Any],
    data: StructuralFrameData,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        return evaluate_sampling_frame(frame, data, contract)
    except Exception as exc:
        return _unavailable_row(
            frame,
            f"INVALID_STRUCTURAL_INPUT:{type(exc).__name__}:{str(exc)}",
        )


def build_sampling_frame_design_atlas(
    contract: Mapping[str, Any],
    *,
    frame_sources: Mapping[str, str | Path],
    timing_no_go_path: str | Path | None,
    created_at_utc: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build and rank the outcome-blind atlas without reading pathway outcomes."""

    contract = validate_design_atlas_contract(contract)
    contract_payload_sha256 = contract.get("_contract_payload_sha256")
    if (
        contract_payload_sha256 != FROZEN_CONTRACT_PAYLOAD_SHA256
        or _contract_payload_sha256(contract) != FROZEN_CONTRACT_PAYLOAD_SHA256
    ):
        raise ValueError(
            "Atlas construction requires the logically frozen YAML contract"
        )
    timing_validation: dict[str, Any] | None = None
    timing_reason = "TIMING_NO_GO_DECISION_NOT_PROVIDED"
    if timing_no_go_path is not None:
        try:
            timing_validation = validate_timing_no_go_file(timing_no_go_path)
            timing_reason = "PASS"
        except Exception as exc:
            timing_reason = f"TIMING_NO_GO_DECISION_INVALID:{type(exc).__name__}"

    cache: dict[str, StructuralFrameData | Exception] = {}
    rows: list[dict[str, Any]] = []
    for frame in contract["sampling_frames"]:
        source_key = str(frame["source_key"])
        if source_key not in frame_sources:
            rows.append(_unavailable_row(frame, "STRUCTURAL_FRAME_SOURCE_NOT_PROVIDED"))
            continue
        if source_key not in cache:
            try:
                cache[source_key] = read_structural_frame_source(
                    source_key, frame_sources[source_key], contract
                )
            except Exception as exc:
                cache[source_key] = exc
        source = cache[source_key]
        if isinstance(source, Exception):
            rows.append(
                _unavailable_row(
                    frame,
                    f"INVALID_STRUCTURAL_SOURCE:{type(source).__name__}:{str(source)}",
                )
            )
        else:
            rows.append(_safe_frame_evaluation(frame, source, contract))

    atlas = pd.DataFrame(rows, columns=list(ATLAS_COLUMNS))
    eligible = atlas[atlas["design_preflight_pass"].eq(True)].copy()
    selected_frame_id: str | None = None
    if not eligible.empty:
        eligible["_mde"] = pd.to_numeric(
            eligible["analytic_standardized_mde_family_adjusted"], errors="coerce"
        )
        eligible = eligible.sort_values(
            [
                "terminal_window_contiguous_bins",
                "minimum_control_support_selected",
                "minimum_case_support_selected",
                "_mde",
                "rank_order",
            ],
            ascending=[False, False, False, True, True],
            kind="mergesort",
        )
        selected_frame_id = str(eligible.iloc[0]["frame_id"])

    timing_decision_pass = timing_validation is not None
    smoke_allowed = bool(selected_frame_id is not None and timing_decision_pass)
    timestamp = created_at_utc or datetime.now(timezone.utc).isoformat()
    sources_manifest = {
        key: {
            "candidate_dir": str(value.candidate_dir),
            "input_hashes": dict(value.input_hashes),
            "trajectory_plan_id": value.trajectory_plan_id,
        }
        for key, value in cache.items()
        if isinstance(value, StructuralFrameData)
    }
    decision = {
        "schema_name": ATLAS_SCHEMA_NAME,
        "schema_version": ATLAS_SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "contract_payload_sha256": contract_payload_sha256,
        "created_at_utc": timestamp,
        "outcome_blind": True,
        "real_pathway_outcomes_read": False,
        "selection_uses_pathway_outcomes": False,
        "timing_decision": "no_go",
        "timing_authorized": False,
        "timing_no_go_validation": timing_reason,
        "timing_no_go_sha256": (
            timing_validation.get("sha256") if timing_validation is not None else None
        ),
        "selected_terminal_frame_id": selected_frame_id,
        "selected_estimand": (
            "average_donor_terminal_window_condition_difference"
            if selected_frame_id is not None
            else None
        ),
        "primary_weighting": "donor_equal",
        "precision_standardized_role": "sensitivity_only",
        "smoke500_allowed": smoke_allowed,
        "smoke500_started": False,
        "smoke500_replicates": 500,
        "screen2000_allowed": False,
        "final10000_allowed": False,
        "formal_discovery_allowed": False,
        "reason_codes": [
            value
            for value in (
                None if selected_frame_id is not None else "NO_FRAME_PASSED_PREFLIGHT",
                None if timing_decision_pass else timing_reason,
            )
            if value is not None
        ],
        "frame_rank_order": atlas.sort_values("rank_order")["frame_id"].tolist(),
        "structural_sources": sources_manifest,
    }
    return atlas, decision


def validate_evidence_snapshot_readiness(
    contract: Mapping[str, Any],
    *,
    frame_sources: Mapping[str, str | Path],
    decision: Mapping[str, Any],
    evidence_version: str,
) -> str:
    """Require complete structural source provenance for post-v1 snapshots."""

    version = validate_evidence_version(evidence_version)
    version_number = int(version[1:])
    if version_number == 1:
        return version

    validated_contract = validate_design_atlas_contract(contract)
    required = {
        str(frame["source_key"])
        for frame in validated_contract["sampling_frames"]
    }
    provided = set(frame_sources)
    if provided != required:
        missing = ",".join(sorted(required - provided)) or "none"
        unexpected = ",".join(sorted(provided - required)) or "none"
        raise ValueError(
            "Post-v1 evidence requires every predeclared structural source binding; "
            f"missing={missing}; unexpected={unexpected}"
        )

    validated_sources = set(decision.get("structural_sources", {}))
    if validated_sources != required:
        missing = ",".join(sorted(required - validated_sources)) or "none"
        raise ValueError(
            "Post-v1 evidence cannot be frozen until every structural source "
            f"validates; missing_or_invalid={missing}"
        )
    _require_exact(
        decision.get("contract_payload_sha256"),
        FROZEN_CONTRACT_PAYLOAD_SHA256,
        "evidence_contract_payload_sha256",
    )
    _require_exact(
        decision.get("real_pathway_outcomes_read"),
        False,
        "evidence_real_pathway_outcomes_read",
    )
    _require_exact(
        decision.get("selection_uses_pathway_outcomes"),
        False,
        "evidence_selection_uses_pathway_outcomes",
    )
    return version


def _evidence_artifact_names(evidence_version: str) -> tuple[str, str, str]:
    version = validate_evidence_version(evidence_version)
    return (
        f"t21_sampling_frame_design_atlas_{version}.tsv",
        f"t21_terminal_amplitude_preflight_decision_{version}.json",
        f"t21_sampling_frame_design_atlas_build_record_{version}.json",
    )


def write_sampling_frame_design_atlas(
    atlas: pd.DataFrame,
    decision: Mapping[str, Any],
    *,
    output_dir: str | Path,
    evidence_version: str = "v1",
) -> dict[str, Any]:
    """Atomically create one evidence snapshot without replacing an earlier one."""

    version = validate_evidence_version(evidence_version)
    _require_exact(
        decision.get("contract_payload_sha256"),
        FROZEN_CONTRACT_PAYLOAD_SHA256,
        "output_contract_payload_sha256",
    )
    _require_exact(
        decision.get("real_pathway_outcomes_read"),
        False,
        "output_real_pathway_outcomes_read",
    )
    _require_exact(
        decision.get("selection_uses_pathway_outcomes"),
        False,
        "output_selection_uses_pathway_outcomes",
    )
    _require_exact(decision.get("smoke500_started"), False, "smoke500_started")
    if tuple(atlas.columns) != ATLAS_COLUMNS:
        raise ValueError("Atlas columns differ from the frozen output schema")

    output = Path(output_dir)
    output_parent = output.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(
            f"Evidence snapshot already exists and cannot be replaced: {output}"
        )

    lock_path = output_parent / f".{output.name}.create.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Evidence snapshot creation is already locked: {lock_path}"
        ) from exc

    temporary_output: Path | None = None
    try:
        os.close(lock_fd)
        if output.exists():
            raise FileExistsError(
                f"Evidence snapshot already exists and cannot be replaced: {output}"
            )
        temporary_output = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output_parent)
        )
        atlas_name, decision_name, manifest_name = _evidence_artifact_names(version)
        temporary_atlas_path = temporary_output / atlas_name
        temporary_decision_path = temporary_output / decision_name
        final_output = output.resolve()
        atlas_path = final_output / atlas_name
        decision_path = final_output / decision_name
        manifest_path = final_output / manifest_name

        atlas.loc[:, list(ATLAS_COLUMNS)].to_csv(
            temporary_atlas_path, sep="\t", index=False
        )
        decision_record = dict(decision)
        existing_version = decision_record.get("evidence_version")
        if existing_version is not None and existing_version != version:
            raise ValueError("Decision evidence version conflicts with output version")
        decision_record.update(
            {
                "evidence_version": version,
                "evidence_revision_only": True,
                "method_amendment_changed": False,
            }
        )
        temporary_decision_path.write_text(
            json.dumps(
                decision_record, indent=2, sort_keys=True, ensure_ascii=True
            )
            + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_name": "t21_sampling_frame_design_atlas_build_record",
            "schema_version": "1.0.0",
            "evidence_version": version,
            "evidence_revision_only": True,
            "method_amendment_changed": False,
            "method_contract_id": decision["contract_id"],
            "contract_payload_sha256": decision["contract_payload_sha256"],
            "implementation_path": str(Path(__file__).resolve()),
            "implementation_sha256": _sha256_file(Path(__file__).resolve()),
            "structural_sources": decision["structural_sources"],
            "atlas_path": str(atlas_path),
            "atlas_sha256": _sha256_file(temporary_atlas_path),
            "decision_path": str(decision_path),
            "decision_sha256": _sha256_file(temporary_decision_path),
            "smoke500_allowed": bool(decision["smoke500_allowed"]),
            "smoke500_started": False,
        }
        temporary_manifest_path = temporary_output / manifest_name
        temporary_manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        if output.exists():
            raise FileExistsError(
                f"Evidence snapshot already exists and cannot be replaced: {output}"
            )
        os.rename(temporary_output, output)
        temporary_output = None
        manifest["manifest_path"] = str(manifest_path)
        manifest["manifest_sha256"] = _sha256_file(manifest_path)
        return manifest
    finally:
        if temporary_output is not None and temporary_output.exists():
            shutil.rmtree(temporary_output)
        lock_path.unlink(missing_ok=True)


def validate_sampling_frame_design_atlas_evidence(
    output_dir: str | Path,
    *,
    evidence_version: str,
) -> dict[str, Any]:
    """Validate the hashes and frozen-boundary fields of one evidence snapshot."""

    version = validate_evidence_version(evidence_version)
    output = Path(output_dir).resolve()
    atlas_name, decision_name, manifest_name = _evidence_artifact_names(version)
    atlas_path = output / atlas_name
    decision_path = output / decision_name
    manifest_path = output / manifest_name
    for path in (atlas_path, decision_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Evidence artifact is missing: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    _require_exact(manifest.get("evidence_version"), version, "evidence_version")
    _require_exact(decision.get("evidence_version"), version, "decision_version")
    _require_exact(
        manifest.get("contract_payload_sha256"),
        FROZEN_CONTRACT_PAYLOAD_SHA256,
        "manifest_contract_payload_sha256",
    )
    _require_exact(
        decision.get("contract_payload_sha256"),
        FROZEN_CONTRACT_PAYLOAD_SHA256,
        "decision_contract_payload_sha256",
    )
    _require_exact(
        manifest.get("method_amendment_changed"),
        False,
        "manifest_method_amendment_changed",
    )
    _require_exact(
        decision.get("method_amendment_changed"),
        False,
        "decision_method_amendment_changed",
    )
    _require_exact(
        decision.get("real_pathway_outcomes_read"),
        False,
        "decision_real_pathway_outcomes_read",
    )
    expected_paths = {
        "atlas_path": atlas_path,
        "decision_path": decision_path,
    }
    for key, expected_path in expected_paths.items():
        _require_exact(Path(manifest.get(key, "")), expected_path, key)
        hash_key = key.replace("_path", "_sha256")
        _require_exact(
            manifest.get(hash_key), _sha256_file(expected_path), hash_key
        )
    written_atlas = pd.read_csv(atlas_path, sep="\t")
    if tuple(written_atlas.columns) != ATLAS_COLUMNS:
        raise ValueError("Written atlas columns differ from the frozen output schema")

    result = dict(manifest)
    result["manifest_path"] = str(manifest_path)
    result["manifest_sha256"] = _sha256_file(manifest_path)
    return result


def parse_frame_source_bindings(values: Sequence[str]) -> dict[str, Path]:
    bindings: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("Frame source must use SOURCE_KEY=CANDIDATE_DIR")
        key, raw_path = value.split("=", 1)
        key = key.strip()
        if key not in {"cd45", "cd235a"} or key in bindings or not raw_path.strip():
            raise ValueError("Frame source key is unknown, duplicated, or empty")
        bindings[key] = Path(raw_path.strip())
    return bindings


__all__ = [
    "ATLAS_COLUMNS",
    "ATLAS_SCHEMA_NAME",
    "ATLAS_SCHEMA_VERSION",
    "CONTRACT_ID",
    "StructuralFrameData",
    "build_sampling_frame_design_atlas",
    "evaluate_sampling_frame",
    "load_design_atlas_contract",
    "parse_frame_source_bindings",
    "read_structural_frame_source",
    "validate_design_atlas_contract",
    "validate_evidence_snapshot_readiness",
    "validate_evidence_version",
    "validate_sampling_frame_design_atlas_evidence",
    "write_sampling_frame_design_atlas",
]
