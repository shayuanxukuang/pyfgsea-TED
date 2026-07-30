"""Reproduce the frozen BNT162b2 RNA analysis and legacy migration fields.

The persisted E/V fields are historical provenance. Current v1.1 typed records
are built by ``build_bib_companion_evidence_contracts.py`` and cannot upgrade
the observed event E code.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.io import mmread
from scipy.stats import rankdata
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyfgsea.ted_evidence import (  # noqa: E402
    EventSupportInputs,
    ValidationProvenanceInputs,
    assign_evidence_boundary,
)
from pyfgsea.ted_flagship import (  # noqa: E402
    entropy_balance,
    exhaustive_sign_flip_max_t,
    leave_one_donor_retention,
    peak_day,
    transient_contrasts,
)
from scripts.run_gse171964_replication import (  # noqa: E402
    matched_random_sets,
    mean_gene_z_per_cell,
    read_gmt,
)


CONFIG_PATH = ROOT / "config" / "ted_bnt162b2_flagship_v1.yaml"
PROTOCOL_FREEZE = ROOT / "results" / "ted_bnt162b2_flagship" / "protocol_freeze_v1"
EXPORT_DIR = ROOT / "data_external" / "bnt162b2_cite_asap_2023" / "rna_masked_export_v1"
OUT_DIR = ROOT / "results" / "ted_bnt162b2_flagship" / "rna_event_freeze_v1"
TIMEPOINTS = (0, 2, 10, 28)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contrast_tables(
    scores: pd.DataFrame,
    pathways: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transient_rows: dict[str, pd.Series] = {}
    activation_rows: dict[str, pd.Series] = {}
    recovery_rows: dict[str, pd.Series] = {}
    long_rows: list[pd.DataFrame] = []
    for pathway in pathways:
        table = scores[["donor_id", "day", pathway]].rename(columns={pathway: "score"})
        contrasts = transient_contrasts(table)
        transient_rows[pathway] = contrasts["transient"]
        activation_rows[pathway] = contrasts["activation"]
        recovery_rows[pathway] = contrasts["recovery"]
        detail = contrasts.reset_index().rename(columns={"index": "donor_id"})
        detail["pathway"] = pathway
        long_rows.append(detail)
    return (
        pd.DataFrame(transient_rows),
        pd.DataFrame(activation_rows),
        pd.DataFrame(recovery_rows),
        pd.concat(long_rows, ignore_index=True),
    )


def verify_existing() -> None:
    status_path = OUT_DIR / "rna_event_status.json"
    manifest_path = OUT_DIR / "rna_event_manifest.tsv"
    if not status_path.is_file() or not manifest_path.is_file():
        raise SystemExit(f"Incomplete existing RNA freeze: {OUT_DIR}")
    manifest = pd.read_csv(manifest_path, sep="\t")
    for row in manifest.itertuples(index=False):
        path = ROOT / row.path if not Path(row.path).is_absolute() else Path(row.path)
        if not path.is_file() or sha256(path) != row.sha256:
            raise SystemExit(f"RNA freeze manifest mismatch: {path}")
    print(f"Verified existing create-only RNA event freeze: {OUT_DIR}")


def main() -> None:
    if OUT_DIR.exists():
        verify_existing()
        return
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    protocol = json.loads((PROTOCOL_FREEZE / "protocol_freeze.json").read_text(encoding="utf-8"))
    if sha256(CONFIG_PATH) != protocol["config_sha256"]:
        raise SystemExit("Primary config differs from its create-only protocol freeze")
    export_status_path = EXPORT_DIR / "rna_masked_export.json"
    matrix_path = EXPORT_DIR / "selected_rna_counts.mtx.gz"
    feature_path = EXPORT_DIR / "rna_features.txt"
    metadata_path = EXPORT_DIR / "selected_cell_metadata.tsv.gz"
    for path in (export_status_path, matrix_path, feature_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    export_status = json.loads(export_status_path.read_text(encoding="utf-8"))
    if export_status.get("adt_assay_values_accessed") is not False:
        raise SystemExit("RNA export did not preserve the declared ADT mask")

    features = feature_path.read_text(encoding="utf-8").splitlines()
    meta = pd.read_csv(metadata_path, sep="\t", dtype={"donor_id": str})
    matrix = mmread(str(matrix_path))
    if matrix.shape == (len(meta), len(features)):
        matrix = matrix.T
    if matrix.shape != (len(features), len(meta)):
        raise ValueError("RNA export matrix, feature and cell dimensions differ")
    counts = matrix.T.tocsr().astype(np.float32)
    del matrix
    if meta["cell_barcode"].duplicated().any():
        raise ValueError("Selected cell barcodes are not unique")
    expected_groups = pd.MultiIndex.from_product(
        [sorted(meta["donor_id"].unique()), TIMEPOINTS], names=["donor_id", "day"]
    )
    observed_groups = pd.MultiIndex.from_frame(meta[["donor_id", "day"]].drop_duplicates())
    if not expected_groups.isin(observed_groups).all():
        raise ValueError("RNA masked export lacks a frozen donor-time group")

    scale = (10000.0 / meta["total_rna_umi"].to_numpy(float)).astype(np.float32)
    logx = counts.multiply(scale[:, None]).tocsr()
    np.log1p(logx.data, out=logx.data)
    global_mean = np.asarray(logx.mean(axis=0)).ravel()
    squared = logx.copy()
    squared.data **= 2
    global_variance = np.maximum(
        np.asarray(squared.mean(axis=0)).ravel() - global_mean**2, 0.0
    )
    del squared
    global_std = np.sqrt(global_variance)
    safe_std = global_std.copy()
    safe_std[safe_std <= np.finfo(float).eps] = 1.0
    detection = np.asarray(logx.getnnz(axis=0)).ravel() / logx.shape[0]
    gene_to_index = {gene: idx for idx, gene in enumerate(features)}

    pathways = read_gmt(PROTOCOL_FREEZE / "locked_pathway_family.gmt")
    pathway_indices = {
        pathway: [gene_to_index[gene] for gene in genes if gene in gene_to_index]
        for pathway, genes in pathways.items()
    }
    minimum_genes = int(config["pathway_family"]["minimum_detected_genes_per_set"])
    if any(len(indices) < minimum_genes for indices in pathway_indices.values()):
        failed = {name: len(idx) for name, idx in pathway_indices.items() if len(idx) < minimum_genes}
        raise ValueError(f"Pathways below detected-gene minimum: {failed}")

    meta["group"] = meta["donor_id"] + "__" + meta["day"].astype(str)
    donors = sorted(meta["donor_id"].unique())
    categories = [f"{donor}__{day}" for donor in donors for day in TIMEPOINTS]
    group_cat = pd.Categorical(meta["group"], categories=categories, ordered=True)
    if np.any(group_cat.codes < 0):
        raise ValueError("Unexpected donor-time group in RNA export")
    indicator = np.zeros((len(categories), len(meta)), dtype=np.float32)
    indicator[group_cat.codes, np.arange(len(meta))] = 1.0
    group_n = indicator.sum(axis=1)
    group_mean = np.asarray(indicator @ logx) / group_n[:, None]
    group_gene_z = (group_mean - global_mean[None, :]) / safe_std[None, :]

    score_table = pd.DataFrame(
        {
            "group": categories,
            "donor_id": [value.split("__")[0] for value in categories],
            "day": [int(value.split("__")[1]) for value in categories],
            "selected_cells": group_n.astype(int),
        }
    )
    rank_scores = score_table.copy()
    for pathway, indices in pathway_indices.items():
        score_table[pathway] = group_gene_z[:, indices].mean(axis=1)
        rank_scores[pathway] = np.asarray(
            [rankdata(row, method="average")[indices].mean() / len(features) for row in group_mean]
        )

    pathway_union = {gene for genes in pathways.values() for gene in genes}
    excluded_hvg = {gene_to_index[g] for g in pathway_union | {"FCGR1A", "SIGLEC1"} if g in gene_to_index}
    hvg_candidates = np.asarray([idx for idx in range(len(features)) if idx not in excluded_hvg])
    hvg_order = hvg_candidates[np.argsort(global_variance[hvg_candidates])[::-1]]
    hvg = hvg_order[: int(config["state_matching"]["highly_variable_genes"])]
    pca_input = logx[:, hvg].toarray().astype(np.float32, copy=False)
    pca_scores = PCA(
        n_components=int(config["state_matching"]["rna_pcs"]),
        svd_solver="randomized",
        random_state=int(config["negative_controls"]["random_seed"]),
    ).fit_transform(pca_input)
    del pca_input
    covariates = pd.DataFrame(
        pca_scores, columns=[f"RNA_PC{i}" for i in range(1, pca_scores.shape[1] + 1)]
    )
    forbidden_markers = pathway_union | set(config["population"]["forbidden_annotation_features"])
    cd14_idx = [gene_to_index[g] for g in config["population"]["marker_panels"]["CD14_like_monocyte"] if g in gene_to_index and g not in forbidden_markers]
    cd16_idx = [gene_to_index[g] for g in config["population"]["marker_panels"]["CD16_like_monocyte"] if g in gene_to_index and g not in forbidden_markers]
    covariates["CD14_minus_CD16_marker_score"] = (
        mean_gene_z_per_cell(logx, cd14_idx) - mean_gene_z_per_cell(logx, cd16_idx)
    )
    covariates["log1p_total_RNA_UMI"] = np.log1p(meta["total_rna_umi"].to_numpy(float))
    covariates["mitochondrial_fraction"] = meta["mitochondrial_fraction"].to_numpy(float)
    s_idx = [gene_to_index[g] for g in config["negative_controls"]["competing_programs"]["S_phase"] if g in gene_to_index]
    g2m_idx = [gene_to_index[g] for g in config["negative_controls"]["competing_programs"]["G2M_phase"] if g in gene_to_index]
    covariates["S_phase_score"] = mean_gene_z_per_cell(logx, s_idx)
    covariates["G2M_phase_score"] = mean_gene_z_per_cell(logx, g2m_idx)

    cell_weights = np.zeros(len(meta), dtype=float)
    balance_rows: list[dict[str, object]] = []
    for donor in donors:
        donor_mask = meta["donor_id"].eq(donor).to_numpy()
        target = covariates.loc[donor_mask].mean(axis=0)
        for day in TIMEPOINTS:
            mask = donor_mask & meta["day"].eq(day).to_numpy()
            weights, diagnostics = entropy_balance(covariates.loc[mask], target)
            cell_weights[np.flatnonzero(mask)] = weights.to_numpy()
            balance_rows.append(
                {"donor_id": donor, "day": day, "n_cells": int(mask.sum()), **diagnostics.__dict__}
            )
    balance = pd.DataFrame(balance_rows)
    weighted_scores = score_table[["group", "donor_id", "day", "selected_cells"]].copy()
    for pathway, indices in pathway_indices.items():
        values: list[float] = []
        for group_index in range(len(categories)):
            mask = group_cat.codes == group_index
            weighted_gene_mean = np.asarray(cell_weights[mask] @ logx[mask, :][:, indices]).ravel()
            values.append(float(np.mean((weighted_gene_mean - global_mean[indices]) / safe_std[indices])))
        weighted_scores[pathway] = values

    pathway_names = list(pathways)
    transient, activation, recovery, contrast_detail = contrast_tables(score_table, pathway_names)
    weighted_transient, _, _, weighted_detail = contrast_tables(weighted_scores, pathway_names)
    rank_transient, rank_activation, rank_recovery, rank_detail = contrast_tables(rank_scores, pathway_names)
    inference = exhaustive_sign_flip_max_t(transient)
    lodo = leave_one_donor_retention(transient, activation, recovery)

    primary = config["pathway_family"]["primary"]
    primary_indices = pathway_indices[primary]
    excluded_random = {idx for indices in pathway_indices.values() for idx in indices}
    random_sets = matched_random_sets(
        primary_indices,
        global_mean,
        detection,
        excluded_random,
        n_sets=int(config["negative_controls"]["matched_random_gene_sets"]),
        seed=int(config["negative_controls"]["random_seed"]),
    )
    control_rows: list[dict[str, object]] = []
    for number, indices in enumerate(random_sets, start=1):
        frame = score_table[["group", "donor_id", "day", "selected_cells"]].copy()
        frame["CONTROL"] = group_gene_z[:, indices].mean(axis=1)
        effect, _, _, _ = contrast_tables(frame, ["CONTROL"])
        control_rows.append(
            {
                "control_type": "matched_random_gene_set",
                "control_id": f"random_{number:03d}",
                "n_genes": len(indices),
                "mean_transient_effect": float(effect["CONTROL"].mean()),
            }
        )
    for program, genes in config["negative_controls"]["competing_programs"].items():
        indices = [gene_to_index[g] for g in genes if g in gene_to_index]
        if not indices:
            continue
        frame = score_table[["group", "donor_id", "day", "selected_cells"]].copy()
        frame["CONTROL"] = group_gene_z[:, indices].mean(axis=1)
        effect, _, _, _ = contrast_tables(frame, ["CONTROL"])
        control_rows.append(
            {
                "control_type": "competing_program",
                "control_id": program,
                "n_genes": len(indices),
                "mean_transient_effect": float(effect["CONTROL"].mean()),
            }
        )
    negative_controls = pd.DataFrame(control_rows)
    random_q95 = float(
        negative_controls.loc[
            negative_controls["control_type"].eq("matched_random_gene_set"),
            "mean_transient_effect",
        ].abs().quantile(float(config["negative_controls"]["control_quantile"]))
    )
    competing_max = float(
        negative_controls.loc[
            negative_controls["control_type"].eq("competing_program"),
            "mean_transient_effect",
        ].abs().max()
    )
    primary_effect = float(transient[primary].mean())
    negative_control_margin = primary_effect - max(random_q95, competing_max)
    weighted_effect = float(weighted_transient[primary].mean())
    attenuation = (
        max(0.0, 1.0 - abs(weighted_effect) / abs(primary_effect))
        if abs(primary_effect) > np.finfo(float).eps
        else 1.0
    )
    primary_inference = inference.set_index("pathway").loc[primary]
    primary_lodo = lodo[lodo["pathway"].eq(primary)]
    lodo_pass = float(primary_lodo["retention_fraction"].iloc[0]) >= float(
        config["gates"]["leave_one_donor_selection_min"]
    )
    direction_fraction = float(np.mean(transient[primary] > 0))
    activation_fraction = float(np.mean(activation[primary] > 0))
    recovery_fraction = float(np.mean(recovery[primary] > 0))
    rank_agreement = bool(
        rank_transient[primary].mean() > 0
        and rank_activation[primary].mean() > 0
        and rank_recovery[primary].mean() > 0
    )
    rank_wide = rank_scores.pivot(index="donor_id", columns="day", values=primary)
    peak_agreement = peak_day(rank_wide) == 2
    balance_complete = bool(
        balance["converged"].all()
        and balance[
            ["max_abs_smd", "effective_sample_size", "max_weight_ratio_to_uniform"]
        ].notna().all().all()
    )
    balance_pass = bool(
        balance_complete
        and balance["max_abs_smd"].max() <= float(config["gates"]["matched_state_smd_max"])
        and balance["effective_sample_size"].min() >= float(config["state_matching"]["minimum_effective_sample_size"])
        and balance["max_weight_ratio_to_uniform"].max() <= float(config["state_matching"]["maximum_weight_ratio_to_uniform"])
        and attenuation <= float(config["gates"]["matched_state_attenuation_max"])
    )
    mode_pass = bool(
        activation[primary].mean() > 0
        and recovery[primary].mean() > 0
        and activation_fraction >= float(config["gates"]["direction_stability_min"])
        and recovery_fraction >= float(config["gates"]["direction_stability_min"])
        and rank_agreement
        and peak_agreement
        and lodo_pass
    )
    control_pass = negative_control_margin > float(config["gates"]["negative_control_margin_min"])
    gates = {
        "evaluable_donors": len(donors) >= int(config["gates"]["evaluable_donors_min"]),
        "family_adjusted_p": float(primary_inference["exact_maxT_p"]) <= float(config["gates"]["family_adjusted_p_max"]),
        "direction_stability": direction_fraction >= float(config["gates"]["direction_stability_min"]),
        "early_activation_direction": activation_fraction >= float(config["gates"]["direction_stability_min"]) and activation[primary].mean() > 0,
        "recovery_direction": recovery_fraction >= float(config["gates"]["direction_stability_min"]) and recovery[primary].mean() > 0,
        "leave_one_donor_selection": lodo_pass,
        "matched_state": balance_pass,
        "negative_controls": control_pass,
        "upstream_score_agreement": rank_agreement,
        "peak_day_agreement": peak_agreement,
    }
    gates = {name: bool(value) for name, value in gates.items()}
    boundary = assign_evidence_boundary(
        EventSupportInputs(
            event_family_declared=True,
            defensible_null_specified=True,
            biological_units_present=True,
            condition_batch_confounded=False,
            identifiability_status="identifiable" if mode_pass else "limited",
            artifact_dominated=not control_pass,
            event_q=float(primary_inference["exact_maxT_p"]),
            retained_module=True,
            basic_controls_pass=control_pass,
            matched_state_required=True,
            matched_state_overlap_pass=balance_pass,
            matched_state_attenuation=attenuation,
            effective_blocks=len(donors),
            block_support_method="exact_paired_sign_permutation",
            minimum_attainable_p=2.0 ** (-len(donors)),
            minimum_attainable_q=None,
            permutation_resolution_pass=True,
            block_q=float(primary_inference["exact_maxT_p"]),
            block_ci_excludes_zero=False,
            direction_stability=direction_fraction,
            mode_identifiable=mode_pass,
            negative_controls_required=True,
            negative_control_pass=control_pass,
            negative_control_margin=negative_control_margin,
        ),
        ValidationProvenanceInputs(),
    )
    if boundary.event_support.code == "E2" and not all(gates.values()):
        raise RuntimeError("Evidence assignment and frozen flagship gates disagree")

    OUT_DIR.mkdir(parents=True, exist_ok=False)
    score_table.to_csv(OUT_DIR / "pathway_donor_time_scores.tsv", sep="\t", index=False)
    weighted_scores.to_csv(OUT_DIR / "pathway_donor_time_scores_state_matched.tsv", sep="\t", index=False)
    rank_scores.to_csv(OUT_DIR / "pathway_donor_time_scores_rank_bridge.tsv", sep="\t", index=False)
    contrast_detail.to_csv(OUT_DIR / "pathway_donor_contrasts.tsv", sep="\t", index=False)
    weighted_detail.to_csv(OUT_DIR / "pathway_donor_contrasts_state_matched.tsv", sep="\t", index=False)
    rank_detail.to_csv(OUT_DIR / "pathway_donor_contrasts_rank_bridge.tsv", sep="\t", index=False)
    inference.to_csv(OUT_DIR / "pathway_family_exact_maxT.tsv", sep="\t", index=False)
    lodo.to_csv(OUT_DIR / "leave_one_donor_refits.tsv", sep="\t", index=False)
    balance.to_csv(OUT_DIR / "state_matching_diagnostics.tsv", sep="\t", index=False)
    negative_controls.to_csv(OUT_DIR / "negative_controls.tsv", sep="\t", index=False)
    pd.DataFrame([{"gate": key, "passed": value} for key, value in gates.items()]).to_csv(
        OUT_DIR / "rna_event_gate_table.tsv", sep="\t", index=False
    )
    result = {
        "protocol_id": config["protocol"]["id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outcome_values_accessed": False,
        "rna_event_freeze_complete": True,
        "adt_unmask_allowed": True,
        "n_evaluable_donors": len(donors),
        "evaluable_donors": donors,
        "n_selected_cells": len(meta),
        "primary_pathway": primary,
        "primary_mean_transient_effect": primary_effect,
        "primary_family_adjusted_p": float(primary_inference["exact_maxT_p"]),
        "direction_fraction": direction_fraction,
        "activation_direction_fraction": activation_fraction,
        "recovery_direction_fraction": recovery_fraction,
        "lodo_retention_fraction": float(primary_lodo["retention_fraction"].iloc[0]),
        "state_matched_effect": weighted_effect,
        "state_match_attenuation": attenuation,
        "state_matching_all_converged": balance_complete,
        "negative_control_margin": negative_control_margin,
        "gates": gates,
        **boundary.as_dict(),
    }
    (OUT_DIR / "rna_event_status.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    input_paths = [
        CONFIG_PATH,
        PROTOCOL_FREEZE / "protocol_freeze.json",
        PROTOCOL_FREEZE / "locked_pathway_family.gmt",
        export_status_path,
        matrix_path,
        feature_path,
        metadata_path,
        Path(__file__).resolve(),
    ]
    pd.DataFrame(
        [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "role": "rna_freeze_input",
            }
            for path in input_paths
        ]
    ).to_csv(OUT_DIR / "rna_event_manifest.tsv", sep="\t", index=False)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
