"""Reproduce the frozen corrected-v2 GSE171964 eligibility analysis.

Conditional E2 display strings are retained as pre-result provenance only.
The observed v1.1 replication facets are emitted by
``build_bib_companion_evidence_contracts.py``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
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

from pyfgsea.ted_evidence import ReplicationFacetInputs, assign_replication_facets
from pyfgsea.ted_flagship import (
    entropy_balance,
    exhaustive_sign_flip_max_t,
    leave_one_donor_retention,
    peak_day,
    transient_contrasts,
)


CONFIG_PATH = ROOT / "config" / "ted_gse171964_replication_v1.yaml"
PRIMARY_CONFIG_PATH = ROOT / "config" / "ted_bnt162b2_flagship_v1.yaml"
FREEZE_DIR = ROOT / "results" / "ted_gse171964_replication" / "protocol_freeze_v1"
SOURCE_DIR = ROOT / "data_external" / "GSE171964_BNT162b2_replication" / "source"
OUT_DIR = ROOT / "results" / "ted_gse171964_replication" / "analysis_v1"
TIMEPOINTS = (21, 22, 28, 42)
TRANSIENT = {21: -0.50, 22: 1.00, 28: -0.25, 42: -0.25}
ACTIVATION = {21: -1.00, 22: 1.00, 28: 0.00, 42: 0.00}
RECOVERY = {21: 0.00, 22: 1.00, 28: -0.50, 42: -0.50}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_r_quoted_vector(path: Path) -> list[str]:
    pattern = re.compile(r'^"\d+"\s+"(.*)"\s*$')

    def parse(handle) -> list[str]:
        values: list[str] = []
        for line in handle:
            match = pattern.match(line.rstrip("\r\n"))
            if match:
                values.append(match.group(1))
        return values

    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            values = parse(handle)
    else:
        with path.open("r", encoding="utf-8") as handle:
            values = parse(handle)
    if not values:
        raise ValueError(f"No values parsed from {path}")
    return values


def read_gmt(path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) >= 3:
            result[fields[0]] = fields[2:]
    if not result:
        raise ValueError(f"No pathways parsed from {path}")
    return result


def robust_abs_z(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    center = float(numeric.median())
    mad = float(np.median(np.abs(numeric - center)))
    if mad <= np.finfo(float).eps:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index)
    return np.abs(numeric - center) / (1.4826 * mad)


def mean_gene_z_per_cell(logx, indices: list[int]) -> np.ndarray:
    if not indices:
        return np.zeros(logx.shape[0], dtype=np.float32)
    block = logx[:, indices].toarray().astype(np.float32, copy=False)
    mean = block.mean(axis=0)
    std = block.std(axis=0)
    std[std <= np.finfo(np.float32).eps] = 1.0
    return ((block - mean) / std).mean(axis=1)


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
        contrasts = transient_contrasts(
            table,
            transient_weights=TRANSIENT,
            activation_weights=ACTIVATION,
            recovery_weights=RECOVERY,
        )
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


def matched_random_sets(
    target_indices: list[int],
    global_mean: np.ndarray,
    detection: np.ndarray,
    excluded: set[int],
    *,
    n_sets: int,
    seed: int,
) -> list[list[int]]:
    n_genes = len(global_mean)
    expression_bin = pd.qcut(
        pd.Series(global_mean).rank(method="first"), 10, labels=False
    ).to_numpy()
    detection_bin = pd.qcut(
        pd.Series(detection).rank(method="first"), 10, labels=False
    ).to_numpy()
    bin_lookup: dict[tuple[int, int], np.ndarray] = {}
    for eb in range(10):
        for db in range(10):
            candidates = np.flatnonzero((expression_bin == eb) & (detection_bin == db))
            bin_lookup[(eb, db)] = np.asarray(
                [idx for idx in candidates if idx not in excluded], dtype=int
            )
    fallback = np.asarray([idx for idx in range(n_genes) if idx not in excluded], dtype=int)
    rng = np.random.default_rng(seed)
    result: list[list[int]] = []
    for _ in range(n_sets):
        chosen: list[int] = []
        used: set[int] = set()
        for target in target_indices:
            pool = bin_lookup[(int(expression_bin[target]), int(detection_bin[target]))]
            available = pool[~np.isin(pool, list(used))] if used else pool
            if not len(available):
                available = fallback[~np.isin(fallback, list(used))] if used else fallback
            pick = int(rng.choice(available))
            chosen.append(pick)
            used.add(pick)
        result.append(chosen)
    return result


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    primary_config = yaml.safe_load(PRIMARY_CONFIG_PATH.read_text(encoding="utf-8"))
    freeze = json.loads((FREEZE_DIR / "protocol_freeze.json").read_text(encoding="utf-8"))
    if sha256(CONFIG_PATH) != freeze["config_sha256"]:
        raise SystemExit("Replication config differs from the create-only freeze")
    if not (SOURCE_DIR.parent / "download_manifest.tsv").is_file():
        raise SystemExit("Run verify_gse171964_replication_download.py first")

    matrix_path = SOURCE_DIR / config["source"]["files"]["count_matrix"]["name"]
    feature_path = SOURCE_DIR / config["source"]["files"]["features"]["name"]
    barcode_path = SOURCE_DIR / config["source"]["files"]["barcodes"]["name"]
    phenotype_path = SOURCE_DIR / config["source"]["files"]["phenotype"]["name"]
    features = read_r_quoted_vector(feature_path)
    barcodes = read_r_quoted_vector(barcode_path)
    phenotype = pd.read_csv(phenotype_path, dtype={"pt_id": str})
    if len(phenotype) != len(barcodes):
        raise ValueError("Phenotype and barcode row counts differ")
    if not np.array_equal(phenotype["barcode"].astype(str).to_numpy(), np.asarray(barcodes)):
        raise ValueError("Corrected v2 phenotype rows are not aligned to the barcode vector")

    print("Reading corrected v2 Matrix Market counts", flush=True)
    matrix = mmread(str(matrix_path))
    if matrix.shape == (len(barcodes), len(features)):
        matrix = matrix.T
    if matrix.shape != (len(features), len(barcodes)):
        raise ValueError(
            f"Unexpected matrix dimensions {matrix.shape}; expected {(len(features), len(barcodes))}"
        )
    matrix = matrix.tocsc()
    adt_indices = [idx for idx, name in enumerate(features) if name.endswith("_ADT")]
    if not adt_indices:
        raise ValueError("The corrected feature panel has no ADT rows")
    n_rna = min(adt_indices)
    if adt_indices != list(range(n_rna, len(features))):
        raise ValueError("ADT rows are not a trailing feature block")
    rna_features = features[:n_rna]
    rna_to_index = {name: idx for idx, name in enumerate(rna_features)}

    total_rna = np.asarray(matrix[:n_rna, :].sum(axis=0)).ravel().astype(float)
    if np.any(total_rna <= 0):
        raise ValueError("At least one cell has zero RNA library size")
    detected_rna = np.asarray(matrix[:n_rna, :].getnnz(axis=0)).ravel().astype(int)
    mt_indices = [idx for idx, name in enumerate(rna_features) if name.upper().startswith("MT-")]
    mt_counts = np.asarray(matrix[mt_indices, :].sum(axis=0)).ravel() if mt_indices else np.zeros(len(barcodes))
    mt_fraction = mt_counts / total_rna

    sample_qc = phenotype[["pt_id", "day", "sample_id"]].copy()
    sample_qc["total_rna_umi"] = total_rna
    sample_qc["detected_rna_genes"] = detected_rna
    sample_qc = (
        sample_qc.groupby(["pt_id", "day", "sample_id"], as_index=False)
        .agg(
            n_cells=("total_rna_umi", "size"),
            median_rna_umi=("total_rna_umi", "median"),
            median_detected_genes=("detected_rna_genes", "median"),
        )
    )
    sample_qc["median_rna_umi_abs_mad_z"] = robust_abs_z(sample_qc["median_rna_umi"])
    sample_qc["median_detected_genes_abs_mad_z"] = robust_abs_z(sample_qc["median_detected_genes"])
    sample_qc["blind_qc_pass"] = (
        (sample_qc["median_rna_umi_abs_mad_z"] <= 3.0)
        & (sample_qc["median_detected_genes_abs_mad_z"] <= 3.0)
    )
    target_qc = sample_qc[sample_qc["day"].isin(TIMEPOINTS)]
    donor_qc = target_qc.groupby("pt_id").agg(
        n_timepoints=("day", "nunique"), all_sample_qc_pass=("blind_qc_pass", "all")
    )
    qc_donors = donor_qc.index[
        (donor_qc["n_timepoints"] == len(TIMEPOINTS)) & donor_qc["all_sample_qc_pass"]
    ].astype(str)

    pathways = read_gmt(FREEZE_DIR / "locked_pathway_family.gmt")
    pathway_union = {gene for genes in pathways.values() for gene in genes}
    marker_panels = primary_config["population"]["marker_panels"]
    forbidden = pathway_union | set(primary_config["population"]["forbidden_annotation_features"])
    usable_panels: dict[str, list[str]] = {
        name: [gene for gene in genes if gene in rna_to_index and gene not in forbidden]
        for name, genes in marker_panels.items()
    }
    if len(usable_panels["CD14_like_monocyte"]) < 5:
        raise ValueError("Too few non-IFN CD14-like annotation markers are present")
    marker_genes = sorted({gene for genes in usable_panels.values() for gene in genes})
    marker_indices = [rna_to_index[gene] for gene in marker_genes]
    marker_counts = matrix[marker_indices, :].toarray().astype(np.float32, copy=False)
    marker_counts *= (10000.0 / total_rna).astype(np.float32)[None, :]
    np.log1p(marker_counts, out=marker_counts)
    marker_mean = marker_counts.mean(axis=1, keepdims=True)
    marker_std = marker_counts.std(axis=1, keepdims=True)
    marker_std[marker_std <= np.finfo(np.float32).eps] = 1.0
    marker_z = (marker_counts - marker_mean) / marker_std
    marker_position = {gene: idx for idx, gene in enumerate(marker_genes)}
    panel_scores = np.column_stack(
        [
            marker_z[[marker_position[gene] for gene in genes], :].mean(axis=0)
            for genes in usable_panels.values()
        ]
    )
    panel_names = list(usable_panels)
    cd14_column = panel_names.index("CD14_like_monocyte")
    cd14_score = panel_scores[:, cd14_column]
    competitor = np.max(np.delete(panel_scores, cd14_column, axis=1), axis=1)
    score_margin = cd14_score - competitor
    # CD14 is excluded from the multi-gene annotation score because it belongs
    # to a frozen pathway-family member, but the protocol separately freezes a
    # low CD14 lineage-inclusion threshold.  Evaluate that declared gate from
    # its own normalized RNA row without adding CD14 back to the marker score.
    if "CD14" not in rna_to_index:
        raise ValueError("Frozen CD14 lineage gate cannot be evaluated")
    cd14_norm = np.asarray(matrix[rna_to_index["CD14"], :].toarray()).ravel().astype(
        np.float32, copy=False
    )
    cd14_norm *= (10000.0 / total_rna).astype(np.float32)
    np.log1p(cd14_norm, out=cd14_norm)
    selected = (
        (score_margin >= float(primary_config["population"]["minimum_score_margin"]))
        & (cd14_norm >= float(primary_config["population"]["minimum_cd14_log_normalized"]))
        & phenotype["pt_id"].astype(str).isin(qc_donors).to_numpy()
        & phenotype["day"].isin(TIMEPOINTS).to_numpy()
    )
    selected_columns = np.flatnonzero(selected)
    selection_meta = phenotype.iloc[selected_columns].reset_index(drop=True).copy()
    selection_meta["annotation_margin"] = score_margin[selected_columns]
    selection_meta["cd14_log_normalized"] = cd14_norm[selected_columns]
    selection_meta["total_rna_umi"] = total_rna[selected_columns]
    selection_meta["detected_rna_genes"] = detected_rna[selected_columns]
    selection_meta["mitochondrial_fraction"] = mt_fraction[selected_columns]
    selected_counts = (
        matrix[:n_rna, selected_columns].T.tocsr().astype(np.float32)
    )
    del matrix, marker_counts, marker_z, panel_scores

    cell_counts = (
        selection_meta.groupby(["pt_id", "day"], as_index=False)
        .size()
        .rename(columns={"size": "selected_cells"})
    )
    complete_cell_donors = (
        cell_counts.assign(
            pass_cells=cell_counts["selected_cells"]
            >= int(primary_config["population"]["minimum_cells_per_donor_time"])
        )
        .groupby("pt_id")
        .agg(n_timepoints=("day", "nunique"), all_cells_pass=("pass_cells", "all"))
    )
    evaluable_donors = complete_cell_donors.index[
        (complete_cell_donors["n_timepoints"] == len(TIMEPOINTS))
        & complete_cell_donors["all_cells_pass"]
    ].astype(str)
    keep_cells = selection_meta["pt_id"].astype(str).isin(evaluable_donors).to_numpy()
    selected_counts = selected_counts[keep_cells, :]
    selection_meta = selection_meta.loc[keep_cells].reset_index(drop=True)
    if len(evaluable_donors) < int(config["design"]["minimum_evaluable_donors"]):
        print("Blind sample-QC failures:", flush=True)
        print(
            sample_qc.loc[
                sample_qc["day"].isin(TIMEPOINTS) & ~sample_qc["blind_qc_pass"]
            ].to_string(index=False),
            flush=True,
        )
        print("Donor QC summary:", flush=True)
        print(donor_qc.to_string(), flush=True)
        print("RNA-only population counts:", flush=True)
        print(cell_counts.to_string(index=False), flush=True)
        facets = assign_replication_facets(
            ReplicationFacetInputs(
                event_analysis_complete=True,
                event_replication_tested=False,
                independent_cohort=True,
                same_event_family=True,
                early_activation_same_direction=None,
                recovery_same_direction=None,
                evaluable_donor_direction_fraction=None,
                family_adjusted_p=None,
                gates_frozen=True,
                additional_declared_gates_pass=False,
                outcome_analysis_complete=True,
                outcome_replication_tested=False,
                outcome_modality_compatible=False,
                outcome_type="protein",
            )
        )
        OUT_DIR.mkdir(parents=True, exist_ok=False)
        sample_qc.to_csv(OUT_DIR / "sample_blind_qc.tsv", sep="\t", index=False)
        donor_qc.reset_index().to_csv(OUT_DIR / "donor_blind_qc.tsv", sep="\t", index=False)
        cell_counts.to_csv(OUT_DIR / "rna_only_population_counts.tsv", sep="\t", index=False)
        pd.DataFrame(
            [
                {
                    "gate": "evaluable_donors",
                    "status": "failed",
                    "passed": False,
                    "observed": len(evaluable_donors),
                    "required": int(config["design"]["minimum_evaluable_donors"]),
                    "protocol_frozen_before_expression": True,
                },
                *[
                    {
                        "gate": name,
                        "status": "not_evaluable_after_prerequisite_failure",
                        "passed": pd.NA,
                        "observed": pd.NA,
                        "required": pd.NA,
                        "protocol_frozen_before_expression": True,
                    }
                    for name in (
                        "family_adjusted_p",
                        "direction_stability",
                        "early_activation_direction",
                        "recovery_direction",
                        "leave_one_donor_selection",
                        "matched_state_smd",
                        "matched_state_effective_sample_size",
                        "matched_state_weight_ratio",
                        "matched_state_attenuation",
                        "negative_control_margin",
                        "upstream_score_agreement",
                        "peak_window_agreement",
                    )
                ],
            ]
        ).to_csv(OUT_DIR / "replication_gate_table.tsv", sep="\t", index=False)
        result = {
            "protocol_id": config["protocol"]["id"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "GSE171964",
            "corrected_release": config["source"]["corrected_release"],
            "sample_sheet_mapping": config["design"]["public_sample_sheet_mapping"],
            "evaluable_donors": sorted(evaluable_donors),
            "n_evaluable_donors": len(evaluable_donors),
            "primary_pathway": config["pathway_family"]["primary"],
            "event_replication_attempt_status": "failed_at_eligibility_prerequisite",
            "event_replication_reason": "insufficient_frozen_QC_donors",
            **facets.as_dict(),
            "protein_outcome": {
                "replication_status": facets.outcome_replication_status,
                "reason": config["protein_outcome_replication"]["fixed_reason"],
                "CD64_ADT_present": False,
                "CD169_ADT_present": False,
            },
            "bounded_display_if_primary_E2_protein_passes": facets.display(
                "E2", within_study_outcome_status="passed"
            ),
        }
        (OUT_DIR / "replication_status.json").write_text(
            json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
        )
        pd.DataFrame(
            [
                {
                    "event_replication_eligibility_status": facets.event_replication_eligibility_status,
                    "event_replication_test_status": facets.event_replication_test_status,
                    "event_replication_status": facets.event_replication_status,
                    "protein_outcome.replication_status": facets.outcome_replication_status,
                    "event_replication_attempt_status": result["event_replication_attempt_status"],
                    "event_replication_reason": result["event_replication_reason"],
                    "bounded_display_if_primary_E2_protein_passes": result[
                        "bounded_display_if_primary_E2_protein_passes"
                    ],
                }
            ]
        ).to_csv(OUT_DIR / "replication_status.tsv", sep="\t", index=False)
        provenance_paths = [
            CONFIG_PATH,
            PRIMARY_CONFIG_PATH,
            FREEZE_DIR / "protocol_freeze.json",
            FREEZE_DIR / "locked_pathway_family.gmt",
            SOURCE_DIR.parent / "download_manifest.tsv",
            Path(__file__).resolve(),
        ]
        pd.DataFrame(
            [
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in provenance_paths
            ]
        ).to_csv(OUT_DIR / "analysis_manifest.tsv", sep="\t", index=False)
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)
        return

    print(f"RNA-only CD14-like selection retained {len(selection_meta)} cells", flush=True)
    scales = (10000.0 / selection_meta["total_rna_umi"].to_numpy(float)).astype(np.float32)
    logx = selected_counts.multiply(scales[:, None]).tocsr()
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

    selection_meta["donor_id"] = selection_meta["pt_id"].astype(str)
    selection_meta["group"] = (
        selection_meta["donor_id"] + "__" + selection_meta["day"].astype(str)
    )
    categories = [f"{donor}__{day}" for donor in sorted(evaluable_donors) for day in TIMEPOINTS]
    group_cat = pd.Categorical(selection_meta["group"], categories=categories, ordered=True)
    if np.any(group_cat.codes < 0):
        raise ValueError("Unexpected donor-time group after filtering")
    n_groups = len(categories)
    indicator = np.zeros((n_groups, len(selection_meta)), dtype=np.float32)
    indicator[group_cat.codes, np.arange(len(selection_meta))] = 1.0
    group_n = indicator.sum(axis=1)
    group_mean = np.asarray(indicator @ logx) / group_n[:, None]
    group_gene_z = (group_mean - global_mean[None, :]) / safe_std[None, :]

    pathway_indices = {
        pathway: [rna_to_index[gene] for gene in genes if gene in rna_to_index]
        for pathway, genes in pathways.items()
    }
    minimum_genes = int(primary_config["pathway_family"]["minimum_detected_genes_per_set"])
    if any(len(indices) < minimum_genes for indices in pathway_indices.values()):
        failed = {name: len(indices) for name, indices in pathway_indices.items() if len(indices) < minimum_genes}
        raise ValueError(f"Pathways below detected-gene minimum: {failed}")
    score_table = pd.DataFrame(
        {
            "group": categories,
            "donor_id": [item.split("__")[0] for item in categories],
            "day": [int(item.split("__")[1]) for item in categories],
            "selected_cells": group_n.astype(int),
        }
    )
    for pathway, indices in pathway_indices.items():
        score_table[pathway] = group_gene_z[:, indices].mean(axis=1)

    rank_scores = score_table[["group", "donor_id", "day", "selected_cells"]].copy()
    for pathway, indices in pathway_indices.items():
        rank_scores[pathway] = np.asarray(
            [rankdata(row, method="average")[indices].mean() / n_rna for row in group_mean]
        )

    excluded_hvg = {rna_to_index[g] for g in pathway_union | {"FCGR1A", "SIGLEC1"} if g in rna_to_index}
    hvg_candidates = np.asarray([idx for idx in range(n_rna) if idx not in excluded_hvg])
    hvg_order = hvg_candidates[np.argsort(global_variance[hvg_candidates])[::-1]]
    hvg = hvg_order[: int(primary_config["state_matching"]["highly_variable_genes"])]
    pca_input = logx[:, hvg].toarray().astype(np.float32, copy=False)
    pca_scores = PCA(
        n_components=int(primary_config["state_matching"]["rna_pcs"]),
        svd_solver="randomized",
        random_state=int(config["negative_controls"]["random_seed"]),
    ).fit_transform(pca_input)
    del pca_input
    covariates = pd.DataFrame(
        pca_scores,
        columns=[f"RNA_PC{i}" for i in range(1, pca_scores.shape[1] + 1)],
    )
    cd14_genes = [rna_to_index[g] for g in usable_panels["CD14_like_monocyte"]]
    cd16_genes = [rna_to_index[g] for g in usable_panels["CD16_like_monocyte"]]
    covariates["CD14_minus_CD16_marker_score"] = (
        mean_gene_z_per_cell(logx, cd14_genes) - mean_gene_z_per_cell(logx, cd16_genes)
    )
    covariates["log1p_total_RNA_UMI"] = np.log1p(selection_meta["total_rna_umi"].to_numpy(float))
    covariates["mitochondrial_fraction"] = selection_meta["mitochondrial_fraction"].to_numpy(float)
    s_genes = [rna_to_index[g] for g in primary_config["negative_controls"]["competing_programs"]["S_phase"] if g in rna_to_index]
    g2m_genes = [rna_to_index[g] for g in primary_config["negative_controls"]["competing_programs"]["G2M_phase"] if g in rna_to_index]
    covariates["S_phase_score"] = mean_gene_z_per_cell(logx, s_genes)
    covariates["G2M_phase_score"] = mean_gene_z_per_cell(logx, g2m_genes)

    cell_weights = np.zeros(len(selection_meta), dtype=float)
    balance_rows: list[dict[str, object]] = []
    for donor in sorted(evaluable_donors):
        donor_mask = selection_meta["donor_id"].eq(donor).to_numpy()
        target = covariates.loc[donor_mask].mean(axis=0)
        for day in TIMEPOINTS:
            mask = donor_mask & selection_meta["day"].eq(day).to_numpy()
            weights, diagnostics = entropy_balance(covariates.loc[mask], target)
            cell_weights[np.flatnonzero(mask)] = weights.to_numpy()
            balance_rows.append(
                {
                    "donor_id": donor,
                    "day": day,
                    "n_cells": int(mask.sum()),
                    **diagnostics.__dict__,
                }
            )
    balance = pd.DataFrame(balance_rows)
    weighted_scores = score_table[["group", "donor_id", "day", "selected_cells"]].copy()
    for pathway, indices in pathway_indices.items():
        values: list[float] = []
        for group_index in range(n_groups):
            mask = group_cat.codes == group_index
            weighted_gene_mean = np.asarray(
                cell_weights[mask] @ logx[mask, :][:, indices]
            ).ravel()
            values.append(float(np.mean((weighted_gene_mean - global_mean[indices]) / safe_std[indices])))
        weighted_scores[pathway] = values

    pathway_names = list(pathways)
    transient, activation, recovery, contrast_detail = contrast_tables(score_table, pathway_names)
    weighted_transient, _, _, weighted_contrast_detail = contrast_tables(weighted_scores, pathway_names)
    rank_transient, rank_activation, rank_recovery, rank_contrast_detail = contrast_tables(rank_scores, pathway_names)
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
    random_rows: list[dict[str, object]] = []
    for number, indices in enumerate(random_sets, start=1):
        random_score = score_table[["group", "donor_id", "day", "selected_cells"]].copy()
        random_score["RANDOM"] = group_gene_z[:, indices].mean(axis=1)
        random_effect, _, _, _ = contrast_tables(random_score, ["RANDOM"])
        random_rows.append(
            {
                "control_type": "matched_random_gene_set",
                "control_id": f"random_{number:03d}",
                "n_genes": len(indices),
                "mean_transient_effect": float(random_effect["RANDOM"].mean()),
            }
        )
    for program, genes in primary_config["negative_controls"]["competing_programs"].items():
        indices = [rna_to_index[g] for g in genes if g in rna_to_index]
        if not indices:
            continue
        program_score = score_table[["group", "donor_id", "day", "selected_cells"]].copy()
        program_score["CONTROL"] = group_gene_z[:, indices].mean(axis=1)
        program_effect, _, _, _ = contrast_tables(program_score, ["CONTROL"])
        random_rows.append(
            {
                "control_type": "competing_program",
                "control_id": program,
                "n_genes": len(indices),
                "mean_transient_effect": float(program_effect["CONTROL"].mean()),
            }
        )
    negative_controls = pd.DataFrame(random_rows)
    random_abs_q95 = float(
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
    control_reference = max(random_abs_q95, competing_max)
    primary_effect = float(transient[primary].mean())
    negative_control_margin = primary_effect - control_reference

    unweighted_effect = primary_effect
    weighted_effect = float(weighted_transient[primary].mean())
    attenuation = (
        max(0.0, 1.0 - abs(weighted_effect) / abs(unweighted_effect))
        if abs(unweighted_effect) > np.finfo(float).eps
        else 1.0
    )
    primary_inference = inference.set_index("pathway").loc[primary]
    primary_lodo = lodo[lodo["pathway"].eq(primary)]
    direction_fraction = float(np.mean(transient[primary] > 0))
    activation_fraction = float(np.mean(activation[primary] > 0))
    recovery_fraction = float(np.mean(recovery[primary] > 0))
    rank_agreement = bool(
        rank_transient[primary].mean() > 0
        and rank_activation[primary].mean() > 0
        and rank_recovery[primary].mean() > 0
    )
    rank_wide = rank_scores.pivot(index="donor_id", columns="day", values=primary)
    peak_agreement = peak_day(rank_wide, TIMEPOINTS) == 22
    gates = {
        "evaluable_donors": len(evaluable_donors) >= int(config["gates"]["evaluable_donors_min"]),
        "family_adjusted_p": float(primary_inference["exact_maxT_p"])
        <= float(config["gates"]["family_adjusted_p_max"]),
        "direction_stability": direction_fraction
        >= float(config["gates"]["direction_stability_min"]),
        "early_activation_direction": activation[primary].mean() > 0
        and activation_fraction >= float(config["gates"]["direction_stability_min"]),
        "recovery_direction": recovery[primary].mean() > 0
        and recovery_fraction >= float(config["gates"]["direction_stability_min"]),
        "leave_one_donor_selection": float(primary_lodo["retention_fraction"].iloc[0])
        >= float(config["gates"]["leave_one_donor_selection_min"]),
        "matched_state_smd": float(balance["max_abs_smd"].max())
        <= float(config["gates"]["matched_state_smd_max"]),
        "matched_state_effective_sample_size": float(balance["effective_sample_size"].min())
        >= float(primary_config["state_matching"]["minimum_effective_sample_size"]),
        "matched_state_weight_ratio": float(balance["max_weight_ratio_to_uniform"].max())
        <= float(primary_config["state_matching"]["maximum_weight_ratio_to_uniform"]),
        "matched_state_attenuation": attenuation
        <= float(config["gates"]["matched_state_attenuation_max"]),
        "negative_control_margin": negative_control_margin
        > float(config["gates"]["negative_control_margin_min"]),
        "upstream_score_agreement": rank_agreement,
        "peak_window_agreement": peak_agreement,
    }
    event_complete = True
    event_tested = True
    event_pass = all(gates.values())
    facets = assign_replication_facets(
        ReplicationFacetInputs(
            event_analysis_complete=event_complete,
            event_replication_tested=event_tested,
            independent_cohort=True,
            same_event_family=True,
            early_activation_same_direction=gates["early_activation_direction"],
            recovery_same_direction=gates["recovery_direction"],
            evaluable_donor_direction_fraction=direction_fraction,
            family_adjusted_p=float(primary_inference["exact_maxT_p"]),
            gates_frozen=True,
            additional_declared_gates_pass=event_pass,
            outcome_analysis_complete=True,
            outcome_replication_tested=False,
            outcome_modality_compatible=False,
            outcome_type="protein",
        )
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_qc.to_csv(OUT_DIR / "sample_blind_qc.tsv", sep="\t", index=False)
    cell_counts.to_csv(OUT_DIR / "rna_only_population_counts.tsv", sep="\t", index=False)
    score_table.to_csv(OUT_DIR / "pathway_donor_time_scores.tsv", sep="\t", index=False)
    weighted_scores.to_csv(OUT_DIR / "pathway_donor_time_scores_state_matched.tsv", sep="\t", index=False)
    rank_scores.to_csv(OUT_DIR / "pathway_donor_time_scores_rank_bridge.tsv", sep="\t", index=False)
    contrast_detail.to_csv(OUT_DIR / "pathway_donor_contrasts.tsv", sep="\t", index=False)
    weighted_contrast_detail.to_csv(OUT_DIR / "pathway_donor_contrasts_state_matched.tsv", sep="\t", index=False)
    rank_contrast_detail.to_csv(OUT_DIR / "pathway_donor_contrasts_rank_bridge.tsv", sep="\t", index=False)
    inference.to_csv(OUT_DIR / "pathway_family_exact_maxT.tsv", sep="\t", index=False)
    lodo.to_csv(OUT_DIR / "leave_one_donor_refits.tsv", sep="\t", index=False)
    balance.to_csv(OUT_DIR / "state_matching_diagnostics.tsv", sep="\t", index=False)
    negative_controls.to_csv(OUT_DIR / "negative_controls.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {
                "gate": name,
                "passed": passed,
                "protocol_frozen_before_expression": True,
            }
            for name, passed in gates.items()
        ]
    ).to_csv(OUT_DIR / "replication_gate_table.tsv", sep="\t", index=False)

    result = {
        "protocol_id": config["protocol"]["id"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "GSE171964",
        "corrected_release": config["source"]["corrected_release"],
        "sample_sheet_mapping": config["design"]["public_sample_sheet_mapping"],
        "evaluable_donors": sorted(evaluable_donors),
        "n_evaluable_donors": len(evaluable_donors),
        "selected_rna_only_cd14_like_cells": len(selection_meta),
        "primary_pathway": primary,
        "primary_mean_transient_effect": primary_effect,
        "primary_family_adjusted_p": float(primary_inference["exact_maxT_p"]),
        "direction_fraction": direction_fraction,
        "activation_direction_fraction": activation_fraction,
        "recovery_direction_fraction": recovery_fraction,
        "lodo_retention_fraction": float(primary_lodo["retention_fraction"].iloc[0]),
        "state_matched_effect": weighted_effect,
        "state_match_attenuation": attenuation,
        "negative_control_margin": negative_control_margin,
        "gates": gates,
        **facets.as_dict(),
        "protein_outcome": {
            "replication_status": facets.outcome_replication_status,
            "reason": config["protein_outcome_replication"]["fixed_reason"],
            "CD64_ADT_present": False,
            "CD169_ADT_present": False,
        },
        "bounded_display_if_primary_E2_protein_passes": facets.display(
            "E2", within_study_outcome_status="passed"
        ),
    }
    (OUT_DIR / "replication_status.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "event_replication_eligibility_status": facets.event_replication_eligibility_status,
                "event_replication_test_status": facets.event_replication_test_status,
                "event_replication_status": facets.event_replication_status,
                "protein_outcome.replication_status": facets.outcome_replication_status,
                "bounded_display_if_primary_E2_protein_passes": result[
                    "bounded_display_if_primary_E2_protein_passes"
                ],
            }
        ]
    ).to_csv(OUT_DIR / "replication_status.tsv", sep="\t", index=False)

    provenance_paths = [
        CONFIG_PATH,
        PRIMARY_CONFIG_PATH,
        FREEZE_DIR / "protocol_freeze.json",
        FREEZE_DIR / "locked_pathway_family.gmt",
        SOURCE_DIR.parent / "download_manifest.tsv",
        Path(__file__).resolve(),
    ]
    pd.DataFrame(
        [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in provenance_paths
        ]
    ).to_csv(OUT_DIR / "analysis_manifest.tsv", sep="\t", index=False)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
