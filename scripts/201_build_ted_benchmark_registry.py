from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = ROOT / "data_external" / "ted_upgrade_candidate_audit"


DATASET_SPECS = {
    "GSE153056": {
        "analysis_role": "main_known_source_outcome",
        "main_text_tier": "Figure2",
        "expected_source": "CRISPR_perturbation_plus_IFNg",
        "expected_event": "IFNg_JAK_STAT_PDL1",
        "expected_outcome": "CD274_RNA_and_PDL1_surface_protein",
        "claim_if_pass": "outcome_supported_event_for_TED_methodology",
        "claim_if_fail": "no_TED_methodology_claim_for_RNA_protein_alignment",
        "claim_boundary": "not_GATA1_T21_biology",
    },
    "GSE93735": {
        "analysis_role": "main_intervention_reversal",
        "main_text_tier": "Figure2",
        "expected_source": "LPS_activation_with_dexamethasone_intervention",
        "expected_event": "LPS_inflammatory_activation",
        "expected_outcome": "Dex_LPS_reversal_fraction",
        "claim_if_pass": "reversal_supported_event_for_claim_boundary_audit",
        "claim_if_fail": "matched_intervention_positive_stratum_not_supported",
        "claim_boundary": "not_GSE271399_rescue",
    },
    "GSE90546": {
        "analysis_role": "main_pathway_branch_specificity",
        "main_text_tier": "Figure2_candidate",
        "expected_source": "CRISPRi_UPR_gene_perturbation",
        "expected_event": "UPR_branch_specific_response",
        "expected_outcome": "expected_branch_greater_than_unrelated_branch",
        "claim_if_pass": "pathway_branch_specific_TED_event",
        "claim_if_fail": "UPR_branch_specificity_not_claimed",
        "claim_boundary": "not_rescue_or_cell_fate_outcome",
    },
    "GSE90063": {
        "analysis_role": "supplement_tf_lps_modulation",
        "main_text_tier": "Supplement",
        "expected_source": "TF_perturbation_plus_LPS",
        "expected_event": "LPS_response_modulation",
        "expected_outcome": "TF_effect_on_response_score",
        "claim_if_pass": "supplementary_inflammatory_perturbation_adapter",
        "claim_if_fail": "supplementary_only_no_claim",
        "claim_boundary": "not_matched_rescue",
    },
    "GSE133344": {
        "analysis_role": "supplement_crispra_combination_scalability",
        "main_text_tier": "Supplement",
        "expected_source": "single_and_combinatorial_CRISPRa",
        "expected_event": "event_mode_or_interaction_deviation",
        "expected_outcome": "interaction_like_transcriptional_deviation",
        "claim_if_pass": "supplementary_scalability_benchmark",
        "claim_if_fail": "scalability_extension_not_claimed",
        "claim_boundary": "not_independent_functional_outcome",
    },
    "SCP1064": {
        "analysis_role": "secondary_known_source_rna_event_protein_outcome",
        "main_text_tier": "SecondaryMainOrSupplement",
        "expected_source": "melanoma_CRISPR_perturbation_TIL_coculture",
        "expected_event": "immune_evasion_antigen_presentation_interferon_response_or_T_cell_interaction_modules",
        "expected_outcome": "protein_readout_from_CITE_or_Protein_expression",
        "claim_if_pass": "outcome_supported_event_for_TED_methodology",
        "claim_if_fail": "no_SCP1064_TED_methodology_claim",
        "claim_boundary": "outcome_supported_event_if_all_gates_pass",
        "source_accession": "SCP1064 / Frangieh_2021",
        "expected_direction": "perturbation_specific_attenuation_or_enhancement_of_immune_response_event",
        "secondary_outcome": "author_regulatory_matrix_effects",
    },
    "GSE298761": {
        "analysis_role": "gata1_cross_dataset_support",
        "main_text_tier": "Figure4",
        "expected_source": "GATA1_N_terminus_or_GATA1s",
        "expected_event": "erythroid_heme_maturation_axis_loss",
        "expected_outcome": "cross_dataset_direction_consistency",
        "claim_if_pass": "cross_dataset_mechanistic_support",
        "claim_if_fail": "not_used_for_GATA1_cross_dataset_support",
        "claim_boundary": "not_human_T21_matched_rescue",
    },
    "GSE315981": {
        "analysis_role": "gata1_cross_dataset_support",
        "main_text_tier": "Figure4",
        "expected_source": "GATA1s_vs_GATA1FL",
        "expected_event": "heme_erythroid_axis_loss",
        "expected_outcome": "cross_dataset_direction_consistency",
        "claim_if_pass": "cross_dataset_mechanistic_support",
        "claim_if_fail": "not_used_for_GATA1_cross_dataset_support",
        "claim_boundary": "not_T21_or_rescue",
    },
    "GSE36787": {
        "analysis_role": "gata1_cross_dataset_support",
        "main_text_tier": "Figure4",
        "expected_source": "human_T21_GATA1s_iPSC_proxy",
        "expected_event": "heme_erythroid_axis_loss",
        "expected_outcome": "cross_dataset_direction_consistency",
        "claim_if_pass": "direct_human_proxy_directional_support",
        "claim_if_fail": "not_used_for_GATA1_cross_dataset_support",
        "claim_boundary": "not_modern_single_cell_rescue",
    },
    "GSE130156": {
        "analysis_role": "gata1_cross_dataset_support",
        "main_text_tier": "Figure4",
        "expected_source": "mouse_Gata1s_with_Gata2_rescue_context",
        "expected_event": "heme_erythroid_axis_loss_and_immature_axis_gain",
        "expected_outcome": "cross_dataset_direction_consistency",
        "claim_if_pass": "mouse_downstream_regulatory_support",
        "claim_if_fail": "not_used_for_GATA1_cross_dataset_support",
        "claim_boundary": "not_human_T21_same_system_rescue",
    },
}


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def access_status(dataset: str, downloads: pd.DataFrame, assessment: pd.DataFrame) -> str:
    row = assessment[assessment["dataset"].astype(str).eq(dataset)]
    if dataset == "SCP1064":
        failed = downloads[
            downloads["dataset"].astype(str).eq(dataset)
            & downloads["status"].astype(str).eq("failed")
        ]
        if not failed.empty:
            return "metadata_public_file_endpoint_401"
    if not row.empty:
        status = str(row.iloc[0].get("runability_status", ""))
        if status:
            return status
    sub = downloads[downloads["dataset"].astype(str).eq(dataset)]
    if sub["status"].astype(str).isin(["downloaded", "already_present"]).any():
        return "downloaded"
    return "not_audited"


def build_registry(download_manifest: Path, assessment_path: Path) -> pd.DataFrame:
    downloads = read_tsv(download_manifest)
    assessment = read_tsv(assessment_path)
    rows = []
    for dataset, spec in DATASET_SPECS.items():
        assess = assessment[assessment["dataset"].astype(str).eq(dataset)]
        source_url = str(assess.iloc[0].get("source_url", "")) if not assess.empty else ""
        large = str(assess.iloc[0].get("large_skipped_files", "")) if not assess.empty else ""
        status = access_status(dataset, downloads, assessment)
        if dataset == "SCP1064":
            downloaded = set(downloads[downloads["dataset"].astype(str).eq(dataset)]["file_name"].astype(str))
            required = {
                "Protein_expression.csv.gz",
                "raw_CITE_expression.csv.gz",
                "RNA_metadata.csv",
                "all_sgRNA_assignments.txt",
            }
            if required.issubset(downloaded):
                status = "portal_protein_file_authenticated_downloaded_and_processed_rna_h5ad_available"
        main_text = spec["main_text_tier"].startswith("Figure")
        rows.append(
            {
                "dataset_id": dataset,
                "dataset": dataset,
                "source_accession": spec.get("source_accession", dataset),
                "analysis_role": spec["analysis_role"],
                "main_text_tier": spec["main_text_tier"],
                "main_text_candidate": main_text,
                "manuscript_role": spec["main_text_tier"],
                "access_status": status,
                "expected_source": spec["expected_source"],
                "expected_event": spec["expected_event"],
                "expected_event_family": spec["expected_event"],
                "expected_direction": spec.get("expected_direction", "expected_direction_defined_by_preregistration"),
                "expected_outcome": spec["expected_outcome"],
                "primary_outcome": spec["expected_outcome"],
                "secondary_outcome": spec.get("secondary_outcome", ""),
                "claim_if_pass": spec["claim_if_pass"],
                "claim_if_fail": spec["claim_if_fail"],
                "claim_boundary": spec["claim_boundary"],
                "source_url": source_url,
                "large_skipped_files": large,
            }
        )
    return pd.DataFrame(rows)


def write_cards(registry: pd.DataFrame, path: Path) -> None:
    cards = registry[
        [
            "dataset",
            "analysis_role",
            "expected_source",
            "expected_event",
            "expected_outcome",
            "claim_if_pass",
            "claim_if_fail",
            "claim_boundary",
        ]
    ].copy()
    cards["status"] = "frozen_before_validation"
    cards.to_csv(path, sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download-manifest",
        type=Path,
        default=DEFAULT_AUDIT / "candidate_download_manifest.tsv",
    )
    parser.add_argument(
        "--assessment",
        type=Path,
        default=DEFAULT_AUDIT / "ted_upgrade_assessment.tsv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "config" / "ted_upgrade_benchmark_registry.tsv",
    )
    args = parser.parse_args()

    registry = build_registry(args.download_manifest, args.assessment)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(args.out, sep="\t", index=False)
    write_cards(registry, args.out.parent / "ted_preregistration_cards.tsv")

    print(f"Wrote {args.out} ({len(registry)} rows)")
    print(f"Wrote {args.out.parent / 'ted_preregistration_cards.tsv'}")


if __name__ == "__main__":
    main()
