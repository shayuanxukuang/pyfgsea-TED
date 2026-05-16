"""Family-level TED-Perturbation inference for GSE271399.

The goal is to move from many pathway-level events to a small number of
mechanistic perturbation event families. This script works from previously
generated hard-validation tables and external validation outputs.
"""

from __future__ import annotations

import argparse
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from run_gse271399_first_round import bh_fdr
from run_gse271399_hard_validation_addons import sign_label, read_gmt


FAMILY_DEFINITIONS = {
    "ERYTHROID_EVENT_LOSS_FAMILY": {
        "description": "Ontology-aware erythroid maturation, heme/globin, GATA/KLF/TAL1, and iron transport loss family.",
        "members": [
            "ERYTHROID_MATURATION",
            "HEME_GLOBIN",
            "GATA_KLF_TAL1_REGULON",
            "IRON_TRANSPORT",
            "HALLMARK_HEME_METABOLISM",
        ],
        "primary_direction": "negative",
        "merge_basis": "ontology_aware_merge",
    },
    "CELL_CYCLE_RIBOSOME_MYC_E2F_FAMILY": {
        "description": "Cell-cycle, E2F/MYC, ribosome and cell-cycle-exit perturbation family.",
        "members": [
            "CELL_CYCLE_E2F_G2M",
            "HALLMARK_E2F_TARGETS",
            "HALLMARK_MYC_TARGETS_V1",
            "HALLMARK_MYC_TARGETS_V2",
            "RIBOSOME_TRANSLATION",
            "CELL_CYCLE_EXIT",
        ],
        "primary_direction": "context_dependent",
        "merge_basis": "pathway_family_compression_plus_curated_axis",
    },
    "INFLAMMATORY_INTERFERON_FAMILY": {
        "description": "Inflammatory and interferon response event family.",
        "members": [
            "INFLAMMATORY_INTERFERON",
            "HALLMARK_INFLAMMATORY_RESPONSE",
            "HALLMARK_INTERFERON_GAMMA_RESPONSE",
        ],
        "primary_direction": "positive",
        "merge_basis": "ontology_aware_merge",
    },
    "HSPC_MYELOID_MK_CONTEXT_FAMILY": {
        "description": "HSPC stemness, myeloid priming and megakaryocyte-context perturbation family.",
        "members": [
            "HSPC_STEMNESS",
            "MYELOID_PRIMING",
            "MEGAKARYOCYTE_MATURATION",
        ],
        "primary_direction": "context_dependent",
        "merge_basis": "lineage_context_merge",
    },
}

PAIRWISE_CONTRASTS = [
    "Euploid_GATA1s_vs_Euploid_wtGATA1",
    "T21_wtGATA1_vs_Euploid_wtGATA1",
    "T21_GATA1s_vs_T21_wtGATA1",
]

EFFECT_TO_ROBUSTNESS_CONTRAST = {
    "T21": ["T21_wtGATA1_vs_Euploid_wtGATA1"],
    "GATA1s": ["Euploid_GATA1s_vs_Euploid_wtGATA1", "T21_GATA1s_vs_T21_wtGATA1"],
    "interaction": ["interaction_did"],
}


def parse_profile(text: object) -> np.ndarray:
    vals: list[float] = []
    for part in str(text).split(","):
        part = part.strip()
        vals.append(np.nan if part in {"", "NA", "nan"} else float(part))
    return np.asarray(vals, dtype=float)


def profile_to_text(profile: np.ndarray) -> str:
    return ",".join("NA" if not np.isfinite(x) else f"{x:.6g}" for x in profile)


def sign_int(value: float) -> int:
    if not np.isfinite(value) or abs(value) < 1e-12:
        return 0
    return 1 if value > 0 else -1


def fisher_p(pvals: list[float]) -> float:
    vals = [min(max(float(p), 1e-300), 1.0) for p in pvals if np.isfinite(float(p))]
    if not vals:
        return 1.0
    stat = -2.0 * float(np.sum(np.log(vals)))
    p = float(stats.chi2.sf(stat, 2 * len(vals)))
    return max(p, 1e-300)


def gene_jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    u = a | b
    if not u:
        return 0.0
    return len(a & b) / len(u)


def family_weights(members: list[str], gene_sets: dict[str, list[str]]) -> tuple[dict[str, float], dict[str, float]]:
    present = [m for m in members if m in gene_sets]
    if not present:
        return {}, {}
    simple = {m: 1.0 / len(present) for m in present}
    raw = {}
    gene_map = {m: set(g.upper() for g in gene_sets.get(m, [])) for m in present}
    for p in present:
        overlap_sum = sum(gene_jaccard(gene_map[p], gene_map[q]) for q in present if q != p)
        raw[p] = 1.0 / (1.0 + overlap_sum)
    total = sum(raw.values())
    redundancy = {m: raw[m] / total for m in present}
    return simple, redundancy


def weighted_profile(pathway_profiles: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    if not pathway_profiles:
        return np.full(8, np.nan)
    max_len = max(len(v) for v in pathway_profiles.values())
    out = np.full(max_len, np.nan)
    for i in range(max_len):
        num = 0.0
        den = 0.0
        for pathway, profile in pathway_profiles.items():
            if pathway not in weights or i >= len(profile):
                continue
            val = profile[i]
            if np.isfinite(val):
                num += weights[pathway] * val
                den += weights[pathway]
        if den > 0:
            out[i] = num / den
    return out


def auc(profile: np.ndarray) -> float:
    return float(np.nanmean(profile)) if np.isfinite(profile).any() else np.nan


def abs_auc(profile: np.ndarray) -> float:
    return float(np.nanmean(np.abs(profile))) if np.isfinite(profile).any() else np.nan


def loss_auc(profile: np.ndarray) -> float:
    if not np.isfinite(profile).any():
        return np.nan
    return float(np.nanmean(np.maximum(-profile, 0)))


def gain_auc(profile: np.ndarray) -> float:
    if not np.isfinite(profile).any():
        return np.nan
    return float(np.nanmean(np.maximum(profile, 0)))


def peak_time(profile: np.ndarray) -> int:
    if not np.isfinite(profile).any():
        return -1
    return int(np.nanargmax(np.abs(profile)))


def split_genes(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [x.strip().upper() for x in str(value).split(",") if x.strip()]


def aggregate_day_robustness(
    family_members: list[str],
    trajectory: str,
    effect_type: str,
    contrast: str,
    day: pd.DataFrame,
    interaction: pd.DataFrame,
) -> float:
    if effect_type == "pairwise_contrast":
        contrasts = [contrast]
    else:
        contrasts = EFFECT_TO_ROBUSTNESS_CONTRAST.get(effect_type, [])
    vals = []
    for c in contrasts:
        if c == "interaction_did":
            sub = interaction[(interaction["trajectory"] == trajectory) & (interaction["pathway"].isin(family_members))]
            vals.extend(sub["day_consistency"].astype(float).tolist())
        else:
            sub = day[
                (day["trajectory"] == trajectory)
                & (day["contrast"] == c)
                & (day["pathway"].isin(family_members))
            ]
            vals.extend(sub["day_robustness_score"].astype(float).tolist())
    return float(np.nanmean(vals)) if vals else np.nan


def aggregate_matched_state(
    family_members: list[str],
    trajectory: str,
    effect_type: str,
    contrast: str,
    matched: pd.DataFrame,
    interaction: pd.DataFrame,
) -> tuple[bool, float]:
    if effect_type == "pairwise_contrast":
        contrasts = [contrast]
    else:
        contrasts = EFFECT_TO_ROBUSTNESS_CONTRAST.get(effect_type, [])
    vals = []
    for c in contrasts:
        if c == "interaction_did":
            sub = interaction[(interaction["trajectory"] == trajectory) & (interaction["pathway"].isin(family_members))]
            vals.extend(sub["matched_state_preserved"].astype(bool).tolist())
        else:
            sub = matched[
                (matched["trajectory"] == trajectory)
                & (matched["contrast"] == c)
                & (matched["pathway"].isin(family_members))
            ]
            vals.extend(sub["direction_preserved"].astype(bool).tolist())
    frac = float(np.mean(vals)) if vals else np.nan
    return bool(frac >= 0.75) if np.isfinite(frac) else False, frac


def aggregate_null_pass(
    family_members: list[str],
    trajectory: str,
    effect_type: str,
    contrast: str,
    nulls: pd.DataFrame,
    interaction: pd.DataFrame,
) -> tuple[bool, float]:
    if effect_type == "interaction":
        sub = interaction[(interaction["trajectory"] == trajectory) & (interaction["pathway"].isin(family_members))]
        vals = ((sub["interaction_q"].astype(float) <= 0.05) & sub["matched_state_preserved"].astype(bool)).tolist()
    else:
        contrasts = [contrast] if effect_type == "pairwise_contrast" else EFFECT_TO_ROBUSTNESS_CONTRAST.get(effect_type, [])
        vals = []
        for c in contrasts:
            if c == "interaction_did":
                continue
            sub = nulls[
                (nulls["trajectory"] == trajectory)
                & (nulls["contrast"] == c)
                & (nulls["pathway"].isin(family_members))
            ]
            vals.extend(
                (
                    sub["null2_candidate"].astype(bool)
                    & sub["fake_genotype_placebo_pass"].astype(bool)
                ).tolist()
            )
    frac = float(np.mean(vals)) if vals else np.nan
    return bool(frac >= 0.75) if np.isfinite(frac) else False, frac


def external_status(
    family_id: str,
    members: list[str],
    effect_type: str,
    contrast: str,
    gse214810: pd.DataFrame,
    marder_event: pd.DataFrame,
    marder_overlap: pd.DataFrame,
) -> tuple[str, str]:
    core = {"ERYTHROID_MATURATION", "HEME_GLOBIN", "GATA_KLF_TAL1_REGULON", "IRON_TRANSPORT"}
    if family_id != "ERYTHROID_EVENT_LOSS_FAMILY":
        return "not_applicable_non_erythroid_family", "not_applicable_non_erythroid_family"

    gse_status = "not_applicable_T21_context_only"
    if effect_type in {"GATA1s", "interaction", "pairwise_contrast"}:
        sub = gse214810[gse214810["pathway"].isin(core)]
        n_dir = int(sub["direction_consistent"].astype(bool).sum()) if not sub.empty else 0
        n_driver = int(sub["alignment_status"].astype(str).str.contains("driver_supported").sum()) if not sub.empty else 0
        gse_status = f"direction_supported_{n_dir}_of_{len(core)};driver_supported_{n_driver}_of_{len(core)}"

    sub_event = marder_event[marder_event["pathway"].isin(core)]
    sub_overlap = marder_overlap[marder_overlap["pathway"].isin(core)]
    n_rna_peak = int(sub_event["support_status"].astype(str).eq("rna_and_peak_supported").sum()) if not sub_event.empty else 0
    n_driver = int((sub_overlap["driver_support_fraction"].astype(float) >= 0.5).sum()) if not sub_overlap.empty else 0
    marder_status = f"rna_peak_supported_{n_rna_peak}_of_{len(core)};driver_overlap_supported_{n_driver}_of_{len(core)}"
    return gse_status, marder_status


def build_family_definition(gene_sets: dict[str, list[str]], out_dir: Path) -> pd.DataFrame:
    rows = []
    for family_id, meta in FAMILY_DEFINITIONS.items():
        simple, redundancy = family_weights(meta["members"], gene_sets)
        for pathway in meta["members"]:
            genes = set(gene_sets.get(pathway, []))
            rows.append(
                {
                    "family_id": family_id,
                    "pathway": pathway,
                    "pathway_present": pathway in gene_sets,
                    "n_genes": len(genes),
                    "simple_weight": simple.get(pathway, np.nan),
                    "redundancy_aware_weight": redundancy.get(pathway, np.nan),
                    "primary_direction": meta["primary_direction"],
                    "merge_basis": meta["merge_basis"],
                    "description": meta["description"],
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "gse271399_event_family_definition_v2.tsv", sep="\t", index=False)
    return df


def load_inputs(dataset_dir: Path, external_dir: Path) -> dict[str, pd.DataFrame]:
    ted = dataset_dir / "ted"
    return {
        "pert": pd.read_csv(ted / "gse271399_perturbation_event_fdr.tsv", sep="\t"),
        "diff": pd.read_csv(ted / "gse271399_differential_event_table.tsv", sep="\t"),
        "day": pd.read_csv(ted / "gse271399_day_robustness_score.tsv", sep="\t"),
        "matched": pd.read_csv(ted / "gse271399_matched_state_contrast_v2.tsv", sep="\t"),
        "drivers": pd.read_csv(ted / "gse271399_event_driver_mechanism_table.tsv", sep="\t"),
        "interaction": pd.read_csv(ted / "gse271399_interaction_did_event_fdr.tsv", sep="\t"),
        "nulls": pd.read_csv(ted / "gse271399_null_sensitivity_summary.tsv", sep="\t"),
        "gse214810": pd.read_csv(external_dir / "gse214810_event_alignment_to_gse271399.tsv", sep="\t"),
        "marder_event": pd.read_csv(external_dir / "marderstein_t21_erythroid_event_validation.tsv", sep="\t"),
        "marder_overlap": pd.read_csv(external_dir / "marderstein_t21_scRNA_multiome_driver_overlap.tsv", sep="\t"),
    }


def pathway_profiles_for_factorial(pert: pd.DataFrame, trajectory: str, effect_type: str, members: list[str]) -> tuple[dict[str, np.ndarray], list[float]]:
    profiles = {}
    pvals = []
    sub = pert[(pert["trajectory"] == trajectory) & (pert["effect_type"] == effect_type) & (pert["pathway"].isin(members))]
    for _, row in sub.iterrows():
        profiles[str(row["pathway"])] = parse_profile(row["effect_profile"])
        pvals.append(float(row["p_value"]))
    return profiles, pvals


def pathway_profiles_for_pairwise(diff: pd.DataFrame, trajectory: str, contrast: str, members: list[str]) -> tuple[dict[str, np.ndarray], list[float]]:
    profiles = {}
    pvals = []
    sub = diff[(diff["trajectory"] == trajectory) & (diff["contrast"] == contrast) & (diff["pathway"].isin(members))]
    for _, row in sub.iterrows():
        profiles[str(row["pathway"])] = parse_profile(row["delta_profile"])
        pvals.append(float(row["p_value"]))
    return profiles, pvals


def family_driver_summary(
    family_id: str,
    trajectory: str,
    effect_type: str,
    contrast: str,
    members: list[str],
    drivers: pd.DataFrame,
    interaction: pd.DataFrame,
    gse214810: pd.DataFrame,
    marder_overlap: pd.DataFrame,
) -> dict[str, object]:
    if effect_type == "pairwise_contrast":
        contrasts = [contrast]
    elif effect_type == "interaction":
        contrasts = ["T21_GATA1s_vs_T21_wtGATA1"]
    elif effect_type == "T21":
        contrasts = ["T21_wtGATA1_vs_Euploid_wtGATA1"]
    else:
        contrasts = ["Euploid_GATA1s_vs_Euploid_wtGATA1", "T21_GATA1s_vs_T21_wtGATA1"]

    support_vals = []
    gene_counter: Counter[str] = Counter()
    class_counter: Counter[str] = Counter()
    for c in contrasts:
        sub = drivers[(drivers["trajectory"] == trajectory) & (drivers["contrast"] == c) & (drivers["pathway"].isin(members))]
        for _, row in sub.iterrows():
            if np.isfinite(float(row["signed_driver_support"])):
                support_vals.append(float(row["signed_driver_support"]))
            for gene in split_genes(row.get("top_signed_driver_genes"))[:10]:
                gene_counter[gene] += 1
            for klass in split_genes(row.get("driver_classes")):
                class_counter[klass] += 1
    if effect_type == "interaction":
        sub_i = interaction[(interaction["trajectory"] == trajectory) & (interaction["pathway"].isin(members))]
        support_vals.extend(sub_i["signed_driver_support"].astype(float).tolist())

    gse_shared = []
    if family_id == "ERYTHROID_EVENT_LOSS_FAMILY":
        for value in gse214810.get("shared_driver_genes", pd.Series(dtype=str)).tolist():
            gse_shared.extend(split_genes(value))
    marder_supported = []
    if family_id == "ERYTHROID_EVENT_LOSS_FAMILY":
        for value in marder_overlap.get("marderstein_multiome_supported_drivers", pd.Series(dtype=str)).tolist():
            marder_supported.extend(split_genes(value))

    top_genes = [g for g, _ in gene_counter.most_common(20)]
    return {
        "family_id": family_id,
        "trajectory": trajectory,
        "effect_type": effect_type,
        "contrast": contrast,
        "signed_driver_support": float(np.nanmean(support_vals)) if support_vals else np.nan,
        "core_driver_genes": ",".join(top_genes),
        "driver_classes": ",".join(f"{k}:{v}" for k, v in class_counter.most_common()),
        "gse214810_shared_driver_genes": ",".join(sorted(set(gse_shared) & set(top_genes))),
        "marderstein_supported_driver_genes": ",".join(sorted(set(marder_supported) & set(top_genes))),
        "driver_summary_pass": bool((np.nanmean(support_vals) if support_vals else np.nan) >= 0.7),
    }


def build_family_inference(dataset_dir: Path, external_dir: Path, out_dir: Path) -> None:
    gene_sets = read_gmt(dataset_dir / "metadata" / "gse271399_ted_gene_sets.gmt")
    family_def = build_family_definition(gene_sets, out_dir)
    inputs = load_inputs(dataset_dir, external_dir)
    pert = inputs["pert"]
    diff = inputs["diff"]
    trajectories = sorted(set(pert["trajectory"]))

    fdr_rows = []
    effect_rows = []
    driver_rows = []
    for family_id, meta in FAMILY_DEFINITIONS.items():
        members = [m for m in meta["members"] if m in gene_sets]
        simple_w, red_w = family_weights(members, gene_sets)
        for trajectory in trajectories:
            jobs: list[tuple[str, str, str]] = [
                ("T21", "NA", "factorial"),
                ("GATA1s", "NA", "factorial"),
                ("interaction", "NA", "factorial"),
            ]
            jobs.extend(("pairwise_contrast", c, "pairwise") for c in PAIRWISE_CONTRASTS)
            for effect_type, contrast, mode in jobs:
                if mode == "factorial":
                    profiles, pvals = pathway_profiles_for_factorial(pert, trajectory, effect_type, members)
                else:
                    profiles, pvals = pathway_profiles_for_pairwise(diff, trajectory, contrast, members)
                if not profiles:
                    continue
                for scheme, weights in [("simple_average", simple_w), ("redundancy_aware", red_w)]:
                    profile = weighted_profile(profiles, weights)
                    family_delta = auc(profile)
                    family_peak = peak_time(profile)
                    p = fisher_p(pvals)
                    fdr_rows.append(
                        {
                            "family_id": family_id,
                            "trajectory": trajectory,
                            "effect_type": effect_type,
                            "contrast": contrast,
                            "weighting_scheme": scheme,
                            "n_member_pathways": len(profiles),
                            "member_pathways": ",".join(profiles.keys()),
                            "family_delta_auc": family_delta,
                            "family_abs_auc": abs_auc(profile),
                            "family_loss_auc": loss_auc(profile),
                            "family_gain_auc": gain_auc(profile),
                            "family_direction": sign_label(family_delta),
                            "family_peak_time": family_peak,
                            "family_profile": profile_to_text(profile),
                            "p_value": p,
                        }
                    )

                # Default decomposition uses redundancy-aware aggregation.
                profile = weighted_profile(profiles, red_w)
                family_delta = auc(profile)
                day_score = aggregate_day_robustness(
                    members, trajectory, effect_type, contrast, inputs["day"], inputs["interaction"]
                )
                matched_pass, matched_frac = aggregate_matched_state(
                    members, trajectory, effect_type, contrast, inputs["matched"], inputs["interaction"]
                )
                null_pass, null_frac = aggregate_null_pass(
                    members, trajectory, effect_type, contrast, inputs["nulls"], inputs["interaction"]
                )
                driver_summary = family_driver_summary(
                    family_id,
                    trajectory,
                    effect_type,
                    contrast,
                    members,
                    inputs["drivers"],
                    inputs["interaction"],
                    inputs["gse214810"],
                    inputs["marder_overlap"],
                )
                gse_status, marder_status = external_status(
                    family_id,
                    members,
                    effect_type,
                    contrast,
                    inputs["gse214810"],
                    inputs["marder_event"],
                    inputs["marder_overlap"],
                )
                signed_support = driver_summary["signed_driver_support"]
                p = fisher_p(pvals)
                effect_rows.append(
                    {
                        "family_id": family_id,
                        "trajectory": trajectory,
                        "effect_type": effect_type,
                        "contrast": contrast,
                        "family_delta_auc": family_delta,
                        "family_direction": sign_label(family_delta),
                        "family_peak_time": peak_time(profile),
                        "family_q": np.nan,  # filled after BH
                        "day_robustness_score": day_score,
                        "matched_state_preserved": matched_pass,
                        "matched_state_preserved_fraction": matched_frac,
                        "signed_driver_support": signed_support,
                        "stratified_null_pass": null_pass,
                        "stratified_null_pass_fraction": null_frac,
                        "gse214810_validation_status": gse_status,
                        "marderstein_multiome_support": marder_status,
                        "internal_candidate_pass": False,
                        "external_supported": False,
                        "strong_candidate_pass": False,
                        "family_profile": profile_to_text(profile),
                        "p_value": p,
                    }
                )
                driver_rows.append(driver_summary)

    fdr = pd.DataFrame(fdr_rows)
    fdr["family_q"] = bh_fdr(fdr["p_value"]) if not fdr.empty else []
    fdr.to_csv(out_dir / "gse271399_family_level_perturbation_fdr.tsv", sep="\t", index=False)

    effect = pd.DataFrame(effect_rows)
    effect["family_q"] = bh_fdr(effect["p_value"]) if not effect.empty else []
    effect["internal_candidate_pass"] = (
        (effect["family_q"].astype(float) <= 0.05)
        & (effect["day_robustness_score"].astype(float) >= (2 / 3))
        & (effect["matched_state_preserved"].astype(bool))
        & (effect["signed_driver_support"].astype(float) >= 0.7)
        & (effect["stratified_null_pass"].astype(bool))
    )
    effect["external_supported"] = (
        effect["gse214810_validation_status"].astype(str).str.contains("direction_supported_4_of_4")
        & (
            effect["marderstein_multiome_support"].astype(str).str.contains("rna_peak_supported_4_of_4")
            | effect["marderstein_multiome_support"].astype(str).str.contains("driver_overlap_supported_4_of_4")
        )
    )
    effect["strong_candidate_pass"] = effect["internal_candidate_pass"] & effect["external_supported"]
    effect = effect[
        [
            "family_id",
            "trajectory",
            "effect_type",
            "contrast",
            "family_delta_auc",
            "family_direction",
            "family_peak_time",
            "family_q",
            "day_robustness_score",
            "matched_state_preserved",
            "signed_driver_support",
            "stratified_null_pass",
            "gse214810_validation_status",
            "marderstein_multiome_support",
            "internal_candidate_pass",
            "external_supported",
            "strong_candidate_pass",
            "family_profile",
            "p_value",
            "matched_state_preserved_fraction",
            "stratified_null_pass_fraction",
        ]
    ]
    effect.to_csv(out_dir / "gse271399_family_level_effect_decomposition.tsv", sep="\t", index=False)

    drivers = pd.DataFrame(driver_rows).drop_duplicates(["family_id", "trajectory", "effect_type", "contrast"])
    drivers.to_csv(out_dir / "gse271399_family_level_driver_summary.tsv", sep="\t", index=False)

    # A compact go/no-go view is useful for reading, but the requested files above
    # remain the canonical outputs.
    compact = effect.sort_values(["strong_candidate_pass", "family_q"], ascending=[False, True]).copy()
    compact.to_csv(out_dir / "gse271399_family_level_candidate_summary.tsv", sep="\t", index=False)


def copy_outputs(out_dir: Path, dataset_dir: Path, root: Path) -> None:
    deliverables = dataset_dir / "deliverables_family_level_inference_v2"
    bundle = root / "deliverables_all_ted_rounds" / "GSE271399_T21_GATA1s"
    deliverables.mkdir(parents=True, exist_ok=True)
    bundle.mkdir(parents=True, exist_ok=True)
    for file in sorted(out_dir.glob("gse271399_family_level*.tsv")) + [
        out_dir / "gse271399_event_family_definition_v2.tsv"
    ]:
        if file.exists():
            shutil.copy2(file, deliverables / file.name)
            shutil.copy2(file, bundle / file.name)
    manifest = pd.DataFrame(
        [
            {"file_name": p.name, "bytes": p.stat().st_size, "path": str(p.resolve())}
            for p in sorted(deliverables.glob("*.tsv"))
        ]
    )
    manifest.to_csv(deliverables / "bundle_manifest.tsv", sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data_external")
    args = parser.parse_args()
    root = Path(args.root)
    dataset_dir = root / "GSE271399_T21_GATA1s"
    external_dir = root / "external_validation" / "deliverables_external_validation"
    out_dir = dataset_dir / "ted"
    build_family_inference(dataset_dir, external_dir, out_dir)
    copy_outputs(out_dir, dataset_dir, root)
    print("[family-v2] done", flush=True)


if __name__ == "__main__":
    main()
