"""Independently recalculate the frozen BNT162b2 flagship evidence summary.

This audit intentionally reads only persisted TSV/JSON artifacts.  It does not
call the analysis functions that created those artifacts and does not access
cell-level RNA or ADT values.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
RNA = ROOT / "results" / "ted_bnt162b2_flagship" / "rna_event_freeze_v1"
PROTEIN = ROOT / "results" / "ted_bnt162b2_flagship" / "orthogonal_outcome_v1"
REPLICATION = ROOT / "results" / "ted_gse171964_replication" / "analysis_v1"
FINAL = ROOT / "results" / "ted_bnt162b2_flagship" / "final_evidence_v1"
CONFIG = ROOT / "config" / "ted_bnt162b2_flagship_v1.yaml"
PRIMARY = "HALLMARK_INTERFERON_ALPHA_RESPONSE"


def exact_max_t(effects: pd.DataFrame) -> pd.DataFrame:
    """Independent exhaustive one-sided sign-flip/maxT implementation."""

    matrix = effects.to_numpy(dtype=float)
    n = len(matrix)
    signs = np.asarray(list(product((-1.0, 1.0), repeat=n)), dtype=float)
    scales = np.sqrt(np.mean(matrix**2, axis=0))
    safe = np.where(scales <= np.finfo(float).eps, 1.0, scales)
    null = (signs @ matrix) / n / safe
    null[:, scales <= np.finfo(float).eps] = 0.0
    observed = np.mean(matrix, axis=0) / safe
    max_null = np.max(null, axis=1)
    return pd.DataFrame(
        {
            "pathway": effects.columns,
            "mean_effect": np.mean(matrix, axis=0),
            "exact_raw_p": np.mean(null >= observed[None, :], axis=0),
            "exact_maxT_p": np.mean(max_null[:, None] >= observed[None, :], axis=0),
        }
    ).set_index("pathway")


def pivot_contrasts(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table = pd.read_csv(path, sep="\t")
    kwargs = {"index": "donor_id", "columns": "pathway"}
    return tuple(table.pivot(values=value, **kwargs) for value in ("transient", "activation", "recovery"))  # type: ignore[return-value]


def lodo_retention(
    transient: pd.DataFrame,
    activation: pd.DataFrame,
    recovery: pd.DataFrame,
    pathway: str,
    *,
    alpha: float = 0.10,
    direction_min: float = 0.80,
) -> float:
    retained: list[bool] = []
    for donor in transient.index:
        keep = transient.index != donor
        inference = exact_max_t(transient.loc[keep])
        a = activation.loc[keep, pathway].to_numpy(float)
        r = recovery.loc[keep, pathway].to_numpy(float)
        d = transient.loc[keep, pathway].to_numpy(float)
        mode = (
            a.mean() > 0
            and r.mean() > 0
            and np.mean(a > 0) >= direction_min
            and np.mean(r > 0) >= direction_min
        )
        retained.append(
            bool(
                inference.loc[pathway, "exact_maxT_p"] <= alpha
                and np.mean(d > 0) >= direction_min
                and mode
            )
        )
    return float(np.mean(retained))


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    rna_status = json.loads((RNA / "rna_event_status.json").read_text(encoding="utf-8"))
    protein_status = json.loads(
        (PROTEIN / "protein_outcome_status.json").read_text(encoding="utf-8")
    )
    replication_status = json.loads(
        (REPLICATION / "replication_status.json").read_text(encoding="utf-8")
    )
    final_status = json.loads(
        (FINAL / "final_evidence_summary.json").read_text(encoding="utf-8")
    )

    rows: list[dict[str, object]] = []

    def check(name: str, recalculated: object, reported: object, source: str) -> None:
        if isinstance(recalculated, (float, np.floating)) and isinstance(
            reported, (float, int)
        ):
            passed = bool(np.isclose(float(recalculated), float(reported), rtol=1e-10, atol=1e-12))
        else:
            passed = recalculated == reported
        rows.append(
            {
                "metric": name,
                "recalculated": recalculated,
                "reported": reported,
                "audit_pass": passed,
                "source": source,
            }
        )

    rna_t, rna_a, rna_r = pivot_contrasts(RNA / "pathway_donor_contrasts.tsv")
    rna_inference = exact_max_t(rna_t)
    primary_rna = rna_t[PRIMARY].to_numpy(float)
    check("rna.primary_mean_transient_effect", primary_rna.mean(), rna_status["primary_mean_transient_effect"], "pathway_donor_contrasts.tsv")
    check("rna.primary_family_adjusted_p", rna_inference.loc[PRIMARY, "exact_maxT_p"], rna_status["primary_family_adjusted_p"], "pathway_donor_contrasts.tsv")
    check("rna.direction_fraction", np.mean(primary_rna > 0), rna_status["direction_fraction"], "pathway_donor_contrasts.tsv")
    check("rna.activation_direction_fraction", np.mean(rna_a[PRIMARY].to_numpy(float) > 0), rna_status["activation_direction_fraction"], "pathway_donor_contrasts.tsv")
    check("rna.recovery_direction_fraction", np.mean(rna_r[PRIMARY].to_numpy(float) > 0), rna_status["recovery_direction_fraction"], "pathway_donor_contrasts.tsv")
    check("rna.lodo_retention_fraction", lodo_retention(rna_t, rna_a, rna_r, PRIMARY), rna_status["lodo_retention_fraction"], "pathway_donor_contrasts.tsv")

    matched = pd.read_csv(RNA / "pathway_donor_contrasts_state_matched.tsv", sep="\t")
    matched_effect = float(matched.loc[matched["pathway"].eq(PRIMARY), "transient"].mean())
    attenuation = max(0.0, 1.0 - abs(matched_effect) / abs(float(primary_rna.mean())))
    check("rna.state_matched_effect", matched_effect, rna_status["state_matched_effect"], "pathway_donor_contrasts_state_matched.tsv")
    check("rna.state_match_attenuation", attenuation, rna_status["state_match_attenuation"], "pathway_donor_contrasts_state_matched.tsv")

    diagnostics = pd.read_csv(RNA / "state_matching_diagnostics.tsv", sep="\t")
    complete = bool(
        diagnostics["converged"].all()
        and diagnostics[["max_abs_smd", "effective_sample_size", "max_weight_ratio_to_uniform"]].notna().all().all()
    )
    check("rna.state_matching_all_converged", complete, rna_status["state_matching_all_converged"], "state_matching_diagnostics.tsv")

    controls = pd.read_csv(RNA / "negative_controls.tsv", sep="\t")
    q = float(config["negative_controls"]["control_quantile"])
    random_q95 = float(controls.loc[controls["control_type"].eq("matched_random_gene_set"), "mean_transient_effect"].abs().quantile(q))
    competing_max = float(controls.loc[controls["control_type"].eq("competing_program"), "mean_transient_effect"].abs().max())
    rna_margin = float(primary_rna.mean()) - max(random_q95, competing_max)
    check("rna.negative_control_margin", rna_margin, rna_status["negative_control_margin"], "negative_controls.tsv")

    protein_table = pd.read_csv(PROTEIN / "protein_donor_contrasts.tsv", sep="\t").set_index("donor_id")
    protein_t = protein_table[["transient"]].rename(columns={"transient": "CD64_CD169_index"})
    protein_a = protein_table[["activation"]].rename(columns={"activation": "CD64_CD169_index"})
    protein_r = protein_table[["recovery"]].rename(columns={"recovery": "CD64_CD169_index"})
    protein_inference = exact_max_t(protein_t)
    check("protein.mean_transient_effect", protein_t.iloc[:, 0].mean(), protein_status["protein_mean_transient_effect"], "protein_donor_contrasts.tsv")
    check("protein.exact_p", protein_inference.loc["CD64_CD169_index", "exact_maxT_p"], protein_status["protein_exact_p"], "protein_donor_contrasts.tsv")
    check("protein.direction_stability", np.mean(protein_t.iloc[:, 0].to_numpy(float) > 0), protein_status["protein_direction_stability"], "protein_donor_contrasts.tsv")
    check("protein.lodo_retention_fraction", lodo_retention(protein_t, protein_a, protein_r, "CD64_CD169_index"), protein_status["protein_lodo_retention_fraction"], "protein_donor_contrasts.tsv")

    common = rna_t.index.intersection(protein_t.index)
    agreement = float(np.mean(np.sign(rna_t.loc[common, PRIMARY]) == np.sign(protein_t.loc[common, "CD64_CD169_index"])))
    check("protein.rna_direction_agreement", agreement, protein_status["rna_protein_donor_direction_agreement"], "RNA and protein donor contrasts")
    protein_controls = pd.read_csv(PROTEIN / "protein_negative_controls.tsv", sep="\t")
    random_p95 = float(protein_controls.loc[protein_controls["control_type"].eq("matched_random_ADT"), "mean_transient_effect"].abs().quantile(float(config["orthogonal_endpoint"]["control_quantile"])))
    stable_max = float(protein_controls.loc[protein_controls["control_type"].eq("stable_ADT"), "mean_transient_effect"].abs().max())
    protein_margin = float(protein_t.iloc[:, 0].mean()) - max(random_p95, stable_max)
    check("protein.negative_control_margin", protein_margin, protein_status["protein_negative_control_margin"], "protein_negative_controls.tsv")

    donor_qc = pd.read_csv(REPLICATION / "donor_blind_qc.tsv", sep="\t")
    n_evaluable = int(donor_qc["all_sample_qc_pass"].sum())
    check("replication.n_evaluable_donors", n_evaluable, replication_status["n_evaluable_donors"], "donor_blind_qc.tsv")
    eligibility = "failed" if n_evaluable < int(config["design"]["minimum_evaluable_donors"]) else "passed"
    test_status = "not_run" if eligibility == "failed" else replication_status["event_replication_test_status"]
    recalculated_replication = "not_evaluable" if eligibility == "failed" else replication_status["event_replication_status"]
    check("replication.event_replication_eligibility_status", eligibility, replication_status["event_replication_eligibility_status"], "donor_blind_qc.tsv + frozen minimum")
    check("replication.event_replication_test_status", test_status, replication_status["event_replication_test_status"], "eligibility prerequisite")
    check("replication.event_replication_status", recalculated_replication, replication_status["event_replication_status"], "eligibility plus event-test status")
    check("replication.protein_outcome_replication_status", "not_tested", replication_status["outcome_replication_status"], "corrected feature-panel audit")

    expected_display = (
        "E0 | protein outcome passed | event replication not_evaluable "
        "(eligibility failed; test not_run) | protein outcome replication not_tested"
    )
    check("final.bounded_display", expected_display, final_status["bounded_display"], "three frozen status JSON files")
    check("final.event_support_code", rna_status["event_support"]["code"], final_status["event_support_code"], "RNA status + final summary")
    check("final.protein_outcome_status", protein_status["protein_outcome_status"], final_status["within_study_protein_outcome_status"], "protein status + final summary")

    audit = pd.DataFrame(rows)
    FINAL.mkdir(parents=True, exist_ok=True)
    audit.to_csv(FINAL / "independent_recalculation_audit.tsv", sep="\t", index=False)
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_method": "independent_recalculation_from_persisted_donor_level_tables",
        "n_checks": int(len(audit)),
        "n_passed": int(audit["audit_pass"].sum()),
        "all_checks_pass": bool(audit["audit_pass"].all()),
        "failed_checks": audit.loc[~audit["audit_pass"], "metric"].tolist(),
    }
    (FINAL / "independent_recalculation_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    if not result["all_checks_pass"]:
        raise SystemExit(f"Independent recalculation failed: {result['failed_checks']}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
