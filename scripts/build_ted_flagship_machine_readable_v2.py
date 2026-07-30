from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ted_manuscript_machine_readable_v2"
RNA = ROOT / "results" / "ted_bnt162b2_flagship" / "rna_event_freeze_v1"
PROTEIN = ROOT / "results" / "ted_bnt162b2_flagship" / "orthogonal_outcome_v1"
REPLICATION = ROOT / "results" / "ted_gse171964_replication" / "analysis_v1"
FINAL = ROOT / "results" / "ted_bnt162b2_flagship" / "final_evidence_v1"
DESIGN_LOCK = ROOT / "config" / "ted_bnt162b2_flagship_v1.yaml"
REPLICATION_DESIGN_LOCK = ROOT / "config" / "ted_gse171964_replication_v1.yaml"
REPLICATION_SOURCE_MANIFEST = (
    ROOT
    / "data_external"
    / "GSE171964_BNT162b2_replication"
    / "download_manifest.tsv"
)
COMMON_TASK_LOCK = ROOT / "config" / "ted_nearest_method_common_task_v1.yml"
TIPS_REPOSITORY = ROOT / "tmp" / "nearest_methods" / "TIPS"
SCTRANSIENT_REPOSITORY = ROOT / "tmp" / "nearest_methods" / "scTransient_notebooks"
TIPS_CONCORDANCE = OUT / "tips_reference_concordance_audit.tsv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_rna_audit(status: dict[str, object]) -> pd.DataFrame:
    gate_rows = [
        ("evaluable_donors", status["n_evaluable_donors"], ">=5 donors and >=100 cells per donor-time", True),
        ("family_adjusted_p", status["primary_family_adjusted_p"], "<=0.10", status["gates"]["family_adjusted_p"]),
        ("direction_stability", status["direction_fraction"], ">=0.80", status["gates"]["direction_stability"]),
        ("early_activation_direction", status["activation_direction_fraction"], ">=0.80 and positive mean", status["gates"]["early_activation_direction"]),
        ("recovery_direction", status["recovery_direction_fraction"], ">=0.80 and positive mean", status["gates"]["recovery_direction"]),
        ("leave_one_donor_selection", status["lodo_retention_fraction"], ">=0.80", status["gates"]["leave_one_donor_selection"]),
        ("matched_state_convergence", status["state_matching_all_converged"], "all fits converge", status["gates"]["matched_state"]),
        ("matched_state_attenuation", status["state_match_attenuation"], "<=0.50", status["gates"]["matched_state"]),
        ("negative_control_margin", status["negative_control_margin"], ">0", status["gates"]["negative_controls"]),
        ("upstream_score_agreement", "positive", "positive", status["gates"]["upstream_score_agreement"]),
        ("peak_day_agreement", 2, "day 2", status["gates"]["peak_day_agreement"]),
    ]
    reason_codes = "|".join(status["event_support"]["reason_codes"])
    return pd.DataFrame(
        [
            {
                "gate": gate,
                "observed": observed,
                "frozen_requirement": requirement,
                "passed": bool(passed),
                "primary_pathway": status["primary_pathway"],
                "event_support": status["event_support"]["code"],
                "event_reason_codes": reason_codes,
                "outcome_values_accessed_before_freeze": bool(status["outcome_values_accessed"]),
            }
            for gate, observed, requirement, passed in gate_rows
        ]
    )


def build_orthogonal_record(
    status: dict[str, object], replication: dict[str, object]
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "record_id": "bnt162b2_cd64_cd169_same_cell",
                "evidence_type": "orthogonal_outcome",
                "status": status["protein_outcome_status"],
                "independence_context": "same_study_same_cells",
                "outcome_type": "protein",
                "measurement": "CD64/CD169 CITE-seq ADT index",
                "contrast": "day2 - 0.5*day0 - 0.25*day10 - 0.25*day28",
                "controls_pass": bool(all(status["gates"].values())),
                "reason_codes": "PROTEIN_OUTCOME_GATES_PASS",
                "prespecified_and_masked": bool(status["unmasked_only_after_rna_event_freeze"]),
                "effect": status["protein_mean_transient_effect"],
                "exact_p": status["protein_exact_p"],
                "donor_direction_fraction": status["protein_direction_stability"],
                "rna_outcome_direction_agreement": status["rna_protein_donor_direction_agreement"],
                "lodo_retention": status["protein_lodo_retention_fraction"],
                "negative_control_margin": status["protein_negative_control_margin"],
                "event_support_after_outcome": status["event_support_code"],
                "replication_status": replication["outcome_replication_status"],
                "replication_dataset_id": "GSE171964",
                "bounded_display": (
                    f"{status['event_support_code']} | protein outcome {status['protein_outcome_status']} | "
                    f"event replication {replication['event_replication_status']} "
                    f"(eligibility {replication['event_replication_eligibility_status']}; "
                    f"test {replication['event_replication_test_status']}) | "
                    f"protein outcome replication {replication['outcome_replication_status']}"
                ),
            }
        ]
    )


def build_replication_audit(status: dict[str, object]) -> pd.DataFrame:
    gate_table = pd.read_csv(REPLICATION / "replication_gate_table.tsv", sep="\t")
    gate_table.insert(0, "dataset", status["dataset"])
    gate_table.insert(1, "corrected_release", status["corrected_release"])
    gate_table["event_replication_eligibility_status"] = status[
        "event_replication_eligibility_status"
    ]
    gate_table["event_replication_test_status"] = status["event_replication_test_status"]
    gate_table["event_replication_status"] = status["event_replication_status"]
    gate_table["protein_outcome_replication_status"] = status["outcome_replication_status"]
    gate_table["evaluable_donors"] = status["n_evaluable_donors"]
    gate_table["protein_panel_compatible"] = False
    gate_table["event_replication_attempt_status"] = status[
        "event_replication_attempt_status"
    ]
    gate_table["event_replication_reason"] = status["event_replication_reason"]
    gate_table["claim_boundary"] = (
        "event replication eligibility failed; event test not_run; "
        "event replication not_evaluable; protein outcome replication not_tested"
    )
    return gate_table


def build_migration() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["V0", "no qualified parallel evidence record", "all evidence record statuses pending/not_tested/failed", "does not change E support"],
            ["V1", "orthogonal outcome", "outcome_record.status=passed", "record outcome type, contrast, controls and independence context"],
            ["V2", "intervention reversal", "reversal_record.status=passed", "record intervention, matched controls and reversal contrast"],
            ["V3", "matched same-system rescue", "rescue_record.status=passed", "record molecular and phenotypic recovery gates"],
            ["V4", "legacy replicated validation aggregate", "set replication_status on the corresponding outcome/reversal/rescue record", "never infer outcome replication from event replication alone"],
            ["E2-V1", "E2 plus qualified orthogonal outcome", "event_support=E2; outcome_record.status=passed", "display: E2 | protein outcome passed"],
            ["E2-V1 with event-only replication", "E2 plus outcome passed plus event replication passed", "event_replication_status=passed; outcome_record.replication_status=not_tested", "display: E2 | protein outcome passed | event independently replicated"],
            ["E2-V1 with same-outcome replication", "E2 plus outcome passed and replicated", "outcome_record.replication_status=passed", "display: E2 | protein outcome passed and independently replicated"],
        ],
        columns=["legacy_code_or_pattern", "new_record_semantics", "machine_rule", "display_or_boundary"],
    )


def build_method_implementation_identity() -> pd.DataFrame:
    lock = yaml.safe_load(COMMON_TASK_LOCK.read_text(encoding="utf-8"))
    implementations = lock["implementations"]
    tips = implementations["TIPS"]
    sctransient = implementations["scTransient"]
    tradeseq = implementations["tradeSeq"]
    runtime = implementations["runtime_image"]
    tips_audit = pd.read_csv(TIPS_CONCORDANCE, sep="\t")
    if len(tips_audit) != 3 or not tips_audit["passed"].astype(bool).all():
        raise RuntimeError("TIPS author-example concordance audit is absent or failed")
    max_correlation_difference = float(tips_audit["abs_correlation_difference"].max())
    max_p_difference = float(tips_audit["p_value_difference"].max())
    wheel = (
        SCTRANSIENT_REPOSITORY
        / "method_evaluations"
        / "wavelet_pseudotime-0.0.0-py3-none-any.whl"
    )
    rows = [
        {
            "method": "TIPS",
            "native_analysis_level": "pathway",
            "upstream_asset_class": "author-released repository",
            "upstream_source": "https://github.com/qingshanni/TIPS",
            "upstream_release_or_commit": tips["repository_commit"],
            "executed_implementation_class": "TED-study wrapper of published pathway-correlation core",
            "executed_package_identity": f"Monocle {tips['monocle_version']}; {runtime['tag']}",
            "upstream_artifact_sha256": sha256(TIPS_REPOSITORY / "base" / "gdata.R"),
            "study_adapter": "none for native pathway ranking; shared curve characterizer only for harmonized event fields",
            "native_output_retained": "tips_native.tsv",
            "concordance_audit": (
                "passed: 3/3 author-example pathways; "
                f"max abs-correlation difference={max_correlation_difference:.3g}; "
                f"max p-value difference={max_p_difference:.3g}; ranks identical"
            ),
            "boundary": "core audit only; not a full Shiny/Seurat preprocessing reproduction",
        },
        {
            "method": "scTransient",
            "native_analysis_level": "gene/protein feature",
            "upstream_asset_class": "author-released notebooks plus bundled early-development wheel",
            "upstream_source": "https://github.com/xomicsdatascience/scTransient_notebooks",
            "upstream_release_or_commit": sctransient["notebooks_commit"],
            "executed_implementation_class": "author-released wheel executed for native gene-level output",
            "executed_package_identity": sctransient["wheel_version"],
            "upstream_artifact_sha256": sha256(wheel),
            "study_adapter": "TED-study pathway adapter: signed gene ranks plus 10,000 expression-matched random sets",
            "native_output_retained": "sctransient_native_gene.tsv",
            "concordance_audit": "version, commit and wheel SHA-256 verified; native gene output retained before adapter",
            "boundary": "no native pathway-level output is claimed; the pathway result is a TED-study adapter",
        },
        {
            "method": "tradeSeq",
            "native_analysis_level": "gene",
            "upstream_asset_class": "author-released Bioconductor package",
            "upstream_source": "https://bioconductor.org/packages/tradeSeq",
            "upstream_release_or_commit": str(tradeseq["version"]),
            "executed_implementation_class": "author-released package executed for native gene-level output",
            "executed_package_identity": f"tradeSeq {tradeseq['version']}; {runtime['tag']}",
            "upstream_artifact_sha256": runtime["image_id"].removeprefix("sha256:"),
            "study_adapter": "TED-study pathway adapter, identical in form to the scTransient adapter",
            "native_output_retained": "tradeseq_native_gene.tsv",
            "concordance_audit": "native package version and container digest verified",
            "boundary": "no native pathway-level output is claimed; the pathway result is a TED-study adapter",
        },
        {
            "method": "score_then_smooth",
            "native_analysis_level": "pathway score",
            "upstream_asset_class": "TED-study internal baseline",
            "upstream_source": "current repository",
            "upstream_release_or_commit": "frozen common-task lock",
            "executed_implementation_class": "internal baseline",
            "executed_package_identity": runtime["tag"],
            "upstream_artifact_sha256": sha256(COMMON_TASK_LOCK),
            "study_adapter": "not applicable",
            "native_output_retained": "direct harmonized method slice",
            "concordance_audit": "deterministic lock and serialized outputs verified",
            "boundary": "not an external published method",
        },
        {
            "method": "TED",
            "native_analysis_level": "downstream pathway event",
            "upstream_asset_class": "study method",
            "upstream_source": "current repository",
            "upstream_release_or_commit": "PyFgsea 0.1.4 common-task implementation",
            "executed_implementation_class": "native TED common-task slice",
            "executed_package_identity": runtime["tag"],
            "upstream_artifact_sha256": sha256(ROOT / "pyfgsea" / "nearest_method_benchmark.py"),
            "study_adapter": "TED-specific E, artifact and evidence gates excluded",
            "native_output_retained": "ted_native_windows.tsv",
            "concordance_audit": "serialized native windows verified before truth reveal",
            "boundary": "common-task output is not the full TED evidence-adjudication protocol",
        },
    ]
    return pd.DataFrame(rows)


def refresh_manifest() -> None:
    rows = []
    self_updating_audits = {
        "manuscript_revision_integrity_audit.tsv",
        "manuscript_revision_integrity_audit.json",
    }
    for path in sorted(OUT.rglob("*")):
        if (
            path.is_file()
            and path.name != "manifest.tsv"
            and path.name not in self_updating_audits
        ):
            rows.append(
                {
                    "file": path.relative_to(OUT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "manifest.tsv", sep="\t", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rna_status = json.loads((RNA / "rna_event_status.json").read_text(encoding="utf-8"))
    protein_status = json.loads((PROTEIN / "protein_outcome_status.json").read_text(encoding="utf-8"))
    replication_status = json.loads((REPLICATION / "replication_status.json").read_text(encoding="utf-8"))

    design = yaml.safe_load(DESIGN_LOCK.read_text(encoding="utf-8"))
    replication_design = yaml.safe_load(
        REPLICATION_DESIGN_LOCK.read_text(encoding="utf-8")
    )
    design["independent_replication_lock"] = replication_design
    replication_source_manifest = pd.read_csv(REPLICATION_SOURCE_MANIFEST, sep="\t")
    design["observed_replication_source_manifest"] = json.loads(
        replication_source_manifest.to_json(orient="records")
    )
    design["reporting_migration"] = {
        "report_schema": "event_support_plus_parallel_evidence_records_v2",
        "source_lock": DESIGN_LOCK.relative_to(ROOT).as_posix(),
        "source_lock_sha256": sha256(DESIGN_LOCK),
        "replication_source_lock": REPLICATION_DESIGN_LOCK.relative_to(ROOT).as_posix(),
        "replication_source_lock_sha256": sha256(REPLICATION_DESIGN_LOCK),
        "replication_source_manifest": REPLICATION_SOURCE_MANIFEST.relative_to(ROOT).as_posix(),
        "replication_source_manifest_sha256": sha256(REPLICATION_SOURCE_MANIFEST),
        "thresholds_changed": False,
        "legacy_v_fields_role": "migration_only",
        "event_replication_eligibility_field": "event_replication_eligibility_status",
        "event_replication_test_field": "event_replication_test_status",
        "event_replication_field": "event_replication_status",
        "protein_outcome_replication_field": "protein_outcome.replication_status",
    }
    (OUT / "flagship_design_lock.yaml").write_text(
        yaml.safe_dump(design, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    build_rna_audit(rna_status).to_csv(OUT / "flagship_rna_event_audit.tsv", sep="\t", index=False)
    build_orthogonal_record(protein_status, replication_status).to_csv(
        OUT / "flagship_orthogonal_evidence_records.tsv", sep="\t", index=False
    )
    build_replication_audit(replication_status).to_csv(
        OUT / "flagship_replication_audit.tsv", sep="\t", index=False
    )
    build_migration().to_csv(OUT / "evidence_schema_migration.tsv", sep="\t", index=False)
    build_method_implementation_identity().to_csv(
        OUT / "method_implementation_identity_audit.tsv", sep="\t", index=False
    )
    for name in ("independent_recalculation_audit.tsv", "independent_recalculation_audit.json"):
        shutil.copy2(FINAL / name, OUT / name)
    refresh_manifest()
    print(f"wrote flagship machine-readable evidence records to {OUT}")


if __name__ == "__main__":
    main()
