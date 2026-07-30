from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RNA = ROOT / "results" / "ted_bnt162b2_flagship" / "rna_event_freeze_v1" / "rna_event_status.json"
PROTEIN = ROOT / "results" / "ted_bnt162b2_flagship" / "orthogonal_outcome_v1" / "protein_outcome_status.json"
REPLICATION = ROOT / "results" / "ted_gse171964_replication" / "analysis_v1" / "replication_status.json"
OUT_DIR = ROOT / "results" / "ted_bnt162b2_flagship" / "final_evidence_v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-status-migration",
        action="store_true",
        help="refresh only after a semantics-only status-schema migration",
    )
    args = parser.parse_args()
    if OUT_DIR.exists() and not args.refresh_status_migration:
        raise SystemExit(f"Final evidence summary is create-only and already exists: {OUT_DIR}")
    rna = json.loads(RNA.read_text(encoding="utf-8"))
    protein = json.loads(PROTEIN.read_text(encoding="utf-8"))
    replication = json.loads(REPLICATION.read_text(encoding="utf-8"))
    event = rna["event_support"]["code"]
    outcome_status = protein["protein_outcome_status"]
    event_replication_eligibility = replication["event_replication_eligibility_status"]
    event_replication_test = replication["event_replication_test_status"]
    event_replication = replication["event_replication_status"]
    outcome_replication = replication["protein_outcome"]["replication_status"]
    display = (
        f"{event} | protein outcome {outcome_status} | "
        f"event replication {event_replication} "
        f"(eligibility {event_replication_eligibility}; test {event_replication_test}) | "
        f"protein outcome replication {outcome_replication}"
    )
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "event_support_code": event,
        "parallel_evidence_records": [
            {
                "record_id": "bnt162b2_cd64_cd169_same_cell",
                "evidence_type": "orthogonal_outcome",
                "status": outcome_status,
                "independence_context": "same_study_same_cells",
                "measurement": "CD64/CD169 CITE-seq ADT index",
                "controls_pass": bool(all(protein["gates"].values())),
                "replication_status": outcome_replication,
                "replication_dataset_id": "GSE171964",
            }
        ],
        "legacy_migration_fields": {
            "validation_provenance_code": protein["validation_provenance_code"],
            "evidence_boundary": protein["evidence_boundary"],
        },
        "within_study_protein_outcome_status": outcome_status,
        "event_replication_eligibility_status": event_replication_eligibility,
        "event_replication_test_status": event_replication_test,
        "event_replication_status": event_replication,
        "protein_outcome_replication_status": outcome_replication,
        "bounded_display": display,
        "manuscript_interpretation": (
            "The same-study CD64/CD169 protein outcome passed, but the RNA event "
            "remained E0; corrected GSE171964 failed the frozen event-replication "
            "eligibility prerequisite, so the event test was not run and the event "
            "replication result is not_evaluable. The same protein outcome was not tested."
        ),
        "forbidden_displays": [
            "E2 | protein outcome passed | event independently replicated",
            "E2 | protein outcome passed and independently replicated",
        ],
        "source_sha256": {
            RNA.relative_to(ROOT).as_posix(): sha256(RNA),
            PROTEIN.relative_to(ROOT).as_posix(): sha256(PROTEIN),
            REPLICATION.relative_to(ROOT).as_posix(): sha256(REPLICATION),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=args.refresh_status_migration)
    (OUT_DIR / "final_evidence_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pd.DataFrame([summary | {"source_sha256": json.dumps(summary["source_sha256"], sort_keys=True)}]).to_csv(
        OUT_DIR / "final_evidence_summary.tsv", sep="\t", index=False
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
