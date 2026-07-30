#!/usr/bin/env python3
"""Build canonical v1.1 evidence records from frozen BNT/GSE status files.

The source status files retain legacy E/V migration fields because they are
byte-preserved analysis artifacts. This adapter does not recompute scientific
results. It fail-closes on the frozen observed states and emits current,
schema-valid parallel-evidence and replication-facet records for the BIB
companion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

RNA_STATUS = Path(
    "results/ted_bnt162b2_flagship/rna_event_freeze_v1/"
    "rna_event_status.json"
)
PROTEIN_STATUS = Path(
    "results/ted_bnt162b2_flagship/orthogonal_outcome_v1/"
    "protein_outcome_status.json"
)
REPLICATION_STATUS = Path(
    "results/ted_gse171964_replication/analysis_v1/"
    "replication_status.json"
)
FINAL_SUMMARY = Path(
    "results/ted_bnt162b2_flagship/final_evidence_v1/"
    "final_evidence_summary.json"
)
DEFAULT_OUTPUT = Path("results/ted_bib_companion_evidence_contract_v1")
BOUNDED_DISPLAY = (
    "E0 | protein outcome passed | event replication not_evaluable "
    "(eligibility failed; test not_run) | protein outcome replication "
    "not_tested"
)


class EvidenceContractError(RuntimeError):
    """Frozen evidence cannot be projected into the canonical v1.1 contract."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceContractError(f"invalid source JSON: {path}") from exc
    if not isinstance(value, dict):
        raise EvidenceContractError(f"source JSON is not an object: {path}")
    return value


def require_equal(observed: object, expected: object, *, field: str) -> None:
    if observed != expected:
        raise EvidenceContractError(
            f"{field} is {observed!r}; expected frozen value {expected!r}"
        )


def validate_schema(
    instance: dict[str, object],
    schema_path: Path,
    *,
    label: str,
) -> None:
    schema = load_json(schema_path)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(instance),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        detail = "; ".join(error.message for error in errors[:5])
        raise EvidenceContractError(
            f"{label} does not satisfy {schema_path.name}: {detail}"
        )


def write_json(path: Path, value: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build_contracts(
    *,
    data_root: Path,
    schema_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    data_root = data_root.resolve()
    schema_root = schema_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise EvidenceContractError(
            f"output directory already exists; refusing overwrite: {output_dir}"
        )

    source_paths = {
        "rna": data_root / RNA_STATUS,
        "protein": data_root / PROTEIN_STATUS,
        "replication": data_root / REPLICATION_STATUS,
        "final": data_root / FINAL_SUMMARY,
    }
    rna = load_json(source_paths["rna"])
    protein = load_json(source_paths["protein"])
    replication = load_json(source_paths["replication"])
    final = load_json(source_paths["final"])

    event_support = rna.get("event_support")
    if not isinstance(event_support, dict):
        raise EvidenceContractError("RNA status lacks event_support")
    require_equal(event_support.get("code"), "E0", field="RNA event support")
    require_equal(
        protein.get("event_support_code"),
        "E0",
        field="protein-record event support",
    )
    require_equal(
        protein.get("protein_outcome_status"),
        "passed",
        field="protein outcome status",
    )
    gates = protein.get("gates")
    if not isinstance(gates, dict) or not gates or not all(
        value is True for value in gates.values()
    ):
        raise EvidenceContractError(
            "protein outcome cannot be passed unless every frozen gate is true"
        )
    require_equal(
        replication.get("event_replication_eligibility_status"),
        "failed",
        field="event replication eligibility",
    )
    require_equal(
        replication.get("event_replication_test_status"),
        "not_run",
        field="event replication test",
    )
    require_equal(
        replication.get("event_replication_status"),
        "not_evaluable",
        field="event replication result",
    )
    require_equal(
        replication.get("outcome_replication_status"),
        "not_tested",
        field="outcome replication result",
    )
    require_equal(
        final.get("event_support_code"),
        "E0",
        field="final event support",
    )
    require_equal(
        final.get("bounded_display"),
        BOUNDED_DISPLAY,
        field="final bounded display",
    )

    final_source_hashes = final.get("source_sha256")
    if not isinstance(final_source_hashes, dict):
        raise EvidenceContractError("final summary lacks source_sha256")
    for key, relative in (
        ("rna", RNA_STATUS),
        ("protein", PROTEIN_STATUS),
        ("replication", REPLICATION_STATUS),
    ):
        require_equal(
            final_source_hashes.get(relative.as_posix()),
            sha256_path(source_paths[key]),
            field=f"final source hash for {relative.as_posix()}",
        )

    parallel_record: dict[str, object] = {
        "record_id": "bnt162b2_cd64_cd169_same_cell",
        "evidence_type": "orthogonal_outcome",
        "status": "passed",
        "independence_context": "same_study_same_cells",
        "outcome_type": "protein",
        "contrast": "day2 - 0.5*day0 - 0.25*day10 - 0.25*day28",
        "controls_pass": True,
        "replication_status": "not_tested",
        "replication_dataset_id": "GSE171964",
        "reason_codes": [
            "PROTEIN_OUTCOME_GATES_PASS",
            "EVENT_E_CODE_UNCHANGED",
            "OUTCOME_REPLICATION_NOT_TESTED_INCOMPATIBLE_READOUT",
        ],
    }
    replication_facets: dict[str, object] = {
        "event_replication_eligibility_status": "failed",
        "event_replication_test_status": "not_run",
        "event_replication_status": "not_evaluable",
        "outcome_replication_status": "not_tested",
        "outcome_type": "protein",
        "replication_dataset_id": "GSE171964",
        "replication_reason_codes": [
            "INSUFFICIENT_FROZEN_QC_DONORS",
            "OUTCOME_REPLICATION_NOT_TESTED_INCOMPATIBLE_READOUT",
        ],
    }
    validate_schema(
        parallel_record,
        schema_root / "parallel_evidence_record_v1.schema.json",
        label="parallel evidence record",
    )
    validate_schema(
        replication_facets,
        schema_root / "replication_facets_v1.schema.json",
        label="replication facets",
    )

    claim_boundary: dict[str, object] = {
        "schema_version": "ted_bib_companion_claim_boundary_v1",
        "event_support_code": "E0",
        "parallel_evidence_record": parallel_record,
        "replication_facets": replication_facets,
        "bounded_display": BOUNDED_DISPLAY,
        "supported_interpretation": (
            "The same-study CD64/CD169 protein outcome passed its frozen "
            "controls while the RNA event remained E0."
        ),
        "unsupported_interpretation_current_evidence": (
            "Corrected GSE171964 was ineligible for the event-replication "
            "test, and its feature panel did not test the CD64/CD169 protein "
            "outcome; neither result upgrades the event E code."
        ),
        "legacy_fields_are_source_provenance_only": True,
        "source_sha256": {
            relative.as_posix(): sha256_path(source_paths[key])
            for key, relative in (
                ("rna", RNA_STATUS),
                ("protein", PROTEIN_STATUS),
                ("replication", REPLICATION_STATUS),
                ("final", FINAL_SUMMARY),
            )
        },
    }

    output_dir.mkdir(parents=True)
    outputs = {
        "parallel_evidence_record_v1.json": parallel_record,
        "replication_facets_v1.json": replication_facets,
        "claim_boundary_v1.json": claim_boundary,
    }
    for name, value in outputs.items():
        write_json(output_dir / name, value)
    with (output_dir / "manifest.tsv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file", "bytes", "sha256", "role"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for name, role in (
            ("parallel_evidence_record_v1.json", "parallel_evidence_record"),
            ("replication_facets_v1.json", "replication_facets"),
            ("claim_boundary_v1.json", "claim_boundary"),
        ):
            path = output_dir / name
            writer.writerow(
                {
                    "file": name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_path(path),
                    "role": role,
                }
            )
    return claim_boundary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--schema-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.data_root / DEFAULT_OUTPUT
    try:
        report = build_contracts(
            data_root=args.data_root,
            schema_root=args.schema_root,
            output_dir=output_dir,
        )
    except (EvidenceContractError, OSError, ValueError) as exc:
        print(f"evidence-contract build failed: {exc}")
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
