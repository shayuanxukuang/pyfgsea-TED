from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_builder():
    path = ROOT / "scripts" / "build_bib_companion_evidence_contracts.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_source_fixture(data_root: Path) -> None:
    module = load_builder()
    paths = {
        "rna": data_root / module.RNA_STATUS,
        "protein": data_root / module.PROTEIN_STATUS,
        "replication": data_root / module.REPLICATION_STATUS,
        "final": data_root / module.FINAL_SUMMARY,
    }
    write_json(paths["rna"], {"event_support": {"code": "E0"}})
    write_json(
        paths["protein"],
        {
            "event_support_code": "E0",
            "protein_outcome_status": "passed",
            "gates": {"prespecified_contrast": True, "controls": True},
        },
    )
    write_json(
        paths["replication"],
        {
            "event_replication_eligibility_status": "failed",
            "event_replication_test_status": "not_run",
            "event_replication_status": "not_evaluable",
            "outcome_replication_status": "not_tested",
        },
    )
    source_hashes = {
        module.RNA_STATUS.as_posix(): sha256(paths["rna"]),
        module.PROTEIN_STATUS.as_posix(): sha256(paths["protein"]),
        module.REPLICATION_STATUS.as_posix(): sha256(paths["replication"]),
    }
    write_json(
        paths["final"],
        {
            "event_support_code": "E0",
            "bounded_display": module.BOUNDED_DISPLAY,
            "source_sha256": source_hashes,
        },
    )


def test_builds_schema_valid_claim_bounded_contracts(tmp_path: Path) -> None:
    module = load_builder()
    data_root = tmp_path / "data"
    output = tmp_path / "output"
    build_source_fixture(data_root)

    report = module.build_contracts(
        data_root=data_root,
        schema_root=ROOT / "schemas",
        output_dir=output,
    )

    assert report["event_support_code"] == "E0"
    assert report["parallel_evidence_record"]["status"] == "passed"
    assert report["replication_facets"] == {
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
    assert (output / "manifest.tsv").is_file()


def test_rejects_any_event_code_upgrade(tmp_path: Path) -> None:
    module = load_builder()
    data_root = tmp_path / "data"
    build_source_fixture(data_root)
    rna_path = data_root / module.RNA_STATUS
    write_json(rna_path, {"event_support": {"code": "E2"}})

    with pytest.raises(
        module.EvidenceContractError,
        match="RNA event support",
    ):
        module.build_contracts(
            data_root=data_root,
            schema_root=ROOT / "schemas",
            output_dir=tmp_path / "output",
        )
