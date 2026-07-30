from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "scripts", ROOT / "reproduce"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from build_ted_bib_companion import (  # noqa: E402
    CompanionBuildError,
    build_bundle,
    dry_run,
)
from verify_and_reproduce_figures import run as reproduce_run  # noqa: E402
from verify_ted_bib_companion import (  # noqa: E402
    CompanionVerificationError,
    archive_name,
    verify_bundle,
)


SPEC_FIELDS = [
    "rule_id",
    "package",
    "mode",
    "source",
    "destination",
    "required",
    "role",
    "member_base",
    "member_destination_prefix",
    "path_column",
    "bytes_column",
    "sha256_column",
    "expected_rows",
    "expected_bytes",
    "expected_sha256",
    "root",
]


CASE_STUDY_PATHS = [
    "data_external/bnt162b2_cite_asap_2023/download_manifest.tsv",
    "data_external/bnt162b2_cite_asap_2023/download_manifest.json",
    "results/ted_bnt162b2_flagship/protocol_freeze_v1/protocol_freeze.json",
    "results/ted_bnt162b2_flagship/protocol_freeze_v1/protocol_manifest.tsv",
    "results/ted_bnt162b2_flagship/rna_event_freeze_v1/rna_event_status.json",
    "results/ted_bnt162b2_flagship/rna_event_freeze_v1/"
    "rna_event_gate_table.tsv",
    "results/ted_bnt162b2_flagship/orthogonal_outcome_v1/"
    "protein_outcome_status.json",
    "results/ted_bnt162b2_flagship/orthogonal_outcome_v1/"
    "protein_outcome_gate_table.tsv",
    "data_external/GSE171964_BNT162b2_replication/download_manifest.tsv",
    "data_external/GSE171964_BNT162b2_replication/download_manifest.json",
    "results/ted_gse171964_replication/protocol_freeze_v1/protocol_freeze.json",
    "results/ted_gse171964_replication/protocol_freeze_v1/protocol_manifest.tsv",
    "results/ted_gse171964_replication/analysis_v1/replication_status.json",
    "results/ted_gse171964_replication/analysis_v1/replication_gate_table.tsv",
    "results/ted_gse171964_replication/analysis_v1/analysis_manifest.tsv",
]


FIGURE3_SOURCES = [
    "figure3_clean_common_task_metrics.tsv",
    "figure3_type_specific_clean_metrics.tsv",
    "figure3_low_signal_noisy_coordinate_metrics.tsv",
    "figure3_artifact_common_task_metrics.tsv",
]


FIGURE5_SOURCES = [
    "figure5_primary_rna_trajectory.tsv",
    "figure5_primary_protein_trajectory.tsv",
    "figure5_rna_protein_donor_contrasts.tsv",
    "figure5_gse171964_blind_qc.tsv",
    "figure5_flagship_design.tsv",
    "figure5_rna_gate_audit.tsv",
    "figure5_evidence_status.tsv",
]

FOCUSED_TEST_FILES = [
    "tests/test_ted_evidence.py",
    "tests/test_ted_schema.py",
    "tests/test_ted_flagship.py",
    "tests/test_gse171964_replication_freeze.py",
    "tests/test_nearest_method_benchmark.py",
]

FOCUSED_EXCLUDED_TESTS = [
    "test_parallel_evidence_types_never_upgrade_event_e_code",
    "test_canonical_parallel_evidence_schema_matches_embedded_v2_contract",
    "test_canonical_parallel_evidence_schema_enforces_controls_and_replication_id",
    "test_canonical_replication_facets_schema_matches_embedded_v2_contract",
    "test_canonical_replication_facets_keep_event_and_outcome_results_separate",
]


def checksum(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_payload(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def write_bytes(root: Path, relative: str, payload: bytes) -> None:
    path = root / Path(*relative.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def write_tsv(
    root: Path,
    relative: str,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        raise ValueError("synthetic TSV needs at least one row")
    path = root / Path(*relative.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def replace_packaged_member(
    bundle: Path,
    member: str,
    replacement: bytes,
) -> None:
    core = bundle / archive_name("core")
    with zipfile.ZipFile(core, "r") as archive:
        payloads = {
            name: archive.read(name)
            for name in archive.namelist()
        }
    assert member in payloads
    payloads[member] = replacement

    manifest_text = payloads["PACKAGE_MANIFEST.tsv"].decode("utf-8")
    reader = csv.DictReader(io.StringIO(manifest_text), delimiter="\t")
    manifest_rows = list(reader)
    assert reader.fieldnames is not None
    target_rows = [
        row for row in manifest_rows if row.get("path") == member
    ]
    assert len(target_rows) == 1
    target_rows[0]["bytes"] = str(len(replacement))
    target_rows[0]["sha256"] = checksum(replacement)
    manifest_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        manifest_buffer,
        fieldnames=reader.fieldnames,
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(manifest_rows)
    payloads["PACKAGE_MANIFEST.tsv"] = manifest_buffer.getvalue().encode(
        "utf-8"
    )

    replacement_archive = core.with_name(f"{core.name}.rewritten")
    with zipfile.ZipFile(
        replacement_archive,
        "w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for name in sorted(payloads):
            archive.writestr(name, payloads[name])
    replacement_archive.replace(core)

    outer_path = bundle / "ARCHIVE_MANIFEST.tsv"
    with outer_path.open("r", encoding="utf-8", newline="") as handle:
        outer_reader = csv.DictReader(handle, delimiter="\t")
        outer_rows = list(outer_reader)
        assert outer_reader.fieldnames is not None
    core_rows = [
        row for row in outer_rows if row.get("file") == core.name
    ]
    assert len(core_rows) == 1
    core_rows[0]["bytes"] = str(core.stat().st_size)
    core_rows[0]["sha256"] = checksum(core.read_bytes())
    with outer_path.open("w", encoding="utf-8", newline="") as handle:
        outer_writer = csv.DictWriter(
            handle,
            fieldnames=outer_reader.fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        outer_writer.writeheader()
        outer_writer.writerows(outer_rows)


def file_rule(
    rule_id: str,
    source: str,
    role: str,
    *,
    package: str = "core",
    root: str = "data",
    destination: str | None = None,
) -> dict[str, str]:
    return {
        "rule_id": rule_id,
        "package": package,
        "mode": "file",
        "source": source,
        "destination": destination or source,
        "required": "true",
        "role": role,
        "root": root,
    }


def manifest_rule(
    rule_id: str,
    *,
    package: str,
    source: str,
    role: str,
    member_base: str,
    member_destination_prefix: str,
    expected_rows: str,
    root: str = "data",
    path_column: str = "file",
    destination: str | None = None,
) -> dict[str, str]:
    return {
        "rule_id": rule_id,
        "package": package,
        "mode": "manifest_members",
        "source": source,
        "destination": destination or source,
        "required": "true",
        "role": role,
        "member_base": member_base,
        "member_destination_prefix": member_destination_prefix,
        "path_column": path_column,
        "bytes_column": "bytes",
        "sha256_column": "sha256",
        "expected_rows": expected_rows,
        "root": root,
    }


def create_fixture(tmp_path: Path) -> tuple[Path, Path, Path, str, str]:
    source = tmp_path / "source"
    source.mkdir()
    rules: list[dict[str, str]] = []
    machine = "results/ted_manuscript_machine_readable_v2"
    status_path = f"{machine}/common_task_status.json"
    write_bytes(
        source,
        status_path,
        (
            json.dumps(
                {
                    "expected_tasks": 480,
                    "completed_tasks": 480,
                    "complete": True,
                    "methods": [
                        "TIPS",
                        "scTransient",
                        "tradeSeq",
                        "score_then_smooth",
                        "TED",
                    ],
                }
            )
            + "\n"
        ).encode(),
    )
    rules.append(file_rule("common.status", status_path, "common_task_status"))
    registry_path = f"{machine}/common_task_scenario_registry.tsv"
    write_bytes(source, registry_path, b"scenario\treplicates\nall\t480\n")
    rules.append(
        file_rule("common.registry", registry_path, "common_task_registry")
    )
    for name, role in [
        ("common_task_truth_masked.tsv", "post_output_truth"),
        ("method_harmonized_event_outputs.tsv", "harmonized_predictions"),
    ]:
        path = f"{machine}/{name}"
        write_bytes(source, path, b"id\tvalue\n1\tfixture\n")
        rules.append(file_rule(f"common.{name}", path, role))

    native_manifest = f"{machine}/method_native_outputs/manifest.tsv"
    native_rows: list[tuple[str, int, str]] = []
    for index in range(2400):
        relative = (
            "results/ted_nearest_method_five_method/locked/"
            f"task_{index:04d}/native.tsv"
        )
        payload = f"task\tvalue\n{index}\t{index % 5}\n".encode()
        write_bytes(source, relative, payload)
        native_rows.append((relative, len(payload), checksum(payload)))
    native_manifest_path = source / Path(*native_manifest.split("/"))
    native_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with native_manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["file", "bytes", "sha256"])
        writer.writerows(native_rows)
    rules.append(
        file_rule(
            "common.native_manifest",
            native_manifest,
            "native_output_manifest",
        )
    )
    rules.append(
        manifest_rule(
            "native.members",
            package="native_outputs",
            source=native_manifest,
            role="native_output",
            member_base="source_root",
            member_destination_prefix="",
            expected_rows="2400",
        )
    )

    figure_source_root = "results/ted_v1_submission/figure_source_data"
    typed_expected = {
        "activation": [0.855, 0.631, 0.599, 0.682, 0.580],
        "suppression": [0.447, 0.239, 0.287, 0.375, 0.447],
        "transient": [0.102, 0.716, 0.681, 0.499, 0.422],
    }
    clean_expected = [0.803, 0.994, 1.000, 1.000, 1.000]
    hard_expected = [0.674, 0.671, 0.751, 0.846, 0.764]
    risk_expected = [0.890, 0.881, 0.745, 0.614, 0.811]
    clean_rows = [
        {
            "method": method,
            "pathway_level_auprc": clean_expected[method_index],
        }
        for method_index, method in enumerate(
            ["TIPS", "scTransient", "tradeSeq", "score_then_smooth", "TED"]
        )
        for _ in range(60)
    ]
    typed_rows = [
        {
            "method": method,
            "event_type": event_type,
            "pathway_auprc": typed_expected[event_type][method_index],
        }
        for method_index, method in enumerate(
            ["TIPS", "scTransient", "tradeSeq", "score_then_smooth", "TED"]
        )
        for event_type in ("activation", "suppression", "transient")
        for _ in range(120)
    ]
    hard_rows = [
        {
            "method": method,
            "pathway_level_auprc": hard_expected[method_index],
        }
        for method_index, method in enumerate(
            ["TIPS", "scTransient", "tradeSeq", "score_then_smooth", "TED"]
        )
        for _ in range(120)
    ]
    artifact_rows = [
        {
            "method": method,
            "artifact": artifact,
            "matched_top_k_artifact_false_promotion_rate": risk_expected[
                method_index
            ],
        }
        for method_index, method in enumerate(
            ["TIPS", "scTransient", "tradeSeq", "score_then_smooth", "TED"]
        )
        for artifact in ("composition", "stress", "partial_batch_time")
        for _ in range(120)
    ]
    for name, rows in zip(
        FIGURE3_SOURCES,
        [clean_rows, typed_rows, hard_rows, artifact_rows],
        strict=True,
    ):
        path = f"{figure_source_root}/{name}"
        write_tsv(source, path, rows)
        rules.append(file_rule(f"fig3.{name}", path, "figure3_source"))

    donors = [f"donor{letter}" for letter in "ABCDEF"]
    days = [0, 2, 10, 28]
    rna_rows = [
        {
            "donor_id": donor,
            "day": day,
            "selected_cells": 100,
            "IFN_alpha_score": (
                {0: 0.0, 2: 0.5, 10: 0.1, 28: 0.0}[day]
                + donor_index * 0.01
            ),
        }
        for donor_index, donor in enumerate(donors)
        for day in days
    ]
    protein_rows = [
        {
            "donor_id": donor,
            "day": day,
            "CD64_CD169_index": (
                {0: 0.0, 2: 0.8, 10: 0.2, 28: 0.0}[day]
                + donor_index * 0.01
            ),
            "CD64": 1.0 + donor_index * 0.01,
            "CD169": 1.2 + donor_index * 0.01,
        }
        for donor_index, donor in enumerate(donors)
        for day in days
    ]
    aligned_rows = [
        {
            "donor_id": donor,
            "RNA_activation": 0.5 + index * 0.01,
            "RNA_recovery": -0.4,
            "RNA_transient": 0.4 + index * 0.01,
            "protein_activation": 0.8 + index * 0.01,
            "protein_recovery": -0.6,
            "protein_transient": 0.7 + index * 0.01,
        }
        for index, donor in enumerate(donors)
    ]
    participants = ["2047", "2049", "2052", "2053", "2051", "2055"]
    qc_rows = [
        {
            "pt_id": participant,
            "day": day,
            "sample_id": f"{participant}_{day}",
            "n_cells": 100,
            "median_rna_umi": 1000,
            "median_detected_genes": 500,
            "median_rna_umi_abs_mad_z": (
                4.0
                if participant in {"2051", "2055"} and day == 22
                else 1.0
            ),
            "median_detected_genes_abs_mad_z": 1.0,
            "blind_qc_pass": not (
                participant in {"2051", "2055"} and day == 22
            ),
            "max_abs_mad_z": (
                4.0
                if participant in {"2051", "2055"} and day == 22
                else 1.0
            ),
        }
        for participant in participants
        for day in [21, 22, 28, 42]
    ]
    design_rows = [
        {
            "donor_id": donor,
            "day": day,
            "modality": modality,
            "availability": "absent" if modality == "ATAC" else "available",
            "masking_or_use": "synthetic fixture",
        }
        for donor in donors
        for day in days
        for modality in ("RNA", "protein_ADT", "ATAC")
    ]
    gate_rows = [
        {
            "gate": gate,
            "observed": observed,
            "frozen_requirement": requirement,
            "passed": passed,
        }
        for gate, observed, requirement, passed in [
            ("family maxT p", 0.0625, "<=0.10", True),
            ("donor direction", 5 / 6, ">=0.80", True),
            ("LODO retention", 0.5, ">=0.80", False),
            ("matched-state attenuation", 0.81, "<=0.50", False),
            ("negative-control margin", -0.014, ">0", False),
        ]
    ]
    evidence_rows = [
        {
            "primary_event_support": "E0",
            "primary_validation_provenance": "V1",
            "primary_evidence_boundary": "E0-V1",
            "primary_protein_outcome_status": "passed",
            "event_replication_eligibility_status": "failed",
            "event_replication_test_status": "not_run",
            "event_replication_status": "not_evaluable",
            "protein_outcome_replication_status": "not_tested",
            "event_replication_attempt_status": (
                "failed_at_eligibility_prerequisite"
            ),
            "event_replication_reason": "insufficient_frozen_QC_donors",
        }
    ]
    for name, rows in zip(
        FIGURE5_SOURCES,
        [
            rna_rows,
            protein_rows,
            aligned_rows,
            qc_rows,
            design_rows,
            gate_rows,
            evidence_rows,
        ],
        strict=True,
    ):
        path = f"{figure_source_root}/{name}"
        write_tsv(source, path, rows)
        rules.append(file_rule(f"fig5.{name}", path, "figure5_source"))

    figure_stems = {
        "figure3": "figure3_primary_heldout_performance",
        "figure5": "figure5_independent_real_data_validation",
    }
    for figure, stem in figure_stems.items():
        for extension in ("pdf", "png"):
            path = (
                "results/bib_manuscript_revision/figures/"
                f"{stem}.{extension}"
            )
            write_bytes(source, path, f"{figure}-{extension}".encode())
            rules.append(
                file_rule(
                    f"{figure}.{extension}",
                    path,
                    f"{figure}_output",
                )
            )

    schema_path = "schemas/ted_event_report_v2.schema.json"
    write_bytes(
        source,
        schema_path,
        (
            json.dumps(
                {
                    "type": "object",
                    "properties": {
                        name: {}
                        for name in [
                            "parallel_evidence_records",
                            "event_replication_eligibility_status",
                            "event_replication_test_status",
                            "event_replication_status",
                            "outcome_replication_status",
                        ]
                    },
                }
            )
            + "\n"
        ).encode(),
    )
    rules.append(
        file_rule(
            "schema.event_v2",
            schema_path,
            "schema",
            root="repository",
        )
    )
    for schema_name in (
        "ted_activity_table_v1.schema.json",
        "ted_event_report_v1.schema.json",
    ):
        package_source = f"pyfgsea/schemas/{schema_name}"
        destination = f"schemas/{schema_name}"
        write_bytes(
            source,
            package_source,
            b'{"type": "object", "properties": {}}\n',
        )
        rules.append(
            file_rule(
                f"schema.package.{schema_name}",
                package_source,
                "schema",
                root="repository",
                destination=destination,
            )
        )
    parallel_schema_path = "schemas/parallel_evidence_record_v1.schema.json"
    write_bytes(
        source,
        parallel_schema_path,
        (
            json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "record_id": {"type": "string"},
                        "evidence_type": {
                            "enum": [
                                "orthogonal_outcome",
                                "intervention_reversal",
                                "matched_rescue",
                            ]
                        },
                        "status": {"type": "string"},
                        "independence_context": {"type": "string"},
                        "controls_pass": {"type": ["boolean", "null"]},
                        "replication_status": {"type": "string"},
                        "reason_codes": {"type": "array"},
                    },
                }
            )
            + "\n"
        ).encode(),
    )
    rules.append(
        file_rule(
            "schema.parallel",
            parallel_schema_path,
            "schema",
            root="repository",
        )
    )
    replication_schema_path = "schemas/replication_facets_v1.schema.json"
    write_bytes(
        source,
        replication_schema_path,
        (
            json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "event_replication_eligibility_status": {},
                        "event_replication_test_status": {},
                        "event_replication_status": {},
                        "outcome_replication_status": {},
                        "replication_dataset_id": {},
                    },
                }
            )
            + "\n"
        ).encode(),
    )
    rules.append(
        file_rule(
            "schema.replication",
            replication_schema_path,
            "schema",
            root="repository",
        )
    )

    bounded_display = (
        "E0 | protein outcome passed | event replication not_evaluable "
        "(eligibility failed; test not_run) | protein outcome replication "
        "not_tested"
    )
    parallel_record = {
        "record_id": "bnt162b2_cd64_cd169_protein_outcome",
        "evidence_type": "orthogonal_outcome",
        "status": "passed",
        "independence_context": "same_study_different_assay",
        "outcome_type": "CD64/CD169 protein",
        "contrast": "day_2_vs_day_0",
        "controls_pass": True,
        "replication_status": "not_tested",
        "replication_dataset_id": None,
        "reason_codes": ["masked_outcome_gate_passed"],
    }
    replication_facets = {
        "event_replication_eligibility_status": "failed",
        "event_replication_test_status": "not_run",
        "event_replication_status": "not_evaluable",
        "outcome_replication_status": "not_tested",
        "outcome_type": "protein",
        "replication_dataset_id": "GSE171964",
        "replication_reason_codes": ["insufficient_frozen_qc_donors"],
    }
    claim_boundary = {
        "event_support_code": "E0",
        "parallel_evidence_record": parallel_record,
        "replication_facets": replication_facets,
        "bounded_display": bounded_display,
        "legacy_fields_are_source_provenance_only": True,
    }
    contract_root = "results/ted_bib_companion_evidence_contract_v1"
    contract_payloads = {
        "parallel_evidence_record_v1.json": json_payload(parallel_record),
        "replication_facets_v1.json": json_payload(replication_facets),
        "claim_boundary_v1.json": json_payload(claim_boundary),
    }
    for name, payload in contract_payloads.items():
        path = f"{contract_root}/{name}"
        write_bytes(source, path, payload)
        rules.append(
            file_rule(
                f"claim.contract.{name}",
                path,
                "canonical_claim_contract",
            )
        )
    contract_manifest_path = f"{contract_root}/manifest.tsv"
    write_tsv(
        source,
        contract_manifest_path,
        [
            {
                "file": name,
                "bytes": len(payload),
                "sha256": checksum(payload),
            }
            for name, payload in contract_payloads.items()
        ],
    )
    rules.append(
        file_rule(
            "claim.contract.manifest",
            contract_manifest_path,
            "canonical_claim_manifest",
        )
    )

    final_root = "results/ted_bnt162b2_flagship/final_evidence_v1"
    final_summary = {
        "event_support_code": "E0",
        "within_study_protein_outcome_status": "passed",
        "event_replication_eligibility_status": "failed",
        "event_replication_test_status": "not_run",
        "event_replication_status": "not_evaluable",
        "protein_outcome_replication_status": "not_tested",
        "bounded_display": bounded_display,
    }
    final_summary_json = f"{final_root}/final_evidence_summary.json"
    write_bytes(source, final_summary_json, json_payload(final_summary))
    rules.append(
        file_rule(
            "claim.final.summary.json",
            final_summary_json,
            "final_claim_boundary",
        )
    )
    final_summary_tsv = f"{final_root}/final_evidence_summary.tsv"
    write_tsv(source, final_summary_tsv, [final_summary])
    rules.append(
        file_rule(
            "claim.final.summary.tsv",
            final_summary_tsv,
            "final_claim_boundary",
        )
    )
    audit = {
        "all_checks_pass": True,
        "event_support_code_recalculated": "E0",
        "parallel_outcome_status_recalculated": "passed",
        "replication_status_recalculated": "not_evaluable",
    }
    audit_json = f"{final_root}/independent_recalculation_audit.json"
    write_bytes(source, audit_json, json_payload(audit))
    rules.append(
        file_rule(
            "claim.final.audit.json",
            audit_json,
            "final_claim_audit",
        )
    )
    audit_tsv = f"{final_root}/independent_recalculation_audit.tsv"
    write_tsv(
        source,
        audit_tsv,
        [
            {
                "check": "canonical_claim_boundary",
                "passed": "true",
                "observed": bounded_display,
            }
        ],
    )
    rules.append(
        file_rule(
            "claim.final.audit.tsv",
            audit_tsv,
            "final_claim_audit",
        )
    )

    claim_builder_path = "scripts/build_bib_companion_evidence_contracts.py"
    write_bytes(
        source,
        claim_builder_path,
        b'"""Synthetic canonical-claim fixture builder."""\n',
    )
    rules.append(
        file_rule(
            "claim.contract.builder",
            claim_builder_path,
            "canonical_claim_builder",
            root="repository",
        )
    )
    claim_readme_path = "release/ted-v1.1.0/CLAIM_BOUNDARY.md"
    write_bytes(
        source,
        claim_readme_path,
        b"# Synthetic claim-boundary fixture\n",
    )
    rules.append(
        file_rule(
            "claim.contract.readme",
            claim_readme_path,
            "claim_boundary_documentation",
            root="repository",
        )
    )

    focused = "results/ted_bib_focused_81"
    collection_path = f"{focused}/focused_81_collection.txt"
    node_ids = [
        (
            f"{FOCUSED_TEST_FILES[index % len(FOCUSED_TEST_FILES)]}"
            f"::test_fixture_{index:03d}"
        )
        for index in range(81)
    ]
    collection_payload = (
        "\n".join(node_ids) + "\n\n81 tests collected in 0.10s\n"
    ).encode("utf-8")
    write_bytes(source, collection_path, collection_payload)
    rules.append(
        file_rule(
            "focused.collection",
            collection_path,
            "focused_81_test_evidence",
            root="repository",
        )
    )
    junit_path = f"{focused}/focused_81_junit.xml"
    junit_payload = (
        b'<?xml version="1.0"?>'
        b'<testsuites><testsuite tests="81" failures="0" errors="0" '
        b'skipped="0"/></testsuites>'
    )
    write_bytes(source, junit_path, junit_payload)
    rules.append(
        file_rule(
            "focused.junit",
            junit_path,
            "focused_81_test_evidence",
            root="repository",
        )
    )
    summary_path = f"{focused}/focused_81_terminal_summary.txt"
    summary_payload = b"================ 81 passed in 1.00s ================\n"
    write_bytes(source, summary_path, summary_payload)
    rules.append(
        file_rule(
            "focused.summary",
            summary_path,
            "focused_81_test_evidence",
            root="repository",
        )
    )
    command_path = f"{focused}/focused_81_command.json"
    write_bytes(source, command_path, b'{"status": "pending analysis lock"}\n')
    rules.append(
        file_rule(
            "focused.command",
            command_path,
            "focused_81_test_evidence",
            root="repository",
        )
    )
    reproduction_assets = [
        (
            "reproduction.entry",
            "reproduce/verify_and_reproduce_figures.py",
            "reproduce/verify_and_reproduce_figures.py",
            ROOT / "reproduce" / "verify_and_reproduce_figures.py",
            "reproduction_script",
        ),
        (
            "reproduction.renderer",
            "reproduce/render_bib_figures.py",
            "reproduce/render_bib_figures.py",
            ROOT / "reproduce" / "render_bib_figures.py",
            "reproduction_script",
        ),
        (
            "reproduction.contract",
            "reproduce/FIGURE_RENDERERS.json",
            "reproduction/FIGURE_RENDERERS.json",
            ROOT / "reproduce" / "FIGURE_RENDERERS.json",
            "reproduction_contract",
        ),
    ]
    for rule_id, source_path, destination, local_path, role in reproduction_assets:
        write_bytes(source, source_path, local_path.read_bytes())
        rules.append(
            file_rule(
                rule_id,
                source_path,
                role,
                root="repository",
                destination=destination,
            )
        )
    lock_path = "requirements-reproduction-py311.txt"
    write_bytes(
        source,
        lock_path,
        (
            b"matplotlib==3.10.0\n"
            b"numpy==2.2.0\n"
            b"pandas==2.2.0\n"
            b"pillow==11.0.0\n"
        ),
    )
    rules.append(
        file_rule(
            "reproduction.lock",
            lock_path,
            "reproduction_dependency_lock",
            root="repository",
        )
    )

    case_json = {
        "results/ted_bnt162b2_flagship/rna_event_freeze_v1/"
        "rna_event_status.json": {
            "event_support": {"code": "E0"},
        },
        "results/ted_bnt162b2_flagship/orthogonal_outcome_v1/"
        "protein_outcome_status.json": {
            "protein_outcome_status": "passed",
        },
        "results/ted_gse171964_replication/analysis_v1/"
        "replication_status.json": {
            "event_replication_eligibility_status": "failed",
            "event_replication_test_status": "not_run",
            "event_replication_status": "not_evaluable",
            "protein_outcome": {"replication_status": "not_tested"},
            "n_evaluable_donors": 4,
        },
    }
    for index, path in enumerate(CASE_STUDY_PATHS):
        payload = (
            (json.dumps(case_json.get(path, {"status": "fixture"})) + "\n").encode()
            if path.endswith(".json")
            else b"field\tvalue\nstatus\tfixture\n"
        )
        write_bytes(source, path, payload)
        rules.append(file_rule(f"case.{index}", path, "case_study"))

    stability_root = (
        "results/ted_submission_supplement/"
        "zscape_repeated_holdout_stability"
    )
    stability_rows: list[tuple[str, int, str]] = []
    for index in range(2):
        name = f"shard_{index}.tsv"
        payload = f"repeat\tvalue\n{index}\t1\n".encode()
        write_bytes(source, f"{stability_root}/{name}", payload)
        stability_rows.append((name, len(payload), checksum(payload)))
    stability_manifest = f"{stability_root}/manifest.tsv"
    stability_manifest_path = source / Path(*stability_manifest.split("/"))
    with stability_manifest_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["file", "bytes", "sha256"])
        writer.writerows(stability_rows)
    reconciled_manifest = (
        "release/ted-v1.1.0/STABILITY_SHARDS_MANIFEST.tsv"
    )
    reconciled_path = source / Path(*reconciled_manifest.split("/"))
    reconciled_path.parent.mkdir(parents=True, exist_ok=True)
    with reconciled_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["file", "source_path", "bytes", "sha256"])
        writer.writerows(
            (
                name,
                f"{stability_root}/{name}",
                size,
                digest,
            )
            for name, size, digest in stability_rows
        )
    rules.append(
        file_rule(
            "stability.historical",
            stability_manifest,
            "historical_provenance",
            package="stability_shards",
            destination=f"{stability_root}/manifest.historical.tsv",
        )
    )
    rules.append(
        manifest_rule(
            "stability.members",
            package="stability_shards",
            source=reconciled_manifest,
            role="stability_shard",
            member_base="data_root",
            member_destination_prefix="",
            expected_rows="2",
            root="repository",
            path_column="source_path",
            destination=stability_manifest,
        )
    )

    stray = "results/ted_nearest_method_five_method/locked/stray.txt"
    write_bytes(source, stray, b"must not be packaged\n")

    spec = tmp_path / "asset_rules.tsv"
    with spec.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SPEC_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rules:
            writer.writerow(row)
    repository = tmp_path / "repository"
    repository.mkdir()
    for row in rules:
        if row.get("root") != "repository":
            continue
        relative = str(row["source"])
        origin = source / Path(*relative.split("/"))
        target = repository / Path(*relative.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(origin), str(target))
    tracked_spec = (
        repository
        / "release"
        / "ted-v1.1.0"
        / "TEST_ASSET_RULES.tsv"
    )
    tracked_spec.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(spec), str(tracked_spec))
    (repository / ".gitignore").write_text(
        "/results/ted_bib_focused_81/\n",
        encoding="utf-8",
    )
    commands = [
        ["git", "init", "-q"],
        ["git", "config", "user.email", "fixture@example.invalid"],
        ["git", "config", "user.name", "Fixture"],
        ["git", "add", "-A"],
        ["git", "commit", "-q", "-m", "fixture analysis lock"],
    ]
    for command in commands:
        subprocess.run(command, cwd=repository, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    selection_payload = ("\n".join(node_ids) + "\n").encode("utf-8")
    command_record = {
        "evidence_kind": "ted_v1.1.0_focused_81",
        "analysis_lock_commit": commit,
        "repository_dirty": False,
        "argv": [
            "/fixture/python",
            "-I",
            "-c",
            "<embedded installed-wheel guard>",
            "/fixture/repository",
            "/fixture/site-packages",
            "-q",
            "-p",
            "no:cacheprovider",
            *FOCUSED_TEST_FILES,
            "-k",
            " and ".join(
                f"not {name}" for name in FOCUSED_EXCLUDED_TESTS
            ),
            "--import-mode=importlib",
            f"--junitxml={junit_path}",
        ],
        "environment_controls": {
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": "removed",
            "isolated_child_python": True,
            "installed_distribution_root_inserted_by_guard": (
                "/fixture/site-packages"
            ),
        },
        "runtime": {
            "python": "3.11.9",
            "python_executable": "/fixture/python",
            "platform": "synthetic-linux",
            "pyfgsea": {
                "version": "0.2.0",
                "import_origin": "/fixture/site-packages/pyfgsea/__init__.py",
                "import_root": "/fixture/site-packages",
            },
            "pytest": "8.3.5",
        },
        "selection": {
            "test_files": FOCUSED_TEST_FILES,
            "excluded_v1_1_extension_tests": FOCUSED_EXCLUDED_TESTS,
            "collected": 81,
            "node_ids": node_ids,
            "selection_manifest_sha256": checksum(selection_payload),
        },
        "result": {
            "tests": 81,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
        },
        "evidence_sha256": {
            "collection": checksum(collection_payload),
            "junit": checksum(junit_payload),
            "terminal_summary": checksum(summary_payload),
        },
    }
    write_bytes(repository, command_path, json_payload(command_record))
    for relative in (
        collection_path,
        junit_path,
        summary_path,
        command_path,
    ):
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", "--", relative],
            cwd=repository,
            check=False,
        )
        assert ignored.returncode == 0
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    assert status == ""
    return repository, source, tracked_spec, native_rows[0][0], commit


def test_build_verify_and_actual_source_data_redraw(tmp_path: Path) -> None:
    repository, data, spec, _, commit = create_fixture(tmp_path)
    output = tmp_path / "bundle"
    built = build_bundle(
        repository_root=repository,
        data_root=data,
        analysis_lock_commit=commit,
        output_dir=output,
        spec_path=spec,
    )
    report = verify_bundle(built)
    assert report["status"] == "verified"
    assert report["common_tasks"] == 480
    assert report["native_method_task_outputs"] == 2400
    assert report["analysis_lock_commit"] == commit
    assert b"\r\n" not in (built / "BUILD_INDEX.json").read_bytes()

    figure_report, return_code = reproduce_run(
        built,
        redraw_requested=True,
        extract_dir=None,
    )
    assert return_code == 0
    assert figure_report["status"] == "verified"
    assert (
        figure_report["redraw_status"]
        == "redrawn_and_semantically_verified"
    )
    assert set(figure_report["redraw_results"]) == {"figure3", "figure5"}
    assert all(
        item["status"] == "redrawn_and_semantically_verified"
        for item in figure_report["redraw_results"].values()
    )

    with zipfile.ZipFile(built / archive_name("native_outputs"), "r") as archive:
        assert (
            "results/ted_nearest_method_five_method/locked/stray.txt"
            not in archive.namelist()
        )


def test_dry_run_hashes_nested_members_and_writes_nothing(
    tmp_path: Path,
) -> None:
    repository, data, spec, first_native, commit = create_fixture(tmp_path)
    output = tmp_path / "must_not_exist"
    report = dry_run(
        repository_root=repository,
        data_root=data,
        analysis_lock_commit=commit,
        spec_path=spec,
    )
    assert report["status"] == "dry_run_verified"
    assert report["writes_performed"] is False
    assert report["analysis_lock_commit"] == commit
    assert not output.exists()

    lock_file = repository / "requirements-reproduction-py311.txt"
    original_lock = lock_file.read_bytes()
    lock_file.write_bytes(original_lock + b"# dirty\n")
    with pytest.raises(CompanionBuildError, match="not tracked-clean"):
        dry_run(
            repository_root=repository,
            data_root=data,
            analysis_lock_commit=commit,
            spec_path=spec,
        )
    lock_file.write_bytes(original_lock)

    write_bytes(data, first_native, b"tampered\n")
    with pytest.raises(CompanionBuildError, match="size mismatch|SHA-256 mismatch"):
        dry_run(
            repository_root=repository,
            data_root=data,
            analysis_lock_commit=commit,
            spec_path=spec,
        )


def test_missing_focused_evidence_fails_closed(tmp_path: Path) -> None:
    repository, data, spec, _, commit = create_fixture(tmp_path)
    missing = (
        repository
        / "results"
        / "ted_bib_focused_81"
        / "focused_81_junit.xml"
    )
    missing.unlink()
    alternative = (
        tmp_path
        / "unbound_evidence_root"
        / "results"
        / "ted_bib_focused_81"
        / "focused_81_junit.xml"
    )
    alternative.parent.mkdir(parents=True)
    alternative.write_text(
        '<testsuite tests="81" failures="0" errors="0"/>',
        encoding="utf-8",
    )
    with pytest.raises(CompanionBuildError, match="required source is missing"):
        build_bundle(
            repository_root=repository,
            data_root=data,
            analysis_lock_commit=commit,
            output_dir=tmp_path / "bundle",
            spec_path=spec,
        )


def test_outer_archive_tamper_is_detected(tmp_path: Path) -> None:
    repository, data, spec, _, commit = create_fixture(tmp_path)
    output = build_bundle(
        repository_root=repository,
        data_root=data,
        analysis_lock_commit=commit,
        output_dir=tmp_path / "bundle",
        spec_path=spec,
    )
    core = output / archive_name("core")
    payload = bytearray(core.read_bytes())
    payload[len(payload) // 2] ^= 1
    core.write_bytes(payload)
    with pytest.raises(
        CompanionVerificationError,
        match="outer member integrity mismatch",
    ):
        verify_bundle(output)


def test_semantic_claim_and_command_tamper_are_detected(
    tmp_path: Path,
) -> None:
    repository, data, spec, _, commit = create_fixture(tmp_path)
    pristine = build_bundle(
        repository_root=repository,
        data_root=data,
        analysis_lock_commit=commit,
        output_dir=tmp_path / "bundle",
        spec_path=spec,
    )
    verify_bundle(pristine)

    claim_bundle = tmp_path / "claim_tamper"
    shutil.copytree(pristine, claim_bundle)
    claim_member = (
        "results/ted_bib_companion_evidence_contract_v1/"
        "claim_boundary_v1.json"
    )
    with zipfile.ZipFile(
        claim_bundle / archive_name("core"),
        "r",
    ) as archive:
        claim = json.loads(archive.read(claim_member))
    claim["event_support_code"] = "E2"
    replace_packaged_member(
        claim_bundle,
        claim_member,
        json_payload(claim),
    )
    with pytest.raises(
        CompanionVerificationError,
        match="canonical BNT/GSE claim contract",
    ):
        verify_bundle(claim_bundle)

    command_bundle = tmp_path / "command_tamper"
    shutil.copytree(pristine, command_bundle)
    command_member = (
        "results/ted_bib_focused_81/focused_81_command.json"
    )
    with zipfile.ZipFile(
        command_bundle / archive_name("core"),
        "r",
    ) as archive:
        command = json.loads(archive.read(command_member))
    command["analysis_lock_commit"] = "0" * 40
    replace_packaged_member(
        command_bundle,
        command_member,
        json_payload(command),
    )
    with pytest.raises(
        CompanionVerificationError,
        match="focused-81 command record disagrees",
    ):
        verify_bundle(command_bundle)


def test_external_asset_template_uses_canonical_manifest_contract() -> None:
    path = (
        ROOT
        / "release"
        / "ted-v1.1.0"
        / "EXTERNAL_ARCHIVE_ASSETS.template.tsv"
    )
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    assert reader.fieldnames == [
        "asset_name",
        "asset_role",
        "archive_location",
        "source_ref_or_manifest",
        "bytes",
        "sha256",
        "verification_status",
        "notes",
    ]
    allowed = {
        "pending_not_built",
        "pending_external_upload",
        "not_applicable",
        "verified",
    }
    assert rows
    assert {row["verification_status"] for row in rows} <= allowed
