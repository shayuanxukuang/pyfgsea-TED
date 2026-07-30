from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "ted_gse171964_replication_v1.yaml"
DEFAULT_OUT = ROOT / "results" / "ted_gse171964_replication" / "protocol_freeze_v1"
PRIMARY_FREEZE = ROOT / "results" / "ted_bnt162b2_flagship" / "protocol_freeze_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def validate_config(config: dict[str, object]) -> None:
    required = {
        "protocol",
        "source",
        "design",
        "population",
        "pathway_family",
        "state_matching",
        "negative_controls",
        "event_replication",
        "protein_outcome_replication",
        "gates",
        "claim_boundary",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing replication protocol sections: {missing}")

    protocol = config["protocol"]
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be a mapping")
    if protocol.get("prospective_preregistration_claimed") is not False:
        raise ValueError("This known-source replication freeze must not claim prospective preregistration")
    if protocol.get("no_retuning_after_freeze") is not True:
        raise ValueError("Replication thresholds must be frozen")
    if protocol.get("expression_values_accessed_before_freeze") is not False:
        raise ValueError("Expression values must not be accessed before the replication freeze")

    source = config["source"]
    if source.get("corrected_release") != "GEO_v2_2022-02-07":
        raise ValueError("Only the corrected GEO v2 release is allowed")
    for spec in source.get("files", {}).values():
        if "_v2" not in str(spec.get("name", "")):
            raise ValueError("Every replication input must be a corrected v2 file")

    mapping = config["design"]["public_sample_sheet_mapping"]
    if mapping != {"baseline": 21, "early_post_dose": 22, "recovery": 28, "late_phase": 42}:
        raise ValueError("The booster-episode time mapping is not the frozen mapping")

    event = config["event_replication"]
    if event.get("evaluable_donor_direction_fraction_min") != 0.80:
        raise ValueError("Event replication requires at least 80% donor-direction agreement")
    if event.get("family_adjusted_p_max") != 0.10 or event.get("no_gate_retuning") is not True:
        raise ValueError("Event replication multiplicity/no-retuning gates changed")

    protein = config["protein_outcome_replication"]
    audit = protein.get("panel_audit_result", {})
    if protein.get("initial_status") != "not_tested":
        raise ValueError("GSE171964 protein replication must start as not_tested")
    if audit.get("CD64_ADT_present") is not False or audit.get("CD169_ADT_present") is not False:
        raise ValueError("The corrected v2 feature-panel audit is inconsistent")

    claims = config["claim_boundary"]
    if claims.get("protein_and_event_pass_text_allowed") is not False:
        raise ValueError("The independently replicated protein claim is prohibited for this panel")


def verify_small_metadata(config: dict[str, object]) -> list[dict[str, object]]:
    source_dir = ROOT / "data_external" / "GSE171964_BNT162b2_replication" / "source"
    rows: list[dict[str, object]] = []
    for role in ("barcodes", "features", "phenotype"):
        spec = config["source"]["files"][role]
        path = source_dir / spec["name"]
        if not path.is_file():
            raise FileNotFoundError(f"Required public metadata file is missing: {path}")
        observed_size = path.stat().st_size
        observed_hash = sha256(path)
        if observed_size != int(spec["size_bytes"]):
            raise ValueError(f"Size mismatch for {path.name}: {observed_size}")
        if observed_hash != spec["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {path.name}")
        rows.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "size_bytes": observed_size,
                "sha256": observed_hash,
                "role": f"public_{role}_metadata_seen_before_expression_freeze",
            }
        )
    return rows


def verify_existing(config_path: Path, outdir: Path) -> None:
    freeze_path = outdir / "protocol_freeze.json"
    manifest_path = outdir / "protocol_manifest.tsv"
    gmt_path = outdir / "locked_pathway_family.gmt"
    for path in (freeze_path, manifest_path, gmt_path):
        if not path.is_file():
            raise SystemExit(f"Incomplete existing freeze: missing {path}")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze["config_sha256"] != sha256(config_path):
        raise SystemExit("Existing replication freeze does not match the current config")
    if freeze["locked_pathway_family_sha256"] != sha256(gmt_path):
        raise SystemExit("Replication pathway-family hash mismatch")
    print(f"Verified existing create-only replication freeze: {outdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or verify the GSE171964 TED replication freeze")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    outdir = args.outdir.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config)

    if outdir.exists():
        verify_existing(config_path, outdir)
        return

    metadata_rows = verify_small_metadata(config)
    primary_gmt = PRIMARY_FREEZE / "locked_pathway_family.gmt"
    primary_registry = PRIMARY_FREEZE / "locked_pathway_registry.tsv"
    if not primary_gmt.is_file() or not primary_registry.is_file():
        raise FileNotFoundError("The primary flagship pathway freeze must exist first")

    outdir.mkdir(parents=True, exist_ok=False)
    gmt_path = outdir / "locked_pathway_family.gmt"
    registry_path = outdir / "locked_pathway_registry.tsv"
    shutil.copyfile(primary_gmt, gmt_path)
    shutil.copyfile(primary_registry, registry_path)

    status_lines = git_output("status", "--porcelain").splitlines()
    freeze = {
        "protocol_id": config["protocol"]["id"],
        "protocol_version": config["protocol"]["version"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "retrospective_known_source_replication": True,
        "prospective_preregistration_claimed": False,
        "known_direction_from_publication": True,
        "feature_and_sample_metadata_accessed_before_freeze": True,
        "expression_values_accessed_before_freeze": False,
        "protein_panel_compatibility_adjudicated_before_expression_access": True,
        "protein_outcome_replication_initial_status": "not_tested",
        "config_sha256": sha256(config_path),
        "locked_pathway_family_sha256": sha256(gmt_path),
        "primary_locked_pathway_family_sha256": sha256(primary_gmt),
        "git_commit": git_output("rev-parse", "HEAD"),
        "worktree_dirty": bool(status_lines),
        "dirty_entry_count": len(status_lines),
    }
    (outdir / "protocol_freeze.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")

    manifest_rows = [
        {
            "path": config_path.relative_to(ROOT).as_posix(),
            "size_bytes": config_path.stat().st_size,
            "sha256": sha256(config_path),
            "role": "replication_analysis_contract",
        },
        {
            "path": "locked_pathway_family.gmt",
            "size_bytes": gmt_path.stat().st_size,
            "sha256": sha256(gmt_path),
            "role": "exact_copy_of_primary_frozen_pathway_family",
        },
        {
            "path": "locked_pathway_registry.tsv",
            "size_bytes": registry_path.stat().st_size,
            "sha256": sha256(registry_path),
            "role": "exact_copy_of_primary_frozen_pathway_registry",
        },
        *metadata_rows,
    ]
    pd.DataFrame(manifest_rows).to_csv(outdir / "protocol_manifest.tsv", sep="\t", index=False)
    print(f"Created GSE171964 replication freeze before expression download: {outdir}")


if __name__ == "__main__":
    main()
