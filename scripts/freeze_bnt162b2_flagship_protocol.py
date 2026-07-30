from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "ted_bnt162b2_flagship_v1.yaml"
DEFAULT_OUT = ROOT / "results" / "ted_bnt162b2_flagship" / "protocol_freeze_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def fetch_library(url: str) -> dict[str, list[str]]:
    request = Request(url, headers={"User-Agent": "PyFgsea-TED/1.0"})
    with urlopen(request, timeout=120) as response:
        text = response.read().decode("utf-8")
    result: dict[str, list[str]] = {}
    for raw in text.splitlines():
        fields = [field.strip() for field in raw.split("\t")]
        if len(fields) < 3 or not fields[0]:
            continue
        genes = [gene for gene in fields[2:] if gene]
        result[fields[0]] = genes
    return result


def validate_config(config: dict[str, object]) -> None:
    required = {
        "protocol",
        "source",
        "design",
        "population",
        "pathway_family",
        "state_matching",
        "negative_controls",
        "orthogonal_endpoint",
        "gates",
        "claim_boundary",
        "outcome_blind_sample_qc",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing protocol sections: {missing}")
    protocol = config["protocol"]
    if not isinstance(protocol, dict) or protocol.get("prospective_preregistration_claimed") is not False:
        raise ValueError("This known-source freeze must not claim prospective preregistration")
    if protocol.get("no_retuning_after_freeze") is not True:
        raise ValueError("The flagship protocol must freeze no-retuning behavior")
    claim = config["claim_boundary"]
    if not isinstance(claim, dict) or claim.get("pass_descriptor") != "E2-V1":
        raise ValueError(
            "The legacy pre-result success criterion must be recorded as E2-V1"
        )
    if (
        claim.get("pass_descriptor_role")
        != "legacy_pre_result_success_criterion_not_observed_result"
    ):
        raise ValueError(
            "The E2-V1 criterion must be marked as conditional and unobserved"
        )
    forbidden = {str(item) for item in claim.get("forbidden_claims", [])}
    required_forbidden = {"prospective external validation", "newly discovered vaccine mechanism", "matched rescue"}
    if not required_forbidden.issubset(forbidden):
        raise ValueError("Known-source claim safeguards are incomplete")


def write_locked_gmt(config: dict[str, object], path: Path) -> list[dict[str, object]]:
    family = config["pathway_family"]
    libraries = family["source_libraries"]
    fetched = {
        key: fetch_library(str(value["url"]))
        for key, value in libraries.items()
    }
    rows: list[dict[str, object]] = []
    lines: list[str] = []
    for output_name, spec in family["members"].items():
        library_key = str(spec["library"])
        source_name = str(spec["source_name"])
        genes = fetched[library_key].get(source_name)
        if not genes:
            raise ValueError(f"Gene set not found: {library_key}/{source_name}")
        source = libraries[library_key]
        lines.append("\t".join([str(output_name), str(source["url"]), *genes]))
        rows.append(
            {
                "pathway": output_name,
                "library": source["name"],
                "source_name": source_name,
                "source_url": source["url"],
                "n_genes": len(genes),
            }
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
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
        raise SystemExit("Existing freeze does not match the current config")
    manifest = pd.read_csv(manifest_path, sep="\t")
    expected = dict(zip(manifest["path"], manifest["sha256"]))
    if expected.get("locked_pathway_family.gmt") != sha256(gmt_path):
        raise SystemExit("Locked pathway family hash mismatch")
    print(f"Verified existing create-only freeze: {outdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or verify the BNT162b2 TED flagship protocol freeze")
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

    outdir.mkdir(parents=True, exist_ok=False)
    gmt_path = outdir / "locked_pathway_family.gmt"
    pathway_rows = write_locked_gmt(config, gmt_path)
    pd.DataFrame(pathway_rows).to_csv(outdir / "locked_pathway_registry.tsv", sep="\t", index=False)

    created_at = datetime.now(timezone.utc).isoformat()
    status_lines = git_output("status", "--porcelain").splitlines()
    freeze = {
        "protocol_id": config["protocol"]["id"],
        "protocol_version": config["protocol"]["version"],
        "created_at_utc": created_at,
        "retrospective_known_source": True,
        "prospective_preregistration_claimed": False,
        "known_direction_from_publication": True,
        "local_expression_data_accessed_before_freeze": False,
        "outcome_values_accessed_before_rna_freeze": False,
        "config_sha256": sha256(config_path),
        "locked_pathway_family_sha256": sha256(gmt_path),
        "git_commit": git_output("rev-parse", "HEAD"),
        "worktree_dirty": bool(status_lines),
        "dirty_entry_count": len(status_lines),
    }
    (outdir / "protocol_freeze.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    manifest = pd.DataFrame(
        [
            {
                "path": config_path.relative_to(ROOT).as_posix(),
                "size_bytes": config_path.stat().st_size,
                "sha256": sha256(config_path),
                "role": "analysis_contract",
            },
            {
                "path": "locked_pathway_family.gmt",
                "size_bytes": gmt_path.stat().st_size,
                "sha256": sha256(gmt_path),
                "role": "frozen_pathway_family",
            },
        ]
    )
    manifest.to_csv(outdir / "protocol_manifest.tsv", sep="\t", index=False)
    print(f"Created BNT162b2 flagship freeze before local expression download: {outdir}")


if __name__ == "__main__":
    main()
