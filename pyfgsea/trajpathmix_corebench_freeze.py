from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import pandas as pd
import yaml


SCHEMA_NAME = "trajpathmix_corebench_freeze_contract"
SCHEMA_VERSION = "1.0.0"
FREEZE_ID = "trajpathmix_corebench_v1"
FROZEN_CONFIG_PAYLOAD_SHA256 = (
    "46ddd94e4624a1654f99f6bdb60a0adadfabfd3e52a01e113d8fe8f2b9948ce9"
)

GENE_FOLD_FILE = "corebench_coordinate_gene_folds_v1.tsv"
EXCLUSION_AUDIT_FILE = "corebench_coordinate_gene_exclusion_audit_v1.json"
CONTRACT_FILE = "corebench_contract_v1.json"
BUILD_RECORD_FILE = "corebench_freeze_build_record_v1.json"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _payload_hash(config: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in config.items()
        if key != "frozen_payload_sha256" and not str(key).startswith("_")
    }
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def _hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise ValueError(
            f"Frozen CoreBench mismatch for {label}: "
            f"expected {expected!r}, observed {value!r}"
        )


def validate_corebench_config(config: Mapping[str, Any]) -> dict[str, Any]:
    _require(config.get("schema_name"), SCHEMA_NAME, "schema_name")
    _require(config.get("schema_version"), SCHEMA_VERSION, "schema_version")
    _require(config.get("freeze_id"), FREEZE_ID, "freeze_id")
    _require(
        config.get("frozen_payload_sha256"),
        FROZEN_CONFIG_PAYLOAD_SHA256,
        "frozen_payload_sha256",
    )
    _require(_payload_hash(config), FROZEN_CONFIG_PAYLOAD_SHA256, "payload_sha256")
    identity = config.get("identity", {})
    for key, expected in {
        "role": "statistical_kernel_benchmark",
        "biological_discovery": False,
        "real_pathway_outcomes": "not_applicable",
        "trajectory_claim": False,
        "is_endoderm_a1_amendment": False,
        "restores_endoderm_biological_timing_claim": False,
        "endoderm_a1_status_remains": "fail_closed",
    }.items():
        _require(identity.get(key), expected, f"identity.{key}")
    scope = config.get("freeze_scope", {})
    for key in (
        "expression_values_read_by_contract_freeze",
        "pathway_outcomes_read",
        "pathway_scoring_performed",
        "pseudo_conditions_generated",
        "injection_results_generated",
        "coordinate_values_materialized",
    ):
        _require(scope.get(key), False, f"freeze_scope.{key}")
    coordinate = config.get("analysis_coordinate", {})
    _require(
        coordinate.get("interpretation"),
        "fixed_analysis_coordinate_not_biological_pseudotime",
        "analysis_coordinate.interpretation",
    )
    _require(
        coordinate.get("coarse_anchor_order"),
        ["day0", "day1", "day2", "day3"],
        "analysis_coordinate.coarse_anchor_order",
    )
    _require(
        coordinate.get("deposited_trajectory_fields_allowed"),
        False,
        "analysis_coordinate.deposited_trajectory_fields_allowed",
    )
    _require(
        coordinate.get("materialization_gate", {}).get(
            "required_before_null_or_injection"
        ),
        True,
        "analysis_coordinate.materialization_gate",
    )
    _require(
        [stage.get("replicates") for stage in config["empirical_null"]["stages"]],
        [500, 2000, 10000],
        "empirical_null.stages",
    )
    _require(
        config.get("execution_state", {}).get("pathway_scoring_authorized"),
        False,
        "execution_state.pathway_scoring_authorized",
    )
    value = deepcopy(dict(config))
    value["_config_payload_sha256"] = FROZEN_CONFIG_PAYLOAD_SHA256
    return value


def load_corebench_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("CoreBench config must be a YAML mapping")
    return validate_corebench_config(value)


def _repo_file(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("CoreBench inputs must be repository-local") from exc
    return path


def _verify_bindings(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, binding in config["bindings"].items():
        path = _repo_file(root, str(binding["relative_path"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        _require(_hash_file(path), binding["sha256"], f"bindings.{name}.sha256")
        paths[name] = path
    return paths


def _fold(feature_id: str, *, seed: int, n_folds: int) -> int:
    digest = hashlib.sha256(f"{seed}:{feature_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n_folds + 1


def _build_artifacts(
    root: Path, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    paths = _verify_bindings(root, config)
    hvg = pd.read_csv(paths["hvg_rank"], sep="\t", dtype={"feature_id": "string", "gene_symbol": "string"})
    universe = pd.read_csv(
        paths["injection_pathway_universe"],
        sep="\t",
        dtype="string",
        usecols=["gene_id", "source_symbol", "pathway_universe_logical_sha256"],
    )
    _require(
        sorted(universe["pathway_universe_logical_sha256"].dropna().unique()),
        [config["bindings"]["injection_pathway_universe"]["logical_sha256"]],
        "injection_pathway_universe.logical_sha256",
    )
    selected = hvg.loc[hvg["selected_primary_500"].astype(str).str.lower().eq("true")].copy()
    coordinate = config["analysis_coordinate"]
    _require(len(selected), coordinate["candidate_hvg_count"], "candidate_hvg_count")
    injection_ids = set(universe["gene_id"].dropna().astype(str))
    injection_symbols = set(
        universe["source_symbol"].dropna().astype(str).str.upper()
    )
    feature_gene_id = selected["feature_id"].astype(str).str.split("_", n=1).str[0]
    symbol_upper = selected["gene_symbol"].fillna("").astype(str).str.upper()
    excluded = feature_gene_id.isin(injection_ids) | symbol_upper.isin(injection_symbols)
    exclusion = coordinate["injection_gene_exclusion"]
    _require(int(excluded.sum()), exclusion["expected_excluded_candidate_hvgs"], "excluded_hvgs")
    _require(int((~excluded).sum()), exclusion["expected_coordinate_genes"], "coordinate_genes")
    folds = coordinate["gene_fold_cross_fitting"]
    selected["ensembl_gene_id"] = feature_gene_id
    selected["coordinate_gene"] = ~excluded
    selected["exclusion_reason"] = ""
    selected.loc[excluded, "exclusion_reason"] = "bound_injection_pathway_universe"
    selected["gene_fold"] = pd.Series(pd.NA, index=selected.index, dtype="Int64")
    selected.loc[~excluded, "gene_fold"] = [
        _fold(
            str(feature_id),
            seed=int(folds["seed"]),
            n_folds=int(folds["n_folds"]),
        )
        for feature_id in selected.loc[~excluded, "feature_id"]
    ]
    manifest = selected[
        [
            "feature_id",
            "ensembl_gene_id",
            "gene_symbol",
            "hvg_rank",
            "coordinate_gene",
            "exclusion_reason",
            "gene_fold",
        ]
    ].sort_values("hvg_rank", kind="stable").reset_index(drop=True)
    fold_counts = {
        str(int(key)): int(value)
        for key, value in manifest.loc[manifest["coordinate_gene"], "gene_fold"]
        .value_counts()
        .sort_index()
        .items()
    }
    _require(len(fold_counts), int(folds["n_folds"]), "nonempty_gene_folds")
    audit = {
        "schema_name": "trajpathmix_corebench_coordinate_gene_exclusion_audit",
        "schema_version": "1.0.0",
        "freeze_id": FREEZE_ID,
        "config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
        "candidate_hvg_count": int(len(manifest)),
        "bound_injection_universe_unique_gene_ids": int(len(injection_ids)),
        "bound_injection_universe_unique_symbols": int(len(injection_symbols)),
        "excluded_candidate_hvgs": int(excluded.sum()),
        "coordinate_gene_count": int((~excluded).sum()),
        "gene_fold_counts": fold_counts,
        "coordinate_injection_gene_overlap": 0,
        "expression_values_read": False,
        "pathway_outcomes_read": False,
        "pathway_scoring_performed": False,
        "audit_status": "pass_coordinate_genes_disjoint_from_injection_universe",
    }
    contract = {
        "schema_name": "trajpathmix_corebench_contract",
        "schema_version": "1.0.0",
        "freeze_id": FREEZE_ID,
        "frozen_at_utc": config["frozen_at_utc"],
        "config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
        "identity": deepcopy(config["identity"]),
        "bindings": deepcopy(config["bindings"]),
        "analysis_coordinate": deepcopy(config["analysis_coordinate"]),
        "statistical_units": deepcopy(config["statistical_units"]),
        "empirical_null": deepcopy(config["empirical_null"]),
        "gene_level_injections": deepcopy(config["gene_level_injections"]),
        "baselines": deepcopy(config["baselines"]),
        "acceptance_policy": deepcopy(config["acceptance_policy"]),
        "terminal_rules": deepcopy(config["terminal_rules"]),
        "pathway_scoring_authorized": False,
        "next_gate": config["execution_state"]["next_gate"],
    }
    return manifest, audit, contract


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_artifacts(
    directory: Path,
    manifest: pd.DataFrame,
    audit: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    manifest.to_csv(directory / GENE_FOLD_FILE, sep="\t", index=False, lineterminator="\n")
    _write_json(audit, directory / EXCLUSION_AUDIT_FILE)
    _write_json(contract, directory / CONTRACT_FILE)


def _build_record(config_path: Path, artifact_dir: Path) -> dict[str, Any]:
    return {
        "schema_name": "trajpathmix_corebench_freeze_build_record",
        "schema_version": "1.0.0",
        "freeze_id": FREEZE_ID,
        "config_file": "config/trajpathmix_corebench_v1.yaml",
        "config_file_sha256": _hash_file(config_path),
        "config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
        "implementation_file": "pyfgsea/trajpathmix_corebench_freeze.py",
        "implementation_sha256": _hash_file(Path(__file__).resolve()),
        "artifacts": {
            name: {"sha256": _hash_file(artifact_dir / name), "bytes": (artifact_dir / name).stat().st_size}
            for name in (GENE_FOLD_FILE, EXCLUSION_AUDIT_FILE, CONTRACT_FILE)
        },
        "expression_values_read": False,
        "pathway_outcomes_read": False,
        "pathway_scoring_authorized": False,
        "evidence_revision_mode": "create_only_append_only",
    }


def build_and_write_corebench_freeze(
    *, config_path: str | Path, repository_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config = load_corebench_config(config_file)
    root = Path(repository_root).resolve()
    manifest, audit, contract = _build_artifacts(root, config)
    output = Path(output_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"CoreBench freeze output exists: {output}")
    lock = output.parent / f".{output.name}.create.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FileExistsError(f"CoreBench freeze is locked: {lock}") from exc
    temporary: Path | None = None
    try:
        os.close(descriptor)
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
        _write_artifacts(temporary, manifest, audit, contract)
        record = _build_record(config_file, temporary)
        _write_json(record, temporary / BUILD_RECORD_FILE)
        os.rename(temporary, output)
        temporary = None
        result = dict(record)
        result["output_dir"] = str(output)
        result["build_record_sha256"] = _hash_file(output / BUILD_RECORD_FILE)
        return result
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
        lock.unlink(missing_ok=True)


def validate_corebench_freeze_output(
    *, config_path: str | Path, repository_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config = load_corebench_config(config_file)
    root = Path(repository_root).resolve()
    manifest, audit, contract = _build_artifacts(root, config)
    output = Path(output_dir).resolve()
    if not output.is_dir():
        raise FileNotFoundError(output)
    expected_names = {GENE_FOLD_FILE, EXCLUSION_AUDIT_FILE, CONTRACT_FILE, BUILD_RECORD_FILE}
    _require({path.name for path in output.iterdir() if path.is_file()}, expected_names, "output_file_set")
    observed_manifest = pd.read_csv(
        output / GENE_FOLD_FILE,
        sep="\t",
        dtype="string",
        keep_default_na=False,
    )
    _require(
        observed_manifest.astype("string").fillna("").to_dict("records"),
        manifest.astype("string").fillna("").to_dict("records"),
        GENE_FOLD_FILE,
    )
    for name, expected in ((EXCLUSION_AUDIT_FILE, audit), (CONTRACT_FILE, contract)):
        observed = json.loads((output / name).read_text(encoding="utf-8"))
        _require(observed, expected, name)
    expected_record = _build_record(config_file, output)
    observed_record = json.loads((output / BUILD_RECORD_FILE).read_text(encoding="utf-8"))
    _require(observed_record, expected_record, BUILD_RECORD_FILE)
    result = dict(observed_record)
    result["output_dir"] = str(output)
    result["build_record_sha256"] = _hash_file(output / BUILD_RECORD_FILE)
    result["validation_status"] = "pass_corebench_contract_and_gene_separation_freeze"
    return result


__all__ = [
    "BUILD_RECORD_FILE",
    "CONTRACT_FILE",
    "EXCLUSION_AUDIT_FILE",
    "FREEZE_ID",
    "FROZEN_CONFIG_PAYLOAD_SHA256",
    "GENE_FOLD_FILE",
    "build_and_write_corebench_freeze",
    "load_corebench_config",
    "validate_corebench_config",
    "validate_corebench_freeze_output",
]
