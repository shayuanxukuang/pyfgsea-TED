from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import yaml


SCHEMA_NAME = "trajpathmix_corebench_cb2_functional_null_amendment"
SCHEMA_VERSION = "1.0.0"
AMENDMENT_ID = "trajpathmix_corebench_cb2_functional_null_v1"
FROZEN_CONFIG_PAYLOAD_SHA256 = (
    "8e27b410a9771670f6ec4be5d9bbcbe6cce00bacdd799ddcaad4102899ed2023"
)

AMENDMENT_FILE = "corebench_cb2_functional_null_amendment_v1.json"
BUILD_RECORD_FILE = "corebench_cb2_functional_null_build_record_v1.json"
IMPLEMENTATION_FILE = "pyfgsea/trajpathmix_corebench_cb2_contract.py"
CONFIG_FILE = "config/trajpathmix_corebench_cb2_functional_null_v1.yaml"


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


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise ValueError(
            f"Frozen CB2 amendment mismatch for {label}: expected "
            f"{expected!r}, observed {observed!r}"
        )


def _repo_file(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("CB2 amendment bindings must be repository-local") from exc
    return path


def validate_cb2_amendment_config(config: Mapping[str, Any]) -> dict[str, Any]:
    _require(config.get("schema_name"), SCHEMA_NAME, "schema_name")
    _require(config.get("schema_version"), SCHEMA_VERSION, "schema_version")
    _require(config.get("amendment_id"), AMENDMENT_ID, "amendment_id")
    _require(
        config.get("frozen_payload_sha256"),
        FROZEN_CONFIG_PAYLOAD_SHA256,
        "frozen_payload_sha256",
    )
    _require(_payload_hash(config), FROZEN_CONFIG_PAYLOAD_SHA256, "payload_sha256")

    semantics = config["amendment_semantics"]
    for key, expected in {
        "append_only": True,
        "parent_corebench_contract_modified": False,
        "cb1_artifacts_overwritten": False,
        "precedence": "cb2_acceptance_semantics_only",
    }.items():
        _require(semantics.get(key), expected, f"amendment_semantics.{key}")

    identity = config["identity"]
    for key, expected in {
        "stage": "CB2",
        "role": "functional_core_empirical_null",
        "is_endoderm_a1_amendment": False,
        "biological_discovery": False,
        "real_condition_contrast": False,
        "pathway_scores_are_statistical_benchmark_outcomes": True,
    }.items():
        _require(identity.get(key), expected, f"identity.{key}")

    _require(
        config["claims_under_test"],
        [
            "simultaneous_pathway_curve_inference",
            "supported_region_integrated_effect",
            "pathway_family_error_control",
            "simultaneous_band_coverage",
            "fail_closed_estimability",
        ],
        "claims_under_test",
    )
    _require(
        config["claims_not_under_test"],
        [
            "onset",
            "duration",
            "phase_shift",
            "peak_location",
            "heterochrony",
            "biological_endoderm_pathway_dynamics",
        ],
        "claims_not_under_test",
    )
    timing = config["outcome_firewall"]["timing"]
    for key in ("compute", "output", "acceptance_decision"):
        _require(timing.get(key), False, f"outcome_firewall.timing.{key}")

    units = config["population_and_units"]
    for key, expected in {
        "expected_donors": 75,
        "frozen_experiment_universe": 28,
        "assignment_unit": "donor_id",
        "independence_unit": "donor_id",
        "nested_line_handling": "nested_within_donor_not_independent",
        "pseudo_case_donors": 37,
        "pseudo_control_donors": 38,
    }.items():
        _require(units.get(key), expected, f"population_and_units.{key}")

    nuisance = config["nuisance_model"]
    _require(
        nuisance.get("primary_encoding"),
        "donor_bin_experiment_cell_fraction_fixed_effects",
        "nuisance_model.primary_encoding",
    )
    _require(
        nuisance.get("bin_specific_reduced_design_required"),
        True,
        "nuisance_model.bin_specific_reduced_design_required",
    )
    availability = config["availability_handling"]
    _require(
        availability.get("residual_mapping_block"),
        ["frozen_restriction_block", "full_20_bin_availability_signature"],
        "availability_handling.residual_mapping_block",
    )
    _require(
        availability.get("experiment_is_a_nuisance_not_a_hard_mapping_block"),
        True,
        "availability_handling.experiment_is_a_nuisance_not_a_hard_mapping_block",
    )
    _require(
        config["stages"]["primary_balanced_null"]["replicates"],
        500,
        "stages.primary_balanced_null.replicates",
    )
    _require(
        config["stages"]["primary_balanced_null"]["formal_status"],
        "acceptance_bearing",
        "stages.primary_balanced_null.formal_status",
    )
    _require(
        config["cb2a_design_precheck"]["pathway_scoring"],
        False,
        "cb2a_design_precheck.pathway_scoring",
    )
    _require(
        config["acceptance_endpoints"]["timing_metrics_affect_go_no_go"],
        False,
        "acceptance_endpoints.timing_metrics_affect_go_no_go",
    )
    state = config["execution_state"]
    _require(state.get("cb2a_authorized"), True, "execution_state.cb2a_authorized")
    _require(
        state.get("pathway_scoring_currently_authorized"),
        False,
        "execution_state.pathway_scoring_currently_authorized",
    )
    _require(
        state.get("cb2_500_currently_authorized"),
        False,
        "execution_state.cb2_500_currently_authorized",
    )
    result = deepcopy(dict(config))
    result["_config_payload_sha256"] = FROZEN_CONFIG_PAYLOAD_SHA256
    return result


def load_cb2_amendment_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("CB2 amendment config must be a YAML mapping")
    return validate_cb2_amendment_config(value)


def verify_cb2_amendment_bindings(
    repository_root: str | Path, config: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    root = Path(repository_root).resolve()
    verified: dict[str, dict[str, Any]] = {}
    for name, binding in config["bindings"].items():
        path = _repo_file(root, str(binding["relative_path"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = _hash_file(path)
        _require(observed, binding["sha256"], f"bindings.{name}.sha256")
        verified[name] = {
            "relative_path": str(binding["relative_path"]),
            "sha256": observed,
            "bytes": int(path.stat().st_size),
        }
    return verified


def _amendment_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(config))
    value.pop("_config_payload_sha256", None)
    value["materialization"] = {
        "evidence_revision_mode": "create_only_append_only",
        "parent_contract_unchanged": True,
        "cb1_artifacts_unchanged": True,
        "expression_values_read": False,
        "pathway_outcomes_read": False,
        "pathway_scoring_performed": False,
        "pseudo_conditions_generated": False,
        "timing_computed": False,
    }
    return value


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _build_record(
    *, config_path: Path, repository_root: Path, artifact_dir: Path
) -> dict[str, Any]:
    config = load_cb2_amendment_config(config_path)
    verified = verify_cb2_amendment_bindings(repository_root, config)
    return {
        "schema_name": "trajpathmix_corebench_cb2_functional_null_build_record",
        "schema_version": "1.0.0",
        "amendment_id": AMENDMENT_ID,
        "config_file": CONFIG_FILE,
        "config_file_sha256": _hash_file(config_path),
        "config_payload_sha256": FROZEN_CONFIG_PAYLOAD_SHA256,
        "implementation_file": IMPLEMENTATION_FILE,
        "implementation_sha256": _hash_file(Path(__file__).resolve()),
        "verified_bindings": verified,
        "artifacts": {
            AMENDMENT_FILE: {
                "sha256": _hash_file(artifact_dir / AMENDMENT_FILE),
                "bytes": int((artifact_dir / AMENDMENT_FILE).stat().st_size),
            }
        },
        "expression_values_read": False,
        "pathway_outcomes_read": False,
        "pathway_scoring_performed": False,
        "pseudo_conditions_generated": False,
        "timing_computed": False,
        "evidence_revision_mode": "create_only_append_only",
    }


def build_and_write_cb2_amendment(
    *, config_path: str | Path, repository_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    root = Path(repository_root).resolve()
    config = load_cb2_amendment_config(config_file)
    verify_cb2_amendment_bindings(root, config)
    output = Path(output_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"CB2 amendment output exists: {output}")
    lock = output.parent / f".{output.name}.create.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FileExistsError(f"CB2 amendment is locked: {lock}") from exc
    temporary: Path | None = None
    try:
        os.close(descriptor)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
        )
        _write_json(_amendment_payload(config), temporary / AMENDMENT_FILE)
        record = _build_record(
            config_path=config_file,
            repository_root=root,
            artifact_dir=temporary,
        )
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


def validate_cb2_amendment_output(
    *, config_path: str | Path, repository_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    root = Path(repository_root).resolve()
    config = load_cb2_amendment_config(config_file)
    output = Path(output_dir).resolve()
    if not output.is_dir():
        raise FileNotFoundError(output)
    _require(
        {path.name for path in output.iterdir() if path.is_file()},
        {AMENDMENT_FILE, BUILD_RECORD_FILE},
        "output_file_set",
    )
    observed_amendment = json.loads(
        (output / AMENDMENT_FILE).read_text(encoding="utf-8")
    )
    _require(observed_amendment, _amendment_payload(config), AMENDMENT_FILE)
    observed_record = json.loads(
        (output / BUILD_RECORD_FILE).read_text(encoding="utf-8")
    )
    expected_record = _build_record(
        config_path=config_file,
        repository_root=root,
        artifact_dir=output,
    )
    _require(observed_record, expected_record, BUILD_RECORD_FILE)
    result = dict(observed_record)
    result["output_dir"] = str(output)
    result["build_record_sha256"] = _hash_file(output / BUILD_RECORD_FILE)
    result["validation_status"] = "pass_append_only_cb2_functional_null_amendment"
    return result


__all__ = [
    "AMENDMENT_FILE",
    "AMENDMENT_ID",
    "BUILD_RECORD_FILE",
    "FROZEN_CONFIG_PAYLOAD_SHA256",
    "build_and_write_cb2_amendment",
    "load_cb2_amendment_config",
    "validate_cb2_amendment_config",
    "validate_cb2_amendment_output",
    "verify_cb2_amendment_bindings",
]
