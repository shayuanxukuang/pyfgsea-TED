from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping
import warnings

import numpy as np

from .t21_calibration_profile import (
    PROFILE_SCHEMA_NAME,
    PROFILE_SCHEMA_VERSION,
    _code_bindings,
    load_calibration_design_profile,
    profile_payload_sha256,
    validate_calibration_design_profile,
    write_calibration_design_profile,
)
from .t21_data_product import sha256_file, stable_json
from .t21_preunblinding_calibration import (
    _positive_log_correlation,
    derive_profile_simulation_parameters,
    load_runner_spec,
)


CORRECTION_SCHEMA_NAME = "t21_calibration_profile_code_binding_correction"
CORRECTION_SCHEMA_VERSION = "1.0.0"
EXPECTED_CHANGED_BINDINGS = {
    "calibration_runner_module_sha256": "pyfgsea/t21_preunblinding_calibration.py",
    "covariate_pseudobulk_core_sha256": (
        "pyfgsea/trajectory_covariate_pseudobulk.py"
    ),
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _relative_file(path: str | Path, root: Path, *, label: str) -> str:
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must be inside the repository") from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return relative.as_posix()


def _resolve_file(relative_path: str, root: Path, *, label: str) -> Path:
    value = Path(str(relative_path))
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"{label} path escapes the repository")
    resolved = (root / value).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes the repository") from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _require_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise ValueError(f"Checksum sidecar is missing for {path.name}")
    declared = sidecar.read_text(encoding="ascii").strip().lower()
    observed = sha256_file(path)
    if not _SHA256_PATTERN.fullmatch(declared) or declared != observed:
        raise ValueError(f"Checksum sidecar differs for {path.name}")
    return observed


def _substantive_payload(profile: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in profile.items()
        if key not in {"code_bindings", "integrity"}
    }


def substantive_payload_sha256(profile: Mapping[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(
        stable_json(_substantive_payload(profile)).encode("utf-8")
    ).hexdigest()


def verify_historical_code_bindings(
    source_bindings: Mapping[str, str],
    current_bindings: Mapping[str, str],
    historical_snapshots: Mapping[str, Path],
) -> list[dict[str, str]]:
    if set(source_bindings) != set(current_bindings):
        raise ValueError("Source and current profile code-binding roles differ")
    changed = {
        role
        for role in source_bindings
        if str(source_bindings[role]) != str(current_bindings[role])
    }
    if changed != set(EXPECTED_CHANGED_BINDINGS):
        raise ValueError(
            "Profile migration requires exactly the reviewed runner and pseudobulk changes"
        )
    if set(historical_snapshots) != changed:
        raise ValueError("Historical source snapshots do not cover changed bindings exactly")
    rows = []
    for role in sorted(changed):
        snapshot = Path(historical_snapshots[role]).resolve()
        if not snapshot.is_file():
            raise FileNotFoundError(snapshot)
        old_sha = sha256_file(snapshot)
        if old_sha != str(source_bindings[role]):
            raise ValueError(f"Historical snapshot differs for {role}")
        rows.append(
            {
                "role": role,
                "repository_path": EXPECTED_CHANGED_BINDINGS[role],
                "source_sha256": old_sha,
                "target_sha256": str(current_bindings[role]),
            }
        )
    return rows


def mean_variance_correction_diagnostics(
    profile: Mapping[str, Any], runner_spec: Mapping[str, Any]
) -> dict[str, Any]:
    bins = profile["pooled_anonymous_log_expression_dispersion"]["bins"]
    mean = np.asarray(
        [max(float(row["mean_log_expression_median"]), 1e-8) for row in bins],
        dtype=float,
    )
    variance = np.asarray(
        [max(float(row["log_expression_variance_median"]), 1e-8) for row in bins],
        dtype=float,
    )
    if len(mean) < 2 or np.any(~np.isfinite(mean)) or np.any(~np.isfinite(variance)):
        raise ValueError("Profile mean-variance bins are invalid")
    order = np.argsort(mean, kind="stable")
    mean = mean[order]
    variance = variance[order]
    n_pathways = int(profile["pathway_structure"]["n_pathways"])
    source_positions = np.linspace(0.0, 1.0, len(mean))
    target_positions = np.linspace(0.0, 1.0, n_pathways)
    baseline = np.interp(target_positions, source_positions, mean)
    noise = np.sqrt(np.interp(target_positions, source_positions, variance))
    permutation = np.random.default_rng(20260714).permutation(n_pathways)
    baseline = baseline[permutation]
    noise = noise[permutation]
    scale = runner_spec["design"]["scale_contract"]
    minimum = float(scale["minimum_noise_multiplier"])
    maximum = float(scale["maximum_noise_multiplier"])
    noise /= np.median(noise)
    noise = np.clip(noise, minimum, maximum)
    noise /= np.median(noise)

    prior_source = float(np.corrcoef(np.log1p(mean), np.log1p(variance))[0, 1])
    prior_simulated = float(
        np.corrcoef(np.log1p(baseline), np.log1p(noise**2))[0, 1]
    )
    corrected_source = _positive_log_correlation(
        mean, variance, label="Corrected source mean-variance relation"
    )
    corrected_simulated = _positive_log_correlation(
        baseline, noise**2, label="Corrected simulated mean-variance relation"
    )
    tolerance = float(scale["mean_variance_log_correlation_tolerance"])
    result = {
        "metric_before": "corr(log1p(mean),log1p(variance))",
        "metric_after": "corr(log(mean),log(variance))",
        "prior_source_correlation": prior_source,
        "prior_simulated_correlation": prior_simulated,
        "prior_absolute_difference": abs(prior_source - prior_simulated),
        "corrected_source_correlation": corrected_source,
        "corrected_simulated_correlation": corrected_simulated,
        "corrected_absolute_difference": abs(
            corrected_source - corrected_simulated
        ),
        "frozen_tolerance": tolerance,
        "minimum_noise_multiplier": minimum,
        "maximum_noise_multiplier": maximum,
        "observed_noise_multiplier_minimum": float(noise.min()),
        "observed_noise_multiplier_maximum": float(noise.max()),
        "n_expression_bins": int(len(mean)),
        "n_pathways": n_pathways,
    }
    if result["prior_absolute_difference"] <= tolerance:
        raise ValueError("Historical mean-variance failure was not reproduced")
    if result["corrected_absolute_difference"] > tolerance:
        raise ValueError("Corrected mean-variance metric still violates tolerance")
    if float(noise.min()) < minimum - 1e-12 or float(noise.max()) > maximum + 1e-12:
        raise ValueError("Corrected diagnostics changed frozen noise bounds")
    return result


def _atomic_write_json_with_sidecar(value: Mapping[str, Any], path: Path) -> None:
    if path.exists() or path.with_suffix(path.suffix + ".sha256").exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)
    digest = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_temp = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
    sidecar_temp.write_text(f"{digest}\n", encoding="ascii", newline="\n")
    os.replace(sidecar_temp, sidecar)


def migrate_profile_code_bindings(
    *,
    source_profile_path: str | Path,
    target_profile_path: str | Path,
    ledger_path: str | Path,
    failed_bindings_path: str | Path,
    failed_output_dir: str | Path,
    historical_runner_snapshot: str | Path,
    historical_pseudobulk_snapshot: str | Path,
    runner_spec_path: str | Path,
    migration_cli_path: str | Path,
    repository_root: str | Path,
) -> tuple[Path, Path]:
    root = Path(repository_root).resolve()
    source_path = Path(source_profile_path).resolve()
    target_path = Path(target_profile_path).resolve()
    ledger = Path(ledger_path).resolve()
    failed_bindings_file = Path(failed_bindings_path).resolve()
    runner_spec_file = Path(runner_spec_path).resolve()
    cli_file = Path(migration_cli_path).resolve()
    for path, label in (
        (source_path, "source profile"),
        (failed_bindings_file, "failed blind bindings"),
        (runner_spec_file, "runner spec"),
        (cli_file, "migration CLI"),
    ):
        _relative_file(path, root, label=label)
    if target_path.exists() or ledger.exists():
        raise FileExistsError("Migration target or ledger already exists")
    if Path(failed_output_dir).resolve().exists():
        raise ValueError("Failed smoke output directory unexpectedly contains artifacts")

    source_sha = _require_sidecar(source_path)
    source = _load_json(source_path, label="source profile")
    if (
        source.get("schema_name") != PROFILE_SCHEMA_NAME
        or source.get("schema_version") != PROFILE_SCHEMA_VERSION
        or source.get("outcome_blinded") is not True
        or source.get("real_pathway_results_read") is not False
        or source.get("integrity", {}).get("profile_payload_sha256")
        != profile_payload_sha256(source)
    ):
        raise ValueError("Source profile is not an intact outcome-blind profile")
    failed_bindings = _load_json(
        failed_bindings_file, label="failed smoke blind bindings"
    )
    if (
        failed_bindings.get("design_profile_sha256") != source_sha
        or failed_bindings.get("design_profile_payload_sha256")
        != source["integrity"]["profile_payload_sha256"]
        or failed_bindings.get("code_dirty") is not True
    ):
        raise ValueError("Failed smoke bindings do not bind the source profile")

    current_bindings = _code_bindings(root)
    snapshots = {
        "calibration_runner_module_sha256": Path(
            historical_runner_snapshot
        ).resolve(),
        "covariate_pseudobulk_core_sha256": Path(
            historical_pseudobulk_snapshot
        ).resolve(),
    }
    changed_rows = verify_historical_code_bindings(
        source["code_bindings"], current_bindings, snapshots
    )
    for row in changed_rows:
        row["historical_snapshot_relative_path"] = _relative_file(
            snapshots[row["role"]], root, label=f"{row['role']} snapshot"
        )
        row["historical_snapshot_bytes"] = snapshots[row["role"]].stat().st_size

    target = deepcopy(source)
    target["code_bindings"] = current_bindings
    target["integrity"] = {"profile_payload_sha256": profile_payload_sha256(target)}
    source_substantive_sha = substantive_payload_sha256(source)
    target_substantive_sha = substantive_payload_sha256(target)
    if (
        stable_json(_substantive_payload(source))
        != stable_json(_substantive_payload(target))
        or source_substantive_sha != target_substantive_sha
    ):
        raise RuntimeError("Profile migration changed substantive anonymous payload")
    validate_calibration_design_profile(target, repository_root=root)

    runner_spec = load_runner_spec(runner_spec_file)
    correction_diagnostics = mean_variance_correction_diagnostics(
        target, runner_spec
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        derived = derive_profile_simulation_parameters(runner_spec, target)
    selected_bins = [int(value) for value in derived["selected_bin_indices"]]
    if sum(bool(value) for value in derived["included_donor_mask"]) != 17:
        raise ValueError("Corrected profile derivation did not retain all 17 donors")

    written_target: Path | None = None
    try:
        written_target = write_calibration_design_profile(
            target, target_path, repository_root=root
        )
        target_sha = sha256_file(written_target)
        module_path = Path(__file__).resolve()
        correction_id = "t21-calibration-profile-correction-" + __import__(
            "hashlib"
        ).sha256(
            stable_json(
                {
                    "source_profile_sha256": source_sha,
                    "target_profile_sha256": target_sha,
                    "changed_code_bindings": changed_rows,
                    "correction_diagnostics": correction_diagnostics,
                }
            ).encode("utf-8")
        ).hexdigest()[:16]
        ledger_value = {
            "schema_name": CORRECTION_SCHEMA_NAME,
            "schema_version": CORRECTION_SCHEMA_VERSION,
            "correction_id": correction_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "pass_outcome_blind_code_binding_migration",
            "outcome_blinded": True,
            "real_pathway_results_read": False,
            "candidate_expression_matrices_read_after_profile": False,
            "candidate_pathway_artifacts_read": False,
            "migration_scope": ["code_bindings", "integrity"],
            "source_profile": {
                "relative_path": _relative_file(
                    source_path, root, label="source profile"
                ),
                "bytes": source_path.stat().st_size,
                "sha256": source_sha,
                "payload_sha256": source["integrity"]["profile_payload_sha256"],
            },
            "target_profile": {
                "relative_path": _relative_file(
                    written_target, root, label="target profile"
                ),
                "bytes": written_target.stat().st_size,
                "sha256": target_sha,
                "payload_sha256": target["integrity"]["profile_payload_sha256"],
            },
            "substantive_payload_proof": {
                "excluded_keys": ["code_bindings", "integrity"],
                "source_sha256": source_substantive_sha,
                "target_sha256": target_substantive_sha,
                "byte_canonical_equality": True,
            },
            "changed_code_bindings": changed_rows,
            "failed_smoke": {
                "phase": "smoke",
                "replicate_simulation_started": False,
                "output_artifacts_written": False,
                "failure_stage": "derive_profile_simulation_parameters",
                "exception_type": "ValueError",
                "exception_message": (
                    "Frozen pathway simulation does not preserve the profile "
                    "mean-variance relation"
                ),
                "bindings_relative_path": _relative_file(
                    failed_bindings_file, root, label="failed blind bindings"
                ),
                "bindings_sha256": sha256_file(failed_bindings_file),
            },
            "scientific_correction": correction_diagnostics,
            "post_correction_derivation": {
                "selected_bin_indices": selected_bins,
                "n_included_donors": 17,
                "n_pathways": int(derived["n_pathways"]),
                "pooled_mean_variance_log_correlation": float(
                    derived["pooled_mean_variance_log_correlation"]
                ),
                "simulated_mean_variance_log_correlation": float(
                    derived["simulated_mean_variance_log_correlation"]
                ),
            },
            "frozen_inputs": {
                "runner_spec_relative_path": _relative_file(
                    runner_spec_file, root, label="runner spec"
                ),
                "runner_spec_sha256": sha256_file(runner_spec_file),
                "tolerance_changed": False,
                "noise_bounds_changed": False,
                "profile_substantive_payload_changed": False,
            },
            "migration_implementation": {
                "module_relative_path": _relative_file(
                    module_path, root, label="migration module"
                ),
                "module_sha256": sha256_file(module_path),
                "cli_relative_path": _relative_file(
                    cli_file, root, label="migration CLI"
                ),
                "cli_sha256": sha256_file(cli_file),
            },
        }
        _atomic_write_json_with_sidecar(ledger_value, ledger)
    except BaseException:
        if written_target is not None:
            written_target.unlink(missing_ok=True)
            written_target.with_suffix(written_target.suffix + ".sha256").unlink(
                missing_ok=True
            )
        raise
    return target_path, ledger


def validate_profile_code_binding_correction(
    ledger_path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    ledger_file = Path(ledger_path).resolve()
    _require_sidecar(ledger_file)
    ledger = _load_json(ledger_file, label="profile correction ledger")
    if (
        ledger.get("schema_name") != CORRECTION_SCHEMA_NAME
        or ledger.get("schema_version") != CORRECTION_SCHEMA_VERSION
        or ledger.get("status") != "pass_outcome_blind_code_binding_migration"
        or ledger.get("outcome_blinded") is not True
        or ledger.get("real_pathway_results_read") is not False
        or ledger.get("candidate_expression_matrices_read_after_profile") is not False
        or ledger.get("candidate_pathway_artifacts_read") is not False
        or ledger.get("migration_scope") != ["code_bindings", "integrity"]
    ):
        raise ValueError("Profile correction ledger violates its blind contract")
    source_entry = ledger["source_profile"]
    target_entry = ledger["target_profile"]
    source_path = _resolve_file(
        source_entry["relative_path"], root, label="source profile"
    )
    target_path = _resolve_file(
        target_entry["relative_path"], root, label="target profile"
    )
    for path, entry, label in (
        (source_path, source_entry, "source profile"),
        (target_path, target_entry, "target profile"),
    ):
        if path.stat().st_size != int(entry["bytes"]) or sha256_file(path) != entry[
            "sha256"
        ]:
            raise ValueError(f"{label} bytes or hash changed")
        _require_sidecar(path)
    source = _load_json(source_path, label="source profile")
    target = load_calibration_design_profile(target_path, repository_root=root)
    if (
        profile_payload_sha256(source) != source_entry["payload_sha256"]
        or target["integrity"]["profile_payload_sha256"]
        != target_entry["payload_sha256"]
    ):
        raise ValueError("Profile payload hashes differ from correction ledger")
    proof = ledger["substantive_payload_proof"]
    source_substantive = substantive_payload_sha256(source)
    target_substantive = substantive_payload_sha256(target)
    if (
        source_substantive != target_substantive
        or source_substantive != proof["source_sha256"]
        or target_substantive != proof["target_sha256"]
        or stable_json(_substantive_payload(source))
        != stable_json(_substantive_payload(target))
    ):
        raise ValueError("Correction changed substantive profile payload")
    current_bindings = _code_bindings(root)
    snapshots = {}
    for row in ledger["changed_code_bindings"]:
        role = str(row["role"])
        snapshot = _resolve_file(
            row["historical_snapshot_relative_path"],
            root,
            label=f"{role} snapshot",
        )
        if snapshot.stat().st_size != int(row["historical_snapshot_bytes"]):
            raise ValueError(f"Historical snapshot bytes changed for {role}")
        snapshots[role] = snapshot
    observed_rows = verify_historical_code_bindings(
        source["code_bindings"], current_bindings, snapshots
    )
    for observed, recorded in zip(observed_rows, ledger["changed_code_bindings"]):
        for key in ("role", "repository_path", "source_sha256", "target_sha256"):
            if observed[key] != recorded[key]:
                raise ValueError(f"Changed code binding ledger differs for {key}")
    implementation = ledger["migration_implementation"]
    for role in ("module", "cli"):
        path = _resolve_file(
            implementation[f"{role}_relative_path"],
            root,
            label=f"migration {role}",
        )
        if sha256_file(path) != implementation[f"{role}_sha256"]:
            raise ValueError(f"Migration {role} source changed")
    spec_entry = ledger["frozen_inputs"]
    spec_path = _resolve_file(
        spec_entry["runner_spec_relative_path"], root, label="runner spec"
    )
    if sha256_file(spec_path) != spec_entry["runner_spec_sha256"]:
        raise ValueError("Runner spec changed after correction")
    diagnostics = mean_variance_correction_diagnostics(
        target, load_runner_spec(spec_path)
    )
    if stable_json(diagnostics) != stable_json(ledger["scientific_correction"]):
        raise ValueError("Correction diagnostics are not reproducible")
    return {
        "status": "pass",
        "correction_id": ledger["correction_id"],
        "source_profile_sha256": source_entry["sha256"],
        "target_profile_sha256": target_entry["sha256"],
        "substantive_payload_sha256": target_substantive,
        "changed_code_binding_count": len(observed_rows),
    }
