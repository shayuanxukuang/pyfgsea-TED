"""Reproduce the frozen masked protein outcome and legacy migration fields.

The observed outcome is projected into the current parallel-record schema by
``build_bib_companion_evidence_contracts.py``; it never upgrades event E0.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.io import mmread


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyfgsea.ted_evidence import (  # noqa: E402
    ValidationProvenanceInputs,
    assign_validation_provenance,
)
from pyfgsea.ted_flagship import (  # noqa: E402
    exhaustive_sign_flip_max_t,
    leave_one_donor_retention,
    transient_contrasts,
)
from scripts.run_bnt162b2_flagship_rna import sha256  # noqa: E402
from scripts.run_gse171964_replication import matched_random_sets  # noqa: E402


CONFIG_PATH = ROOT / "config" / "ted_bnt162b2_flagship_v1.yaml"
RNA_DIR = ROOT / "results" / "ted_bnt162b2_flagship" / "rna_event_freeze_v1"
EXPORT_DIR = ROOT / "data_external" / "bnt162b2_cite_asap_2023" / "adt_unmasked_export_v1"
OUT_DIR = ROOT / "results" / "ted_bnt162b2_flagship" / "orthogonal_outcome_v1"


def feature_match(features: list[str], label: str) -> list[int]:
    normalized = [value.upper().replace("_", "-") for value in features]
    target = label.upper()
    return [
        idx
        for idx, value in enumerate(normalized)
        if value == target
        or value.startswith(target + "-")
        or value.endswith("-" + target)
        or (target == "CD64" and "FCGR1A" in value)
        or (target == "CD169" and "SIGLEC1" in value)
    ]


def group_score_table(meta: pd.DataFrame, cell_score: np.ndarray, value_name: str) -> pd.DataFrame:
    frame = meta[["donor_id", "day"]].copy()
    frame[value_name] = cell_score
    return frame.groupby(["donor_id", "day"], as_index=False)[value_name].mean()


def main() -> None:
    if OUT_DIR.exists():
        raise SystemExit(f"Orthogonal outcome output is create-only and already exists: {OUT_DIR}")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    rna_status_path = RNA_DIR / "rna_event_status.json"
    manifest_path = EXPORT_DIR / "adt_unmask_manifest.json"
    matrix_path = EXPORT_DIR / "selected_adt_values.mtx.gz"
    feature_path = EXPORT_DIR / "adt_features.txt"
    metadata_path = EXPORT_DIR / "adt_cell_metadata.tsv.gz"
    for path in (rna_status_path, manifest_path, matrix_path, feature_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    rna_status = json.loads(rna_status_path.read_text(encoding="utf-8"))
    unmask = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not rna_status.get("rna_event_freeze_complete") or rna_status.get("outcome_values_accessed") is not False:
        raise SystemExit("RNA freeze is incomplete or its mask audit failed")
    if unmask.get("unmasked_only_after_rna_event_freeze") is not True:
        raise SystemExit("ADT export provenance does not prove post-freeze unmasking")

    features = feature_path.read_text(encoding="utf-8").splitlines()
    meta = pd.read_csv(metadata_path, sep="\t", dtype={"donor_id": str})
    matrix = mmread(str(matrix_path))
    if matrix.shape == (len(meta), len(features)):
        matrix = matrix.T
    if matrix.shape != (len(features), len(meta)):
        raise ValueError("ADT matrix, features and selected-cell metadata differ")
    values = matrix.toarray().astype(np.float32, copy=False)
    if not unmask["values_are_seurat_normalized"]:
        denominator = np.exp(np.log1p(values).sum(axis=0) / len(features))
        values = np.log1p(values / denominator[None, :])
    feature_mean = values.mean(axis=1)
    feature_std = values.std(axis=1)
    feature_std[feature_std <= np.finfo(np.float32).eps] = 1.0
    feature_z = (values - feature_mean[:, None]) / feature_std[:, None]
    detection = np.mean(values > 0, axis=1)

    cd64 = feature_match(features, "CD64")
    cd169 = feature_match(features, "CD169")
    if len(cd64) != 1 or len(cd169) != 1:
        raise ValueError(f"Expected one CD64 and one CD169 ADT feature, found {cd64} and {cd169}")
    targets = cd64 + cd169
    outcome_index = feature_z[targets, :].mean(axis=0)
    outcome_scores = group_score_table(meta, outcome_index, "CD64_CD169_index")
    for label, feature_index in (("CD64", cd64[0]), ("CD169", cd169[0])):
        single = group_score_table(meta, feature_z[feature_index, :], label)
        outcome_scores = outcome_scores.merge(
            single, on=["donor_id", "day"], how="inner", validate="one_to_one"
        )
    contrasts = transient_contrasts(
        outcome_scores.rename(columns={"CD64_CD169_index": "score"})
    )
    transient = pd.DataFrame({"CD64_CD169_index": contrasts["transient"]})
    activation = pd.DataFrame({"CD64_CD169_index": contrasts["activation"]})
    recovery = pd.DataFrame({"CD64_CD169_index": contrasts["recovery"]})
    inference = exhaustive_sign_flip_max_t(transient)
    lodo = leave_one_donor_retention(transient, activation, recovery)

    stable_rows: list[dict[str, object]] = []
    stable_indices: set[int] = set()
    for label in config["orthogonal_endpoint"]["stable_ADT_controls"]:
        matches = feature_match(features, str(label))
        if len(matches) != 1:
            continue
        idx = matches[0]
        stable_indices.add(idx)
        scores = group_score_table(meta, feature_z[idx, :], "score")
        effect = transient_contrasts(scores)
        stable_rows.append(
            {
                "control_type": "stable_ADT",
                "control_id": label,
                "feature": features[idx],
                "mean_transient_effect": float(effect["transient"].mean()),
            }
        )
    random_sets = matched_random_sets(
        targets,
        feature_mean,
        detection,
        set(targets) | stable_indices,
        n_sets=int(config["orthogonal_endpoint"]["matched_random_ADT_controls"]),
        seed=int(config["negative_controls"]["random_seed"]),
    )
    random_rows: list[dict[str, object]] = []
    for number, indices in enumerate(random_sets, start=1):
        scores = group_score_table(meta, feature_z[indices, :].mean(axis=0), "score")
        effect = transient_contrasts(scores)
        random_rows.append(
            {
                "control_type": "matched_random_ADT",
                "control_id": f"random_adt_{number:03d}",
                "feature": "|".join(features[idx] for idx in indices),
                "mean_transient_effect": float(effect["transient"].mean()),
            }
        )
    controls = pd.DataFrame(stable_rows + random_rows)
    random_q95 = float(
        controls.loc[
            controls["control_type"].eq("matched_random_ADT"), "mean_transient_effect"
        ].abs().quantile(float(config["orthogonal_endpoint"]["control_quantile"]))
    )
    stable_max = float(
        controls.loc[
            controls["control_type"].eq("stable_ADT"), "mean_transient_effect"
        ].abs().max()
    )
    if not np.isfinite(stable_max):
        stable_max = 0.0
    outcome_effect = float(transient["CD64_CD169_index"].mean())
    control_margin = outcome_effect - max(random_q95, stable_max)

    rna_contrasts = pd.read_csv(RNA_DIR / "pathway_donor_contrasts.tsv", sep="\t")
    primary = config["pathway_family"]["primary"]
    rna_primary = rna_contrasts[rna_contrasts["pathway"].eq(primary)].set_index("donor_id")
    outcome_primary = transient.rename(columns={"CD64_CD169_index": "transient"})
    common = rna_primary.index.intersection(outcome_primary.index)
    direction_agreement = float(
        np.mean(
            np.sign(rna_primary.loc[common, "transient"].to_numpy(float))
            == np.sign(outcome_primary.loc[common, "transient"].to_numpy(float))
        )
    )
    direction_stability = float(np.mean(transient["CD64_CD169_index"] > 0))
    lodo_retention = float(lodo["retention_fraction"].iloc[0])
    exact_p = float(inference.set_index("pathway").loc["CD64_CD169_index", "exact_maxT_p"])
    gates = {
        "same_transient_contrast": True,
        "positive_alignment": outcome_effect > 0,
        "exact_p": exact_p <= float(config["orthogonal_endpoint"]["exact_p_max"]),
        "donor_direction_stability": direction_stability
        >= float(config["orthogonal_endpoint"]["donor_direction_stability_min"]),
        "rna_protein_donor_direction_agreement": direction_agreement
        >= float(config["orthogonal_endpoint"]["RNA_protein_donor_direction_agreement_min"]),
        "leave_one_donor_retention": lodo_retention
        >= float(config["orthogonal_endpoint"]["leave_one_donor_retention_min"]),
        "negative_control_margin": control_margin > 0,
    }
    gates = {name: bool(value) for name, value in gates.items()}
    outcome_pass = all(gates.values())
    provenance = assign_validation_provenance(
        ValidationProvenanceInputs(
            orthogonal_outcome_observed=True,
            outcome_assessment_prespecified=True,
            outcome_aligned=outcome_effect > 0 and direction_agreement
            >= float(config["orthogonal_endpoint"]["RNA_protein_donor_direction_agreement_min"]),
            outcome_controls_pass=outcome_pass,
        )
    )
    event_code = rna_status["event_support"]["code"]
    evidence_boundary = f"{event_code}-{provenance.code}"

    OUT_DIR.mkdir(parents=True, exist_ok=False)
    outcome_scores.to_csv(OUT_DIR / "protein_donor_time_scores.tsv", sep="\t", index=False)
    contrasts.reset_index().to_csv(OUT_DIR / "protein_donor_contrasts.tsv", sep="\t", index=False)
    inference.to_csv(OUT_DIR / "protein_exact_sign_flip.tsv", sep="\t", index=False)
    lodo.to_csv(OUT_DIR / "protein_leave_one_donor_refits.tsv", sep="\t", index=False)
    controls.to_csv(OUT_DIR / "protein_negative_controls.tsv", sep="\t", index=False)
    pd.DataFrame([{"gate": key, "passed": value} for key, value in gates.items()]).to_csv(
        OUT_DIR / "protein_outcome_gate_table.tsv", sep="\t", index=False
    )
    result = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "unmasked_only_after_rna_event_freeze": True,
        "event_support_code": event_code,
        "validation_provenance_code": provenance.code,
        "evidence_boundary": evidence_boundary,
        "protein_outcome_status": "passed" if outcome_pass else "failed",
        "protein_outcome_type": "same-study same-cells CD64/CD169 CITE-seq ADT",
        "CD64_feature": features[cd64[0]],
        "CD169_feature": features[cd169[0]],
        "protein_mean_transient_effect": outcome_effect,
        "protein_exact_p": exact_p,
        "protein_direction_stability": direction_stability,
        "rna_protein_donor_direction_agreement": direction_agreement,
        "protein_lodo_retention_fraction": lodo_retention,
        "protein_negative_control_margin": control_margin,
        "gates": gates,
        "validation_provenance": provenance.as_dict(),
    }
    (OUT_DIR / "protein_outcome_status.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    provenance_paths = [
        CONFIG_PATH,
        rna_status_path,
        RNA_DIR / "pathway_donor_contrasts.tsv",
        manifest_path,
        matrix_path,
        feature_path,
        metadata_path,
        Path(__file__).resolve(),
    ]
    pd.DataFrame(
        [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in provenance_paths
        ]
    ).to_csv(OUT_DIR / "protein_outcome_manifest.tsv", sep="\t", index=False)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
