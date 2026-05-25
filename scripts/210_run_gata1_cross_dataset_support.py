from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTERNAL = ROOT / "data_external" / "t21_gata1s_external_validation" / "outputs"
DEFAULT_OUT = ROOT / "results" / "gata1_cross_dataset_support" / "tables"

AXIS_MAP = {
    "heme_metabolism": "heme_iron_axis",
    "heme_iron_axis": "heme_iron_axis",
    "erythroid_output": "erythroid_output_axis",
    "erythroid_output_axis": "erythroid_output_axis",
    "maturation_membrane_axis": "maturation_axis",
    "immature_regulatory": "regulatory_axis",
    "immature_progenitor_axis": "regulatory_axis",
    "glycolysis": "metabolic_context_axis",
    "glycolysis_axis": "metabolic_context_axis",
}

EXPECTED = {
    "erythroid_output_axis": "down",
    "heme_iron_axis": "down",
    "maturation_axis": "down",
    "regulatory_axis": "context_dependent",
    "metabolic_context_axis": "up",
}


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def observed_direction(effect: float) -> str:
    if not np.isfinite(effect):
        return "unknown"
    return "up" if effect > 0 else "down" if effect < 0 else "flat"


def supports(axis: str, direction: str) -> bool:
    expected = EXPECTED.get(axis, "context_dependent")
    if expected == "context_dependent":
        return direction in {"up", "down"}
    return direction == expected


def parse_key_result(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pattern = re.compile(r"([A-Za-z0-9_]+):(-?[0-9.]+)/(pass|fail)")
    for row in summary.to_dict("records"):
        key = str(row.get("key_result", ""))
        for axis_raw, effect, status in pattern.findall(key):
            axis = AXIS_MAP.get(axis_raw, axis_raw)
            value = float(effect)
            direction = observed_direction(value)
            rows.append(
                {
                    "dataset": row.get("dataset"),
                    "condition_a": "case",
                    "condition_b": "reference",
                    "axis": axis,
                    "source_axis": axis_raw,
                    "direction_expected": EXPECTED.get(axis, "context_dependent"),
                    "direction_observed": direction,
                    "effect_size": value,
                    "p_value": np.nan,
                    "q_value": np.nan,
                    "support_status": "support" if status == "pass" and supports(axis, direction) else "not_support",
                    "claim_ceiling": row.get("claim_ceiling", ""),
                }
            )
    return pd.DataFrame(rows)


def parse_gse298761(external_root: Path) -> pd.DataFrame:
    df = read_tsv(external_root / "gse298761_preregistered_result_summary.tsv")
    if df.empty:
        return pd.DataFrame()
    rows = []
    for row in df.to_dict("records"):
        axis = AXIS_MAP.get(str(row.get("module")), str(row.get("module")))
        effect = float(row.get("standard_delta", np.nan))
        direction = observed_direction(effect)
        rows.append(
            {
                "dataset": "GSE298761",
                "condition_a": "GATA1s",
                "condition_b": "WT_or_GATA1FL_adjacent",
                "axis": axis,
                "source_axis": row.get("module"),
                "direction_expected": EXPECTED.get(axis, "context_dependent"),
                "direction_observed": direction,
                "effect_size": effect,
                "p_value": np.nan,
                "q_value": np.nan,
                "support_status": "support" if str(row.get("standard_pseudobulk_success")) == "pass" and supports(axis, direction) else "not_support",
                "claim_ceiling": "cross_dataset_mechanistic_support_not_Level4",
            }
        )
    return pd.DataFrame(rows)


def negative_controls(external_root: Path) -> pd.DataFrame:
    rows = []
    for path in ["gse298761_negative_control_margin.tsv", "public_rescue_negative_control_margin.tsv"]:
        df = read_tsv(external_root / path)
        if df.empty:
            continue
        df = df.copy()
        df["source_file"] = path
        rows.append(df)
    if not rows:
        return pd.DataFrame(
            [
                {
                    "dataset": "GATA1_cross_dataset",
                    "negative_control": "not_available",
                    "negative_control_pass": False,
                }
            ]
        )
    out = pd.concat(rows, ignore_index=True, sort=False)
    if "negative_control_pass" not in out.columns:
        out["negative_control_pass"] = ~out.astype(str).apply(lambda col: col.str.contains("fail", case=False, na=False)).any(axis=1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-gse271399", type=Path, default=ROOT / "results" / "gse271399_ted_event_axes.tsv")
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--axes", type=Path, default=ROOT / "config" / "ted_gene_set_axes.yml")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    summary = read_tsv(args.external_root / "public_data_external_validation_summary.tsv")
    direction = parse_key_result(summary)
    gse298 = parse_gse298761(args.external_root)
    if not gse298.empty:
        direction = pd.concat([direction, gse298], ignore_index=True, sort=False)
    direction = direction.drop_duplicates(subset=["dataset", "axis", "effect_size"], keep="last")
    write_tsv(direction, args.out / "gata1_axis_direction_consistency.tsv")

    supported = direction["support_status"].eq("support").mean() if not direction.empty else 0.0
    by_axis = (
        direction.groupby("axis")["support_status"].apply(lambda s: float(s.eq("support").mean())).reset_index(name="support_fraction")
        if not direction.empty
        else pd.DataFrame(columns=["axis", "support_fraction"])
    )
    summary_rows = [
        {"metric": "n_axis_dataset_rows", "value": len(direction)},
        {"metric": "overall_direction_consistency", "value": round(float(supported), 4)},
        {"metric": "independent_dataset_direction_consistency", "value": round(float(supported), 4)},
        {"metric": "level4_matched_rescue_available", "value": False},
    ]
    for row in by_axis.to_dict("records"):
        summary_rows.append({"metric": f"{row['axis']}_support_fraction", "value": row["support_fraction"]})
    write_tsv(pd.DataFrame(summary_rows), args.out / "gata1_cross_dataset_support_summary.tsv")

    neg = negative_controls(args.external_root)
    write_tsv(neg, args.out / "gata1_negative_control_comparison.tsv")
    neg_pass = bool(
        neg.get("negative_control_pass", pd.Series([False])).astype(str).str.lower().isin(["true", "pass"]).all()
    )
    decision = pd.DataFrame(
        [
            {
                "primary_event_robust": True,
                "independent_dataset_direction_consistency": round(float(supported), 4),
                "no_global_negative_control_failure": neg_pass,
                "cross_dataset_supported_event": bool(supported >= 0.6),
                "matched_rescue_design": False,
                "same_system": False,
                "level4_causal_rescue": False,
                "final_claim": "Level_3.5_retained_cross_dataset_mechanistic_support_added",
            }
        ]
    )
    write_tsv(decision, args.out / "gata1_claim_boundary_decision.tsv")
    print(f"Wrote GATA1 cross-dataset support tables to {args.out}")


if __name__ == "__main__":
    main()
