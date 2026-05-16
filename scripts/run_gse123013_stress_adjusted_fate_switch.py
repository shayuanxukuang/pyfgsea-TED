#!/usr/bin/env python
"""Stress-adjusted fate-switch audit for GSE123013."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts" / "run_gse123013_plant_root_fate_switch_ted_lite.py"
OUTDIR = ROOT / "data_external" / "ted_generalization_panel" / "GSE123013"


def load_base():
    spec = importlib.util.spec_from_file_location("gse123013_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def write_tsv(df: pd.DataFrame, name: str) -> Path:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / name
    df.to_csv(path, sep="\t", index=False, na_rep="")
    return path


def sample_module_scores(gene_df: pd.DataFrame, sample_meta: pd.DataFrame, modules: dict[str, list[str]]) -> pd.DataFrame:
    sample_cols = [c for c in gene_df.columns if c.startswith("sample::")]
    rows = []
    upper = {g.upper(): g for g in gene_df["gene"]}
    for module, genes in modules.items():
        present = [upper[g.upper()] for g in genes if g.upper() in upper]
        sub = gene_df[gene_df["gene"].isin(present)]
        for col in sample_cols:
            sample = col.split("::", 1)[1]
            score = float(sub[col].mean()) if not sub.empty else np.nan
            rows.append(
                {
                    "sample": sample,
                    "module": module,
                    "score": score,
                    "n_genes_present": len(present),
                    "genes_present": ";".join(present),
                }
            )
    scores = pd.DataFrame(rows)
    meta = sample_meta[["sample", "condition", "n_cells", "mean_library_size"]].drop_duplicates()
    return scores.merge(meta, on="sample", how="left")


def ridge_residualize(y: np.ndarray, cov: np.ndarray, alpha: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(y) & np.all(np.isfinite(cov), axis=1)
    residual = np.full_like(y, np.nan, dtype=float)
    beta = np.full(cov.shape[1] + 1, np.nan, dtype=float)
    if mask.sum() < cov.shape[1] + 1:
        return residual, beta
    x = cov[mask].astype(float)
    x = (x - x.mean(axis=0)) / np.where(x.std(axis=0) > 1e-9, x.std(axis=0), 1.0)
    X = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(X.shape[1]) * alpha
    penalty[0, 0] = 0
    beta_fit = np.linalg.solve(X.T @ X + penalty, X.T @ y[mask])
    fitted = X @ beta_fit
    residual[mask] = y[mask] - fitted + np.nanmean(y[mask])
    beta[:] = beta_fit
    return residual, beta


def delta_by_condition(df: pd.DataFrame, score_col: str) -> dict[str, float]:
    means = df.groupby("condition")[score_col].mean()
    wt = means.get("WT", np.nan)
    return {
        "WT_mean": wt,
        "gl2_minus_WT": means.get("gl2", np.nan) - wt,
        "rhd6_minus_WT": means.get("rhd6", np.nan) - wt,
    }


def build_stress_adjusted(gene_df: pd.DataFrame, sample_meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    modules = BASE.GENE_SETS
    scores = sample_module_scores(gene_df, sample_meta, modules)
    wide = scores.pivot_table(index=["sample", "condition", "n_cells", "mean_library_size"], columns="module", values="score").reset_index()
    cov_cols = ["stress_axis", "housekeeping_negative_control", "wound_response_axis"]
    cov = wide[cov_cols].to_numpy(dtype=float)
    fate_modules = ["root_hair_axis", "non_hair_axis", "root_epidermis_axis"]
    rows = []
    adjusted_wide = wide.copy()
    betas = []
    for fate in fate_modules:
        y = wide[fate].to_numpy(dtype=float)
        residual, beta = ridge_residualize(y, cov, alpha=0.25)
        adjusted_col = f"{fate}_stress_adjusted"
        adjusted_wide[adjusted_col] = residual
        raw = delta_by_condition(wide[["condition", fate]].rename(columns={fate: "score"}), "score")
        adj = delta_by_condition(adjusted_wide[["condition", adjusted_col]].rename(columns={adjusted_col: "score"}), "score")
        rows.append(
            {
                "fate_axis": fate,
                "raw_WT_mean": raw["WT_mean"],
                "raw_gl2_minus_WT": raw["gl2_minus_WT"],
                "raw_rhd6_minus_WT": raw["rhd6_minus_WT"],
                "stress_adjusted_WT_mean": adj["WT_mean"],
                "stress_adjusted_gl2_minus_WT": adj["gl2_minus_WT"],
                "stress_adjusted_rhd6_minus_WT": adj["rhd6_minus_WT"],
                "adjusted_signal_retained": max(abs(adj["gl2_minus_WT"]), abs(adj["rhd6_minus_WT"])) >= 0.15,
                "model": "ridge residualization: fate ~ stress + housekeeping + wound",
                "n_samples": len(wide),
            }
        )
        betas.append({"fate_axis": fate, "intercept": beta[0], "beta_stress": beta[1], "beta_housekeeping": beta[2], "beta_wound": beta[3]})
    return pd.DataFrame(rows), adjusted_wide, pd.DataFrame(betas)


def build_specificity(adjusted: pd.DataFrame, marker: pd.DataFrame) -> pd.DataFrame:
    control = marker[marker["axis"].isin(["wound_response_axis", "stress_axis", "cell_cycle_axis", "housekeeping_negative_control"])].copy()
    control_strength = float(control[["gl2_minus_WT", "rhd6_minus_WT"]].abs().max(axis=1).max())
    rows = []
    for _, row in adjusted.iterrows():
        raw_strength = max(abs(float(row["raw_gl2_minus_WT"])), abs(float(row["raw_rhd6_minus_WT"])))
        adj_strength = max(abs(float(row["stress_adjusted_gl2_minus_WT"])), abs(float(row["stress_adjusted_rhd6_minus_WT"])))
        rows.append(
            {
                "fate_axis": row["fate_axis"],
                "raw_fate_strength": raw_strength,
                "stress_adjusted_fate_strength": adj_strength,
                "control_strength_raw": control_strength,
                "raw_fate_vs_control_margin": raw_strength - control_strength,
                "adjusted_fate_vs_control_margin": adj_strength - control_strength,
                "specificity_interpretation": "stress-adjusted fate signal retained" if row["adjusted_signal_retained"] else "fate signal largely explained by stress/housekeeping/wound axes",
            }
        )
    return pd.DataFrame(rows)


def build_axisfree_concordance(axis_modules: pd.DataFrame, adjusted: pd.DataFrame) -> pd.DataFrame:
    root = adjusted[adjusted["fate_axis"].eq("root_hair_axis")].iloc[0]
    root_signs = (np.sign(root["stress_adjusted_gl2_minus_WT"]), np.sign(root["stress_adjusted_rhd6_minus_WT"]))
    rows = []
    for _, row in axis_modules.iterrows():
        gl2 = float(row["gl2_score"] - row["WT_score"])
        rhd6 = float(row["rhd6_score"] - row["WT_score"])
        signs = (np.sign(gl2), np.sign(rhd6))
        rows.append(
            {
                "axis_free_module": row["module"],
                "module_gl2_minus_WT": gl2,
                "module_rhd6_minus_WT": rhd6,
                "stress_adjusted_root_hair_gl2_minus_WT": root["stress_adjusted_gl2_minus_WT"],
                "stress_adjusted_root_hair_rhd6_minus_WT": root["stress_adjusted_rhd6_minus_WT"],
                "event_direction_concordance": "same_sign_as_adjusted_root_hair" if signs == root_signs else "not_same_sign_as_adjusted_root_hair",
                "interpretation": "axis-free concordance after stress adjustment is supportive only, not a causal fate claim",
            }
        )
    return pd.DataFrame(rows)


def build_claim(adjusted: pd.DataFrame, specificity: pd.DataFrame) -> pd.DataFrame:
    retained = bool(adjusted["adjusted_signal_retained"].any())
    positive_margin = bool((specificity["adjusted_fate_vs_control_margin"] > 0).any())
    if retained and positive_margin:
        ceiling = "cautious_Level_3_stress_adjusted_plant_root_fate_switch_candidate"
        allowed = "GSE123013 retains a stress-adjusted root fate-switch signal in TED-lite."
    elif retained:
        ceiling = "Level_2.5_to_cautious_Level_3_stress_adjusted_signal_retained_but_controls_strong"
        allowed = "GSE123013 shows stress-adjusted fate signal but remains control-sensitive."
    else:
        ceiling = "Level_2.5_plant_root_stress_sensitive_fate_switch_candidate"
        allowed = "GSE123013 remains a stress-sensitive plant root fate-switch candidate; TED correctly avoids overclaiming."
    return pd.DataFrame(
        [
            {
                "dataset": "GSE123013",
                "previous_claim_ceiling": "Level_2.5_plant_root_fate_switch_scaffold_with_stress_control_sensitivity",
                "updated_claim_ceiling": ceiling,
                "allowed_claim": allowed,
                "forbidden_claim": "do not claim strong plant fate mechanism, strict delay/loss classification, or functional rescue",
                "missing_evidence": "more biological replicates, robust sample-block model, protoplasting/wound controls, time/state trajectory",
            }
        ]
    )


def main() -> None:
    gene_df, _, sample_meta = BASE.parse_matrix()
    adjusted, adjusted_wide, betas = build_stress_adjusted(gene_df, sample_meta)
    marker = BASE.build_marker_audit(gene_df)
    axis_modules, _ = BASE.build_axis_free(gene_df)
    specificity = build_specificity(adjusted, marker)
    axis_concordance = build_axisfree_concordance(axis_modules, adjusted)
    claim = build_claim(adjusted, specificity)
    paths = [
        write_tsv(adjusted, "gse123013_stress_adjusted_fate_switch.tsv"),
        write_tsv(specificity, "gse123013_fate_vs_stress_specificity.tsv"),
        write_tsv(axis_concordance, "gse123013_axisfree_stress_adjusted_concordance.tsv"),
        write_tsv(adjusted_wide, "gse123013_sample_level_stress_adjusted_scores.tsv"),
        write_tsv(betas, "gse123013_stress_adjustment_coefficients.tsv"),
        write_tsv(claim, "gse123013_claim_ceiling_update.tsv"),
    ]
    manifest = pd.DataFrame({"output_file": [rel(p) for p in paths], "n_rows": [len(pd.read_csv(p, sep="\t")) for p in paths]})
    write_tsv(manifest, "gse123013_stress_adjusted_output_manifest.tsv")
    print(f"Wrote {len(paths) + 1} GSE123013 stress-adjusted files to {rel(OUTDIR)}")


if __name__ == "__main__":
    main()
