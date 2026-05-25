from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from scp1064_utils import (  # noqa: E402
    RESULTS,
    axis_protein_pairs,
    protein_matrix,
    read_tsv,
    write_tsv,
)


N_PERMUTATIONS = 200
CELL_SUBSAMPLE = 50_000
RANDOM_SEED = 1064


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float("nan")
    res = stats.spearmanr(x[mask], y[mask], nan_policy="omit")
    return float(res.statistic)


def _permute_spearman(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_permutations: int,
) -> tuple[float, float, float]:
    observed = _safe_spearman(x, y)
    if not np.isfinite(observed):
        return observed, float("nan"), float("nan")
    null = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        null[i] = _safe_spearman(x, rng.permutation(y))
    empirical_p = (1.0 + float((np.abs(null) >= abs(observed)).sum())) / (n_permutations + 1.0)
    return observed, float(np.nanmax(np.abs(null))), empirical_p


def run_lightweight_shuffles() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    outcome_summary = read_tsv(RESULTS / "scp1064_outcome_alignment_summary.tsv")
    gate_pairs: set[tuple[str, str, str]] = set()
    if not outcome_summary.empty:
        summary = outcome_summary.copy()
        summary["alignment_pass"] = summary["alignment_pass"].astype(str).str.lower().isin({"true", "1", "yes"})
        summary["spearman_abs"] = pd.to_numeric(summary["spearman"], errors="coerce").abs()
        primary = summary[summary["alignment_pass"] & (summary["spearman_abs"] >= 0.20)]
        for row in primary.itertuples(index=False):
            gate_pairs.add((str(row.level), str(row.axis), str(row.protein_name)))

    cell_events = read_tsv(RESULTS / "scp1064_cell_level_event_scores.tsv")
    if cell_events.empty:
        raise FileNotFoundError("Run SCP1064 RNA-event scoring before lightweight shuffles.")
    wide_events = cell_events.pivot_table(index="cell_id", columns="axis", values="event_score", aggfunc="mean")
    prot, _ = protein_matrix()
    shared = wide_events.index.intersection(prot.index)
    if len(shared) == 0:
        raise RuntimeError("No shared cells between SCP1064 event scores and protein matrix.")
    if len(shared) > CELL_SUBSAMPLE:
        sample = pd.Index(rng.choice(shared.to_numpy(), size=CELL_SUBSAMPLE, replace=False))
    else:
        sample = shared
    wide_sample = wide_events.loc[sample]
    prot_sample = np.log1p(prot.loc[sample])

    rows: list[dict[str, object]] = []
    for axis, protein in axis_protein_pairs():
        if axis not in wide_sample.columns or protein not in prot_sample.columns:
            continue
        observed, null_max, empirical_p = _permute_spearman(
            wide_sample[axis].to_numpy(dtype=float),
            prot_sample[protein].to_numpy(dtype=float),
            rng,
            N_PERMUTATIONS,
        )
        rows.append(
            {
                "shuffle_type": "cell_protein_label_shuffle",
                "level": "cell",
                "axis": axis,
                "protein_name": protein,
                "n_units": len(sample),
                "n_permutations": N_PERMUTATIONS,
                "observed_spearman": observed,
                "max_abs_null_spearman": null_max,
                "empirical_p": empirical_p,
                "shuffle_gate_pass": bool(abs(observed) > null_max and empirical_p <= 0.05),
                "evaluated_for_gate": bool(("cell", axis, protein) in gate_pairs),
                "notes": "Protein labels were permuted across a deterministic 50,000-cell subsample; this is a lightweight label-shuffle gate, not an exhaustive heavy shuffle.",
            }
        )

    def grouped_shuffle(level: str, key: str, event_file: str, protein_file: str) -> None:
        events = read_tsv(RESULTS / event_file)
        proteins = read_tsv(RESULTS / protein_file)
        if events.empty or proteins.empty:
            return
        for axis, protein in axis_protein_pairs():
            e = events[events["axis"].eq(axis)][[key, "effect_size_vs_control"]].rename(
                columns={"effect_size_vs_control": "rna_event_effect"}
            )
            p = proteins[proteins["protein_name"].eq(protein)][[key, "effect_size_vs_control"]].rename(
                columns={"effect_size_vs_control": "protein_effect"}
            )
            merged = e.merge(p, on=key).dropna()
            if len(merged) < 5:
                continue
            observed, null_max, empirical_p = _permute_spearman(
                merged["rna_event_effect"].to_numpy(dtype=float),
                merged["protein_effect"].to_numpy(dtype=float),
                rng,
                N_PERMUTATIONS,
            )
            rows.append(
                {
                    "shuffle_type": f"{level}_effect_label_shuffle",
                    "level": level,
                    "axis": axis,
                    "protein_name": protein,
                    "n_units": len(merged),
                    "n_permutations": N_PERMUTATIONS,
                    "observed_spearman": observed,
                    "max_abs_null_spearman": null_max,
                    "empirical_p": empirical_p,
                    "shuffle_gate_pass": bool(abs(observed) > null_max and empirical_p <= 0.05),
                    "evaluated_for_gate": bool((level, axis, protein) in gate_pairs),
                    "notes": f"{level.capitalize()}-level protein effects were permuted relative to RNA event effects; this is a lightweight label-shuffle gate.",
                }
            )

    grouped_shuffle(
        "guide",
        "guide_id",
        "scp1064_guide_level_event_scores.tsv",
        "protein_marker_summary_by_guide.tsv",
    )
    grouped_shuffle(
        "target",
        "target_gene",
        "scp1064_target_level_event_scores.tsv",
        "protein_marker_summary_by_target.tsv",
    )

    shuffle = pd.DataFrame(rows)
    write_tsv(shuffle, RESULTS / "scp1064_lightweight_shuffle_summary.tsv")

    primary = read_tsv(RESULTS / "scp1064_outcome_alignment_summary.tsv")
    negative = read_tsv(RESULTS / "scp1064_negative_control_alignment.tsv")
    specificity = read_tsv(RESULTS / "scp1064_specificity_summary.tsv")
    primary_max = float(pd.to_numeric(primary["spearman"], errors="coerce").abs().max())
    negative_max = float(pd.to_numeric(negative["spearman"], errors="coerce").abs().max())
    gate_shuffle = shuffle[shuffle["evaluated_for_gate"].astype(bool)].copy() if not shuffle.empty else pd.DataFrame()
    shuffle_pass = bool(gate_shuffle["shuffle_gate_pass"].all()) if not gate_shuffle.empty else False
    max_shuffle = float(pd.to_numeric(gate_shuffle["max_abs_null_spearman"], errors="coerce").max()) if not gate_shuffle.empty else np.nan

    if specificity.empty:
        specificity = pd.DataFrame([{"dataset": "SCP1064"}])
    specificity.loc[:, "primary_max_abs_alignment"] = primary_max
    specificity.loc[:, "negative_max_abs_alignment"] = negative_max
    specificity.loc[:, "specificity_vs_random"] = (
        "completed_lightweight_label_shuffle_pass" if shuffle_pass else "completed_lightweight_label_shuffle_fail"
    )
    specificity.loc[:, "lightweight_shuffle_max_abs_null"] = max_shuffle
    specificity.loc[:, "lightweight_shuffle_n_gate_pairs"] = int(len(gate_shuffle))
    specificity.loc[:, "lightweight_shuffle_pass"] = shuffle_pass
    specificity.loc[:, "heavy_shuffle_status"] = "deferred_exhaustive_heavy_shuffle"
    specificity.loc[:, "specificity_vs_stress"] = "covered_by_melanoma_state_or_stress_control"
    specificity.loc[:, "specificity_vs_ribosome"] = "pass" if primary_max > negative_max + 0.02 else "fail"
    specificity.loc[:, "specificity_vs_mitochondrial"] = "pass" if primary_max > negative_max + 0.02 else "fail"
    specificity.loc[:, "negative_control_pass"] = bool((primary_max > negative_max + 0.02) and shuffle_pass)
    write_tsv(specificity, RESULTS / "scp1064_specificity_summary.tsv")

    return {"shuffle": shuffle, "specificity": specificity}


def main() -> None:
    outputs = run_lightweight_shuffles()
    print("SCP1064 lightweight shuffles complete")
    for name, df in outputs.items():
        print(f"{name}: {df.shape}")
    print(outputs["specificity"].to_string(index=False))


if __name__ == "__main__":
    main()
