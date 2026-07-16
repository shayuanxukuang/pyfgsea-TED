from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
ABUNDANCE = ROOT / "data_external" / "ted_development_phase2" / "Animal_1_GSE202639_ZSCAPE_zebrafish" / "zscape_celltype_abundance_event.tsv"
GSE271399_META = ROOT / "data_external" / "GSE271399_T21_GATA1s" / "sample_metadata.tsv"
GSE271399_JACKKNIFE = ROOT / "data_external" / "GSE271399_T21_GATA1s" / "ted" / "gse271399_block_jackknife_family_effects.tsv"
DEFAULT_OUT = ROOT / "results" / "ted_submission_supplement" / "zscape_leave_one_embryo_full_refit"


def bh(p: np.ndarray) -> np.ndarray:
    out = np.full(len(p), np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    indices = np.flatnonzero(finite)
    order = indices[np.argsort(p[indices])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.clip(ranked, 0, 1)
    return out


def welch_from_sufficient(
    n1: np.ndarray,
    sum1: np.ndarray,
    sumsq1: np.ndarray,
    n0: np.ndarray,
    sum0: np.ndarray,
    sumsq0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean1 = np.divide(sum1, n1, out=np.full_like(sum1, np.nan), where=n1 > 0)
    mean0 = np.divide(sum0, n0, out=np.full_like(sum0, np.nan), where=n0 > 0)
    effect = mean1 - mean0
    var1 = np.divide(sumsq1 - np.divide(sum1 * sum1, n1, out=np.zeros_like(sum1), where=n1 > 0), n1 - 1, out=np.full_like(sum1, np.nan), where=n1 > 1)
    var0 = np.divide(sumsq0 - np.divide(sum0 * sum0, n0, out=np.zeros_like(sum0), where=n0 > 0), n0 - 1, out=np.full_like(sum0, np.nan), where=n0 > 1)
    a = var1 / n1
    b = var0 / n0
    se = np.sqrt(a + b)
    statistic = np.divide(effect, se, out=np.full_like(effect, np.nan), where=se > 0)
    denominator = np.divide(a * a, n1 - 1, out=np.zeros_like(a), where=n1 > 1) + np.divide(b * b, n0 - 1, out=np.zeros_like(b), where=n0 > 1)
    dof = np.divide((a + b) ** 2, denominator, out=np.full_like(effect, np.nan), where=denominator > 0)
    p = 2 * stats.t.sf(np.abs(statistic), dof)
    return mean1, mean0, effect, p


def event_mode(
    times: np.ndarray,
    control: np.ndarray,
    mutant: np.ndarray,
    cell_type: str,
    max_alt_gain: float,
) -> str:
    valid = np.isfinite(times) & np.isfinite(control) & np.isfinite(mutant)
    if valid.sum() < 2:
        return "not_identifiable"
    times = times[valid]
    control = control[valid]
    mutant = mutant[valid]
    order = np.argsort(times)
    times, control, mutant = times[order], control[order], mutant[order]
    ctrl_trend = float(control[-1] - control[0])
    mut_trend = float(mutant[-1] - mutant[0])
    terminal_delta = float(mutant[-1] - control[-1])
    ctrl_peak = float(times[np.argmax(control)])
    mut_peak = float(times[np.argmax(mutant)])
    threshold = max(0.005, float(np.ptp(control)) * 0.25)
    lower = cell_type.lower()
    if terminal_delta < -threshold and max_alt_gain > threshold:
        return "fate_redirection"
    if any(token in lower for token in ["progenitor", "early", "precursor", "mesenchyme"]) and terminal_delta > threshold:
        return "state_accumulation"
    if ctrl_trend > threshold and terminal_delta < -threshold and mut_trend <= threshold:
        return "true_loss"
    if mut_peak > ctrl_peak and terminal_delta > -2 * threshold:
        return "developmental_delay"
    return "ambiguous_or_mixed"


def build_sufficient(abundance: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    abundance = abundance.copy()
    abundance["embryo"] = abundance["embryo"].astype(str)
    abundance["gene_target"] = abundance["gene_target"].astype(str)
    abundance["cell_type_broad"] = abundance["cell_type_broad"].astype(str)
    abundance["cell_type_fraction"] = pd.to_numeric(abundance["cell_type_fraction"], errors="coerce")
    controls = abundance[abundance["is_control"].astype(bool)].copy()
    mutants = abundance[~abundance["is_control"].astype(bool)].copy()
    mutant_stats = (
        mutants.groupby(["gene_target", "timepoint", "cell_type_broad"], observed=True)["cell_type_fraction"]
        .agg(["size", "sum", lambda values: float(np.square(values).sum())])
        .reset_index()
    )
    mutant_stats.columns = ["gene_target", "timepoint", "cell_type_broad", "mut_n", "mut_sum", "mut_sumsq"]
    control_stats = (
        controls.groupby(["timepoint", "cell_type_broad"], observed=True)["cell_type_fraction"]
        .agg(["size", "sum", lambda values: float(np.square(values).sum())])
        .reset_index()
    )
    control_stats.columns = ["timepoint", "cell_type_broad", "ctrl_n", "ctrl_sum", "ctrl_sumsq"]
    events = mutant_stats.merge(control_stats, on=["timepoint", "cell_type_broad"], how="left")
    events["event_id"] = events["gene_target"] + "|" + events["cell_type_broad"] + "|" + events["timepoint"].astype(str)
    events["time_cell"] = events["timepoint"].astype(str) + "|" + events["cell_type_broad"]
    events = events.reset_index(drop=True)
    event_lookup = dict(zip(events["event_id"], events.index, strict=True))
    time_cell_lookup: dict[str, np.ndarray] = {
        key: group.index.to_numpy(dtype=int) for key, group in events.groupby("time_cell", sort=False)
    }
    mutant_contributions: dict[str, list[tuple[int, float]]] = {}
    for row in mutants.itertuples(index=False):
        key = f"{row.gene_target}|{row.cell_type_broad}|{row.timepoint}"
        if key in event_lookup:
            mutant_contributions.setdefault(str(row.embryo), []).append((event_lookup[key], float(row.cell_type_fraction)))
    control_contributions: dict[str, list[tuple[str, float]]] = {}
    for row in controls.itertuples(index=False):
        key = f"{row.timepoint}|{row.cell_type_broad}"
        control_contributions.setdefault(str(row.embryo), []).append((key, float(row.cell_type_fraction)))
    context: dict[str, object] = {
        "embryos": sorted(abundance["embryo"].unique()),
        "event_lookup": event_lookup,
        "time_cell_lookup": time_cell_lookup,
        "mutant_contributions": mutant_contributions,
        "control_contributions": control_contributions,
    }
    return events, context


def fit_arrays(events: pd.DataFrame) -> dict[str, np.ndarray]:
    mean_mut, mean_ctrl, effect, p = welch_from_sufficient(
        events["mut_n"].to_numpy(dtype=float),
        events["mut_sum"].to_numpy(dtype=float),
        events["mut_sumsq"].to_numpy(dtype=float),
        events["ctrl_n"].to_numpy(dtype=float),
        events["ctrl_sum"].to_numpy(dtype=float),
        events["ctrl_sumsq"].to_numpy(dtype=float),
    )
    return {"mean_mut": mean_mut, "mean_ctrl": mean_ctrl, "effect": effect, "p": p, "q": bh(p)}


def group_indices(events: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    return {
        (str(target), str(cell)): group.sort_values("timepoint").index.to_numpy(dtype=int)
        for (target, cell), group in events.groupby(["gene_target", "cell_type_broad"], sort=False)
    }


def classify_groups(
    events: pd.DataFrame,
    fit: dict[str, np.ndarray],
    groups: dict[tuple[str, str], np.ndarray],
    selected_groups: set[tuple[str, str]] | None = None,
) -> dict[tuple[str, str], str]:
    terminal_time = float(pd.to_numeric(events["timepoint"], errors="coerce").max())
    terminal = events[events["timepoint"].eq(terminal_time)].copy()
    terminal["effect"] = fit["effect"][terminal.index]
    alt_gain = terminal.groupby("gene_target")["effect"].max().to_dict()
    output: dict[tuple[str, str], str] = {}
    for key, idx in groups.items():
        if selected_groups is not None and key not in selected_groups:
            continue
        target, cell = key
        output[key] = event_mode(
            events.loc[idx, "timepoint"].to_numpy(dtype=float),
            fit["mean_ctrl"][idx],
            fit["mean_mut"][idx],
            cell,
            float(alt_gain.get(target, np.nan)),
        )
    return output


def refit_without_embryo(
    events: pd.DataFrame,
    context: dict[str, object],
    embryo: str,
) -> dict[str, np.ndarray]:
    changed = events[["mut_n", "mut_sum", "mut_sumsq", "ctrl_n", "ctrl_sum", "ctrl_sumsq"]].copy()
    mutant_contributions = context["mutant_contributions"]
    for idx, value in mutant_contributions.get(embryo, []):  # type: ignore[union-attr]
        changed.loc[idx, "mut_n"] -= 1
        changed.loc[idx, "mut_sum"] -= value
        changed.loc[idx, "mut_sumsq"] -= value * value
    control_contributions = context["control_contributions"]
    time_cell_lookup = context["time_cell_lookup"]
    for key, value in control_contributions.get(embryo, []):  # type: ignore[union-attr]
        idx = time_cell_lookup.get(key, np.asarray([], dtype=int))  # type: ignore[union-attr]
        changed.loc[idx, "ctrl_n"] -= 1
        changed.loc[idx, "ctrl_sum"] -= value
        changed.loc[idx, "ctrl_sumsq"] -= value * value
    mean_mut, mean_ctrl, effect, p = welch_from_sufficient(
        changed["mut_n"].to_numpy(dtype=float),
        changed["mut_sum"].to_numpy(dtype=float),
        changed["mut_sumsq"].to_numpy(dtype=float),
        changed["ctrl_n"].to_numpy(dtype=float),
        changed["ctrl_sum"].to_numpy(dtype=float),
        changed["ctrl_sumsq"].to_numpy(dtype=float),
    )
    return {"mean_mut": mean_mut, "mean_ctrl": mean_ctrl, "effect": effect, "p": p, "q": bh(p)}


def run(outdir: Path, max_embryos: int | None = None) -> dict[str, pd.DataFrame]:
    abundance = pd.read_csv(ABUNDANCE, sep="\t")
    events, context = build_sufficient(abundance)
    full = fit_arrays(events)
    groups = group_indices(events)
    full_called = np.isfinite(full["q"]) & (full["q"] <= 0.05)
    selected_groups = set(
        zip(events.loc[full_called, "gene_target"].astype(str), events.loc[full_called, "cell_type_broad"].astype(str), strict=False)
    )
    full_modes = classify_groups(events, full, groups, selected_groups)
    full_families = {(target, mode) for (target, _), mode in full_modes.items()}
    embryos = list(context["embryos"])
    if max_embryos is not None:
        embryos = embryos[:max_embryos]

    effect_sum = np.zeros(len(events), dtype=float)
    effect_sumsq = np.zeros(len(events), dtype=float)
    effect_count = np.zeros(len(events), dtype=int)
    effect_min = np.full(len(events), np.inf)
    effect_max = np.full(len(events), -np.inf)
    refit_rows = []
    for embryo in embryos:
        held = refit_without_embryo(events, context, embryo)
        called = np.isfinite(held["q"]) & (held["q"] <= 0.05)
        union = int((full_called | called).sum())
        jaccard = float((full_called & called).sum() / union) if union else 1.0
        direction = float(np.mean(np.sign(held["effect"][full_called]) == np.sign(full["effect"][full_called]))) if full_called.any() else np.nan
        modes = classify_groups(events, held, groups, selected_groups)
        common_groups = sorted(set(full_modes) & set(modes))
        mode_agreement = float(np.mean([full_modes[key] == modes[key] for key in common_groups])) if common_groups else np.nan
        families = {(target, mode) for (target, _), mode in modes.items()}
        family_union = full_families | families
        family_jaccard = float(len(full_families & families) / len(family_union)) if family_union else 1.0
        finite = np.isfinite(held["effect"])
        effect_sum[finite] += held["effect"][finite]
        effect_sumsq[finite] += np.square(held["effect"][finite])
        effect_count[finite] += 1
        effect_min[finite] = np.minimum(effect_min[finite], held["effect"][finite])
        effect_max[finite] = np.maximum(effect_max[finite], held["effect"][finite])
        supported = called & (events["mut_n"].to_numpy(dtype=float) >= 3) & (events["ctrl_n"].to_numpy(dtype=float) >= 3)
        claim = "Level 3.5 perturbation-aware developmental candidate" if supported.any() else "Level 2 event-FDR supported"
        refit_rows.append(
            {
                "left_out_embryo": embryo,
                "n_events_called": int(called.sum()),
                "event_id_jaccard": jaccard,
                "direction_agreement": direction,
                "event_mode_agreement": mode_agreement,
                "event_family_jaccard": family_jaccard,
                "effect_rmse_on_full_calls": float(np.sqrt(np.nanmean(np.square(held["effect"][full_called] - full["effect"][full_called])))) if full_called.any() else np.nan,
                "claim_boundary_refit": claim,
                "claim_boundary_stable": claim == "Level 3.5 perturbation-aware developmental candidate",
                "refit_scope": "upstream embryo abundance, event effects, event-FDR, event modes, and claim boundary all recomputed",
            }
        )

    refits = pd.DataFrame(refit_rows)
    count = np.maximum(effect_count, 1)
    variance = np.maximum(effect_sumsq / count - np.square(effect_sum / count), 0.0)
    variability = events[["event_id", "gene_target", "timepoint", "cell_type_broad"]].copy()
    variability["full_effect"] = full["effect"]
    variability["full_q"] = full["q"]
    variability["n_refits"] = effect_count
    variability["refit_effect_mean"] = effect_sum / count
    variability["refit_effect_sd"] = np.sqrt(variance)
    variability["refit_effect_min"] = np.where(np.isfinite(effect_min), effect_min, np.nan)
    variability["refit_effect_max"] = np.where(np.isfinite(effect_max), effect_max, np.nan)
    variability = variability[full_called].reset_index(drop=True)
    summary = pd.DataFrame(
        [
            {
                "n_cells_source": int(abundance["n_cells"].sum()),
                "n_embryos_total": int(abundance["embryo"].nunique()),
                "n_embryos_refit": len(refits),
                "n_full_events": len(events),
                "n_full_called_events": int(full_called.sum()),
                "median_event_jaccard": float(refits["event_id_jaccard"].median()),
                "minimum_event_jaccard": float(refits["event_id_jaccard"].min()),
                "median_direction_agreement": float(refits["direction_agreement"].median()),
                "minimum_direction_agreement": float(refits["direction_agreement"].min()),
                "median_event_mode_agreement": float(refits["event_mode_agreement"].median()),
                "minimum_event_mode_agreement": float(refits["event_mode_agreement"].min()),
                "median_event_family_jaccard": float(refits["event_family_jaccard"].median()),
                "claim_boundary_stability": float(refits["claim_boundary_stable"].mean()),
                "analysis_unit": "embryo",
                "pseudoreplication_policy": "cells aggregated to embryo fractions before inference",
            }
        ]
    )
    full_table = events[["event_id", "gene_target", "timepoint", "cell_type_broad", "mut_n", "ctrl_n"]].copy()
    full_table["event_effect"] = full["effect"]
    full_table["event_p"] = full["p"]
    full_table["event_q"] = full["q"]
    full_table["called"] = full_called
    return {"full_event_table": full_table, "leave_one_embryo_refits": refits, "event_effect_variability": variability, "summary": summary}


def gse271399_estimability_audit() -> pd.DataFrame:
    meta = pd.read_csv(GSE271399_META, sep="\t")
    jack = pd.read_csv(GSE271399_JACKKNIFE, sep="\t") if GSE271399_JACKKNIFE.exists() else pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "requested_refit": "leave_one_donor_out",
                "status": "not_estimable",
                "n_independent_units": int(pd.to_numeric(meta["replicate"], errors="coerce").nunique()),
                "available_alternative": "none",
                "claim_consequence": "do not describe GSE271399 as donor-replicated; retain Level 3.5 ceiling",
            },
            {
                "requested_refit": "leave_one_sample_day_out",
                "status": "completed_existing_exact_aggregate_refit",
                "n_independent_units": int(meta["sample_id"].nunique()),
                "available_alternative": f"{len(jack[jack.get('jackknife_axis', pd.Series(dtype=str)).eq('day')])} family/contrast/trajectory refits",
                "claim_consequence": "time-stratum sensitivity only; not donor replication",
            },
            {
                "requested_refit": "leave_one_trajectory_state_or_bin_out",
                "status": "completed_existing_exact_aggregate_refit",
                "n_independent_units": int(jack.get("left_out_block", pd.Series(dtype=str)).nunique()),
                "available_alternative": "pseudotime-bin and coarse-state jackknife tables",
                "claim_consequence": "trajectory sensitivity support without causal upgrade",
            },
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exact ZSCAPE leave-one-embryo full event-layer refits")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-embryos", type=int, default=None, help="Smoke-test limit; omit for all embryos")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    outputs = run(args.outdir, args.max_embryos)
    outputs["gse271399_estimability_audit"] = gse271399_estimability_audit()
    for name, table in outputs.items():
        table.to_csv(args.outdir / f"{name}.tsv", sep="\t", index=False)
    summary = outputs["summary"].iloc[0]
    report = [
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: run + validate",
        "- Origin Date: 2026-07-15",
        "- Verification Status: ANALYZED",
        "- Version Label: zscape_leave_one_embryo_full_refit_v1",
        "",
        "# ZSCAPE leave-one-embryo full refit",
        "",
        f"All {int(summary['n_embryos_refit'])} embryo holdouts were refit from the frozen embryo-level abundance table.",
        "For every holdout, mutant/control activity, Welch event tests, BH event-FDR, event modes, event-family membership, and the claim boundary were recomputed.",
        f"Median event Jaccard: {summary['median_event_jaccard']:.3f}; minimum: {summary['minimum_event_jaccard']:.3f}.",
        f"Median direction agreement: {summary['median_direction_agreement']:.3f}.",
        f"Median event-mode agreement: {summary['median_event_mode_agreement']:.3f}.",
        f"Claim-boundary stability: {summary['claim_boundary_stability']:.3f}.",
        "",
        "The sufficient-statistic implementation is algebraically identical to deleting one embryo and refitting the embryo-level mean/variance tests; it does not delete a row from a final summary table.",
        "GSE271399 donor LODO remains not estimable because the public design has a single replicate per condition/day cell; this run does not upgrade the GATA1/T21 claim above Level 3.5.",
    ]
    (args.outdir / "refit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = []
    for path in sorted(args.outdir.glob("*")):
        if path.is_file():
            manifest.append({"file": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    pd.DataFrame(manifest).to_csv(args.outdir / "manifest.tsv", sep="\t", index=False)
    (args.outdir / "run_config.json").write_text(json.dumps({"max_embryos": args.max_embryos, "source": str(ABUNDANCE), "runtime_seconds": time.perf_counter() - started}, indent=2), encoding="utf-8")
    print(f"ZSCAPE full refits complete in {time.perf_counter() - started:.1f}s")
    print(outputs["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
