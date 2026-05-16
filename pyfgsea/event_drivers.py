from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


_DRIVER_OUTPUT_FILENAMES = {
    "event_driver_score": "event_driver_score.tsv",
    "event_regulator_activity": "event_regulator_activity.tsv",
    "event_driver_network": "event_driver_network.tsv",
    "driver_specificity_report": "driver_specificity_report.tsv",
}


_DEFAULT_REGULATOR_TARGETS = {
    "GATA1": {
        "ALAS2",
        "ALAD",
        "ANK1",
        "EPB42",
        "ERMAP",
        "FECH",
        "GYPA",
        "HBA-A1",
        "HBA-A2",
        "HBB-BS",
        "HBB-BT",
        "HBB",
        "HMBS",
        "KLF1",
        "SLC4A1",
        "TAL1",
        "TFRC",
        "UROD",
    },
    "KLF1": {
        "ALAS2",
        "BCL2L1",
        "EPB42",
        "ERMAP",
        "GYPA",
        "HBA-A1",
        "HBA-A2",
        "HBB-BS",
        "HBB-BT",
        "HBB",
        "SLC4A1",
        "TFRC",
    },
    "TAL1": {"ALAS2", "GATA1", "GYPA", "HBB", "KLF1", "SLC4A1", "TFRC"},
    "STAT5A": {"BCL2L1", "CISH", "EPOR", "PIM1", "SOCS2", "SOCS3"},
    "STAT5B": {"BCL2L1", "CISH", "EPOR", "PIM1", "SOCS2", "SOCS3"},
    "FOXO3": {"BCL2L11", "BTG2", "CCNG2", "CDKN1A", "CDKN1B", "SOD2"},
    "MYC": {"EEF1A1", "NPM1", "RPL13A", "RPL18", "RPLP0", "RPS3", "RPS6"},
    "SPI1": {"CSF1R", "CSF2RB", "FCER1G", "ITGAM", "LYZ2", "MPO", "TYROBP"},
    "CEBPA": {"CSF3R", "ELANE", "MPO", "PRTN3", "SPI1"},
}

_TF_SYMBOLS = {
    "ATF4",
    "BCL11A",
    "CEBPA",
    "CEBPB",
    "E2F1",
    "E2F2",
    "E2F4",
    "FOXO3",
    "GATA1",
    "GATA2",
    "GFI1B",
    "KLF1",
    "KLF3",
    "KLF6",
    "MYB",
    "MYC",
    "NFE2",
    "NFE2L2",
    "RUNX1",
    "SPI1",
    "STAT5A",
    "STAT5B",
    "TAL1",
    "ZBTB7A",
}
_SURFACE_RECEPTORS = {
    "CD34",
    "CD44",
    "CD47",
    "CD71",
    "CSF1R",
    "CSF2RB",
    "CSF3R",
    "EPOR",
    "FCER1G",
    "FTH1",
    "ITGAM",
    "KIT",
    "MPL",
    "SLC11A2",
    "SLC25A37",
    "SLC40A1",
    "SLC4A1",
    "TFRC",
}
_ENZYMES = {
    "ALAD",
    "ALAS2",
    "CPOX",
    "FECH",
    "FTL1",
    "GCLC",
    "GLRX5",
    "GSR",
    "HMBS",
    "PPOX",
    "STEAP3",
    "UROD",
    "UROS",
}
_KNOWN_REGULATORS = _TF_SYMBOLS | {"BTG2", "CCNG2", "CDKN1A", "CDKN1B", "CISH", "SOCS2", "SOCS3"}


def _pick_column(df: pd.DataFrame, *candidates: Optional[str]) -> Optional[str]:
    lower_to_original = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate is None:
            continue
        if candidate in df.columns:
            return candidate
        lower = str(candidate).lower()
        if lower in lower_to_original:
            return lower_to_original[lower]
    return None


def _as_symbol(gene: object) -> str:
    return str(gene).strip()


def _symbol_key(gene: object) -> str:
    return _as_symbol(gene).upper()


def _safe_float(value: object, default: float = np.nan) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(out) if np.isfinite(out) else float(default)


def _parse_genes(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    genes = []
    for token in str(value).replace(",", ";").split(";"):
        gene = token.strip()
        if gene and gene.lower() != "nan":
            genes.append(gene)
    return genes


def _event_label(row: pd.Series, label_col: Optional[str]) -> str:
    if label_col is not None and label_col in row.index and pd.notna(row[label_col]):
        return str(row[label_col])
    for col in ("pathway", "Pathway", "module", "event_id"):
        if col in row.index and pd.notna(row[col]):
            return str(row[col])
    return ""


def _prepare_events(
    events: pd.DataFrame,
    *,
    event_id_col: Optional[str],
    label_col: Optional[str],
    condition_col: Optional[str],
    dataset_col: Optional[str],
) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame()
    event_id_col = _pick_column(events, event_id_col, "event_id")
    label_col = label_col or _pick_column(events, "pathway", "Pathway", "module")
    rows = []
    for idx, row in events.reset_index(drop=True).iterrows():
        label = _event_label(row, label_col)
        event_id = str(row[event_id_col]) if event_id_col and pd.notna(row.get(event_id_col)) else f"{label}|event_{idx + 1:03d}"
        dataset = str(row[dataset_col]) if dataset_col and dataset_col in row.index and pd.notna(row[dataset_col]) else ""
        condition = (
            str(row[condition_col])
            if condition_col and condition_col in row.index and pd.notna(row[condition_col])
            else ""
        )
        future_fate = str(row["future_fate"]) if "future_fate" in row.index and pd.notna(row["future_fate"]) else ""
        rows.append(
            {
                "__event_id": event_id,
                "__event_label": label,
                "__dataset": dataset,
                "__condition": condition,
                "__future_fate": future_fate,
                "__peak_effect": _event_peak_effect(row),
                "__event_q": _safe_float(row.get("event_q", row.get("event_fdr", np.nan))),
                **{str(col): row[col] for col in events.columns},
            }
        )
    return pd.DataFrame(rows)


def _event_peak_effect(row: pd.Series) -> float:
    for col in ("peak_effect", "abs_peak_effect", "event_peak_score", "peak_NES", "event_score", "effect_size"):
        if col in row.index:
            value = _safe_float(row[col])
            if np.isfinite(value) and abs(value) > 0:
                return abs(value)
    if "observed_event_statistic" in row.index:
        value = _safe_float(row["observed_event_statistic"])
        if np.isfinite(value):
            return abs(value)
    if "AUC_abs" in row.index:
        value = _safe_float(row["AUC_abs"])
        if np.isfinite(value):
            return abs(value)
    return 1.0


def _normalise_target_map(regulator_targets: Optional[dict[str, set[str] | list[str]]]) -> dict[str, set[str]]:
    raw = regulator_targets or _DEFAULT_REGULATOR_TARGETS
    return {str(reg).upper(): {_symbol_key(gene) for gene in targets} for reg, targets in raw.items()}


def _gene_class(gene: object) -> str:
    key = _symbol_key(gene)
    if key in _TF_SYMBOLS:
        return "transcription_factor"
    if key in _SURFACE_RECEPTORS or (key.startswith("CD") and key[2:].isdigit()):
        return "surface_receptor_or_transporter"
    if key in _ENZYMES or key.endswith("ASE"):
        return "enzyme"
    if key in _KNOWN_REGULATORS:
        return "known_regulator"
    return "target_gene"


def _regulatory_weight(gene: object, overrides: Optional[dict[str, float]]) -> float:
    key = _symbol_key(gene)
    if overrides and key in {str(k).upper(): v for k, v in overrides.items()}:
        return float({str(k).upper(): v for k, v in overrides.items()}[key])
    klass = _gene_class(gene)
    if klass == "transcription_factor":
        return 1.6
    if klass == "known_regulator":
        return 1.45
    if klass == "surface_receptor_or_transporter":
        return 1.3
    if klass == "enzyme":
        return 1.2
    return 1.0


def _event_match_mask(leading: pd.DataFrame, event: pd.Series, event_id_col: Optional[str], label_col: Optional[str]) -> pd.Series:
    mask = pd.Series(True, index=leading.index)
    if event_id_col and event_id_col in leading.columns:
        direct = leading[event_id_col].astype(str) == str(event["__event_id"])
        if direct.any():
            return direct
    lead_label_col = label_col or _pick_column(leading, "pathway", "Pathway", "module")
    if lead_label_col and lead_label_col in leading.columns:
        mask &= leading[lead_label_col].astype(str) == str(event["__event_label"])
    for src, target in (
        ("dataset", "__dataset"),
        ("condition", "__condition"),
        ("future_fate", "__future_fate"),
    ):
        if src in leading.columns and str(event[target]):
            same = leading[src].astype(str) == str(event[target])
            if same.any():
                mask &= same
    return mask


def _leading_probability(row: pd.Series, group: pd.DataFrame) -> float:
    for col in ("bootstrap_probability", "leading_edge_probability", "leading_edge_frequency", "probability"):
        if col in row.index:
            value = _safe_float(row[col])
            if np.isfinite(value):
                return max(min(value, 1.0), 0.0)
    if "relative_weight" in row.index:
        value = _safe_float(row["relative_weight"])
        if np.isfinite(value):
            return max(min(value, 1.0), 0.0)
    if "module_weight" in row.index:
        max_weight = pd.to_numeric(group["module_weight"], errors="coerce").max()
        value = _safe_float(row["module_weight"])
        if np.isfinite(value) and np.isfinite(max_weight) and max_weight > 0:
            return max(min(value / float(max_weight), 1.0), 0.0)
    return 1.0


def _gene_peak_effect(row: pd.Series, event: pd.Series) -> float:
    if "module_weight" in row.index and "event_peak_score" in row.index:
        weight = _safe_float(row["module_weight"])
        peak = _safe_float(row["event_peak_score"])
        if np.isfinite(weight) and np.isfinite(peak):
            return abs(weight * peak)
    for col in ("peak_effect", "abs_peak_effect", "gene_peak_effect", "event_peak_score", "effect_size"):
        if col in row.index:
            value = _safe_float(row[col])
            if np.isfinite(value):
                return abs(value)
    return float(event["__peak_effect"])


def _driver_rows_from_gene_table(
    events: pd.DataFrame,
    leading_edges: pd.DataFrame,
    *,
    event_id_col: Optional[str],
    label_col: Optional[str],
    gene_col: str,
) -> pd.DataFrame:
    if leading_edges is None or leading_edges.empty:
        return pd.DataFrame()
    gene_col = _pick_column(leading_edges, gene_col, "gene", "Gene")
    if gene_col is None:
        return pd.DataFrame()
    lead_event_id = _pick_column(leading_edges, event_id_col, "event_id")
    lead_label_col = _pick_column(leading_edges, label_col, "pathway", "Pathway", "module")
    rows = []
    for _, event in events.iterrows():
        group = leading_edges.loc[_event_match_mask(leading_edges, event, lead_event_id, lead_label_col)].copy()
        if group.empty:
            continue
        for _, row in group.iterrows():
            gene = _as_symbol(row[gene_col])
            if not gene:
                continue
            rows.append(
                {
                    "dataset": event["__dataset"],
                    "condition": event["__condition"],
                    "future_fate": event["__future_fate"],
                    "event_id": event["__event_id"],
                    "event": event["__event_label"],
                    "gene": gene,
                    "bootstrap_leading_edge_probability": _leading_probability(row, group),
                    "peak_effect": _gene_peak_effect(row, event),
                    "event_q": event["__event_q"],
                    "reported_specificity": max(
                        _safe_float(row.get("condition_specificity_delta", np.nan), default=np.nan),
                        0.0,
                    )
                    if "condition_specificity_delta" in row.index
                    else np.nan,
                    "source": "leading_edge_gene_table",
                }
            )
    return pd.DataFrame(rows)


def _driver_rows_from_window_table(
    events: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    label_col: Optional[str],
    gene_list_col: str,
    time_col: str,
) -> pd.DataFrame:
    if windows is None or windows.empty or gene_list_col not in windows:
        return pd.DataFrame()
    lead_label_col = _pick_column(windows, label_col, "pathway", "Pathway", "module")
    if lead_label_col is None or time_col not in windows:
        return pd.DataFrame()
    rows = []
    for _, event in events.iterrows():
        group = windows[windows[lead_label_col].astype(str) == str(event["__event_label"])].copy()
        if "dataset" in group.columns and str(event["__dataset"]):
            same = group["dataset"].astype(str) == str(event["__dataset"])
            if same.any():
                group = group[same]
        if "condition" in group.columns and str(event["__condition"]):
            same = group["condition"].astype(str) == str(event["__condition"])
            if same.any():
                group = group[same]
        if group.empty:
            continue
        peak_time = _safe_float(event.get("peak_time", event.get("event_peak", np.nan)))
        if np.isfinite(peak_time):
            group["__distance"] = (pd.to_numeric(group[time_col], errors="coerce") - peak_time).abs()
            selected = group.nsmallest(max(min(len(group), 3), 1), "__distance")
        else:
            selected = group
        counts: dict[str, int] = {}
        for value in selected[gene_list_col]:
            for gene in _parse_genes(value):
                counts[gene] = counts.get(gene, 0) + 1
        denom = max(len(selected), 1)
        for gene, count in counts.items():
            rows.append(
                {
                    "dataset": event["__dataset"],
                    "condition": event["__condition"],
                    "future_fate": event["__future_fate"],
                    "event_id": event["__event_id"],
                    "event": event["__event_label"],
                    "gene": gene,
                    "bootstrap_leading_edge_probability": float(count) / float(denom),
                    "peak_effect": float(event["__peak_effect"]),
                    "event_q": event["__event_q"],
                    "source": "window_leading_edge_table",
                }
            )
    return pd.DataFrame(rows)


def _specificity(values: pd.DataFrame) -> pd.Series:
    out = pd.Series(1.0, index=values.index, dtype=float)
    if values.empty:
        return out
    pi = pd.to_numeric(values["bootstrap_leading_edge_probability"], errors="coerce").fillna(0.0)
    for gene, idx in values.groupby("gene").groups.items():
        gene_idx = list(idx)
        if len(gene_idx) == 1:
            out.loc[gene_idx] = 1.0
            continue
        gene_pi = pi.loc[gene_idx]
        for row_idx in gene_idx:
            cur = float(gene_pi.loc[row_idx])
            ratios = []
            for other_idx in gene_idx:
                if other_idx == row_idx:
                    continue
                other = float(gene_pi.loc[other_idx])
                denom = max(cur, other)
                ratios.append(min(cur, other) / denom if denom > 0 else 1.0)
            out.loc[row_idx] = float(1.0 - max(ratios)) if ratios else 1.0
    return out.clip(lower=0.0, upper=1.0)


def _attach_driver_scores(
    drivers: pd.DataFrame,
    *,
    regulatory_weights: Optional[dict[str, float]],
    top_genes_per_event: Optional[int],
) -> pd.DataFrame:
    if drivers.empty:
        return drivers
    work = drivers.copy()
    work["bootstrap_leading_edge_probability"] = pd.to_numeric(
        work["bootstrap_leading_edge_probability"], errors="coerce"
    ).fillna(0.0)
    work["peak_effect"] = pd.to_numeric(work["peak_effect"], errors="coerce").abs().fillna(0.0)
    work["specificity"] = _specificity(work)
    if "reported_specificity" in work.columns:
        reported = pd.to_numeric(work["reported_specificity"], errors="coerce").clip(lower=0.0, upper=1.0)
        work["specificity"] = np.maximum(work["specificity"], reported.fillna(0.0))
    work["regulatory_class"] = work["gene"].map(_gene_class)
    work["regulatory_weight"] = work["gene"].map(lambda gene: _regulatory_weight(gene, regulatory_weights))
    work["driver_score"] = (
        work["bootstrap_leading_edge_probability"]
        * work["peak_effect"]
        * work["specificity"]
        * work["regulatory_weight"]
    )
    work["condition_specificity"] = work["specificity"]
    work["bootstrap_support"] = work["bootstrap_leading_edge_probability"]
    work = work.sort_values(["event_id", "driver_score", "gene"], ascending=[True, False, True])
    if top_genes_per_event is not None and int(top_genes_per_event) > 0:
        work = work.groupby("event_id", sort=False).head(int(top_genes_per_event)).reset_index(drop=True)
    return work.reset_index(drop=True)


def _regulators_for_gene(gene: str, target_map: dict[str, set[str]]) -> list[tuple[str, str]]:
    key = _symbol_key(gene)
    regulators = [(reg, "target_network") for reg, targets in target_map.items() if key in targets]
    klass = _gene_class(gene)
    if klass in {"transcription_factor", "known_regulator", "surface_receptor_or_transporter", "enzyme"}:
        regulators.append((_symbol_key(gene), "self_driver"))
    if not regulators:
        regulators.append(("UNMAPPED_DRIVER", "unmapped_driver"))
    return regulators


def _regulator_tables(
    drivers: pd.DataFrame,
    *,
    regulator_targets: Optional[dict[str, set[str] | list[str]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if drivers.empty:
        return pd.DataFrame(), pd.DataFrame()
    target_map = _normalise_target_map(regulator_targets)
    network_rows = []
    for row in drivers.itertuples(index=False):
        for regulator, edge_type in _regulators_for_gene(row.gene, target_map):
            network_rows.append(
                {
                    "dataset": row.dataset,
                    "condition": row.condition,
                    "future_fate": row.future_fate,
                    "event_id": row.event_id,
                    "event": row.event,
                    "regulator": regulator,
                    "target_gene": row.gene,
                    "edge_type": edge_type,
                    "driver_score_contribution": float(row.driver_score),
                    "target_driver_score": float(row.driver_score),
                    "target_regulatory_class": row.regulatory_class,
                }
            )
    network = pd.DataFrame(network_rows)
    if network.empty:
        return pd.DataFrame(), network
    rows = []
    grouped = network.groupby(["dataset", "condition", "future_fate", "event_id", "event", "regulator"], dropna=False, sort=False)
    for keys, group in grouped:
        dataset, condition, future_fate, event_id, event, regulator = keys
        targets = (
            group.sort_values("driver_score_contribution", ascending=False)["target_gene"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        rows.append(
            {
                "dataset": dataset,
                "condition": condition,
                "future_fate": future_fate,
                "event_id": event_id,
                "event": event,
                "regulator": regulator,
                "regulator_activity": float(pd.to_numeric(group["driver_score_contribution"], errors="coerce").sum()),
                "n_target_drivers": int(group["target_gene"].nunique()),
                "top_target_genes": ";".join(targets[:10]),
                "edge_types": ";".join(sorted(set(group["edge_type"].astype(str)))),
            }
        )
    activity = pd.DataFrame(rows).sort_values(
        ["event_id", "regulator_activity", "regulator"], ascending=[True, False, True]
    )
    return activity.reset_index(drop=True), network.reset_index(drop=True)


def _specificity_report(drivers: pd.DataFrame) -> pd.DataFrame:
    if drivers.empty:
        return pd.DataFrame()
    rows = []
    for gene, group in drivers.groupby("gene", sort=False):
        best = group.sort_values("driver_score", ascending=False).iloc[0]
        rows.append(
            {
                "gene": gene,
                "regulatory_class": _gene_class(gene),
                "n_events": int(group["event_id"].nunique()),
                "n_conditions": int(group["condition"].replace("", np.nan).dropna().nunique()),
                "mean_bootstrap_probability": float(
                    pd.to_numeric(group["bootstrap_leading_edge_probability"], errors="coerce").mean()
                ),
                "max_bootstrap_probability": float(
                    pd.to_numeric(group["bootstrap_leading_edge_probability"], errors="coerce").max()
                ),
                "mean_specificity": float(pd.to_numeric(group["specificity"], errors="coerce").mean()),
                "max_driver_score": float(pd.to_numeric(group["driver_score"], errors="coerce").max()),
                "top_event_id": best["event_id"],
                "top_event": best["event"],
                "top_condition": best["condition"],
            }
        )
    return pd.DataFrame(rows).sort_values(["max_driver_score", "gene"], ascending=[False, True]).reset_index(drop=True)


def score_event_drivers(
    events: pd.DataFrame,
    leading_edges: Optional[pd.DataFrame] = None,
    *,
    event_windows: Optional[pd.DataFrame] = None,
    regulator_targets: Optional[dict[str, set[str] | list[str]]] = None,
    regulatory_weights: Optional[dict[str, float]] = None,
    event_id_col: Optional[str] = "event_id",
    label_col: Optional[str] = None,
    condition_col: Optional[str] = "condition",
    dataset_col: Optional[str] = "dataset",
    gene_col: str = "gene",
    gene_list_col: str = "leading_edge",
    window_time_col: str = "center_time",
    top_genes_per_event: Optional[int] = 25,
) -> dict[str, pd.DataFrame]:
    """
    Score event-level driver genes and regulators.

    ``DriverScore_g(E)`` combines bootstrap leading-edge probability, peak
    effect, event specificity, and a small regulatory weight for TFs,
    receptors/transporters, enzymes, and known regulators. Regulator activity
    is then the sum of driver scores over target genes.
    """
    prepared_events = _prepare_events(
        events,
        event_id_col=event_id_col,
        label_col=label_col,
        condition_col=condition_col,
        dataset_col=dataset_col,
    )
    if prepared_events.empty:
        raise ValueError("events must contain at least one event")
    rows = []
    if leading_edges is not None and not leading_edges.empty:
        gene_rows = _driver_rows_from_gene_table(
            prepared_events,
            leading_edges,
            event_id_col=event_id_col,
            label_col=label_col,
            gene_col=gene_col,
        )
        if not gene_rows.empty:
            rows.append(gene_rows)
    if event_windows is not None and not event_windows.empty:
        window_time = _pick_column(event_windows, window_time_col, "pt_mid", "center_time")
        if window_time is not None:
            window_rows = _driver_rows_from_window_table(
                prepared_events,
                event_windows,
                label_col=label_col,
                gene_list_col=gene_list_col,
                time_col=window_time,
            )
            if not window_rows.empty:
                rows.append(window_rows)
    raw_drivers = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    drivers = _attach_driver_scores(
        raw_drivers.drop_duplicates(["event_id", "gene", "source"]) if not raw_drivers.empty else raw_drivers,
        regulatory_weights=regulatory_weights,
        top_genes_per_event=top_genes_per_event,
    )
    activity, network = _regulator_tables(drivers, regulator_targets=regulator_targets)
    specificity = _specificity_report(drivers)
    return {
        "event_driver_score": drivers,
        "event_regulator_activity": activity,
        "event_driver_network": network,
        "driver_specificity_report": specificity,
    }


def write_event_driver_scores(
    tables: dict[str, pd.DataFrame],
    outdir: str | Path,
    *,
    sep: str = "\t",
) -> dict[str, Path]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, filename in _DRIVER_OUTPUT_FILENAMES.items():
        table = tables.get(key, pd.DataFrame())
        path = out / filename
        table.to_csv(path, sep=sep, index=False, na_rep="NA")
        paths[key] = path
    return paths
