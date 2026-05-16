from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


_GRAPH_OUTPUT_FILENAMES = {
    "event_order_probability_matrix": "event_order_probability_matrix.tsv",
    "condition_event_graph_edges": "condition_event_graph_edges.tsv",
    "event_graph_rewiring": "event_graph_rewiring.tsv",
    "event_graph_bootstrap_support": "event_graph_bootstrap_support.tsv",
}


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


def _node_label(row: pd.Series, label_col: str) -> str:
    return str(row.get(label_col, row.get("event_id", "")))


def _event_id(row: pd.Series, id_col: Optional[str], label_col: str) -> str:
    if id_col is not None and id_col in row.index and pd.notna(row[id_col]):
        return str(row[id_col])
    return _node_label(row, label_col)


def _prepare_events(
    events: pd.DataFrame,
    *,
    condition_col: str,
    time_col: str,
    event_id_col: Optional[str],
    label_col: Optional[str],
) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame()
    if condition_col not in events:
        raise ValueError(f"Missing condition column '{condition_col}'")
    if time_col not in events:
        raise ValueError(f"Missing event time column '{time_col}'")
    label_col = label_col or _pick_column(events, "pathway", "Pathway", "module", "event_id")
    if label_col is None:
        raise ValueError("Could not infer event label column")
    event_id_col = _pick_column(events, event_id_col, "event_id")
    rows = []
    for _, row in events.iterrows():
        time = pd.to_numeric(pd.Series([row[time_col]]), errors="coerce").iloc[0]
        if not np.isfinite(time):
            continue
        rows.append(
            {
                "condition": str(row[condition_col]),
                "event_id": _event_id(row, event_id_col, label_col),
                "event_label": _node_label(row, label_col),
                "event_time": float(time),
                "event_q": row.get("event_q", np.nan),
                "effect_type": row.get("effect_type", row.get("event_label", "")),
            }
        )
    return pd.DataFrame(rows).drop_duplicates(["condition", "event_id"])


def _bootstrap_probability(
    bootstrap_events: pd.DataFrame,
    *,
    condition: str,
    source_id: str,
    target_id: str,
    condition_col: str,
    time_col: str,
    event_id_col: Optional[str],
    label_col: str,
    bootstrap_col: str,
) -> tuple[float, int]:
    if bootstrap_events is None or bootstrap_events.empty:
        return np.nan, 0
    if bootstrap_col not in bootstrap_events or condition_col not in bootstrap_events:
        return np.nan, 0
    if time_col not in bootstrap_events:
        return np.nan, 0
    event_id_col = _pick_column(bootstrap_events, event_id_col, "event_id")
    work = bootstrap_events[bootstrap_events[condition_col].astype(str) == str(condition)].copy()
    if work.empty:
        return np.nan, 0
    if event_id_col is not None:
        work["__event_id"] = work[event_id_col].astype(str)
    else:
        work["__event_id"] = work[label_col].astype(str)
    flags = []
    for _, group in work.groupby(bootstrap_col, sort=False):
        left = pd.to_numeric(
            group.loc[group["__event_id"] == str(source_id), time_col],
            errors="coerce",
        ).dropna()
        right = pd.to_numeric(
            group.loc[group["__event_id"] == str(target_id), time_col],
            errors="coerce",
        ).dropna()
        if left.empty or right.empty:
            continue
        flags.append(float(left.min()) < float(right.min()))
    if not flags:
        return np.nan, 0
    return float(np.mean(flags)), int(len(flags))


def _probability_rows(
    nodes: pd.DataFrame,
    bootstrap_events: Optional[pd.DataFrame],
    *,
    condition_col: str,
    time_col: str,
    event_id_col: Optional[str],
    label_col: str,
    bootstrap_col: str,
) -> pd.DataFrame:
    rows = []
    for condition, group in nodes.groupby("condition", sort=False):
        group = group.sort_values("event_time").reset_index(drop=True)
        for i, source in group.iterrows():
            for j, target in group.iterrows():
                if i == j:
                    continue
                p_boot, n_boot = _bootstrap_probability(
                    bootstrap_events,
                    condition=condition,
                    source_id=source["event_id"],
                    target_id=target["event_id"],
                    condition_col=condition_col,
                    time_col=time_col,
                    event_id_col=event_id_col,
                    label_col=label_col,
                    bootstrap_col=bootstrap_col,
                )
                if np.isfinite(p_boot):
                    prob = p_boot
                    support_type = "bootstrap"
                else:
                    delta = float(target["event_time"] - source["event_time"])
                    prob = 1.0 if delta > 0 else (0.0 if delta < 0 else 0.5)
                    support_type = "deterministic"
                rows.append(
                    {
                        "condition": condition,
                        "source_event_id": source["event_id"],
                        "target_event_id": target["event_id"],
                        "source_event": source["event_label"],
                        "target_event": target["event_label"],
                        "source_time": float(source["event_time"]),
                        "target_time": float(target["event_time"]),
                        "delta_time": float(target["event_time"] - source["event_time"]),
                        "order_probability": float(prob),
                        "n_bootstrap": int(n_boot),
                        "support_type": support_type,
                    }
                )
    return pd.DataFrame(rows)


def _edge_table(prob: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if prob is None or prob.empty:
        return pd.DataFrame()
    edges = prob[
        (pd.to_numeric(prob["order_probability"], errors="coerce") >= float(threshold))
        & (pd.to_numeric(prob["delta_time"], errors="coerce") > 0)
    ].copy()
    if edges.empty:
        return edges
    edges["edge_id"] = (
        edges["condition"].astype(str)
        + "|"
        + edges["source_event"].astype(str)
        + "->"
        + edges["target_event"].astype(str)
    )
    edges["edge_type"] = "stable_order"
    return edges.sort_values(
        ["condition", "source_time", "target_time", "source_event", "target_event"]
    ).reset_index(drop=True)


def _rewiring_table(edges: pd.DataFrame, nodes: pd.DataFrame, reference: Optional[str], query: Optional[str]) -> pd.DataFrame:
    if nodes is None or nodes.empty:
        return pd.DataFrame()
    conditions = sorted(nodes["condition"].astype(str).unique(), key=str)
    if len(conditions) < 2:
        return pd.DataFrame()
    reference = str(reference if reference is not None else conditions[0])
    query = str(query if query is not None else conditions[1])
    ref_nodes = set(nodes.loc[nodes["condition"] == reference, "event_label"].astype(str))
    qry_nodes = set(nodes.loc[nodes["condition"] == query, "event_label"].astype(str))
    rows = []
    for node in sorted(qry_nodes - ref_nodes):
        rows.append({"reference": reference, "query": query, "rewiring_type": "node_gained", "source_event": "", "target_event": node})
    for node in sorted(ref_nodes - qry_nodes):
        rows.append({"reference": reference, "query": query, "rewiring_type": "node_lost", "source_event": "", "target_event": node})
    if edges is None or edges.empty:
        return pd.DataFrame(rows)
    ref_edges = {
        (row.source_event, row.target_event): row
        for row in edges[edges["condition"].astype(str) == reference].itertuples()
    }
    qry_edges = {
        (row.source_event, row.target_event): row
        for row in edges[edges["condition"].astype(str) == query].itertuples()
    }
    for edge in sorted(set(qry_edges) - set(ref_edges)):
        reverse = (edge[1], edge[0])
        rows.append(
            {
                "reference": reference,
                "query": query,
                "rewiring_type": "edge_reversed" if reverse in ref_edges else "edge_gained",
                "source_event": edge[0],
                "target_event": edge[1],
                "query_order_probability": qry_edges[edge].order_probability,
                "reference_order_probability": ref_edges[reverse].order_probability if reverse in ref_edges else np.nan,
            }
        )
    for edge in sorted(set(ref_edges) - set(qry_edges)):
        reverse = (edge[1], edge[0])
        if reverse in qry_edges:
            continue
        rows.append(
            {
                "reference": reference,
                "query": query,
                "rewiring_type": "edge_lost",
                "source_event": edge[0],
                "target_event": edge[1],
                "reference_order_probability": ref_edges[edge].order_probability,
                "query_order_probability": np.nan,
            }
        )
    for edge in sorted(set(ref_edges) & set(qry_edges)):
        rows.append(
            {
                "reference": reference,
                "query": query,
                "rewiring_type": "edge_shared",
                "source_event": edge[0],
                "target_event": edge[1],
                "reference_order_probability": ref_edges[edge].order_probability,
                "query_order_probability": qry_edges[edge].order_probability,
            }
        )
    return pd.DataFrame(rows)


def build_event_graph(
    events: pd.DataFrame,
    *,
    bootstrap_events: Optional[pd.DataFrame] = None,
    condition_col: str = "condition",
    time_col: str = "peak_time",
    event_id_col: Optional[str] = "event_id",
    label_col: Optional[str] = None,
    bootstrap_col: str = "boot_id",
    order_probability_threshold: float = 0.9,
    reference: Optional[str] = None,
    query: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    Build condition-specific event graphs and graph rewiring summaries.

    Nodes are events. Directed edges are stable event-order relations
    ``P(T_source < T_target) >= order_probability_threshold``.
    """
    if events is None or events.empty:
        raise ValueError("events must be a non-empty event table")
    label_col = label_col or _pick_column(events, "pathway", "Pathway", "module", "event_id")
    if label_col is None:
        raise ValueError("Could not infer event label column")
    nodes = _prepare_events(
        events,
        condition_col=condition_col,
        time_col=time_col,
        event_id_col=event_id_col,
        label_col=label_col,
    )
    prob = _probability_rows(
        nodes,
        bootstrap_events,
        condition_col=condition_col,
        time_col=time_col,
        event_id_col=event_id_col,
        label_col=label_col,
        bootstrap_col=bootstrap_col,
    )
    edges = _edge_table(prob, order_probability_threshold)
    rewiring = _rewiring_table(edges, nodes, reference, query)
    support = edges[
        [
            "condition",
            "source_event_id",
            "target_event_id",
            "source_event",
            "target_event",
            "order_probability",
            "n_bootstrap",
            "support_type",
        ]
    ].copy() if not edges.empty else pd.DataFrame()
    return {
        "event_order_probability_matrix": prob,
        "condition_event_graph_edges": edges,
        "event_graph_rewiring": rewiring,
        "event_graph_bootstrap_support": support,
    }


def write_event_graph(
    tables: dict[str, pd.DataFrame],
    outdir: str | Path,
    *,
    sep: str = "\t",
) -> dict[str, Path]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, filename in _GRAPH_OUTPUT_FILENAMES.items():
        table = tables.get(key, pd.DataFrame())
        path = out / filename
        table.to_csv(path, sep=sep, index=False, na_rep="NA")
        paths[key] = path
    return paths

