"""Raw-count common-task benchmark for dynamic pathway methods.

This module deliberately generates counts and truth without calling TED.  It
provides a small executable development profile and the locked design used for
the planned nearest-method comparison.  Native external methods remain in
their own wrappers; the helpers here only create shared inputs and evaluate a
common output schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse


EVENT_MODES = ("activation", "suppression", "transient")
ARTIFACTS = ("none", "composition", "stress", "partial_batch_time")


@dataclass(frozen=True)
class RawCountDesign:
    n_blocks: int = 4
    n_cells: int = 2_000
    n_genes: int = 5_000
    n_pathways: int = 30
    pathway_size_min: int = 30
    pathway_size_max: int = 80
    true_per_mode: int = 3
    signal_strength: str = "low"
    coordinate_quality: str = "true"
    artifact: str = "none"
    seed: int = 1

    def validate(self) -> None:
        if self.n_blocks < 2:
            raise ValueError("n_blocks must be at least 2")
        if self.n_cells < self.n_blocks * 10:
            raise ValueError("n_cells is too small for the requested blocks")
        if self.n_pathways < self.true_per_mode * len(EVENT_MODES) + 1:
            raise ValueError("n_pathways must include true and null pathways")
        if self.pathway_size_max > self.n_genes:
            raise ValueError("pathway size cannot exceed n_genes")
        if self.signal_strength not in {"low", "high"}:
            raise ValueError("signal_strength must be low or high")
        if self.coordinate_quality not in {"true", "noisy"}:
            raise ValueError("coordinate_quality must be true or noisy")
        if self.artifact not in ARTIFACTS and self.artifact != "complete_batch_time":
            raise ValueError(f"unsupported artifact: {self.artifact}")


@dataclass
class RawCountDataset:
    counts: sparse.csr_matrix
    cells: pd.DataFrame
    pathways: dict[str, np.ndarray]
    truth: pd.DataFrame
    gene_metadata: pd.DataFrame
    scenario: dict[str, object]


def _pathways(design: RawCountDesign, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Generate prespecified sets with moderate, generator-level overlap."""
    sizes = rng.integers(
        design.pathway_size_min,
        design.pathway_size_max + 1,
        size=design.n_pathways,
    )
    pathways: dict[str, np.ndarray] = {}
    reusable_pool = rng.choice(design.n_genes, size=max(1, design.n_genes // 5), replace=False)
    for j, size in enumerate(sizes):
        # Up to 25% of a set is drawn from a common pool.  The realized overlap
        # is reported and is not forced to favor any analysis method.
        shared_n = int(round(size * rng.uniform(0.0, 0.25)))
        shared = rng.choice(reusable_pool, size=shared_n, replace=False)
        remaining_pool = np.setdiff1d(np.arange(design.n_genes), shared, assume_unique=False)
        unique = rng.choice(remaining_pool, size=int(size) - shared_n, replace=False)
        pathways[f"pathway_{j + 1:02d}"] = np.unique(np.concatenate([shared, unique])).astype(int)
    return pathways


def _mode_truth(design: RawCountDesign, pathways: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    names = list(pathways)
    cursor = 0
    centers = {"activation": 0.45, "suppression": 0.55, "transient": 0.50}
    for mode in EVENT_MODES:
        for _ in range(design.true_per_mode):
            name = names[cursor]
            rows.append(
                {
                    "pathway": name,
                    "is_dynamic": True,
                    "event_mode": mode,
                    "direction": "positive" if mode != "suppression" else "negative",
                    "true_center": centers[mode],
                    "true_width": 0.22 if mode == "transient" else np.nan,
                }
            )
            cursor += 1
    for name in names[cursor:]:
        rows.append(
            {
                "pathway": name,
                "is_dynamic": False,
                "event_mode": "null",
                "direction": "none",
                "true_center": np.nan,
                "true_width": np.nan,
            }
        )
    return pd.DataFrame(rows)


def _negative_binomial(
    mu: np.ndarray, theta: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Gamma-Poisson sampler with Var(Y)=mu+mu^2/theta."""
    shape = np.broadcast_to(theta[None, :], mu.shape)
    scale = mu / shape
    lam = rng.gamma(shape=shape, scale=scale)
    return rng.poisson(lam).astype(np.int32)


def simulate_raw_count_dataset(design: RawCountDesign) -> RawCountDataset:
    design.validate()
    rng = np.random.default_rng(design.seed)
    pathways = _pathways(design, rng)
    truth = _mode_truth(design, pathways)

    block = np.arange(design.n_cells) % design.n_blocks
    rng.shuffle(block)
    true_time = rng.uniform(0.0, 1.0, size=design.n_cells)
    library_size = rng.lognormal(mean=np.log(5_000.0), sigma=0.35, size=design.n_cells)
    library_offset = library_size / np.median(library_size)

    if design.artifact == "composition":
        state_prob = 1.0 / (1.0 + np.exp(-8.0 * (true_time - 0.5)))
    else:
        state_prob = np.full(design.n_cells, 0.5)
    state = (rng.random(design.n_cells) < state_prob).astype(int)

    if design.artifact == "complete_batch_time":
        batch = (true_time >= 0.5).astype(int)
    elif design.artifact == "partial_batch_time":
        batch_prob = 0.2 + 0.6 * true_time
        batch = (rng.random(design.n_cells) < batch_prob).astype(int)
    else:
        batch = rng.integers(0, 2, size=design.n_cells)

    if design.coordinate_quality == "noisy":
        ordered = np.clip(true_time + rng.normal(0.0, 0.18, size=design.n_cells), 0.0, 1.0)
    else:
        ordered = true_time.copy()

    alpha = rng.normal(-1.3, 0.55, size=design.n_genes)
    theta = rng.lognormal(mean=np.log(8.0), sigma=0.25, size=design.n_genes)
    block_effect = rng.normal(0.0, 0.18, size=(design.n_blocks, design.n_genes))
    eta = alpha[None, :] + block_effect[block, :]

    amplitude = 0.45 if design.signal_strength == "low" else 0.90
    for row in truth.itertuples(index=False):
        if not bool(row.is_dynamic):
            continue
        genes = pathways[str(row.pathway)]
        if row.event_mode == "activation":
            curve = 1.0 / (1.0 + np.exp(-12.0 * (true_time - 0.45)))
        elif row.event_mode == "suppression":
            curve = -(1.0 / (1.0 + np.exp(-12.0 * (true_time - 0.55))))
        else:
            curve = np.exp(-0.5 * ((true_time - 0.50) / 0.11) ** 2)
        gene_loading = rng.normal(1.0, 0.12, size=len(genes))
        eta[:, genes] += amplitude * curve[:, None] * gene_loading[None, :]

    null_names = truth.loc[~truth["is_dynamic"], "pathway"].tolist()
    artifact_targets = null_names[: min(3, len(null_names))]
    artifact_gene_mask = np.zeros(design.n_genes, dtype=bool)
    for name in artifact_targets:
        artifact_gene_mask[pathways[name]] = True
    artifact_genes = np.flatnonzero(artifact_gene_mask)
    if design.artifact == "composition":
        eta[:, artifact_genes] += 0.9 * state[:, None]
    elif design.artifact == "stress":
        stress_curve = np.exp(-0.5 * ((true_time - 0.62) / 0.09) ** 2)
        eta[:, artifact_genes] += 1.0 * stress_curve[:, None]
    elif design.artifact in {"partial_batch_time", "complete_batch_time"}:
        eta[:, artifact_genes] += 0.75 * batch[:, None]

    mu = np.exp(np.clip(eta, -8.0, 6.0)) * library_offset[:, None]
    counts = _negative_binomial(mu, theta, rng)
    counts_csr = sparse.csr_matrix(counts)
    observed_spearman = pd.Series(true_time).corr(pd.Series(ordered), method="spearman")

    cells = pd.DataFrame(
        {
            "cell_id": [f"cell_{i + 1:05d}" for i in range(design.n_cells)],
            "block": [f"block_{x + 1:02d}" for x in block],
            "true_time_private": true_time,
            "ordered_coordinate": ordered,
            "state": np.where(state == 1, "B", "A"),
            "batch": np.where(batch == 1, "batch_2", "batch_1"),
            "library_size_target": library_size,
        }
    )
    truth = truth.assign(
        artifact_scenario=design.artifact,
        artifact_target=truth["pathway"].isin(artifact_targets),
    )
    gene_metadata = pd.DataFrame(
        {
            "gene": [f"gene_{i + 1:05d}" for i in range(design.n_genes)],
            "baseline_log_mean": alpha,
            "dispersion_theta": theta,
            "artifact_program_member": artifact_gene_mask,
        }
    )
    scenario = {
        "seed": design.seed,
        "n_blocks": design.n_blocks,
        "n_cells": design.n_cells,
        "n_genes": design.n_genes,
        "n_pathways": design.n_pathways,
        "signal_strength": design.signal_strength,
        "coordinate_quality": design.coordinate_quality,
        "artifact": design.artifact,
        "observed_coordinate_spearman": float(observed_spearman),
        "complete_confounding": design.artifact == "complete_batch_time",
    }
    return RawCountDataset(counts_csr, cells, pathways, truth, gene_metadata, scenario)


def pathway_scores(counts: sparse.csr_matrix, pathways: dict[str, np.ndarray]) -> pd.DataFrame:
    """Library-normalized mean log-expression; shared input for simple adapters."""
    totals = np.asarray(counts.sum(axis=1)).ravel()
    scale = np.divide(10_000.0, totals, out=np.zeros_like(totals, dtype=float), where=totals > 0)
    normalized = counts.multiply(scale[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    out: dict[str, np.ndarray] = {}
    for name, genes in pathways.items():
        out[name] = np.asarray(normalized[:, genes].mean(axis=1)).ravel()
    return pd.DataFrame(out)


def score_then_smooth_common_task(
    counts: sparse.csr_matrix,
    cells: pd.DataFrame,
    pathways: dict[str, np.ndarray],
    *,
    n_bins: int = 20,
) -> pd.DataFrame:
    """Minimal shared-task baseline; not a substitute for any native package."""
    scores = pathway_scores(counts, pathways)
    coord = cells["ordered_coordinate"].to_numpy(dtype=float)
    order = np.argsort(coord)
    chunks = np.array_split(order, n_bins)
    bin_center = np.array([float(np.mean(coord[idx])) for idx in chunks])
    rows: list[dict[str, object]] = []
    for name in scores:
        values = scores[name].to_numpy(dtype=float)
        curve = np.array([float(np.mean(values[idx])) for idx in chunks])
        curve = pd.Series(curve).rolling(3, center=True, min_periods=1).mean().to_numpy()
        baseline = float(np.mean(np.r_[curve[:3], curve[-3:]]))
        peak_index = int(np.argmax(np.abs(curve - baseline)))
        endpoint_delta = float(np.mean(curve[-3:]) - np.mean(curve[:3]))
        interior_peak = 2 <= peak_index <= len(curve) - 3
        peak_delta = float(curve[peak_index] - baseline)
        transient_like = interior_peak and abs(peak_delta) > 1.35 * abs(endpoint_delta)
        if transient_like:
            event_mode = "transient"
            direction = "positive" if peak_delta >= 0 else "negative"
        else:
            event_mode = "activation" if endpoint_delta >= 0 else "suppression"
            direction = "positive" if endpoint_delta >= 0 else "negative"
        spread = float(np.std(values, ddof=1))
        statistic = float((np.max(curve) - np.min(curve)) / max(spread, 1e-8))
        rows.append(
            {
                "pathway": name,
                "method": "score_then_smooth",
                "ranking_score": statistic,
                "event_detected": statistic >= 0.35,
                "direction": direction,
                "event_mode": event_mode,
                "event_center": float(bin_center[peak_index]),
                "event_width": np.nan,
                "p_value": np.nan,
                "q_value": np.nan,
                "formal_p_value_available": False,
                "status": "ok",
            }
        )
    return pd.DataFrame(rows)


def evaluate_common_task(predictions: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Evaluate only fields shared by the compared methods."""
    from sklearn.metrics import average_precision_score, ndcg_score

    merged = truth.merge(predictions, on="pathway", how="left", validate="one_to_one")
    ok = merged["status"].eq("ok") & merged["ranking_score"].notna()
    if not ok.any():
        return pd.DataFrame([{"method": predictions["method"].iloc[0], "status": "no_evaluable_outputs"}])
    y = merged.loc[ok, "is_dynamic"].astype(int).to_numpy()
    score = merged.loc[ok, "ranking_score"].astype(float).to_numpy()
    k = int(max(1, merged["is_dynamic"].sum()))
    ranked = merged.loc[ok].sort_values("ranking_score", ascending=False)
    top = ranked.head(k)
    artifact_targets = merged["artifact_target"] & ~merged["is_dynamic"]
    matched_top_k_artifact_false_promotion_rate = (
        float(merged.loc[artifact_targets, "pathway"].isin(set(top["pathway"])).mean())
        if artifact_targets.any()
        else np.nan
    )
    direction_rows = merged[merged["is_dynamic"] & merged["direction_y"].notna()]
    transient_rows = merged[(merged["event_mode_x"] == "transient") & merged["event_center"].notna()]
    return pd.DataFrame(
        [
            {
                "method": str(predictions["method"].iloc[0]),
                "status": "ok",
                "pathway_level_auprc": float(average_precision_score(y, score)),
                "top_k": k,
                "top_k_precision": float(top["is_dynamic"].mean()),
                "top_k_recall": float(top["is_dynamic"].sum() / max(1, merged["is_dynamic"].sum())),
                "ndcg": float(ndcg_score(y[None, :], score[None, :])),
                "direction_accuracy": float(
                    (direction_rows["direction_x"] == direction_rows["direction_y"]).mean()
                )
                if len(direction_rows)
                else np.nan,
                "transient_center_mae": float(
                    np.mean(np.abs(transient_rows["event_center"] - transient_rows["true_center"]))
                )
                if len(transient_rows)
                else np.nan,
                "artifact_only_false_call_rate": float(
                    merged.loc[merged["artifact_target"] & ~merged["is_dynamic"], "event_detected"].mean()
                )
                if (merged["artifact_target"] & ~merged["is_dynamic"]).any()
                else np.nan,
                "matched_top_k_artifact_false_promotion_rate": matched_top_k_artifact_false_promotion_rate,
                "formal_fdp_available": bool(merged["formal_p_value_available"].fillna(False).any()),
            }
        ]
    )


def realized_pairwise_overlap(pathways: dict[str, Iterable[int]]) -> float:
    sets = [set(map(int, genes)) for genes in pathways.values()]
    maximum = 0.0
    for i, left in enumerate(sets):
        for right in sets[i + 1 :]:
            union = left | right
            maximum = max(maximum, len(left & right) / len(union) if union else 0.0)
    return float(maximum)
