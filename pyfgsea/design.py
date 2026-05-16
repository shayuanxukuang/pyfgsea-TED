"""Experimental design diagnostics for trajectory pathway comparisons."""

from __future__ import annotations

from collections import Counter
from typing import Optional

import pandas as pd


def detect_experimental_design(
    adata,
    condition_key: Optional[str] = None,
    replicate_key: Optional[str] = None,
    sample_key: Optional[str] = None,
    *,
    min_replicates_per_condition: int = 3,
) -> pd.DataFrame:
    """Classify whether a trajectory comparison supports replicate-level inference.

    The detector is intentionally conservative. If a biological replicate appears
    in more than one condition, PyFgsea-TED treats it as a mixed/paired sample
    design that needs a dedicated sensitivity or paired analysis rather than a
    between-sample permutation test.
    """

    obs = adata.obs
    sample_key = sample_key or replicate_key
    row = {
        "condition_key": condition_key,
        "replicate_key": sample_key,
        "design": "unknown",
        "replicate_inference": "not_supported",
        "recommended_mode": "descriptive",
        "n_cells": int(adata.n_obs),
        "n_conditions": 0,
        "n_replicates": 0,
        "min_replicates_per_condition": 0,
        "mixed_samples": "",
    }

    if condition_key is None:
        row.update(
            {
                "design": "single_trajectory",
                "replicate_inference": "not_applicable",
                "recommended_mode": "trajectory_event_discovery",
            }
        )
        return pd.DataFrame([row])

    if condition_key not in obs:
        raise KeyError(f"condition_key '{condition_key}' is not present in adata.obs")

    cond = obs[condition_key].astype(str)
    row["n_conditions"] = int(cond.nunique(dropna=True))

    if sample_key is None:
        row.update(
            {
                "design": "cell_level_condition_descriptive",
                "recommended_mode": "descriptive_or_add_replicate_key",
            }
        )
        return pd.DataFrame([row])

    if sample_key not in obs:
        raise KeyError(f"replicate_key '{sample_key}' is not present in adata.obs")

    sample = obs[sample_key].astype(str)
    row["n_replicates"] = int(sample.nunique(dropna=True))

    sample_to_conditions = (
        pd.DataFrame({"sample": sample, "condition": cond})
        .dropna()
        .drop_duplicates()
        .groupby("sample")["condition"]
        .apply(lambda s: sorted(set(s)))
    )
    mixed_samples = [
        str(sample_id)
        for sample_id, conditions in sample_to_conditions.items()
        if len(conditions) > 1
    ]
    if mixed_samples:
        row.update(
            {
                "design": "mixed_sample_condition",
                "recommended_mode": "descriptive_or_within_sample_sensitivity",
                "mixed_samples": ",".join(mixed_samples),
            }
        )
        return pd.DataFrame([row])

    sample_condition = pd.DataFrame({"sample": sample, "condition": cond}).drop_duplicates()
    counts = Counter(sample_condition["condition"])
    min_reps = min(counts.values()) if counts else 0
    row["min_replicates_per_condition"] = int(min_reps)

    if row["n_replicates"] <= 1:
        row.update(
            {
                "design": "single_sample_descriptive",
                "recommended_mode": "descriptive_only",
            }
        )
    elif min_reps < min_replicates_per_condition:
        row.update(
            {
                "design": "between_sample_design",
                "replicate_inference": "limited_low_replicate_count",
                "recommended_mode": "replicate_aware_descriptive",
            }
        )
    else:
        row.update(
            {
                "design": "between_sample_design",
                "replicate_inference": "supported",
                "recommended_mode": "replicate_aware",
            }
        )

    return pd.DataFrame([row])
