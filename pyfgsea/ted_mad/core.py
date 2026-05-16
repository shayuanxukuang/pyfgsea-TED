"""Core TED-MAD/ARD implementation.

TED-MAD/ARD is intentionally table driven. The model is not meant to be a
black-box perturbation predictor; it turns an evidence ledger into a mechanism
posterior, a claim ceiling, and a ranked list of falsifiable rescue experiments.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import textwrap
import html
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .schema import (
    validate_evidence,
    validate_experiments,
    validate_hypotheses,
    validate_observed_rescue,
)

try:  # pragma: no cover - exercised in normal environments
    import yaml
except Exception:  # pragma: no cover
    yaml = None


TED_MAD_VERSION = "0.1.0"
DEFAULT_RANDOM_SEED = 20260513

DEFAULT_MODEL_CONFIG = {
    "base_likelihood_ratio": 2.0,
    "max_item_abs_log_lr": 2.0,
    "max_family_abs_log_lr": 2.5,
    "dependency_aggregation": "mean",
    "support_threshold_log_lr": 0.05,
}

CLAIM_LEVELS = [
    {
        "level": 1.0,
        "label": "L1 descriptive association",
        "claim": "descriptive association",
        "required": ["any_supportive_evidence"],
        "missing": ["at least one mechanism-relevant evidence family"],
    },
    {
        "level": 2.0,
        "label": "L2 robust erythroid event",
        "claim": "robust erythroid event",
        "required": ["family_block_robustness", "negative_controls"],
        "missing": ["family/block robustness", "negative mediator or shuffled controls"],
    },
    {
        "level": 3.0,
        "label": "L3 adjusted causal-compatible event",
        "claim": "adjusted causal-compatible event",
        "required_any_count": (
            ["mediation", "counterfactual_ot", "state_adjustment", "proliferation_adjustment"],
            2,
        ),
        "missing": [
            "two independent adjusted causal-compatible evidence families",
        ],
    },
    {
        "level": 3.5,
        "label": "L3.5 mechanism-prioritized event",
        "claim": "computationally adjudicated, rescue-ready mechanism model",
        "required_any_count": (
            ["timing", "dynamic_precedence", "rescue_prediction", "external_support"],
            2,
        ),
        "missing": [
            "timing/dynamic precedence, external perturbation support, or rescue prediction evidence",
        ],
    },
    {
        "level": 4.0,
        "label": "L4 rescue-supported mechanism",
        "claim": "rescue-supported mechanism",
        "required": ["direct_rescue"],
        "missing": ["pre-registered matched rescue result"],
    },
    {
        "level": 5.0,
        "label": "L5 orthogonal perturbation-validated mechanism",
        "claim": "orthogonal perturbation-validated mechanism",
        "required": ["orthogonal_perturbation", "independent_replication"],
        "missing": ["orthogonal perturbation or assay and independent replication"],
    },
]

CATEGORY_ALIASES = {
    "family_block_robustness": [
        "e1",
        "family level block robustness",
        "family-level block robustness",
        "family block robustness",
        "block robustness",
        "family_block",
    ],
    "negative_controls": [
        "e5",
        "negative mediator controls",
        "negative mediator",
        "negative control",
        "shuffled control",
    ],
    "mediation": [
        "e2",
        "mediation",
        "proliferation adjusted mediation",
        "proliferation-adjusted mediation",
    ],
    "counterfactual_ot": [
        "e3",
        "counterfactual ot",
        "counterfactual optimal transport",
        "ot event effect",
    ],
    "state_adjustment": [
        "state matched",
        "state-matched",
        "composition adjusted",
        "composition artifact control",
    ],
    "proliferation_adjustment": [
        "proliferation adjusted",
        "proliferation-adjusted",
        "cell cycle adjusted",
        "cell-cycle adjusted",
    ],
    "timing": ["e4", "day stratified timing", "day-stratified timing", "timing"],
    "dynamic_precedence": ["e6", "dynamic precedence", "precedence"],
    "rescue_prediction": ["e7", "rescue prediction", "rescue prediction table"],
    "external_support": [
        "e8",
        "e9",
        "external gata1 kd",
        "external gata1 knockdown",
        "external t21 multiome",
        "external support",
    ],
    "direct_rescue": [
        "direct rescue",
        "wet rescue",
        "matched rescue",
        "pre registered rescue",
        "pre-registered rescue",
        "rescue-supported",
    ],
    "orthogonal_perturbation": [
        "orthogonal perturbation",
        "loss of function",
        "loss-of-function",
        "gain of function",
        "cut&tag",
        "cut and tag",
        "atac validation",
    ],
    "independent_replication": [
        "independent replication",
        "multi batch",
        "multi-batch",
        "cross dataset",
        "cross-dataset",
    ],
}

DEFAULT_HYPOTHESES = [
    {
        "hypothesis_id": "H0",
        "label": "noise/batch/family artifact",
        "prior": 1 / 6,
    },
    {
        "hypothesis_id": "H1",
        "label": "state/composition artifact",
        "prior": 1 / 6,
    },
    {
        "hypothesis_id": "H2",
        "label": "proliferation-confounded mechanism",
        "prior": 1 / 6,
    },
    {
        "hypothesis_id": "H3",
        "label": "GATA1-regulatory mechanism",
        "prior": 1 / 6,
    },
    {
        "hypothesis_id": "H4",
        "label": "downstream heme/maturation mechanism",
        "prior": 1 / 6,
    },
    {
        "hypothesis_id": "H5",
        "label": "T21-specific chromatin/GATA1 interaction",
        "prior": 1 / 6,
    },
]


def load_ted_mad_yaml(path: str | Path) -> Any:
    """Load a YAML or JSON TED-MAD input file."""

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json" or yaml is None:
        return json.loads(text)
    return yaml.safe_load(text)


def _write_yaml(path: Path, payload: Any) -> None:
    if yaml is None:
        path.with_suffix(".json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        return
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pyfgsea_version() -> str:
    try:
        from importlib.metadata import version

        return version("pyfgsea")
    except Exception:
        try:
            from pyfgsea import __version__  # type: ignore

            return str(__version__)
        except Exception:
            return "unknown"


def _git_commit_hash(cwd: str | Path | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd or Path.cwd()),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def build_provenance(
    input_files: Mapping[str, str | Path] | None = None,
    *,
    random_seed: int | None = DEFAULT_RANDOM_SEED,
    run_timestamp: str | None = None,
    git_cwd: str | Path | None = None,
) -> dict[str, Any]:
    """Build a reproducibility metadata block for TED-MAD outputs."""

    files = {key: str(Path(value)) for key, value in (input_files or {}).items() if value}
    checksums = {key: _sha256_file(path) for key, path in files.items() if Path(path).exists()}
    return {
        "ted_mad_version": TED_MAD_VERSION,
        "pyfgsea_version": _pyfgsea_version(),
        "git_commit": _git_commit_hash(git_cwd),
        "input_sha256": checksums,
        "run_timestamp": run_timestamp or datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
        "input_files": files,
    }


def merge_provenance(*blocks: Mapping[str, Any] | None) -> dict[str, Any]:
    """Merge provenance blocks, preserving all input-file checksums."""

    merged: dict[str, Any] = {}
    input_files: dict[str, Any] = {}
    input_sha256: dict[str, Any] = {}
    for block in blocks:
        if not block:
            continue
        merged.update({key: value for key, value in block.items() if key not in {"input_files", "input_sha256"}})
        input_files.update(block.get("input_files", {}) or {})
        input_sha256.update(block.get("input_sha256", {}) or {})
    if input_files:
        merged["input_files"] = input_files
    if input_sha256:
        merged["input_sha256"] = input_sha256
    return merged


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [str(key).strip() for key in value if str(key).strip()]
    if isinstance(value, str):
        if not value.strip():
            return []
        return [part.strip() for part in re.split(r"[,;]", value) if part.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _as_weight_map(value: Any) -> dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {
            str(key).strip(): _clip(_to_float(weight, 1.0), 0.0, 1.0)
            for key, weight in value.items()
            if str(key).strip()
        }
    return {hyp: 1.0 for hyp in _as_list(value)}


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _canonical_text(value: Any) -> str:
    value = str(value or "").lower()
    value = re.sub(r"[_/\-]+", " ", value)
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _parse_hypotheses(raw: Mapping[str, Any] | Sequence[Any] | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if raw is None:
        return [dict(item) for item in DEFAULT_HYPOTHESES], dict(DEFAULT_MODEL_CONFIG)

    model_config = dict(DEFAULT_MODEL_CONFIG)
    if isinstance(raw, Mapping):
        model_config.update(raw.get("model", {}) or {})
        entries = raw.get("hypotheses", DEFAULT_HYPOTHESES)
    else:
        entries = raw

    hypotheses: list[dict[str, Any]] = []
    if isinstance(entries, Mapping):
        for key, spec in entries.items():
            spec = dict(spec or {})
            hypotheses.append(
                {
                    "hypothesis_id": str(spec.get("hypothesis_id") or spec.get("id") or key),
                    "label": str(spec.get("label") or spec.get("name") or key),
                    "description": str(spec.get("description") or ""),
                    "prior": _to_float(spec.get("prior"), 0.0),
                    "expected_evidence": dict(spec.get("expected_evidence") or {}),
                    "falsifiers": _as_list(spec.get("falsifiers")),
                }
            )
    else:
        for i, item in enumerate(entries or []):
            spec = dict(item or {})
            hypotheses.append(
                {
                    "hypothesis_id": str(
                        spec.get("hypothesis_id") or spec.get("id") or spec.get("name") or f"H{i}"
                    ),
                    "label": str(spec.get("label") or spec.get("name") or f"H{i}"),
                    "description": str(spec.get("description") or ""),
                    "prior": _to_float(spec.get("prior"), 0.0),
                    "expected_evidence": dict(spec.get("expected_evidence") or {}),
                    "falsifiers": _as_list(spec.get("falsifiers")),
                }
            )

    if not hypotheses:
        raise ValueError("hypotheses input must define at least one hypothesis")

    priors = np.asarray([max(_to_float(h["prior"], 0.0), 0.0) for h in hypotheses], dtype=float)
    if priors.sum() <= 0:
        priors = np.repeat(1.0 / len(hypotheses), len(hypotheses))
    else:
        priors = priors / priors.sum()
    for hyp, prior in zip(hypotheses, priors):
        hyp["prior"] = float(prior)

    return hypotheses, model_config


def _parse_evidence(raw: Mapping[str, Any] | Sequence[Any], event: str | None = None) -> list[dict[str, Any]]:
    entries = raw.get("evidence", raw.get("ledger", [])) if isinstance(raw, Mapping) else raw
    if entries is None:
        entries = []

    evidence: list[dict[str, Any]] = []
    for i, item in enumerate(entries):
        row = dict(item or {})
        target_event = str(row.get("target_event") or row.get("event_id") or "")
        if event and target_event and target_event != event:
            continue

        evidence_id = str(row.get("evidence_id") or row.get("id") or f"E{i + 1}")
        evidence_family = str(
            row.get("evidence_family") or row.get("family") or row.get("evidence_type") or evidence_id
        )
        supports = (
            row.get("which_hypotheses_it_supports")
            or row.get("supports_hypotheses")
            or row.get("supports")
        )
        weakens = (
            row.get("which_hypotheses_it_weakens")
            or row.get("weakens_hypotheses")
            or row.get("weakens")
        )
        assumptions = row.get("assumptions", row.get("assumption", []))
        failure_modes = row.get("failure_modes", row.get("failure_mode", []))
        row.update(
            {
                "evidence_id": evidence_id,
                "evidence_family": evidence_family,
                "target_event": target_event or str(event or row.get("target_event") or "all"),
                "which_hypotheses_it_supports": _as_list(supports),
                "which_hypotheses_it_weakens": _as_list(weakens),
                "support_weights": _as_weight_map(supports),
                "weaken_weights": _as_weight_map(weakens),
                "dependency_group": str(row.get("dependency_group") or evidence_id),
                "claim_tags": _as_list(row.get("claim_tags")),
                "assumptions": _as_list(assumptions),
                "failure_modes": _as_list(failure_modes),
            }
        )
        evidence.append(row)

    if not evidence:
        raise ValueError("evidence ledger is empty after optional event filtering")
    return evidence


def _evidence_strength(row: Mapping[str, Any]) -> float:
    if row.get("strength") is not None:
        return _clip(_to_float(row.get("strength"), 1.0), 0.0, 3.0)

    effect = abs(_to_float(row.get("effect_size"), 0.0))
    uncertainty = abs(_to_float(row.get("standard_error", row.get("uncertainty")), 0.0))
    if uncertainty > 0:
        return _clip(effect / uncertainty, 0.0, 3.0)
    if effect > 0:
        return _clip(effect, 0.0, 3.0)
    return 1.0


def _mapping_for_keys(row: Mapping[str, Any], keys: Sequence[str]) -> Mapping[str, Any] | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, Mapping):
            return value
    return None


def _item_log_lrs(
    row: Mapping[str, Any],
    hypothesis_ids: Sequence[str],
    config: Mapping[str, Any],
) -> dict[str, float]:
    explicit_log = _mapping_for_keys(row, ["log_likelihood_ratios", "log_lr", "log_lrs"])
    if explicit_log is not None:
        return {
            hyp: _clip(
                _to_float(explicit_log.get(hyp), 0.0),
                -_to_float(config.get("max_item_abs_log_lr"), 2.0),
                _to_float(config.get("max_item_abs_log_lr"), 2.0),
            )
            for hyp in hypothesis_ids
        }

    explicit_lr = _mapping_for_keys(row, ["likelihood_ratios", "lr", "lrs"])
    if explicit_lr is not None:
        out = {}
        for hyp in hypothesis_ids:
            lr = max(_to_float(explicit_lr.get(hyp), 1.0), 1e-12)
            out[hyp] = _clip(
                math.log(lr),
                -_to_float(config.get("max_item_abs_log_lr"), 2.0),
                _to_float(config.get("max_item_abs_log_lr"), 2.0),
            )
        return out

    strength = _evidence_strength(row)
    weight = _clip(_to_float(row.get("weight"), 1.0), 0.0, 5.0)
    reliability = _clip(_to_float(row.get("reliability"), 1.0), 0.0, 1.0)
    base_lr = max(_to_float(row.get("base_likelihood_ratio"), _to_float(config.get("base_likelihood_ratio"), 2.0)), 1.0001)
    log_step = math.log(base_lr) * strength * weight * reliability
    log_step = _clip(
        log_step,
        -_to_float(config.get("max_item_abs_log_lr"), 2.0),
        _to_float(config.get("max_item_abs_log_lr"), 2.0),
    )

    support_weights = {
        hyp: _clip(_to_float(value, 1.0), 0.0, 1.0)
        for hyp, value in dict(row.get("support_weights", {})).items()
    }
    weaken_weights = {
        hyp: _clip(_to_float(value, 1.0), 0.0, 1.0)
        for hyp, value in dict(row.get("weaken_weights", {})).items()
    }
    for hyp in row.get("which_hypotheses_it_supports", []):
        support_weights.setdefault(hyp, 1.0)
    for hyp in row.get("which_hypotheses_it_weakens", []):
        weaken_weights.setdefault(hyp, 1.0)
    out = {hyp: 0.0 for hyp in hypothesis_ids}
    for hyp in hypothesis_ids:
        out[hyp] += log_step * support_weights.get(hyp, 0.0)
        out[hyp] -= log_step * weaken_weights.get(hyp, 0.0)
    return out


def _aggregate_evidence(
    evidence: Sequence[Mapping[str, Any]],
    hypotheses: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hypothesis_ids = [str(h["hypothesis_id"]) for h in hypotheses]
    item_rows: list[dict[str, Any]] = []
    for row in evidence:
        log_lrs = _item_log_lrs(row, hypothesis_ids, config)
        for hyp, log_lr in log_lrs.items():
            item_rows.append(
                {
                    "evidence_id": row["evidence_id"],
                    "evidence_family": row["evidence_family"],
                    "target_event": row["target_event"],
                    "dependency_group": row["dependency_group"],
                    "hypothesis": hyp,
                    "log_likelihood_ratio": float(log_lr),
                    "likelihood_ratio": float(math.exp(log_lr)),
                    "strength": float(_evidence_strength(row)),
                    "data_source": row.get("data_source", ""),
                    "assumption": ";".join(row.get("assumptions", _as_list(row.get("assumption")))),
                    "failure_mode": ";".join(
                        row.get("failure_modes", _as_list(row.get("failure_mode")))
                    ),
                    "claim_tags": ";".join(row.get("claim_tags", [])),
                }
            )

    item_df = pd.DataFrame(item_rows)
    if item_df.empty:
        raise ValueError("no evidence contribution rows could be computed")

    mode = str(config.get("dependency_aggregation", "mean"))
    grouped = item_df.groupby(
        ["evidence_family", "dependency_group", "hypothesis"], as_index=False
    )["log_likelihood_ratio"]
    if mode == "max_abs":
        dep_df = grouped.agg(lambda s: s.iloc[int(np.argmax(np.abs(s.to_numpy())))])
    elif mode == "sum":
        dep_df = grouped.sum()
    else:
        dep_df = grouped.mean()

    family = dep_df.groupby(["evidence_family", "hypothesis"], as_index=False)[
        "log_likelihood_ratio"
    ].sum()
    cap = _to_float(config.get("max_family_abs_log_lr"), 2.5)
    family["log_likelihood_ratio"] = family["log_likelihood_ratio"].map(
        lambda x: _clip(float(x), -cap, cap)
    )
    family["likelihood_ratio"] = np.exp(family["log_likelihood_ratio"])

    support_threshold = _to_float(config.get("support_threshold_log_lr"), 0.05)
    family["supports_hypothesis"] = family["log_likelihood_ratio"] > support_threshold
    family["weakens_hypothesis"] = family["log_likelihood_ratio"] < -support_threshold
    return item_df, family


def _posterior_from_contributions(
    hypotheses: Sequence[Mapping[str, Any]],
    family_df: pd.DataFrame,
    exclude_family: str | None = None,
) -> pd.DataFrame:
    rows = []
    use_family = family_df
    if exclude_family is not None:
        use_family = family_df[family_df["evidence_family"] != exclude_family]

    contribution = (
        use_family.groupby("hypothesis")["log_likelihood_ratio"].sum().to_dict()
        if not use_family.empty
        else {}
    )
    log_values = []
    for hyp in hypotheses:
        prior = max(_to_float(hyp.get("prior"), 0.0), 1e-12)
        log_values.append(math.log(prior) + _to_float(contribution.get(hyp["hypothesis_id"]), 0.0))
    log_values_np = np.asarray(log_values, dtype=float)
    log_values_np = log_values_np - log_values_np.max()
    probs = np.exp(log_values_np)
    probs = probs / probs.sum()

    for hyp, posterior in zip(hypotheses, probs):
        rows.append(
            {
                "hypothesis": hyp["hypothesis_id"],
                "label": hyp.get("label", hyp["hypothesis_id"]),
                "description": hyp.get("description", ""),
                "prior": float(hyp["prior"]),
                "log_likelihood_ratio_total": float(
                    contribution.get(hyp["hypothesis_id"], 0.0)
                ),
                "posterior": float(posterior),
            }
        )
    return pd.DataFrame(rows).sort_values("posterior", ascending=False).reset_index(drop=True)


def _leave_one_family_out(
    hypotheses: Sequence[Mapping[str, Any]], family_df: pd.DataFrame
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for family in sorted(family_df["evidence_family"].unique()):
        post = _posterior_from_contributions(hypotheses, family_df, exclude_family=family)
        post.insert(0, "hidden_evidence_family", family)
        rows.append(post)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _leave_one_family_out_summary(
    evidence: Sequence[Mapping[str, Any]],
    hypotheses: Sequence[Mapping[str, Any]],
    family_df: pd.DataFrame,
    baseline_posterior: pd.DataFrame,
    baseline_claim: Mapping[str, Any],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Summarize how each evidence-family ablation changes posterior and claim ceiling."""

    baseline_leading = str(baseline_posterior.iloc[0]["hypothesis"])
    baseline_level = _to_float(baseline_claim.get("current_level_numeric"), 0.0)
    family_assumptions: dict[str, list[str]] = defaultdict(list)
    family_failure_modes: dict[str, list[str]] = defaultdict(list)
    for row in evidence:
        family = str(row["evidence_family"])
        family_assumptions[family].extend(_as_list(row.get("assumptions")))
        family_failure_modes[family].extend(_as_list(row.get("failure_modes")))
    rows = []
    for family in sorted(family_df["evidence_family"].unique()):
        ablated_family_df = family_df[family_df["evidence_family"] != family]
        posterior = _posterior_from_contributions(hypotheses, family_df, exclude_family=family)
        claim = _claim_ceiling(evidence, ablated_family_df, posterior, config)
        baseline_row = posterior[posterior["hypothesis"] == baseline_leading]
        baseline_rank = (
            int(baseline_row.index[0]) + 1 if not baseline_row.empty else len(posterior) + 1
        )
        baseline_posterior_after = (
            float(baseline_row.iloc[0]["posterior"]) if not baseline_row.empty else 0.0
        )
        rows.append(
            {
                "family_removed": family,
                "leading_hypothesis": str(posterior.iloc[0]["hypothesis"]),
                "leading_posterior": float(posterior.iloc[0]["posterior"]),
                "baseline_leading_hypothesis": baseline_leading,
                "baseline_leading_posterior_after_removal": baseline_posterior_after,
                "baseline_leading_rank_after_removal": baseline_rank,
                "leading_changed": str(posterior.iloc[0]["hypothesis"]) != baseline_leading,
                "claim_ceiling": claim["current_level"],
                "claim_level_numeric": float(claim["current_level_numeric"]),
                "claim_level_delta": float(claim["current_level_numeric"]) - baseline_level,
                "posterior_drop_for_baseline_leading": float(
                    baseline_posterior.iloc[0]["posterior"]
                )
                - baseline_posterior_after,
                "assumptions": ";".join(dict.fromkeys(family_assumptions.get(family, []))),
                "failure_modes": ";".join(dict.fromkeys(family_failure_modes.get(family, []))),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["posterior_drop_for_baseline_leading", "family_removed"],
        ascending=[False, True],
    ).reset_index(drop=True)


def _categories_for_family(family: str, claim_tags: Sequence[str] | None = None) -> set[str]:
    text = _canonical_text(family)
    tags = {_canonical_text(tag) for tag in (claim_tags or [])}
    categories: set[str] = set()
    for category, aliases in CATEGORY_ALIASES.items():
        for alias in aliases:
            alias_text = _canonical_text(alias)
            if alias_text and (alias_text in text or alias_text in tags):
                categories.add(category)
                break
    return categories


def _present_claim_categories(
    evidence: Sequence[Mapping[str, Any]],
    family_df: pd.DataFrame,
    leading_hypothesis: str,
    support_threshold: float,
) -> set[str]:
    positive_families = set(
        family_df[
            (family_df["hypothesis"] == leading_hypothesis)
            & (family_df["log_likelihood_ratio"] > support_threshold)
        ]["evidence_family"]
    )
    by_family_tags: dict[str, set[str]] = defaultdict(set)
    for row in evidence:
        by_family_tags[str(row["evidence_family"])].update(row.get("claim_tags", []))

    categories: set[str] = set()
    if positive_families:
        categories.add("any_supportive_evidence")
    for family in positive_families:
        categories.update(_categories_for_family(family, sorted(by_family_tags[family])))
    return categories


def _claim_ceiling(
    evidence: Sequence[Mapping[str, Any]],
    family_df: pd.DataFrame,
    posterior_df: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    leading = posterior_df.iloc[0]
    support_threshold = _to_float(config.get("support_threshold_log_lr"), 0.05)
    present = _present_claim_categories(
        evidence, family_df, str(leading["hypothesis"]), support_threshold
    )

    current = CLAIM_LEVELS[0]
    passed_level = 0.0
    missing_for_next: list[str] = []
    for level in CLAIM_LEVELS:
        required = set(level.get("required", []))
        any_count_spec = level.get("required_any_count")
        ok = required.issubset(present)
        if any_count_spec is not None:
            choices, count = any_count_spec
            ok = ok and len(set(choices).intersection(present)) >= int(count)
        if ok:
            current = level
            passed_level = float(level["level"])
        else:
            missing_for_next = list(level.get("missing", []))
            break

    if passed_level >= CLAIM_LEVELS[-1]["level"]:
        missing_for_next = []

    next_level = next((lvl for lvl in CLAIM_LEVELS if float(lvl["level"]) > passed_level), None)
    if next_level is not None and not missing_for_next:
        missing_for_next = list(next_level.get("missing", []))

    reason = (
        "All defined evidence categories are present."
        if not missing_for_next
        else "Claim cannot exceed this level because the next level still needs: "
        + "; ".join(missing_for_next)
    )
    return {
        "leading_hypothesis": str(leading["hypothesis"]),
        "leading_hypothesis_label": str(leading["label"]),
        "leading_posterior": float(leading["posterior"]),
        "current_level_numeric": passed_level,
        "current_level": current["label"],
        "current_claim": current["claim"],
        "present_evidence_categories": sorted(present),
        "missing_evidence_for_next_level": missing_for_next,
        "reason_claim_cannot_exceed_current_level": reason,
    }


def _copy_hypotheses_with_priors(
    hypotheses: Sequence[Mapping[str, Any]], priors: Sequence[float]
) -> list[dict[str, Any]]:
    out = [dict(hyp) for hyp in hypotheses]
    priors_np = np.asarray(priors, dtype=float)
    priors_np = np.maximum(priors_np, 1e-12)
    priors_np = priors_np / priors_np.sum()
    for hyp, prior in zip(out, priors_np):
        hyp["prior"] = float(prior)
    return out


def _copy_evidence_with_weights(
    evidence: Sequence[Mapping[str, Any]],
    *,
    weight_factors: Mapping[str, float] | None = None,
    family_counts: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    out = []
    weight_factors = weight_factors or {}
    family_counts = family_counts or {}
    for row in evidence:
        family = str(row["evidence_family"])
        count = int(family_counts.get(family, 1))
        if count <= 0:
            continue
        copied = dict(row)
        copied["weight"] = (
            _to_float(copied.get("weight"), 1.0)
            * float(weight_factors.get(str(row["evidence_id"]), 1.0))
            * count
        )
        out.append(copied)
    return out


def _sensitivity_iteration(
    evidence: Sequence[Mapping[str, Any]],
    hypotheses: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    iteration: int,
    scenario: str,
    family_counts: Mapping[str, int] | None = None,
    weight_factors: Mapping[str, float] | None = None,
    priors: Sequence[float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    iter_hypotheses = (
        _copy_hypotheses_with_priors(hypotheses, priors)
        if priors is not None
        else [dict(hyp) for hyp in hypotheses]
    )
    iter_evidence = _copy_evidence_with_weights(
        evidence,
        weight_factors=weight_factors,
        family_counts=family_counts,
    )
    _, iter_family_df = _aggregate_evidence(iter_evidence, iter_hypotheses, config)
    posterior = _posterior_from_contributions(iter_hypotheses, iter_family_df)
    claim = _claim_ceiling(iter_evidence, iter_family_df, posterior, config)
    rows = []
    for row in posterior.to_dict(orient="records"):
        rows.append(
            {
                "iteration": iteration,
                "scenario": scenario,
                "hypothesis": row["hypothesis"],
                "posterior": float(row["posterior"]),
                "prior": float(row["prior"]),
                "leading_hypothesis": str(posterior.iloc[0]["hypothesis"]),
                "is_leading": str(row["hypothesis"]) == str(posterior.iloc[0]["hypothesis"]),
                "claim_ceiling": claim["current_level"],
                "claim_level_numeric": float(claim["current_level_numeric"]),
            }
        )
    return rows, claim


def _posterior_sensitivity_analysis(
    evidence: Sequence[Mapping[str, Any]],
    hypotheses: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    baseline_posterior: pd.DataFrame,
    baseline_claim: Mapping[str, Any],
    prior_grid: bool = False,
    weight_jitter: float = 0.0,
    n_bootstrap: int = 0,
    random_seed: int | None = DEFAULT_RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run prior/weight/bootstrap perturbations and summarize posterior stability."""

    rows: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    iteration = 0
    base_rows, base_claim = _sensitivity_iteration(
        evidence,
        hypotheses,
        config,
        iteration=iteration,
        scenario="baseline",
    )
    rows.extend(base_rows)
    claims.append({"iteration": iteration, "claim_level_numeric": base_claim["current_level_numeric"]})

    base_priors = np.asarray([float(hyp["prior"]) for hyp in hypotheses], dtype=float)
    if prior_grid:
        for hyp_idx, hyp in enumerate(hypotheses):
            for factor in (0.5, 2.0):
                iteration += 1
                priors = base_priors.copy()
                priors[hyp_idx] *= factor
                iter_rows, claim = _sensitivity_iteration(
                    evidence,
                    hypotheses,
                    config,
                    iteration=iteration,
                    scenario=f"prior_grid:{hyp['hypothesis_id']}x{factor:g}",
                    priors=priors,
                )
                rows.extend(iter_rows)
                claims.append(
                    {"iteration": iteration, "claim_level_numeric": claim["current_level_numeric"]}
                )

    rng = np.random.default_rng(random_seed)
    families = sorted({str(row["evidence_family"]) for row in evidence})
    n_bootstrap = max(int(n_bootstrap), 0)
    jitter = max(float(weight_jitter), 0.0)
    for _ in range(n_bootstrap):
        iteration += 1
        sampled = rng.choice(families, size=len(families), replace=True)
        family_counts = {family: int(np.sum(sampled == family)) for family in families}
        weight_factors = None
        if jitter > 0:
            low = max(0.0, 1.0 - jitter)
            high = 1.0 + jitter
            weight_factors = {
                str(row["evidence_id"]): float(rng.uniform(low, high)) for row in evidence
            }
        iter_rows, claim = _sensitivity_iteration(
            evidence,
            hypotheses,
            config,
            iteration=iteration,
            scenario="bootstrap_weight_jitter" if jitter > 0 else "bootstrap",
            family_counts=family_counts,
            weight_factors=weight_factors,
        )
        rows.extend(iter_rows)
        claims.append({"iteration": iteration, "claim_level_numeric": claim["current_level_numeric"]})

    sensitivity_df = pd.DataFrame(rows)
    interval_rows = []
    for hyp, group in sensitivity_df.groupby("hypothesis", sort=False):
        interval_rows.append(
            {
                "hypothesis": hyp,
                "posterior_median": float(group["posterior"].median()),
                "posterior_q025": float(group["posterior"].quantile(0.025)),
                "posterior_q975": float(group["posterior"].quantile(0.975)),
                "posterior_min": float(group["posterior"].min()),
                "posterior_max": float(group["posterior"].max()),
                "leading_frequency": float(group["is_leading"].mean()),
            }
        )
    interval_df = pd.DataFrame(interval_rows).sort_values(
        "posterior_median", ascending=False
    ).reset_index(drop=True)

    dominant = (
        sensitivity_df[["iteration", "leading_hypothesis"]]
        .drop_duplicates()
        .groupby("leading_hypothesis")
        .size()
        .reset_index(name="n_iterations_leading")
    )
    total_iterations = sensitivity_df["iteration"].nunique()
    dominant["dominance_frequency"] = dominant["n_iterations_leading"] / max(total_iterations, 1)
    dominant = dominant.sort_values("dominance_frequency", ascending=False).reset_index(drop=True)

    leading = str(baseline_posterior.iloc[0]["hypothesis"])
    leading_interval = interval_df[interval_df["hypothesis"] == leading]
    claim_levels = pd.DataFrame(claims)["claim_level_numeric"] if claims else pd.Series(dtype=float)
    claim_stable_frequency = float(
        (claim_levels == _to_float(baseline_claim.get("current_level_numeric"), 0.0)).mean()
    ) if not claim_levels.empty else 1.0
    leading_frequency = (
        float(leading_interval.iloc[0]["leading_frequency"]) if not leading_interval.empty else 0.0
    )
    robustness = {
        "baseline_leading_hypothesis": leading,
        "leading_stability_frequency": leading_frequency,
        "leading_stable": bool(leading_frequency >= 0.8),
        "claim_ceiling_stability_frequency": claim_stable_frequency,
        "claim_ceiling_stable": bool(claim_stable_frequency >= 0.8),
        "n_sensitivity_iterations": int(total_iterations),
    }
    if not leading_interval.empty:
        robustness.update(
            {
                "leading_posterior_median": float(leading_interval.iloc[0]["posterior_median"]),
                "leading_posterior_q025": float(leading_interval.iloc[0]["posterior_q025"]),
                "leading_posterior_q975": float(leading_interval.iloc[0]["posterior_q975"]),
            }
        )
    return sensitivity_df, interval_df, dominant, robustness


def _naive_fusion_posterior(
    evidence: Sequence[Mapping[str, Any]],
    hypotheses: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Compute posterior by summing every evidence item directly."""

    hypothesis_ids = [str(hyp["hypothesis_id"]) for hyp in hypotheses]
    totals = {hyp: 0.0 for hyp in hypothesis_ids}
    for row in evidence:
        item = _item_log_lrs(row, hypothesis_ids, config)
        for hyp, value in item.items():
            totals[hyp] += float(value)
    rows = []
    log_values = []
    for hyp in hypotheses:
        prior = max(_to_float(hyp.get("prior"), 0.0), 1e-12)
        log_values.append(math.log(prior) + totals[str(hyp["hypothesis_id"])])
    log_values_np = np.asarray(log_values, dtype=float)
    log_values_np = log_values_np - log_values_np.max()
    probs = np.exp(log_values_np)
    probs = probs / probs.sum()
    for hyp, posterior in zip(hypotheses, probs):
        rows.append(
            {
                "hypothesis": hyp["hypothesis_id"],
                "label": hyp.get("label", hyp["hypothesis_id"]),
                "prior": float(hyp["prior"]),
                "log_likelihood_ratio_total": float(totals[str(hyp["hypothesis_id"])]),
                "posterior": float(posterior),
            }
        )
    return pd.DataFrame(rows).sort_values("posterior", ascending=False).reset_index(drop=True)


def _fusion_comparison(
    aware_posterior: pd.DataFrame,
    naive_posterior: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    aware = aware_posterior.copy()
    naive = naive_posterior.copy()
    aware.insert(0, "fusion_model", "dependency_aware")
    naive.insert(0, "fusion_model", "naive")
    comparison = pd.concat([aware, naive], ignore_index=True, sort=False)
    aware_leading = str(aware_posterior.iloc[0]["hypothesis"])
    aware_value = float(
        aware_posterior[aware_posterior["hypothesis"] == aware_leading].iloc[0]["posterior"]
    )
    naive_value = float(
        naive_posterior[naive_posterior["hypothesis"] == aware_leading].iloc[0]["posterior"]
    )
    diff = naive_value - aware_value
    if diff > 0.05:
        warning = (
            "# Overconfidence warning\n\n"
            f"Naive fusion assigns posterior {naive_value:.3f} to {aware_leading}, "
            f"versus {aware_value:.3f} under dependency-aware fusion. "
            "This suggests item-level fusion may overstate confidence by double counting "
            "correlated evidence."
        )
    else:
        warning = (
            "# Overconfidence warning\n\n"
            "Naive and dependency-aware fusion are similar for the leading hypothesis; "
            "no large overconfidence signal was detected."
        )
    return comparison, warning


def _augment_robustness_summary(
    robustness: Mapping[str, Any] | None,
    lofo_summary: pd.DataFrame,
) -> dict[str, Any]:
    summary = dict(robustness or {})
    if not lofo_summary.empty:
        most = lofo_summary.sort_values("posterior_drop_for_baseline_leading", ascending=False).iloc[0]
        summary["most_influential_evidence_family"] = str(most["family_removed"])
        summary["most_influential_posterior_drop"] = float(
            most["posterior_drop_for_baseline_leading"]
        )
        assumptions = _as_list(most.get("assumptions", ""))
        failure_modes = _as_list(most.get("failure_modes", ""))
        summary["most_fragile_assumption"] = assumptions[0] if assumptions else "not available"
        summary["most_relevant_failure_mode"] = failure_modes[0] if failure_modes else "not available"
        summary["lofo_any_leading_change"] = bool(lofo_summary["leading_changed"].any())
        summary["lofo_min_claim_level"] = float(lofo_summary["claim_level_numeric"].min())
    return summary


def adjudicate_mechanism(
    evidence_input: Mapping[str, Any] | Sequence[Any],
    hypotheses_input: Mapping[str, Any] | Sequence[Any] | None = None,
    *,
    event: str | None = None,
    strict: bool = False,
    provenance: Mapping[str, Any] | None = None,
    sensitivity: bool = False,
    prior_grid: bool = False,
    weight_jitter: float = 0.0,
    n_bootstrap: int = 0,
    leave_one_family_out: bool = True,
    compare_naive: bool = False,
    random_seed: int | None = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    """Compute a TED-MAD mechanism posterior and claim ceiling."""

    if strict:
        if not isinstance(hypotheses_input, Mapping):
            raise ValueError("strict TED-MAD validation requires a hypotheses mapping")
        if not isinstance(evidence_input, Mapping):
            raise ValueError("strict TED-MAD validation requires an evidence mapping")
        hypothesis_ids = validate_hypotheses(hypotheses_input)
        validate_evidence(evidence_input, hypothesis_ids)

    hypotheses, config = _parse_hypotheses(hypotheses_input)
    evidence = _parse_evidence(evidence_input, event=event)
    item_df, family_df = _aggregate_evidence(evidence, hypotheses, config)
    posterior_df = _posterior_from_contributions(hypotheses, family_df)
    claim = _claim_ceiling(evidence, family_df, posterior_df, config)
    lofo_df = _leave_one_family_out(hypotheses, family_df) if leave_one_family_out else pd.DataFrame()
    lofo_summary_df = (
        _leave_one_family_out_summary(evidence, hypotheses, family_df, posterior_df, claim, config)
        if leave_one_family_out
        else pd.DataFrame()
    )

    sensitivity_df = pd.DataFrame()
    interval_df = pd.DataFrame()
    dominance_df = pd.DataFrame()
    robustness_summary: dict[str, Any] = {}
    if sensitivity or prior_grid or weight_jitter > 0 or n_bootstrap > 0:
        sensitivity_df, interval_df, dominance_df, robustness_summary = (
            _posterior_sensitivity_analysis(
                evidence,
                hypotheses,
                config,
                baseline_posterior=posterior_df,
                baseline_claim=claim,
                prior_grid=prior_grid,
                weight_jitter=weight_jitter,
                n_bootstrap=n_bootstrap,
                random_seed=random_seed,
            )
        )
    robustness_summary = _augment_robustness_summary(robustness_summary, lofo_summary_df)

    fusion_comparison_df = pd.DataFrame()
    overconfidence_warning = ""
    if compare_naive:
        naive_posterior = _naive_fusion_posterior(evidence, hypotheses, config)
        fusion_comparison_df, overconfidence_warning = _fusion_comparison(
            posterior_df, naive_posterior
        )

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_config": dict(config),
        "hypotheses": hypotheses,
        "evidence": evidence,
        "evidence_item_contribution": item_df,
        "evidence_contribution": family_df,
        "posterior": posterior_df,
        "leave_one_evidence_family_out": lofo_df,
        "leave_one_evidence_family_out_summary": lofo_summary_df,
        "posterior_sensitivity": sensitivity_df,
        "posterior_interval": interval_df,
        "dominance_frequency": dominance_df,
        "robustness_summary": robustness_summary,
        "fusion_comparison": fusion_comparison_df,
        "overconfidence_warning_markdown": overconfidence_warning,
        "claim_ceiling": claim,
        "provenance": dict(provenance or {}),
    }


def write_adjudication_outputs(result: Mapping[str, Any], outdir: str | Path) -> dict[str, str]:
    """Write standard TED-MAD adjudication output files."""

    out = _ensure_dir(outdir)
    paths = {
        "mechanism_posterior": out / "mechanism_posterior.csv",
        "evidence_contribution": out / "evidence_contribution.csv",
        "evidence_item_contribution": out / "evidence_item_contribution.csv",
        "claim_ceiling": out / "claim_ceiling.json",
        "provenance": out / "provenance.json",
        "provenance_yaml": out / "provenance.yaml",
        "posterior_bundle": out / "posterior.json",
        "posterior_yaml": out / "posterior.yaml",
        "leave_one_evidence_family_out": out / "leave_one_evidence_family_out.csv",
        "leave_one_evidence_family_out_summary": out / "leave_one_evidence_family_out_summary.csv",
        "posterior_sensitivity": out / "posterior_sensitivity.csv",
        "posterior_interval": out / "posterior_interval.csv",
        "dominance_frequency": out / "dominance_frequency.csv",
        "robustness_summary": out / "robustness_summary.json",
        "fusion_comparison": out / "fusion_comparison.csv",
        "overconfidence_warning": out / "overconfidence_warning.md",
    }

    result["posterior"].to_csv(paths["mechanism_posterior"], index=False)
    result["evidence_contribution"].to_csv(paths["evidence_contribution"], index=False)
    result["evidence_item_contribution"].to_csv(paths["evidence_item_contribution"], index=False)
    result["leave_one_evidence_family_out"].to_csv(
        paths["leave_one_evidence_family_out"], index=False
    )
    result["leave_one_evidence_family_out_summary"].to_csv(
        paths["leave_one_evidence_family_out_summary"], index=False
    )
    if not result.get("posterior_sensitivity", pd.DataFrame()).empty:
        result["posterior_sensitivity"].to_csv(paths["posterior_sensitivity"], index=False)
    if not result.get("posterior_interval", pd.DataFrame()).empty:
        result["posterior_interval"].to_csv(paths["posterior_interval"], index=False)
    if not result.get("dominance_frequency", pd.DataFrame()).empty:
        result["dominance_frequency"].to_csv(paths["dominance_frequency"], index=False)
    if not result.get("fusion_comparison", pd.DataFrame()).empty:
        result["fusion_comparison"].to_csv(paths["fusion_comparison"], index=False)
    if result.get("overconfidence_warning_markdown"):
        paths["overconfidence_warning"].write_text(
            result["overconfidence_warning_markdown"], encoding="utf-8"
        )
    paths["claim_ceiling"].write_text(
        json.dumps(result["claim_ceiling"], indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    paths["robustness_summary"].write_text(
        json.dumps(
            result.get("robustness_summary", {}),
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    paths["provenance"].write_text(
        json.dumps(result.get("provenance", {}), indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    _write_yaml(paths["provenance_yaml"], result.get("provenance", {}))

    bundle = {
        "created_at": result["created_at"],
        "model_config": result["model_config"],
        "hypotheses": result["hypotheses"],
        "target_events": sorted(
            {
                str(row.get("target_event"))
                for row in result.get("evidence", [])
                if row.get("target_event")
            }
        ),
        "posterior": result["posterior"].to_dict(orient="records"),
        "claim_ceiling": result["claim_ceiling"],
        "robustness_summary": result.get("robustness_summary", {}),
        "posterior_interval": result.get("posterior_interval", pd.DataFrame()).to_dict(
            orient="records"
        )
        if isinstance(result.get("posterior_interval"), pd.DataFrame)
        else [],
        "dominance_frequency": result.get("dominance_frequency", pd.DataFrame()).to_dict(
            orient="records"
        )
        if isinstance(result.get("dominance_frequency"), pd.DataFrame)
        else [],
        "leave_one_evidence_family_out_summary": result.get(
            "leave_one_evidence_family_out_summary", pd.DataFrame()
        ).to_dict(orient="records")
        if isinstance(result.get("leave_one_evidence_family_out_summary"), pd.DataFrame)
        else [],
        "overconfidence_warning_markdown": result.get("overconfidence_warning_markdown", ""),
        "provenance": result.get("provenance", {}),
    }
    paths["posterior_bundle"].write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    _write_yaml(paths["posterior_yaml"], bundle)
    return {key: str(path) for key, path in paths.items() if path.exists()}


def _read_posterior_bundle(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() in {".json", ".yaml", ".yml"}:
        payload = load_ted_mad_yaml(path)
        if "posterior" in payload:
            return payload
    df = pd.read_csv(path)
    return {
        "posterior": df.to_dict(orient="records"),
        "claim_ceiling": {},
        "hypotheses": [
            {"hypothesis_id": row["hypothesis"], "label": row.get("label", row["hypothesis"])}
            for row in df.to_dict(orient="records")
        ],
    }


def _parse_experiments(raw: Mapping[str, Any] | Sequence[Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = {
        "lambda_claim": 1.0,
        "gamma_falsification": 0.5,
        "cost_weight": 0.2,
        "risk_weight": 0.3,
        "pattern_reliability": 0.8,
    }
    if isinstance(raw, Mapping):
        config.update(raw.get("design_model", {}) or {})
        entries = raw.get("experiments", raw.get("candidate_experiments", []))
    else:
        entries = raw

    experiments: list[dict[str, Any]] = []
    for i, item in enumerate(entries or []):
        exp = dict(item or {})
        exp["experiment_id"] = str(
            exp.get("experiment_id") or exp.get("id") or exp.get("name") or f"A{i + 1}"
        )
        exp["name"] = str(exp.get("name") or exp.get("label") or exp["experiment_id"])
        exp["readouts"] = _as_list(exp.get("readouts") or exp.get("minimal_readout_panel"))
        exp["supports_hypotheses"] = _as_list(exp.get("supports_hypotheses") or exp.get("supports"))
        if "falsifiers" in exp and "falsifies" not in exp:
            exp["falsifies"] = exp["falsifiers"]
        experiments.append(exp)
    if not experiments:
        raise ValueError("experiment library is empty")
    return experiments, config


def _entropy(probs: np.ndarray) -> float:
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def _pattern_signature(pattern: Any) -> str:
    if pattern is None:
        return "__unspecified__"
    if isinstance(pattern, Mapping):
        items = sorted((str(k), str(v)) for k, v in pattern.items())
        return "|".join(f"{k}={v}" for k, v in items)
    return str(pattern)


def _expected_information_gain(
    posterior: Mapping[str, float],
    expected_patterns: Mapping[str, Any],
    reliability: float,
) -> float:
    hypotheses = list(posterior)
    prior = np.asarray([posterior[h] for h in hypotheses], dtype=float)
    if prior.sum() <= 0:
        return 0.0
    prior = prior / prior.sum()

    signatures = {h: _pattern_signature(expected_patterns.get(h)) for h in hypotheses}
    outcomes = sorted({sig for sig in signatures.values() if sig != "__unspecified__"})
    if len(outcomes) <= 1:
        return 0.0

    reliability = _clip(reliability, 1 / len(outcomes), 0.999)
    likelihood = np.zeros((len(outcomes), len(hypotheses)), dtype=float)
    for oi, outcome in enumerate(outcomes):
        for hi, hyp in enumerate(hypotheses):
            sig = signatures[hyp]
            if sig == "__unspecified__":
                likelihood[oi, hi] = 1.0 / len(outcomes)
            elif sig == outcome:
                likelihood[oi, hi] = reliability
            else:
                likelihood[oi, hi] = (1.0 - reliability) / (len(outcomes) - 1)

    h_prior = _entropy(prior)
    expected_h = 0.0
    for oi in range(len(outcomes)):
        p_outcome = float((prior * likelihood[oi, :]).sum())
        if p_outcome <= 0:
            continue
        post = prior * likelihood[oi, :] / p_outcome
        expected_h += p_outcome * _entropy(post)
    return max(0.0, h_prior - expected_h)


def _cost_or_risk(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    if isinstance(value, str):
        lookup = {
            "none": 0.0,
            "very_low": 0.05,
            "very low": 0.05,
            "low": 0.2,
            "medium": 0.5,
            "moderate": 0.5,
            "high": 0.8,
            "very_high": 1.0,
            "very high": 1.0,
        }
        return lookup.get(_canonical_text(value), _to_float(value, default))
    return _to_float(value, default)


def _falsification_score(exp: Mapping[str, Any], leading_hypothesis: str, patterns: Mapping[str, Any]) -> float:
    raw = exp.get("falsifies") or exp.get("falsification_rules") or []
    if isinstance(raw, Mapping):
        falsifiers = [raw]
    elif isinstance(raw, Sequence) and not isinstance(raw, str):
        falsifiers = list(raw)
    elif raw:
        falsifiers = [raw]
    else:
        falsifiers = []

    score = 0.0
    if falsifiers:
        score += 0.35
        text = _canonical_text(json.dumps(falsifiers, ensure_ascii=False))
        if _canonical_text(leading_hypothesis) in text:
            score += 0.35

    leading_sig = _pattern_signature(patterns.get(leading_hypothesis))
    other_sigs = {_pattern_signature(v) for h, v in patterns.items() if h != leading_hypothesis}
    if leading_sig != "__unspecified__" and leading_sig not in other_sigs:
        score += 0.30
    return _clip(score, 0.0, 1.0)


def _claim_delta(
    exp: Mapping[str, Any],
    posterior: Mapping[str, float],
    leading_hypothesis: str,
    current_level: float,
) -> float:
    target = _to_float(exp.get("claim_level_if_success"), current_level)
    if target <= current_level:
        upgrade_evidence = str(exp.get("claim_upgrade_evidence") or "")
        if _categories_for_family(upgrade_evidence).intersection({"direct_rescue"}):
            target = max(target, 4.0)
        elif _categories_for_family(upgrade_evidence).intersection({"orthogonal_perturbation"}):
            target = max(target, 5.0)

    support_hypotheses = _as_list(exp.get("supports_hypotheses") or exp.get("supports"))
    if support_hypotheses:
        success_prob = sum(posterior.get(h, 0.0) for h in support_hypotheses)
    else:
        success_prob = posterior.get(leading_hypothesis, 0.0)
    return max(0.0, target - current_level) * _clip(success_prob, 0.0, 1.0)


def _pattern_rows(experiments: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for exp in experiments:
        patterns = exp.get("expected_patterns") or exp.get("expected_result_patterns") or {}
        for hyp, readout_map in dict(patterns).items():
            if isinstance(readout_map, Mapping):
                for readout, pattern in readout_map.items():
                    rows.append(
                        {
                            "experiment_id": exp["experiment_id"],
                            "experiment_name": exp["name"],
                            "hypothesis": hyp,
                            "readout": readout,
                            "expected_pattern": pattern,
                        }
                    )
            else:
                rows.append(
                    {
                        "experiment_id": exp["experiment_id"],
                        "experiment_name": exp["name"],
                        "hypothesis": hyp,
                        "readout": "overall",
                        "expected_pattern": readout_map,
                    }
                )
    return pd.DataFrame(rows)


def _norm_token(value: Any) -> str:
    return _canonical_text(value).replace(" ", "_")


def _rescue_strength(value: Any) -> float:
    text = _norm_token(value)
    if not text or text == "__unspecified__":
        return 0.0
    if "strong" in text and "no_rescue" not in text:
        return 1.0
    if "partial_to_strong" in text:
        return 0.85
    if "partial" in text or "moderate" in text or "reduced" in text or "decrease" in text:
        return 0.55
    if "weak" in text or "delayed" in text or "incomplete" in text:
        return 0.25
    if "no_rescue" in text or "unchanged" in text or "none" in text or "no" == text:
        return 0.0
    if "worse" in text or "increase" in text:
        return -0.25
    return 0.35


def _contrast_label(score: float) -> str:
    if score >= 0.65:
        return "strong"
    if score >= 0.35:
        return "moderate"
    if score >= 0.15:
        return "weak"
    return "none"


def _claim_upgrade_label(delta: float) -> str:
    if delta >= 0.75:
        return "high"
    if delta >= 0.25:
        return "medium"
    if delta > 0:
        return "low"
    return "none"


def _extract_falsifier_hypotheses(exp: Mapping[str, Any]) -> set[str]:
    raw = exp.get("falsifies") or exp.get("falsification_rules") or []
    if isinstance(raw, Mapping):
        raw = [raw]
    if isinstance(raw, str):
        return set()
    out = set()
    for item in raw or []:
        if isinstance(item, Mapping) and item.get("hypothesis"):
            out.add(str(item["hypothesis"]))
    return out


def _pattern_contrast(pattern_a: Mapping[str, Any], pattern_b: Mapping[str, Any]) -> float:
    readouts = set(pattern_a) | set(pattern_b)
    if not readouts:
        return 0.0
    diffs = [
        abs(_rescue_strength(pattern_a.get(readout)) - _rescue_strength(pattern_b.get(readout)))
        for readout in readouts
    ]
    return float(np.mean(diffs))


def _experiment_contrast_matrix(
    experiments: Sequence[Mapping[str, Any]],
    leading_hypothesis: str,
    current_level: float,
) -> pd.DataFrame:
    rows = []
    for exp in experiments:
        patterns = exp.get("expected_patterns") or exp.get("expected_result_patterns") or {}
        leading_pattern = patterns.get(leading_hypothesis, {})
        row = {
            "experiment_id": exp["experiment_id"],
            "name": exp["name"],
        }
        for hyp in sorted(patterns):
            if hyp == leading_hypothesis:
                continue
            score = _pattern_contrast(leading_pattern, patterns.get(hyp, {}))
            row[f"{leading_hypothesis}_vs_{hyp}"] = _contrast_label(score)
            row[f"{leading_hypothesis}_vs_{hyp}_score"] = score
        row["claim_upgrade"] = _claim_upgrade_label(
            _to_float(exp.get("claim_level_if_success"), current_level) - current_level
        )
        row[f"falsifies_{leading_hypothesis}"] = leading_hypothesis in _extract_falsifier_hypotheses(exp)
        rows.append(row)
    return pd.DataFrame(rows)


def _readout_panel_rows(experiments: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for exp in experiments:
        panel = exp.get("minimal_readout_panel") or {}
        required = _as_list(panel.get("required") if isinstance(panel, Mapping) else None)
        optional = _as_list(panel.get("optional") if isinstance(panel, Mapping) else None)
        controls = _as_list(panel.get("negative_controls") if isinstance(panel, Mapping) else None)
        if not required:
            required = _as_list(exp.get("readouts"))
        patterns = exp.get("expected_patterns") or exp.get("expected_result_patterns") or {}
        pattern_readouts = []
        for hyp_pattern in patterns.values():
            if isinstance(hyp_pattern, Mapping):
                pattern_readouts.extend(str(readout) for readout in hyp_pattern)
        if not required and pattern_readouts:
            required = sorted(dict.fromkeys(pattern_readouts))

        for category, readouts in [
            ("required", required),
            ("optional", optional),
            ("negative_control", controls),
        ]:
            for readout in readouts:
                rows.append(
                    {
                        "experiment_id": exp["experiment_id"],
                        "experiment_name": exp["name"],
                        "category": category,
                        "readout": readout,
                    }
                )
    return pd.DataFrame(rows)


def _design_sensitivity_analysis(
    rank_df: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    cost_risk_jitter: float = 0.2,
    n_bootstrap: int = 100,
    random_seed: int | None = DEFAULT_RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stress-test rescue ranking under cost/risk weight perturbations."""

    if rank_df.empty or n_bootstrap <= 0:
        return pd.DataFrame(), pd.DataFrame()
    rng = np.random.default_rng(random_seed)
    jitter = max(float(cost_risk_jitter), 0.0)
    low = max(0.0, 1.0 - jitter)
    high = 1.0 + jitter
    base_cost_weight = _to_float(config.get("cost_weight"), 0.2)
    base_risk_weight = _to_float(config.get("risk_weight"), 0.3)
    lambda_claim = _to_float(config.get("lambda_claim"), 1.0)
    gamma_falsification = _to_float(config.get("gamma_falsification"), 0.5)

    rows = []
    for iteration in range(1, int(n_bootstrap) + 1):
        cost_weight = base_cost_weight * float(rng.uniform(low, high))
        risk_weight = base_risk_weight * float(rng.uniform(low, high))
        scored = rank_df.copy()
        scored["utility"] = (
            scored["expected_information_gain"]
            + lambda_claim * scored["expected_claim_delta"]
            + gamma_falsification * scored["falsification_score"]
            - cost_weight * scored["cost_normalized"]
            - risk_weight * scored["risk_normalized"]
        )
        scored = scored.sort_values("utility", ascending=False).reset_index(drop=True)
        scored["rank"] = np.arange(1, len(scored) + 1)
        for row in scored.to_dict(orient="records"):
            rows.append(
                {
                    "iteration": iteration,
                    "experiment_id": row["experiment_id"],
                    "name": row["name"],
                    "utility": float(row["utility"]),
                    "rank": int(row["rank"]),
                    "is_top": int(row["rank"]) == 1,
                    "cost_weight": float(cost_weight),
                    "risk_weight": float(risk_weight),
                }
            )
    sensitivity_df = pd.DataFrame(rows)
    summary_rows = []
    for exp_id, group in sensitivity_df.groupby("experiment_id", sort=False):
        summary_rows.append(
            {
                "experiment_id": exp_id,
                "name": str(group.iloc[0]["name"]),
                "top_frequency": float(group["is_top"].mean()),
                "median_rank": float(group["rank"].median()),
                "rank_q025": float(group["rank"].quantile(0.025)),
                "rank_q975": float(group["rank"].quantile(0.975)),
                "utility_median": float(group["utility"].median()),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["top_frequency", "utility_median"], ascending=[False, False]
    ).reset_index(drop=True)
    return sensitivity_df, summary_df


def design_rescue_experiments(
    posterior_bundle: Mapping[str, Any],
    experiments_input: Mapping[str, Any] | Sequence[Any],
    *,
    claim_ceiling: Mapping[str, Any] | None = None,
    strict: bool = False,
    provenance: Mapping[str, Any] | None = None,
    design_sensitivity: bool = False,
    cost_risk_jitter: float = 0.2,
    n_design_bootstrap: int = 0,
    random_seed: int | None = DEFAULT_RANDOM_SEED,
    lambda_claim: float | None = None,
    gamma_falsification: float | None = None,
    cost_weight: float | None = None,
    risk_weight: float | None = None,
) -> dict[str, Any]:
    """Rank rescue experiments by EIG, claim gain, falsification value, cost, and risk."""

    if strict:
        if not isinstance(experiments_input, Mapping):
            raise ValueError("strict TED-MAD validation requires an experiment mapping")
        hypothesis_ids = {str(row["hypothesis"]) for row in posterior_bundle["posterior"]}
        validate_experiments(experiments_input, hypothesis_ids)

    experiments, config = _parse_experiments(experiments_input)
    if lambda_claim is not None:
        config["lambda_claim"] = lambda_claim
    if gamma_falsification is not None:
        config["gamma_falsification"] = gamma_falsification
    if cost_weight is not None:
        config["cost_weight"] = cost_weight
    if risk_weight is not None:
        config["risk_weight"] = risk_weight

    posterior_rows = posterior_bundle["posterior"]
    posterior = {str(row["hypothesis"]): _to_float(row["posterior"], 0.0) for row in posterior_rows}
    posterior_total = sum(posterior.values())
    if posterior_total <= 0:
        raise ValueError("posterior probabilities must sum to a positive value")
    posterior = {hyp: value / posterior_total for hyp, value in posterior.items()}
    leading_hypothesis = max(posterior, key=posterior.get)
    claim = dict(claim_ceiling or posterior_bundle.get("claim_ceiling") or {})
    current_level = _to_float(claim.get("current_level_numeric"), 1.0)

    raw_costs = [_cost_or_risk(exp.get("cost"), 0.5) for exp in experiments]
    raw_risks = [_cost_or_risk(exp.get("risk"), 0.2) for exp in experiments]
    cost_scale = max(max(raw_costs), 1.0)
    risk_scale = max(max(raw_risks), 1.0)

    rank_rows: list[dict[str, Any]] = []
    falsification_lines: list[str] = []
    for exp, raw_cost, raw_risk in zip(experiments, raw_costs, raw_risks):
        patterns = exp.get("expected_patterns") or exp.get("expected_result_patterns") or {}
        reliability = _to_float(exp.get("pattern_reliability"), _to_float(config.get("pattern_reliability"), 0.8))
        eig = _expected_information_gain(posterior, patterns, reliability)
        delta_claim = _claim_delta(exp, posterior, leading_hypothesis, current_level)
        falsification = _falsification_score(exp, leading_hypothesis, patterns)
        cost_norm = _clip(raw_cost / cost_scale, 0.0, 1.0)
        risk_norm = _clip(raw_risk / risk_scale, 0.0, 1.0)
        utility = (
            eig
            + _to_float(config.get("lambda_claim"), 1.0) * delta_claim
            + _to_float(config.get("gamma_falsification"), 0.5) * falsification
            - _to_float(config.get("cost_weight"), 0.2) * cost_norm
            - _to_float(config.get("risk_weight"), 0.3) * risk_norm
        )
        rank_rows.append(
            {
                "experiment_id": exp["experiment_id"],
                "name": exp["name"],
                "description": exp.get("description", ""),
                "utility": float(utility),
                "expected_information_gain": float(eig),
                "expected_claim_delta": float(delta_claim),
                "falsification_score": float(falsification),
                "cost": float(raw_cost),
                "risk": float(raw_risk),
                "cost_normalized": float(cost_norm),
                "risk_normalized": float(risk_norm),
                "readouts": ";".join(exp.get("readouts", [])),
                "claim_upgrade_evidence": exp.get("claim_upgrade_evidence", ""),
            }
        )

        raw_falsifies = exp.get("falsifies") or exp.get("falsification_rules")
        if raw_falsifies:
            falsification_lines.append(f"### {exp['experiment_id']} {exp['name']}")
            if isinstance(raw_falsifies, str):
                falsification_lines.append(raw_falsifies)
            else:
                falsification_lines.append(
                    "```json\n"
                    + json.dumps(raw_falsifies, indent=2, ensure_ascii=False, default=_json_default)
                    + "\n```"
                )

    rank_df = pd.DataFrame(rank_rows).sort_values("utility", ascending=False).reset_index(drop=True)
    rank_df["rank"] = np.arange(1, len(rank_df) + 1)
    rank_df["next_best_experiment"] = rank_df["rank"] == 1
    pattern_df = _pattern_rows(experiments)
    contrast_df = _experiment_contrast_matrix(experiments, leading_hypothesis, current_level)
    readout_panel_df = _readout_panel_rows(experiments)
    best = rank_df.iloc[0].to_dict()
    design_sensitivity_df, design_stability_df = (
        _design_sensitivity_analysis(
            rank_df,
            config,
            cost_risk_jitter=cost_risk_jitter,
            n_bootstrap=n_design_bootstrap,
            random_seed=random_seed,
        )
        if design_sensitivity or n_design_bootstrap > 0
        else (pd.DataFrame(), pd.DataFrame())
    )
    merged_provenance = merge_provenance(
        posterior_bundle.get("provenance", {}),
        provenance,
    )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "design_model": dict(config),
        "leading_hypothesis": leading_hypothesis,
        "current_claim_ceiling": claim,
        "ranked_experiments": rank_df,
        "expected_result_patterns": pattern_df,
        "experiment_contrast_matrix": contrast_df,
        "minimal_readout_panel": readout_panel_df,
        "falsification_rules_markdown": "\n\n".join(falsification_lines).strip(),
        "next_best_experiment": best,
        "design_sensitivity": design_sensitivity_df,
        "design_rank_stability": design_stability_df,
        "provenance": merged_provenance,
    }


def write_design_outputs(result: Mapping[str, Any], outdir: str | Path) -> dict[str, str]:
    out = _ensure_dir(outdir)
    paths = {
        "rescue_design_rank": out / "rescue_design_rank.csv",
        "expected_result_patterns": out / "expected_result_patterns.csv",
        "experiment_contrast_matrix": out / "experiment_contrast_matrix.csv",
        "minimal_readout_panel": out / "minimal_readout_panel.csv",
        "falsification_rules": out / "falsification_rules.md",
        "design_sensitivity": out / "design_sensitivity.csv",
        "design_rank_stability": out / "design_rank_stability.csv",
        "provenance": out / "provenance.json",
        "provenance_yaml": out / "provenance.yaml",
        "design_bundle": out / "design.json",
        "design_yaml": out / "design.yaml",
    }
    result["ranked_experiments"].to_csv(paths["rescue_design_rank"], index=False)
    result["expected_result_patterns"].to_csv(paths["expected_result_patterns"], index=False)
    result["experiment_contrast_matrix"].to_csv(
        paths["experiment_contrast_matrix"], index=False
    )
    result["minimal_readout_panel"].to_csv(paths["minimal_readout_panel"], index=False)
    if not result.get("design_sensitivity", pd.DataFrame()).empty:
        result["design_sensitivity"].to_csv(paths["design_sensitivity"], index=False)
    if not result.get("design_rank_stability", pd.DataFrame()).empty:
        result["design_rank_stability"].to_csv(paths["design_rank_stability"], index=False)
    paths["falsification_rules"].write_text(
        result.get("falsification_rules_markdown", ""), encoding="utf-8"
    )
    paths["provenance"].write_text(
        json.dumps(result.get("provenance", {}), indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    _write_yaml(paths["provenance_yaml"], result.get("provenance", {}))
    bundle = {
        "created_at": result["created_at"],
        "design_model": result["design_model"],
        "leading_hypothesis": result["leading_hypothesis"],
        "current_claim_ceiling": result["current_claim_ceiling"],
        "ranked_experiments": result["ranked_experiments"].to_dict(orient="records"),
        "expected_result_patterns": result["expected_result_patterns"].to_dict(orient="records"),
        "experiment_contrast_matrix": result["experiment_contrast_matrix"].to_dict(
            orient="records"
        ),
        "minimal_readout_panel": result["minimal_readout_panel"].to_dict(orient="records"),
        "design_rank_stability": result.get("design_rank_stability", pd.DataFrame()).to_dict(
            orient="records"
        )
        if isinstance(result.get("design_rank_stability"), pd.DataFrame)
        else [],
        "falsification_rules_markdown": result.get("falsification_rules_markdown", ""),
        "next_best_experiment": result["next_best_experiment"],
        "provenance": result.get("provenance", {}),
    }
    paths["design_bundle"].write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
    )
    _write_yaml(paths["design_yaml"], bundle)
    return {key: str(path) for key, path in paths.items()}


def _read_design_bundle(path: str | Path) -> dict[str, Any]:
    payload = load_ted_mad_yaml(path)
    if "ranked_experiments" not in payload:
        raise ValueError("design bundle must contain ranked_experiments")
    return payload


def _markdown_table(df: pd.DataFrame, columns: Sequence[str], max_rows: int = 8) -> str:
    if df.empty:
        return "_None._"
    use = df.loc[:, [col for col in columns if col in df.columns]].head(max_rows)
    return use.to_markdown(index=False)


def _provenance_markdown(provenance: Mapping[str, Any]) -> str:
    if not provenance:
        return "_No provenance block supplied._"
    lines = [
        f"- TED-MAD version: `{provenance.get('ted_mad_version', 'unknown')}`",
        f"- pyfgsea version: `{provenance.get('pyfgsea_version', 'unknown')}`",
        f"- git commit: `{provenance.get('git_commit', 'unknown')}`",
        f"- run timestamp: `{provenance.get('run_timestamp', 'unknown')}`",
        f"- random seed: `{provenance.get('random_seed', 'unknown')}`",
    ]
    input_files = provenance.get("input_files", {}) or {}
    input_sha256 = provenance.get("input_sha256", {}) or {}
    for key in sorted(set(input_files) | set(input_sha256)):
        path = input_files.get(key, "unknown")
        digest = input_sha256.get(key, "unknown")
        lines.append(f"- {key}: `{path}` sha256 `{digest}`")
    return "\n".join(lines)


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _robustness_markdown(
    robustness: Mapping[str, Any],
    posterior_interval: pd.DataFrame,
    dominance_frequency: pd.DataFrame,
) -> str:
    if not robustness and posterior_interval.empty and dominance_frequency.empty:
        return "_No sensitivity analysis was requested for this run._"
    leading = robustness.get("baseline_leading_hypothesis", "unknown")
    lines = [
        f"- Leading hypothesis stable under perturbation: {_yes_no(robustness.get('leading_stable', False))}",
        f"- Leading stability frequency: {float(robustness.get('leading_stability_frequency', 0.0)):.3f}",
        f"- Claim ceiling stable: {_yes_no(robustness.get('claim_ceiling_stable', False))}",
        f"- Claim ceiling stability frequency: {float(robustness.get('claim_ceiling_stability_frequency', 0.0)):.3f}",
        f"- Most influential evidence family: {robustness.get('most_influential_evidence_family', 'not available')}",
        f"- Most fragile assumption: {robustness.get('most_fragile_assumption', 'not available')}",
        f"- Most relevant failure mode: {robustness.get('most_relevant_failure_mode', 'not available')}",
    ]
    if "most_influential_posterior_drop" in robustness:
        lines.append(
            "- Largest posterior drop for leading hypothesis: "
            f"{float(robustness['most_influential_posterior_drop']):.3f}"
        )
    if not posterior_interval.empty and "hypothesis" in posterior_interval.columns:
        row = posterior_interval[posterior_interval["hypothesis"] == leading]
        if not row.empty:
            lines.append(
                "- Leading posterior median and interval: "
                f"{float(row.iloc[0]['posterior_median']):.3f} "
                f"({float(row.iloc[0]['posterior_q025']):.3f}-"
                f"{float(row.iloc[0]['posterior_q975']):.3f})"
            )
    if not dominance_frequency.empty:
        lines.append("")
        lines.append(_markdown_table(dominance_frequency, ["leading_hypothesis", "dominance_frequency"]))
    return "\n".join(lines)


def _design_robustness_markdown(design_rank_stability: pd.DataFrame) -> str:
    if design_rank_stability.empty:
        return "_No cost/risk ranking sensitivity analysis was requested for this run._"
    return _markdown_table(
        design_rank_stability,
        ["experiment_id", "top_frequency", "median_rank", "rank_q025", "rank_q975"],
    )


def _minimal_panel_markdown(panel_df: pd.DataFrame, experiment_id: str) -> str:
    if panel_df.empty:
        return "_No minimal readout panel supplied._"
    use = panel_df[panel_df["experiment_id"] == experiment_id]
    if use.empty:
        return "_No minimal readout panel supplied for the next-best experiment._"
    labels = {
        "required": "Required",
        "optional": "Optional",
        "negative_control": "Negative/control readouts",
    }
    lines = []
    for category in ("required", "optional", "negative_control"):
        readouts = use[use["category"] == category]["readout"].tolist()
        if readouts:
            lines.append(f"{labels[category]}:")
            lines.extend(f"- {readout}" for readout in readouts)
    return "\n".join(lines) if lines else "_No minimal readout panel supplied._"


def _candidate_event_label(posterior_bundle: Mapping[str, Any]) -> str:
    event = posterior_bundle.get("target_event")
    if event:
        return str(event)
    events = posterior_bundle.get("target_events") or []
    if isinstance(events, Sequence) and not isinstance(events, str) and events:
        return ", ".join(str(item) for item in events)
    return "not supplied"


def _claim_limitation_sentence(claim: Mapping[str, Any]) -> str:
    level = _to_float(claim.get("current_level_numeric"), 0.0)
    missing = claim.get("missing_evidence_for_next_level", []) or []
    if level == 3.5 and any("rescue" in str(item).lower() for item in missing):
        return (
            "The current data support a computationally adjudicated, rescue-ready "
            "mechanism model, but not a rescue-supported mechanism claim."
        )
    reason = str(claim.get("reason_claim_cannot_exceed_current_level", "")).strip()
    if reason:
        return reason
    if missing:
        return "Claim cannot be higher until this evidence is added: " + ", ".join(map(str, missing))
    return "No higher claim level is defined for the current evidence set."


def _ambiguous_result_markdown(design_df: pd.DataFrame) -> str:
    if design_df.empty:
        return (
            "If the result pattern is ambiguous, keep the current claim ceiling, update the "
            "posterior, and add a targeted contrast experiment rather than upgrading the claim."
        )
    next_options = design_df.head(3)[["experiment_id", "name"]].to_dict(orient="records")
    lines = [
        "If the result pattern splits across competing hypotheses, do not upgrade the claim ceiling.",
        "Update the posterior with `ted-mad update`, then choose the highest-ranked remaining contrast experiment:",
    ]
    lines.extend(f"- {row['experiment_id']} {row['name']}" for row in next_options)
    return "\n".join(lines)


def _reviewer_defense_markdown(
    *,
    overconfidence_warning: str,
    robustness: Mapping[str, Any],
    design_rank_stability: pd.DataFrame,
) -> str:
    lines = [
        "- Competing mechanisms are explicit; the report does not only score the favored model.",
        "- Evidence is fused at evidence-family/dependency-group level to reduce double counting.",
        "- The claim ceiling is capped when direct rescue or orthogonal perturbation evidence is missing.",
        "- The next experiment is selected for information gain, claim upgrade, falsification value, cost, and risk.",
        "- Falsification rules are reported before rescue data are observed.",
    ]
    if overconfidence_warning:
        lines.append("- Naive fusion was compared against dependency-aware fusion and flagged for overconfidence.")
    if robustness:
        lines.append("- Posterior and claim robustness summaries are included for prior and evidence-weight perturbations.")
    if not design_rank_stability.empty:
        lines.append("- Active-design ranking stability was stress-tested under cost/risk perturbations.")
    return "\n".join(lines)


def _sections_to_markdown(title: str, sections: Sequence[tuple[str, str]]) -> str:
    lines = [f"# {title}", ""]
    for i, (heading, body) in enumerate(sections, start=1):
        lines.append(f"## {i}. {heading}")
        lines.append("")
        lines.append(body.strip() if body.strip() else "_None._")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def generate_decision_report(
    posterior_bundle: Mapping[str, Any],
    design_bundle: Mapping[str, Any],
    *,
    evidence_contribution: pd.DataFrame | None = None,
    title: str = "TED Mechanism Decision Report",
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate reviewer-facing TED-MAD Mechanism Claim Card artifacts."""

    posterior_df = pd.DataFrame(posterior_bundle["posterior"]).sort_values(
        "posterior", ascending=False
    )
    claim = posterior_bundle.get("claim_ceiling") or design_bundle.get("current_claim_ceiling") or {}
    robustness = posterior_bundle.get("robustness_summary", {}) or {}
    posterior_interval = pd.DataFrame(posterior_bundle.get("posterior_interval", []))
    dominance_frequency = pd.DataFrame(posterior_bundle.get("dominance_frequency", []))
    overconfidence_warning = posterior_bundle.get("overconfidence_warning_markdown", "")
    design_df = pd.DataFrame(design_bundle["ranked_experiments"]).sort_values("rank")
    design_rank_stability = pd.DataFrame(design_bundle.get("design_rank_stability", []))
    contrast_df = pd.DataFrame(design_bundle.get("experiment_contrast_matrix", []))
    readout_panel_df = pd.DataFrame(design_bundle.get("minimal_readout_panel", []))
    pattern_df = pd.DataFrame(design_bundle.get("expected_result_patterns", []))
    leading = posterior_df.iloc[0]
    best = design_df.iloc[0] if not design_df.empty else pd.Series(dtype=object)

    if evidence_contribution is None:
        evidence_contribution = pd.DataFrame()
    if {"hypothesis", "log_likelihood_ratio"}.issubset(evidence_contribution.columns):
        leading_mask = evidence_contribution["hypothesis"] == leading["hypothesis"]
        supporting = evidence_contribution[
            leading_mask & (evidence_contribution["log_likelihood_ratio"] > 0)
        ].sort_values("log_likelihood_ratio", ascending=False)
        opposing = evidence_contribution[
            leading_mask & (evidence_contribution["log_likelihood_ratio"] < 0)
        ].sort_values("log_likelihood_ratio")
    else:
        supporting = pd.DataFrame()
        opposing = pd.DataFrame()

    non_independent = (
        evidence_contribution.groupby("evidence_family")["hypothesis"].count().reset_index()
        if {"evidence_family", "hypothesis"}.issubset(evidence_contribution.columns)
        else pd.DataFrame()
    )

    pattern_best = (
        pattern_df[pattern_df["experiment_id"] == best.get("experiment_id")]
        if not pattern_df.empty and "experiment_id" in pattern_df.columns and not best.empty
        else pd.DataFrame()
    )
    merged_provenance = merge_provenance(
        posterior_bundle.get("provenance", {}),
        design_bundle.get("provenance", {}),
        provenance,
    )

    readouts = str(best.get("readouts", "")).replace(";", ", ") if not best.empty else ""
    best_experiment_id = str(best.get("experiment_id", ""))
    competing = posterior_df[posterior_df["hypothesis"] != leading["hypothesis"]]
    dependency_warning = (
        "Evidence is aggregated by `evidence_family` and `dependency_group`; "
        "rows within a dependency group do not simply get multiplied together.\n\n"
        f"{_markdown_table(non_independent, ['evidence_family', 'hypothesis'])}\n\n"
        f"{overconfidence_warning or '_Naive fusion comparison was not requested for this run._'}"
    )
    best_summary = (
        f"**{best.get('experiment_id', 'NA')} {best.get('name', 'NA')}**\n\n"
        f"Utility: {best.get('utility', float('nan')):.3f}; "
        f"EIG: {best.get('expected_information_gain', float('nan')):.3f}; "
        f"expected claim delta: {best.get('expected_claim_delta', float('nan')):.3f}; "
        f"falsification score: {best.get('falsification_score', float('nan')):.3f}."
    )
    sections = [
        ("Candidate TED Event", _candidate_event_label(posterior_bundle)),
        (
            "Leading Hypothesis",
            f"**{leading['hypothesis']} {leading.get('label', '')}** with posterior **{leading['posterior']:.3f}**.",
        ),
        (
            "Competing Hypotheses",
            _markdown_table(
                competing,
                ["hypothesis", "label", "prior", "posterior", "log_likelihood_ratio_total"],
                max_rows=12,
            ),
        ),
        (
            "Posterior Distribution",
            _markdown_table(
                posterior_df,
                ["hypothesis", "label", "prior", "posterior", "log_likelihood_ratio_total"],
                max_rows=12,
            ),
        ),
        (
            "Evidence-Family Contribution",
            "Supporting leading hypothesis:\n\n"
            + _markdown_table(supporting, ["evidence_family", "log_likelihood_ratio", "likelihood_ratio"])
            + "\n\nEvidence against leading hypothesis:\n\n"
            + _markdown_table(opposing, ["evidence_family", "log_likelihood_ratio", "likelihood_ratio"]),
        ),
        ("Evidence Dependency Warning", dependency_warning),
        (
            "Current Claim Ceiling",
            f"**{claim.get('current_level', 'not assigned')}**: {claim.get('current_claim', 'not assigned')}.",
        ),
        (
            "Why The Claim Cannot Be Higher",
            _claim_limitation_sentence(claim)
            + "\n\nMissing evidence for next level: "
            + (", ".join(claim.get("missing_evidence_for_next_level", [])) or "none")
            + ".",
        ),
        ("Next Best Experiment", best_summary),
        (
            "Minimal Readout Panel",
            _minimal_panel_markdown(readout_panel_df, best_experiment_id)
            if best_experiment_id
            else (readouts or "_No minimal readout panel supplied._"),
        ),
        (
            "Expected Result Pattern",
            _markdown_table(pattern_best, ["hypothesis", "readout", "expected_pattern"], max_rows=60),
        ),
        (
            "Falsification Rule",
            design_bundle.get("falsification_rules_markdown")
            or "_No explicit falsification rule supplied._",
        ),
        ("Ambiguous-Result Handling", _ambiguous_result_markdown(design_df)),
        (
            "Sensitivity Summary",
            "Posterior and claim robustness:\n\n"
            + _robustness_markdown(robustness, posterior_interval, dominance_frequency)
            + "\n\nActive-design robustness:\n\n"
            + _design_robustness_markdown(design_rank_stability),
        ),
        (
            "Reviewer Defense Notes",
            _reviewer_defense_markdown(
                overconfidence_warning=overconfidence_warning,
                robustness=robustness,
                design_rank_stability=design_rank_stability,
            )
            + "\n\nProvenance:\n\n"
            + _provenance_markdown(merged_provenance),
        ),
    ]

    report = _sections_to_markdown("Mechanism Claim Card", sections)
    card = report
    return {
        "report_markdown": report,
        "claim_card_markdown": card,
        "claim_card_sections": sections,
        "plot_data": {
            "posterior": posterior_df,
            "posterior_interval": posterior_interval,
            "evidence_contribution": evidence_contribution,
            "leave_one_evidence_family_out_summary": pd.DataFrame(
                posterior_bundle.get("leave_one_evidence_family_out_summary", [])
            ),
            "ranked_experiments": design_df,
            "experiment_contrast_matrix": contrast_df,
            "expected_result_patterns": pattern_df,
            "current_claim_ceiling": claim,
        },
        "provenance": merged_provenance,
    }


def _write_pdf_claim_card(path: Path, markdown_text: str) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception:
        return

    lines = []
    for line in markdown_text.replace("**", "").splitlines():
        if line.startswith("# "):
            lines.append(line[2:])
        elif line.strip():
            lines.extend(textwrap.wrap(line, width=88) or [""])
        else:
            lines.append("")

    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("white")
        y = 0.95
        for i, line in enumerate(lines[:42]):
            size = 16 if i == 0 else 10
            weight = "bold" if i == 0 else "normal"
            fig.text(0.08, y, line, ha="left", va="top", fontsize=size, weight=weight)
            y -= 0.045 if i == 0 else 0.032
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _plot_barh(
    path: Path,
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str,
    xlabel: str,
    colors: Sequence[str] | None = None,
    xerr: Sequence[Sequence[float]] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if not labels:
        return
    fig, ax = plt.subplots(figsize=(8, max(3.2, 0.42 * len(labels) + 1.8)))
    y = np.arange(len(labels))
    ax.barh(y, values, color=list(colors or ["#4C78A8"] * len(labels)), xerr=xerr)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_posterior_figure(plot_data: Mapping[str, Any], path: Path) -> None:
    posterior = pd.DataFrame(plot_data.get("posterior", []))
    if posterior.empty or "posterior" not in posterior.columns:
        return
    posterior = posterior.sort_values("posterior", ascending=True)
    labels = posterior["hypothesis"].astype(str).tolist()
    values = posterior["posterior"].astype(float).tolist()
    intervals = pd.DataFrame(plot_data.get("posterior_interval", []))
    xerr = None
    if not intervals.empty and {"hypothesis", "posterior_q025", "posterior_q975"}.issubset(intervals.columns):
        by_hyp = intervals.set_index("hypothesis")
        lower = []
        upper = []
        for _, row in posterior.iterrows():
            hyp = row["hypothesis"]
            center = float(row["posterior"])
            if hyp in by_hyp.index:
                lower.append(max(0.0, center - float(by_hyp.loc[hyp, "posterior_q025"])))
                upper.append(max(0.0, float(by_hyp.loc[hyp, "posterior_q975"]) - center))
            else:
                lower.append(0.0)
                upper.append(0.0)
        xerr = [lower, upper]
    colors = ["#2F855A" if value == max(values) else "#6B7280" for value in values]
    _plot_barh(
        path,
        labels,
        values,
        title="Figure A. Hypothesis posterior with sensitivity interval",
        xlabel="Posterior probability",
        colors=colors,
        xerr=xerr,
    )


def _save_evidence_waterfall(plot_data: Mapping[str, Any], path: Path) -> None:
    evidence = pd.DataFrame(plot_data.get("evidence_contribution", []))
    posterior = pd.DataFrame(plot_data.get("posterior", []))
    if evidence.empty or posterior.empty or "log_likelihood_ratio" not in evidence.columns:
        return
    leading = str(posterior.sort_values("posterior", ascending=False).iloc[0]["hypothesis"])
    use = evidence[evidence["hypothesis"] == leading].sort_values("log_likelihood_ratio")
    if use.empty:
        return
    values = use["log_likelihood_ratio"].astype(float).tolist()
    colors = ["#2F855A" if value >= 0 else "#C2410C" for value in values]
    _plot_barh(
        path,
        use["evidence_family"].astype(str).tolist(),
        values,
        title="Evidence-family contribution waterfall",
        xlabel="Log likelihood ratio for leading hypothesis",
        colors=colors,
    )


def _save_lofo_figure(plot_data: Mapping[str, Any], path: Path) -> None:
    lofo = pd.DataFrame(plot_data.get("leave_one_evidence_family_out_summary", []))
    if lofo.empty or "family_removed" not in lofo.columns:
        return
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    metrics = [
        col
        for col in ("posterior_drop_for_baseline_leading", "claim_level_delta", "leading_changed")
        if col in lofo.columns
    ]
    if not metrics:
        return
    matrix = lofo[metrics].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(8, max(3.2, 0.35 * len(lofo) + 1.8)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r")
    ax.set_title("Figure B. Evidence-family ablation")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(lofo)))
    ax.set_yticklabels(lofo["family_removed"].astype(str))
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_experiment_ranking(plot_data: Mapping[str, Any], path: Path) -> None:
    design = pd.DataFrame(plot_data.get("ranked_experiments", []))
    if design.empty or "utility" not in design.columns:
        return
    use = design.sort_values("utility", ascending=True)
    _plot_barh(
        path,
        use["experiment_id"].astype(str).tolist(),
        use["utility"].astype(float).tolist(),
        title="Experiment ranking by active design utility",
        xlabel="Utility",
        colors=["#805AD5"] * len(use),
    )


def _label_to_score(value: Any) -> float:
    text = str(value).lower()
    if text in {"true", "yes"}:
        return 1.0
    if "strong" in text or "high" in text:
        return 1.0
    if "moderate" in text or "medium" in text or "partial" in text:
        return 0.6
    if "weak" in text or "low" in text:
        return 0.3
    if "none" in text or "false" in text or "no" == text:
        return 0.0
    return _rescue_strength(value)


def _save_contrast_matrix(plot_data: Mapping[str, Any], path: Path) -> None:
    contrast = pd.DataFrame(plot_data.get("experiment_contrast_matrix", []))
    if contrast.empty or "experiment_id" not in contrast.columns:
        return
    label_cols = [
        col
        for col in contrast.columns
        if (
            "_vs_" in col
            and not col.endswith("_score")
        )
        or col == "claim_upgrade"
        or col.startswith("falsifies_")
    ]
    if not label_cols:
        return
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    matrix = contrast[label_cols].apply(lambda col: col.map(_label_to_score)).to_numpy()
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(label_cols) + 3), max(3.2, 0.4 * len(contrast) + 1.8)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_title("Figure C. Active rescue design matrix")
    ax.set_xticks(np.arange(len(label_cols)))
    ax.set_xticklabels(label_cols, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(contrast)))
    ax.set_yticklabels(contrast["experiment_id"].astype(str))
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_expected_pattern_matrix(plot_data: Mapping[str, Any], path: Path) -> None:
    patterns = pd.DataFrame(plot_data.get("expected_result_patterns", []))
    ranked = pd.DataFrame(plot_data.get("ranked_experiments", []))
    if patterns.empty or ranked.empty:
        return
    best_id = str(ranked.sort_values("rank").iloc[0]["experiment_id"])
    use = patterns[patterns["experiment_id"] == best_id]
    if use.empty:
        return
    pivot = use.pivot_table(
        index="hypothesis",
        columns="readout",
        values="expected_pattern",
        aggfunc="first",
    )
    matrix = pivot.apply(lambda col: col.map(_label_to_score)).to_numpy()
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(pivot.columns) + 3), max(3.2, 0.38 * len(pivot) + 1.8)))
    im = ax.imshow(matrix, aspect="auto", cmap="PuBuGn", vmin=0, vmax=1)
    ax.set_title("Hypothesis x readout expected pattern matrix")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(col) for col in pivot.columns], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(idx) for idx in pivot.index])
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_claim_ladder(plot_data: Mapping[str, Any], path: Path) -> None:
    claim = plot_data.get("current_claim_ceiling", {}) or {}
    current = _to_float(claim.get("current_level_numeric"), 0.0)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    levels = [float(item["level"]) for item in CLAIM_LEVELS]
    labels = [str(item["label"]) for item in CLAIM_LEVELS]
    colors = ["#2F855A" if level <= current else "#CBD5E1" for level in levels]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.scatter(levels, [1] * len(levels), s=260, c=colors, zorder=3)
    ax.plot(levels, [1] * len(levels), color="#94A3B8", zorder=1)
    for level, label in zip(levels, labels):
        ax.text(level, 0.86, label, rotation=35, ha="right", va="top", fontsize=8)
    ax.set_ylim(0.6, 1.25)
    ax.set_yticks([])
    ax.set_xlabel("Claim level")
    ax.set_title("Claim ceiling ladder")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_dependency_graph(plot_data: Mapping[str, Any], path: Path) -> None:
    evidence = pd.DataFrame(plot_data.get("evidence_contribution", []))
    if evidence.empty or not {"evidence_family", "hypothesis", "log_likelihood_ratio"}.issubset(evidence.columns):
        return
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    families = sorted(evidence["evidence_family"].astype(str).unique())
    hypotheses = sorted(evidence["hypothesis"].astype(str).unique())
    fig, ax = plt.subplots(figsize=(9, max(4, 0.34 * max(len(families), len(hypotheses)) + 2)))
    fy = {family: i for i, family in enumerate(families)}
    hy = {hyp: i for i, hyp in enumerate(hypotheses)}
    for family, y in fy.items():
        ax.scatter(0, y, s=80, color="#4C78A8")
        ax.text(-0.03, y, family, ha="right", va="center", fontsize=8)
    for hyp, y in hy.items():
        ax.scatter(1, y, s=80, color="#F58518")
        ax.text(1.03, y, hyp, ha="left", va="center", fontsize=8)
    for row in evidence.to_dict(orient="records"):
        weight = abs(float(row.get("log_likelihood_ratio", 0.0)))
        if weight < 0.05:
            continue
        color = "#2F855A" if float(row["log_likelihood_ratio"]) >= 0 else "#C2410C"
        ax.plot(
            [0, 1],
            [fy[str(row["evidence_family"])], hy[str(row["hypothesis"])]],
            color=color,
            alpha=min(0.75, 0.2 + weight / 2.5),
            linewidth=0.8 + weight,
        )
    ax.set_xlim(-0.45, 1.45)
    ax.set_axis_off()
    ax.set_title("Dependency-aware evidence graph")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_report_figures(report: Mapping[str, Any], out: Path) -> dict[str, Path]:
    plot_data = report.get("plot_data", {}) or {}
    specs = [
        ("figure_a_hypothesis_posterior_sensitivity", _save_posterior_figure, "figure_a_hypothesis_posterior_sensitivity.png"),
        ("evidence_contribution_waterfall", _save_evidence_waterfall, "evidence_contribution_waterfall.png"),
        ("figure_b_evidence_family_ablation", _save_lofo_figure, "figure_b_evidence_family_ablation.png"),
        ("experiment_ranking_plot", _save_experiment_ranking, "experiment_ranking_plot.png"),
        ("figure_c_active_rescue_design_matrix", _save_contrast_matrix, "figure_c_active_rescue_design_matrix.png"),
        ("hypothesis_readout_expected_pattern_matrix", _save_expected_pattern_matrix, "hypothesis_readout_expected_pattern_matrix.png"),
        ("claim_ceiling_ladder", _save_claim_ladder, "claim_ceiling_ladder.png"),
        ("dependency_graph", _save_dependency_graph, "dependency_graph.png"),
    ]
    figure_paths: dict[str, Path] = {}
    for key, writer, filename in specs:
        path = out / filename
        try:
            writer(plot_data, path)
        except Exception:
            continue
        if path.exists():
            figure_paths[key] = path
    return figure_paths


def _figure_links_markdown(figure_paths: Mapping[str, Path]) -> str:
    if not figure_paths:
        return ""
    lines = ["", "## Report Figures", ""]
    for key, path in figure_paths.items():
        title = key.replace("_", " ").title()
        lines.append(f"![{title}]({path.name})")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _claim_card_html(report: Mapping[str, Any], figure_paths: Mapping[str, Path]) -> str:
    sections = report.get("claim_card_sections", []) or []
    figure_html = "\n".join(
        f'<figure><img src="{html.escape(path.name)}" alt="{html.escape(key)}">'
        f"<figcaption>{html.escape(key.replace('_', ' ').title())}</figcaption></figure>"
        for key, path in figure_paths.items()
    )
    section_html = "\n".join(
        "<section>"
        f"<h2>{i}. {html.escape(str(title))}</h2>"
        f"<pre>{html.escape(str(body).strip())}</pre>"
        "</section>"
        for i, (title, body) in enumerate(sections, start=1)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Mechanism Claim Card</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #111827; line-height: 1.45; }}
    h1 {{ margin-bottom: 0.2rem; }}
    section {{ border-top: 1px solid #E5E7EB; padding: 18px 0; }}
    pre {{ white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; background: #F8FAFC; padding: 12px; border-radius: 6px; overflow-x: auto; }}
    figure {{ margin: 24px 0; }}
    img {{ max-width: 100%; border: 1px solid #E5E7EB; border-radius: 6px; }}
    figcaption {{ color: #4B5563; font-size: 0.9rem; margin-top: 6px; }}
  </style>
</head>
<body>
  <h1>Mechanism Claim Card</h1>
  <p>Reviewer-facing TED-MAD/ARD mechanism adjudication and active rescue design report.</p>
  <h2>Figures</h2>
  {figure_html or '<p>No figures were generated for this run.</p>'}
  {section_html}
</body>
</html>
"""


def write_report_outputs(
    report: Mapping[str, Any],
    outdir: str | Path,
    *,
    formats: Sequence[str] | None = None,
    write_pdf: bool = True,
) -> dict[str, str]:
    out = _ensure_dir(outdir)
    requested = set(formats or ["markdown"])
    if formats is None and write_pdf:
        requested.add("pdf")
    if not write_pdf:
        requested.discard("pdf")
    figure_paths = _save_report_figures(report, out)
    figure_markdown = _figure_links_markdown(figure_paths)
    paths = {
        "report_markdown": out / "ted_mechanism_decision_report.md",
        "claim_card_markdown": out / "reviewer_claim_card.md",
        "claim_card_html": out / "mechanism_claim_card.html",
        "provenance": out / "provenance.json",
        "provenance_yaml": out / "provenance.yaml",
        "claim_card_pdf": out / "reviewer_claim_card.pdf",
    }
    markdown_text = str(report["report_markdown"]) + figure_markdown
    claim_card_markdown = str(report["claim_card_markdown"]) + figure_markdown
    if "markdown" in requested:
        paths["report_markdown"].write_text(markdown_text, encoding="utf-8")
        paths["claim_card_markdown"].write_text(claim_card_markdown, encoding="utf-8")
    if "html" in requested:
        paths["claim_card_html"].write_text(
            _claim_card_html(report, figure_paths),
            encoding="utf-8",
        )
    paths["provenance"].write_text(
        json.dumps(report.get("provenance", {}), indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    _write_yaml(paths["provenance_yaml"], report.get("provenance", {}))
    if "pdf" in requested:
        _write_pdf_claim_card(paths["claim_card_pdf"], report["claim_card_markdown"])
    output_paths = {key: str(path) for key, path in paths.items() if path.exists()}
    output_paths.update({key: str(path) for key, path in figure_paths.items()})
    return output_paths


def _posterior_rows_to_dict(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    values = {str(row["hypothesis"]): _to_float(row.get("posterior"), 0.0) for row in rows}
    total = sum(values.values())
    if total <= 0:
        raise ValueError("posterior probabilities must sum to a positive value")
    return {hyp: value / total for hyp, value in values.items()}


def _pattern_match_score(expected: Any, observed: Any) -> float:
    if expected is None or observed is None:
        return 0.0
    diff = abs(_rescue_strength(expected) - _rescue_strength(observed))
    return _clip(1.0 - diff, -1.0, 1.0)


def _observed_result_likelihoods(
    experiment_patterns: Mapping[str, Any],
    observed_results: Mapping[str, Any],
    *,
    quality: float = 1.0,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    quality = _clip(float(quality), 0.0, 1.0)
    log_lrs: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for hyp, readout_pattern in experiment_patterns.items():
        if not isinstance(readout_pattern, Mapping):
            continue
        matches = []
        for readout, observed in observed_results.items():
            expected = readout_pattern.get(readout)
            score = _pattern_match_score(expected, observed)
            if expected is None:
                continue
            matches.append(score)
            rows.append(
                {
                    "hypothesis": hyp,
                    "readout": readout,
                    "expected_pattern": expected,
                    "observed_pattern": observed,
                    "match_score": float(score),
                }
            )
        if matches:
            centered = [(_clip(score, 0.0, 1.0) - 0.5) * 2.0 for score in matches]
            cumulative_match = _clip(float(np.sum(centered)), -2.0, 2.0)
        else:
            cumulative_match = 0.0
        log_lrs[str(hyp)] = math.log(2.0) * quality * cumulative_match
    return log_lrs, rows


def _posterior_update_from_log_lrs(
    prior: Mapping[str, float],
    log_lrs: Mapping[str, float],
) -> dict[str, float]:
    hypotheses = list(prior)
    log_values = np.asarray(
        [math.log(max(prior[hyp], 1e-12)) + _to_float(log_lrs.get(hyp), 0.0) for hyp in hypotheses],
        dtype=float,
    )
    log_values = log_values - log_values.max()
    probs = np.exp(log_values)
    probs = probs / probs.sum()
    return {hyp: float(prob) for hyp, prob in zip(hypotheses, probs)}


def interpret_rescue_results(
    posterior_bundle: Mapping[str, Any],
    design_bundle: Mapping[str, Any],
    observed_input: Mapping[str, Any],
    *,
    strict: bool = False,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Update posterior and claim ceiling from observed rescue readout patterns."""

    if strict:
        validate_observed_rescue(observed_input)
    experiment_id = str(observed_input["experiment_id"])
    observed_results = dict(observed_input.get("observed_results", {}))
    quality = _to_float(observed_input.get("quality"), 1.0)
    patterns_df = pd.DataFrame(design_bundle.get("expected_result_patterns", []))
    if patterns_df.empty:
        raise ValueError("design bundle does not contain expected_result_patterns")
    use = patterns_df[patterns_df["experiment_id"] == experiment_id]
    if use.empty:
        raise ValueError(f"experiment_id '{experiment_id}' is not present in design bundle")
    experiment_patterns: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in use.to_dict(orient="records"):
        experiment_patterns[str(row["hypothesis"])][str(row["readout"])] = row["expected_pattern"]

    prior = _posterior_rows_to_dict(posterior_bundle["posterior"])
    log_lrs, match_rows = _observed_result_likelihoods(
        experiment_patterns,
        observed_results,
        quality=quality,
    )
    updated = _posterior_update_from_log_lrs(prior, log_lrs)
    posterior_rows = []
    prior_rows = {str(row["hypothesis"]): row for row in posterior_bundle["posterior"]}
    for hyp, posterior in sorted(updated.items(), key=lambda item: item[1], reverse=True):
        prior_row = prior_rows.get(hyp, {})
        posterior_rows.append(
            {
                "hypothesis": hyp,
                "label": prior_row.get("label", hyp),
                "prior_posterior": float(prior.get(hyp, 0.0)),
                "observed_log_likelihood_ratio": float(log_lrs.get(hyp, 0.0)),
                "updated_posterior": float(posterior),
                "posterior_delta": float(posterior - prior.get(hyp, 0.0)),
            }
        )
    updated_df = pd.DataFrame(posterior_rows)
    match_df = pd.DataFrame(match_rows)

    previous_claim = posterior_bundle.get("claim_ceiling", {})
    leading_before = str(pd.DataFrame(posterior_bundle["posterior"]).iloc[0]["hypothesis"])
    leading_after = str(updated_df.iloc[0]["hypothesis"])
    matched_score = (
        float(match_df[match_df["hypothesis"] == leading_after]["match_score"].mean())
        if not match_df.empty
        else 0.0
    )
    current_level = _to_float(previous_claim.get("current_level_numeric"), 1.0)
    claim_level = current_level
    claim_label = previous_claim.get("current_level", "not assigned")
    if leading_after == leading_before and matched_score >= 0.65:
        claim_level = max(current_level, 4.0)
        claim_label = "L4 rescue-supported mechanism"
    elif leading_after != leading_before or matched_score < 0.35:
        claim_level = min(current_level, 3.0)
        claim_label = "L3 adjusted causal-compatible event"

    interpretation_lines = []
    if leading_after == leading_before and matched_score >= 0.65:
        interpretation_lines.append("Observed rescue pattern supports the leading mechanism.")
    elif leading_after != leading_before:
        interpretation_lines.append(
            f"Observed rescue pattern shifts the leading mechanism from {leading_before} to {leading_after}."
        )
    else:
        interpretation_lines.append("Observed rescue pattern is mixed or weak for the leading mechanism.")
    for row in updated_df.head(3).to_dict(orient="records"):
        direction = "increases" if row["posterior_delta"] > 0 else "decreases"
        interpretation_lines.append(
            f"{row['hypothesis']} {direction} by {abs(row['posterior_delta']):.3f}."
        )

    claim = {
        "previous_level_numeric": current_level,
        "updated_level_numeric": claim_level,
        "previous_level": previous_claim.get("current_level", "not assigned"),
        "updated_level": claim_label,
        "leading_hypothesis_before": leading_before,
        "leading_hypothesis_after": leading_after,
        "leading_pattern_match_score": matched_score,
    }
    merged_provenance = merge_provenance(
        posterior_bundle.get("provenance", {}),
        design_bundle.get("provenance", {}),
        provenance,
    )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "observed_results": observed_results,
        "updated_posterior": updated_df,
        "result_pattern_match": match_df,
        "updated_claim_ceiling": claim,
        "interpretation_markdown": "\n".join(f"- {line}" for line in interpretation_lines),
        "provenance": merged_provenance,
    }


def write_interpretation_outputs(result: Mapping[str, Any], outdir: str | Path) -> dict[str, str]:
    out = _ensure_dir(outdir)
    paths = {
        "updated_posterior": out / "updated_posterior.csv",
        "result_pattern_match": out / "result_pattern_match.csv",
        "updated_claim_ceiling": out / "updated_claim_ceiling.json",
        "interpretation": out / "interpretation.md",
        "provenance": out / "provenance.json",
        "interpretation_bundle": out / "interpretation.json",
    }
    result["updated_posterior"].to_csv(paths["updated_posterior"], index=False)
    result["result_pattern_match"].to_csv(paths["result_pattern_match"], index=False)
    paths["updated_claim_ceiling"].write_text(
        json.dumps(result["updated_claim_ceiling"], indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    paths["interpretation"].write_text(result["interpretation_markdown"], encoding="utf-8")
    paths["provenance"].write_text(
        json.dumps(result.get("provenance", {}), indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    bundle = {
        "created_at": result["created_at"],
        "experiment_id": result["experiment_id"],
        "observed_results": result["observed_results"],
        "updated_posterior": result["updated_posterior"].to_dict(orient="records"),
        "result_pattern_match": result["result_pattern_match"].to_dict(orient="records"),
        "updated_claim_ceiling": result["updated_claim_ceiling"],
        "interpretation_markdown": result["interpretation_markdown"],
        "provenance": result.get("provenance", {}),
    }
    paths["interpretation_bundle"].write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return {key: str(path) for key, path in paths.items() if path.exists()}


def load_posterior_bundle(path: str | Path) -> dict[str, Any]:
    return _read_posterior_bundle(path)


def load_design_bundle(path: str | Path) -> dict[str, Any]:
    return _read_design_bundle(path)
