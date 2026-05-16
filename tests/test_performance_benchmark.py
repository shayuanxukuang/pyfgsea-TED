from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest


def _load_benchmark_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_ted_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_ted_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_performance_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_ted_performance.py"
    spec = importlib.util.spec_from_file_location("benchmark_ted_performance", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.performance
def test_ted_benchmark_profile_and_metric_schema():
    bench = _load_benchmark_module()
    spec = bench.profile_for("tiny")
    assert spec.cells == 500
    assert spec.pathways == 50
    assert "wall_time" in bench.METRIC_COLUMNS
    assert "event_fdr_time_per_perm" in bench.METRIC_COLUMNS
    assert "bootstrap_time_per_resample" in bench.METRIC_COLUMNS


@pytest.mark.performance
def test_ted_benchmark_core_smoke():
    bench = _load_benchmark_module()
    out = bench.run_benchmark_suite(profile="tiny", suites=["core"], seed=101)
    assert set(bench.METRIC_COLUMNS).issubset(out.columns)
    assert len(out) == 1
    assert out.loc[0, "benchmark_level"] == "core_gsea"
    assert out.loc[0, "status"] == "ok"
    assert out.loc[0, "result_rows"] > 0


@pytest.mark.performance
def test_ted_performance_wrapper_core_smoke():
    perf = _load_performance_module()
    out = perf.run_performance_benchmark(
        profiles=["core"],
        sizes=["tiny"],
        repeats=1,
        seed=103,
        phase_probes=False,
    )
    assert {"ranker_time", "gsea_time", "event_summary_time"}.issubset(out.columns)
    assert len(out) == 1
    assert out.loc[0, "benchmark_level"] == "core_gsea"
    assert out.loc[0, "status"] == "ok"
    assert out.loc[0, "gsea_time"] > 0


@pytest.mark.performance
def test_ted_performance_regression_gate_detects_runtime_regression():
    perf = _load_performance_module()
    current = pd.DataFrame(
        {
            "benchmark_level": ["core_gsea"],
            "profile": ["tiny"],
            "case": ["run_gsea"],
            "cells": [500],
            "genes": [1000],
            "pathways": [50],
            "windows_target": [20],
            "wall_time": [2.0],
            "peak_rss_mb": [120.0],
            "pathway_windows_per_second": [100.0],
            "status": ["ok"],
        }
    )
    baseline = current.copy()
    baseline["wall_time"] = 1.0
    baseline["peak_rss_mb"] = 100.0
    baseline["pathway_windows_per_second"] = 150.0

    regression = perf.compare_to_baseline(
        current,
        baseline,
        runtime_threshold=1.25,
        memory_threshold=1.25,
        throughput_threshold=0.80,
    )

    assert regression.loc[0, "runtime_regression"]
    assert regression.loc[0, "throughput_regression"]
    assert regression.loc[0, "regression_status"] == "regression"
