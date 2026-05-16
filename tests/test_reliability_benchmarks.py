import numpy as np
import pandas as pd

import pyfgsea
import pyfgsea.trajectory as traj


def test_reliability_synthetic_truth_benchmark_columns():
    traj.HAS_SCANPY = True
    table = pyfgsea.run_reliability_synthetic_truth_benchmark(
        truth_types=["monotonic_up", "condition_amplitude_loss", "graph_branch_mixing"],
        rankers=["mean_diff"],
        n_cells=80,
        n_genes=40,
        pathway_size=6,
        seed=61,
        n_perm=1,
        calibrate=True,
        window_size=20,
        step=20,
        nperm_nes=8,
        sample_size=6,
    )

    assert {
        "truth_type",
        "ranker",
        "power",
        "false_positive_rate",
        "event_label_accuracy",
        "peak_time_error",
        "onset_error",
        "duration_error",
        "branch_assignment_accuracy",
        "condition_delta_error",
        "ranker_support",
        "event_q_calibration",
        "target_event_confidence_class",
        "background_event_confidence_class",
        "target_event_windows",
        "background_event_windows",
        "raw_power",
        "raw_false_positive_rate",
        "screening_power",
        "screening_false_positive_rate",
        "candidate_power",
        "candidate_false_positive_rate",
        "discovery_power",
        "discovery_false_positive_rate",
        "eligible_for_discovery",
        "discovery_status",
        "gated_power",
        "gated_false_positive_rate",
        "target_gate_reason",
        "background_gate_reason",
        "target_n_counts_sensitivity_flag",
    }.issubset(table.columns)
    assert set(table["truth_type"]) == {
        "monotonic_up",
        "condition_amplitude_loss",
        "graph_branch_mixing",
    }


def test_synthetic_discovery_gate_blocks_single_window_false_positive():
    table = pd.DataFrame(
        [
            {
                "truth_type": "sparse_noise",
                "ranker": "mean_diff",
                "window_mode": "cell_count",
                "event_stat": "max_abs_NES",
                "null_model": "pseudotime_permutation",
                "power": 1.0,
                "false_positive_rate": 1.0,
                "ranker_support": 1.0,
                "seed_support": 1.0,
                "technical_confound_score": 0.0,
                "target_event_q": 0.01,
                "background_event_q": 0.01,
                "target_event_windows": 3,
                "background_event_windows": 1,
                "target_duration": 0.2,
                "background_duration": 0.0,
                "target_event_confidence_class": "multi_window_event",
                "background_event_confidence_class": "single_window_pulse",
            },
            {
                "truth_type": "monotonic_up",
                "ranker": "detection_weighted",
                "window_mode": "adaptive",
                "event_stat": "AUC_abs",
                "null_model": "pseudotime_permutation",
                "power": 1.0,
                "false_positive_rate": 0.0,
                "ranker_support": 1.0,
                "seed_support": 1.0,
                "technical_confound_score": 0.0,
                "target_event_q": 0.01,
                "background_event_q": 0.5,
                "target_event_windows": 3,
                "background_event_windows": 0,
                "target_duration": 0.2,
                "background_duration": 0.0,
                "target_event_confidence_class": "multi_window_event",
                "background_event_confidence_class": "missing",
            },
        ]
    )

    gated = pyfgsea.apply_synthetic_discovery_gate(table, mode="candidate")
    assert gated.loc[0, "target_candidate_pass"]
    assert not gated.loc[0, "background_discovery_pass"]
    assert "single_window_pulse" in gated.loc[0, "background_candidate_gate_reason"]
    assert gated["candidate_false_positive_rate"].mean() == 0.0

    sweep = pyfgsea.apply_synthetic_gate_sweep(table)
    assert {
        "screening_power",
        "candidate_power",
        "discovery_power",
        "screening_false_positive_rate",
        "candidate_false_positive_rate",
        "discovery_false_positive_rate",
    }.issubset(sweep.columns)

    breakdown = pyfgsea.summarize_synthetic_fpr_breakdown(sweep)
    assert {"dimension", "raw_false_positive_rate", "candidate_false_positive_rate"}.issubset(
        breakdown.columns
    )
    assert "background_event_confidence_class" in set(breakdown["dimension"])


def test_synthetic_gate_excludes_uncalibrated_and_tightens_transition_rankers():
    table = pd.DataFrame(
        [
            {
                "truth_type": "branch",
                "ranker": "mean_diff",
                "null_model": "branch_contrast_uncalibrated",
                "target_event_p": 0.01,
                "background_event_p": 0.01,
                "target_event_q": 0.01,
                "background_event_q": 0.01,
                "target_event_windows": 4,
                "background_event_windows": 4,
                "target_event_confidence_class": "multi_window_event",
                "background_event_confidence_class": "multi_window_event",
                "ranker_support": 1.0,
                "seed_support": 1.0,
            },
            {
                "truth_type": "transition",
                "ranker": "neighbor_contrast",
                "null_model": "pseudotime_permutation",
                "target_event_p": 0.01,
                "background_event_p": 0.01,
                "target_event_q": 0.01,
                "background_event_q": 0.01,
                "target_event_windows": 2,
                "background_event_windows": 2,
                "target_event_confidence_class": "multi_window_event",
                "background_event_confidence_class": "multi_window_event",
                "ranker_support": 1.0,
                "seed_support": 1.0,
            },
        ]
    )

    gated = pyfgsea.apply_synthetic_discovery_gate(table, mode="candidate")
    assert not gated.loc[0, "eligible_for_discovery"]
    assert gated.loc[0, "discovery_status"] == "uncalibrated_null_model"
    assert "uncalibrated_null_model" in gated.loc[0, "target_candidate_gate_reason"]
    assert "min_event_windows" in gated.loc[1, "target_candidate_gate_reason"]


def test_synthetic_gate_uses_graded_and_event_local_balance():
    table = pd.DataFrame(
        [
            {
                "truth_type": "condition",
                "ranker": "detection_weighted",
                "null_model": "condition_label_permutation_within_pseudotime_bins",
                "target_event_p": 0.05,
                "background_event_p": 0.50,
                "target_event_q": 0.08,
                "background_event_q": 0.80,
                "target_event_windows": 3,
                "background_event_windows": 0,
                "target_event_confidence_class": "condition_comparison",
                "background_event_confidence_class": "missing",
                "ranker_support": 0.5,
                "seed_support": 0.7,
                "balance_pass_rate": 0.33,
                "median_balance_score": 0.65,
                "target_event_balance_coverage": 0.6,
                "background_event_balance_coverage": np.nan,
            }
        ]
    )

    screening = pyfgsea.apply_synthetic_discovery_gate(table, mode="screening")
    assert screening.loc[0, "target_screening_pass"]

    candidate = pyfgsea.apply_synthetic_discovery_gate(table, mode="candidate")
    assert not candidate.loc[0, "target_candidate_pass"]
    assert "median_balance_score" in candidate.loc[0, "target_candidate_gate_reason"]
    assert "event_balance_coverage" in candidate.loc[0, "target_candidate_gate_reason"]


def test_null_calibration_benchmark_reports_false_positive_controls():
    traj.HAS_SCANPY = True
    table = pyfgsea.run_null_calibration_benchmark(
        nulls=["pseudotime", "condition", "branch"],
        n_perm_values=[1],
        n_cells=80,
        n_genes=40,
        pathway_size=6,
        seed=62,
        window_size=20,
        step=20,
        nperm_nes=8,
        sample_size=6,
    )

    assert {"null", "q_lt_005_rate", "robust_event_count", "event_q_calibration"}.issubset(
        table.columns
    )
    assert set(table["null"]) == {"pseudotime", "condition", "branch"}
    assert (table["minimum_attainable_p"] == 0.5).all()


def test_reliability_ablation_study_table():
    traj.HAS_SCANPY = True
    table = pyfgsea.run_reliability_ablation_study(
        n_cells=80,
        n_genes=40,
        pathway_size=6,
        seed=63,
        n_perm=1,
        window_size=20,
        step=20,
        nperm_nes=8,
        sample_size=6,
    )

    assert {"Module added", "Failure mode reduced", "Evidence"}.issubset(table.columns)
    assert {"detection_weighted", "event_fdr", "graph_adaptive"}.issubset(
        set(table["Module added"])
    )
