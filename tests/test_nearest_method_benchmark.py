import numpy as np

from pyfgsea.nearest_method_benchmark import (
    RawCountDesign,
    evaluate_common_task,
    realized_pairwise_overlap,
    score_then_smooth_common_task,
    simulate_raw_count_dataset,
)


def tiny_design(**kwargs):
    values = dict(
        n_blocks=4,
        n_cells=120,
        n_genes=300,
        n_pathways=12,
        pathway_size_min=12,
        pathway_size_max=20,
        true_per_mode=1,
        seed=17,
    )
    values.update(kwargs)
    return RawCountDesign(**values)


def test_raw_count_generator_is_deterministic_and_truth_is_separate():
    left = simulate_raw_count_dataset(tiny_design())
    right = simulate_raw_count_dataset(tiny_design())
    assert left.counts.shape == (120, 300)
    assert (left.counts != right.counts).nnz == 0
    assert "is_dynamic" not in left.cells.columns
    assert set(left.truth["event_mode"]) == {"activation", "suppression", "transient", "null"}
    assert np.isfinite(left.scenario["observed_coordinate_spearman"])


def test_noisy_coordinate_is_imperfect_and_overlap_is_bounded():
    dataset = simulate_raw_count_dataset(tiny_design(coordinate_quality="noisy"))
    rho = dataset.scenario["observed_coordinate_spearman"]
    assert 0.65 < rho < 0.95
    assert realized_pairwise_overlap(dataset.pathways) < 0.40


def test_composition_creates_artifact_targets_without_changing_dynamic_truth():
    dataset = simulate_raw_count_dataset(tiny_design(artifact="composition"))
    targets = dataset.truth[dataset.truth["artifact_target"]]
    assert len(targets) > 0
    assert not targets["is_dynamic"].any()
    assert dataset.cells.groupby("state")["true_time_private"].mean().diff().abs().max() > 0.1


def test_common_task_adapter_and_metrics_use_only_shared_fields():
    dataset = simulate_raw_count_dataset(tiny_design(signal_strength="high"))
    public_cells = dataset.cells.drop(columns=["true_time_private"])
    predictions = score_then_smooth_common_task(dataset.counts, public_cells, dataset.pathways)
    metrics = evaluate_common_task(predictions, dataset.truth)
    assert len(predictions) == 12
    assert predictions["q_value"].isna().all()
    assert not predictions["formal_p_value_available"].any()
    assert 0.0 <= metrics.loc[0, "pathway_level_auprc"] <= 1.0
    assert 0.0 <= metrics.loc[0, "matched_top_k_artifact_false_promotion_rate"] <= 1.0
    assert metrics.loc[0, "formal_fdp_available"] in (False, np.bool_(False))
