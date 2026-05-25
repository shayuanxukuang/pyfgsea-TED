# TED-Development Output Schema

## Phase 4.5 Adversarial Benchmark

Core files:

- `phase4_5_noise_sweep.tsv`
- `phase4_5_dropout_sweep.tsv`
- `phase4_5_block_imbalance_sweep.tsv`
- `phase4_5_missing_timepoint_sweep.tsv`
- `phase4_5_batch_confounding_sweep.tsv`
- `phase4_5_rare_lineage_sweep.tsv`
- `phase4_5_performance_ci.tsv`
- `phase4_5_failure_modes.tsv`

Important columns:

- `sweep_factor`, `sweep_value`: adversarial condition being varied.
- `method`: TED or comparison method.
- `*_mean`, `*_ci95_low`, `*_ci95_high`: replicate mean and 95% confidence interval.
- `TED_adversarial_behavior`: whether TED passed, downgraded, or needs review.
- `failure_modes`: human-readable failure flags.

## Phase 4.6 Serious Baseline Suite

Core files:

- `phase4_6_baseline_task_matrix.tsv`
- `phase4_6_baseline_metric_table.tsv`
- `phase4_6_baseline_failure_modes.tsv`
- `phase4_6_method_capability_coverage.tsv`

Important columns:

- `closest_baseline`: the most relevant task-specific existing-method proxy.
- `TED_additional_object`: what TED contributes beyond the baseline output.
- `event_type_accuracy_mean`: whether the method distinguishes delay, loss, artifact, lag, or lineage mode.
- `overclaim_rate_mean`: estimated rate of stronger-than-supported claims.
- `coverage_fraction`: fraction of TED object capabilities natively covered by the method.

## Reproducibility Outputs

- `main_table_manifest.tsv`: source-to-main-table map.
- `main_figure_manifest.tsv`: source-to-main-figure map.

These manifests are intended as the audit trail between raw Phase 4 outputs and manuscript-ready artifacts.
