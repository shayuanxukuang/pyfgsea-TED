# TED-Development Output Schema

## TED event report v2: event support plus parallel evidence

`pyfgsea/schemas/ted_event_report_v2.schema.json` replaces the single legacy
ladder and horizontal V projection with three explicit components:

- `event_support_code`: `E0` (unsupported, non-estimable,
  non-identifiable, artifact-dominated, or missing required design), `E1`
  (statistically supported event), or `E2` (block-robust and
  mode-identifiable event). E0 is a fail-closed support state; it must not be
  interpreted as evidence that no biological event exists.
- `e0_reason_code`: one stable reason is required for every E0 row:
  `E0_not_supported`, `E0_not_estimable`, `E0_not_identifiable`,
  `E0_artifact_dominated`, or `E0_missing_required_design`. E1 and E2 rows
  must leave this field null.
- `event_test_status`: `not_run`, `run_not_supported`, or `run_supported`.
  This separates an unavailable test from a valid test that did not support an
  event.
- `event_q`: null exactly for `event_test_status=not_run`; numeric and finite
  in `[0, 1]` for both `run_not_supported` and `run_supported`.
- `event_q_missing_reason`: required for `not_run` and null after a valid test
  run. Stable reasons are `undeclared_family`, `no_defensible_null`,
  `insufficient_blocks`, `complete_confounding`,
  `insufficient_permutation_resolution`, and `other`.
- `parallel_evidence_records`: independently adjudicated
  `orthogonal_outcome`, `intervention_reversal`, and `matched_rescue` records.
  Each record stores its own status, independence context, control result,
  replication status and stable reason codes. A passed parallel record never
  repairs or upgrades the event E code.
- Event-replication eligibility, test execution and result are distinct
  fields: `event_replication_eligibility_status`,
  `event_replication_test_status`, and `event_replication_status`.
  `outcome_replication_status` is a separate projection of the corresponding
  outcome record; an unavailable outcome is `not_tested`, not failed or passed
  by substituting an RNA readout.
- `supported_interpretation` and
  `unsupported_interpretation_current_evidence`: nonblank statements that make
  the current interpretation boundary auditable.
- `resampling_selection_frequency`: optional conservative selection frequency
  across declared resampling schemes. `discovery_stability_status` must be
  `stable_core` for values at least 0.80, `intermediate` for 0.50 to below
  0.80, and `unstable` below 0.50; unassessed rows use null plus
  `not_evaluable`.
- `upstream_method_agreement`: optional fraction of prespecified upstream
  combinations agreeing on direction and mode. If
  `upstream_disagreement_flag=true`, schema semantics forbid E2 and require an
  E1 ceiling or ambiguity return.

The v2 schema keeps `validation_provenance_code`, `evidence_boundary`,
`evidence_tier`, `claim_ceiling` and `matched_functional_rescue` only as
optional migration fields for archived reports. The legacy V fields do not
drive event support. If present, their internal consistency is still checked;
new reports should use the canonical
`parallel_evidence_record_v1.schema.json` and
`replication_facets_v1.schema.json` contracts. E2 requires
`identifiability_status=identifiable` and a declared block-support method.

For the BIB companion, the schema-valid observed instances are generated
fail-closed from the frozen BNT162b2/GSE171964 status files into
`results/ted_bib_companion_evidence_contract_v1/`. The release verifier
requires `E0`, a passed same-study protein outcome record, failed replication
eligibility, a not-run event test, a not-evaluable event result, and an
untested protein-outcome replication result.

`ted validate --kind event` auto-selects v2 when any v2-only field is present.
This is deliberately fail-closed: a partially migrated table is checked as v2
and reports its missing fields instead of falling back to v1. Use
`--schema-version v1` or `--schema-version v2` to require a specific contract.
Legacy event tables without v2 event-support fields continue to use the
unchanged v1 schema and semantic gate.

## TrajPathMix v1 two-layer output

The default high-level `TrajectoryEventResult.to_tables()` projection is the
functional core. It retains pathway identity, curve magnitude, integrated
effect, calibration, robustness, and diagnostics, and adds
`timing_module_status`, `timing_claim_allowed`, and `timing_failed_gates`.

When timing is `conditional_only`, the default event table excludes onset,
duration, peak/trough location, direction-switch, recurrence, and
transient/sustained label fields. The complete calculation table remains in
`TrajectoryEventResult.events` for preregistered calibration work and can be
requested with `to_tables(include_conditional_timing=True)`; doing so does not
upgrade claim permission.

Timing fields enter the primary projection only when all six project-level
activation gates pass. See `config/trajpathmix_scope_v1.yaml`.

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
