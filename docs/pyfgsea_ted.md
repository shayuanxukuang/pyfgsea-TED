# PyFgsea-TED: Trajectory Pathway Event Discovery

PyFgsea-TED is the next development direction for PyFgsea. The goal is not only
to draw pathway NES curves along pseudotime, but to discover stable pathway
events in single-cell trajectories, branches, and condition comparisons.

The event object is the central unit:

- pathway activation or suppression
- delayed or accelerated activation between conditions
- transient pathway pulses
- recurrent or biphasic programs
- branch-specific or divergent pathway programs

This positions PyFgsea-TED between classical preranked GSEA and gene-level
trajectory methods. Classical GSEA handles one ranked list; AUCell, UCell,
GSVA, and ssGSEA score pathways per cell or sample; tradeSeq and CellRank are
mostly gene-trend or fate-oriented; Lamian focuses on multi-sample pseudotime
statistics. PyFgsea-TED uses the high-performance fgseaMultilevel-aligned core
to turn trajectory-scale pathway dynamics into testable, summarized,
comparable event tables.

For the broader comparison against tradeSeq, CellRank, AUCell/UCell,
GSVA/ssGSEA, decoupler, irGSEA, SCPA, Lamian, and GSDensity, see
[`method_positioning.md`](method_positioning.md).

## Reliability Layers

The reliability framework is organized in five layers.

1. Input and API checks
   - `validate_inputs(...)` checks pseudotime, gene names, expression source,
     GMT overlap, pathway filtering, group balance, and constant genes.
   - `run_trajectory_gsea(...)` now supports explicit `layer`, `use_raw`,
     `dropna`, and `make_var_names_unique` behavior.

2. Result object consistency
   - `validate_trajectory_result(...)` checks finite ES/NES/p-values, FDR
     ranges, monotonic window times, event timing consistency, event label
     direction, leading-edge membership, and core leading-edge consistency.

3. Statistical invariants
   - Unit tests cover deterministic behavior, pseudotime reversal, constant
     expression behavior, adaptive-window bounds, pseudotime-span bounds,
     monotonic local slope, sparse/dense consistency, cell-weighted rankers,
     multi-ranker consensus, and comparison label permutation behavior.

4. Semi-synthetic truth benchmarks
   - `make_synthetic_trajectory_truth(...)` creates compact AnnData objects
     with known pathway dynamics.
   - `run_synthetic_truth_benchmark(...)` produces ranker-by-truth benchmark
     tables with detection, peak-time error, label accuracy, and random
     background false-positive metrics.
   - `run_trajectory_gsea_grid(...)` supports consensus across window sizes,
     step sizes, rankers, and seeds.
   - `ranker="smooth_slope"` and its alias `ranker="gam_slope"` provide a
     lightweight smoothed-trend ranker inspired by GAM trajectory tools.
   - `gene_set_mode="split_signed"` splits signed resources into positive and
     negative arms for sign-aware event discovery.

5. Event and comparison calibration
   - `run_event_permutation_fdr(...)` permutes pseudotime labels, summarizes
     null pathway events, and adds empirical event-level p-values/FDRs.
   - `estimate_event_fdr(...)` is the high-level API for event discovery. It
     returns a long pathway-by-statistic table with `event_p` and `event_q`.
     It supports `null="pseudotime_within_replicate_permutation"` for
     trajectory event discovery,
     `null="condition_label_permutation_by_replicate"` for case/control event
     differences, and
     `null="branch_label_permutation_within_pseudotime_bins"` for branch
     divergence calibration. `null="pseudotime_permutation"` and
     `null="gene_label_permutation"` remain available for simpler designs.
   - `run_comparison_permutation_calibration(...)` and
     `run_branch_permutation_calibration(...)` shuffle condition or branch
     labels while preserving group sizes, then calibrate `delta_AUC` and
     `delta_peak_time`.
   - `compare_trajectory_gsea(..., mode="pseudobulk")` aggregates cells within
     each sample-window before ranking genes, then calibrates condition events
     by sample-label permutation.
   - `compare_trajectory_gsea(..., mode="mixed_effect")` fits pathway-level
     mixed-effect models on per-sample window NES values and reports
     `mixed_event_p` / `mixed_event_fdr`.
   - The high-level event table reports pathway-level empirical p-values as
     `(1 + # null_stat >= observed_stat) / (1 + n_perm)`, then applies BH
     correction across pathways for each event statistic. `window_q` remains a
     local visualization aid; `event_q` is the trajectory event discovery
     quantity.

6. Biological validation
   - Real-data validation should start with the existing GSE155254 erythroid
     trajectory, the GSE126085 dropout stress-test data, and at least one
     branch-structured trajectory dataset.

## Calibration Examples

## Unified Result Object

`TrajectoryEventResult` is the recommended container once an analysis has more
than one table. It keeps exploratory window curves, event calls, trajectory-wide
event FDR, robustness evidence, diagnostics, and reproducibility metadata
together.

```python
ted = pyfgsea.make_trajectory_event_result(
    adata=adata,
    gmt_path="hallmark.gmt",
    results=res,
    events=events,
    event_fdr=event_table,
    bootstrap=bands,
    consensus=consensus,
    leading_edges=leading_edges,
    comparisons=comparison_table,
    seed=1,
    replicate_key="donor",
    condition_key="condition",
)

ted.metadata["calibration_status"]
ted.diagnostics
ted.summary()
```

The object uses three fixed evidence layers:

| Layer | Outputs | Role |
| --- | --- | --- |
| window-level | NES, `window_q` / per-window `padj` | Local visualization, not trajectory-wide discovery |
| event-level | onset, peak, duration, AUC, `event_q` | Primary pathway event discovery evidence |
| robustness-level | bootstrap CI, ranker support, seed support, replicate support, leading-edge and baseline agreement | Stability and interpretation support |

**Rule of thumb:** `window_q` is for visualization; `event_q` is for trajectory
event discovery.

## TED-v3 Alignment-Aware Contrasts

TED-v3 upgrades condition comparison from absolute pathway events to
alignment-aware differential events:

```text
S_p^A(t), S_p^B(t) -> phi_A_to_B(t) -> D_p(t) = S_p^A(t) - S_p^B(phi(t))
```

Use anchor pathways that mark stable biological state, not the mechanism being
tested. The returned differential event table separates three common cases:
trajectory speed differences that disappear after alignment, amplitude
rewiring that remains at the same aligned state, and event-order rewiring where
the aligned pathway peak/order still shifts.

```python
tables = pyfgsea.run_aligned_trajectory_contrast(
    window_results,
    condition_col="condition",
    condition_a="basal",
    condition_b="fetal_liver",
    anchor_pathways=[
        "MOUSE_ERYTHROID_HEME_GLOBIN",
        "MOUSE_GATA_KLF_TAL1_REGULON",
    ],
    contrast_threshold=0.5,
    n_permutations=200,
)

pyfgsea.write_aligned_trajectory_contrast(tables, "results_ted_v3/08_tables")
```

The standard TED-v3 outputs are:

| File | Purpose |
| --- | --- |
| `trajectory_alignment_functions.tsv` | monotone state mapping `phi_A_to_B(t)` plus anchor RMSE and quality |
| `alignment_anchor_pathways.tsv` | anchors used for the alignment and raw/aligned anchor agreement |
| `aligned_pathway_score_process.tsv` | `S_A(t)`, aligned `S_B(phi(t))`, and contrast `D_A_minus_B` |
| `differential_event_table.tsv` | connected events detected on the contrast process with `contrast_C`, `contrast_C_plus`, `contrast_C_minus`, effect type, and peak/AUC shifts |
| `differential_event_fdr.tsv` | optional pathway-label permutation event p/q values after alignment |
| `alignment_sensitivity_report.tsv` | linear-time and leave-one-anchor-out sensitivity checks |

The high-level condition API also supports:

```python
comparison = pyfgsea.compare_trajectory_gsea(
    adata,
    gene_sets,
    condition_key="condition",
    mode="aligned_contrast",
    control="basal",
    case="fetal_liver",
    alignment_anchor_pathways=["MOUSE_ERYTHROID_HEME_GLOBIN"],
    n_permutations=200,
)
```

For a no-replicate screen, the high-level `adata` entry point can also do the
full null route: permute cell condition labels within pseudotime bins, rerun
trajectory GSEA, rebuild `D_p(t)`, extract contrast events, and recalibrate
`C(E)`.

```python
comparison = pyfgsea.compare_trajectory_gsea(
    adata,
    gene_sets,
    condition_key="condition",
    mode="aligned_contrast",
    alignment_permutation="condition_within_pseudotime_bins",
    permutation_bins=5,
    n_permutations=100,
)
```

The Tusi erythroid v3 tables can be rebuilt from existing priority TED window
results with:

```bash
python scripts/build_ted_v3_alignment.py --n-permutations 200
```

## TED-v3 Fate-Predictive Events

Branch validation should not stop at post-branch pathway separation. TED-v3 also
supports pre-branch fate-predictive events: pathway score processes are
interpolated onto cells before a split time, then each pathway is scored for how
well its pre-branch signal predicts a future fate label or fate probability.

```python
tables = pyfgsea.run_fate_predictive_events(
    window_results,
    adata=adata,
    pseudotime_key="dpt_pseudotime",
    fate_key="ted_branch",
    split_time=0.45,
    n_permutations=200,
)

pyfgsea.write_fate_predictive_events(tables, "results_ted_v3/08_tables")
```

The standard fate-predictive outputs are:

| File | Purpose |
| --- | --- |
| `prebranch_fate_predictive_events.tsv` | dataset, pathway/module, future fate, event timing, exposure-model theta, CV AUC, balanced accuracy, macro-F1, FPES, event-q, drivers, evidence level |
| `fate_prediction_model_performance.tsv` | per-fate summary of the strongest exposure-based predictive pathway models |
| `prebranch_event_fdr.tsv` | FPES p/q values from future-fate label permutation within pseudotime bins |
| `fate_predictive_driver_genes.tsv` | pre-branch driver genes, leading-edge frequency, and driver score |
| `fate_predictive_leading_edge.tsv` | backward-compatible alias for the fate-predictive driver gene table |

The Paul15 pre-branch fate-predictive tables can be rebuilt with:

```bash
python scripts/build_ted_v3_fate_predictive.py --n-permutations 200
```

## TED-v3 De Novo Dynamic Modules

Pathway databases are useful annotations, not the only discovery space. TED-v3
can now discover dynamic gene modules first, then annotate those modules against
pathway resources:

```python
tables = pyfgsea.discover_dynamic_gene_modules(
    adata,
    gmt_path="erythroid_timing_custom_mouse.gmt",
    pseudotime_key="pseudotime_pba",
    n_modules=6,
    layer="log1p",
    n_permutations=100,
)

pyfgsea.write_dynamic_gene_modules(tables, "results_ted_v3/08_tables")
```

The standard module outputs are:

| File | Purpose |
| --- | --- |
| `dynamic_gene_modules.tsv` | de novo module genes and sparse NMF weights |
| `module_time_profiles.tsv` | module activity trajectories over pseudotime |
| `module_event_table.tsv` | module-level events from module profiles |
| `module_event_fdr.tsv` | screening module-event p/q values from time permutation |
| `module_pathway_annotation.tsv` | pathway enrichment annotations for module genes |
| `module_driver_score.tsv` | module-event driver genes and module-weighted driver score |
| `module_leading_edge_drivers.tsv` | backward-compatible alias for module driver scores |

## TED-v3 Event Graphs

Event lists are rigorous, but graph summaries are easier to interpret
biologically. TED-v3 builds condition-specific event graphs where nodes are
events and directed edges represent stable timing relations
`P(T_source < T_target) >= threshold`. With bootstrap event tables, edge
probabilities are bootstrap estimates; otherwise they are deterministic timing
relations and marked as such.

```python
graph = pyfgsea.build_event_graph(
    events,
    bootstrap_events=boot_events,
    condition_col="condition",
    time_col="peak_time",
    reference="basal",
    query="fetal_liver",
)

pyfgsea.write_event_graph(graph, "results_ted_v3/08_tables")
```

The standard event-graph outputs are:

| File | Purpose |
| --- | --- |
| `event_order_probability_matrix.tsv` | pairwise `P(T_source < T_target)` for every condition |
| `condition_event_graph_edges.tsv` | stable directed event-order edges per condition |
| `event_graph_rewiring.tsv` | node/edge gained, lost, reversed, and shared calls between conditions |
| `event_graph_bootstrap_support.tsv` | edge probability and bootstrap count/support type |

Tusi module and event graph outputs can be rebuilt with:

```bash
python scripts/build_ted_v3_modules_graph.py
```

## TED-v3 Driver Scores

Leading-edge genes are now treated as inputs to a driver-level mechanism score,
not as the final mechanistic claim. TED-v3 computes:

```text
DriverScore_g(E) =
  leading_edge_probability_g(E)
  * abs(peak_effect_g(E))
  * event_specificity_g(E)
  * regulatory_weight_g
```

The default regulatory weighting marks transcription factors, known regulators,
surface receptors/transporters, and enzymes slightly above generic target
genes. Optional regulator-target maps can then aggregate target driver scores
into regulator-level event activity.

```python
drivers = pyfgsea.score_event_drivers(
    event_table,
    leading_edge_table,
    regulator_targets={"GATA1": {"ALAS2", "KLF1", "TFRC"}},
)

pyfgsea.write_event_driver_scores(drivers, "results_ted_v3/08_tables")
```

The standard driver outputs are:

| File | Purpose |
| --- | --- |
| `event_driver_score.tsv` | per event-gene driver score with leading-edge probability, peak effect, specificity, regulatory class, and regulatory weight |
| `event_regulator_activity.tsv` | regulator activity per event from summed target driver scores |
| `event_driver_network.tsv` | regulator-target-event edge table used to compute regulator activity |
| `driver_specificity_report.tsv` | gene-level specificity and top event summaries |

## TED-v3 Cross-Dataset Replication

Single-dataset events are screening observations. TED-v3 now supports event
matching across datasets using pathway/module gene overlap, normalized timing
IoU, score-curve correlation, and leading-edge/driver overlap. Missing
components are skipped and the remaining weights are renormalized.

```python
replication = pyfgsea.match_cross_dataset_events(
    all_events,
    score_process=all_window_scores,
    driver_scores=drivers["event_driver_score"],
    gene_sets="erythroid_timing_custom_mouse.gmt",
    match_threshold=0.55,
)

pyfgsea.write_cross_dataset_replication(replication, "results_ted_v3/08_tables")
```

The standard replication outputs are:

| File | Purpose |
| --- | --- |
| `event_match_matrix.tsv` | all cross-dataset event pairs and component match scores |
| `cross_dataset_event_replication.tsv` | event pairs passing the replication threshold |
| `meta_event_score.tsv` | per-event replicated dataset count, meta score, and evidence level |
| `dataset_event_coverage.tsv` | event labels, observed datasets, and best replication support |

Driver and replication tables can be rebuilt with:

```bash
python scripts/build_ted_v3_drivers_replication.py --match-threshold 0.55
```

## TED-v3 Phenotype-Linked Events

The most interpretable TED question is often supervised: which trajectory
events explain mutation, drug, stress, response, severity, or survival at the
sample/replicate level? TED-v3 now converts each event into a sample-level
burden score:

```text
B_rp = integral over event E of S_rp(t) dt
```

Then it fits phenotype association models per event. Continuous phenotypes use
linear regression, binary phenotypes use logistic regression, and survival
phenotypes use Cox regression when `lifelines` is installed.

```python
phenotype_tables = pyfgsea.associate_phenotype_events(
    event_scores=sample_window_scores,
    events=event_table,
    phenotype=sample_phenotype,
    sample_col="sample_id",
    phenotype_col="response",
    phenotype_type="binary",
)

pyfgsea.write_phenotype_event_association(
    phenotype_tables,
    "results_ted_v3/08_tables",
)
```

The standard phenotype-linked outputs are:

| File | Purpose |
| --- | --- |
| `event_burden_score.tsv` | sample-by-event burden scores over event intervals |
| `phenotype_event_association.tsv` | per-event beta, standard error, statistic, p/q value, effect direction, and model status |
| `phenotype_prediction_performance.tsv` | cross-validated all-event phenotype prediction performance |
| `phenotype_linked_event_report.tsv` | compact phenotype-linked event evidence table |

The plumbing can be smoke-tested with a deterministic synthetic phenotype:

```bash
python scripts/build_ted_v3_phenotype_linked.py --synthetic-demo
```

For real supervised discovery, pass real sample-window scores, event intervals,
and sample phenotype metadata:

```bash
python scripts/build_ted_v3_phenotype_linked.py \
  --event-scores sample_window_scores.tsv \
  --events event_table.tsv \
  --phenotype sample_phenotype.tsv \
  --sample-col sample_id \
  --phenotype-col response \
  --phenotype-type binary
```

## Biological Discovery Score

TED-v3 also provides an event-level discovery score to keep broad, expected
programs from dominating the top of the table:

```text
BDS(E) = Q(E) * R(E) * N(E) * P(E) * D(E) * M(E)
```

where `Q` is event-q support, `R` is robustness, `N` is novelty after generic
pathway penalties, `P` is pre-branch or phenotype predictiveness, `D` is
differential specificity, and `M` is driver/regulator support.

```python
score = pyfgsea.score_biological_discovery(
    events,
    driver_scores=event_driver_score,
    replication=meta_event_score,
    phenotype_report=phenotype_linked_event_report,
)

pyfgsea.write_biological_discovery_score(score, "results_ted_v3/08_tables")
```

The standard output is `biological_discovery_score.tsv`. It reports the six
score components, the generic pathway penalty, replication support, and a
`meaningful_discovery` flag.

```bash
python scripts/build_ted_v3_discovery_score.py
```

```python
event_table = pyfgsea.estimate_event_fdr(
    adata=adata,
    gmt_path="hallmark.gmt",
    pseudotime_key="dpt",
    event_stats=["max_abs_nes", "auc_abs", "longest_run", "peak_sharpness"],
    null="pseudotime_within_replicate_permutation",
    replicate_key="donor",
    ranker="detection_weighted",
    n_perm=200,
    window_size=120,
    step=60,
)
```

```python
comparison_table = pyfgsea.estimate_event_fdr(
    adata=adata,
    gmt_path="hallmark.gmt",
    null="condition_label_permutation_by_replicate",
    condition_key="condition",
    replicate_key="donor",
    control="control",
    case="case",
    event_stats=["delta_AUC", "delta_peak_time"],
    ranker="detection_weighted",
    n_perm=200,
)
```

```python
branch_table = pyfgsea.estimate_event_fdr(
    adata=adata,
    gmt_path="hallmark.gmt",
    null="branch_label_permutation_within_pseudotime_bins",
    branch_key="lineage",
    branch_a="erythroid",
    branch_b="myeloid",
    pseudotime_key="dpt",
    event_stats=["delta_AUC", "delta_peak_time"],
    n_pseudotime_bins=10,
    n_perm=200,
)
```

For exploratory runs, small `n_permutations` values are useful smoke tests. For
publication-grade discovery, increase the permutation count and report both the
window-level FDR and the event/comparison-level empirical FDR.

For experiments with biological replicates, prefer replicate-aware calibration.
`replicate_key` is the public alias for biological sample identity; `sample_key`
remains accepted for backward compatibility.

```python
replicate_events = pyfgsea.compare_trajectory_gsea(
    adata,
    "hallmark.gmt",
    condition_key="genotype",
    replicate_key="donor",
    mode="replicate_aware",
    pseudotime_key="dpt",
    ranker="detection_weighted",
    window_mode="adaptive",
    min_cells_per_replicate=10,
    min_replicates_per_condition=3,
    n_permutations=200,
)
```

This mode uses sample-balanced ranking: within each condition-window it computes
gene statistics per replicate, averages replicate statistics with equal donor
weight, summarizes control/case pathway events, then calibrates
`delta_AUC`, `delta_peak_time`, and `delta_duration` by sample-label
permutation. It reports `n_replicates_control`, `n_replicates_case`,
`replicate_support`, `sample_consistency`, `replicate_aware_p`, and
`replicate_aware_q`. If either group has fewer than three biological
replicates, `calibration_status` is
`descriptive_only_low_replicate_count`.

Pseudobulk differential trajectory GSEA remains available when the direct
question is "which pathways are case-enriched or case-depleted along
pseudotime?":

```python
pseudobulk_events = pyfgsea.compare_trajectory_gsea(
    adata,
    "hallmark.gmt",
    condition_key="genotype",
    replicate_key="donor",
    mode="pseudobulk",
    pseudotime_key="dpt",
    pseudobulk_ranker="t_stat",
    min_cells_per_sample=10,
    min_samples_per_condition=2,
    n_permutations=200,
)
```

```python
mixed_events = pyfgsea.compare_trajectory_gsea(
    adata,
    "hallmark.gmt",
    condition_key="genotype",
    replicate_key="donor",
    mode="mixed_effect",
    pseudotime_key="dpt",
)
```

The equivalent high-level calibration entry points are
`estimate_event_fdr(..., null="sample_label_permutation", replicate_key="donor")`,
`estimate_event_fdr(..., null="pseudobulk_permutation", replicate_key="donor")`,
and `estimate_event_fdr(..., null="mixed_effect", replicate_key="donor")`.

## Weighted And Consensus Examples

`cell_weight_key` lets users supply fate probabilities, lineage assignment
weights, or other non-negative cell weights. The window layout is still defined
by pseudotime, but ranking statistics use weighted in-window and out-of-window
summaries.

```python
weighted = pyfgsea.run_branch_gsea(
    adata,
    "hallmark.gmt",
    branch_key="lineage",
    pseudotime_key="dpt",
    cell_weight_key="erythroid_fate_prob",
    ranker="detection_weighted",
)
```

Experimental graph-aware windows define local trajectory states with pseudotime,
kNN graph proximity, branch purity, and optional fate probabilities. They are
useful for bifurcating trajectories where cells at similar pseudotime can belong
to different lineages.

```python
graph_res = pyfgsea.run_trajectory_gsea(
    adata,
    "hallmark.gmt",
    pseudotime_key="dpt",
    window_mode="graph_adaptive",
    graph_key="connectivities",
    graph_radius=2,
    target_span=0.03,
    min_cells=100,
    max_cells=600,
    branch_key="lineage",
    min_branch_purity=0.75,
    cell_weight_key="erythroid_fate_prob",
    ranker="detection_weighted",
    experimental=True,
)

graph_diagnostics = graph_res.attrs["graph_window_diagnostics"]
```

The result table includes topology diagnostics such as `anchor_pseudotime`,
`effective_n_cells`, `mean_graph_distance`, `branch_purity`, `weight_entropy`,
and `fate_weight_mean`. Windows below `min_branch_purity` are skipped and kept
in `graph_window_diagnostics` with `skip_reason="low_branch_purity"`.

Multi-ranker consensus runs the same trajectory through multiple rankers,
window parameters, and seeds, then summarizes event stability.

```python
consensus = pyfgsea.run_ranker_consensus(
    adata,
    "hallmark.gmt",
    pseudotime_key="dpt",
    rankers=["mean_diff", "detection_weighted", "local_slope", "neighbor_contrast"],
    window_mode="adaptive",
    window_size=120,
    step=60,
)
```

Bootstrap confidence bands quantify curve stability. Use cell-level resampling
for technical stability diagnostics and sample-level resampling when
biological replicates are available.

```python
bands = pyfgsea.bootstrap_trajectory_gsea(
    adata,
    "hallmark.gmt",
    pseudotime_key="dpt",
    ranker="detection_weighted",
    window_mode="adaptive",
    resample="samples",
    sample_key="donor",
    n_boot=100,
)
```

Signed resources can be supplied as dictionary-valued gene sets. In
`split_signed` mode, positive and negative arms are tested separately.

```python
regulons = {
    "GATA1_regulon": {"Klf1": 1.0, "Alas2": 1.0, "Spi1": -1.0}
}
signed = pyfgsea.run_trajectory_gsea(
    adata,
    regulons,
    pseudotime_key="dpt",
    gene_set_mode="split_signed",
    ranker="smooth_slope",
)
```

For same-pseudotime branch divergence, `branch_contrast` compares each target
branch window against cells from the other branch at matched pseudotime.

```python
branch = pyfgsea.run_branch_gsea(
    adata,
    "hallmark.gmt",
    branch_key="lineage",
    branches=["erythroid", "myeloid"],
    ranker="branch_contrast",
    pseudotime_key="dpt",
)
```

## Score-Then-Smooth Baselines

Score-then-smooth methods are baseline evidence, not the TED core method. They
ask whether a pathway event discovered by rolling-window fgsea is also visible
after first scoring every cell or sample and then smoothing those scores along
pseudotime.

```python
baseline = pyfgsea.run_score_then_smooth_baseline(
    adata,
    gene_sets="hallmark.gmt",
    pseudotime_key="dpt",
    method="rank_auc",       # rank_auc, mean_zscore, decoupler_ulm, gsva, ssgsea
    smoother="rolling",      # rolling, lowess, spline
    window_mode="adaptive",
    min_cells=100,
    max_cells=600,
    target_span=0.03,
)
baseline_events = pyfgsea.summarize_events(baseline)
```

The baseline output is intentionally compatible with TED event summaries:
`Pathway`, `window_id`, `pt_mid` / `window_midpoint`, `activity_score`,
`activity_z`, and `NES` where `NES = activity_z`.

```python
baseline_check = pyfgsea.compare_event_tables(
    pyfgsea_events,
    baseline_events,
    left_name="PyFgsea-TED",
    right_name="rank_auc_smooth",
)
```

The comparison table reports `event_label_agreement`, `peak_time_delta`,
`AUC_correlation`, `top_event_overlap`,
`false_positive_under_random_sets`, and `runtime`. Agreement with a
rank-based score-then-smooth baseline is useful external consistency evidence.
If an event is unique to TED, prioritize `event_fdr`, leading-edge stability,
and replicate support before treating it as a discovery.

Replicate-aware condition comparison summarizes events per biological sample,
then calibrates condition differences by permuting sample labels rather than
cell labels.

```python
cmp = pyfgsea.compare_trajectory_gsea(
    adata,
    "hallmark.gmt",
    condition_key="genotype",
    replicate_key="donor",
    mode="replicate_aware",
    pseudotime_key="dpt",
    n_permutations=200,
)
```
