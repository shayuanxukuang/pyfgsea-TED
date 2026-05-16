# Reproducibility Check Report

## Check Summary

| Check | Status | Exit code | Log |
| --- | --- | ---: | --- |
| py_compile_release_scripts | pass | 0 | `data_external\ted_development_reproducibility\check_logs\py_compile_release_scripts.log` |
| ruff_check_release_scope | pass | 0 | `data_external\ted_development_reproducibility\check_logs\ruff_check_release_scope.log` |
| direct_external_baseline_quick | pass | 0 | `data_external\ted_development_reproducibility\check_logs\direct_external_baseline_quick.log` |
| quick_benchmark_suite | pass | 0 | `data_external\ted_development_reproducibility\check_logs\quick_benchmark_suite.log` |
| reproduce_all_main_tables | pass | 0 | `data_external\ted_development_reproducibility\check_logs\reproduce_all_main_tables.log` |
| reproduce_all_main_figures | pass | 0 | `data_external\ted_development_reproducibility\check_logs\reproduce_all_main_figures.log` |
| ruff_check_full_repo_historical_debt | documented_non_gate | 1 | `data_external\ted_development_reproducibility\check_logs\ruff_check.log` |

## Artifact Summary

- Hashed processed outputs: 78
- SHA256 manifest: `sha256_processed_outputs.tsv`
- File manifest: `reproducibility_file_manifest.tsv`

## Reproduction Modes

- Mode 1 minimal demo writes `demo_output_event_objects.tsv`, `demo_claim_ceiling.tsv` and `demo_report.md`.
- Mode 2 table reproduction writes `reproduced_main_tables/`.
- Mode 2 figure reproduction writes `reproduced_main_figures/`.
- Direct external baseline execution writes package-status, metric and Docker report files under `direct_external_baseline/`.
- Quick benchmark execution writes `quick_benchmark_run_manifest.tsv` and `quick_benchmark_run_report.md`.
- Mode 3 full analysis remains long-running and is represented by dataset-specific scripts and manifests.
