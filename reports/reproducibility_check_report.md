# Reproducibility Check Report

Generated for the release companion repository after direct external baseline container validation.

## Check Summary

| Check | Status | Evidence |
| --- | --- | --- |
| Docker Desktop available | pass | `docker info` returned Docker Desktop server 29.4.3 |
| Baseline image build | pass | `docker build -f Dockerfile.baselines -t ted-baselines:gb-rc2 .` completed |
| Baseline container run | pass | `docker run --rm -v "G:\pyfgsea-TED-release:/workspace" -w /workspace ted-baselines:gb-rc2` completed |
| tradeSeq wrapper | pass | `executed`, package version 1.24.0 |
| GSVA wrapper | pass | `executed`, package version 2.4.4 |
| AUCell wrapper | pass | `executed`, package version 1.32.0 |
| POT wrapper | pass | `executed`, package version 0.9.6.post1 |

## Output Files

- `tables/direct_external_baseline_registry.tsv`
- `tables/direct_external_baseline_execution_manifest.tsv`
- `tables/direct_external_baseline_metric_table.tsv`
- `tables/direct_external_baseline_to_ted_object_adapter.tsv`
- `reports/direct_external_baseline_docker_report.md`

## Scope

`Dockerfile.baselines` is scoped to the direct external package comparison used for the Genome Biology baseline narrative. The broader full benchmark orchestration script remains available, but it requires the full processed benchmark input set and is not used as the baseline-container smoke test.
