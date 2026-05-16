# Direct External Baseline Docker Report

Generated UTC: 2026-05-16T05:09:15.173314+00:00
Local Docker CLI available: False

## Baseline environment

- `Dockerfile.baselines` installs `environment.baselines.yml`.
- The baseline environment includes R/Bioconductor tradeSeq, GSVA and AUCell plus Python POT.
- The execution manifest records direct package wrappers run in the active baseline runtime.
- When this report is generated inside the container, Docker CLI availability is expected to be false and is not used as a success criterion.

## Reviewer commands

```bash
docker build -f Dockerfile.baselines -t ted-external-baselines .
docker run --rm -v "$PWD:/workspace" -w /workspace ted-external-baselines
```

## Direct package execution summary

| method   | direct_package   | local_status   | package_version   |
|:---------|:-----------------|:---------------|:------------------|
| tradeSeq | tradeSeq         | executed       | 1.24.0            |
| GSVA     | GSVA             | executed       | 2.4.4             |
| AUCell   | AUCell           | executed       | 1.32.0            |
| POT_OT   | POT              | executed       | 0.9.6.post1       |

## Commands executed in the active runtime

| method      | status   |   exit_code |   runtime_seconds | log                                                                                     |
|:------------|:---------|------------:|------------------:|:----------------------------------------------------------------------------------------|
| tradeSeq    | executed |           0 |            10.019 | data_external/ted_development_phase4_benchmark/direct_external_baseline/tradeSeq.log    |
| GSVA_AUCell | executed |           0 |             8.343 | data_external/ted_development_phase4_benchmark/direct_external_baseline/GSVA_AUCell.log |
| POT_OT      | executed |           0 |             1.559 | data_external/ted_development_phase4_benchmark/direct_external_baseline/POT_OT.log      |
