# Direct External Baseline Docker Report

Generated UTC: 2026-05-16T02:20:32.798811+00:00
Local Docker CLI available: False

## Baseline environment

- `Dockerfile.baselines` installs `environment.baselines.yml`.
- The baseline environment includes R/Bioconductor tradeSeq, GSVA and AUCell plus Python POT.
- The local execution manifest records which packages were available in the current workstation environment.

## Reviewer commands

```bash
docker build -f Dockerfile.baselines -t ted-external-baselines .
docker run --rm -v "$PWD/data_external:/workspace/data_external" ted-external-baselines
```

## Local execution summary

| method   | direct_package   | local_status              | package_version   |
|:---------|:-----------------|:--------------------------|:------------------|
| tradeSeq | tradeSeq         | not_run_missing_R_package | nan               |
| GSVA     | GSVA             | not_run_missing_R_package | nan               |
| AUCell   | AUCell           | not_run_missing_R_package | nan               |
| POT_OT   | POT              | executed                  | 0.9.4             |

## Commands executed locally

| method      | status   |   exit_code |   runtime_seconds | log                                                                                     |
|:------------|:---------|------------:|------------------:|:----------------------------------------------------------------------------------------|
| tradeSeq    | executed |           0 |             0.205 | data_external\ted_development_phase4_benchmark\direct_external_baseline\tradeSeq.log    |
| GSVA_AUCell | executed |           0 |             0.363 | data_external\ted_development_phase4_benchmark\direct_external_baseline\GSVA_AUCell.log |
| POT_OT      | executed |           0 |             4.28  | data_external\ted_development_phase4_benchmark\direct_external_baseline\POT_OT.log      |
