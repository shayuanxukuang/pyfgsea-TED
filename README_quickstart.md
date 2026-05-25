# TED-Development Reproducibility Quickstart

This repository provides an auditable release-candidate workflow for TED. It supports a minimal demo, main figure/table reproduction from processed source data, and a full benchmark entry point. The portable environment is `environment.yml`; the tested dependency snapshot is `requirements-lock.txt` with `environment.lock.yml`.

## Local Conda

```bash
conda env create -f environment.yml
conda activate ted-development
python reproducibility/run_minimal_demo.py
python scripts/run_direct_external_baseline_suite.py --quick
python reproduce_all_main_tables.py
python reproduce_all_main_figures.py
python scripts/run_full_benchmark_suite.py --quick
```

For the locked reviewer snapshot, install from `requirements-lock.txt` inside a Python 3.12.7 environment. The lock file records the package versions used to generate the current submission package.

## Docker

```bash
docker build -t ted-development .
docker run --rm -v "$PWD/data_external:/workspace/data_external" ted-development

docker build -f Dockerfile.baselines -t ted-external-baselines .
docker run --rm -v "$PWD/data_external:/workspace/data_external" ted-external-baselines
```

The Docker command runs the minimal demo, a quick benchmark smoke run, and main table/figure reproduction. A full benchmark run can be launched inside the container with:

```bash
python scripts/run_full_benchmark_suite.py
```

`Dockerfile.baselines` is a package-complete direct-baseline image. It installs R/Bioconductor tradeSeq, GSVA and AUCell plus Python POT and runs `scripts/run_direct_external_baseline_suite.py`. On workstations where some external packages are not installed, the same script records missing-package statuses in `direct_external_baseline_execution_manifest.tsv` rather than silently replacing them with internal approximations.

## Reproduction Modes

Mode 1 minimal demo:

```bash
python reproducibility/run_minimal_demo.py
```

Expected outputs:

- `data_external/ted_development_reproducibility/minimal_demo/demo_output_event_objects.tsv`
- `data_external/ted_development_reproducibility/minimal_demo/demo_claim_ceiling.tsv`
- `data_external/ted_development_reproducibility/minimal_demo/demo_report.md`

Mode 2 main figures and tables:

```bash
python reproduce_all_main_tables.py
python reproduce_all_main_figures.py
```

Mode 3 full benchmark suite:

```bash
python scripts/run_full_benchmark_suite.py
```

Use `--quick` for a small reviewer smoke test.

## Main Outputs

- `data_external/ted_development_phase4_benchmark/adversarial_benchmark/`
- `data_external/ted_development_phase4_benchmark/serious_baseline_suite/`
- `data_external/ted_development_phase4_benchmark/baseline_comparison/`
- `data_external/ted_development_phase4_benchmark/ablation/`
- `data_external/ted_development_phase4_benchmark/algorithm_sensitivity/`
- `data_external/ted_development_reproducibility/main_tables/`
- `data_external/ted_development_reproducibility/main_figures/`
- `data_external/ted_development_reproducibility/full_benchmark_run_manifest.tsv`

The expected scientific behavior is not perfect performance. In adversarial low-identifiability regimes, TED should lower the claim ceiling instead of promoting a strong biological claim.
