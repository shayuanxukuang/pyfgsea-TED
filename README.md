# pyfgsea-TED

This repository contains the code, benchmark entry points, selected result tables and figure source data for TED: a claim-aware framework for dynamic pathway event interpretation in single-cell genomics.

Archived release DOI: [10.5281/zenodo.20224518](https://doi.org/10.5281/zenodo.20224518)

TED converts dynamic single-cell pathway/module signals into auditable event-and-claim objects. A TED event object records effect direction, event-FDR, block robustness, matched-state evidence, negative-control behavior, event mode, identifiability and claim ceiling.

## What is included

- Core Python package code under `pyfgsea/`.
- Benchmark and baseline scripts under `scripts/`.
- Direct external baseline wrappers for tradeSeq, GSVA, AUCell and POT under `scripts/baselines/`.
- Conda and Docker environments, including `Dockerfile.baselines`.
- Selected machine-readable result tables under `tables/`.
- Main figure images and source data under `figures/`.
- Reproducibility and release audit reports under `reports/`.

## What is not included

This repository intentionally excludes article text, journal letters, compiled article PDFs, and submission-specific documents. It is intended as a software/data companion suitable for archival release.

## Quick checks

```bash
python -m py_compile scripts/run_direct_external_baseline_suite.py scripts/run_full_benchmark_suite.py
python scripts/run_direct_external_baseline_suite.py --quick
python scripts/run_full_benchmark_suite.py --quick --keep-going
```

## Direct external baselines

```bash
docker build -f Dockerfile.baselines -t ted-external-baselines .
docker run --rm -v "$PWD/data_external:/workspace/data_external" ted-external-baselines
```

The local workstation used for the draft executed POT directly and recorded missing R/Bioconductor packages rather than substituting internal approximations. `Dockerfile.baselines` defines the complete reviewer-side baseline environment.

## License

MIT License.

## Remote

Prepared for: https://github.com/shayuanxukuang/pyfgsea-TED.git
