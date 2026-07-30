# TED v1.1.0 candidate package/schema quickstart

This is the canonical installed-package smoke for the v1.1.0 companion
candidate. Its purpose is deliberately narrow: install the Python package, run a deterministic
controlled-data generator through the installed package's schema validator, and
exercise the public `ted validate` command.

Archive and Python distribution identifiers are distinct: the
`ted-v1.1.0` candidate carries `pyfgsea==0.2.0`. The v1.0.1 release-engineering
patch candidate carries `pyfgsea==0.1.5`.

> **Scope boundary:** this smoke test does not reproduce the submitted
> *Briefings in Bioinformatics* Figure 3 480-task common-task comparison,
> Figure 5 BNT162b2 masked protein outcome or corrected GSE171964 replication.
> Those analyses are verified through the separate v1.1.0 companion
> reproduction entry and external archives, not through this schema smoke.

## Python environments

| Meaning | Version | Evidence |
| --- | --- | --- |
| Supported by package metadata | Python 3.9--3.13 | `pyproject.toml` declares `requires-python = ">=3.9,<3.14"`. |
| Candidate exact-tag installed-package matrix | Linux, Python 3.11, 3.12 and 3.13 | The CI matrix is configured to build and install a wheel, run `pip check`, import outside the source tree, exercise `ted --version`/`ted run --help`, and run the schema smoke. A target is release-tested only after its immutable-tag job passes. |
| Full test/integration environment | Linux, Python 3.11 | Fast, integration, slow and dedicated validation-demo jobs use Python 3.11. |
| Canonical locked reproduction environment | Linux, Python 3.11 | `requirements-reproduction-py311.txt` is the uv-compiled lock; `environment.yml` supplies the portable Conda base used below. |
| Historical dependency snapshot | Python 3.12.7 | `requirements-lock.txt` records a local release-candidate snapshot; it is not a complete cross-platform lock. |
| External-baseline environment | Python 3.12 | `environment.baselines.yml` is specific to tradeSeq/GSVA/AUCell/POT execution. |

## Canonical local smoke test

Run these commands from a clean clone:

```bash
conda env create -f environment.yml
conda activate ted-development
python -m pip install uv
uv pip sync --python python requirements-reproduction-py311.txt
maturin build --release --out dist
python -m pip install --no-deps dist/*.whl

python -I scripts/run_ted_validation_demo.py \
  --outdir results/ted_validation_demo

ted validate results/ted_validation_demo/demo_activity.tsv \
  --kind activity \
  --report results/ted_validation_demo/activity_cli_validation.tsv

ted validate results/ted_validation_demo/demo_events_v2.tsv \
  --kind event \
  --schema-version v2 \
  --report results/ted_validation_demo/event_cli_validation.tsv
```

`python -I` prevents the repository root from satisfying the package import, so
the demo must use the installed distribution. Successful completion verifies:

- the wheel/source installation can import `pyfgsea`;
- the deterministic generator completes;
- the built-in activity-v1 and event-v2 schemas are packaged;
- both tables pass the Python and console-command validators.

It does not verify biological truth, external replication, the full TED
inference stack, benchmark performance or manuscript figure reproduction.

Expected outputs are:

- `results/ted_validation_demo/demo_activity.tsv`
- `results/ted_validation_demo/demo_events_v2.tsv`
- `results/ted_validation_demo/demo_events.tsv`
- `results/ted_validation_demo/demo_validation.tsv`
- `results/ted_validation_demo/demo_report.html`
- `results/ted_validation_demo/demo_manifest.json`
- the two CLI validation reports requested above

See [docs/ted_validation_demo.md](docs/ted_validation_demo.md) for the
event-support and no-parallel-evidence contract.

## Real `ted run` interface

`ted run` is the rolling-window trajectory GSEA pipeline, not a generic
activity-table TED adjudication command:

```bash
ted run \
  --h5ad input.h5ad \
  --gmt gene_sets.gmt \
  --out results/
```

Required options:

- `--h5ad`: AnnData input.
- `--gmt`: gene-set collection.

Important defaults:

- `--out results`
- `--pseudotime-key dpt_pseudotime`
- `--window-size 800`
- `--step 50`
- `--ranker mean_diff`
- `--window-mode cell_count`

Use `ted run --help` for optional metadata merge, layer/raw selection,
alternative rankers, adaptive/graph windows and diagnostics controls.
Use `ted --version` to report the installed distribution version.

The command does **not** accept `--activity`, `--metadata`,
`--gene-sets`, `--design` or `--negative-controls`. Any document showing that
interface is an illustration from earlier manuscript planning, not an
executable command.

## Docker behavior

The main image builds and installs the package. Its default command is
`ted --help`; it does not run a demo, benchmark or figure workflow:

```bash
docker build -t ted-v1.0.x .
docker run --rm ted-v1.0.x

docker run --rm \
  -v "$PWD/results:/out" \
  ted-v1.0.x \
  python -I scripts/run_ted_validation_demo.py --outdir /out/ted_validation_demo
```

The baseline image installs R/Bioconductor tradeSeq, GSVA and AUCell, Python
POT, and the local package. Its default command only imports `pyfgsea` and
prints the package version:

```bash
docker build -f Dockerfile.baselines -t ted-external-baselines .
docker run --rm ted-external-baselines

docker run --rm \
  -v "$PWD/data_external:/workspace/data_external" \
  ted-external-baselines \
  python /workspace/scripts/run_direct_external_baseline_suite.py --quick
```

The explicit baseline command records missing-package or execution statuses in
`direct_external_baseline_execution_manifest.tsv`; the no-argument container
run is only an import/version smoke test.

## Historical v1.0.x workflows

The archive retains historical known-source audits, a 40-packet shifted audit,
external-baseline wrappers and other release-local outputs. Those facts remain
part of the v1.0.x provenance, but they are not the submitted BIB 480-task
benchmark or BNT162b2/GSE171964 companion.

The former `reproducibility/run_minimal_demo.py` is preserved at
`legacy/pre_ev_schema/run_minimal_demo.py`. It implements an older standalone
Level 1--4 claim-ceiling illustration and does not import the installed
package. It is not an installation check and is not part of this quickstart.
