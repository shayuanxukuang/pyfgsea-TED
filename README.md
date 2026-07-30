# Trajectory Pathway Event Discovery (TED)

**Trajectory Pathway Event Discovery (TED)** is an artifact-aware protocol for structured interpretation of dynamic pathway events in single-cell genomics.

TED starts from pathway, module or perturbation activity profiles and writes
row-wise event objects. Each event row records event mode, effect and
uncertainty, block support, matched-state and negative-control behavior,
identifiability, and an E0--E2 within-study event-support code. Orthogonal
outcome, intervention reversal, and matched rescue are parallel typed evidence
records; they do not upgrade the event E code. Event-replication and
outcome-replication eligibility, test, and result states are separate facets.

The V0--V4 projection from v1.0.x is retained only for explicit migration of
historical records. It is not the v1.1 reviewer-facing evidence model. Escort
addresses upstream trajectory suitability; TED starts after a trajectory,
time, or state representation has been supplied. PyFgsea can generate upstream
activity profiles, but TED is a separate downstream inference task and accepts
activity matrices produced by other scoring or trajectory methods.

## Release status

The journal-neutral semantic release `ted-v1.0.0` contains the July 2026 E/V v2 schema, current-task baselines, repeated embryo holdouts, heavy-control outputs, scaling results and validation-demo records. Its version-specific archive DOI is [10.5281/zenodo.21403133](https://doi.org/10.5281/zenodo.21403133).

The local `ted-v1.0.1` candidate is limited to release engineering,
dependency, test and documentation compatibility. It preserves the frozen
v1.0 scientific artifacts and does not add a new manuscript analysis. Archive
and distribution identifiers are distinct: `ted-v1.0.1` carries the
`pyfgsea==0.1.5` Python package.

The local `ted-v1.1.0` candidate is the separately versioned BIB manuscript
companion and carries `pyfgsea==0.2.0`. Its Git tree contains the locked
analysis code, protocols, schemas, tests, explicit asset rules, and
reproduction/verification entry points. The task registry, truth,
harmonized/native outputs, Figure 3/5 source data, and flagship records are
packaged as checksummed external release assets rather than committed by
recursively adding the local `results/` tree.

The canonical flagship contract is claim-bounded to `E0 | protein outcome
passed | event replication not_evaluable (eligibility failed; test not_run) |
protein outcome replication not_tested`. Legacy E/V fields and hypothetical
success strings remain byte-preserved provenance only and are not current
conclusions.

`ted-v1.1.0` is not yet a public release. A clean analysis-lock commit,
verified external archives, exact-tag CI, and a new version-specific DOI are
required before it can be cited.

> **Manuscript-companion boundary:** the `ted-v1.0.x` archive is not the
> computational companion for the submitted *Briefings in Bioinformatics*
> Figure 3 480-task common-task comparison or Figure 5 BNT162b2/GSE171964
> analyses. It does not contain the corresponding 480 task registry, 2,400
> method-task native outputs, BNT162b2 masked protein-outcome analysis, or
> corrected GSE171964 replication package. Those materials require a
> separately versioned `ted-v1.1.0` manuscript-companion release. No v1.0.x
> command should be represented as reproducing those figures.

The historical May 2026 archive is:

- Historical Git tag: `ted-gb-rc7`
- Commit: `3ffec1a1dcb4261303fc130b81ccd6b29a2fa34f`
- Zenodo DOI: [10.5281/zenodo.20378158](https://doi.org/10.5281/zenodo.20378158)
- License: MIT

That DOI remains valid for the historical snapshot, but it does not contain the complete July 2026 materials. Cite `10.5281/zenodo.21403133` for the E/V v2 release.

### Analysis lock and final release

The analysis lock identifies the exact estimator code, thresholds, seeds, inputs and machine-readable output hashes used for evaluation. The final-release commit may add packaging, documentation, citation metadata and archive manifests, but it must not silently change locked estimators or thresholds. Release metadata must record both identifiers and a file-level difference audit. If analysis code or thresholds change, the affected analyses must be rerun and a new analysis lock recorded.

## What is in this repository

| Path | Purpose |
| --- | --- |
| `pyfgsea/` | Python package code for PyFgsea and TED event-analysis components. |
| `scripts/` | Reproducibility scripts for known-source validation, GATA1/GATA1s support, external baselines, benchmarks and figure generation. |
| `tests/` | Unit and validation tests used for the release snapshot. |
| `config/` | Event axes, negative-control axes, evidence-boundary rules and preregistration cards. Some historical filenames retain legacy `claim_boundary` names. |
| `tables/` | Machine-readable event objects, benchmark summaries, evidence-boundary outputs, validation summaries and release-audit tables. Some historical filenames retain legacy terminology for provenance. |
| `figures/` | Main figure PDFs/PNGs and their source-data TSV files. |
| `scripts/run_ted_validation_demo.py` | Deterministic installed-package and E/V schema smoke test. |
| `scripts/build_ted_bib_companion.py` | Fail-closed, explicit-allowlist builder for the v1.1.0 external companion archives. |
| `scripts/verify_ted_bib_companion.py` | Independent archive member, size, SHA-256, and required-role verification. |
| `reproduce/` | Companion verification and Figure 3/5 source-to-figure reproduction entry points. |
| `schemas/parallel_evidence_record_v1.schema.json` | Canonical outcome/reversal/rescue parallel-record contract. |
| `schemas/replication_facets_v1.schema.json` | Canonical event/outcome replication-facet contract. |
| `release/ted-v1.1.0/` | Candidate metadata, asset rules, focused-test evidence, and publication gates. |
| `release/ted-v1.1.0/CLAIM_BOUNDARY.md` | Observed BNT162b2/GSE171964 state and treatment of frozen legacy success criteria. |
| `legacy/pre_ev_schema/` | Historical pre-E/V development demonstration; not a package or manuscript reproduction test. |
| `results/ted_v1_submission/` | Final figure source data, controlled-benchmark summaries, public-data audits and checksummed release outputs. |
| `Dockerfile`, `Dockerfile.baselines`, `environment*.yml` | Runtime environments for TED analyses and direct external baseline execution. |

Journal submission files are managed separately from this software archive. Large public raw datasets are also kept at their original repositories; the manifests in `tables/` record accessions, file provenance and processed-output checks.

## Quick start

The installed-package smoke is documented in
[README_quickstart.md](README_quickstart.md). It creates the canonical Python
3.11 environment, installs the package, runs the deterministic validation demo
in isolated mode, and validates its activity-v1 and event-v2 tables through the
installed `ted` console command.

That sequence is a package/schema smoke test. It is not a benchmark, external
validation, or main-figure reproduction. The v1.1.0 companion build and
verification path is documented separately in
[`release/ted-v1.1.0/README.md`](release/ted-v1.1.0/README.md).

### Public CLI boundary

The v1.0.x `ted run` command is the rolling-window trajectory GSEA entry point:

```bash
ted run --h5ad input.h5ad --gmt gene_sets.gmt --out results/
```

Only `--h5ad` and `--gmt` are required. `--out` defaults to `results`,
`--pseudotime-key` defaults to `dpt_pseudotime`, `--window-size` defaults to
800, `--step` defaults to 50, `--ranker` defaults to `mean_diff`, and
`--window-mode` defaults to `cell_count`; use `ted run --help` for the remaining
optional window, graph, layer and metadata controls.

`ted run` does not accept `--activity`, `--metadata`, `--gene-sets`, `--design`
or `--negative-controls`, and it does not implement the review-facing
activity-table adjudication interface described in later manuscript materials.
Use `ted --version` to report the installed distribution version. Schema
validation is a separate command:

```bash
ted validate TABLE --kind activity
ted validate TABLE --kind event --schema-version v2
```

### Python and container support

- **Supported by package metadata:** Python 3.9 through 3.13.
- **Candidate exact-tag compatibility matrix (pending):** Linux with Python
  3.11, 3.12 and 3.13 in `.github/workflows/ci.yml`; these become
  release-tested targets only after the immutable tag jobs pass.
- **Full test/integration environment:** Linux with Python 3.11.
- **Canonical locked reproduction environment:** Linux Python 3.11 from
  `requirements-reproduction-py311.txt`; `environment.yml` is the portable
  Conda base specification.
- **Historical dependency snapshot:** `requirements-lock.txt` records a local
  Python 3.12.7 snapshot; it is not the canonical Conda environment or a
  complete cross-platform lock.
- **External-baseline environment:** `environment.baselines.yml` uses Python
  3.12 for tradeSeq/GSVA/AUCell/POT execution and does not expand the core
  package's release-tested matrix.

The main `Dockerfile` builds and installs the wheel, then defaults to
`ted --help`. `Dockerfile.baselines` defaults to importing `pyfgsea` and
printing its version. Neither default command runs the validation demo,
benchmark suite, external baselines, manuscript tables or manuscript figures;
explicit invocations are documented in the canonical quickstart.

## Key validation outputs in the historical snapshot

TED was evaluated with a combination of same-input benchmarks, public known-source datasets and evidence-boundary audits. The most useful starting points in the historical snapshot are:

The figure numbers and filenames in this section are release-local historical
names. In particular, `figure5_claim_upgrade_block_audit.pdf` is not the
submitted BIB Figure 5.

| File | What it records |
| --- | --- |
| `tables/known_source_validation_summary.tsv` | Public known-source validation results for GSE153056, GSE93735 and SCP1064. |
| `tables/ted_dataset_level_claim_boundary.tsv` | Dataset-level evidence boundaries assigned by TED; legacy filename. |
| `tables/benchmark_audit_table.tsv` | Benchmark truth sources, scored units, uncertainty reporting and frozen/optimization status. |
| `tables/benchmark_non_circular_evaluation_table.tsv` | Separation of biological correctness metrics from reporting-completeness fields. |
| `tables/dynamic_pathway_event_table.tsv` | Standardized dynamic pathway-event grammar rows. |
| `tables/scp1064_lightweight_shuffle_summary.tsv` | Lightweight label-shuffle audit for SCP1064 outcome alignment. |
| `tables/gata1_cross_dataset_support_summary.tsv` | Independent GATA1/GATA1s directional-support summary. |
| `figures/figure2_known_source_validation.pdf` | Public known-source outcome and reversal validation figure. |
| `figures/figure4_gse271399_gata1_cross_dataset_support.pdf` | GSE271399 and independent GATA1/GATA1s support figure. |
| `figures/figure5_claim_upgrade_block_audit.pdf` | Evidence-promotion/block audit figure; legacy filename. |

## Legacy v1.0.x evidence descriptors

The rows below reproduce the archived v1.0.x E/V projection for migration and
provenance only. They are not the v1.1 reviewer-facing evidence model and must
not be used to upgrade an event E code. In v1.1, outcome, reversal and rescue
are parallel typed records, while event and outcome replication remain
separate facets.

| Dataset | Readout | Historical E/V projection | Qualification |
| --- | --- | --- | --- |
| GSE153056 | IFNG/PD-L1 RNA event aligned with PD-L1 protein effects | E1--V1 | Formal locked assignment; a retrospective three-block audit met E2 eligibility but does not replace the locked endpoint. |
| GSE93735 | Partial dexamethasone reversal of an LPS-associated signal | E0--V2 | Two samples per group and no event-level q value; reversal provenance is descriptive. |
| SCP1064 | CRISPR-linked RNA events aligned with CITE-seq protein readouts | E1--V1 (partial) | Mandatory within-block guide-label shuffling passed only 1 of 4 axes, so E2 promotion is blocked. |
| GSE271399 | GATA1/T21 computational event with cross-dataset directional context | E1--V0 | Available design strata are not independent biological blocks, and no same-system matched full-length GATA1 rescue is available. |

Positive outcome alignment cannot override a failed mandatory control. Same-system matched rescue is represented separately from orthogonal outcomes or intervention-reversal provenance.

## Current benchmark status

The July 2026 packet split is retrospective: replicates 1--8 per event mode are development data, 9--12 are used for baseline tuning, and 13--16 form a 40-packet shifted audit whose labels were masked during prediction. Because the generator and TED rules pre-date this split, the audit is not an untouched final test and no truly post-freeze external dataset has yet been evaluated.

This 40-packet shifted audit is a historical v1.0.x result. It is not the
480-task common-task benchmark used for the submitted BIB Figure 3.

On the shifted audit, TED aggregate packet-class macro-F1 was 0.741 and controlled E-level macro-F1 was 0.761. Supervised models classified packet and artifact states more accurately. TED made no upward E errors in this audit, at the cost of a 0.225 false-demotion rate; each downgrade was linked to a recorded failed gate. TED is therefore evaluated as a conservative evidence-assignment protocol rather than an accuracy-maximizing classifier.

Across 100 repeated 20% ZSCAPE embryo holdouts, median event-set Jaccard was 0.654, direction agreement 1.000 and mode agreement 0.936. Across 50 balanced split halves, median event-set Jaccard was 0.207, showing that event discovery is sample-sensitive even when directions of common calls are stable.

## Direct external baselines

The direct external baseline suite includes wrappers for representative upstream tools, including tradeSeq, GSVA, AUCell and POT. These runs are used to check executable upstream outputs and to define how native outputs can be carried into the downstream TED-object comparison.

```bash
python scripts/run_direct_external_baseline_suite.py --quick
```

For the package-complete baseline image, use `Dockerfile.baselines`.
Running that image without an explicit command only imports `pyfgsea` and
prints its version; it does not execute the baseline suite automatically.

## Data access

The release uses public datasets from GEO, Single Cell Portal, STOmicsDB/CNGB and related public resources. Raw public archives are not mirrored here. The relevant accessions, download status, checksums and analysis roles are tracked in:

- `tables/availability_accession_audit.tsv`
- `tables/candidate_download_manifest.tsv`
- `RELEASE_MANIFEST.tsv`

## Citation

Machine-readable citation metadata are provided in `CITATION.cff`. Cite the `ted-v1.0.0` archive as [10.5281/zenodo.21403133](https://doi.org/10.5281/zenodo.21403133). The historical DOI [10.5281/zenodo.20378158](https://doi.org/10.5281/zenodo.20378158) refers only to the May 2026 development snapshot.
