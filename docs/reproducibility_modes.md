# Reproducibility Modes

This document defines what the `ted-v1.0.x` archive can and cannot reproduce.
The only canonical first-run workflow is the installed-package schema smoke in
[`README_quickstart.md`](../README_quickstart.md).

## Mode 1: package/schema smoke

- Uses deterministic controlled data.
- Installs the package before execution and uses isolated Python mode.
- Verifies packaged activity-v1/event-v2 schemas and the `ted validate` console
  command.
- Does not establish biological truth, inference performance or external
  validation.
- Canonical instructions:
  [`README_quickstart.md`](../README_quickstart.md).

## Mode 2: historical v1.0.x artifact inspection

- Uses the frozen tables, figures, source-data files and manifests already
  present in the v1.0.x archive.
- Covers release-local known-source audits, the historical 40-packet shifted
  audit, robustness summaries and evidence-boundary records.
- Historical figure numbers are local to this archive and must not be mapped to
  the submitted BIB figure numbering.
- `reproduce_all_main_tables.py` and `reproduce_all_main_figures.py` are not
  members of the v1.0.x release tree and are not valid public entry points.

## Mode 3: explicit historical workflow reruns

- Uses the dataset-specific and benchmark scripts under `scripts/`.
- May require public data downloads, substantial storage, optional R packages
  or long runtimes.
- Must be scoped to the named historical v1.0.x artifact; the presence of a
  script is not evidence that all required raw data are archived locally.
- Direct external baselines can be invoked explicitly with
  `python scripts/run_direct_external_baseline_suite.py --quick` or through the
  explicit baseline-container command in the canonical quickstart.

## BIB manuscript-companion exclusion

None of the three v1.0.x modes reproduces:

- the submitted BIB Figure 3 480-task common-task comparison;
- the 2,400 native method-task outputs behind that comparison;
- the submitted BIB Figure 5 BNT162b2 masked protein outcome;
- the corrected GSE171964 eligibility/replication analysis; or
- the associated current-manuscript tables and figure source package.

Those artifacts require a separately versioned manuscript-companion release.
The historical v1.0.x 40-packet audit and release-local Figure 5 filename must
not be described as the BIB Figure 3 or Figure 5 analyses.

## Historical pre-E/V demonstration

The former `reproducibility/run_minimal_demo.py` is retained at
`legacy/pre_ev_schema/run_minimal_demo.py`. It independently implements an
older Level 1--4 claim-ceiling illustration, does not import `pyfgsea`, and is
not a package smoke test or manuscript reproduction workflow.

## Release verification record

The v1.0.1 candidate GitHub workflow is configured to build and install wheels
and run installed-package compatibility smoke tests outside the source tree on
Linux with Python 3.11, 3.12 and 3.13. These are pending candidate checks, not
release-tested claims, until the immutable-tag jobs pass. The full fast,
integration and slow suites and the dedicated validation-demo job use Linux
with Python 3.11, which is also the locked reproduction target. The workflow
additionally smoke-tests both Docker images; the separate external-baseline
container uses the Python 3.12 environment declared in
`environment.baselines.yml`.

If any release check or historical workflow is not run, record that status
explicitly rather than inferring success from the presence of code or a
container recipe.
