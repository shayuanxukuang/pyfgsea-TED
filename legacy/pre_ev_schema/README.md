# Historical pre-E/V schema demonstration

> **Legacy status:** this directory is retained for provenance only. It is not
> part of the canonical v1.0.x quickstart, package validation, current E/V
> schema contract, or submitted manuscript reproduction workflow.

`run_minimal_demo.py` was formerly located at
`reproducibility/run_minimal_demo.py`. It creates deterministic toy
trajectories and applies a standalone Level 1--4 claim-ceiling illustration.
The script does not import `pyfgsea`, invoke the installed `ted` command, or
validate its outputs against the packaged activity-v1/event-v2 schemas.

The file is preserved unchanged so historical development outputs remain
interpretable. Its historical outputs were:

- `data_external/ted_development_reproducibility/minimal_demo/demo_output_event_objects.tsv`
- `data_external/ted_development_reproducibility/minimal_demo/demo_claim_ceiling.tsv`
- `data_external/ted_development_reproducibility/minimal_demo/demo_report.md`

Those outputs are not evidence that the Python package installed successfully
and were not used to reproduce the submitted BIB Figure 3 480-task comparison
or Figure 5 BNT162b2/GSE171964 analyses.

For the supported installed-package and schema smoke test, follow the sole
canonical quickstart at [`../../README_quickstart.md`](../../README_quickstart.md).
