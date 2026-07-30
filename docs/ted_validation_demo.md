# TED package/schema validation demo

The deterministic validation demo generates a controlled block-aware activity
table and a synthetic event table, then validates both with schemas packaged in
the installed `pyfgsea` distribution. It completes in seconds on a normal
workstation.

This is a package/schema smoke test. The event-calling illustration is local to
the demo script; it is not the production `ted run` pipeline, an external truth
source, a biological validation, or a manuscript figure reproduction.

## Canonical invocation

Follow the installation and invocation sequence in
[the sole canonical quickstart](../README_quickstart.md). That sequence installs
the package, uses `python -I` so the source tree cannot satisfy the import, and
then exercises `ted validate` independently on the generated activity-v1 and
event-v2 tables.

Do not substitute the historical
`legacy/pre_ev_schema/run_minimal_demo.py`; that file does not import or test
the installed package.

The fixed random seed is `20260715`. The demo contains six biological blocks
and four prespecified pathway behaviors. Cells are not used as independent
inferential replicates. Event schema v2 is auto-detected when
`--schema-version` is omitted because the table contains v2 event-support and
test-status fields.

## Outputs

| File | Purpose |
| --- | --- |
| `demo_activity.tsv` | Input activity table with dataset, block, trajectory, time, pathway, activity and weight fields. |
| `demo_events_v2.tsv` | Canonical v2 event report with E0–E2 event support, E0 reason code, test/resampling status and interpretation fields. |
| `demo_events.tsv` | Byte-equivalent compatibility alias. |
| `demo_validation.tsv` | Combined activity/event schema validation report. |
| `demo_report.html` | Human-readable event table for validation inspection. |
| `demo_manifest.json` | Seed, file roles, schema version and explicit no-parallel-evidence statement. |

The demo retains deprecated `evidence_tier`, `claim_ceiling` and
`matched_functional_rescue` columns to exercise transition compatibility.
They do not determine current v2 event support.

## Release and manuscript boundary

The deprecated horizontal V fields are deliberately absent. The demo emits
only an E0–E2 event record because it does not run an orthogonal outcome,
reversal, rescue or independent-replication analysis. Current v1.1 evidence
represents any such analyses as parallel typed records and separate
event/outcome replication facets.

The demo is not a reproduction entry point for the submitted BIB Figure 3
480-task common-task comparison or Figure 5 BNT162b2/GSE171964 analyses. The
corresponding task registry, native method outputs, masked protein outcome,
corrected replication package and figure source data are handled by the
separately versioned v1.1.0 companion build and reproduction entry.

## Fail-closed evidence gates

For v2, `ted validate` requires the E0–E2 event-support fields and their
fail-closed reason/status fields. The legacy `validation_provenance_code` and
`evidence_boundary` columns are optional migration projections and are not
emitted by this demo. Every E0 row carries one of the five stable
`e0_reason_code` values; E1 and E2 rows leave it null. E1 and E2 require a
numeric event q value, and E2 must be identifiable with a declared block
support method. A null event q is accepted only when the event test did not
run and a stable missing-reason code is present. E0 is a support/design outcome
and does not mean that no event exists.
