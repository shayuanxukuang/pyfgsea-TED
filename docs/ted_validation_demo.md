# TED v1.0.x package/schema validation demo

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
`--schema-version` is omitted because the table contains E/V fields.

## Outputs

| File | Purpose |
| --- | --- |
| `demo_activity.tsv` | Input activity table with dataset, block, trajectory, time, pathway, activity and weight fields. |
| `demo_events_v2.tsv` | Canonical v2 event report with event support (E), E0 reason code, validation provenance (V), evidence boundary and interpretation fields. |
| `demo_events.tsv` | Byte-equivalent compatibility alias. |
| `demo_validation.tsv` | Combined activity/event schema validation report. |
| `demo_report.html` | Human-readable event table for validation inspection. |
| `demo_manifest.json` | Seed, file roles, schema version and evidence-boundary statement. |

The demo retains deprecated `evidence_tier`, `claim_ceiling` and
`matched_functional_rescue` columns to exercise transition compatibility.
They do not determine a v2 evidence boundary.

## Release and manuscript boundary

The E/V v2 fields are retained because this demo tests the v1.0.x schema
contract. They are not evidence that an external validation was performed.

The demo is not a reproduction entry point for the submitted BIB Figure 3
480-task common-task comparison or Figure 5 BNT162b2/GSE171964 analyses. The
corresponding task registry, native method outputs, masked protein outcome,
corrected replication package and figure source data are outside the v1.0.x
archive and require a separately versioned manuscript companion.

## Fail-closed evidence gates

For v2, `ted validate` requires `evidence_boundary` to match
`event_support_code` and `validation_provenance_code`. Every E0 row must carry
one of the five stable `e0_reason_code` values; E1 and E2 rows leave it null.
E1 and E2 require a numeric event q value, E2 must be identifiable, and V3 is
rejected unless `matched_functional_rescue=true`. A null event q is accepted
only for `E0_not_estimable`; other E0 reasons require a numeric q. E0 is a
support/design outcome and does not mean that no event exists. V0 means
computational evidence only.
