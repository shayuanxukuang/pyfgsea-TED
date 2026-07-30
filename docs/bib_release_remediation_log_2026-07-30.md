# BIB release-remediation revision log — 2026-07-30

This log records the manuscript-facing correction made after the GitHub/BIB
submission-readiness audit. The change is computational and
documentation-only. It does not alter a dataset, estimator, threshold,
reported number, biological conclusion, or claim ceiling.

| Audit concern | Affected source | Previous representation | Corrected representation | Evidence and status |
| --- | --- | --- | --- | --- |
| The advertised “Minimal TED run” was not an executable CLI contract. | `scripts/build_genome_biology_submission_package.py`; `GenomeBiology_submission_files_only/minimal_ted_run_box.md`; generated supplementary LaTeX copies under `GenomeBiology_submission_files_only/`, `GenomeBiology_known_source_submission_package/`, `latex_submission_package/`, and the retained BIB/initial-upload directories | `ted run --activity ... --metadata ... --gene-sets ... --design ... --negative-controls ...` | The installed-package smoke now runs `scripts/run_ted_validation_demo.py` and validates `demo_activity.tsv` and `demo_events_v2.tsv` with the real `ted validate` command. Text explicitly states that `ted run` is the trajectory-GSEA interface requiring `--h5ad` and `--gmt`. | CLI source: `pyfgsea/cli/main.py`. All occurrences of the five-argument fictional interface were removed from the listed generator and generated sources. Complete locally; the corrected files must be used in the next manuscript revision upload. |
| A package/schema smoke could be mistaken for manuscript reproduction. | Same supplementary-material locations | The command was described as a review-facing input/output contract and listed event-object outputs that the public command does not produce. | The command is labelled a historical activity-v1/event-v2 schema/package smoke. It explicitly does not run the locked 480-task common task and does not reproduce Figure 3 or Figure 5. | Scope checked against the current BIB machine-readable asset audit. Complete locally. |

## Integrity notes

- No reference, citation, or bibliography entry was added, removed, or
  renumbered by this correction.
- No verbatim source quotation was introduced.
- The old `ted-v1.0.0` tag and its DOI remain immutable.
- The v1.1 companion projects byte-preserved BNT162b2/GSE171964 status
  artifacts into schema-valid parallel-evidence and replication-facet
  records. The observed state remains E0 with a passed same-study protein
  outcome; the GSE171964 event test was not run after failed eligibility.
- A separate `ted-v1.0.1` release-engineering candidate and
  `ted-v1.1.0` manuscript-companion candidate are prepared outside the
  manuscript source tree; neither is represented as public before exact-tag
  verification and a new DOI.
