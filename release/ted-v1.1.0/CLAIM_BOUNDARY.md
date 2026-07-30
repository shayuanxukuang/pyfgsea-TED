# BNT162b2 and GSE171964 claim boundary

The BNT162b2 protocol was frozen before local expression scoring. Its retained
`E2-V1` text is a historical, conditional success criterion in the legacy
migration notation; it is not the observed result and is not the v1.1 evidence
model.

The observed, release-gating result is:

- BNT162b2 RNA event support: `E0`.
- Same-study CD64/CD169 protein outcome: `passed` as a parallel orthogonal
  outcome record; it does not alter `E0`.
- Corrected GSE171964 event-replication eligibility: `failed`.
- GSE171964 event-replication test: `not_run`.
- GSE171964 event-replication result: `not_evaluable`.
- GSE171964 CD64/CD169 protein-outcome replication: `not_tested`.

The canonical machine-readable projection is under
`results/ted_bib_companion_evidence_contract_v1/`. The companion verifier
validates those instances against the standalone parallel-evidence and
replication-facet schemas and rejects any event-code upgrade.

Legacy `validation_provenance_code`, `evidence_boundary`, hypothetical
success-display strings, and the pre-result protocol wording are retained only
to preserve analysis provenance. They must not be read as current conclusions.
