# TED v1.1.0 — BIB manuscript companion

Status: verified release candidate. Do not publish, tag, or upload until a
new v1.1.0 Zenodo version DOI has been reserved and the exact-tag source
attestation succeeds.

This release is the computational companion for the submitted *Briefings in
Bioinformatics* manuscript. It is distinct from the immutable `ted-v1.0.0`
baseline (`10.5281/zenodo.21403133`) and from the release-engineering-only
`ted-v1.0.1` patch.

Analysis lock:
`32e099c780bf0103bbcfadb2993e59254c6d9e12`.

The three locally built and independently verified archives contain:

- the 480-task registry and post-output truth;
- harmonized predictions and exactly 2,400 native method-task outputs;
- all Figure 3 and Figure 5 source tables and frozen outputs;
- BNT162b2 RNA/protein records and the corrected-v2 GSE171964
  eligibility/readout audit;
- schema-valid parallel-evidence and replication-facet instances;
- stability/resolution shards; and
- exact focused-test evidence (`81 passed`, `0 failed`, `0 errors`,
  `0 skipped`).

Observed claim boundary:

`E0 | protein outcome passed | event replication not_evaluable (eligibility failed; test not_run) | protein outcome replication not_tested`

The passed protein record is parallel evidence and does not upgrade the event
E code. Frozen E/V fields and hypothetical success strings are retained only
as provenance.

Figure 3 and Figure 5 were independently redrawn from packaged source data.
Both redraws passed semantic checks and nonblank pixel/PDF QA. Cross-platform
plot byte identity is not claimed.

The historical ten-job Linux workflow cited for v1.0.0 validated release
candidate commit `a312b387fc77f59d390ebd14db7fe7bfcddfd31d`, not the final
`ted-v1.0.0` tag commit. The later green workflow validated post-release
compatibility commit `c6db8bff70aaa491be2c6f73ed00b65d2f487231`. Neither is
described here as an exact-tag ten-job validation.

Archive sizes and SHA-256 values are recorded in
`release/ted-v1.1.0/EXTERNAL_ARCHIVE_ASSETS.tsv`. The new v1.1.0 version DOI
must be inserted into `RELEASE_METADATA.json` before the tag can pass its
publication gate; the v1.0.0 DOI must not be reused or overwritten.
