# TED v1.0.1 — release-engineering patch

Status in this file: **candidate text; do not publish before the exact-tag
gates pass**.

TED v1.0.1 corrects release engineering around the immutable v1.0.0
scientific baseline. It updates the Python/Rust distribution to pyfgsea
0.1.5, restores a complete runtime dependency, makes the real CLI contract
explicit, supplies a canonical locked Python 3.11 environment, and replaces
checkout-derived integrity claims with Git-object-backed manifests.

This patch does not change the 64 declared scientific artifacts or the two
provenance artifacts from v1.0.0. The exact-tag attestation must report all 66
as byte-identical before publication.

This is not the BIB manuscript companion. The locked 480-task common task,
native outputs, Figure 3/5 source data, BNT162b2/GSE171964 records, and
focused 81-test evidence will be released separately as `ted-v1.1.0`.

## Verification assets required before publication

- exact-tag JUnit and terminal summary
- installed-wheel smoke for Python 3.11, 3.12, and 3.13
- canonical locked Python 3.11 smoke
- no-cache main and baseline Docker logs
- `FROZEN_SCIENTIFIC_ARTIFACTS.tsv`
- `RELEASE_TREE_MANIFEST.tsv`
- `EXTERNAL_ARCHIVE_ASSETS.tsv`
- `manifest_summary.json`
- a new version-specific DOI (not the v1.0.0 DOI)

The existing `ted-v1.0.0` tag and DOI
`10.5281/zenodo.21403133` remain unchanged.
