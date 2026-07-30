# TED release status

## Public release

`ted-v1.0.0` remains the immutable public baseline:

- annotated tag object: `992e7569e2a7cb47fc06557302f7480496f7eeab`
- tagged commit: `5cb7b25458b41437b54623488d37b4872e79f474`
- analysis-lock parent: `54ec1269344b6a1392928324ae4a54c7d68d0260`
- version-specific DOI: `10.5281/zenodo.21403133`
- Python distribution version: `0.1.4`

The tag and the Zenodo record must not be moved, replaced, or overwritten.
`CITATION.cff` continues to describe this released baseline until a later
version has its own immutable tag and reserved DOI.

## Release candidates

`ted-v1.0.1` is a local release-engineering patch candidate. Its Python/Rust
distribution version is `0.1.5`. The candidate corrects packaging,
dependency, CLI-documentation, test, CI-provenance, and manifest defects. It
does not change the declared v1.0.0 scientific artifacts.

`ted-v1.1.0` is reserved for the BIB manuscript companion. It is a separate
deliverable and is not part of v1.0.x. It must receive a new immutable tag,
release archive, analysis lock, and version-specific DOI.

Neither candidate is public or citable yet. No DOI is assigned in this
repository before the corresponding external record is reserved.

## CI provenance correction

The previously cited 10-job release-candidate run
(`29526380376`, commit `a312b387fc77f59d390ebd14db7fe7bfcddfd31d`)
did not run against the `ted-v1.0.0` tag or an ancestor of that tag.

The post-release green main-branch run
(`29546657448`, commit
`c6db8bff70aaa491be2c6f73ed00b65d2f487231`) ran against the direct child of
the tag commit after a dependency and test-compatibility repair. It is useful
post-release evidence, but it is not exact-tag evidence.

The v1.0.0 tag's known-source validation assertion is inconsistent with the
tagged table, so a clean exact-tag full-suite pass must not be claimed. The
v1.0.1 workflow emits JUnit, terminal summaries, a Python 3.11/3.12/3.13
installed-wheel matrix, Docker logs, and a Git-object-backed attestation.
Those rows remain pending until the immutable v1.0.1 tag workflow completes.

## Frozen scientific-artifact gate

The v1.0.1 audit compares raw Git blob bytes, not checkout-normalized files.
The declared baseline set contains:

- 64 scientific result files, 5,170,330 bytes, aggregate SHA-256
  `7c85a6b3df7317804f54d92b8444e5a77166f12839274ebbbaae32d69eec4346`
- 2 provenance files, aggregate SHA-256
  `fcb650e51b0a85914c35f3d637a9dd05c4546dd14492aba6c8fd8aa39f7f8f39`

All 66 files are byte-identical between `ted-v1.0.0` and the post-release
repair commit `c6db8bff70aaa491be2c6f73ed00b65d2f487231`.
`scripts/build_release_manifests.py` fails closed if any declared file is
missing or differs in the release candidate.

The historical `RELEASE_MANIFEST.tsv` and the two historical
`evidence_manifest.tsv` files remain provenance records, not authoritative
tag-tree attestations. They contain checkout-normalized hashes and/or paths
that were not members of the tagged Git tree.

## v1.0.1 publication gates

Do not publish or tag v1.0.1 until all of the following are true:

1. the candidate commit is clean and reviewed;
2. the frozen-artifact audit reports zero differences;
3. the installed-wheel matrix, locked Python 3.11 environment, test jobs, and
   both Docker smoke jobs pass for the exact commit;
4. an immutable `ted-v1.0.1` tag is created without moving v1.0.0;
5. the exact-tag workflow passes and its JUnit, terminal summaries, Docker
   log, and release-tree attestation are retained;
6. a new version-specific DOI is reserved and the tag-derived archive is
   uploaded to that new record;
7. `CITATION.cff`, release notes, and
   `EXTERNAL_ARCHIVE_ASSETS.tsv` are updated with the new DOI and verified
   asset hashes in a final, auditable release commit.

## Manuscript boundary

The v1.0.x E/V baseline is not the sole computational companion for the
submitted BIB manuscript. It does not contain the locked 480-task common
task, all five methods' native outputs, the current Figure 3 and Figure 5
source data, the BNT162b2/GSE171964 evidence records, or the focused 81-test
evidence. Those materials belong to `ted-v1.1.0` under the repository's
computational-only research scope and existing claim ceilings.
