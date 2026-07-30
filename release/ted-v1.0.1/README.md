# TED v1.0.1 release-manifest policy

`ted-v1.0.1` is a release-engineering patch over the immutable
`ted-v1.0.0` baseline. It must not change the declared frozen scientific
artifacts. As a second fail-closed gate, the complete
`results/ted_v1_submission/` Git subtree must remain byte-identical.

After the release candidate is committed, generate the audit outside the Git
tree:

```bash
python scripts/build_release_manifests.py \
  --baseline-ref ted-v1.0.0 \
  --release-ref HEAD \
  --external-assets-template release/ted-v1.0.1/EXTERNAL_ARCHIVE_ASSETS.template.tsv \
  --outdir ../ted-v1.0.1-release-attestation
```

Run the same command against the immutable `ted-v1.0.1` tag after it is
created. The tag-derived attestation, not a recursively scanned local
directory, is the authoritative release record.

The generated files are an external attestation because a complete tree
manifest cannot include its own final hash without recursion.

The historical `RELEASE_MANIFEST.tsv` and
`results/ted_v1_submission/evidence_manifest.tsv` remain untouched for
provenance. They are not authoritative release-tree manifests: the former
contains checkout-normalized hashes, while the latter includes a path that was
not a member of `ted-v1.0.0`.
