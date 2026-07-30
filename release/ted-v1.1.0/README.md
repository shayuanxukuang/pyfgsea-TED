# TED BIB manuscript companion v1.1.0

This directory defines the release contract for the manuscript companion. It
does not declare that `ted-v1.1.0` has been tagged, archived, or assigned a DOI.
The analysis-lock commit and a new Zenodo version DOI remain explicit external
release gates in `EXTERNAL_ARCHIVE_ASSETS.template.tsv`. The immutable
`ted-v1.0.0` tag and its Zenodo files must not be moved or overwritten.

## Allowlisted package layout

`COMPANION_ASSET_RULES.template.tsv` is the default, configurable asset
allowlist. Its `root` column resolves every path against exactly one of two
explicit roots: `repository` for the clean analysis-lock checkout
(code/schemas/renderer/focused evidence), or `data` for ignored result and
public-data artifacts. Resolution is containment-checked independently for
both roots. The builder never copies new code into the old dirty data
workspace and never falls back from one root to the other. The build produces
three independent ZIP64 archives:

- `ted-bib-companion-v1.1.0-core.zip`: the encoded 480-task registry/status,
  post-output truth, harmonized predictions, nested native-output manifest,
  all four Figure 3 and seven Figure 5 source tables and final PDF/PNG files,
  BNT162b2 RNA-freeze and masked protein-outcome records, corrected-v2
  GSE171964 eligibility outputs and download manifests, compact
  stability/resolution summaries, the schema-valid canonical E0/protein/GSE
  claim contract, parallel-evidence/replication schemas, and the focused
  81-test evidence.
- `ted-bib-companion-v1.1.0-native-outputs.zip`: exactly the 2,400 paths named
  by the native-output manifest. The builder reads those rows directly; it
  never scans the approximately 17 GB result tree.
- `ted-bib-companion-v1.1.0-stability-shards.zip`: exactly the paths named by
  the reconciled stability manifest, kept separate from the core archive.

Each archive contains `PACKAGE_MANIFEST.tsv`, which records every other member
including any nested input manifest. The package manifest excludes itself to
avoid an impossible recursive hash. `ARCHIVE_MANIFEST.tsv` records all three
archives and `BUILD_INDEX.json`; it likewise excludes itself.

The historical stability manifest is retained as
`manifest.historical.tsv`. Its `rerun_stdout.log` row records an empty file,
whereas the preserved data-root log is 2,271 bytes with SHA-256
`fcc1aab736ae33d5cf237e09be2841774a5b42d5cf0cd143f05133d06362eab9`.
`STABILITY_SHARDS_MANIFEST.tsv` records both the historical values and the
reconciled current bytes; it is the fail-closed member selector for v1.1.0.
No historical evidence file is edited.

## Candidate build

First project the byte-preserved BNT162b2/GSE171964 status files into the
canonical v1.1 schema instances. This is a deterministic, fail-closed adapter;
it does not recompute the RNA, protein, or replication analyses:

```powershell
python scripts/build_bib_companion_evidence_contracts.py `
  --data-root G:\pyfgsea `
  --schema-root G:\pyfgsea-ted-bib-companion\schemas
```

First generate and freeze the exact focused-test evidence described by
`FOCUSED_81_EVIDENCE.template.tsv`. The default allowlist intentionally treats
all four evidence files as required and fails if they are absent.

The evidence location is deliberately singular: run the locked test selection
from a clean checkout of the intended analysis-lock commit, then materialize
the collection list, JUnit XML, terminal summary, and command/commit record
directly under
`<repository-root>/results/ted_bib_focused_81/`. The builder has no separate
evidence-root fallback, so logs from the data workspace or an ad-hoc working
directory cannot silently satisfy the release gate. The command record must
name the same 40-character source commit that will become the analysis lock.
This directory is intentionally Git-ignored: the evidence is an external
release input generated *from* the clean analysis lock, not a self-referential
member of that commit.

```powershell
python scripts/build_ted_bib_companion.py `
  --repository-root G:\pyfgsea-ted-bib-companion `
  --data-root G:\pyfgsea `
  --analysis-lock-commit <40-character-clean-HEAD> `
  --asset-spec release\ted-v1.1.0\COMPANION_ASSET_RULES.template.tsv `
  --output-dir G:\release-assets\ted-bib-companion-v1.1.0 `
  --dry-run
```

`--dry-run` recomputes every nested-manifest SHA-256, including all native
outputs, and performs no writes. It is intentionally not a fast existence-only
check. The supplied analysis-lock must equal `--repository-root` HEAD; the
repository and tracked asset specification must be clean. The same full commit
is embedded in `BUILD_INDEX.json` and every archive's
`PACKAGE_METADATA.json`, and the verifier requires all copies to agree.

After the dry run passes, omit `--dry-run` to build. The builder refuses to
overwrite an existing output directory and verifies the completed bundle before
returning success.

```powershell
python scripts/build_ted_bib_companion.py `
  --output-dir G:\release-assets\ted-bib-companion-v1.1.0 `
  --verify-only

python scripts/verify_ted_bib_companion.py `
  G:\release-assets\ted-bib-companion-v1.1.0
```

Both verification paths recompute the outer and inner hashes, reject missing or
unmanifested members, cross-check the nested 2,400-output and stability
manifests, and enforce the 480-task, Figure 3/5, schema, case-study, and
focused-81 contracts.

The case-study gate validates the canonical instances in
`results/ted_bib_companion_evidence_contract_v1/` against the standalone
parallel-evidence and replication-facet schemas. It requires the observed
combination `E0 / protein outcome passed / eligibility failed / test not_run /
event not_evaluable / outcome not_tested`. Frozen E/V fields and hypothetical
success strings remain provenance only; see `CLAIM_BOUNDARY.md`.

## Figure reproduction status

The clean companion entry is:

```powershell
python -m pip install -r requirements-reproduction-py311.txt

python reproduce\verify_and_reproduce_figures.py `
  G:\release-assets\ted-bib-companion-v1.1.0 --redraw
```

The command first verifies the source-to-final-figure integrity chain, then
runs the packaged `reproduction/FIGURE_RENDERERS.json` argument-array contract.
It redraws Figure 3 from the four frozen source tables and Figure 5 from the
seven frozen source tables plus the three serialized status JSON records.
Semantic checks bind the headline values and evidence states; PNG pixel QA
rejects blank or undersized output. The report records generated and reference
hashes but does not require cross-environment PDF/PNG byte identity.

Running without `--redraw` remains a pure integrity check. The explicit
`not_available_renderer_contract_not_packaged` status is retained only to
diagnose an incomplete custom candidate; a conforming v1.1.0 bundle must
include the renderer, contract, entry point, and dependency lock.

## Publication gates

Do not publish the companion until all of the following hold:

1. the four focused-test evidence files independently certify exactly
   81 passed tests and no failures or errors;
2. a dry run and a completed-bundle verification pass from the same immutable
   analysis-lock commit;
3. `EXTERNAL_ARCHIVE_ASSETS.template.tsv` is copied to a final manifest and
   filled with the exact file-asset sizes and SHA-256 values; the resulting
   `BUILD_INDEX.json` and `RELEASE_METADATA.json` bind those assets to the
   immutable analysis-lock commit;
4. `RELEASE_METADATA.json` records a newly reserved v1.1.0 Zenodo version DOI
   and the exact-tag source-attestation gate succeeds; and
5. the GitHub release and Zenodo deposit use `ted-v1.1.0`, not the immutable
   baseline tag or DOI.
