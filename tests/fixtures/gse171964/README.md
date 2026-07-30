# GSE171964 contract fixtures

These two compact fixtures exercise the frozen corrected-v2 time mapping and
feature-panel parser without requiring the 1.47 GB public count matrix during
package CI.

- `sample_sheet_contract.tsv` contains only the 6 participant identifiers and
  four frozen booster-episode days needed by the contract test. It is derived
  from `GSE171964_geo_pheno_v2.csv.gz` (source SHA-256
  `04eec09116d50b29100a7f3056cde375e39c44f4a08b782bb8ad11db404e91a0`).
- `feature_panel_contract.tsv` uses the public file's quoted-vector syntax and
  contains two RNA features that must be present. The contract test also
  asserts that the unavailable CD64/CD169 ADT names are absent. The full source
  is `GSE171964_feats_v2.tsv.gz` (source SHA-256
  `ed7f73e13048211f4ec40aa0faeee8fb1cb2c961e68a704a9756be42557ee6eb`).

The release companion separately carries and verifies the complete download
manifest. These fixtures are parser/design-contract inputs, not substitutes
for the public analysis data and not evidence of an evaluable replication.
