# pyfgsea-TED

Public software/data companion for TED: an evidence-gated framework for dynamic pathway event interpretation in single-cell genomics.

This repository intentionally excludes article manuscripts, cover letters, compiled submission PDFs, LaTeX submission packages, and journal-specific internal documents. It contains only code, configuration, tests, selected result tables, figure source data, and reproducibility manifests needed for archival release and review.

## Archived release

The manuscript-facing archived release is `ted-gb-rc7`, commit `3ffec1a1dcb4261303fc130b81ccd6b29a2fa34f`, archived at Zenodo DOI [10.5281/zenodo.20378158](https://doi.org/10.5281/zenodo.20378158). The release audit tables in `tables/` use this version-specific DOI and commit.

## Current release content

- Core `pyfgsea`/TED Python package code.
- Known-source validation scripts for GSE153056, GSE93735 and SCP1064.
- SCP1064 lightweight label-shuffle audit (`scripts/223_run_scp1064_lightweight_shuffles.py`) and result tables.
- GATA1/GATA1s cross-dataset support scripts and tables.
- Direct external baseline wrappers and Docker/conda environments.
- Machine-readable event-object, benchmark, claim-boundary, source-data and release-audit tables.
- Main figure PDFs/PNGs and source-data TSV files.
- Benchmark audit table with truth sources, scored units, uncertainty reporting, frozen status and threshold-optimization role.

## Not included

- Manuscript text or PDFs.
- Cover letters.
- Supplementary Information PDFs or LaTeX submission packages.
- Large raw public datasets; raw data remain available from GEO/SCP/STOmicsDB/CNGB accessions listed in the manifests.

## Quick validation

```bash
python -m pytest tests/test_scp1064_file_qc.py tests/test_scp1064_cell_alignment.py tests/test_scp1064_event_outcome_alignment.py tests/test_scp1064_claim_boundary.py tests/test_ted_known_source_validation.py
```

## Key outputs

- `tables/ted_dataset_level_claim_boundary.tsv`
- `tables/scp1064_lightweight_shuffle_summary.tsv`
- `tables/scp1064_specificity_summary.tsv`
- `tables/known_source_boundary_mapping.tsv`
- `tables/dynamic_pathway_event_table.tsv`
- `figures/figure2_known_source_validation.pdf`
- `figures/figure4_gse271399_gata1_cross_dataset_support.pdf`
- `figures/figure5_claim_upgrade_block_audit.pdf`

## License

MIT License.
