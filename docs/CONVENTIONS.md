# Conventions

## File Naming
- Library files use lowercase snake_case Python modules, for example `trajectory_compare.py` and `ted_perturbation.py`.
- Dataset workflows use `scripts/run_<dataset_or_scope>_<analysis>.py`.
- Output tables are TSV by default and include dataset/accession labels where practical.
- Pre-registered analyses write an explicit plan or hash before expression-level scoring when the workflow is intended as a claim-boundary gate.

## Python Style
- Public functions use snake_case and return pandas DataFrames or dictionaries of DataFrames for reusable TED tables.
- Result containers expose `to_tables()` when multiple tables need to move together.
- Dataset scripts use small helper functions such as `write_tsv`, `sha256`, and `log` rather than hidden notebook state.
- Claims and caveats are represented as machine-readable table fields, not only prose.

## Data Artifacts
- Download manifests should include accession, URL, local path, byte size, and hash when feasible.
- Large raw archives may be skipped when a processed matrix or file-list audit answers the runability question.
- Public proxy datasets must carry explicit forbidden-claim fields when they cannot serve as matched rescue.

## Tests
- Unit and reliability tests live in `tests/`.
- External dataset workflows are script-level reproducibility checks and may be slow; use targeted tests for package behavior.

## Git Notes
The repository may contain many untracked generated artifacts. Do not delete or reset them unless the user explicitly requests cleanup.

