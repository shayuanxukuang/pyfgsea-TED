# Quality

## Definition of Done
- The relevant script or library path runs locally with the declared project dependencies.
- New or changed TED outputs include enough provenance to identify input files, accession, source URL, and claim boundary.
- Tests are added or run in proportion to risk. For TED core logic, prefer `pytest` coverage; for public data workflows, prefer manifest plus summary table checks.
- Manuscript-facing claims match the machine-readable claim ceiling.

## Review Checklist
- Does the output distinguish event support from functional rescue?
- Are negative controls or specificity checks present when a dataset could be confounded by stress, proliferation, composition, or batch?
- Are skipped files and large archives explicitly recorded?
- Are thresholds and gene sets locked before expression scoring for positive-stratum claims?
- Are paths and generated artifacts scoped to `data_external/`, `results/`, or submission folders rather than hidden local locations?

## Suggested Smoke Tests
```powershell
python -m pytest tests/test_ted_perturbation.py tests/test_ted_mad.py tests/test_trajectory_features.py
python scripts/run_ted_benchmark.py --profile tiny --suite core,trajectory,rankers,windows
```

## External Data Checks
For public data additions, produce at least:
- downloadability or file manifest;
- dataset suitability/claim-boundary table;
- concise summary table with pass/fail gates;
- explicit allowed and forbidden claims.

