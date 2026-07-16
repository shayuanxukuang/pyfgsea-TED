# Trajectory Pathway Event Discovery (TED)

TED is a structured evidence protocol for dynamic pathway interpretation in single-cell genomics. It converts pathway, module, or perturbation activity profiles into auditable event objects with explicit biological-mode, artifact, identifiability, within-study E-support, and external V-provenance fields.

This repository is the software and machine-readable evidence companion for the manuscript. PyFgsea is included as an optional Rust-backed upstream activity engine; TED also accepts activity matrices from other methods.

## Release identity

- Release tag: `ted-v1.0.0`
- Software version: `1.0.0`
- License: MIT

The version-specific Zenodo DOI is recorded here only after Zenodo has reserved or minted it. No provisional DOI is used.

## Evidence contract

TED separates five targets that must not be collapsed into one label:

1. biological event mode: activation, suppression, delay, loss, or redirection;
2. artifact classification: none, composition, stress, or other declared artifacts;
3. identifiability: identifiable, ambiguous, or not identifiable;
4. within-study support: E0, E1, or E2 with a reason code and test status;
5. external provenance: outcome, reversal, rescue, or replication tags.

Controlled simulations use the neutral term **controlled packet class** only for legacy combined-class comparisons. Rule variations are reported as **rule-perturbation sensitivity profiles**.

## Quick start

```bash
git clone https://github.com/shayuanxukuang/pyfgsea-TED.git
cd pyfgsea-TED
python -m pip install -e ".[dev]"
python -I scripts/run_ted_validation_demo.py --outdir ted_demo
ted validate ted_demo/demo_activity.tsv --kind activity --report ted_demo/activity_validation.tsv
ted validate ted_demo/demo_events_v2.tsv --kind event --report ted_demo/event_validation.tsv
```

## Verified test groups

The release suite is divided into independently bounded jobs:

```bash
pytest -q -m "not integration and not slow and not external_data"
pytest -q -m integration
pytest -q -m slow
pytest -q -m external_data
```

Every job has a 120-second per-test timeout in CI. Public-data tests use archived, fixed local fixtures and do not download data during test execution.

## Containers

```bash
docker build --no-cache -t ted:1.0.0 .
docker build --no-cache -f Dockerfile.baselines -t ted-baselines:1.0.0 .
docker run --rm ted:1.0.0 ted --help
docker run --rm ted:1.0.0
```

The main image runs the release validation demo by default. The baseline image runs the quick direct external-baseline suite.

## Main release evidence

| Path | Purpose |
| --- | --- |
| `results/ted_adaptive_window_multiplicity/` | Full adaptive-window search simulation with per-event max-window + BH, family-wide maxT, FDR, FWER, power, and timing error. |
| `results/ted_factorized_ablation/` | Factorized biological-mode, artifact, identifiability, E-support, V-provenance, gate-ablation, reason-code, and schema audits. |
| `results/ted_current_task_benchmark/` | Current benchmark and supervised baseline comparisons with evidence-risk metrics. |
| `results/ted_real_data_upstream_sensitivity/` | Real-data upstream-method disagreement and E2 fail-closed checks. |
| `results/ted_submission_calibration/` | Controlled truth, E benchmark, external outcome, baseline comparison, and rule-perturbation sensitivity profiles. |
| `results/ted_post_freeze_protocol_candidate/` | Draft, not-activated protocol candidate; not evidence of prospective validation. |
| `results/bib_manuscript_revision/figure_source_data/` | Machine-readable source data for manuscript figures. |
| `schemas/ted_event_report_v2.schema.json` | E0/test-status compatible event-object schema. |
| `RELEASE_MANIFEST.tsv` | File sizes and SHA256 checksums for the release tree. |

## Claim boundaries

TED is not presented as the most accurate packet classifier. Supervised classifiers optimize packet-level prediction, whereas TED provides a fail-closed evidence contract with abstention, reason codes, explicit evidence boundaries, and auditable external provenance. GSE271399 and related GATA1/T21 evidence remain below a direct matched-rescue claim unless the predeclared functional gate passes.

## Citation

Use the version-specific Zenodo DOI listed in this section after final publication of `ted-v1.0.0`, together with the journal article when available.
