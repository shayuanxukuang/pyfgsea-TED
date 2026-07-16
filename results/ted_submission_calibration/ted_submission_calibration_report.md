## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-07-15
- Verification Status: ANALYZED
- Version Label: ted_submission_calibration_v1

# TED submission calibration report

- Packets: 160; five rule-perturbation sensitivity profiles.
- TED vs controlled synthetic packet-class truth macro-F1: 0.760.
- Evidence-tier false-upgrade rate: 0.056.
- Event-FDR is reported by nominal q with empirical FDP, 95% Monte Carlo intervals and the worst configuration; pass counts are secondary.
- Relative calibration criterion: upper 95% bound <= (1 + 0.50) x q.
- Packet-class compatibility-set coverage under shift: 0.875.

This is an internal benchmark with controlled synthetic truth. The rule-perturbation sensitivity profiles are supplementary sensitivity analyses only, not an independent truth source or external validation.
The ten legacy labels are controlled packet classes spanning biological mode, artifact, identifiability and V-provenance domains; they are not ten biological event modes.
The packet-class compatibility sets are rule-defined sets, not conformal prediction sets.
No result in this package upgrades GSE271399/T21/GATA1 above Level 3.5.

Simulation settings: packets_per_class=16, FDR replicates=10000 (q=0.01), 5000 (q=0.05), or 500 (q=0.10/0.20); confounded-null replicates=500, confounded-signal replicates=500, seed=20260715.
