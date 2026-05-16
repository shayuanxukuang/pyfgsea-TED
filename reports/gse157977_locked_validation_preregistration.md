# GSE157977 Locked Validation Preregistration

Generated UTC: 2026-05-15T13:35:28.699835+00:00

## Rationale

GSE157977 is used as an independent locked validation of TED's event-and-claim behavior. It is not used to tune TED parameters, choose the GSE271399 mechanism, select algorithm variants, or define the benchmark scoring rules. The validation question is not whether TED proves a neurodevelopmental mechanism. The locked question is whether TED can identify plausible guide-level adapter candidates while downgrading or rejecting guides whose neural/glial signal is weak, low-support, or dominated by negative controls.

## Locked Event Families

- neural_axis_shift_candidate
- glial_axis_shift_candidate
- control_sensitive_or_mixed
- low_support_guide

## Locked Negative Controls

- housekeeping_control
- stress_axis
- proliferation_axis

## Locked Claim Gates

- guide support: n_samples >= 8 and total_cells >= 100
- event strength: max_abs_neural_delta >= 0.15
- negative-control margin: negative_control_margin >= 0.05
- missing guide-target map caps claims at adapter level
- missing cell-state annotation forbids delay/loss/redirection or cell-type-specific fate claims

## Allowed Interpretation

GSE157977 can support independent guide-level in vivo Perturb-seq adapter validation of TED's claim discipline. It can show that TED identifies a small number of neural/glial-axis adapter candidates while downgrading or rejecting control-sensitive guides.

## Forbidden Interpretation

Do not claim validated ASD/NDD gene mechanisms, cell-type-specific fate loss, developmental delay, fate redirection, or functional causality from this locked validation.
