"""TED-MAD/ARD mechanism adjudication and active rescue design."""

from .core import (
    adjudicate_mechanism,
    build_provenance,
    design_rescue_experiments,
    generate_decision_report,
    interpret_rescue_results,
    load_ted_mad_yaml,
    merge_provenance,
    write_adjudication_outputs,
    write_design_outputs,
    write_interpretation_outputs,
    write_report_outputs,
)
from .benchmark import (
    make_benchmark_experiments,
    make_benchmark_hypotheses,
    make_synthetic_evidence,
    run_negative_control_benchmark,
    run_retrospective_evidence_hiding_benchmark,
    run_synthetic_mechanism_benchmark,
    write_benchmark_outputs,
)
from .schema import (
    TedMadValidationError,
    validate_evidence,
    validate_experiments,
    validate_hypotheses,
    validate_observed_rescue,
)

__all__ = [
    "adjudicate_mechanism",
    "build_provenance",
    "design_rescue_experiments",
    "generate_decision_report",
    "interpret_rescue_results",
    "load_ted_mad_yaml",
    "merge_provenance",
    "make_benchmark_experiments",
    "make_benchmark_hypotheses",
    "make_synthetic_evidence",
    "run_negative_control_benchmark",
    "run_retrospective_evidence_hiding_benchmark",
    "run_synthetic_mechanism_benchmark",
    "TedMadValidationError",
    "validate_evidence",
    "validate_experiments",
    "validate_hypotheses",
    "validate_observed_rescue",
    "write_adjudication_outputs",
    "write_design_outputs",
    "write_benchmark_outputs",
    "write_interpretation_outputs",
    "write_report_outputs",
]
