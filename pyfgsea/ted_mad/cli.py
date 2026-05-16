"""Command line interface for TED-MAD/ARD."""

from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd

from .core import (
    adjudicate_mechanism,
    build_provenance,
    design_rescue_experiments,
    generate_decision_report,
    interpret_rescue_results,
    load_design_bundle,
    load_posterior_bundle,
    load_ted_mad_yaml,
    merge_provenance,
    write_adjudication_outputs,
    write_design_outputs,
    write_interpretation_outputs,
    write_report_outputs,
)
from .benchmark import (
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


@click.group(name="ted-mad")
def cli() -> None:
    """TED-MAD/ARD mechanism adjudication and active rescue design."""


@cli.group("benchmark")
def benchmark_group() -> None:
    """Reviewer-facing TED-MAD benchmark suites."""


@benchmark_group.command("synthetic")
@click.option("--out", "outdir", default="ted_mad_results/benchmark/synthetic", show_default=True)
@click.option("--n-replicates", default=5, show_default=True, type=int)
@click.option("--seed", default=20260513, show_default=True, type=int)
@click.option(
    "--correlated-evidence/--no-correlated-evidence",
    default=True,
    show_default=True,
    help="Inject repeated same-source evidence to compare naive and dependency-aware fusion.",
)
def benchmark_synthetic_cmd(
    outdir: str,
    n_replicates: int,
    seed: int,
    correlated_evidence: bool,
) -> None:
    """Run synthetic mechanism recovery and double-counting benchmark."""

    results = run_synthetic_mechanism_benchmark(
        n_replicates=n_replicates,
        correlated_evidence=correlated_evidence,
        random_seed=seed,
    )
    paths = write_benchmark_outputs(results, outdir)
    metrics = results["synthetic_metrics"].iloc[0]
    click.echo(
        "Synthetic benchmark: "
        f"top-1 recovery={metrics['top1_hypothesis_recovery']:.3f}; "
        f"false claim upgrade={metrics['false_claim_upgrade_rate']:.3f}; "
        f"mean naive overconfidence delta={metrics['mean_naive_overconfidence_delta']:.3f}"
    )
    click.echo(json.dumps(paths, indent=2))


@benchmark_group.command("hide-evidence")
@click.argument("evidence_yaml", type=click.Path(exists=True, dir_okay=False))
@click.argument("hypotheses_yaml", type=click.Path(exists=True, dir_okay=False))
@click.argument("experiments_yaml", type=click.Path(exists=True, dir_okay=False))
@click.option("--event", default=None, help="Optional target_event to adjudicate.")
@click.option(
    "--families",
    default=None,
    help="Comma-separated evidence families to hide. Defaults to all observed families.",
)
@click.option(
    "--out",
    "outdir",
    default="ted_mad_results/benchmark/evidence_hiding",
    show_default=True,
)
def benchmark_hide_evidence_cmd(
    evidence_yaml: str,
    hypotheses_yaml: str,
    experiments_yaml: str,
    event: str | None,
    families: str | None,
    outdir: str,
) -> None:
    """Run retrospective evidence-family hiding benchmark."""

    hidden_families = [item.strip() for item in families.split(",") if item.strip()] if families else None
    table = run_retrospective_evidence_hiding_benchmark(
        evidence_yaml,
        hypotheses_yaml,
        experiments_yaml,
        event=event,
        hide_families=hidden_families,
    )
    paths = write_benchmark_outputs({"retrospective_evidence_hiding": table}, outdir)
    click.echo(f"Evidence-hiding benchmark: {len(table)} families evaluated")
    click.echo(json.dumps(paths, indent=2))


@benchmark_group.command("negative-control")
@click.option(
    "--out",
    "outdir",
    default="ted_mad_results/benchmark/negative_control",
    show_default=True,
)
@click.option("--n-controls", default=6, show_default=True, type=int)
@click.option("--seed", default=20260513, show_default=True, type=int)
@click.option("--promotion-threshold", default=0.5, show_default=True, type=float)
def benchmark_negative_control_cmd(
    outdir: str,
    n_controls: int,
    seed: int,
    promotion_threshold: float,
) -> None:
    """Run negative-control false mechanism promotion benchmark."""

    results = run_negative_control_benchmark(
        n_controls=n_controls,
        random_seed=seed,
        promotion_threshold=promotion_threshold,
    )
    paths = write_benchmark_outputs(results, outdir)
    metrics = results["negative_control_metrics"].iloc[0]
    click.echo(
        "Negative-control benchmark: "
        f"false mechanism promotion={metrics['false_mechanism_promotion_rate']:.3f}; "
        f"mean max mechanism posterior={metrics['max_mechanism_posterior_mean']:.3f}"
    )
    click.echo(json.dumps(paths, indent=2))


@cli.command("adjudicate")
@click.argument("evidence_yaml", type=click.Path(exists=True, dir_okay=False))
@click.argument("hypotheses_yaml", type=click.Path(exists=True, dir_okay=False), required=False)
@click.option("--event", default=None, help="Optional target_event to adjudicate.")
@click.option("--out", "outdir", default="ted_mad_results/adjudication", show_default=True)
@click.option("--seed", default=20260513, show_default=True, type=int, help="Recorded random seed.")
@click.option("--sensitivity", is_flag=True, help="Run posterior sensitivity analysis.")
@click.option("--prior-grid", is_flag=True, help="Perturb each hypothesis prior up/down.")
@click.option("--weight-jitter", default=0.0, show_default=True, type=float, help="Uniform evidence weight jitter fraction.")
@click.option("--n-bootstrap", default=0, show_default=True, type=int, help="Evidence-family bootstrap iterations.")
@click.option(
    "--leave-one-family-out/--no-leave-one-family-out",
    default=True,
    show_default=True,
    help="Write LOFO claim and posterior summary.",
)
@click.option("--compare-naive", is_flag=True, help="Compare dependency-aware fusion to naive item-level fusion.")
def adjudicate_cmd(
    evidence_yaml: str,
    hypotheses_yaml: str | None,
    event: str | None,
    outdir: str,
    seed: int,
    sensitivity: bool,
    prior_grid: bool,
    weight_jitter: float,
    n_bootstrap: int,
    leave_one_family_out: bool,
    compare_naive: bool,
) -> None:
    """Compute posterior, evidence waterfall, and claim ceiling."""

    evidence = load_ted_mad_yaml(evidence_yaml)
    hypotheses = load_ted_mad_yaml(hypotheses_yaml) if hypotheses_yaml else None
    if hypotheses is None:
        raise click.ClickException("Missing required hypotheses YAML for strict TED-MAD validation")
    try:
        hypothesis_ids = validate_hypotheses(hypotheses)
        validate_evidence(evidence, hypothesis_ids)
    except TedMadValidationError as exc:
        raise click.ClickException(str(exc)) from exc
    provenance = build_provenance(
        {"evidence": evidence_yaml, "hypotheses": hypotheses_yaml},
        random_seed=seed,
    )
    result = adjudicate_mechanism(
        evidence,
        hypotheses,
        event=event,
        strict=True,
        provenance=provenance,
        sensitivity=sensitivity,
        prior_grid=prior_grid,
        weight_jitter=weight_jitter,
        n_bootstrap=n_bootstrap,
        leave_one_family_out=leave_one_family_out,
        compare_naive=compare_naive,
        random_seed=seed,
    )
    paths = write_adjudication_outputs(result, outdir)
    leading = result["posterior"].iloc[0]
    click.echo(
        f"Leading hypothesis: {leading['hypothesis']} "
        f"({leading['posterior']:.3f}); claim ceiling: "
        f"{result['claim_ceiling']['current_level']}"
    )
    click.echo(json.dumps(paths, indent=2))


@cli.command("design")
@click.argument("posterior", type=click.Path(exists=True, dir_okay=False))
@click.argument("experiments_yaml", type=click.Path(exists=True, dir_okay=False))
@click.option("--claim-ceiling", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--out", "outdir", default="ted_mad_results/design", show_default=True)
@click.option("--lambda-claim", type=float, default=None, help="Weight for expected claim upgrade.")
@click.option("--gamma-falsification", type=float, default=None, help="Weight for falsification value.")
@click.option("--cost-weight", type=float, default=None, help="Penalty weight for cost.")
@click.option("--risk-weight", type=float, default=None, help="Penalty weight for technical risk.")
@click.option("--seed", default=20260513, show_default=True, type=int, help="Recorded random seed.")
@click.option("--design-sensitivity", is_flag=True, help="Stress-test rescue ranking under cost/risk weights.")
@click.option("--cost-risk-jitter", default=0.2, show_default=True, type=float, help="Cost/risk weight jitter fraction.")
@click.option("--n-design-bootstrap", default=0, show_default=True, type=int, help="Design sensitivity iterations.")
def design_cmd(
    posterior: str,
    experiments_yaml: str,
    claim_ceiling: str | None,
    outdir: str,
    lambda_claim: float | None,
    gamma_falsification: float | None,
    cost_weight: float | None,
    risk_weight: float | None,
    seed: int,
    design_sensitivity: bool,
    cost_risk_jitter: float,
    n_design_bootstrap: int,
) -> None:
    """Rank rescue experiments by EIG, claim gain, falsification, cost, and risk."""

    posterior_bundle = load_posterior_bundle(posterior)
    experiments = load_ted_mad_yaml(experiments_yaml)
    try:
        hypothesis_ids = {str(row["hypothesis"]) for row in posterior_bundle["posterior"]}
        validate_experiments(experiments, hypothesis_ids)
    except TedMadValidationError as exc:
        raise click.ClickException(str(exc)) from exc
    claim = None
    if claim_ceiling:
        claim = json.loads(Path(claim_ceiling).read_text(encoding="utf-8"))
    provenance = build_provenance(
        {"experiments": experiments_yaml},
        random_seed=seed,
    )
    result = design_rescue_experiments(
        posterior_bundle,
        experiments,
        claim_ceiling=claim,
        strict=True,
        provenance=provenance,
        design_sensitivity=design_sensitivity,
        cost_risk_jitter=cost_risk_jitter,
        n_design_bootstrap=n_design_bootstrap,
        random_seed=seed,
        lambda_claim=lambda_claim,
        gamma_falsification=gamma_falsification,
        cost_weight=cost_weight,
        risk_weight=risk_weight,
    )
    paths = write_design_outputs(result, outdir)
    best = result["next_best_experiment"]
    click.echo(
        f"Next best experiment: {best['experiment_id']} {best['name']} "
        f"(utility={best['utility']:.3f})"
    )
    click.echo(json.dumps(paths, indent=2))


@cli.command("report")
@click.argument("posterior", type=click.Path(exists=True, dir_okay=False))
@click.argument("design", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--evidence-contribution",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Optional evidence_contribution.csv from adjudicate.",
)
@click.option("--out", "outdir", default="ted_mad_results/report", show_default=True)
@click.option("--title", default="TED Mechanism Decision Report", show_default=True)
@click.option(
    "--format",
    "formats",
    type=click.Choice(["markdown", "html", "pdf"]),
    multiple=True,
    default=("markdown", "html"),
    show_default=True,
    help="Output format. Repeat to request multiple formats.",
)
@click.option("--no-pdf", is_flag=True, help="Deprecated compatibility flag; suppresses PDF output.")
@click.option("--seed", default=20260513, show_default=True, type=int, help="Recorded random seed.")
def report_cmd(
    posterior: str,
    design: str,
    evidence_contribution: str | None,
    outdir: str,
    title: str,
    formats: tuple[str, ...],
    no_pdf: bool,
    seed: int,
) -> None:
    """Generate the reviewer-facing mechanism decision report."""

    posterior_bundle = load_posterior_bundle(posterior)
    design_bundle = load_design_bundle(design)
    evidence_df = pd.read_csv(evidence_contribution) if evidence_contribution else None
    provenance_inputs = {"posterior": posterior, "design": design}
    if evidence_contribution:
        provenance_inputs["evidence_contribution"] = evidence_contribution
    provenance = merge_provenance(
        build_provenance(provenance_inputs, random_seed=seed),
        posterior_bundle.get("provenance", {}),
        design_bundle.get("provenance", {}),
    )
    report = generate_decision_report(
        posterior_bundle,
        design_bundle,
        evidence_contribution=evidence_df,
        title=title,
        provenance=provenance,
    )
    selected_formats = tuple(fmt for fmt in formats if fmt != "pdf" or not no_pdf)
    paths = write_report_outputs(
        report,
        outdir,
        formats=selected_formats,
        write_pdf=not no_pdf,
    )
    click.echo(json.dumps(paths, indent=2))


@cli.command("update")
@click.argument("posterior", type=click.Path(exists=True, dir_okay=False))
@click.argument("design", type=click.Path(exists=True, dir_okay=False))
@click.argument("observed_rescue_yaml", type=click.Path(exists=True, dir_okay=False))
@click.option("--out", "outdir", default="ted_mad_results/update", show_default=True)
@click.option("--seed", default=20260513, show_default=True, type=int, help="Recorded random seed.")
def update_cmd(posterior: str, design: str, observed_rescue_yaml: str, outdir: str, seed: int) -> None:
    """Update mechanism posterior from observed rescue readouts."""

    posterior_bundle = load_posterior_bundle(posterior)
    design_bundle = load_design_bundle(design)
    observed = load_ted_mad_yaml(observed_rescue_yaml)
    try:
        validate_observed_rescue(observed)
    except TedMadValidationError as exc:
        raise click.ClickException(str(exc)) from exc
    provenance = merge_provenance(
        build_provenance(
            {"posterior": posterior, "design": design, "observed_rescue": observed_rescue_yaml},
            random_seed=seed,
        ),
        posterior_bundle.get("provenance", {}),
        design_bundle.get("provenance", {}),
    )
    result = interpret_rescue_results(
        posterior_bundle,
        design_bundle,
        observed,
        strict=True,
        provenance=provenance,
    )
    paths = write_interpretation_outputs(result, outdir)
    leading = result["updated_posterior"].iloc[0]
    click.echo(
        f"Updated leading hypothesis: {leading['hypothesis']} "
        f"({leading['updated_posterior']:.3f}); claim ceiling: "
        f"{result['updated_claim_ceiling']['updated_level']}"
    )
    click.echo(json.dumps(paths, indent=2))


@cli.command("interpret")
@click.argument("posterior", type=click.Path(exists=True, dir_okay=False))
@click.argument("design", type=click.Path(exists=True, dir_okay=False))
@click.argument("observed_rescue_yaml", type=click.Path(exists=True, dir_okay=False))
@click.option("--out", "outdir", default="ted_mad_results/interpret", show_default=True)
@click.option("--seed", default=20260513, show_default=True, type=int, help="Recorded random seed.")
def interpret_cmd(posterior: str, design: str, observed_rescue_yaml: str, outdir: str, seed: int) -> None:
    """Alias for update: interpret observed rescue readouts."""

    update_cmd.callback(posterior, design, observed_rescue_yaml, outdir, seed)


@cli.command("init-example")
@click.option("--out", "outdir", default="examples/ted_mad", show_default=True)
def init_example_cmd(outdir: str) -> None:
    """Write minimal TED-MAD example YAML files."""

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    source_dir = Path(__file__).resolve().parents[2] / "examples" / "ted_mad"
    if source_dir.exists():
        for name in (
            "hypotheses.yaml",
            "evidence.yaml",
            "experiments.yaml",
            "observed_rescue_supports_h3.yaml",
            "observed_rescue_supports_h4.yaml",
        ):
            (out / name).write_text((source_dir / name).read_text(encoding="utf-8"), encoding="utf-8")
        click.echo(f"Wrote TED-MAD example files to {out}")
        return
    examples = {
        "hypotheses.yaml": _example_hypotheses(),
        "evidence.yaml": _example_evidence(),
        "experiments.yaml": _example_experiments(),
    }
    for name, text in examples.items():
        (out / name).write_text(text, encoding="utf-8")
    click.echo(f"Wrote TED-MAD example files to {out}")


def _example_hypotheses() -> str:
    return """model:
  base_likelihood_ratio: 2.0
  dependency_aggregation: mean
  max_family_abs_log_lr: 2.5
hypotheses:
  H0:
    label: noise/batch/family artifact
    prior: 0.08
  H1:
    label: state/composition artifact
    prior: 0.12
  H2:
    label: proliferation-confounded mechanism
    prior: 0.15
  H3:
    label: GATA1-regulatory mechanism
    prior: 0.30
  H4:
    label: downstream heme/maturation mechanism
    prior: 0.18
  H5:
    label: T21-specific chromatin/GATA1 interaction
    prior: 0.17
"""


def _example_evidence() -> str:
    return """evidence:
  - evidence_id: E1a
    evidence_family: E1 family-level block robustness
    target_event: gata1_erythroid_ted
    effect_size: 1.4
    uncertainty: 0.7
    data_source: internal TED family-block analysis
    assumption: donor/family blocks capture major sample structure
    failure_mode: hidden batch structure aligned with condition
    which_hypotheses_it_supports: [H3, H5]
    which_hypotheses_it_weakens: [H0]
    dependency_group: family_block
  - evidence_id: E2a
    evidence_family: E2 proliferation-adjusted mediation
    target_event: gata1_erythroid_ted
    effect_size: 1.0
    uncertainty: 0.8
    data_source: internal mediation analysis
    assumption: proliferation score captures main cell-cycle axis
    failure_mode: unmeasured cycling state remains
    which_hypotheses_it_supports: [H3]
    which_hypotheses_it_weakens: [H2]
    dependency_group: adjusted_causal
  - evidence_id: E3a
    evidence_family: E3 counterfactual OT event effect
    target_event: gata1_erythroid_ted
    effect_size: 1.3
    uncertainty: 0.8
    data_source: counterfactual OT
    assumption: transport features cover matched developmental state
    failure_mode: poor overlap between compared states
    which_hypotheses_it_supports: [H3, H5]
    which_hypotheses_it_weakens: [H1]
    dependency_group: adjusted_causal
  - evidence_id: E4a
    evidence_family: E4 day-stratified timing
    target_event: gata1_erythroid_ted
    effect_size: 1.1
    uncertainty: 0.7
    data_source: day-stratified TED timing
    assumption: sampled days bracket the relevant onset
    failure_mode: onset lies between sampled days
    which_hypotheses_it_supports: [H3]
    which_hypotheses_it_weakens: [H4]
    dependency_group: timing
  - evidence_id: E5a
    evidence_family: E5 negative mediator controls
    target_event: gata1_erythroid_ted
    strength: 1.0
    data_source: shuffled and implausible mediator controls
    assumption: controls represent non-mechanistic alternatives
    failure_mode: control set misses relevant confounder
    which_hypotheses_it_supports: [H3, H5]
    which_hypotheses_it_weakens: [H0, H1, H2]
    dependency_group: controls
  - evidence_id: E7a
    evidence_family: E7 rescue prediction table
    target_event: gata1_erythroid_ted
    strength: 0.8
    data_source: internal rescue-readout prediction
    assumption: compact readouts cover the mechanism-relevant state
    failure_mode: rescue affects unmeasured readout
    which_hypotheses_it_supports: [H3]
    which_hypotheses_it_weakens: []
    dependency_group: rescue_prediction
  - evidence_id: E8a
    evidence_family: E8 external GATA1 KD support
    target_event: gata1_erythroid_ted
    strength: 1.1
    data_source: external GATA1 perturbation support
    assumption: perturbation direction maps onto this differentiation context
    failure_mode: external context differs from T21 differentiation
    which_hypotheses_it_supports: [H3]
    which_hypotheses_it_weakens: [H4]
    dependency_group: external_gata1
  - evidence_id: E9a
    evidence_family: E9 external T21 multiome support
    target_event: gata1_erythroid_ted
    strength: 1.0
    data_source: external T21 multiome support
    assumption: accessibility changes are comparable across systems
    failure_mode: multiome cohort captures a different stage
    which_hypotheses_it_supports: [H5, H3]
    which_hypotheses_it_weakens: [H0]
    dependency_group: external_t21
"""


def _example_experiments() -> str:
    return """design_model:
  lambda_claim: 1.0
  gamma_falsification: 0.6
  cost_weight: 0.15
  risk_weight: 0.25
experiments:
  - experiment_id: A1
    name: full-length GATA1 rescue at D7, read D9/D11
    description: Restore full-length GATA1 before the predicted regulatory defect.
    cost: medium
    risk: medium
    claim_upgrade_evidence: direct rescue
    claim_level_if_success: 4
    supports_hypotheses: [H3]
    readouts:
      - D9 erythroid regulatory module
      - D9 GATA1 target module
      - D11 maturation module
      - D11 heme/hemoglobinization
      - event-level TED score
    expected_patterns:
      H3:
        D9 regulatory module: strong rescue
        D9 GATA1 target module: strong rescue
        D11 maturation/heme: partial-to-strong rescue
        TED event score: decreases
      H4:
        D9 regulatory module: weak rescue
        D11 maturation/heme: partial downstream improvement
        TED event score: incomplete decrease
      H5:
        D9 regulatory module: partial rescue
        chromatin-linked targets: incomplete rescue
        TED event score: partial decrease
      H1:
        state-matched signal: unstable or nonspecific
      H2:
        proliferation readout: not normalized
      H0:
        replicate pattern: inconsistent
    falsifies:
      hypothesis: H3
      rule: If full-length GATA1 is restored but the D9 regulatory module and event-level TED score do not rescue, H3 should drop substantially.
  - experiment_id: A4
    name: hemin / heme pathway rescue
    description: Test whether downstream hemoglobinization is the primary bottleneck.
    cost: low
    risk: low
    claim_level_if_success: 3.5
    supports_hypotheses: [H4]
    readouts:
      - heme/hemoglobinization
      - maturation module
      - D9 regulatory module
      - event-level TED score
    expected_patterns:
      H3:
        D9 regulatory module: weak rescue
        heme/hemoglobinization: partial late improvement
        TED event score: incomplete decrease
      H4:
        heme/hemoglobinization: strong rescue
        maturation module: strong rescue
        D9 regulatory module: weak rescue
      H5:
        heme/hemoglobinization: partial rescue
        chromatin-linked targets: no rescue
      H1:
        pattern: nonspecific
      H2:
        proliferation readout: unchanged
      H0:
        replicate pattern: inconsistent
    falsifies:
      hypothesis: H4
      rule: If heme readouts do not improve despite adequate exposure, H4 loses support.
  - experiment_id: A6
    name: sorted state-matched progenitor comparison
    description: Directly test whether the event survives state matching.
    cost: high
    risk: medium
    claim_level_if_success: 3
    supports_hypotheses: [H3, H5]
    readouts:
      - state composition
      - GATA1 target module
      - event-level TED score
    expected_patterns:
      H3:
        state-matched TED event: persists
        GATA1 target module: reduced
      H1:
        state-matched TED event: disappears
        GATA1 target module: nonspecific
      H5:
        state-matched TED event: persists with chromatin specificity
      H2:
        proliferation readout: explains residual signal
      H4:
        heme/hemoglobinization: later-only defect
      H0:
        replicate pattern: inconsistent
    falsifies:
      hypothesis: H1
      rule: If the event persists after stringent state matching, H1 becomes unlikely.
"""


if __name__ == "__main__":
    cli()
