import json
import math
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from pyfgsea.ted_mad import (
    adjudicate_mechanism,
    build_provenance,
    design_rescue_experiments,
    generate_decision_report,
    interpret_rescue_results,
    load_ted_mad_yaml,
    run_negative_control_benchmark,
    run_retrospective_evidence_hiding_benchmark,
    run_synthetic_mechanism_benchmark,
    TedMadValidationError,
    validate_evidence,
    validate_experiments,
    validate_hypotheses,
    write_adjudication_outputs,
    write_design_outputs,
    write_interpretation_outputs,
    write_report_outputs,
)
from pyfgsea.ted_mad.cli import cli as ted_mad_cli


GOLDEN = Path(__file__).resolve().parent / "golden"


def _hypotheses():
    return {
        "hypotheses": {
            "H0": {"label": "noise/batch/family artifact", "prior": 1 / 6},
            "H1": {"label": "state/composition artifact", "prior": 1 / 6},
            "H2": {"label": "proliferation-confounded mechanism", "prior": 1 / 6},
            "H3": {"label": "GATA1-regulatory mechanism", "prior": 1 / 6},
            "H4": {"label": "downstream heme/maturation mechanism", "prior": 1 / 6},
            "H5": {"label": "T21-specific chromatin/GATA1 interaction", "prior": 1 / 6},
        }
    }


def _evidence():
    return {
        "evidence": [
            {
                "evidence_id": "E1a",
                "evidence_family": "E1 family-level block robustness",
                "target_event": "gata1_event",
                "strength": 1.0,
                "which_hypotheses_it_supports": ["H3"],
                "which_hypotheses_it_weakens": ["H0"],
                "dependency_group": "family_block",
            },
            {
                "evidence_id": "E5a",
                "evidence_family": "E5 negative mediator controls",
                "target_event": "gata1_event",
                "strength": 1.0,
                "which_hypotheses_it_supports": ["H3"],
                "which_hypotheses_it_weakens": ["H0", "H1", "H2"],
                "dependency_group": "negative_controls",
            },
            {
                "evidence_id": "E2a",
                "evidence_family": "E2 proliferation-adjusted mediation",
                "target_event": "gata1_event",
                "strength": 1.0,
                "which_hypotheses_it_supports": ["H3"],
                "which_hypotheses_it_weakens": ["H2"],
                "dependency_group": "adjusted_causal",
            },
            {
                "evidence_id": "E3a",
                "evidence_family": "E3 counterfactual OT event effect",
                "target_event": "gata1_event",
                "strength": 1.0,
                "which_hypotheses_it_supports": ["H3"],
                "which_hypotheses_it_weakens": ["H1"],
                "dependency_group": "adjusted_causal",
            },
            {
                "evidence_id": "E4a",
                "evidence_family": "E4 day-stratified timing",
                "target_event": "gata1_event",
                "strength": 1.0,
                "which_hypotheses_it_supports": ["H3"],
                "which_hypotheses_it_weakens": ["H4"],
                "dependency_group": "timing",
            },
            {
                "evidence_id": "E8a",
                "evidence_family": "E8 external GATA1 KD support",
                "target_event": "gata1_event",
                "strength": 1.0,
                "which_hypotheses_it_supports": ["H3"],
                "which_hypotheses_it_weakens": ["H0"],
                "dependency_group": "external",
            },
        ]
    }


def test_adjudication_assigns_l35_without_direct_rescue():
    result = adjudicate_mechanism(_evidence(), _hypotheses(), event="gata1_event")

    posterior = result["posterior"].set_index("hypothesis")
    assert posterior["posterior"].idxmax() == "H3"
    assert result["claim_ceiling"]["current_level_numeric"] == 3.5
    assert "pre-registered matched rescue result" in result["claim_ceiling"][
        "missing_evidence_for_next_level"
    ]
    assert not result["leave_one_evidence_family_out"].empty


def test_dependency_groups_are_not_multiplied_like_independent_evidence():
    evidence = {
        "evidence": [
            {
                "evidence_id": "E1a",
                "evidence_family": "E1 family-level block robustness",
                "strength": 1.0,
                "which_hypotheses_it_supports": ["H3"],
                "dependency_group": "same_block",
            },
            {
                "evidence_id": "E1b",
                "evidence_family": "E1 family-level block robustness",
                "strength": 1.0,
                "which_hypotheses_it_supports": ["H3"],
                "dependency_group": "same_block",
            },
        ]
    }
    result = adjudicate_mechanism(evidence, _hypotheses())
    contrib = result["evidence_contribution"]
    h3 = contrib[
        (contrib["evidence_family"] == "E1 family-level block robustness")
        & (contrib["hypothesis"] == "H3")
    ].iloc[0]

    assert math.isclose(h3["log_likelihood_ratio"], math.log(2.0), rel_tol=1e-6)


def test_active_rescue_design_prefers_falsifiable_claim_upgrading_rescue():
    adjudication = adjudicate_mechanism(_evidence(), _hypotheses(), event="gata1_event")
    posterior_bundle = {
        "posterior": adjudication["posterior"].to_dict(orient="records"),
        "claim_ceiling": adjudication["claim_ceiling"],
    }
    experiments = {
        "experiments": [
            {
                "experiment_id": "A1",
                "name": "full-length GATA1 rescue",
                "cost": "medium",
                "risk": "medium",
                "claim_level_if_success": 4,
                "supports_hypotheses": ["H3"],
                "readouts": ["D9 regulatory module", "D11 maturation", "TED event score"],
                "expected_patterns": {
                    "H3": {"D9 regulatory module": "strong rescue", "TED event score": "decreases"},
                    "H4": {"D9 regulatory module": "weak rescue", "TED event score": "partial"},
                    "H1": {"pattern": "nonspecific"},
                    "H0": {"pattern": "inconsistent"},
                },
                "falsifies": {
                    "hypothesis": "H3",
                    "rule": "If GATA1 is restored but D9 regulatory module does not rescue, H3 drops.",
                },
            },
            {
                "experiment_id": "A4",
                "name": "hemin rescue",
                "cost": "low",
                "risk": "low",
                "supports_hypotheses": ["H4"],
                "expected_patterns": {
                    "H3": {"heme": "partial"},
                    "H4": {"heme": "strong rescue"},
                },
            },
        ]
    }

    design = design_rescue_experiments(posterior_bundle, experiments)

    assert design["next_best_experiment"]["experiment_id"] == "A1"
    assert not design["expected_result_patterns"].empty
    assert "H3 drops" in design["falsification_rules_markdown"]


def test_design_sensitivity_reports_cost_risk_rank_stability():
    adjudication = adjudicate_mechanism(_evidence(), _hypotheses(), event="gata1_event")
    posterior_bundle = {
        "posterior": adjudication["posterior"].to_dict(orient="records"),
        "claim_ceiling": adjudication["claim_ceiling"],
    }
    experiments = {
        "experiments": [
            {
                "experiment_id": "A1",
                "name": "full-length GATA1 rescue",
                "cost": 0.7,
                "risk": 0.4,
                "claim_level_if_success": 4,
                "supports_hypotheses": ["H3"],
                "readouts": ["D9 regulatory module"],
                "expected_patterns": {
                    "H3": {"D9 regulatory module": "strong rescue"},
                    "H4": {"D9 regulatory module": "weak rescue"},
                },
                "falsifies": {"hypothesis": "H3", "rule": "No D9 rescue falsifies H3."},
            },
            {
                "experiment_id": "A4",
                "name": "hemin rescue",
                "cost": 0.2,
                "risk": 0.1,
                "supports_hypotheses": ["H4"],
                "expected_patterns": {
                    "H3": {"heme": "partial"},
                    "H4": {"heme": "strong rescue"},
                },
            },
        ]
    }

    design = design_rescue_experiments(
        posterior_bundle,
        experiments,
        design_sensitivity=True,
        cost_risk_jitter=0.2,
        n_design_bootstrap=20,
        random_seed=3,
    )

    assert not design["design_sensitivity"].empty
    assert not design["design_rank_stability"].empty
    assert {"top_frequency", "median_rank"}.issubset(design["design_rank_stability"].columns)


def test_design_outputs_contrast_matrix_and_minimal_readout_panel():
    hypotheses = load_ted_mad_yaml(GOLDEN / "example_hypotheses.yaml")
    evidence = load_ted_mad_yaml(GOLDEN / "example_evidence.yaml")
    experiments = load_ted_mad_yaml(GOLDEN / "example_experiments.yaml")
    adjudication = adjudicate_mechanism(
        evidence,
        hypotheses,
        event="erythroid_event_001",
        strict=True,
    )
    design = design_rescue_experiments(
        {
            "posterior": adjudication["posterior"].to_dict(orient="records"),
            "claim_ceiling": adjudication["claim_ceiling"],
        },
        experiments,
        strict=True,
    )

    contrast = design["experiment_contrast_matrix"]
    panel = design["minimal_readout_panel"]

    assert "H3_GATA1_regulatory_vs_H4_downstream_heme" in contrast.columns
    assert contrast.iloc[0]["claim_upgrade"] == "medium"
    assert bool(contrast.iloc[0]["falsifies_H3_GATA1_regulatory"])
    assert {"experiment_id", "category", "readout"}.issubset(panel.columns)
    assert "D9_regulatory_module" in set(panel["readout"])


def test_interpret_rescue_results_updates_posterior_and_claim(tmp_path):
    hypotheses = load_ted_mad_yaml(GOLDEN / "example_hypotheses.yaml")
    evidence = load_ted_mad_yaml(GOLDEN / "example_evidence.yaml")
    experiments = load_ted_mad_yaml(GOLDEN / "example_experiments.yaml")
    adjudication = adjudicate_mechanism(
        evidence,
        hypotheses,
        event="erythroid_event_001",
        strict=True,
    )
    posterior_bundle = {
        "posterior": adjudication["posterior"].to_dict(orient="records"),
        "claim_ceiling": adjudication["claim_ceiling"],
    }
    design = design_rescue_experiments(posterior_bundle, experiments, strict=True)
    design_bundle = {
        "expected_result_patterns": design["expected_result_patterns"].to_dict(orient="records"),
        "ranked_experiments": design["ranked_experiments"].to_dict(orient="records"),
    }
    observed = {
        "experiment_id": "A1_GATA1_FL_D7_rescue",
        "quality": 1.0,
        "observed_results": {
            "D9_regulatory_module": "strong_rescue",
            "D11_maturation_module": "partial_rescue",
            "D11_hemoglobinization": "partial_rescue",
        },
    }

    updated = interpret_rescue_results(posterior_bundle, design_bundle, observed, strict=True)
    paths = write_interpretation_outputs(updated, tmp_path / "interpret")

    h3 = updated["updated_posterior"][
        updated["updated_posterior"]["hypothesis"] == "H3_GATA1_regulatory"
    ].iloc[0]
    assert h3["updated_posterior"] > h3["prior_posterior"]
    assert updated["updated_claim_ceiling"]["updated_level_numeric"] == 4.0
    assert "supports the leading mechanism" in updated["interpretation_markdown"]
    assert Path(paths["updated_posterior"]).exists()


def test_interpret_rescue_results_can_weaken_leading_mechanism():
    hypotheses = load_ted_mad_yaml(GOLDEN / "example_hypotheses.yaml")
    evidence = load_ted_mad_yaml(GOLDEN / "example_evidence.yaml")
    experiments = load_ted_mad_yaml(GOLDEN / "example_experiments.yaml")
    adjudication = adjudicate_mechanism(
        evidence,
        hypotheses,
        event="erythroid_event_001",
        strict=True,
    )
    posterior_bundle = {
        "posterior": adjudication["posterior"].to_dict(orient="records"),
        "claim_ceiling": adjudication["claim_ceiling"],
    }
    design = design_rescue_experiments(posterior_bundle, experiments, strict=True)
    design_bundle = {
        "expected_result_patterns": design["expected_result_patterns"].to_dict(orient="records"),
        "ranked_experiments": design["ranked_experiments"].to_dict(orient="records"),
    }
    observed = {
        "experiment_id": "A1_GATA1_FL_D7_rescue",
        "observed_results": {
            "D9_regulatory_module": "no_rescue",
            "D11_hemoglobinization": "strong_rescue",
        },
    }

    updated = interpret_rescue_results(posterior_bundle, design_bundle, observed, strict=True)
    h3 = updated["updated_posterior"][
        updated["updated_posterior"]["hypothesis"] == "H3_GATA1_regulatory"
    ].iloc[0]
    h4 = updated["updated_posterior"][
        updated["updated_posterior"]["hypothesis"] == "H4_downstream_heme"
    ].iloc[0]

    assert h3["updated_posterior"] < h3["prior_posterior"]
    assert h4["updated_posterior"] > h4["prior_posterior"]


def test_output_writers_and_report(tmp_path):
    adjudication = adjudicate_mechanism(_evidence(), _hypotheses(), event="gata1_event")
    adj_paths = write_adjudication_outputs(adjudication, tmp_path / "adjudication")
    posterior_bundle = json.loads(
        (tmp_path / "adjudication" / "posterior.json").read_text(encoding="utf-8")
    )

    design = design_rescue_experiments(
        posterior_bundle,
        {
            "experiments": [
                {
                    "experiment_id": "A1",
                    "name": "full-length GATA1 rescue",
                    "claim_level_if_success": 4,
                    "supports_hypotheses": ["H3"],
                    "readouts": ["D9 regulatory module"],
                    "expected_patterns": {
                        "H3": {"D9 regulatory module": "strong rescue"},
                        "H4": {"D9 regulatory module": "weak rescue"},
                    },
                    "falsifies": {"hypothesis": "H3", "rule": "No D9 rescue falsifies H3."},
                }
            ]
        },
    )
    design_paths = write_design_outputs(design, tmp_path / "design")
    report = generate_decision_report(
        posterior_bundle,
        json.loads((tmp_path / "design" / "design.json").read_text(encoding="utf-8")),
        evidence_contribution=pd.read_csv(adj_paths["evidence_contribution"]),
    )
    report_paths = write_report_outputs(
        report,
        tmp_path / "report",
        formats=("markdown", "html"),
        write_pdf=False,
    )

    assert "Mechanism Claim Card" in report["report_markdown"]
    assert "Why The Claim Cannot Be Higher" in report["report_markdown"]
    assert (tmp_path / "report" / "ted_mechanism_decision_report.md").exists()
    assert (tmp_path / "report" / "mechanism_claim_card.html").exists()
    assert (tmp_path / "report" / "figure_a_hypothesis_posterior_sensitivity.png").exists()
    assert (tmp_path / "report" / "figure_c_active_rescue_design_matrix.png").exists()
    assert "design_bundle" in design_paths
    assert "claim_card_markdown" in report_paths
    assert "claim_card_html" in report_paths


def test_posterior_sensitivity_outputs_stability_tables(tmp_path):
    hypotheses = load_ted_mad_yaml(GOLDEN / "example_hypotheses.yaml")
    evidence = load_ted_mad_yaml(GOLDEN / "example_evidence.yaml")

    result = adjudicate_mechanism(
        evidence,
        hypotheses,
        event="erythroid_event_001",
        strict=True,
        sensitivity=True,
        prior_grid=True,
        weight_jitter=0.2,
        n_bootstrap=20,
        random_seed=7,
    )
    paths = write_adjudication_outputs(result, tmp_path / "sensitivity")

    assert not result["posterior_sensitivity"].empty
    assert not result["posterior_interval"].empty
    assert not result["dominance_frequency"].empty
    assert "leading_stability_frequency" in result["robustness_summary"]
    assert Path(paths["posterior_sensitivity"]).exists()
    assert Path(paths["posterior_interval"]).exists()
    assert Path(paths["dominance_frequency"]).exists()


def test_leave_one_family_summary_reports_claim_and_rank_effects():
    hypotheses = load_ted_mad_yaml(GOLDEN / "example_hypotheses.yaml")
    evidence = load_ted_mad_yaml(GOLDEN / "example_evidence.yaml")

    result = adjudicate_mechanism(
        evidence,
        hypotheses,
        event="erythroid_event_001",
        strict=True,
        leave_one_family_out=True,
    )
    lofo = result["leave_one_evidence_family_out_summary"]

    assert {
        "family_removed",
        "leading_hypothesis",
        "claim_ceiling",
        "baseline_leading_rank_after_removal",
        "posterior_drop_for_baseline_leading",
    }.issubset(lofo.columns)
    assert "most_influential_evidence_family" in result["robustness_summary"]


def test_compare_naive_flags_overconfidence_when_correlated_items_repeat():
    hypotheses = {
        "hypotheses": {
            "H0": {"label": "noise", "prior": 0.5},
            "H3": {"label": "driver", "prior": 0.5},
        }
    }
    evidence = {
        "evidence": [
            {
                "evidence_id": f"E{i}",
                "evidence_family": "E1 family-level block robustness",
                "target_event": "event",
                "strength": 1.0,
                "which_hypotheses_it_supports": ["H3"],
                "which_hypotheses_it_weakens": ["H0"],
                "dependency_group": "same_source",
            }
            for i in range(4)
        ]
    }

    result = adjudicate_mechanism(evidence, hypotheses, compare_naive=True)
    comparison = result["fusion_comparison"]
    aware_h3 = comparison[
        (comparison["fusion_model"] == "dependency_aware") & (comparison["hypothesis"] == "H3")
    ].iloc[0]["posterior"]
    naive_h3 = comparison[
        (comparison["fusion_model"] == "naive") & (comparison["hypothesis"] == "H3")
    ].iloc[0]["posterior"]

    assert naive_h3 > aware_h3
    assert "Overconfidence warning" in result["overconfidence_warning_markdown"]


def test_synthetic_mechanism_benchmark_reports_recovery_and_calibration():
    results = run_synthetic_mechanism_benchmark(n_replicates=1, random_seed=11)

    cases = results["synthetic_case_results"]
    metrics = results["synthetic_metrics"].iloc[0]
    naive = results["naive_dependency_summary"]

    assert len(cases) == 6
    assert {
        "top1_hypothesis_recovery",
        "top2_hypothesis_coverage",
        "false_claim_upgrade_rate",
        "best_experiment_recovery",
        "falsifier_recovery",
        "mean_naive_overconfidence_delta",
    }.issubset(results["synthetic_metrics"].columns)
    assert metrics["top1_hypothesis_recovery"] >= 0.8
    assert metrics["false_claim_upgrade_rate"] <= 0.2
    assert (naive["naive_overconfidence_delta_mean"] > 0).any()


def test_retrospective_evidence_hiding_benchmark_reports_reveal_delta():
    table = run_retrospective_evidence_hiding_benchmark(
        GOLDEN / "example_evidence.yaml",
        GOLDEN / "example_hypotheses.yaml",
        GOLDEN / "example_experiments.yaml",
        event="erythroid_event_001",
        hide_families=["external_GATA1_KD"],
    )

    assert len(table) == 1
    row = table.iloc[0]
    assert row["hidden_family"] == "external_GATA1_KD"
    assert row["after_leading_hypothesis"] == "H3_GATA1_regulatory"
    assert row["posterior_delta_after_reveal"] > 0
    assert row["recommended_experiment_before_reveal"] == "A1_GATA1_FL_D7_rescue"


def test_negative_control_benchmark_estimates_false_promotion_rate():
    results = run_negative_control_benchmark(n_controls=3, random_seed=13)

    controls = results["negative_control_results"]
    metrics = results["negative_control_metrics"].iloc[0]

    assert len(controls) == 3
    assert 0.0 <= metrics["false_mechanism_promotion_rate"] <= 1.0
    assert controls["max_mechanism_posterior"].max() < 0.5


def _golden_provenance():
    return {
        "ted_mad_version": "0.1.0",
        "pyfgsea_version": "0.1.4",
        "git_commit": "abc1234",
        "input_sha256": {
            "evidence": "sha-e",
            "hypotheses": "sha-h",
            "experiments": "sha-x",
        },
        "run_timestamp": "2026-05-13T00:00:00+00:00",
        "random_seed": 20260513,
        "input_files": {
            "evidence": "tests/golden/example_evidence.yaml",
            "hypotheses": "tests/golden/example_hypotheses.yaml",
            "experiments": "tests/golden/example_experiments.yaml",
        },
    }


def test_strict_schema_reports_missing_required_field():
    hypotheses = load_ted_mad_yaml(GOLDEN / "example_hypotheses.yaml")
    evidence = load_ted_mad_yaml(GOLDEN / "example_evidence.yaml")
    hypothesis_ids = validate_hypotheses(hypotheses)
    bad = deepcopy(evidence)
    del bad["evidence"][0]["dependency_group"]

    with pytest.raises(TedMadValidationError, match="Missing required field: dependency_group"):
        validate_evidence(bad, hypothesis_ids)


def test_strict_schema_reports_invalid_evidence_family():
    hypotheses = load_ted_mad_yaml(GOLDEN / "example_hypotheses.yaml")
    evidence = load_ted_mad_yaml(GOLDEN / "example_evidence.yaml")
    hypothesis_ids = validate_hypotheses(hypotheses)
    bad = deepcopy(evidence)
    bad["evidence"][0]["evidence_family"] = "dynamic_precendence"

    with pytest.raises(TedMadValidationError, match="Invalid evidence_family: dynamic_precendence"):
        validate_evidence(bad, hypothesis_ids)


def test_strict_schema_reports_invalid_hypothesis_id():
    hypotheses = load_ted_mad_yaml(GOLDEN / "example_hypotheses.yaml")
    evidence = load_ted_mad_yaml(GOLDEN / "example_evidence.yaml")
    hypothesis_ids = validate_hypotheses(hypotheses)
    bad = deepcopy(evidence)
    bad["evidence"][0]["supports"] = {"H3_GATA1": 0.8}

    with pytest.raises(TedMadValidationError, match="Invalid hypothesis id: H3_GATA1"):
        validate_evidence(bad, hypothesis_ids)


def test_cli_validation_error_is_actionable(tmp_path):
    hypotheses_path = GOLDEN / "example_hypotheses.yaml"
    evidence = load_ted_mad_yaml(GOLDEN / "example_evidence.yaml")
    del evidence["evidence"][0]["dependency_group"]
    bad_path = tmp_path / "bad_evidence.yaml"
    bad_path.write_text(
        "schema_version: '0.1'\nevidence:\n"
        "  - evidence_id: bad\n"
        "    evidence_family: family_block_robustness\n"
        "    target_event: erythroid_event_001\n"
        "    direction: supports\n"
        "    effect_size: 1.0\n"
        "    standard_error: 0.5\n"
        "    p_value: 0.01\n"
        "    weight: 1.0\n"
        "    assumptions: [ok]\n"
        "    failure_modes: [bad]\n"
        "    supports: {H3_GATA1_regulatory: 0.8}\n"
        "    weakens: {H0_noise_batch: 0.5}\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        ted_mad_cli,
        ["adjudicate", str(bad_path), str(hypotheses_path), "--event", "erythroid_event_001"],
    )

    assert result.exit_code != 0
    assert "Missing required field: dependency_group" in result.output


def test_cli_update_command_writes_interpretation_outputs(tmp_path):
    adj_dir = tmp_path / "adjudication"
    design_dir = tmp_path / "design"
    update_dir = tmp_path / "update"
    runner = CliRunner()
    adjudicate = runner.invoke(
        ted_mad_cli,
        [
            "adjudicate",
            str(GOLDEN / "example_evidence.yaml"),
            str(GOLDEN / "example_hypotheses.yaml"),
            "--event",
            "erythroid_event_001",
            "--out",
            str(adj_dir),
        ],
    )
    assert adjudicate.exit_code == 0
    design = runner.invoke(
        ted_mad_cli,
        [
            "design",
            str(adj_dir / "posterior.yaml"),
            str(GOLDEN / "example_experiments.yaml"),
            "--out",
            str(design_dir),
        ],
    )
    assert design.exit_code == 0
    observed_path = tmp_path / "observed.yaml"
    observed_path.write_text(
        "experiment_id: A1_GATA1_FL_D7_rescue\n"
        "observed_results:\n"
        "  D9_regulatory_module: strong_rescue\n"
        "  D11_maturation_module: partial_rescue\n"
        "  D11_hemoglobinization: partial_rescue\n",
        encoding="utf-8",
    )
    update = runner.invoke(
        ted_mad_cli,
        [
            "update",
            str(adj_dir / "posterior.yaml"),
            str(design_dir / "design.yaml"),
            str(observed_path),
            "--out",
            str(update_dir),
        ],
    )

    assert update.exit_code == 0
    assert (update_dir / "updated_posterior.csv").exists()
    assert (update_dir / "interpretation.md").exists()


def test_cli_report_html_format_writes_claim_card_and_figures(tmp_path):
    adj_dir = tmp_path / "adjudication"
    design_dir = tmp_path / "design"
    report_dir = tmp_path / "report"
    runner = CliRunner()
    adjudicate = runner.invoke(
        ted_mad_cli,
        [
            "adjudicate",
            str(GOLDEN / "example_evidence.yaml"),
            str(GOLDEN / "example_hypotheses.yaml"),
            "--event",
            "erythroid_event_001",
            "--out",
            str(adj_dir),
            "--sensitivity",
            "--prior-grid",
            "--n-bootstrap",
            "2",
            "--compare-naive",
        ],
    )
    assert adjudicate.exit_code == 0
    design = runner.invoke(
        ted_mad_cli,
        [
            "design",
            str(adj_dir / "posterior.yaml"),
            str(GOLDEN / "example_experiments.yaml"),
            "--out",
            str(design_dir),
        ],
    )
    assert design.exit_code == 0
    report = runner.invoke(
        ted_mad_cli,
        [
            "report",
            str(adj_dir / "posterior.yaml"),
            str(design_dir / "design.yaml"),
            "--evidence-contribution",
            str(adj_dir / "evidence_contribution.csv"),
            "--format",
            "html",
            "--out",
            str(report_dir),
        ],
    )

    assert report.exit_code == 0
    assert (report_dir / "mechanism_claim_card.html").exists()
    assert (report_dir / "figure_a_hypothesis_posterior_sensitivity.png").exists()
    assert (report_dir / "figure_b_evidence_family_ablation.png").exists()
    assert (report_dir / "figure_c_active_rescue_design_matrix.png").exists()


def test_cli_benchmark_synthetic_writes_tables(tmp_path):
    outdir = tmp_path / "benchmark"
    result = CliRunner().invoke(
        ted_mad_cli,
        [
            "benchmark",
            "synthetic",
            "--n-replicates",
            "1",
            "--seed",
            "17",
            "--out",
            str(outdir),
        ],
    )

    assert result.exit_code == 0
    assert "Synthetic benchmark" in result.output
    assert (outdir / "synthetic_case_results.csv").exists()
    assert (outdir / "synthetic_metrics.csv").exists()
    assert (outdir / "naive_dependency_summary.csv").exists()


def test_experiment_schema_cross_checks_hypothesis_ids():
    hypotheses = load_ted_mad_yaml(GOLDEN / "example_hypotheses.yaml")
    experiments = load_ted_mad_yaml(GOLDEN / "example_experiments.yaml")
    hypothesis_ids = validate_hypotheses(hypotheses)
    bad = deepcopy(experiments)
    bad["experiments"][0]["distinguishes"].append("H3_GATA1")

    with pytest.raises(TedMadValidationError, match="Invalid hypothesis id: H3_GATA1"):
        validate_experiments(bad, hypothesis_ids)


def test_ted_mad_golden_posterior_claim_design_and_report():
    hypotheses = load_ted_mad_yaml(GOLDEN / "example_hypotheses.yaml")
    evidence = load_ted_mad_yaml(GOLDEN / "example_evidence.yaml")
    experiments = load_ted_mad_yaml(GOLDEN / "example_experiments.yaml")
    provenance = _golden_provenance()

    adjudication = adjudicate_mechanism(
        evidence,
        hypotheses,
        event="erythroid_event_001",
        strict=True,
        provenance=provenance,
    )
    expected_posterior = pd.read_csv(GOLDEN / "expected_posterior.csv")
    pd.testing.assert_frame_equal(
        adjudication["posterior"].reset_index(drop=True),
        expected_posterior,
        check_dtype=False,
        atol=1e-12,
        rtol=1e-9,
    )

    expected_claim = json.loads((GOLDEN / "expected_claim_ceiling.json").read_text())
    assert adjudication["claim_ceiling"] == expected_claim

    posterior_bundle = {
        "posterior": adjudication["posterior"].to_dict(orient="records"),
        "claim_ceiling": adjudication["claim_ceiling"],
        "target_events": ["erythroid_event_001"],
        "provenance": provenance,
    }
    design = design_rescue_experiments(
        posterior_bundle,
        experiments,
        strict=True,
        provenance=provenance,
    )
    expected_rank = pd.read_csv(GOLDEN / "expected_design_rank.csv")
    expected_rank = expected_rank.fillna("")
    pd.testing.assert_frame_equal(
        design["ranked_experiments"].reset_index(drop=True),
        expected_rank,
        check_dtype=False,
        atol=1e-12,
        rtol=1e-9,
    )

    design_bundle = {
        "ranked_experiments": design["ranked_experiments"].to_dict(orient="records"),
        "expected_result_patterns": design["expected_result_patterns"].to_dict(orient="records"),
        "experiment_contrast_matrix": design["experiment_contrast_matrix"].to_dict(orient="records"),
        "minimal_readout_panel": design["minimal_readout_panel"].to_dict(orient="records"),
        "falsification_rules_markdown": design["falsification_rules_markdown"],
        "provenance": provenance,
    }
    report = generate_decision_report(
        posterior_bundle,
        design_bundle,
        evidence_contribution=adjudication["evidence_contribution"],
        provenance=provenance,
    )
    expected_report = (GOLDEN / "expected_report.md").read_text(encoding="utf-8")
    assert report["report_markdown"].strip() == expected_report.strip()


def test_provenance_records_checksums_and_files():
    provenance = build_provenance(
        {
            "evidence": GOLDEN / "example_evidence.yaml",
            "hypotheses": GOLDEN / "example_hypotheses.yaml",
        },
        random_seed=20260513,
        run_timestamp="2026-05-13T00:00:00+00:00",
        git_cwd=GOLDEN,
    )

    assert provenance["ted_mad_version"] == "0.1.0"
    assert provenance["random_seed"] == 20260513
    assert set(provenance["input_sha256"]) == {"evidence", "hypotheses"}
    assert provenance["input_files"]["evidence"].endswith("example_evidence.yaml")
