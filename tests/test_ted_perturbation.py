import numpy as np
import pandas as pd

import pyfgsea


def _synthetic_perturbation_input(seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    scores = []
    conditions = [
        ("Euploid_wtGATA1", 0, 0),
        ("Euploid_GATA1s", 0, 1),
        ("T21_wtGATA1", 1, 0),
        ("T21_GATA1s", 1, 1),
    ]
    for day in ("D7", "D9", "D11"):
        for bin_id in range(4):
            for state in ("erythroid", "cycling"):
                for condition, t21, gata1s in conditions:
                    for replicate in range(3):
                        pseudotime = bin_id / 3
                        erythroid_base = 1.2 + 0.4 * pseudotime + (state == "erythroid") * 0.5
                        proliferation_base = 0.8 + (state == "cycling") * 0.5 - 0.1 * pseudotime
                        erythroid_effect = -1.0 * gata1s - 0.25 * t21 * gata1s
                        proliferation_effect = -0.18 * gata1s - 0.04 * t21 * gata1s
                        rows.append(
                            {
                                "trajectory": "erythroid",
                                "day": day,
                                "pseudotime_bin": f"bin{bin_id}",
                                "state": state,
                                "condition": condition,
                                "T21": t21,
                                "GATA1s": gata1s,
                                "replicate": replicate,
                            }
                        )
                        scores.append(
                            {
                                "ERYTHROID_MATURATION": erythroid_base
                                + erythroid_effect
                                + rng.normal(0, 0.03),
                                "HEME_GLOBIN": erythroid_base
                                + erythroid_effect
                                + rng.normal(0, 0.03),
                                "RIBOSOME_TRANSLATION": proliferation_base
                                + proliferation_effect
                                + rng.normal(0, 0.03),
                                "MYC_E2F": proliferation_base
                                + proliferation_effect
                                + rng.normal(0, 0.03),
                            }
                        )
    metadata = pd.DataFrame(rows)
    score_matrix = pd.DataFrame(scores, index=metadata.index)
    return score_matrix, metadata


def test_run_ted_perturbation_returns_productized_result_object():
    score_matrix, metadata = _synthetic_perturbation_input()
    result = pyfgsea.run_ted_perturbation(
        score_matrix,
        metadata,
        trajectory_col="trajectory",
        condition_col="condition",
        factor_columns=("T21", "GATA1s"),
        strata_cols=("day", "pseudotime_bin", "state"),
        pathway_families={
            "ERYTHROID_EVENT_LOSS_FAMILY": [
                "ERYTHROID_MATURATION",
                "HEME_GLOBIN",
            ],
            "PROLIFERATION_TRANSLATION_FAMILY": [
                "RIBOSOME_TRANSLATION",
                "MYC_E2F",
            ],
        },
        contrasts={
            "T21_GATA1s_vs_T21_wtGATA1": (
                "T21_GATA1s",
                "T21_wtGATA1",
            )
        },
        proliferation_family_id="PROLIFERATION_TRANSLATION_FAMILY",
        n_perm=99,
        random_state=7,
    )

    assert isinstance(result, pyfgsea.PerturbationEventResult)
    assert {
        "event_table",
        "family_table",
        "factorial_effect_table",
        "block_null_table",
        "driver_table",
        "external_support_table",
        "claim_ceiling_table",
    } == set(result.to_tables())
    assert result.metadata["family_level_primary"] is True
    assert result.evidence_layers["block_permutation_q"] == "robust perturbation discovery"

    family_effects = result.factorial_effect_table
    gata1s = family_effects[
        (family_effects["family_id"] == "ERYTHROID_EVENT_LOSS_FAMILY")
        & (family_effects["effect_type"] == "GATA1s")
    ].iloc[0]
    interaction = family_effects[
        (family_effects["family_id"] == "ERYTHROID_EVENT_LOSS_FAMILY")
        & (family_effects["effect_type"] == "interaction")
    ].iloc[0]
    assert gata1s["beta"] < 0
    assert abs(gata1s["beta"]) > abs(interaction["beta"])

    block_row = result.block_null_table[
        (result.block_null_table["family_id"] == "ERYTHROID_EVENT_LOSS_FAMILY")
        & (result.block_null_table["contrast"] == "T21_GATA1s_vs_T21_wtGATA1")
    ].iloc[0]
    assert block_row["observed_family_delta_auc"] < 0
    assert block_row["block_perm_q"] <= 0.05
    assert block_row["direction_stability"] == 1.0

    specificity = result.family_table[
        (result.family_table["family_id"] == "ERYTHROID_EVENT_LOSS_FAMILY")
        & (result.family_table["contrast_or_effect"] == "T21_GATA1s_vs_T21_wtGATA1")
    ].iloc[0]
    assert specificity["specificity_pass"] is True
    assert specificity["specificity_classification"] in {
        "erythroid_specific",
        "partly_proliferation_associated",
    }

