from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_claim_boundary_rules_require_same_system_matched_rescue_for_level4():
    rules = yaml.safe_load((ROOT / "config" / "ted_claim_boundary_rules.yml").read_text())
    level4 = rules["level4_causal_rescue"]["requires"]
    assert level4["matched_rescue_design"] is True
    assert level4["same_system"] is True
    assert level4["event_family_restoration"] is True
    assert level4["phenotype_restoration"] is True


def test_gata1_cross_dataset_support_does_not_promote_to_level4():
    decision = pd.read_csv(
        ROOT / "results" / "gata1_cross_dataset_support" / "tables" / "gata1_claim_boundary_decision.tsv",
        sep="\t",
    ).iloc[0]
    assert str(decision["cross_dataset_supported_event"]).lower() == "true"
    assert str(decision["matched_rescue_design"]).lower() == "false"
    assert str(decision["same_system"]).lower() == "false"
    assert str(decision["level4_causal_rescue"]).lower() == "false"
    assert "Level_3.5" in decision["final_claim"]


def test_outcome_and_reversal_claims_do_not_imply_causal_rescue():
    claims = pd.read_csv(
        ROOT / "results" / "ted_known_source_validation" / "tables" / "ted_dataset_level_claim_boundary.tsv",
        sep="\t",
    )
    promoted = claims[claims["claim_boundary"].isin(["outcome_supported_event", "reversal_supported_event"])]
    assert not promoted.empty
    assert promoted["level4_causal_rescue"].astype(str).str.lower().eq("false").all()
