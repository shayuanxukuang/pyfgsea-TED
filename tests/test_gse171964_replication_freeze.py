from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "ted_gse171964_replication_v1.yaml"
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "gse171964"


def load_script():
    path = ROOT / "scripts" / "freeze_gse171964_replication_protocol.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_replication_contract_splits_event_and_protein_status() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["event_replication"]["initial_status"] == "pending"
    assert config["protein_outcome_replication"]["initial_status"] == "not_tested"
    assert config["protein_outcome_replication"]["panel_audit_result"]["CD64_ADT_present"] is False
    assert config["protein_outcome_replication"]["panel_audit_result"]["CD169_ADT_present"] is False
    assert config["claim_boundary"]["protein_and_event_pass_text_allowed"] is False


def test_replication_contract_freezes_corrected_v2_booster_design() -> None:
    module = load_script()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    module.validate_config(config)
    assert config["design"]["public_sample_sheet_mapping"] == {
        "baseline": 21,
        "early_post_dose": 22,
        "recovery": 28,
        "late_phase": 42,
    }
    assert config["event_replication"]["evaluable_donor_direction_fraction_min"] == 0.80


def test_validator_rejects_posthoc_protein_substitution_or_time_change() -> None:
    module = load_script()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    bad_protein = deepcopy(config)
    bad_protein["protein_outcome_replication"]["initial_status"] = "pending"
    try:
        module.validate_config(bad_protein)
    except ValueError as error:
        assert "not_tested" in str(error)
    else:
        raise AssertionError("An incompatible ADT panel must remain not_tested")

    bad_time = deepcopy(config)
    bad_time["design"]["public_sample_sheet_mapping"]["early_post_dose"] = 23
    try:
        module.validate_config(bad_time)
    except ValueError as error:
        assert "time mapping" in str(error)
    else:
        raise AssertionError("Post-hoc time remapping must fail validation")


def test_corrected_public_sample_sheet_covers_frozen_booster_episode() -> None:
    pheno = pd.read_csv(
        FIXTURE_ROOT / "sample_sheet_contract.tsv",
        sep="\t",
        dtype={"pt_id": str},
    )
    frozen_days = {21, 22, 28, 42}
    coverage = pheno[pheno["day"].isin(frozen_days)].groupby("pt_id")["day"].nunique()
    assert len(coverage) == 6
    assert coverage.eq(4).all()


def test_corrected_v2_panel_has_rna_but_not_cd64_cd169_adt() -> None:
    module_path = ROOT / "scripts" / "run_gse171964_replication.py"
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    features = module.read_r_quoted_vector(
        FIXTURE_ROOT / "feature_panel_contract.tsv"
    )
    assert "FCGR1A" in features
    assert "SIGLEC1" in features
    assert "CD64_ADT" not in features
    assert "CD169_ADT" not in features
