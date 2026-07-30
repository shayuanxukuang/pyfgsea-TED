from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_flagship_config_marks_legacy_success_criterion_as_unobserved() -> None:
    config = yaml.safe_load((ROOT / "config" / "ted_bnt162b2_flagship_v1.yaml").read_text(encoding="utf-8"))
    assert config["protocol"]["prospective_preregistration_claimed"] is False
    assert config["protocol"]["known_direction_from_source_publication"] is True
    assert config["claim_boundary"]["pass_descriptor"] == "E2-V1"
    assert (
        config["claim_boundary"]["pass_descriptor_role"]
        == "legacy_pre_result_success_criterion_not_observed_result"
    )
    assert "observed result did not meet" in config["claim_boundary"]["allowed_claim"]
    forbidden = set(config["claim_boundary"]["forbidden_claims"])
    assert "prospective external validation" in forbidden
    assert "newly discovered vaccine mechanism" in forbidden
    assert "matched rescue" in forbidden


def test_freeze_validator_rejects_prospective_claim() -> None:
    module = load_script("freeze_bnt162b2_flagship_protocol.py")
    config = yaml.safe_load((ROOT / "config" / "ted_bnt162b2_flagship_v1.yaml").read_text(encoding="utf-8"))
    module.validate_config(config)
    config["protocol"]["prospective_preregistration_claimed"] = True
    try:
        module.validate_config(config)
    except ValueError as error:
        assert "must not claim prospective" in str(error)
    else:
        raise AssertionError("Prospective claim should fail validation")
