from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "config" / "ted_upgrade_benchmark_registry.tsv"


def test_registry_has_roles_access_and_claims():
    registry = pd.read_csv(REGISTRY, sep="\t")
    required = [
        "dataset",
        "analysis_role",
        "access_status",
        "expected_event",
        "claim_if_pass",
        "claim_if_fail",
        "claim_boundary",
    ]
    assert set(required).issubset(registry.columns)
    for column in required:
        assert registry[column].astype(str).str.len().gt(0).all(), column
        assert not registry[column].astype(str).str.lower().eq("unknown").any(), column


def test_main_text_datasets_are_not_failed_access():
    registry = pd.read_csv(REGISTRY, sep="\t")
    main = registry[registry["main_text_candidate"].astype(str).str.lower().eq("true")]
    assert not main.empty
    assert not main["access_status"].astype(str).str.lower().str.contains("failed").any()
    assert "SCP1064" not in set(main["dataset"])


def test_registry_freezes_gata1_as_cross_dataset_support_only():
    registry = pd.read_csv(REGISTRY, sep="\t")
    gata1 = registry[registry["analysis_role"].eq("gata1_cross_dataset_support")]
    assert not gata1.empty
    assert gata1["claim_boundary"].astype(str).str.contains("not_.*rescue|not_T21_or_rescue", regex=True).all()
