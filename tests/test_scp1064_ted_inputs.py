from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "external_data"


def test_scp1064_fixed_fixtures_are_manifested():
    manifest = pd.read_csv(FIXTURES / "fixture_manifest.tsv", sep="\t")
    expected = {
        "tests/fixtures/external_data/scp1064_cell_metadata_100.tsv",
        "tests/fixtures/external_data/scp1064_protein_outcome_100.tsv",
    }
    assert expected.issubset(set(manifest["fixture_path"]))
    assert manifest["source_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert manifest["fixture_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()


def test_scp1064_event_axes_include_primary_and_negative_controls():
    axes = yaml.safe_load((ROOT / "config" / "scp1064_event_axes.yml").read_text())
    for axis in [
        "immune_evasion_antigen_presentation",
        "ifn_jak_stat_response",
        "t_cell_interaction_or_cytokine_response",
    ]:
        assert axis in axes
        assert axes[axis]["positive_markers"]
    assert "negative_controls" in axes
    for control in ["ribosome", "mitochondrial", "housekeeping"]:
        assert control in axes["negative_controls"]


def test_scp1064_protein_outcome_fixture_has_required_schema():
    outcome = pd.read_csv(FIXTURES / "scp1064_protein_outcome_100.tsv", sep="\t")
    required = {
        "cell_id",
        "guide_id",
        "target_gene",
        "protein_name",
        "protein_value",
        "protein_value_normalized",
    }
    assert required.issubset(outcome.columns)
    assert len(outcome) == 100
