from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "external_data"


def _fixture(name: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURES / name, sep="\t")


def test_gse153056_fixed_fixture_and_large_output_manifest_are_complete():
    meta = _fixture("gse153056_cell_metadata_100.tsv")
    assert {"cell_id", "condition", "perturbed_gene", "guide_id"}.issubset(meta.columns)
    assert meta["condition"].notna().all()
    qc = _fixture("gse153056_qc_summary.tsv").set_index("metric")["value"]
    assert qc["processing_status"] == "pass"
    outputs = set(_fixture("gse153056_processing_manifest.tsv")["output"])
    assert {"expression_matrix.tsv.gz", "protein_matrix.tsv.gz", "guide_matrix.tsv.gz"}.issubset(outputs)


def test_gse93735_fixed_fixture_is_complete():
    meta = _fixture("gse93735_sample_metadata.tsv")
    assert {"sample", "condition", "timepoint", "intervention"}.issubset(meta.columns)
    assert {"Control", "LPS", "Dex_LPS"}.issubset(set(meta["condition"]))
    qc = _fixture("gse93735_qc_summary.tsv").set_index("metric")["value"]
    assert qc["processing_status"] == "pass"


def test_gse90546_records_raw_required_boundary():
    qc = _fixture("gse90546_qc_summary.tsv").set_index("metric")["value"]
    assert qc["processing_status"] == "metadata_only_raw_required_not_present"
