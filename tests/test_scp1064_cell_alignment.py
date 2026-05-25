from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCP = ROOT / "data" / "processed" / "ted_known_source" / "SCP1064"


def _qc_value(qc: pd.DataFrame, metric: str) -> str:
    return str(qc.loc[qc["metric"].eq(metric), "value"].iloc[0])


def test_scp1064_alignment_has_nonzero_shared_cells():
    qc = pd.read_csv(SCP / "raw_index" / "scp1064_cell_alignment_qc.tsv", sep="\t")
    assert int(_qc_value(qc, "rna_metadata_guide_intersection")) > 0
    assert int(_qc_value(qc, "rna_protein_guide_intersection")) > 0
    assert _qc_value(qc, "cell_id_alignment_mode") == "explicit_CELL_id"
    assert _qc_value(qc, "cell_level_alignment_allowed").lower() == "true"
    assert int(_qc_value(qc, "duplicate_cell_ids")) == 0


def test_scp1064_alignment_table_has_unique_cells_and_guides():
    alignment = pd.read_csv(SCP / "metadata" / "cell_alignment_table.tsv", sep="\t")
    assert alignment["cell_id"].is_unique
    ted_cells = alignment[alignment["include_for_ted_rna"].astype(str).str.lower().eq("true")]
    outcome_cells = alignment[alignment["include_for_outcome_alignment"].astype(str).str.lower().eq("true")]
    assert not ted_cells.empty
    assert not outcome_cells.empty
    assert ted_cells["guide_id"].notna().all()
    assert outcome_cells["guide_id"].notna().all()


def test_scp1064_primary_perturbations_have_rna_and_protein_cells():
    perturbations = pd.read_csv(SCP / "metadata" / "perturbation_metadata.tsv", sep="\t")
    primary = perturbations[perturbations["include_in_primary_analysis"].astype(str).str.lower().eq("true")]
    assert not primary.empty
    assert primary["n_cells_with_rna"].astype(int).gt(0).all()
    assert primary["n_cells_with_protein"].astype(int).gt(0).all()
