from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCP = ROOT / "data" / "processed" / "ted_known_source" / "SCP1064"


def test_scp1064_required_files_pass_qc():
    qc = pd.read_csv(SCP / "provenance" / "scp1064_file_qc.tsv", sep="\t")
    required = {
        "all_sgRNA_assignments.txt",
        "RNA_metadata.csv",
        "frangieh_2021_rna.h5ad",
        "Protein_expression.csv.gz",
        "raw_CITE_expression.csv.gz",
        "frangieh_2021_protein.h5ad",
    }
    present = qc[qc["file_name"].isin(required)]
    assert set(present["file_name"]) == required
    assert present["exists"].astype(str).str.lower().eq("true").all()
    assert present["readable"].astype(str).str.lower().eq("true").all()
    assert present["qc_status"].eq("pass").all()


def test_scp1064_inventory_records_sha_and_provenance():
    inventory = pd.read_csv(SCP / "raw_index" / "scp1064_file_inventory.tsv", sep="\t")
    protein = inventory[inventory["file_name"].eq("Protein_expression.csv.gz")].iloc[0]
    assert protein["sha256"]
    assert str(protein["is_original_scp_file"]).lower() == "true"
    assert str(protein["readable"]).lower() == "true"

    provenance = pd.read_csv(SCP / "provenance" / "scp1064_download_provenance.tsv", sep="\t")
    protein_prov = provenance[provenance["file_name"].eq("Protein_expression.csv.gz")].iloc[0]
    assert protein_prov["access_method"] == "authenticated SCP download"
    assert str(protein_prov["auth_config_retained"]).lower() == "false"


def test_scp1064_h5ad_structure_contains_required_fields():
    structure = pd.read_csv(SCP / "raw_index" / "scp1064_h5ad_structure.tsv", sep="\t")
    rna = structure[structure["file_name"].eq("frangieh_2021_rna.h5ad")].iloc[0]
    protein = structure[structure["file_name"].eq("frangieh_2021_protein.h5ad")].iloc[0]
    assert int(rna["n_obs"]) > 0
    assert int(protein["n_obs"]) > 0
    assert str(rna["has_X"]).lower() == "true"
    for field in ["guide_id", "celltype", "MOI", "UMI_count"]:
        assert field in rna["obs_columns"].split(";")
