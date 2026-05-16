import numpy as np
import pandas as pd

import pyfgsea


def test_design_detector_flags_mixed_sample_condition():
    import anndata as ad

    adata = ad.AnnData(np.ones((4, 3)))
    adata.obs["condition"] = ["control", "case", "control", "case"]
    adata.obs["donor"] = ["d1", "d1", "d2", "d2"]

    design = pyfgsea.detect_experimental_design(
        adata,
        condition_key="condition",
        replicate_key="donor",
    )

    row = design.iloc[0]
    assert row["design"] == "mixed_sample_condition"
    assert row["replicate_inference"] == "not_supported"
    assert row["recommended_mode"] == "descriptive_or_within_sample_sensitivity"
    assert row["mixed_samples"] == "d1,d2"


def test_design_detector_supports_between_sample_replicates():
    import anndata as ad

    adata = ad.AnnData(np.ones((6, 3)))
    adata.obs["condition"] = ["control", "control", "control", "case", "case", "case"]
    adata.obs["donor"] = ["c1", "c2", "c3", "k1", "k2", "k3"]

    design = pyfgsea.detect_experimental_design(
        adata,
        condition_key="condition",
        replicate_key="donor",
        min_replicates_per_condition=3,
    )

    row = design.iloc[0]
    assert row["design"] == "between_sample_design"
    assert row["replicate_inference"] == "supported"
    assert row["recommended_mode"] == "replicate_aware"
    assert row["min_replicates_per_condition"] == 3


def test_technical_confound_diagnostics_from_window_metrics():
    result = pd.DataFrame(
        {
            "Pathway": ["A", "A", "A", "B", "B", "B"],
            "window_id": [0, 1, 2, 0, 1, 2],
            "NES": [3.0, 1.0, 0.5, 0.1, 0.2, 0.1],
            "padj": [0.01, 0.2, 0.8, 0.9, 0.8, 0.7],
            "mean_cell_detection_rate": [0.1, 0.5, 0.9, 0.1, 0.5, 0.9],
        }
    )

    diag = pyfgsea.technical_confound_diagnostics(result)

    assert set(diag["Pathway"]) == {"A", "B"}
    assert "technical_confound_score" in diag.attrs["summary"]
    a = diag[diag["Pathway"] == "A"].iloc[0]
    assert a["n_significant_windows"] == 1
    assert a["n_low_detection_significant_windows"] == 1
    assert a["low_detection_sig_fraction"] == 1.0
