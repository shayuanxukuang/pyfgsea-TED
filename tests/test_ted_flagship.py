import numpy as np
import pandas as pd

from pyfgsea.ted_flagship import (
    entropy_balance,
    exhaustive_sign_flip_max_t,
    leave_one_donor_retention,
    peak_day,
    transient_contrasts,
)


def test_transient_contrasts_match_frozen_weights():
    table = pd.DataFrame(
        {
            "donor_id": np.repeat(["D1", "D2"], 4),
            "day": [0, 2, 10, 28] * 2,
            "score": [0.0, 2.0, 1.0, 1.0, 1.0, 3.0, 2.0, 2.0],
        }
    )
    result = transient_contrasts(table)
    assert result.loc["D1", "activation"] == 2.0
    assert result.loc["D1", "recovery"] == 1.0
    assert result.loc["D1", "transient"] == 1.5
    assert result.loc["D2", "transient"] == 1.5


def test_exact_sign_flip_and_max_t_use_all_configurations():
    effects = pd.DataFrame(
        {
            "primary": [1.0] * 6,
            "null_like": [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        },
        index=[f"D{i}" for i in range(6)],
    )
    result = exhaustive_sign_flip_max_t(effects).set_index("pathway")
    assert result.loc["primary", "n_sign_configurations"] == 64
    assert result.loc["primary", "exact_raw_p"] == 1 / 64
    assert result.loc["primary", "exact_maxT_p"] <= 0.10
    assert result.loc["null_like", "exact_raw_p"] >= 0.50


def test_leave_one_donor_recomputes_the_family_test():
    donors = [f"D{i}" for i in range(6)]
    transient = pd.DataFrame({"primary": [1.0] * 6}, index=donors)
    activation = pd.DataFrame({"primary": [1.0] * 6}, index=donors)
    recovery = pd.DataFrame({"primary": [1.0] * 6}, index=donors)
    result = leave_one_donor_retention(transient, activation, recovery)
    assert len(result) == 6
    assert result["selection_retained"].all()
    assert result["retention_fraction"].eq(1.0).all()
    assert result["exact_maxT_p"].eq(1 / 32).all()


def test_entropy_balance_matches_target_and_reports_weight_diagnostics():
    covariates = pd.DataFrame(
        {"x": [-2.0, -1.0, 0.0, 1.0, 2.0], "nuisance": [0.0, 1.0, 0.0, 1.0, 0.0]},
        index=[f"c{i}" for i in range(5)],
    )
    target = pd.Series({"x": 0.5, "nuisance": 0.4})
    weights, diagnostics = entropy_balance(covariates, target)
    observed = weights.to_numpy() @ covariates.to_numpy()
    assert np.allclose(observed, target.to_numpy(), atol=1e-6)
    assert np.isclose(weights.sum(), 1.0)
    assert diagnostics.converged
    assert diagnostics.max_abs_smd < 1e-6
    assert diagnostics.effective_sample_size > 1


def test_replication_contrast_supports_separately_frozen_booster_grid():
    table = pd.DataFrame(
        {
            "donor_id": ["a"] * 4 + ["b"] * 4,
            "day": [21, 22, 28, 42] * 2,
            "score": [0.0, 2.0, 0.5, 0.0, 1.0, 3.0, 1.0, 1.0],
        }
    )
    weights = {
        "transient_weights": {21: -0.5, 22: 1.0, 28: -0.25, 42: -0.25},
        "activation_weights": {21: -1.0, 22: 1.0, 28: 0.0, 42: 0.0},
        "recovery_weights": {21: 0.0, 22: 1.0, 28: -0.5, 42: -0.5},
    }
    result = transient_contrasts(table, **weights)
    assert result.loc["a", "activation"] == 2.0
    assert result.loc["a", "recovery"] == 1.75
    assert result.loc["a", "transient"] == 1.875
    wide = table.pivot(index="donor_id", columns="day", values="score")
    assert peak_day(wide, (21, 22, 28, 42)) == 22
