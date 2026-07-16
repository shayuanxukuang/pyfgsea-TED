"""Release-suite grouping with mutually exclusive execution classes."""

from __future__ import annotations

from pathlib import Path

import pytest


EXTERNAL_DATA_FILES = {
    "test_scp1064_cell_alignment.py",
    "test_scp1064_claim_boundary.py",
    "test_scp1064_event_outcome_alignment.py",
    "test_scp1064_file_qc.py",
    "test_scp1064_ted_inputs.py",
    "test_ted_claim_boundary.py",
    "test_ted_dataset_registry.py",
    "test_ted_known_source_validation.py",
    "test_ted_preprocessing_outputs.py",
}

SLOW_FILES = {
    "test_performance_benchmark.py",
}

INTEGRATION_FILES = {
    "test_calibration.py",
    "test_reliability_benchmarks.py",
    "test_result_object.py",
    "test_ted_developmental.py",
    "test_ted_mad.py",
    "test_ted_perturbation.py",
    "test_ted_reliability.py",
    "test_trajectory_features.py",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Assign every release test to one and only one CI execution class."""

    for item in items:
        name = Path(str(item.fspath)).name
        if name in EXTERNAL_DATA_FILES:
            item.add_marker(pytest.mark.external_data)
        elif name in SLOW_FILES:
            item.add_marker(pytest.mark.slow)
        elif name in INTEGRATION_FILES:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)
