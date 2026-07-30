from __future__ import annotations

import importlib.util
from pathlib import Path

from pyfgsea.ted_schema import ted_table_is_valid, validate_ted_table


ROOT = Path(__file__).resolve().parents[1]


def load_demo_module():
    path = ROOT / "scripts" / "run_ted_validation_demo.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_validation_demo_uses_current_event_support_contract() -> None:
    module = load_demo_module()
    events = module.call_events(module.make_activity())
    report = validate_ted_table(events, "event", schema_version="v2")

    assert ted_table_is_valid(report), report.to_string(index=False)
    assert "validation_provenance_code" not in events
    assert "evidence_boundary" not in events
    assert set(events["event_support_code"]) <= {"E0", "E1", "E2"}
    assert events["block_support_method"].eq("parametric_block_test").all()
    assert events["discovery_stability_status"].eq("not_evaluable").all()
