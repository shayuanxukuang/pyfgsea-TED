from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PACKAGE_VERSION = "0.1.5"
EXPECTED_REQUIRES_PYTHON = ">=3.9,<3.14"


def _toml_section(text: str, section: str) -> str:
    match = re.search(
        rf"(?ms)^\[{re.escape(section)}\]\s*$\n(.*?)(?=^\[|\Z)",
        text,
    )
    assert match is not None, f"missing TOML section [{section}]"
    return match.group(1)


def _toml_string(text: str, section: str, key: str) -> str:
    body = _toml_section(text, section)
    match = re.search(
        rf'(?m)^{re.escape(key)}\s*=\s*"([^"]+)"\s*$',
        body,
    )
    assert match is not None, f"missing TOML key [{section}].{key}"
    return match.group(1)


def _python_package_version() -> str:
    source = (ROOT / "pyfgsea" / "__init__.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            continue
        assert isinstance(node.value, ast.Constant)
        assert isinstance(node.value.value, str)
        return node.value.value
    raise AssertionError("pyfgsea.__version__ assignment is missing")


def _cargo_lock_version() -> str:
    text = (ROOT / "Cargo.lock").read_text(encoding="utf-8")
    versions: list[str] = []
    for match in re.finditer(
        r"(?ms)^\[\[package\]\]\s*$\n(.*?)(?=^\[\[package\]\]|\Z)",
        text,
    ):
        body = match.group(1)
        name = re.search(r'(?m)^name\s*=\s*"([^"]+)"\s*$', body)
        if name is None or name.group(1) != "pyfgsea":
            continue
        version = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', body)
        assert version is not None, "pyfgsea Cargo.lock entry has no version"
        versions.append(version.group(1))
    assert len(versions) == 1, (
        "expected exactly one pyfgsea Cargo.lock entry, "
        f"found {len(versions)}"
    )
    return versions[0]


def test_python_and_rust_package_versions_are_synchronized() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    cargo_toml = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    actual = {
        "pyproject.toml": _toml_string(pyproject, "project", "version"),
        "pyfgsea/__init__.py": _python_package_version(),
        "Cargo.toml": _toml_string(cargo_toml, "package", "version"),
        "Cargo.lock": _cargo_lock_version(),
    }
    assert actual == {
        source: EXPECTED_PACKAGE_VERSION for source in actual
    }


def test_python_compatibility_declaration_is_bounded() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    # This is distribution metadata, not evidence that every version was tested.
    assert (
        _toml_string(pyproject, "project", "requires-python")
        == EXPECTED_REQUIRES_PYTHON
    )
