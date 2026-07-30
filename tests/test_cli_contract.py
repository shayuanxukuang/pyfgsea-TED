from __future__ import annotations

from click.testing import CliRunner

from pyfgsea import __version__
from pyfgsea.cli.main import cli
from pyfgsea.ted_mad.cli import cli as ted_mad_cli


def test_ted_run_help_matches_the_trajectory_gsea_interface():
    result = CliRunner().invoke(cli, ["run", "--help"])
    assert result.exit_code == 0, result.output
    assert "--h5ad" in result.output
    assert "--gmt" in result.output
    assert "--out" in result.output
    for unsupported_option in (
        "--activity",
        "--metadata",
        "--gene-sets",
        "--design",
        "--negative-controls",
    ):
        assert unsupported_option not in result.output


def test_ted_exposes_validate_nested_ted_mad_and_version():
    runner = CliRunner()
    help_result = runner.invoke(cli, ["--help"])
    assert help_result.exit_code == 0, help_result.output
    assert "validate" in help_result.output
    assert "ted-mad" in help_result.output

    version_result = runner.invoke(cli, ["--version"])
    assert version_result.exit_code == 0, version_result.output
    assert version_result.output.strip() == f"ted, version {__version__}"

    mad_help = runner.invoke(ted_mad_cli, ["--help"])
    assert mad_help.exit_code == 0, mad_help.output
    assert "adjudicate" in mad_help.output
