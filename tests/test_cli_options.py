"""Bad option values are usage errors before anything is mounted or read."""

from pathlib import Path
from typing import Any

from click.testing import CliRunner, Result
import pytest

from tboardfs import TensorBoardFS, cli, file_cli
from tboardfs.file_tree import SingleEventTree

from test_file_cli import FIXTURE_EVENT


def _mount(tmp_path: Path, monkeypatch: Any, *option: str) -> Result:
    (tmp_path / "runs").mkdir()
    (tmp_path / "mnt").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "FUSE", lambda *args, **kwargs: None)
    return CliRunner().invoke(cli.main, ["runs", "mnt", *option])


@pytest.mark.parametrize(
    ("option", "message"),
    [
        (["--scalar-format", "json,parquet"], "parquet"),
        (["--scalar-format", "clipboard"], "clipboard"),
        (["--step-digits", "-1"], "not in the range"),
        (["--step-digits", "21"], "not in the range"),
        (["--loglevel", "LOUD"], "LOUD"),
    ],
)
def test_mount_cli_refuses_bad_values(
    tmp_path: Path, monkeypatch: Any, option: list[str], message: str
) -> None:
    """The error names the option and the bad value."""
    result = _mount(tmp_path, monkeypatch, *option)

    assert result.exit_code == 2
    assert option[0] in result.stderr
    assert message in result.stderr


@pytest.mark.parametrize(
    "option",
    [
        ["--scalar-format", " CSV, .tsv "],
        ["--step-digits", "1"],
        ["--step-digits", "20"],
        ["--loglevel", "debug"],
    ],
)
def test_mount_cli_accepts_good_values(
    tmp_path: Path, monkeypatch: Any, option: list[str]
) -> None:
    """Supported formats, the digit range ends, and any level case pass."""
    assert _mount(tmp_path, monkeypatch, *option).exit_code == 0


@pytest.mark.parametrize(
    "args",
    [
        ["list", str(FIXTURE_EVENT), "--scalar-format", "xml"],
        ["list", str(FIXTURE_EVENT), "--step-digits", "0"],
        ["--loglevel", "LOUD", "list", str(FIXTURE_EVENT)],
    ],
)
def test_file_cli_refuses_bad_values(args: list[str]) -> None:
    """The single-file commands check the same options."""
    result = CliRunner().invoke(file_cli.main, args)

    assert result.exit_code == 2
    assert "Invalid value" in result.stderr


def test_csv_tables_are_comma_separated(tmp_path: Path) -> None:
    """csv is a supported scalar format next to the default ones."""
    tree = SingleEventTree(FIXTURE_EVENT, scalar_format="csv")

    assert tree.read_file("/scalars/loss.csv").startswith(b"epoch,step,wall_time,")


@pytest.mark.parametrize("formats", ["json,parquet", ("tsv", "html")])
def test_library_refuses_unsupported_formats(tmp_path: Path, formats: Any) -> None:
    """The library checks formats when it is built, not when a file is read."""
    with pytest.raises(ValueError, match="unsupported scalar format"):
        TensorBoardFS(tmp_path, scalar_formats=formats)
    with pytest.raises(ValueError, match="unsupported scalar format"):
        SingleEventTree(FIXTURE_EVENT, scalar_format=",".join(formats))
