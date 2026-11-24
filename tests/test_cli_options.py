"""Bad option values are usage errors before anything is mounted or read."""

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner, Result
from test_file_cli import FIXTURE_EVENT

from tboardfs import TensorBoardFS, cli, file_cli
from tboardfs.file_tree import SingleEventTree


def _mount(tmp_path: Path, monkeypatch: Any, *option: str) -> Result:
    (tmp_path / "runs").mkdir()
    (tmp_path / "mnt").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "mount", lambda *args, **kwargs: None)
    return CliRunner().invoke(cli.main, ["runs", "mnt", *option])


@pytest.mark.parametrize(
    ("option", "message"),
    [
        (["--scalar-format", "json,parquet"], "parquet"),
        (["--scalar-format", "clipboard"], "clipboard"),
        (["--histogram-format", "json,parquet"], "unsupported histogram format"),
        (["--step-digits", "-1"], "not in the range"),
        (["--step-digits", "21"], "not in the range"),
        (["--loglevel", "LOUD"], "LOUD"),
        (["--refresh-age-seconds", "-1"], "not in the range"),
        (["--refresh-age-seconds", "nan"], "not a number"),
        (["--source-check-seconds", "-0.5"], "not in the range"),
        (["--source-check-seconds", "-inf"], "not in the range"),
        (["--source-check-seconds", "nan"], "not a number"),
        (["--source-check-seconds", "inf"], "not finite"),
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
        ["--refresh-age-seconds", "0"],
        ["--refresh-age-seconds", "inf"],
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
        ["list", str(FIXTURE_EVENT), "--histogram-format", "xml"],
        ["list", str(FIXTURE_EVENT), "--step-digits", "0"],
        ["--loglevel", "LOUD", "list", str(FIXTURE_EVENT)],
    ],
)
def test_file_cli_refuses_bad_values(args: list[str]) -> None:
    """The single-file commands check the same options."""
    result = CliRunner().invoke(file_cli.main, args)

    assert result.exit_code == 2
    assert "Invalid value" in result.stderr


@pytest.mark.parametrize("directory", [False, True], ids=["file", "directory"])
def test_export_formats(tmp_path: Path, directory: bool) -> None:
    source = tmp_path if directory else FIXTURE_EVENT
    if directory:
        (source / FIXTURE_EVENT.name).write_bytes(FIXTURE_EVENT.read_bytes())
    runner = CliRunner()
    args = ["list", str(source)]
    default = runner.invoke(file_cli.main, args)
    assert default.exit_code == 0, default.stderr
    selected = runner.invoke(
        file_cli.main,
        [
            *args,
            "--scalar-format",
            "csv,npz",
            "--histogram-format",
            "csv",
            "--pr-curve-npy",
        ],
    )
    assert selected.exit_code == 0, selected.stderr
    for result, expected in [
        (default, [("json", "tsv"), ("json", "npz", "tsv"), ("json", "tsv")]),
        (selected, [("csv", "npz"), ("csv",), ("json", "npy", "tsv")]),
    ]:
        files = result.stdout.splitlines()
        stems = (
            ("/scalars/loss", expected[0]),
            ("/histograms/weights/000001", expected[1]),
            ("/distributions/weights/000001", expected[1]),
            ("/pr_curves/quality/pr/000001", expected[2]),
        )
        for stem, suffixes in stems:
            assert sorted(path for path in files if path.startswith(f"{stem}.")) == [
                f"{stem}.{suffix}" for suffix in suffixes
            ]


def test_csv_tables_are_comma_separated(tmp_path: Path) -> None:
    """csv is a supported scalar format next to the default ones."""
    tree = SingleEventTree(FIXTURE_EVENT, scalar_format="csv")

    assert tree.read_file("/scalars/loss.csv").startswith(b"epoch,step,wall_time,")


@pytest.mark.parametrize("tab", ["histograms", "distributions"])
def test_csv_applies_to_histogram_tables(tab: str) -> None:
    """Each step table has the csv format too, with the columns of tsv."""
    csv = SingleEventTree(FIXTURE_EVENT, histogram_format="csv")
    tsv = SingleEventTree(FIXTURE_EVENT, histogram_format="tsv")

    text = csv.read_file(f"/{tab}/weights/000001.csv")
    assert text == tsv.read_file(f"/{tab}/weights/000001.tsv").replace(b"\t", b",")


@pytest.mark.parametrize("formats", ["json,parquet", ("tsv", "html")])
def test_library_refuses_unsupported_formats(tmp_path: Path, formats: Any) -> None:
    """The library checks formats when it is built, not when a file is read."""
    with pytest.raises(ValueError, match="unsupported scalar format"):
        TensorBoardFS(tmp_path, scalar_formats=formats)
    with pytest.raises(ValueError, match="unsupported scalar format"):
        SingleEventTree(FIXTURE_EVENT, scalar_format=",".join(formats))


@pytest.mark.parametrize("seconds", [-1.0, float("nan"), float("inf")])
def test_library_refuses_unusable_source_intervals(
    tmp_path: Path, seconds: float
) -> None:
    with pytest.raises(ValueError, match="source_check_seconds"):
        TensorBoardFS(tmp_path, source_check_seconds=seconds)
