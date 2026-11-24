"""tboardfs-file reads a run or log directory as the mount shows it."""

from pathlib import Path

from click.testing import CliRunner
from test_file_cli import FIXTURE_EVENT
from test_incremental_parse import _files
from test_open_handles import _append

from tboardfs import TensorBoardFS
from tboardfs.file_cli import main

EVENTS = "events.out.tfevents.host"


def _logdir(tmp_path: Path) -> Path:
    for name in ("a", "b"):
        (tmp_path / "logs" / name).mkdir(parents=True)
        _append(tmp_path / "logs" / name / EVENTS, 0, 1)
    (tmp_path / "logs" / "a" / "projector_config.pbtxt").write_text("config")
    return tmp_path / "logs"


def test_list_of_a_log_directory_matches_the_mount(tmp_path: Path) -> None:
    """Each run is a directory, with its sidecars, as in filesystem mode."""
    logs = _logdir(tmp_path)

    result = CliRunner().invoke(main, ["list", str(logs)])

    assert result.exit_code == 0
    mounted = _files(TensorBoardFS(logs))
    assert result.output.splitlines() == sorted(mounted)
    assert "/a/projector/projector_config.pbtxt" in result.output.splitlines()


def test_get_from_a_run_directory_has_tabs_at_the_root(tmp_path: Path) -> None:
    """A run directory is one run, so its tabs are at the root."""
    run = _logdir(tmp_path) / "b"

    result = CliRunner().invoke(
        main, ["get", str(run), "/scalars/loss.json", "-o", "-"]
    )

    assert result.exit_code == 0
    assert result.stdout_bytes == TensorBoardFS(run).read(
        "/scalars/loss.json", 1 << 20, 0
    )


def test_copy_all_of_a_log_directory_copies_every_run(tmp_path: Path) -> None:
    """copy-all writes the files of all runs, with the mount's bytes."""
    logs = _logdir(tmp_path)
    fs = TensorBoardFS(logs, raw=True)

    result = CliRunner().invoke(
        main, ["copy-all", str(logs), str(tmp_path / "out"), "--view", "both"]
    )

    assert result.exit_code == 0
    for path in _files(fs):
        assert (tmp_path / "out" / path[1:]).read_bytes() == fs.read(path, 1 << 20, 0)


def test_directory_listing_uses_the_format_options(tmp_path: Path) -> None:
    """--step-digits and --scalar-format apply to a directory too."""
    logs = _logdir(tmp_path)
    (logs / "c").mkdir()
    (logs / "c" / FIXTURE_EVENT.name).write_bytes(FIXTURE_EVENT.read_bytes())
    options = ["--step-digits", "3", "--scalar-format", "csv"]

    result = CliRunner().invoke(main, ["list", str(logs), *options])

    expected = TensorBoardFS(logs, step_digits=3, scalar_formats="csv")
    assert result.exit_code == 0
    assert result.output.splitlines() == sorted(_files(expected))
    assert "/a/scalars/loss.csv" in result.output.splitlines()
    assert "/c/tensors/activations/001.npy" in result.output.splitlines()


def test_run_named_raw_does_not_change_view(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    run = logs / "raw"
    run.mkdir(parents=True)
    _append(run / EVENTS, 0, 1)
    runner = CliRunner()
    typed = runner.invoke(main, ["list", str(logs)])
    raw = runner.invoke(main, ["list", str(logs), "--view", "raw"])
    both = runner.invoke(main, ["list", str(logs), "--view", "both"])
    assert typed.exit_code == raw.exit_code == both.exit_code == 0
    typed_paths = set(typed.stdout.splitlines())
    raw_paths = set(raw.stdout.splitlines())
    assert typed_paths and raw_paths and typed_paths.isdisjoint(raw_paths)
    assert set(both.stdout.splitlines()) == typed_paths | raw_paths
    assert any(p.endswith("/scalars/loss.json") for p in typed_paths)
    prefix = next(p for p in typed_paths if p.endswith("/scalars/loss.json")).rsplit(
        "/scalars/", 1
    )[0]
    assert runner.invoke(main, ["list", str(logs), prefix]).stdout.splitlines() == (
        typed.stdout.splitlines()
    )
