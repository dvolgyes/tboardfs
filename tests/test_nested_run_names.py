"""A run folder with the name of an entry of the run above it gets its own name."""

from collections.abc import Iterator
import json
from pathlib import Path

from loguru import logger
import pytest

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2

from test_incremental_parse import _frame


def _write_run(run_dir: Path, step: int, **values: float) -> None:
    event = compat_pb2.Event(wall_time=float(step), step=step)
    for tag, value in values.items():
        event.summary.value.add(tag=tag, simple_value=value)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "events.out.tfevents.host").open("ab") as handle:
        handle.write(_frame(event))


def _values(fs: TensorBoardFS, path: str) -> list[float]:
    return [row["value"] for row in json.loads(fs.read(path, 1 << 20, 0))]


@pytest.fixture
def warnings() -> Iterator[list[str]]:
    """Return the warning messages that tboardfs logs during the test."""
    messages: list[str] = []
    sink = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(sink)


def test_a_run_folder_named_like_a_tab_does_not_mix_with_it(
    tmp_path: Path, warnings: list[str]
) -> None:
    """The tab keeps its files; the nested run is under scalars~2."""
    _write_run(tmp_path / "exp", 1, value=1.0)
    _write_run(tmp_path / "exp" / "scalars", 1, value=2.0)
    fs = TensorBoardFS(tmp_path)

    listing = fs.readdir("/exp/scalars", None)
    assert "value.json" in listing
    assert "images" not in listing
    assert _values(fs, "/exp/scalars/value.json") == [1.0]
    assert _values(fs, "/exp/scalars~2/scalars/value.json") == [2.0]
    assert len(warnings) == 1
    assert "'exp/scalars'" in warnings[0]
    assert "'exp/scalars~2'" in warnings[0]


def test_a_run_folder_below_a_file_name_does_not_break_the_mount(
    tmp_path: Path,
) -> None:
    """exp/scalars/loss.json is a file of the run exp and a folder on disk."""
    _write_run(tmp_path / "exp", 1, loss=1.0)
    _write_run(tmp_path / "exp" / "scalars" / "loss.json", 1, loss=2.0)
    fs = TensorBoardFS(tmp_path)

    assert _values(fs, "/exp/scalars/loss.json") == [1.0]
    assert _values(fs, "/exp/scalars~2/loss.json/scalars/loss.json") == [2.0]


def test_a_run_folder_below_the_log_directory_run_gets_a_suffix(
    tmp_path: Path,
) -> None:
    """The log directory itself is a run, and its tabs keep their names."""
    _write_run(tmp_path, 1, value=1.0)
    _write_run(tmp_path / "scalars", 1, value=2.0)
    fs = TensorBoardFS(tmp_path)

    assert _values(fs, "/scalars/value.json") == [1.0]
    assert _values(fs, "/scalars~2/scalars/value.json") == [2.0]


def test_runs_below_a_renamed_folder_follow_it(
    tmp_path: Path, warnings: list[str]
) -> None:
    """A folder named scalars~2 on disk keeps that name; the moved one skips it."""
    _write_run(tmp_path / "exp", 1, value=1.0)
    _write_run(tmp_path / "exp" / "scalars", 1, value=2.0)
    _write_run(tmp_path / "exp" / "scalars" / "sub", 1, value=3.0)
    _write_run(tmp_path / "exp" / "scalars~2", 1, value=4.0)
    fs = TensorBoardFS(tmp_path)

    assert _values(fs, "/exp/scalars~3/scalars/value.json") == [2.0]
    assert _values(fs, "/exp/scalars~3/sub/scalars/value.json") == [3.0]
    assert _values(fs, "/exp/scalars~2/scalars/value.json") == [4.0]
    assert len(warnings) == 1


def test_a_renamed_run_folder_is_logged_once(
    tmp_path: Path, warnings: list[str]
) -> None:
    """A run that grows rebuilds the tree, but does not repeat the warning."""
    _write_run(tmp_path / "exp", 1, value=1.0)
    _write_run(tmp_path / "exp" / "scalars", 1, value=2.0)
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    fs.readdir("/exp", None)
    _write_run(tmp_path / "exp" / "scalars", 2, value=5.0)

    assert _values(fs, "/exp/scalars~2/scalars/value.json") == [2.0, 5.0]
    assert len(warnings) == 1
