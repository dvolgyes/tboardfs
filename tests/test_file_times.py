"""File times come from the data, so they stay the same until the data changes."""

from pathlib import Path

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2

from test_incremental_parse import _frame


def _scalar(run_dir: Path, wall_time: float, tag: str) -> None:
    event = compat_pb2.Event(step=1, wall_time=wall_time)
    event.summary.value.add(tag=tag, simple_value=1.0)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "events.out.tfevents.host").open("ab") as handle:
        handle.write(_frame(event))


def _times(fs: TensorBoardFS, path: str) -> tuple[float, float, float]:
    status = fs.getattr(path)
    return status["st_mtime"], status["st_ctime"], status["st_atime"]


def test_a_file_has_its_event_time_as_all_three_times(tmp_path: Path) -> None:
    """tar compares the change time before and after it reads a file."""
    _scalar(tmp_path, 1000.0, "loss")
    fs = TensorBoardFS(tmp_path)

    assert _times(fs, "/scalars/loss.json") == (1000.0, 1000.0, 1000.0)


def test_a_directory_has_the_newest_time_below_it(tmp_path: Path) -> None:
    """An empty directory has the time of the directory above it."""
    _scalar(tmp_path / "a", 1000.0, "loss")
    _scalar(tmp_path / "b", 2000.0, "acc")
    fs = TensorBoardFS(tmp_path)

    assert _times(fs, "/a/scalars") == (1000.0, 1000.0, 1000.0)
    assert _times(fs, "/a") == (1000.0, 1000.0, 1000.0)
    assert _times(fs, "/a/images") == (1000.0, 1000.0, 1000.0)
    assert _times(fs, "/") == (2000.0, 2000.0, 2000.0)


def test_a_directory_time_follows_new_data(tmp_path: Path) -> None:
    _scalar(tmp_path, 1000.0, "loss")
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    assert fs.getattr("/scalars")["st_mtime"] == 1000.0
    _scalar(tmp_path, 3000.0, "loss")

    assert fs.getattr("/scalars")["st_mtime"] == 3000.0
