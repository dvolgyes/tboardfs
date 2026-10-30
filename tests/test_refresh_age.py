"""The mount reads the logs again only when the refresh age has passed."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tboardfs import TensorBoardFS
import tboardfs.filesystem
from tboardfs.proto import compat_pb2

from test_incremental_parse import _frame


def _scalar(run_dir: Path, step: int) -> None:
    event = compat_pb2.Event(step=step, wall_time=1.0)
    event.summary.value.add(tag="loss", simple_value=1.0)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "events.out.tfevents.host").open("ab") as handle:
        handle.write(_frame(event))


def test_new_data_and_new_runs_show_after_the_refresh_age(
    tmp_path: Path, monkeypatch: Any
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(
        tboardfs.filesystem, "time", SimpleNamespace(time=lambda: clock[0])
    )
    _scalar(tmp_path / "a", 1)
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=60)
    size = fs.getattr("/a/scalars/loss.json")["st_size"]
    _scalar(tmp_path / "a", 2)
    _scalar(tmp_path / "b", 1)

    clock[0] += 59
    assert fs.getattr("/a/scalars/loss.json")["st_size"] == size
    assert fs.readdir("/")[2:] == ["a"]

    clock[0] += 1
    assert fs.getattr("/a/scalars/loss.json")["st_size"] > size
    assert fs.readdir("/")[2:] == ["a", "b"]
    assert fs.getattr("/b/scalars/loss.json")["st_size"] > 0

    _scalar(tmp_path / "b", 2)
    clock[0] += 59
    assert fs.getattr("/b/scalars/loss.json")["st_size"] == size
    clock[0] += 1
    assert fs.getattr("/b/scalars/loss.json")["st_size"] > size
