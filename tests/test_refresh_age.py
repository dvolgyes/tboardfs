"""Discovery is throttled locally, independently of checks on known files."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from test_incremental_parse import _frame

import tboardfs.filesystem
from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2


def _scalar(run_dir: Path, step: int) -> None:
    event = compat_pb2.Event(step=step, wall_time=1.0)
    event.summary.value.add(tag="loss", simple_value=1.0)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "events.out.tfevents.host").open("ab") as handle:
        handle.write(_frame(event))


def test_known_data_refreshes_before_new_runs_are_discovered(
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
    assert fs.getattr("/a/scalars/loss.json")["st_size"] > size
    assert fs.readdir("/")[2:] == ["a"]

    clock[0] += 1
    assert fs.getattr("/a/scalars/loss.json")["st_size"] > size
    assert fs.readdir("/")[2:] == ["a", "b"]
    assert fs.getattr("/b/scalars/loss.json")["st_size"] > 0

    _scalar(tmp_path / "b", 2)
    clock[0] += 59
    assert fs.getattr("/b/scalars/loss.json")["st_size"] > size
    clock[0] += 1
    assert fs.getattr("/b/scalars/loss.json")["st_size"] > size


def test_new_file_discovery_is_local_but_deletion_is_immediate(
    tmp_path: Path, monkeypatch: Any
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(
        tboardfs.filesystem, "time", SimpleNamespace(time=lambda: clock[0])
    )
    run = tmp_path / "a"
    _scalar(run, 0)
    fs = TensorBoardFS(tmp_path)
    assert "loss.json" in fs.readdir("/a/scalars")
    event = compat_pb2.Event(step=1, wall_time=1.0)
    event.summary.value.add(tag="accuracy", simple_value=0.5)
    (run / "events.out.tfevents.second").write_bytes(_frame(event))
    (run / "events.out.tfevents.host").unlink()

    clock[0] += 149
    assert "scalars" not in fs.readdir("/a")
    clock[0] += 1
    assert fs.readdir("/a/scalars")[2:] == [
        "accuracy.json",
        "accuracy.npz",
        "accuracy.tsv",
    ]


def test_infinite_interval_freezes_each_directory_after_first_access(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    _scalar(run, 0)
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=float("inf"))
    before = fs.read("/run/scalars/loss.json", 4096, 0)
    _scalar(run, 1)
    _scalar(tmp_path / "new", 2)

    assert fs.read("/run/scalars/loss.json", 4096, 0) == before
    assert fs.readdir("/")[2:] == ["run"]
