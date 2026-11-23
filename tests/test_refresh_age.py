"""Discovery is throttled locally, independently of checks on known files."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_incremental_parse import _frame, _loss
from test_refresh_io import _spy_listings

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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(tboardfs.filesystem, "_source_clock", lambda: clock[0])
    _scalar(tmp_path / "a", 1)
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=60, source_check_seconds=0)
    size = fs.getattr("/a/scalars/loss.json")["st_size"]
    _scalar(tmp_path / "a", 2)
    _scalar(tmp_path / "b", 1)

    clock[0] += 59
    assert fs.getattr("/a/scalars/loss.json")["st_size"] > size
    assert fs.readdir("/")[2:] == ["a", "llms.txt"]

    clock[0] += 1
    assert fs.getattr("/a/scalars/loss.json")["st_size"] > size
    assert fs.readdir("/")[2:] == ["a", "b", "llms.txt"]
    assert fs.getattr("/b/scalars/loss.json")["st_size"] > 0

    _scalar(tmp_path / "b", 2)
    clock[0] += 59
    assert fs.getattr("/b/scalars/loss.json")["st_size"] > size
    clock[0] += 1
    assert fs.getattr("/b/scalars/loss.json")["st_size"] > size


def test_new_file_discovery_is_local_but_deletion_is_immediate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(tboardfs.filesystem, "_source_clock", lambda: clock[0])
    run = tmp_path / "a"
    _scalar(run, 0)
    fs = TensorBoardFS(tmp_path, source_check_seconds=0)
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = 0.0
    monkeypatch.setattr(tboardfs.filesystem, "_source_clock", lambda: now)
    run = tmp_path / "run"
    _scalar(run, 0)
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=float("inf"))
    before = fs.read("/run/scalars/loss.json", 4096, 0)
    _scalar(run, 1)
    _scalar(tmp_path / "new", 2)
    now = 1000.0

    assert fs.read("/run/scalars/loss.json", 4096, 0) == before
    assert fs.readdir("/")[2:] == ["llms.txt", "run"]


def test_default_source_deadline_is_not_slid_by_reads_or_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = 0.0
    wall = 1000.0
    monkeypatch.setattr(tboardfs.filesystem, "_source_clock", lambda: now)
    monkeypatch.setattr(tboardfs.filesystem, "time", SimpleNamespace(time=lambda: wall))
    a, b = tmp_path / "a", tmp_path / "b"
    _scalar(a, 1)
    _scalar(b, 1)
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=90)
    first, second = "/a/scalars/loss.json", "/b/scalars/loss.json"
    before = fs.read(first, 4096, 0)

    _scalar(a, 2)
    _scalar(b, 2)
    assert fs.read(first, 4096, 0) == before

    wall = 50_000.0
    listed = _spy_listings(monkeypatch)
    now = 89.0
    assert fs.readdir("/a/scalars")[2:] == ["loss.json", "loss.npz", "loss.tsv"]
    assert not listed

    now = 90.0
    assert fs.readdir("/a/scalars")[2:] == ["loss.json", "loss.npz", "loss.tsv"]
    assert a in listed
    assert fs.read(first, 4096, 0) == before
    assert fs._state.runs[("a",)].refreshed_at == 1000.0
    second_before = fs.read(second, 4096, 0)
    assert second_before != before
    _scalar(b, 3)

    now = 119.999
    assert fs.read(first, 4096, 0) == before

    wall = 500.0
    now = 120.0
    assert fs.read(first, 4096, 0) == second_before
    assert fs._state.runs[("a",)].refreshed_at == wall
    assert fs.read(second, 4096, 0) == second_before

    now = 209.999
    assert fs.read(second, 4096, 0) == second_before

    now = 210.0
    assert fs.read(second, 4096, 0) != second_before


@pytest.mark.parametrize("change", ["rewrite", "replace", "truncate", "unlink"])
def test_source_expiry_updates_scalar_formats_but_not_open_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    now = 0.0
    monkeypatch.setattr(tboardfs.filesystem, "_source_clock", lambda: now)
    event = tmp_path / "events.out.tfevents.host"
    first, last = _loss(1, 1.0), _loss(2, 2.0)
    event.write_bytes(first + last)
    fs = TensorBoardFS(tmp_path, scalar_formats=("json", "tsv"))
    json_path, tsv_path = "/scalars/loss.json", "/scalars/loss.tsv"
    before = fs.read(json_path, 4096, 0)
    tsv_before = fs.read(tsv_path, 4096, 0)
    handle = fs.open(json_path, os.O_RDONLY)
    try:
        if change == "unlink":
            event.unlink()
        elif change == "truncate":
            event.write_bytes(first)
        else:
            changed = _loss(1, 9.0) + last
            assert len(changed) == len(first + last)
            target = event.with_name("replacement") if change == "replace" else event
            target.write_bytes(changed)
            if change == "replace":
                target.replace(event)
            else:
                os.utime(event, ns=(1, 1))

        now = 119.999
        assert fs.read(json_path, 4096, 0) == before
        assert fs.read(tsv_path, 4096, 0) == tsv_before

        now = 120.0
        if change == "unlink":
            with pytest.raises(FileNotFoundError):
                fs.open(json_path, os.O_RDONLY)
            with pytest.raises(FileNotFoundError):
                fs.getattr(tsv_path)
        else:
            fresh = TensorBoardFS(tmp_path, scalar_formats=("json", "tsv"))
            expected = fresh.read(json_path, 4096, 0)
            current = fs.open(json_path, os.O_RDONLY)
            try:
                assert fs.read(json_path, 4096, 0, current) == expected != before
            finally:
                fs.release(json_path, current)
            expected_tsv = fresh.read(tsv_path, 4096, 0)
            assert fs.read(tsv_path, 4096, 0) == expected_tsv != tsv_before
        assert fs.read(json_path, 4096, 0, handle) == before
    finally:
        fs.release(json_path, handle)
