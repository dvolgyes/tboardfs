"""A refresh lists each run directory once and opens no unchanged event file.

On a network filesystem each listing and each open is a round trip, so a
log directory with many runs is checked with as few of them as possible.
"""

from collections.abc import Iterator
import os
from pathlib import Path
import time
from typing import Any

import pytest

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2

from test_incremental_parse import _frame, _loss
from test_open_handles import _append

EVENTS = "events.out.tfevents.host"


def _runs(tmp_path: Path, *names: str) -> list[Path]:
    events = []
    for name in names:
        (tmp_path / name).mkdir()
        _append(tmp_path / name / EVENTS, 0)
        events.append(tmp_path / name / EVENTS)
    return events


def _spy_opens(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    opened: list[Path] = []
    real_open = Path.open

    def spy(self: Path, *args: Any, **kwargs: Any) -> Any:
        opened.append(self)
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy)
    return opened


def _spy_listings(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    listed: list[Path] = []
    real_scandir = os.scandir

    def spy(path: Path) -> Iterator[os.DirEntry[str]]:
        listed.append(Path(path))
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", spy)
    return listed


def test_an_unchanged_event_file_is_not_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (event,) = _runs(tmp_path, "run")
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    fs.readdir("/run/scalars")
    opened = _spy_opens(monkeypatch)

    fs.readdir("/run/scalars")

    assert event not in opened


def test_a_grown_event_file_is_read_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (event,) = _runs(tmp_path, "run")
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    before = fs.read("/run/scalars/loss.json", 1 << 20, 0)
    opened = _spy_opens(monkeypatch)
    _append(event, 1)

    after = fs.read("/run/scalars/loss.json", 1 << 20, 0)

    assert event in opened
    assert len(after) > len(before)


def test_a_refresh_lists_each_run_directory_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _runs(tmp_path, "a", "b")
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    fs.readdir("/")
    listed = _spy_listings(monkeypatch)

    fs.readdir("/")

    assert sorted(listed) == [tmp_path, tmp_path / "a", tmp_path / "b"]


def test_a_listing_of_an_earlier_discovery_is_not_reused(tmp_path: Path) -> None:
    """The first refresh after the mount lists the run again."""
    _runs(tmp_path, "run")
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=3600)
    event = compat_pb2.Event(wall_time=1.0, step=1)
    event.summary.value.add(tag="accuracy", simple_value=0.5)
    (tmp_path / "run" / f"{EVENTS}.2").write_bytes(_frame(event))

    assert "accuracy.json" in fs.readdir("/run/scalars")


def test_a_rewrite_in_place_with_the_old_mtime_is_read_again(tmp_path: Path) -> None:
    """A write changes the ctime, which no program can set back."""
    event = tmp_path / EVENTS
    event.write_bytes(_loss(0, 1.0))
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    before = fs.read("/scalars/loss.json", 1 << 20, 0)
    times = event.stat()
    time.sleep(0.02)  # past the clock tick of coarse file timestamps
    with event.open("r+b") as handle:
        handle.write(_loss(0, 2.0))
    os.utime(event, ns=(times.st_atime_ns, times.st_mtime_ns))

    assert fs.read("/scalars/loss.json", 1 << 20, 0) != before


def test_a_touched_event_file_is_opened_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (event,) = _runs(tmp_path, "run")
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    fs.readdir("/run/scalars")
    os.utime(event, ns=(1, 1))
    opened = _spy_opens(monkeypatch)

    fs.readdir("/run/scalars")
    fs.readdir("/run/scalars")

    assert opened.count(event) == 1
