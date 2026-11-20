"""A refresh lists each run directory once and opens no unchanged event file.

On a network filesystem each listing and each open is a round trip, so a
log directory with many runs is checked with as few of them as possible.
"""

import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from test_incremental_parse import _frame, _loss
from test_open_handles import _append

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2

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
    real_os_open = os.open

    def spy(self: Path, *args: Any, **kwargs: Any) -> Any:
        opened.append(self)
        return real_open(self, *args, **kwargs)

    def spy_fd(
        path: str | bytes | int | os.PathLike[str] | os.PathLike[bytes],
        *args: Any,
        **kwargs: Any,
    ) -> int:
        if isinstance(path, (str, Path)):
            opened.append(Path(path))
        return real_os_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy)
    monkeypatch.setattr(os, "open", spy_fd)
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
    _append(event, 1)
    opened = _spy_opens(monkeypatch)

    after = fs.read("/run/scalars/loss.json", 1 << 20, 0)

    assert event in opened
    assert len(after) > len(before)


def test_root_and_active_run_do_not_read_idle_subtrees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = [f"run{i:02}" for i in range(32)]
    active, *idle = _runs(tmp_path, *names)
    for event in (active, *idle):
        profile = event.parent / "plugins" / "profile"
        profile.mkdir(parents=True)
        (profile / "trace").write_bytes(b"trace")
    listed = _spy_listings(monkeypatch)
    opened = _spy_opens(monkeypatch)
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    assert fs.readdir("/")[2:] == names
    assert listed == [tmp_path]
    assert opened == []

    assert fs.read("/run00/profile/trace", 16, 0) == b"trace"
    _append(idle[-1], 1)
    opened.clear()
    listed.clear()
    fs.readdir("/run00/scalars")
    assert all(path.is_relative_to(active.parent) for path in listed)
    assert not set(idle).intersection(opened)


def test_a_listing_of_an_earlier_discovery_is_not_reused(tmp_path: Path) -> None:
    """The first refresh after the mount lists the run again."""
    _runs(tmp_path, "run")
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=3600)
    event = compat_pb2.Event(wall_time=1.0, step=1)
    event.summary.value.add(tag="accuracy", simple_value=0.5)
    (tmp_path / "run" / f"{EVENTS}.2").write_bytes(_frame(event))

    assert "accuracy.json" in fs.readdir("/run/scalars")


@pytest.mark.parametrize("tail_bytes", [0, 9], ids=["same-size", "shrinking-tail"])
def test_rewrite_before_unchanged_checkpoint_refreshes_scalars(
    tmp_path: Path, tail_bytes: int
) -> None:
    """Rewrites and shrinks invalidate the whole index, even with the old mtime."""
    event = tmp_path / EVENTS
    frames = _loss(0, 1.0) + _loss(1, 1.0)
    event.write_bytes(frames + _loss(2, 1.0)[:tail_bytes])
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    before = fs.read("/scalars/loss.json", 1 << 20, 0)
    times = event.stat()
    time.sleep(0.02)  # past the clock tick of coarse file timestamps
    with event.open("r+b") as handle:
        handle.write(_loss(0, 2.0))
        if tail_bytes:
            handle.truncate(len(frames) + tail_bytes - 1)
    os.utime(event, ns=(times.st_atime_ns, times.st_mtime_ns))

    after = fs.read("/scalars/loss.json", 1 << 20, 0)
    assert after != before
    assert after == TensorBoardFS(tmp_path).read("/scalars/loss.json", 1 << 20, 0)


def test_unreadable_replacement_drops_only_its_previous_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad, _ = _runs(tmp_path, "bad", "good")
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    good_path = "/good/scalars/loss.json"
    bad_path = "/bad/scalars/loss.json"
    expected = fs.read(good_path, 4096, 0)
    assert fs.read(bad_path, 4096, 0) == expected
    replacement = bad.with_name("replacement")
    replacement.write_bytes(_loss(9, 2.0))
    replacement.replace(bad)
    original = os.open

    def deny_replacement(path: Path, *args: Any, **kwargs: Any) -> int:
        if path == bad:
            raise PermissionError(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", deny_replacement)
    with pytest.raises(FileNotFoundError):
        fs.read(bad_path, 4096, 0)
    assert fs.read(good_path, 4096, 0) == expected


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


def test_stat_of_frozen_raw_frame_needs_no_source_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (event,) = _runs(tmp_path, "run")
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=float("inf"))
    directory = f"/run/raw/{EVENTS}"
    (filename,) = fs.readdir(directory)[2:]
    opened = _spy_opens(monkeypatch)
    assert fs.getattr(f"{directory}/{filename}")["st_size"] == event.stat().st_size
    assert not opened
