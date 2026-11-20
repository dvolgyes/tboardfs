"""File times and sizes come from the data, so they change only with the data."""

import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from tboardfs import TensorBoardFS
import tboardfs.filesystem
from tboardfs.proto import compat_pb2

from test_incremental_parse import _files, _frame
from test_refresh_io import _spy_listings, _spy_opens


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
    assert _times(fs, "/") == (1000.0, 1000.0, 1000.0)
    assert _times(fs, "/b/scalars") == (2000.0, 2000.0, 2000.0)
    assert _times(fs, "/") == (2000.0, 2000.0, 2000.0)


def test_stat_of_cold_run_uses_the_parent_time_until_descent(tmp_path: Path) -> None:
    _scalar(tmp_path, 1000.0, "root")
    _scalar(tmp_path / "a", 2000.0, "a")
    _scalar(tmp_path / "a" / "b", 3000.0, "b")
    os.utime(tmp_path, (500.0, 500.0))
    os.utime(tmp_path / "a", (550.0, 550.0))
    os.utime(tmp_path / "a" / "b", (575.0, 575.0))
    fs = TensorBoardFS(tmp_path)

    assert _times(fs, "/") == (500.0, 500.0, 500.0)
    assert "a" in fs.readdir("/")
    assert stat.S_ISDIR(fs.getattr("/a")["st_mode"])
    assert _times(fs, "/a") == (1000.0, 1000.0, 1000.0)
    assert _times(fs, "/") == (1000.0, 1000.0, 1000.0)

    assert _times(fs, "/a/b") == (2000.0, 2000.0, 2000.0)
    assert _times(fs, "/a/b/scalars/b.json") == (3000.0, 3000.0, 3000.0)

    _scalar(tmp_path / "a" / "b", 4000.0, "b")
    assert _times(fs, "/a/b") == (4000.0, 4000.0, 4000.0)


def test_stat_separates_a_cold_run_from_the_colliding_virtual_dir(
    tmp_path: Path,
) -> None:
    _scalar(tmp_path, 1000.0, "root")
    _scalar(tmp_path / "scalars", 2000.0, "child")
    fs = TensorBoardFS(tmp_path)

    assert _times(fs, "/scalars~2") == (1000.0, 1000.0, 1000.0)
    _scalar(tmp_path, 3000.0, "root")
    assert _times(fs, "/scalars") == (3000.0, 3000.0, 3000.0)
    assert "child.json" in fs.readdir("/scalars~2/scalars")
    assert _times(fs, "/scalars~2") == (2000.0, 2000.0, 2000.0)


def test_stat_follows_a_new_collision_for_a_visited_run(tmp_path: Path) -> None:
    _scalar(tmp_path / "scalars", 2000.0, "child")
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    assert "child.json" in fs.readdir("/scalars/scalars")
    assert _times(fs, "/scalars") == (2000.0, 2000.0, 2000.0)

    _scalar(tmp_path, 3000.0, "root")
    assert _times(fs, "/scalars") == (3000.0, 3000.0, 3000.0)
    assert _times(fs, "/scalars~2") == (2000.0, 2000.0, 2000.0)


def test_cold_stat_uses_backing_time_without_publishing_it(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _scalar(run, 2000.0, "loss")
    os.utime(run, (500.0, 500.0))
    fs = TensorBoardFS(tmp_path)
    assert "run" in fs.readdir("/")

    assert _times(fs, "/run") == (500.0, 500.0, 500.0)
    assert _times(fs, "/") == (0.0, 0.0, 0.0)
    assert "scalars" in fs.readdir("/run")
    assert _times(fs, "/run") == (2000.0, 2000.0, 2000.0)
    assert _times(fs, "/") == (2000.0, 2000.0, 2000.0)


def test_cold_stat_does_not_read_retargeted_directory_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    mount = tmp_path / "mount"
    outside = tmp_path / "outside"
    source.mkdir()
    mount.mkdir()
    outside.mkdir()
    cache = tmp_path / "cache"
    storage = cache / "tboardfs" / "indexes-v1"
    storage.mkdir(parents=True)
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    os.utime(outside, (500.0, 500.0))
    run = source / "run"
    run.symlink_to(outside, target_is_directory=True)
    fs = TensorBoardFS(source, mountpoint=mount, disk_index=True)
    assert "run" in fs.readdir("/")
    assert _times(fs, "/run") == (500.0, 500.0, 500.0)

    run.unlink()
    run.symlink_to(mount, target_is_directory=True)
    assert _times(fs, "/run") == (0.0, 0.0, 0.0)

    run.unlink()
    run.symlink_to(storage, target_is_directory=True)
    assert _times(fs, "/run") == (0.0, 0.0, 0.0)

    run.unlink()
    run.symlink_to(outside / "missing", target_is_directory=True)
    assert _times(fs, "/run") == (0.0, 0.0, 0.0)


def test_listing_then_stat_does_not_open_or_scan_cold_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(32):
        _scalar(tmp_path / f"run{i:02}", 1000.0 + i, "loss")
    os.utime(tmp_path, (500.0, 500.0))
    fs = TensorBoardFS(tmp_path)
    opened = _spy_opens(monkeypatch)
    listed = _spy_listings(monkeypatch)

    assert _times(fs, "/") == (500.0, 500.0, 500.0)
    assert listed == []
    names = fs.readdir("/")[2:]
    assert len(names) == 32
    assert all(stat.S_ISDIR(fs.getattr(f"/{name}")["st_mode"]) for name in names)
    assert listed == [tmp_path]
    assert opened == []

    assert "scalars" in fs.readdir("/run00")
    assert listed == [tmp_path, tmp_path / "run00"]
    assert tmp_path / "run00" / "events.out.tfevents.host" in opened


def test_stat_validates_cold_topology_at_the_discovery_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(
        tboardfs.filesystem, "time", SimpleNamespace(time=lambda: clock[0])
    )
    _scalar(tmp_path, 1000.0, "root")
    _scalar(tmp_path / "old", 2000.0, "loss")
    (tmp_path / "replaced").mkdir()
    fs = TensorBoardFS(tmp_path)
    assert "old" in fs.readdir("/")

    (tmp_path / "old").rename(tmp_path / "new")
    (tmp_path / "replaced").rmdir()
    (tmp_path / "replaced").write_bytes(b"not a directory")
    _scalar(tmp_path / "new" / "child", 3000.0, "loss")
    clock[0] += 149
    assert _times(fs, "/old") == (1000.0, 1000.0, 1000.0)

    clock[0] += 1
    for missing in ("/old", "/replaced"):
        with pytest.raises(FileNotFoundError):
            fs.getattr(missing)
    assert _times(fs, "/new/child") == (2000.0, 2000.0, 2000.0)
    assert "scalars" in fs.readdir("/new/child")


def test_stat_does_not_use_a_real_directory_placeholder_for_a_sidecar(
    tmp_path: Path,
) -> None:
    _scalar(tmp_path, 1000.0, "root")
    plugins = tmp_path / "plugins"
    _scalar(plugins, 2000.0, "nested")
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    assert "plugins~2" in fs.readdir("/")
    assert _times(fs, "/plugins~2") == (1000.0, 1000.0, 1000.0)

    (plugins / "events.out.tfevents.host").unlink()
    profile = plugins / "profile"
    profile.mkdir()
    trace = profile / "trace"
    trace.write_bytes(b"trace")
    os.utime(trace, (3000.0, 3000.0))

    with pytest.raises(FileNotFoundError):
        fs.getattr("/plugins~2")
    assert _times(fs, "/profile/trace") == (3000.0, 3000.0, 3000.0)
    assert _times(fs, "/profile") == (3000.0, 3000.0, 3000.0)
    assert fs.read("/profile/trace", 32, 0) == b"trace"


def test_a_directory_time_follows_new_data(tmp_path: Path) -> None:
    _scalar(tmp_path, 1000.0, "loss")
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    assert fs.getattr("/scalars")["st_mtime"] == 1000.0
    _scalar(tmp_path, 3000.0, "loss")

    assert fs.getattr("/scalars")["st_mtime"] == 3000.0


def test_a_file_has_the_blocks_of_its_size(tmp_path: Path) -> None:
    """du counts 512-byte blocks, not the size."""
    _scalar(tmp_path, 1000.0, "loss")
    fs = TensorBoardFS(tmp_path)

    status = fs.getattr("/scalars/loss.json")
    assert status["st_size"] > 0
    assert status["st_blocks"] == (status["st_size"] + 511) // 512
    assert fs.getattr("/scalars")["st_blocks"] == 0


def test_each_fixture_file_has_the_size_of_its_bytes() -> None:
    """A size that differs from the bytes cuts reads or pads them with zeros."""
    fs = TensorBoardFS(Path(__file__).resolve().parent.parent / "test-logs")
    sizes = {path: fs.getattr(path)["st_size"] for path in _files(fs)}

    assert len(sizes) > 300
    assert {path: len(fs.read(path, size + 1, 0)) for path, size in sizes.items()} == (
        sizes
    )
