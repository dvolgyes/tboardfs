"""Reject source replacements without following a path into the mount."""

from errno import EINVAL, ENOENT
import os
from pathlib import Path

import pytest

from tboardfs import TensorBoardFS, parallel_index
from tboardfs.event_source import EventSource
from tboardfs.paths import _Paths


@pytest.mark.parametrize("mounted", [False, True])
def test_replacement_fifo_is_rejected_on_the_open_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mounted: bool
) -> None:
    source = tmp_path / "events.out.tfevents.test"
    source.write_bytes(b"event")
    mountpoint = tmp_path / "mount" if mounted else None
    original_open = os.open
    swapped = False

    def replace_on_open(
        path: str | Path, flags: int, *, dir_fd: int | None = None
    ) -> int:
        nonlocal swapped
        if Path(path).name == source.name:
            assert flags & os.O_NONBLOCK
            source.unlink()
            os.mkfifo(source)
            swapped = True
        return original_open(path, flags, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", replace_on_open)

    with pytest.raises(OSError) as raised:
        EventSource.open(source, mountpoint=mountpoint)

    assert swapped
    assert raised.value.errno == EINVAL


def test_replaced_parent_is_not_followed_into_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_dir = tmp_path / "run"
    source_dir.mkdir()
    mountpoint = tmp_path / "mount"
    mountpoint.mkdir()
    source = source_dir / "events.out.tfevents.test"
    source.write_bytes(b"event")
    original_open = os.open
    swapped = False

    def replace_on_open(
        path: str | Path, flags: int, *, dir_fd: int | None = None
    ) -> int:
        nonlocal swapped
        if path == source_dir.name:
            assert flags & os.O_NOFOLLOW
            source_dir.rename(tmp_path / "old-run")
            source_dir.symlink_to(mountpoint, target_is_directory=True)
            swapped = True
        return original_open(path, flags, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", replace_on_open)

    with pytest.raises(OSError) as raised:
        EventSource.open(source, mountpoint=mountpoint)
    assert swapped
    assert raised.value.errno == ENOENT


def test_cold_fallback_rechecks_retargeted_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_is_file = _Paths.is_file
    run = tmp_path / "run"
    run.mkdir()
    external = tmp_path / "external"
    with external.open("wb") as stream:
        stream.truncate(8 << 20)
    mountpoint = tmp_path / "mount"
    mountpoint.mkdir()
    source = run / "events.out.tfevents.test"
    source.symlink_to(Path("..") / external.name)
    inside_mount = mountpoint / "loop"
    inside_mount.touch()
    original_open = os.open
    attempted = False

    def guard_open(path: str | Path, flags: int, *, dir_fd: int | None = None) -> int:
        assert Path(path).name not in {mountpoint.name, inside_mount.name}
        return original_open(path, flags, dir_fd=dir_fd)

    def guard_is_file(entry: os.DirEntry[str]) -> bool:
        assert not (attempted and entry.name == source.name)
        return original_is_file(entry)

    def decline(path: Path, workers: int, **kwargs: object) -> None:
        nonlocal attempted
        assert path == source
        assert workers == 2
        assert kwargs["mountpoint"] == mountpoint
        source.unlink()
        source.symlink_to(inside_mount)
        attempted = True

    fs = TensorBoardFS(run, mountpoint=mountpoint, index_workers=2)
    assert EventSource.stat(source, mountpoint=mountpoint).st_size == 8 << 20
    with EventSource.open(source, mountpoint=mountpoint) as stream:
        assert stream.read(1) == b"\0"
    monkeypatch.setattr(os, "open", guard_open)
    monkeypatch.setattr(parallel_index, "parallel_index", decline)
    monkeypatch.setattr(_Paths, "is_file", guard_is_file)

    assert fs.getattr("/")["st_size"] == 0
    assert attempted
    assert not fs._state.runs[()].files
