"""Reject source replacements without following a path into the mount."""

import ctypes
import json
import os
import subprocess
import sys
from errno import EACCES, EINVAL, ENOENT
from pathlib import Path
from textwrap import dedent

import pytest
from test_incremental_parse import _frame

from tboardfs import TensorBoardFS, _directory, parallel_index
from tboardfs.event_source import EventSource
from tboardfs.paths import _Paths
from tboardfs.proto import compat_pb2


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

    fs.readdir("/")
    assert attempted
    assert not fs._state.runs[()].files


def _scalar_record(step: int) -> bytes:
    event = compat_pb2.Event(wall_time=100.0 + step, step=step)
    event.summary.value.add(tag="loss", simple_value=float(step))
    return _frame(event)


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux openat2")
def test_deep_sibling_status_under_descriptor_limit(tmp_path: Path) -> None:
    """A deep ordinary parent does not need a descriptor for every ancestor."""

    class OpenHow(ctypes.Structure):
        _fields_ = [
            ("flags", ctypes.c_uint64),
            ("mode", ctypes.c_uint64),
            ("resolve", ctypes.c_uint64),
        ]

    libc = ctypes.CDLL(None, use_errno=True)
    openat2 = getattr(libc, "openat2", None)
    if openat2 is None:
        pytest.skip("libc does not export openat2")
    openat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(OpenHow),
        ctypes.c_size_t,
    )
    openat2.restype = ctypes.c_int
    how = OpenHow(os.O_PATH | os.O_DIRECTORY | os.O_CLOEXEC, 0, 0x04)
    fd = openat2(-100, b".", ctypes.byref(how), ctypes.sizeof(how))
    if fd < 0:
        pytest.skip(f"openat2 unavailable: {os.strerror(ctypes.get_errno())}")
    os.close(fd)

    run = tmp_path.resolve().joinpath(*(["level"] * 64))
    run.mkdir(parents=True)
    for index in (1, 2):
        (run / f"events.out.tfevents.{index}").write_bytes(bytes(index))

    script = dedent("""
        import resource
        import sys
        from pathlib import Path

        from tboardfs.event_source import EventSource

        run, mount, storage = map(Path, sys.argv[1:])
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        limit = 32 if soft == resource.RLIM_INFINITY else min(soft, 32)
        resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))
        with EventSource._sibling_stat(run, mountpoint=mount, storage=storage) as stat:
            for index in (1, 2):
                path = run / f"events.out.tfevents.{index}"
                assert stat(path) == path.stat()
        """)
    result = subprocess.run(
        [sys.executable, "-c", script, run, tmp_path / "mount", tmp_path / "storage"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_live_siblings_share_a_guarded_walk_and_refresh_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "training" / "live"
    run.mkdir(parents=True)
    mountpoint = tmp_path / "mount"
    mountpoint.mkdir()
    paths = [run / f"events.out.tfevents.{step:02}" for step in range(24)]
    for step, path in enumerate(paths):
        path.write_bytes(_scalar_record(step))
    fs = TensorBoardFS(run, mountpoint=mountpoint, refresh_age_seconds=3600)
    virtual = "/scalars/loss.json"

    def steps() -> list[int]:
        return [row["step"] for row in json.loads(fs.read(virtual, 1 << 20, 0))]

    assert steps() == list(range(24))
    assert len(fs._state.runs[()].files) == len(paths)
    original_open = os.open
    ancestor_opens = 0

    def count_open(path: str | Path, flags: int, *, dir_fd: int | None = None) -> int:
        nonlocal ancestor_opens
        assert path != mountpoint.name
        if path == run.name:
            ancestor_opens += 1
        return original_open(path, flags, dir_fd=dir_fd)

    with monkeypatch.context() as patch:
        patch.setattr(_directory, "_OPEN_DIRECTORY", None)
        patch.setattr(os, "open", count_open)
        assert steps() == list(range(24))
        assert 0 < ancestor_opens < len(paths) // 2

    with paths[-1].open("ab") as stream:
        stream.write(_scalar_record(30))
    assert steps() == [*range(24), 30]

    replacement = run / "replacement"
    replacement.write_bytes(_scalar_record(31))
    os.replace(replacement, paths[-1])
    assert steps() == [*range(23), 31]

    paths[0].unlink()
    assert steps() == [*range(1, 23), 31]
    assert paths[0] not in fs._state.runs[()].files

    blocked = mountpoint / "blocked"
    blocked.write_bytes(_scalar_record(40))
    paths[-1].unlink()
    paths[-1].symlink_to(blocked)
    assert steps() == list(range(1, 23))
    assert paths[-1] not in fs._state.runs[()].files


def test_sibling_status_resolves_aliases_again_after_retargeting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    alias = tmp_path / "alias"
    mountpoint = second / "mount"
    first.mkdir()
    mountpoint.mkdir(parents=True)
    for directory, content in ((first, b"first"), (second, b"second")):
        (directory / "source").write_bytes(content)
        (directory / "link").symlink_to(Path("..") / directory.name / "source")
    (first / mountpoint.name).touch()
    nested = first / "nested"
    nested.mkdir()
    (nested / "source").write_bytes(b"nested source")
    second_status = second.stat()
    original_open = os.open
    original_stat = os.stat

    def guard_mount(path: str | Path, dir_fd: int | None) -> None:
        if path == mountpoint.name and dir_fd is not None:
            assert not os.path.samestat(os.fstat(dir_fd), second_status)

    def guard_open(path: str | Path, flags: int, *, dir_fd: int | None = None) -> int:
        guard_mount(path, dir_fd)
        return original_open(path, flags, dir_fd=dir_fd)

    def guard_stat(
        path: str | Path, *, dir_fd: int | None = None, follow_symlinks: bool = True
    ) -> os.stat_result:
        guard_mount(path, dir_fd)
        return original_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(os, "open", guard_open)
    monkeypatch.setattr(os, "stat", guard_stat)
    for target in (first, second):
        alias.unlink(missing_ok=True)
        alias.symlink_to(target, target_is_directory=True)
        with EventSource._sibling_stat(alias, mountpoint=mountpoint) as stat:
            for name in ("source", "link"):
                assert stat(alias / name) == EventSource.stat(
                    alias / name, mountpoint=mountpoint
                )
                assert stat(alias / name).st_size == len(target.name)
            assert stat(second / "source").st_size == 6
            if target == first:
                assert stat(alias / "nested" / "source").st_size == len(
                    b"nested source"
                )
                assert stat(alias / mountpoint.name).st_size == 0
            else:
                with pytest.raises(FileNotFoundError):
                    stat(alias / mountpoint.name)

    alias.unlink()
    alias.symlink_to(mountpoint, target_is_directory=True)
    with (
        EventSource._sibling_stat(alias, mountpoint=mountpoint) as stat,
        pytest.raises(FileNotFoundError),
    ):
        stat(alias / "source")


@pytest.mark.parametrize("root_name", ["mount", "storage"])
def test_sibling_status_rejects_lexical_protected_ancestor(
    tmp_path: Path, root_name: str
) -> None:
    root = tmp_path.resolve()
    mount = root / "mount"
    storage = root / "storage"
    mount.mkdir()
    storage.mkdir()
    protected = root / root_name
    nested = protected / "subdir"
    nested.mkdir()
    hidden = "events.out.tfevents.hidden"
    for directory in (protected, nested):
        (directory / hidden).touch()
        with (
            EventSource._sibling_stat(
                directory, mountpoint=mount, storage=storage
            ) as stat,
            pytest.raises(FileNotFoundError),
        ):
            stat(directory / hidden)
    run = root / f"{root_name}-run"
    run.mkdir()
    source = run / "events.out.tfevents.source"
    source.write_bytes(b"safe")

    with EventSource._sibling_stat(run, mountpoint=mount, storage=storage) as stat:
        assert stat(source) == source.stat()
        with pytest.raises(FileNotFoundError):
            stat(protected / hidden)
    with (
        EventSource._sibling_stat(root, mountpoint=mount, storage=storage) as stat,
        pytest.raises(FileNotFoundError),
    ):
        stat(protected)

    lexical_run = protected / ".." / run.name
    context = EventSource._sibling_stat(lexical_run, mountpoint=mount, storage=storage)
    with context as stat, pytest.raises(FileNotFoundError):
        stat(lexical_run / source.name)


def test_sibling_status_falls_back_when_parent_cannot_be_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    run = tmp_path / "run"
    run.symlink_to(target, target_is_directory=True)
    source = run / "source"
    source.touch()
    mountpoint = tmp_path / "mount"
    original_open = os.open
    faults = 0

    def deny_once(path: str | Path, flags: int, *, dir_fd: int | None = None) -> int:
        nonlocal faults
        if path == run.name and not faults:
            faults += 1
            raise PermissionError(EACCES, "permission denied", path)
        return original_open(path, flags, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", deny_once)
    with EventSource._sibling_stat(run, mountpoint=mountpoint) as stat:
        assert faults == 1
        assert stat(source).st_size == 0
        with pytest.raises(FileNotFoundError):
            stat(run / "missing")


@pytest.mark.parametrize("relative", [False, True])
def test_sibling_status_pins_parent_until_next_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: bool
) -> None:
    if relative:
        monkeypatch.chdir(tmp_path)
    root = Path(".") if relative else tmp_path
    run = root / "run"
    previous = root / "previous"
    mountpoint = tmp_path / "mount"
    source = run / "source"
    run.mkdir()
    mountpoint.mkdir()
    source.write_bytes(b"old")
    (mountpoint / source.name).write_bytes(b"protected")

    with EventSource._sibling_stat(run, mountpoint=mountpoint) as stat:
        run.rename(previous)
        run.symlink_to(mountpoint, target_is_directory=True)
        assert stat(source).st_size == 3

    with (
        EventSource._sibling_stat(run, mountpoint=mountpoint) as stat,
        pytest.raises(FileNotFoundError),
    ):
        stat(source)

    run.unlink()
    run.mkdir()
    source.write_bytes(b"replacement")
    with EventSource._sibling_stat(run, mountpoint=mountpoint) as stat:
        assert stat(source).st_size == len(b"replacement")


@pytest.mark.parametrize("mounted", [False, True])
def test_directory_status_rejects_a_retarget_into_private_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mounted: bool
) -> None:
    storage = tmp_path / "indexes"
    storage.mkdir()
    (storage / "run").mkdir()
    source = tmp_path / "source"
    source.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(source)
    mountpoint = tmp_path / "mount" if mounted else None
    assert (
        EventSource.directory_stat(alias, mountpoint=mountpoint, storage=storage)
        == source.stat()
    )
    original_open = os.open

    def retarget_on_open(
        path: str | Path, flags: int, *, dir_fd: int | None = None
    ) -> int:
        assert path != storage.name
        if path == alias.name:
            alias.unlink()
            alias.symlink_to(storage / "run")
        return original_open(path, flags, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", retarget_on_open)
    with pytest.raises(FileNotFoundError):
        EventSource.directory_stat(alias, mountpoint=mountpoint, storage=storage)
