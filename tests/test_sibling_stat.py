"""Check that the accelerated status comparison cannot approve a stale sibling."""

import os
from collections.abc import Callable
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

from tboardfs.file_status import _HAS_BIRTH_TIME, FileIdentity

pytestmark = pytest.mark.skipif(
    _HAS_BIRTH_TIME or find_spec("tboardfs._sibling_stat") is None,
    reason="native comparison unavailable for this Python build",
)


def test_native_comparison_checks_every_fingerprint_field(tmp_path: Path) -> None:
    matches = import_module("tboardfs._sibling_stat").matches
    names = (b"first", b"event-\xff")
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for name in names:
            (tmp_path / os.fsdecode(name)).write_bytes(b"event")
        os.utime(names[-1], ns=(0, -123_456_789), dir_fd=fd)
        rows = tuple(
            (name, FileIdentity.fingerprint(os.lstat(name, dir_fd=fd)))
            for name in names
        )
        assert matches(fd, rows)
        fingerprint = rows[-1][1]
        for index, value in enumerate(fingerprint):
            replacement = 0 if value is None else value + 1
            changed = (*fingerprint[:index], replacement, *fingerprint[index + 1 :])
            assert not matches(fd, (rows[0], (names[-1], changed)))

        os.rename(names[-1], b"target", src_dir_fd=fd, dst_dir_fd=fd)
        assert not matches(fd, rows)
        os.symlink(b"target", names[-1], dir_fd=fd)
        assert not matches(fd, rows)
    finally:
        os.close(fd)


def test_native_comparison_requires_an_immediate_regular_file(tmp_path: Path) -> None:
    matches = import_module("tboardfs._sibling_stat").matches
    directory = tmp_path / "run"
    directory.mkdir()
    event = tmp_path / "event"
    event.touch()
    (directory / "neighbor").symlink_to(event)
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        status = FileIdentity.fingerprint(event.stat())
        for name in (b"../event", os.fsencode(event), b"neighbor", b"neighbor\0suffix"):
            assert not matches(fd, ((name, status),))
        assert not matches(fd, ((b".", FileIdentity.fingerprint(os.fstat(fd))),))
        (directory / "subdir").mkdir()
        status = FileIdentity.fingerprint((directory / "subdir").stat())
        assert not matches(fd, ((b"subdir", status),))
    finally:
        os.close(fd)


_PrepareSiblings = Callable[[object, object], Callable[[], bool] | None]


@pytest.fixture
def prepare_siblings(native_paths: ModuleType) -> _PrepareSiblings:
    prepare = getattr(native_paths, "prepare_siblings", None)
    assert callable(prepare)
    return cast(_PrepareSiblings, prepare)


def _row(path: Path) -> tuple[bytes, tuple[int | None, ...]]:
    return os.fsencode(path.name), FileIdentity.fingerprint(path.lstat())


def _assert_no_open_parent(parent: Path) -> None:
    descriptors = Path("/proc/self/fd")
    if not descriptors.is_dir():
        return
    for fd in descriptors.iterdir():
        try:
            target = fd.readlink()
        except FileNotFoundError:
            continue
        assert target != parent


def test_prepared_siblings_owns_inputs_and_checks_the_final_row(
    prepare_siblings: _PrepareSiblings, tmp_path: Path
) -> None:
    parent = tmp_path.resolve() / os.fsdecode(b"run-\xff")
    parent.mkdir()
    siblings = [parent / f"event-{i}" for i in range(8)]
    final = parent / os.fsdecode(b"event-\xfe")
    siblings.append(final)
    for path in siblings:
        path.write_bytes(b"old")
    os.utime(final, ns=(0, -123_456_789))

    check = prepare_siblings(
        os.fsencode(parent), tuple(_row(path) for path in siblings)
    )
    assert check is not None
    with siblings[0].open("rb") as reader:
        for _ in range(3):
            assert check() is True
            _assert_no_open_parent(parent)
            assert os.pread(reader.fileno(), 3, 0) == b"old"

        final.write_bytes(b"new")
        os.utime(final, ns=(0, -123_456_788))
        for _ in range(3):
            assert check() is False
            _assert_no_open_parent(parent)
            assert os.pread(reader.fileno(), 3, 0) == b"old"


@pytest.mark.parametrize("change", ["replacement", "rename", "symlink"])
def test_prepared_siblings_detects_leaf_changes(
    prepare_siblings: _PrepareSiblings, tmp_path: Path, change: str
) -> None:
    parent = tmp_path.resolve()
    event = parent / "event"
    event.write_bytes(b"event")
    check = prepare_siblings(os.fsencode(parent), (_row(event),))
    assert check is not None
    assert check() is True

    target = parent / "target"
    if change == "replacement":
        target.write_bytes(b"event")
        target.replace(event)
    else:
        event.rename(target)
        if change == "symlink":
            event.symlink_to(target)
            check_target = prepare_siblings(
                os.fsencode(parent), ((b"event", _row(target)[1]),)
            )
            assert check_target is not None
            assert check_target() is False
    assert check() is False


def test_prepared_siblings_reopens_parent_and_rejects_parent_symlink(
    prepare_siblings: _PrepareSiblings, tmp_path: Path
) -> None:
    parent = tmp_path.resolve() / "run"
    parent.mkdir()
    event = parent / "event"
    event.touch()
    check = prepare_siblings(os.fsencode(parent), (_row(event),))
    assert check is not None
    assert check() is True

    moved = parent.rename(parent.with_name("moved"))
    parent.mkdir()
    assert check() is False
    _assert_no_open_parent(parent)
    parent.rmdir()
    parent.symlink_to(moved, target_is_directory=True)
    assert check() is False
    parent.unlink()
    moved.rename(parent)
    assert check() is True
    _assert_no_open_parent(parent)


def test_prepared_siblings_rejects_mutable_and_malformed_rows(
    prepare_siblings: _PrepareSiblings, tmp_path: Path
) -> None:
    parent = tmp_path.resolve()
    event = parent / "event"
    event.touch()
    name, fingerprint = _row(event)
    parent_name = os.fsencode(parent)
    row = (name, fingerprint)

    class IntegerSubclass(int):
        pass

    dev = fingerprint[0]
    assert dev is not None
    for bad_parent in (bytearray(parent_name), parent_name + b"\0suffix"):
        assert prepare_siblings(bad_parent, (row,)) is None
    for bad_rows in (
        [row],
        ([name, fingerprint],),
        ((bytearray(name), fingerprint),),
        ((name + b"\0suffix", fingerprint),),
        ((b"../" + name, fingerprint),),
        ((name, list(fingerprint)),),
        ((name, fingerprint[:-1]),),
        ((name, (True, *fingerprint[1:])),),
        ((name, (IntegerSubclass(dev), *fingerprint[1:])),),
        ((name, (1 << 100, *fingerprint[1:])),),
    ):
        assert prepare_siblings(parent_name, bad_rows) is None
