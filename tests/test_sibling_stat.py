"""Check that the accelerated status comparison cannot approve a stale sibling."""

import os
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

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
