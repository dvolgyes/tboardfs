"""Verify the native no-symlink path checks and their descriptor ownership."""

import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType


def test_native_directory_fd_survives_parent_replacement(
    native_paths: ModuleType, tmp_path: Path
) -> None:
    live = tmp_path.resolve() / "live"
    live.mkdir()
    (live / "event").write_bytes(b"original")
    replacement = live.with_name("replacement")
    replacement.mkdir()
    (replacement / "event").write_bytes(b"replacement")

    fd = native_paths.open_directory(os.fsencode(live))
    assert fd >= 0
    try:
        other_fd = native_paths.open_directory(os.fsencode(live))
        assert other_fd >= 0
        assert other_fd != fd
        os.close(other_fd)
        assert not os.get_inheritable(fd)

        live.rename(live.with_name("original"))
        live.symlink_to(replacement, target_is_directory=True)
        assert (live / "event").read_bytes() == b"replacement"
        event_fd = os.open("event", os.O_RDONLY, dir_fd=fd)
        with os.fdopen(event_fd, "rb") as event:
            assert event.read() == b"original"
        assert native_paths.open_directory(os.fsencode(live)) == -1
    finally:
        os.close(fd)


def test_native_path_checks_refuse_symlink_components(
    native_paths: ModuleType, tmp_path: Path
) -> None:
    root = tmp_path.resolve()
    directory = root / "directory"
    directory.mkdir()
    (directory / "nested").mkdir()
    event = directory / "event"
    event.touch()
    dir_link = root / "dir-link"
    dir_link.symlink_to(directory, target_is_directory=True)
    file_link = root / "file-link"
    file_link.symlink_to(event)

    for path in (dir_link, dir_link / "nested", dir_link / "event", file_link):
        name = os.fsencode(path)
        assert native_paths.plain_path(name) is False, path
        assert native_paths.open_directory(name) == -1, path


def test_native_plain_path_accepts_files_without_retaining_a_descriptor(
    native_paths: ModuleType, tmp_path: Path
) -> None:
    directory = tmp_path.resolve()
    event = directory / "event"
    event.touch()
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
import sys

from tboardfs._sibling_stat import open_directory, plain_path

directory, event = map(os.fsencode, sys.argv[1:])
fd = os.open(directory, os.O_PATH)
os.close(fd)
assert plain_path(directory) is True
assert plain_path(event) is True
assert open_directory(event) == -1
after = os.open(directory, os.O_PATH)
os.close(after)
assert after == fd
""",
            str(directory),
            str(event),
        ],
        check=True,
    )


def test_native_paths_refuse_embedded_nul(
    native_paths: ModuleType, tmp_path: Path
) -> None:
    directory_name = os.fsencode(tmp_path.resolve())
    malformed = directory_name + b"\0suffix"
    assert native_paths.open_directory(malformed) == -1
    assert native_paths.plain_path(malformed) is False
