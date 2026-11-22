"""Verify the native no-symlink path checks and their descriptor ownership."""

import ctypes
import errno
import os
import subprocess
import sys
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType

import pytest

from tboardfs._directory import _AT_FDCWD, _RESOLVE_NO_SYMLINKS, _OpenHow


@pytest.fixture(scope="module")
def native_paths() -> ModuleType:
    """Require the optional native build and a working kernel syscall."""
    name = "tboardfs._sibling_stat"
    if find_spec(name) is None:
        pytest.skip("native extension unavailable for this build")
    native = import_module(name)
    has_open_directory = hasattr(native, "open_directory")
    assert has_open_directory == hasattr(native, "plain_path")
    if not has_open_directory:
        pytest.skip("native openat2 bindings not compiled")

    fd = native.open_directory(b"/")
    if fd >= 0:
        os.close(fd)
        return native

    libc_open = getattr(ctypes.CDLL(None, use_errno=True), "openat2", None)
    if libc_open is not None:
        libc_open.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.POINTER(_OpenHow),
            ctypes.c_size_t,
        )
        libc_open.restype = ctypes.c_int
        how = _OpenHow(
            os.O_PATH | os.O_DIRECTORY | os.O_CLOEXEC, 0, _RESOLVE_NO_SYMLINKS
        )
        libc_fd = libc_open(_AT_FDCWD, b"/", ctypes.byref(how), ctypes.sizeof(how))
        if libc_fd >= 0:
            os.close(libc_fd)
        else:
            error = ctypes.get_errno()
            if error in (errno.ENOSYS, errno.EPERM):
                pytest.skip(f"openat2 unavailable: {os.strerror(error)}")
    pytest.fail("native open_directory rejected the root directory")


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
