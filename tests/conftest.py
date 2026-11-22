"""Share capability fixtures across the test suite."""

import ctypes
import errno
import os
from importlib import import_module
from importlib.util import find_spec
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
