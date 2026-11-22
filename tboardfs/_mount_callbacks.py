"""Route native FUSE 2 stat callbacks to the handler owned by their mount."""

import ctypes
import platform
import sys
import sysconfig
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import version
from importlib.util import find_spec
from types import ModuleType
from typing import Self

import mfusepy

type _GetattrCallback = Callable[[bytes | None, mfusepy.c_stat_p], int]
type _StatHandler = Callable[
    [bytes | None, mfusepy.c_stat_p, mfusepy.fuse_fi_p | None], int
]
type _StatReply = Callable[
    [bytes | None, mfusepy.fuse_fi_p | None], mfusepy.c_stat | int
]

_EXTENSION = "tboardfs._stat_extern"


def _supported() -> bool:
    """Limit native entry to the audited platform and mfusepy ABI."""
    return (
        sys.platform == "linux"
        and sys.implementation.name == "cpython"
        and platform.machine() == "x86_64"
        and ctypes.sizeof(ctypes.c_void_p) == ctypes.sizeof(ctypes.c_long) == 8
        and not sysconfig.get_config_var("Py_GIL_DISABLED")
        and (mfusepy.fuse_version_major, mfusepy.fuse_version_minor) == (2, 9)
        and version("mfusepy") == "3.1.1"
    )


def _load_extension() -> ModuleType | None:
    """Distinguish an optional missing build from a broken installed extension."""
    return (
        None
        if find_spec(_EXTENSION) is None
        else import_module("tboardfs._stat_dispatch")
    )


@dataclass(frozen=True, slots=True)
class StatCallbacks:
    """Keep the per-mount CFFI handle alive until libfuse returns."""

    getattr: _GetattrCallback
    fgetattr: _StatHandler
    user_data: ctypes.c_void_p
    _handle: object

    @classmethod
    def try_create(cls, handler: _StatReply) -> Self | None:
        """Return callbacks only when supported and the optional extension was built."""
        if not _supported() or (extension := _load_extension()) is None:
            return None
        ffi = extension.ffi
        handle = ffi.new_handle(handler)
        private_data = ctypes.c_void_p(int(ffi.cast("uintptr_t", handle)))
        return cls(extension.GETATTR, extension.FGETATTR, private_data, handle)
