"""Resolve guarded source directories without following symlinks into blocked roots."""

import ctypes
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

_AT_FDCWD = -100
_RESOLVE_NO_SYMLINKS = 0x04
_GUARD_PLAN_MAX_PREFIX_CHARS = 4096


def _guard_plan(
    blocked: frozenset[str], path: Path
) -> tuple[tuple[str, ...], bool] | None:
    """Plan bounded lexical probes before the first blocked prefix."""
    parts = path.parts
    if (
        path.anchor != "/"
        or len(parts) * len(str(path)) > _GUARD_PLAN_MAX_PREFIX_CHARS
        or ".." in parts
    ):
        return None
    prefixes: list[str] = []
    prefix = ""
    for part in parts[1:]:
        prefix += "/" + part
        if prefix in blocked:
            return tuple(prefixes), False
        prefixes.append(prefix)
    return tuple(prefixes), True


class _OpenHow(ctypes.Structure):
    """Linux openat2 arguments from <linux/openat2.h>."""

    _fields_: ClassVar[list[tuple[str, type]]] = [
        ("flags", ctypes.c_uint64),
        ("mode", ctypes.c_uint64),
        ("resolve", ctypes.c_uint64),
    ]


def _bind_open() -> Callable[[bytes], int] | None:
    if sys.platform != "linux" or not hasattr(os, "O_PATH"):
        return None
    call = getattr(ctypes.CDLL(None), "openat2", None)
    if call is None:
        return None
    call.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(_OpenHow),
        ctypes.c_size_t,
    )
    call.restype = ctypes.c_int
    how = _OpenHow(os.O_PATH | os.O_DIRECTORY | os.O_CLOEXEC, 0, _RESOLVE_NO_SYMLINKS)
    pointer, size = ctypes.pointer(how), ctypes.sizeof(how)
    return lambda path: call(_AT_FDCWD, path, pointer, size)


_OPEN_DIRECTORY = _bind_open()


def open_plain_directory(path: Path, blocked: tuple[str, ...]) -> int | None:
    """Return an owned descriptor, or defer to the guarded component walk."""
    text = os.fspath(path)
    if (
        _OPEN_DIRECTORY is None
        or path.anchor != "/"
        or ".." in path.parts
        or "\0" in text
        or any(
            text == root or text.startswith(root.rstrip("/") + "/") for root in blocked
        )
    ):
        return None
    fd = _OPEN_DIRECTORY(os.fsencode(text))
    return fd if fd >= 0 else None
