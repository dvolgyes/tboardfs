"""Resolve guarded source directories without following symlinks into blocked roots."""

import ctypes
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import ClassVar, Self

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

    @classmethod
    def bind(cls, *, directory: bool) -> Callable[[bytes], int] | None:
        """Bind either a fresh directory fd lookup or an ordinary path probe."""
        if sys.platform != "linux" or not hasattr(os, "O_PATH"):
            return None
        if find_spec("tboardfs._sibling_stat") is not None:
            native = import_module("tboardfs._sibling_stat")
            probe: Callable[[bytes], int] | None = getattr(
                native, "open_directory" if directory else "plain_path", None
            )
            if probe is not None:
                return probe
        call = getattr(ctypes.CDLL(None), "openat2", None)
        if call is None:
            return None
        call.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.POINTER(cls),
            ctypes.c_size_t,
        )
        call.restype = ctypes.c_int
        flags = os.O_PATH | os.O_CLOEXEC
        if directory:
            flags |= os.O_DIRECTORY
        how = cls(flags, 0, _RESOLVE_NO_SYMLINKS)
        pointer, size = ctypes.pointer(how), ctypes.sizeof(how)

        if directory:
            return lambda path: call(_AT_FDCWD, path, pointer, size)
        return lambda path: cls.close_probe(call(_AT_FDCWD, path, pointer, size))

    @staticmethod
    def close_probe(fd: int) -> bool:
        """Release a successful libc path probe before reporting its result."""
        if fd < 0:
            return False
        os.close(fd)
        return True


_OPEN_DIRECTORY = _OpenHow.bind(directory=True)
_OPEN_PATH = _OpenHow.bind(directory=False)


def _plain_path(path: Path) -> bool:
    """Confirm a lexically permitted path exists without traversing a symlink."""
    if _OPEN_PATH is None:
        return False
    try:
        name = os.fsencode(path)
    except UnicodeEncodeError:
        return False
    if b"\0" in name:
        return False
    return bool(_OPEN_PATH(name))


@dataclass(frozen=True, slots=True)
class _PlainDirectory:
    """Lexically eligible directory name; resolving it still requires a fresh open."""

    name: bytes

    @classmethod
    def prepare(cls, path: Path, blocked: tuple[str, ...]) -> Self | None:
        """Encode a canonical absolute name outside the protected roots without doing I/O."""
        text = os.fspath(path)
        if (
            path.anchor != "/"
            or ".." in path.parts
            or "\0" in text
            or any(
                text == root or text.startswith(root.rstrip("/") + "/")
                for root in blocked
            )
        ):
            return None
        try:
            return cls(os.fsencode(text))
        except UnicodeEncodeError:
            return None

    def open(self) -> int | None:
        """Return a new no-follow directory descriptor or defer to guarded lookup."""
        if _OPEN_DIRECTORY is None:
            return None
        fd = _OPEN_DIRECTORY(self.name)
        return fd if fd >= 0 else None
