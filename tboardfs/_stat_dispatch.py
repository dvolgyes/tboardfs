"""Register native FUSE 2 stat entrypoints once through the import lock."""

import ctypes
import errno
from importlib import import_module
from typing import cast

import mfusepy

from ._mount_callbacks import _GetattrCallback, _StatHandler

_native = import_module("tboardfs._stat_extern")
ffi = _native.ffi
lib = _native.lib


def _dispatch(path: object, stat: object, fi: object) -> int:
    """Recover the runner's pinned handler from libfuse's mount private data."""
    if stat == ffi.NULL:
        return -errno.EFAULT
    context = mfusepy._libfuse.fuse_get_context()
    if not context or not (private_data := context.contents.private_data):
        return -errno.EFAULT
    handler: _StatHandler = ffi.from_handle(ffi.cast("void *", private_data))
    return handler(
        None if path == ffi.NULL else ffi.string(path),
        ctypes.cast(int(ffi.cast("uintptr_t", stat)), mfusepy.c_stat_p),
        None
        if fi == ffi.NULL
        else ctypes.cast(int(ffi.cast("uintptr_t", fi)), mfusepy.fuse_fi_p),
    )


def _slot_callbacks() -> tuple[_GetattrCallback, _StatHandler]:
    """Cast the compiled entrypoints to mfusepy's actual slot prototypes."""
    slot = {field[0]: field[1] for field in mfusepy.fuse_operations._fields_}
    return (
        cast(
            _GetattrCallback,
            ctypes.cast(
                int(ffi.cast("uintptr_t", ffi.addressof(lib, "stat_getattr"))),
                slot["getattr"],
            ),
        ),
        cast(
            _StatHandler,
            ctypes.cast(
                int(ffi.cast("uintptr_t", ffi.addressof(lib, "stat_fgetattr"))),
                slot["fgetattr"],
            ),
        ),
    )


ffi.def_extern(name="stat_getattr", error=-errno.EFAULT)(
    lambda path, stat: _dispatch(path, stat, ffi.NULL)
)
ffi.def_extern(name="stat_fgetattr", error=-errno.EFAULT)(_dispatch)
GETATTR, FGETATTR = _slot_callbacks()
