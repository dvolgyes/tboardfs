"""Encode resolved TensorBoardFS attributes for the mfusepy ABI."""

from collections.abc import Callable
from functools import lru_cache

import mfusepy

from tboardfs.file_attributes import _FileAttributes
from tboardfs.mounted import MountedTensorBoardFS


class MountBinding(mfusepy.FUSE):  # type: ignore[misc]
    """Reuse native stat encoding without caching path or handle lookups."""

    operations: MountedTensorBoardFS
    _native_stat: Callable[[_FileAttributes, bool], mfusepy.c_stat]

    def __init__(
        self,
        operations: MountedTensorBoardFS,
        mountpoint: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        self._native_stat = lru_cache(maxsize=256)(self._encode_stat)
        super().__init__(operations, mountpoint, *args, **kwargs)

    @staticmethod
    def _encode_stat(attrs: _FileAttributes, use_ns: bool) -> mfusepy.c_stat:
        st = mfusepy.c_stat()
        mfusepy.set_st_attrs(st, attrs.as_dict(), use_ns=use_ns)
        return st

    def fgetattr(
        self, path: bytes | None, buf: mfusepy.c_stat_p, fip: mfusepy.fuse_fi_p | None
    ) -> int:
        """Resolve and encode fresh attributes, leaving a zeroed buffer on error."""
        try:
            decoded_path = (
                None if path is None else path.decode(self.encoding, self.errors)
            )
            handles = (fip.contents.fh,) if fip else ()
            attrs = self.operations._attributes(decoded_path, handles)
            st = self._native_stat(attrs, self.use_ns)
        except BaseException:
            buf[0] = mfusepy.c_stat()
            raise
        buf[0] = st
        return 0
