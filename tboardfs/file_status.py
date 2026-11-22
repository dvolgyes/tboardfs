"""Identify and compare event-file generations using portable stat evidence."""

import os
from typing import NamedTuple, Self

_HAS_BIRTH_TIME = hasattr(os.stat_result, "st_birthtime_ns") or hasattr(
    os.stat_result, "st_birthtime"
)


def _birth_ns(status: os.stat_result) -> int | None:
    birth: int | None = getattr(status, "st_birthtime_ns", None)
    if birth is not None:
        return birth
    seconds: float | None = getattr(status, "st_birthtime", None)
    return None if seconds is None else round(seconds * 1_000_000_000)


class FileIdentity(NamedTuple):
    """Filesystem identity, including creation time where available."""

    device: int
    inode: int
    birth_ns: int | None

    @classmethod
    def of(cls, status: os.stat_result) -> Self:
        """Read creation time without interpreting Linux ctime as birth time."""
        birth = _birth_ns(status) if _HAS_BIRTH_TIME else None
        return cls(status.st_dev, status.st_ino, birth)

    @staticmethod
    def fingerprint(status: os.stat_result) -> tuple[int | None, ...]:
        """Return status used to skip unchanged sources without reopening them."""
        return (
            status.st_dev,
            status.st_ino,
            _birth_ns(status) if _HAS_BIRTH_TIME else None,
            status.st_size,
            status.st_mtime_ns,
            status.st_ctime_ns,
        )
