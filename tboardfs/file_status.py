"""Identify and compare event-file generations using portable stat evidence."""

import os
from typing import NamedTuple, Self


class FileIdentity(NamedTuple):
    """Filesystem identity, including creation time where available."""

    device: int
    inode: int
    birth_ns: int | None

    @classmethod
    def of(cls, status: os.stat_result) -> Self:
        """Read creation time without interpreting Linux ctime as birth time."""
        birth = getattr(status, "st_birthtime_ns", None)
        if (
            birth is None
            and (seconds := getattr(status, "st_birthtime", None)) is not None
        ):
            birth = round(seconds * 1_000_000_000)
        return cls(status.st_dev, status.st_ino, birth)

    @classmethod
    def fingerprint(cls, status: os.stat_result) -> tuple[int | None, ...]:
        """Return status used to skip unchanged sources without reopening them."""
        return (*cls.of(status), status.st_size, status.st_mtime_ns, status.st_ctime_ns)
