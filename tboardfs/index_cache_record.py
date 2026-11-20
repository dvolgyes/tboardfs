"""Immutable metadata for a persisted event-file index."""

import re
import time
from dataclasses import dataclass

from .constants import SOURCE_DIGEST_BYTES

_EXPIRY = 30 * 24 * 60 * 60
_PAYLOAD = re.compile(r"index-[0-9a-f]{32}\.idx")


@dataclass(frozen=True, slots=True)
class CacheRecord:
    """Metadata identifying one immutable payload, not a path to the source."""

    digest: bytes
    name: str
    checksum: bytes
    length: int
    written: float

    @staticmethod
    def expiry_cutoff(now: float) -> float:
        """Return the latest write timestamp that has expired."""
        return now - _EXPIRY

    def valid(self, limit: int) -> bool:
        """Check persisted metadata before using it as a path, size, or digest."""
        now = time.time()
        return (
            isinstance(self.digest, bytes)
            and isinstance(self.checksum, bytes)
            and (len(self.digest), len(self.checksum)) == (SOURCE_DIGEST_BYTES, 32)
            and isinstance(self.name, str)
            and bool(_PAYLOAD.fullmatch(self.name))
            and isinstance(self.length, int)
            and 0 <= self.length <= limit
            and isinstance(self.written, (int, float))
            and self.expiry_cutoff(now) < self.written <= now
        )
