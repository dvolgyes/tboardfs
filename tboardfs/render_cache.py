"""Least-recently-used cache of rendered virtual file bytes."""

from collections import OrderedDict
from collections.abc import Hashable
from threading import Lock

# Approximate memory of one entry besides its bytes: key, dict slot, header.
ENTRY_OVERHEAD_BYTES = 200


class RenderCache:
    """Keep rendered file bytes within an approximate byte budget.

    When the total goes over the limit, the least recently used entries are
    dropped until at most half the limit is left. So each cleanup frees much
    memory, and recently used entries stay. An entry larger than half the
    limit is not kept, because its own cleanup would drop it.

    The cache has its own lock and calls nothing while it holds it, so it
    can be used with or without the filesystem lock.

    Attributes:
        limit_bytes: byte budget; zero turns the cache off
        size_bytes: approximate bytes of all entries
    """

    limit_bytes: int
    size_bytes: int
    _entries: OrderedDict[Hashable, bytes]
    _lock: Lock

    def __init__(self, limit_bytes: int) -> None:
        """Create an empty cache.

        Args:
            limit_bytes: byte budget, zero or more

        Raises:
            ValueError: when ``limit_bytes`` is negative
        """
        if limit_bytes < 0:
            raise ValueError(f"render cache size must be >= 0, got {limit_bytes}")
        self.limit_bytes = limit_bytes
        self.size_bytes = 0
        self._entries = OrderedDict()
        self._lock = Lock()

    def get(self, key: Hashable) -> bytes | None:
        """Return the bytes for ``key`` and mark them recently used."""
        with self._lock:
            data = self._entries.get(key)
            if data is not None:
                self._entries.move_to_end(key)
            return data

    def put(self, key: Hashable, data: bytes) -> None:
        """Keep ``data`` for ``key``; drop old entries when over the limit."""
        size = len(data) + ENTRY_OVERHEAD_BYTES
        if size > self.limit_bytes // 2:
            return
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self.size_bytes -= len(old) + ENTRY_OVERHEAD_BYTES
            self._entries[key] = data
            self.size_bytes += size
            if self.size_bytes <= self.limit_bytes:
                return
            while self.size_bytes > self.limit_bytes // 2:
                _, dropped = self._entries.popitem(last=False)
                self.size_bytes -= len(dropped) + ENTRY_OVERHEAD_BYTES
