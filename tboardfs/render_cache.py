"""Least-recently-used cache of rendered virtual file bytes."""

from collections import OrderedDict
from threading import Lock

# Approximate memory of one entry besides its bytes: key, dict slot, header.
ENTRY_OVERHEAD_BYTES = 200


class _RenderScope:
    """Identity of one generation and the paths cached for it."""

    __slots__: tuple[str, ...] = ("_expired", "_paths")

    _expired: bool
    _paths: set[tuple[str, ...]]

    def __init__(self) -> None:
        self._expired = False
        self._paths = set()


type RenderKey = tuple[_RenderScope, tuple[str, ...]]


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
    _entries: OrderedDict[RenderKey, bytes]
    _scopes: dict[int | None, _RenderScope]
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
        self._scopes = {}
        self._lock = Lock()

    def scope(self, version: int | None) -> _RenderScope:
        """Return the scope for one run version, or for the active name map."""
        with self._lock:
            scope = self._scopes.get(version)
            if scope is None:
                scope = self._scopes[version] = _RenderScope()
            return scope

    def get(self, key: RenderKey) -> bytes | None:
        """Return the bytes for ``key`` and mark them recently used."""
        with self._lock:
            data = self._entries.get(key)
            if data is not None:
                self._entries.move_to_end(key)
            return data

    def put(self, key: RenderKey, data: bytes) -> None:
        """Keep ``data`` for ``key``; drop old entries when over the limit."""
        size = len(data) + ENTRY_OVERHEAD_BYTES
        if size > self.limit_bytes // 2:
            return
        with self._lock:
            scope, path = key
            if scope._expired:
                return
            old = self._entries.pop(key, None)
            if old is not None:
                self.size_bytes -= len(old) + ENTRY_OVERHEAD_BYTES
            self._entries[key] = data
            scope._paths.add(path)
            self.size_bytes += size
            if self.size_bytes <= self.limit_bytes:
                return
            while self.size_bytes > self.limit_bytes // 2:
                (evicted_scope, evicted_path), dropped = self._entries.popitem(
                    last=False
                )
                evicted_scope._paths.remove(evicted_path)
                self.size_bytes -= len(dropped) + ENTRY_OVERHEAD_BYTES

    def expire(self, version: int | None) -> None:
        """Drop this generation and refuse further inserts for it."""
        with self._lock:
            scope = self._scopes.pop(version, None)
            if scope is None:
                return
            scope._expired = True
            for path in scope._paths:
                data = self._entries.pop((scope, path))
                self.size_bytes -= len(data) + ENTRY_OVERHEAD_BYTES
            scope._paths.clear()
