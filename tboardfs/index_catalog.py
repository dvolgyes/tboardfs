"""Store optional whole-file indexes in a private, metadata-only SQLite catalog."""

import fcntl
import hashlib
import os
import sqlite3
import time
import uuid
import weakref
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from threading import Lock

from loguru import logger

from .constants import MAX_INDEX_BYTES
from .index_cache_files import IndexCacheFiles
from .index_cache_record import CacheRecord

_METADATA_LIMIT = 64 * 1024 * 1024
_CATALOG = "catalog.sqlite3"
_LOCK_TIMEOUT = 0.3
_LOCK_POLL = 0.01
_ERRORS = (OSError, sqlite3.Error, ValueError, OverflowError)
_SCHEMA = """CREATE TABLE entries (
    path BLOB NOT NULL, size INTEGER NOT NULL, policy TEXT NOT NULL,
    digest BLOB NOT NULL, name TEXT NOT NULL, checksum BLOB NOT NULL,
    length INTEGER NOT NULL CHECK (length >= 0), written REAL NOT NULL,
    PRIMARY KEY (path, size, policy)
) STRICT, WITHOUT ROWID"""


class _CorruptCatalog(sqlite3.DatabaseError):
    """A damaged image at the catalog's reserved, private filename."""


class _Metadata:
    """Validate and update an in-memory SQLite image."""

    @staticmethod
    def trim(connection: sqlite3.Connection, limit: int, now: float) -> None:
        """Evict expired records, then keep the newest records within the budget."""
        connection.execute(
            "DELETE FROM entries WHERE written <= ?", (CacheRecord.expiry_cutoff(now),)
        )
        connection.execute(
            """DELETE FROM entries WHERE (path, size, policy) IN (
                SELECT path, size, policy FROM (
                    SELECT path, size, policy,
                        sum(length) OVER (ORDER BY written DESC, name DESC) AS used
                    FROM entries
                ) WHERE used > ?
            )""",
            (limit,),
        )

    @staticmethod
    def configure(connection: sqlite3.Connection) -> None:
        """Restrict SQLite to in-memory operation with untrusted schema rules."""
        connection.execute("PRAGMA trusted_schema = OFF")
        connection.execute("PRAGMA temp_store = MEMORY")
        defensive = getattr(sqlite3, "SQLITE_DBCONFIG_DEFENSIVE", None)
        if defensive is not None:
            connection.setconfig(defensive, True)
        if not hasattr(connection, "serialize") or not hasattr(
            connection, "deserialize"
        ):
            raise sqlite3.NotSupportedError("SQLite image APIs are unavailable")

    @staticmethod
    def deserialize(connection: sqlite3.Connection, body: bytes) -> None:
        """Load the expected rollback-mode image and reject foreign schemas."""
        if body[:16] != b"SQLite format 3\x00":
            raise _CorruptCatalog("unsupported catalog image")
        if body[18:20] != b"\x01\x01":
            raise sqlite3.DatabaseError("catalog must use rollback mode")
        try:
            connection.deserialize(body)
            schema = connection.execute(
                "SELECT type, name, sql FROM sqlite_schema"
            ).fetchall()
        except sqlite3.DatabaseError as error:
            if _Metadata.is_damage(error):
                raise _CorruptCatalog("damaged index catalog") from error
            raise
        if schema != [("table", "entries", _SCHEMA)]:
            raise sqlite3.DatabaseError("unexpected index catalog schema")

    @staticmethod
    def is_damage(error: sqlite3.DatabaseError) -> bool:
        """Recognize SQLite primary and extended database corruption codes."""
        return getattr(error, "sqlite_errorcode", 0) & 255 in (
            sqlite3.SQLITE_CORRUPT,
            sqlite3.SQLITE_NOTADB,
        )

    @staticmethod
    def read(fd: int, size: int) -> bytes:
        """Read a metadata image only within its independent size ceiling."""
        if not 0 < size <= _METADATA_LIMIT:
            raise _CorruptCatalog("invalid index catalog size")
        body = IndexCacheFiles.read(fd, size + 1)
        if len(body) != size:
            raise sqlite3.DatabaseError("catalog changed while reading")
        return body

    @classmethod
    def load(cls, directory: int, connection: sqlite3.Connection) -> None:
        """Load the current catalog through a checked file descriptor."""
        with IndexCacheFiles.open_file(directory, _CATALOG) as (fd, status):
            cls.deserialize(connection, cls.read(fd, status.st_size))

    @staticmethod
    def record(
        row: tuple[bytes, str, bytes, int, float] | None, limit: int
    ) -> CacheRecord | None:
        """Validate a row before passing its reference to the caller."""
        if row is None:
            return None
        record = CacheRecord(*row)
        return record if record.valid(limit) else None


class CacheCatalog:
    """Store index metadata and payloads; expected I/O or SQLite failures are misses.

    The budget counts payload bytes. SQLite works on a bounded private image in
    memory. A process lock serializes writers, which publish the payload before
    atomically replacing the catalog. Readers see complete generations only.
    """

    _directory: Path
    _limit: int
    _payload_limit: int
    _mutex: Lock
    _image: sqlite3.Connection | None
    _close_image: Callable[[], None] | None
    _version: tuple[int, int, int, int] | None

    def __init__(self, directory: Path, limit: int) -> None:
        self._directory = directory.absolute()
        self._limit = limit
        self._payload_limit = min(limit, MAX_INDEX_BYTES)
        self._mutex = Lock()
        self._image = None
        self._close_image = None
        self._version = None

    def _lookup(
        self, directory: int, path: Path, size: int, policy: str
    ) -> CacheRecord | None:
        with (
            self._mutex,
            IndexCacheFiles.open_file(directory, _CATALOG) as (fd, status),
        ):
            version = (status.st_dev, status.st_ino, status.st_ctime_ns, status.st_size)
            if version != self._version:
                self._refresh(fd, status.st_size, version)
            if self._image is None:
                return None
            row = self._image.execute(
                "SELECT digest, name, checksum, length, written FROM entries WHERE path = ? AND size = ? AND policy = ?",
                (os.fsencode(path), size, policy),
            ).fetchone()
            return _Metadata.record(row, self._payload_limit)

    def _refresh(self, fd: int, size: int, version: tuple[int, int, int, int]) -> None:
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        try:
            _Metadata.configure(connection)
            _Metadata.deserialize(connection, _Metadata.read(fd, size))
        except sqlite3.Error:
            connection.close()
            raise
        if self._close_image is not None:
            self._close_image()
        self._image = connection
        self._close_image = weakref.finalize(self, connection.close)
        self._version = version

    def find(self, path: Path, size: int, policy: str) -> CacheRecord | None:
        """Look up same-key metadata without opening or reading the payload."""
        if not path.is_absolute():
            return None
        try:
            with IndexCacheFiles.directory(self._directory, create=False) as directory:
                return self._lookup(directory, path, size, policy)
        except _ERRORS:
            return None

    def read(self, record: CacheRecord) -> bytes | None:
        """Read a bounded immutable payload only if its checksum matches."""
        if not record.valid(self._payload_limit):
            return None
        try:
            with (
                IndexCacheFiles.directory(self._directory, create=False) as directory,
                IndexCacheFiles.open_file(directory, record.name) as (fd, status),
            ):
                if status.st_size != record.length:
                    return None
                body = IndexCacheFiles.read(fd, record.length + 1)
        except _ERRORS:
            return None
        return (
            body
            if len(body) == record.length
            and hashlib.sha256(body).digest() == record.checksum
            else None
        )

    def _save_entry(
        self,
        directory: int,
        connection: sqlite3.Connection,
        path: Path,
        size: int,
        policy: str,
        digest: bytes,
        body: bytes,
    ) -> None:
        name = f"index-{uuid.uuid4().hex}.idx"
        now = time.time()
        connection.execute(
            "INSERT OR REPLACE INTO entries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                os.fsencode(path),
                size,
                policy,
                digest,
                name,
                hashlib.sha256(body).digest(),
                len(body),
                now,
            ),
        )
        _Metadata.trim(connection, self._limit, now)
        connection.commit()
        image = connection.serialize()
        if len(image) > _METADATA_LIMIT:
            return
        IndexCacheFiles.publish(directory, name, body)
        IndexCacheFiles.publish(directory, _CATALOG, image)
        logger.debug("persistent index stored: {}", path)
        retained = {name for (name,) in connection.execute("SELECT name FROM entries")}
        IndexCacheFiles.sweep(directory, retained)

    def _store(
        self,
        directory: int,
        path: Path,
        size: int,
        policy: str,
        digest: bytes,
        body: bytes,
    ) -> None:
        with closing(sqlite3.connect(":memory:")) as connection:
            _Metadata.configure(connection)
            _Metadata.load(directory, connection)
            try:
                self._save_entry(
                    directory, connection, path, size, policy, digest, body
                )
            except sqlite3.DatabaseError as error:
                if _Metadata.is_damage(error):
                    raise _CorruptCatalog("damaged index catalog entries") from error
                raise

    def _initialize(
        self,
        directory: int,
        path: Path,
        size: int,
        policy: str,
        digest: bytes,
        body: bytes,
    ) -> None:
        with closing(sqlite3.connect(":memory:")) as connection:
            _Metadata.configure(connection)
            connection.execute(_SCHEMA)
            self._save_entry(directory, connection, path, size, policy, digest, body)

    def put(
        self, path: Path, size: int, policy: str, digest: bytes, body: bytes
    ) -> None:
        """Publish one record, evicting expired and over-budget payloads."""
        if (
            not path.is_absolute()
            or len(digest) != 64
            or len(body) > self._payload_limit
        ):
            return
        try:
            with IndexCacheFiles.directory(self._directory, create=True) as directory:
                if self._lock_writer(directory):
                    self._save(directory, path, size, policy, digest, body)
        except _ERRORS:
            return

    @staticmethod
    def _lock_writer(directory: int) -> bool:
        deadline = time.monotonic() + _LOCK_TIMEOUT
        while True:
            try:
                fcntl.flock(directory, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                time.sleep(min(_LOCK_POLL, remaining))

    def _save(
        self,
        directory: int,
        path: Path,
        size: int,
        policy: str,
        digest: bytes,
        body: bytes,
    ) -> None:
        try:
            self._store(directory, path, size, policy, digest, body)
        except (FileNotFoundError, _CorruptCatalog):
            self._initialize(directory, path, size, policy, digest, body)
