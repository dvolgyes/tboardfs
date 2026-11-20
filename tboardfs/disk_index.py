"""Reuse complete file indexes only after validating the current source bytes."""

import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BufferedReader
from pathlib import Path
from typing import Any, BinaryIO, cast

import xxhash
from loguru import logger

from tboardfs.event_source import EventSource
from tboardfs.file_status import FileIdentity
from tboardfs.index_cache_record import CacheRecord
from tboardfs.index_catalog import CacheCatalog
from tboardfs.index_codec import IndexCodec
from tboardfs.index_policy import IndexPolicy
from tboardfs.model import EventFileCache

type _Fingerprint = tuple[int | None, ...]


@dataclass(frozen=True, slots=True)
class _IndexHit:
    """A size-matched catalog entry and the digest observed on a stable source."""

    path: Path
    fingerprint: _Fingerprint
    digest: bytes
    record: CacheRecord


class _Source:
    """Fence source reads with descriptor and mount-aware pathname checks."""

    mountpoint: Path | None

    def __init__(self, mountpoint: Path | None) -> None:
        self.mountpoint = mountpoint

    def matches(self, path: Path, handle: BinaryIO, expected: _Fingerprint) -> bool:
        """Require both the open file and the named source to match the snapshot."""
        return FileIdentity.fingerprint(os.fstat(handle.fileno())) == expected and (
            FileIdentity.fingerprint(EventSource.stat(path, mountpoint=self.mountpoint))
            == expected
        )

    def digest(self, path: Path, expected: _Fingerprint) -> bytes | None:
        """Hash exactly the observed size; never chase a continuously growing tail."""
        with EventSource.open(path, mountpoint=self.mountpoint) as handle:
            status = os.fstat(handle.fileno())
            if FileIdentity.fingerprint(status) != expected:
                return None
            remaining = status.st_size
            scratch = memoryview(bytearray(min(1 << 20, remaining)))
            digest = xxhash.xxh3_128()
            stream = cast(BufferedReader, handle)
            while remaining:
                size = stream.readinto(scratch[:remaining])
                if not size:
                    return None
                digest.update(scratch[:size])
                remaining -= size
            if not self.matches(path, handle, expected):
                return None
            result: bytes = digest.digest()
            return result


class DiskIndex:
    """Coordinate advisory storage without weakening source access or refresh rules."""

    _source: _Source
    _catalog: CacheCatalog
    _policy: IndexPolicy | None
    _limit: int

    def __init__(self, directory: Path, *, limit: int, mountpoint: Path | None) -> None:
        self._source = _Source(mountpoint)
        self._limit = limit
        self._catalog = CacheCatalog(directory, limit)
        self._policy = IndexPolicy.capture(mountpoint=mountpoint) if limit > 0 else None

    def _candidate(self, path: Path) -> _IndexHit | None:
        policy = self._policy
        if policy is None:
            return None
        try:
            status = EventSource.stat(path, mountpoint=self._source.mountpoint)
            record = self._catalog.find(path.absolute(), status.st_size, policy.id)
            if record is None:
                return None
            expected = FileIdentity.fingerprint(status)
            digest = self._source.digest(path, expected)
            if digest is not None:
                return _IndexHit(path, expected, digest, record)
        except OSError:
            return None
        return None

    def candidates(self, paths: Iterable[Path], workers: int) -> dict[Path, _IndexHit]:
        """Check only cold sources in the active run, with bounded hash concurrency."""
        if self._policy is None or not self._policy.unchanged():
            return {}
        sources = list(paths)
        count = min(len(sources), workers, 4)
        if count < 2:
            return {hit.path: hit for path in sources if (hit := self._candidate(path))}
        with ThreadPoolExecutor(max_workers=count) as pool:
            return {hit.path: hit for hit in pool.map(self._candidate, sources) if hit}

    def load(
        self, hit: _IndexHit, path: Path, metadata: dict[str, dict[str, Any]]
    ) -> EventFileCache | None:
        """Restore source-bound values only after the whole-file digest matches."""
        if hit.path != path or hit.digest != hit.record.digest or self._policy is None:
            return None
        try:
            with EventSource.open(path, mountpoint=self._source.mountpoint) as source:
                if not self._source.matches(path, source, hit.fingerprint):
                    return None
                body = self._catalog.read(hit.record)
                if body is None:
                    return None
                cache = IndexCodec.decode(
                    body, path, os.fstat(source.fileno()), source, metadata
                )
                if (
                    cache is not None
                    and self._source.matches(path, source, hit.fingerprint)
                    and self._policy.unchanged()
                ):
                    logger.debug("persistent index hit: {}", path)
                    return cache
        except (OSError, ValueError) as error:
            logger.debug("persistent index unavailable for {}: {}", path, error)
        return None

    def save(self, cache: EventFileCache, hit: _IndexHit | None) -> None:
        """Publish only a complete index and digest from the same source generation."""
        policy = self._policy
        progress = cache.progress
        if (
            policy is None
            or not progress.provenance.cacheable
            or progress.resumed
            or not progress.processed_pos
        ):
            return
        if (
            progress.processed_pos != progress.last_checked_size
            or not policy.unchanged()
        ):
            return
        path = cache.source_path
        expected = progress.checked_status
        try:
            digest = (
                hit.digest
                if hit is not None and hit.path == path and hit.fingerprint == expected
                else self._source.digest(path, expected)
            )
            if digest is None:
                return
            body = IndexCodec.encode(cache, limit=self._limit)
            if (
                FileIdentity.fingerprint(
                    EventSource.stat(path, mountpoint=self._source.mountpoint)
                )
                == expected
                and policy.unchanged()
            ):
                self._catalog.put(
                    path.absolute(), progress.last_checked_size, policy.id, digest, body
                )
        except (OSError, ValueError) as error:
            logger.debug("could not save persistent index for {}: {}", path, error)
