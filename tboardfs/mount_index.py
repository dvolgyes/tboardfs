"""Refresh the active run and reuse complete persisted event indexes."""

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
import os
from pathlib import Path
from typing import Any, TypedDict

from loguru import logger

from tboardfs.disk_index import DiskIndex, _IndexHit
from tboardfs.event_source import EventSource
from tboardfs.file_status import FileIdentity
from tboardfs.indexer import _EventIndexer
from tboardfs.model import EventFileCache, RunCache


_DEFAULT_DISK_INDEX_SIZE_MB = 2048


class _MountIndexKeywords(TypedDict, total=False):
    """Optional indexing settings for the mount constructor."""

    index_workers: int
    disk_index: bool
    disk_index_size_mb: int


def _disk_index_directory(source: Path, mountpoint: Path | None) -> Path | None:
    """Reject unsafe cache paths before FUSE can hide any part of the path."""
    try:
        xdg = Path(os.environ.get("XDG_CACHE_HOME", ""))
        base = xdg if xdg.is_absolute() else Path.home() / ".cache"
        directory = base / "tboardfs" / "indexes-v1"
        if (
            not directory.is_absolute()
            or ".." in directory.parts
            or _has_symlink(directory)
        ):
            logger.warning("unsafe persistent index directory {}", directory)
            return None
        roots = (source.resolve(), mountpoint)
    except (OSError, RuntimeError) as error:
        logger.warning("cannot resolve persistent index directory: {}", error)
        return None
    for root in roots:
        if root is not None and (
            directory.is_relative_to(root) or root.is_relative_to(directory)
        ):
            logger.warning("persistent index directory {} overlaps {}", directory, root)
            return None
    return directory


def _has_symlink(directory: Path) -> bool:
    return any(parent.is_symlink() for parent in reversed(directory.parents)) or (
        directory.is_symlink()
    )


@dataclass(frozen=True, slots=True)
class _IndexOptions:
    """Event-source lookup, worker limits, and persistence for this mount."""

    mountpoint: Path | None
    workers: int
    parallel_min_bytes: int
    disk: DiskIndex | None = None
    storage: Path | None = None

    @classmethod
    def create(
        cls,
        source: Path,
        mountpoint: str | Path | None,
        parallel_min_bytes: int,
        *,
        index_workers: int = 1,
        disk_index: bool = False,
        disk_index_size_mb: int = _DEFAULT_DISK_INDEX_SIZE_MB,
    ) -> "_IndexOptions":
        """Build a passive service only when the caller opts into safe storage."""
        if index_workers < 1:
            raise ValueError("index_workers must be positive")
        if not isinstance(disk_index, bool):
            raise ValueError("disk_index must be a bool")
        if type(disk_index_size_mb) is not int or disk_index_size_mb < 1:
            raise ValueError("disk_index_size_mb must be a positive integer")
        mounted_at = Path(mountpoint).resolve() if mountpoint is not None else None
        storage = _disk_index_directory(source, mounted_at) if disk_index else None
        if storage is None:
            return cls(mounted_at, index_workers, parallel_min_bytes)
        disk = DiskIndex(storage, limit=disk_index_size_mb << 20, mountpoint=mounted_at)
        return cls(mounted_at, index_workers, parallel_min_bytes, disk, storage)

    def candidates(self, paths: list[Path], run: RunCache) -> dict[Path, _IndexHit]:
        """Check only the not-yet-indexed files of this run."""
        if self.disk is None:
            return {}
        cold = [path for path in paths if path not in run.files]
        return self.disk.candidates(cold, self.workers) if cold else {}


class _RunData:
    """Refresh event-file caches and collect run entries."""

    @staticmethod
    def refresh_event_files(
        run: RunCache,
        files: list[Path],
        *,
        force: bool,
        log: Callable[[str, object], None] | None,
        options: _IndexOptions,
    ) -> tuple[bool, EventFileCache | None]:
        """Report a change and the base index if exactly one source resumed."""
        changed = False
        base = None
        carry: dict[str, dict[str, Any]] = {}
        hits = options.candidates(files, run)
        for source_path in sorted(files):
            old = run.files.get(source_path)
            try:
                if _RunData.refresh_event_file(
                    run,
                    source_path,
                    force=force
                    or (old is not None and old.progress.initial_metadata != carry),
                    log=log,
                    initial_metadata=carry,
                    options=options,
                    hit=hits.get(source_path),
                ):
                    base = (
                        old
                        if not changed and run.files[source_path].progress.resumed
                        else None
                    )
                    changed = True
            except OSError as error:
                if not isinstance(error, FileNotFoundError):
                    logger.warning("cannot read event file {}: {}", source_path, error)
                if run.files.pop(source_path, None) is not None:
                    changed, base = True, None
            if source_path in run.files:
                carry = run.files[source_path].progress.effective_metadata
        for deleted in set(run.files) - set(files):
            if log is not None:
                log("event file removed {}", deleted)
            del run.files[deleted]
            changed, base = True, None
        return changed, base

    @staticmethod
    def refresh_event_file(
        run: RunCache,
        source_path: Path,
        *,
        force: bool,
        log: Callable[[str, object], None] | None,
        initial_metadata: dict[str, dict[str, Any]],
        options: _IndexOptions,
        hit: _IndexHit | None,
    ) -> bool:
        """Check a known source; open and parse it only when its status changes."""
        old = run.files.get(source_path)
        if (
            not force
            and old is not None
            and FileIdentity.fingerprint(
                EventSource.stat(source_path, mountpoint=options.mountpoint)
            )
            == old.progress.checked_status
        ):
            return False
        disk = options.disk if old is None else None
        if disk is not None and hit is not None:
            cached = disk.load(hit, source_path, initial_metadata)
            if cached is not None:
                run.files[source_path] = cached
                return True
        if log is not None:
            log("parsing {}", source_path)
        cache = _RunData.parse_event_file(
            source_path,
            old=old,
            initial_metadata=initial_metadata,
            options=options,
            force=force,
        )
        run.files[source_path] = cache
        if (
            disk is not None
            and cache.progress.provenance.cacheable
            and cache.progress.processed_pos == cache.progress.last_checked_size
        ):
            disk.save(cache, hit)
        return True

    @staticmethod
    def parse_event_file(
        path: Path,
        *,
        old: EventFileCache | None,
        initial_metadata: dict[str, dict[str, Any]],
        options: _IndexOptions,
        force: bool,
    ) -> EventFileCache:
        """Select the complete cold parser, with the live serial parser as fallback."""
        if (
            old is None
            and options.workers > 1
            and EventSource.stat(path, mountpoint=options.mountpoint).st_size
            >= options.parallel_min_bytes
        ):
            parallel = import_module("tboardfs.parallel_index")
            cache: EventFileCache | None = parallel.parallel_index(
                path,
                options.workers,
                initial_metadata=initial_metadata,
                mountpoint=options.mountpoint,
            )
            if cache is not None:
                return cache
        return _EventIndexer.parse_event_file(
            path,
            ignore_truncated=True,
            base_cache=None if force else old,
            initial_metadata=initial_metadata,
            mountpoint=options.mountpoint,
            retain_tensor_extents=old is None and options.disk is not None,
        )
