"""Refresh the active run and reuse complete persisted event indexes."""

import os
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from stat import S_ISREG
from typing import Any

from loguru import logger

from tboardfs._directory import open_plain_directory
from tboardfs.disk_index import DiskIndex, _IndexHit
from tboardfs.event_source import EventSource
from tboardfs.file_status import FileIdentity
from tboardfs.indexer import _EventIndexer
from tboardfs.model import EventFileCache, RunCache

_DEFAULT_DISK_INDEX_SIZE_MB = 2048


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
            raise TypeError("disk_index must be a bool")
        if type(disk_index_size_mb) is not int or disk_index_size_mb < 1:
            raise ValueError("disk_index_size_mb must be a positive integer")
        mounted_at = Path(mountpoint).resolve() if mountpoint is not None else None
        storage = _disk_index_directory(source, mounted_at) if disk_index else None
        if storage is None:
            return cls(mounted_at, index_workers, parallel_min_bytes)
        disk = DiskIndex(storage, limit=disk_index_size_mb << 20, mountpoint=mounted_at)
        return cls(mounted_at, index_workers, parallel_min_bytes, disk, storage)

    def directory_time(self, directory: Path) -> float:
        """Read a backing directory's timestamp without scanning its entries."""
        try:
            return EventSource.directory_stat(
                directory, mountpoint=self.mountpoint, storage=self.storage
            ).st_mtime
        except OSError:
            return 0.0

    def candidates(self, paths: list[Path], run: RunCache) -> dict[Path, _IndexHit]:
        """Check only the not-yet-indexed files of this run."""
        if self.disk is None:
            return {}
        cold = [path for path in paths if path not in run.files]
        return self.disk.candidates(cold, self.workers) if cold else {}


@dataclass(frozen=True, slots=True)
class _SiblingSnapshot:
    """Compare the last indexed siblings without reconstructing the update plan."""

    parent: Path
    blocked: tuple[str, ...]
    files: tuple[tuple[str, tuple[int | None, ...]], ...]

    @classmethod
    def capture(
        cls, run: RunCache, options: _IndexOptions
    ) -> "_SiblingSnapshot | None":
        """Keep checked fingerprints only for immediate siblings of one guarded parent."""
        if options.mountpoint is None or len(run.files) < 2:
            return None
        parent = next(iter(run.files)).parent
        roots = (options.mountpoint, options.storage)
        files = []
        for path, cache in run.files.items():
            if path.parent != parent or path in roots or path.name in ("", ".", ".."):
                return None
            files.append((path.name, cache.progress.checked_status))
        return cls(
            parent, tuple(str(root) for root in roots if root is not None), tuple(files)
        )

    def matches(self) -> bool:
        """Check fresh no-follow status; defer any uncertain or changed source."""
        fd = open_plain_directory(self.parent, self.blocked)
        if fd is None:
            return False
        try:
            for name, expected in self.files:
                status = os.stat(name, dir_fd=fd, follow_symlinks=False)
                if (
                    not S_ISREG(status.st_mode)
                    or FileIdentity.fingerprint(status) != expected
                ):
                    return False
        except OSError:
            return False
        finally:
            os.close(fd)
        return True


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
        parent = files[0].parent if len(files) > 1 and run.files and not force else None
        with EventSource._sibling_stat(
            parent,
            mountpoint=options.mountpoint,
            storage=options.storage if options.mountpoint is not None else None,
        ) as stat:
            return _RunData._refresh_sources(
                run, files, force=force, log=log, options=options, stat=stat
            )

    @staticmethod
    def _refresh_sources(
        run: RunCache,
        files: list[Path],
        *,
        force: bool,
        log: Callable[[str, object], None] | None,
        options: _IndexOptions,
        stat: Callable[[Path], os.stat_result],
    ) -> tuple[bool, EventFileCache | None]:
        """Apply status changes and carry metadata in source order."""
        changed = False
        base = None
        carry: dict[str, dict[str, Any]] = {}
        hits = options.candidates(files, run)
        retained = 0
        for source_path in sorted(files):
            old = run.files.get(source_path)
            reparse = force or (
                old is not None and old.progress.initial_metadata != carry
            )
            try:
                if (
                    old is None
                    or FileIdentity.fingerprint(stat(source_path))
                    != old.progress.checked_status
                    or reparse
                ):
                    current = _RunData._update_source(
                        source_path,
                        old=old,
                        force=reparse,
                        log=log,
                        initial_metadata=carry,
                        options=options,
                        hit=hits.get(source_path),
                    )
                    run.files[source_path] = current
                    base = old if not changed and current.progress.resumed else None
                    changed = True
                else:
                    current = old
            except OSError as error:
                if not isinstance(error, FileNotFoundError):
                    logger.warning("cannot read event file {}: {}", source_path, error)
                if run.files.pop(source_path, None) is not None:
                    changed, base = True, None
                continue
            carry = current.progress.effective_metadata
            retained += 1
        if retained != len(run.files):
            for deleted in set(run.files) - set(files):
                if log is not None:
                    log("event file removed {}", deleted)
                del run.files[deleted]
                changed, base = True, None
        return changed, base

    @staticmethod
    def _update_source(
        source_path: Path,
        *,
        old: EventFileCache | None,
        force: bool,
        log: Callable[[str, object], None] | None,
        initial_metadata: dict[str, dict[str, Any]],
        options: _IndexOptions,
        hit: _IndexHit | None,
    ) -> EventFileCache:
        """Load or parse an index after detecting a new or changed source."""
        disk = options.disk if old is None else None
        if disk is not None and hit is not None:
            cached = disk.load(hit, source_path, initial_metadata)
            if cached is not None:
                return cached
        if log is not None:
            log("parsing {}", source_path)
        cache = _RunData.parse_event_file(
            source_path,
            old=old,
            initial_metadata=initial_metadata,
            options=options,
            force=force,
        )
        if (
            disk is not None
            and cache.progress.provenance.cacheable
            and cache.progress.processed_pos == cache.progress.last_checked_size
        ):
            disk.save(cache, hit)
        return cache

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
