"""Serve the mounted, read-only view of TensorBoard event data."""

import os
import stat
import time
from errno import EINVAL, EIO, EISDIR, ENOENT, ENOTDIR, EROFS
from functools import lru_cache, partial
from importlib.resources import files as package_files
from io import FileIO
from math import inf
from pathlib import Path
from threading import RLock
from time import monotonic as _source_clock
from typing import Any, Unpack

from loguru import logger

from tboardfs._directory import (
    _GUARD_PLAN_MAX_PATH_CHARS,
    _guard_plan,
    _plain_path,
)
from tboardfs._guarded_render import _render_miss
from tboardfs.constants import (
    DEFAULT_CACHE_SIZE_MB,
    DEFAULT_DISCOVERY_SECONDS,
    DEFAULT_SCALAR_FORMATS,
)
from tboardfs.errors import fuse_error
from tboardfs.export_formats import FormatList, _ExportFormats
from tboardfs.file_attributes import _FileAttributes, _unrendered_size
from tboardfs.file_times import FileTimes
from tboardfs.leaf_cache import _CachedLeaf, _Location, _resolve_node
from tboardfs.model import (
    BinaryEntry,
    EventFileCache,
    GraphEntry,
    RunCache,
    SidecarEntry,
)
from tboardfs.mount_index import (
    MountIndexKeywords,
    _IndexOptions,
    _RunData,
    _SiblingSnapshot,
)
from tboardfs.mount_state import _Directory, _FsState, _mount_formats, _MountKeywords
from tboardfs.node_source import _NodeSource
from tboardfs.paths import _DirectoryNode, _Paths
from tboardfs.render_cache import RenderCache, RenderKey
from tboardfs.render_node import render_node
from tboardfs.run_places import RunPlaces
from tboardfs.run_tree import RunTree
from tboardfs.sidecar_scan import SidecarScan
from tboardfs.verified_span import _VerifiedSpan

_PARALLEL_MIN_BYTES = 8 << 20

_GUARD_PLAN_LIMIT = 256


class TensorBoardFS:
    """Read-only virtual filesystem over TensorBoard event files.

    FUSE calls the callbacks from several threads. Discovery, refresh and
    tree lookup run under ``_lock``; rendering a node runs outside it,
    because nodes and event-file caches are not changed after creation.
    ``open`` pins a source descriptor or rendered bytes until ``release``.
    Rendered bytes are cached by the data versions of the runs on the path,
    so a change in one run makes only its own files render again.

    :ivar source: source log directory
    :ivar step_digits: zero-padding width for step filenames
    :ivar refresh_age_seconds: minimum interval between local directory scans
    :ivar source_check_seconds: minimum interval between checks of known event files
    :ivar scalar_formats: enabled scalar export formats
    :ivar index_workers: maximum workers for a cold event-file index
    :ivar log_to_screen: whether operations log status messages
    :ivar owner: user and group ids that own every file and directory
    :ivar _renders: rendered file bytes by run versions and path
    :ivar _state: mutable run discovery and tree state
    :ivar _lock: serializes every change and read of ``_state``
    :ivar _run_places: places the run trees in one tree, and keeps the run
        folders with a suffix so each is logged once
    """

    _guide: dict[str, Any]
    _formats: _ExportFormats
    _raw: bool
    source: Path
    step_digits: int
    refresh_age_seconds: float
    log_to_screen: bool
    owner: tuple[int, int]
    _renders: RenderCache
    _state: "_FsState"
    _lock: RLock
    _run_places: RunPlaces

    def __init__(
        self,
        source: str | Path,
        *,
        step_digits: int = 6,
        refresh_age_seconds: float = DEFAULT_DISCOVERY_SECONDS,
        scalar_formats: FormatList = DEFAULT_SCALAR_FORMATS,
        log_to_screen: bool = False,
        cache_size_mb: int = DEFAULT_CACHE_SIZE_MB,
        owner: tuple[int, int] = (0, 0),
        mountpoint: str | Path | None = None,
        **options: Unpack[_MountKeywords],
    ) -> None:
        self.source = Path(source)
        self._guide = {
            "type": "bytes",
            "data": package_files("tboardfs").joinpath("llms.txt").read_bytes(),
            "mtime": 0.0,
        }
        self._raw = options.pop("raw", False)
        self.step_digits = step_digits
        self.refresh_age_seconds = refresh_age_seconds
        self._formats = _mount_formats(scalar_formats, options)
        self.log_to_screen = log_to_screen
        self.owner = owner
        self._renders = RenderCache(cache_size_mb * 1024 * 1024)
        index_options: MountIndexKeywords = options
        index = _IndexOptions.create(
            self.source, mountpoint, _PARALLEL_MIN_BYTES, **index_options
        )
        blocked = frozenset(
            str(path) for path in (index.mountpoint, index.storage) if path is not None
        )
        self._state = _FsState(
            index,
            lru_cache(maxsize=_GUARD_PLAN_LIMIT)(partial(_guard_plan, blocked))
            if blocked
            else None,
        )
        if index.disk is not None:
            self.source = self.source.absolute()
        self._lock = RLock()
        self._run_places = RunPlaces()
        self._state.directories[()] = _Directory()
        self._ensure_run(())

    @property
    def scalar_formats(self) -> tuple[str, ...]:
        """Return the enabled scalar export formats."""
        return self._formats.scalars

    @property
    def source_check_seconds(self) -> float:
        """Return the interval between checks of a run's known event sources."""
        return self._state.index.source_check_seconds

    @property
    def index_workers(self) -> int:
        """Return the configured limit for cold indexing workers."""
        return self._state.index.workers

    def virtual_tree(self) -> dict[str, Any]:
        """Return the current root node; nodes are not changed after creation."""
        with self._lock:
            pending: list[tuple[str, ...]] = [()]
            while pending:
                parts = pending.pop()
                self._activate(parts)
                pending.extend(
                    sorted(self._state.directories[parts].children, reverse=True)
                )
            return self._cached_tree()

    def getattr(self, path: str, *file_handles: object) -> dict[str, Any]:
        """Return stat data for a virtual path, or for an open handle."""
        return self._attributes(path, file_handles).as_dict()

    def _attributes(
        self, path: str | None, file_handles: tuple[object, ...]
    ) -> _FileAttributes:
        """Read current path metadata or the metadata pinned by an open handle."""
        opened = self._open_file(file_handles)
        if opened is not None:
            content, mtime = opened
            size = content.size if isinstance(content, _VerifiedSpan) else len(content)
            return _FileAttributes(False, size, mtime, self.owner)
        if path is None:
            raise fuse_error(EINVAL)
        return self._path_attributes(path)

    def readdir(self, path: str, *file_handles: object) -> list[str]:
        """Return virtual directory entries."""
        del file_handles
        _, names = self._listed_directory(path)
        return [".", "..", *names]

    def _listed_directory(self, path: str) -> tuple[_DirectoryNode, tuple[str, ...]]:
        """Return the refreshed directory snapshot and its sorted names."""
        with self._lock:
            node = self._lookup(path)
            if not isinstance(node, _DirectoryNode):
                raise fuse_error(ENOTDIR)
            return node, node.sorted_names()

    def open(self, path: str, flags: int) -> int:
        """Open a virtual file and pin its current version for the handle."""
        del flags
        node, key = self._locate(path)
        if node["type"] == "dir":
            raise fuse_error(EISDIR)
        mtime = _CachedLeaf.time_of(node)
        span = self._node_span(node)
        opened = (self._node_bytes(node, key) if span is None else span, mtime)
        with self._lock:
            handle = self._state.next_handle
            self._state.next_handle += 1
            self._state.open_files[handle] = opened
        return handle

    def release(self, path: str, handle: int) -> int:
        """Release the version pinned by an open handle."""
        del path
        with self._lock:
            opened = self._state.open_files.pop(handle, None)
        if opened is not None and isinstance(opened[0], _VerifiedSpan):
            opened[0].close()
        return 0

    def read(self, path: str, size: int, offset: int, *file_handles: object) -> bytes:
        """Read a byte range from an open handle, or from the current file."""
        opened = self._open_file(file_handles)
        if opened is not None:
            content = opened[0]
            if isinstance(content, _VerifiedSpan):
                return content.read(size, offset)
            return content[offset : offset + size]
        node, key = self._locate(path)
        if node["type"] == "dir":
            raise fuse_error(EISDIR)
        span = self._node_span(node)
        if span is not None:
            try:
                return span.read(size, offset)
            finally:
                span.close()
        data = self._node_bytes(node, key)
        return data[offset : offset + size]

    def symlink(self, target: str, source: str) -> None:
        """Reject symlink creation because the filesystem is read-only."""
        del target, source
        raise fuse_error(EROFS)

    def unlink(self, path: str) -> None:
        """Reject file removal because the filesystem is read-only."""
        del path
        raise fuse_error(EROFS)

    def mkdir(self, path: str, mode: int) -> None:
        """Reject directory creation because the filesystem is read-only."""
        del path, mode
        raise fuse_error(EROFS)

    def mknod(self, path: str, mode: int, dev: int) -> None:
        """Reject node creation because the filesystem is read-only."""
        del path, mode, dev
        raise fuse_error(EROFS)

    def create(self, path: str, mode: int, *file_info: object) -> None:
        """Reject file creation because the filesystem is read-only."""
        del path, mode, file_info
        raise fuse_error(EROFS)

    def write(self, path: str, data: bytes, offset: int, *file_handles: object) -> None:
        """Reject writes because the filesystem is read-only."""
        del path, data, offset, file_handles
        raise fuse_error(EROFS)

    def rename(self, old: str, new: str) -> None:
        """Reject renames because the filesystem is read-only."""
        del old, new
        raise fuse_error(EROFS)

    def rmdir(self, path: str) -> None:
        """Reject directory removal because the filesystem is read-only."""
        del path
        raise fuse_error(EROFS)

    def truncate(self, path: str, length: int, *file_handles: object) -> None:
        """Reject truncation because the filesystem is read-only."""
        del path, length, file_handles
        raise fuse_error(EROFS)

    def chmod(self, path: str, mode: int) -> None:
        """Reject chmod because the filesystem is read-only."""
        del path, mode
        raise fuse_error(EROFS)

    def chown(self, path: str, uid: int, gid: int) -> None:
        """Reject chown because the filesystem is read-only."""
        del path, uid, gid
        raise fuse_error(EROFS)

    def _lookup(self, path: str) -> dict[str, Any]:
        return self._locate(path)[0]

    def _path_attributes(self, path: str) -> _FileAttributes:
        with self._lock:
            (node, key), checked_paths = self._locate_locked(path, directory_attrs=True)
            mtime = _CachedLeaf.time_of(node)
            owner = self._run_places.disk_parts.get(key[1]) if not mtime else None
            if (
                owner is not None
                and self._state.directories[owner].scanned_at is not None
            ):
                owner = None
        if owner is not None:
            mtime = self._state.index.directory_time(self.source.joinpath(*owner))
        directory = node["type"] == "dir"
        size = (
            0 if directory else self._node_size(node, key, checked_paths=checked_paths)
        )
        return _FileAttributes(directory, size, mtime, self.owner)

    def _locate(self, path: str) -> _Location:
        """Return the node at ``path`` and its generation-scoped render key.

        The key is read under the same lock as the node, so both match.
        """
        with self._lock:
            return self._locate_locked(path, directory_attrs=False)[0]

    def _locate_locked(
        self, path: str, *, directory_attrs: bool
    ) -> tuple[_Location, frozenset[Path] | None]:
        """Locate a node and return any paths checked by this unchanged lookup."""
        snapshot = self._state.tree
        leaves = snapshot.leaves
        parts, entry, parent = leaves.probe(path)
        refreshed: tuple[str, ...] | None
        if entry is not None:
            refreshed = entry[0]
        else:
            refreshed = parent.owner if parent is not None else None
        generation = snapshot.generation
        checked_paths = self._activate(refreshed) if refreshed is not None else None
        unchanged = generation == snapshot.generation
        if unchanged and entry is not None:
            return entry[1], checked_paths
        sibling = parent.leaf(parts[-1]) if unchanged and parent is not None else None
        if sibling is not None and refreshed is not None:
            scope = self._renders.scope(self._state.runs[refreshed].version)
            location = leaves.remember(path, refreshed, sibling, (scope, parts))
            return location, checked_paths
        tree, owner = self._refresh_lookup(parts, refreshed, directory_attrs)
        node, children = _resolve_node(tree, parts, path)
        run = self._state.runs.get(owner)
        version = None if node["type"] == "name_map" or run is None else run.version
        key = self._renders.scope(version), parts
        if snapshot.root is tree and version is not None:
            location = leaves.remember(
                path,
                owner,
                node,
                key,
                children=children,
                disk_parts=self._run_places.disk_parts,
            )
        else:
            location = node, key
        return location, None

    def _refresh_lookup(
        self,
        parts: tuple[str, ...],
        refreshed: tuple[str, ...] | None,
        directory_attrs: bool,
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        tree = self._cached_tree()
        depth, owner = self._lookup_start(parts, directory_attrs=directory_attrs)
        while True:
            if (
                directory_attrs
                and depth == len(parts)
                and self._state.directories[owner].scanned_at is None
            ):
                return tree, owner
            if owner != refreshed:
                self._activate(owner)
                tree = self._cached_tree()
            if depth == len(parts):
                break
            child_owner = self._run_places.disk_parts.get(parts[: depth + 1])
            if child_owner is None:
                break
            owner = child_owner
            refreshed = None
            depth += 1
        return tree, self._run_places.mapped_owner(parts, depth)[1]

    def _lookup_start(
        self, parts: tuple[str, ...], *, directory_attrs: bool
    ) -> tuple[int, tuple[str, ...]]:
        depth, owner = self._run_places.mapped_owner(parts, len(parts))
        if directory_attrs and depth == len(parts) and depth:
            now = _source_clock()
            disk_parts = self._run_places.disk_parts
            for ancestor_depth in range(depth):
                ancestor = disk_parts[parts[:ancestor_depth]]
                scanned_at = self._state.directories[ancestor].scanned_at
                if scanned_at is None or now - scanned_at >= self.refresh_age_seconds:
                    return ancestor_depth, ancestor
        return depth, owner

    def _build_tree(self) -> dict[str, Any]:
        tree = self._run_places.build(
            {
                parts: self._visible_run(parts) or _DirectoryNode()
                for parts in self._state.visible_dirs()
            }
        )
        # The run trees have their times, so only the run folders get them.
        FileTimes.stamp_dirs(tree)
        return tree

    def _visible_run(self, parts: tuple[str, ...]) -> dict[str, Any] | None:
        """Return the run's own tree, or leave its mount directory empty."""
        run = self._state.runs.get(parts)
        return self._run_tree(run) if run is not None and run.files else None

    def _run_tree(self, run: RunCache) -> dict[str, Any]:
        cached = self._state.run_trees.get(run.run_parts)
        if cached is not None and cached[0] == run.version:
            return cached[1]
        tree = RunTree.build(run, self._formats, self.step_digits, raw=self._raw)
        tree["children"]["llms.txt"] = self._guide
        self._state.run_trees[run.run_parts] = (run.version, tree)
        return tree

    def _open_file(
        self, file_handles: tuple[object, ...]
    ) -> tuple[bytes | _VerifiedSpan, float] | None:
        handle = file_handles[0] if file_handles else None
        if not isinstance(handle, int):
            return None
        with self._lock:
            return self._state.open_files.get(handle)

    def _node_size(
        self,
        node: dict[str, Any],
        key: RenderKey,
        *,
        checked_paths: frozenset[Path] | None = None,
    ) -> int:
        size = _unrendered_size(node, self._allowed, checked_paths)
        return size if size is not None else len(self._node_bytes(node, key))

    def _node_span(self, node: dict[str, Any]) -> _VerifiedSpan | None:
        if node["type"] not in {"binary", "graph"}:
            return None
        entry: BinaryEntry | GraphEntry = node["entry"]
        blob = entry.blob
        if blob.span is None or not blob.range_checksums:
            return None
        if not self._allowed(entry.source_path):
            raise fuse_error(ENOENT)
        try:
            file = FileIO(
                entry.source_path,
                "rb",
                opener=lambda path, flags: os.open(path, flags | os.O_NONBLOCK),
            )
        except OSError as error:
            raise fuse_error(EIO) from error
        try:
            if not self._allowed(entry.source_path):
                raise fuse_error(ENOENT)
            return _VerifiedSpan(
                file,
                blob.frame,
                blob.span,
                blob.range_checksums,
                self._state.frame_validations,
            )
        except BaseException:
            file.close()
            raise

    def _node_bytes(self, node: dict[str, Any], key: RenderKey) -> bytes:
        # Sidecars are files on disk that can change without a new run
        # version, so they are read each time.
        if node["type"] == "sidecar":
            if not self._allowed(node["entry"].source_path):
                raise fuse_error(ENOENT)
            return render_node(node)
        data = self._renders.get(key)
        if data is None:
            data = _render_miss(
                node,
                self._allowed,
                self._state.guard_plans,
                self._state.index.mountpoint,
            )
            self._renders.put(key, data)
        else:
            index = self._state.index
            _NodeSource.require_readable(
                node,
                plans=self._state.guard_plans,
                mountpoint=index.mountpoint,
                storage=index.storage,
            )
        if isinstance(node, _CachedLeaf) and node.size is None:
            with self._lock:
                node.size = len(data)
        return data

    def _allowed(self, path: Path) -> bool:
        plans = self._state.guard_plans
        if plans is None:
            return True
        if len(str(path)) > _GUARD_PLAN_MAX_PATH_CHARS:
            return self._allowed_resolved(path)
        plan = plans(path)
        if plan is not None:
            prefixes, allowed_if_no_links = plan
            if allowed_if_no_links and _plain_path(path):
                return True
            for prefix in prefixes:
                try:
                    mode = os.lstat(prefix).st_mode
                except (FileNotFoundError, NotADirectoryError):
                    continue
                except (OSError, ValueError):
                    return False
                if stat.S_ISLNK(mode):
                    return self._allowed_resolved(path, first_link=Path(prefix))
            return allowed_if_no_links
        return self._allowed_resolved(path)

    def _allowed_resolved(self, path: Path, *, first_link: Path | None = None) -> bool:
        """Follow components from the root or resume at a detected symlink."""
        blocked = (self._state.index.mountpoint, self._state.index.storage)
        if first_link is None:
            pending = list(reversed(path.absolute().parts[1:]))
            current = Path("/")
            links = 0
        else:
            pending = list(reversed(path.parts[len(first_link.parts) :]))
            current = self._queue_link_target(first_link, pending)
            links = 1
        while pending:
            part = pending.pop()
            if part == "..":
                current = current.parent
                continue
            current /= part
            if current in blocked:
                return False
            try:
                mode = os.lstat(current).st_mode
            except (FileNotFoundError, NotADirectoryError):
                continue
            except (OSError, ValueError):
                return False
            if stat.S_ISLNK(mode):
                links += 1
                if links > 40:
                    return False
                current = self._queue_link_target(current, pending)
        return True

    @staticmethod
    def _queue_link_target(link: Path, pending: list[str]) -> Path:
        """Prepend the current link target and return its starting directory."""
        target = link.readlink()
        absolute = target.is_absolute()
        pending.extend(reversed(target.parts[int(absolute) :]))
        return Path("/") if absolute else link.parent

    def _activate(self, parts: tuple[str, ...]) -> frozenset[Path] | None:
        """Refresh a run and return the paths verified by a fresh sibling match."""
        run = self._ensure_run(parts)
        local = self._state.directories[parts]
        previous = local.scanned_at
        if previous is not None and self.refresh_age_seconds == inf:
            return None
        now = _source_clock()
        scan = previous is None or now - previous >= self.refresh_age_seconds
        reuse = local.reusable(now, self.source_check_seconds)
        verified = local.match(now) if not scan and not reuse else None
        if verified is not None:
            run.refreshed_at = time.time()
        if not scan and (reuse or verified is not None):
            if local.owner != parts or local.pieces or run.sidecars:
                self._refresh_sidecars(run, None)
            return verified
        self._refresh_local(run, local, now, scan=scan, reuse=reuse)
        return None

    def _refresh_local(
        self, run: RunCache, local: _Directory, now: float, *, scan: bool, reuse: bool
    ) -> None:
        """Refresh runs and branch visibility after the cached checks miss."""
        first_scan = local.scanned_at is None
        directory = self.source.joinpath(*run.run_parts)
        entries = None
        if scan:
            entries = _Paths.mount_entries(directory, self._allowed)
            local.scanned_at = now
        files, entries = self._run_sources(run, directory, entries)
        if reuse and run.files.keys() == set(files):
            self._refresh_sidecars(run, entries)
        else:
            if self._refresh_run(run, files, entries):
                self._log("finished parsing TensorBoard files")
            local.checked_at = now
        if first_scan and not run.files and not local.children:
            self._invalidate_tree()

    def _scan_branches(
        self,
        parts: tuple[str, ...],
        entries: list[os.DirEntry[str]],
        owner: tuple[str, ...] | None,
    ) -> list[SidecarEntry]:
        """Keep log branches visible and collect sidecars from hidden branches."""
        directory = self.source.joinpath(*parts)
        local = self._state.directories[parts]
        run_dir = None if owner is None else self.source.joinpath(*owner)
        names, sidecars = SidecarScan.branches(
            directory,
            entries,
            local.owned,
            run_dir,
            self._allowed,
            self._state.index.mountpoint,
        )
        self._update_children(parts, {(*parts, name) for name in names})
        return sidecars

    def _update_children(
        self, parts: tuple[str, ...], children: set[tuple[str, ...]]
    ) -> None:
        local = self._state.directories[parts]
        old = local.children
        restored = self._state.restore_branches(children & old)
        if old == children and not restored:
            return
        local.children = children
        for child in children - old:
            self._state.directories[child] = _Directory()
        for removed in old - children:
            self._forget(removed)
        self._invalidate_tree()

    def _run_sources(
        self,
        run: RunCache,
        directory: Path,
        entries: list[os.DirEntry[str]] | None,
    ) -> tuple[list[Path], list[os.DirEntry[str]] | None]:
        """Prefilter private storage only when refresh uses native stat without a mount."""
        if entries is None:
            index = self._state.index
            if index.mountpoint is None and index.storage is not None:
                return [path for path in run.files if self._allowed(path)], None
            return list(run.files), None
        files = _Paths.event_files(
            directory, entries, mountpoint=self._state.index.mountpoint
        )
        event_names = {path.name for path in files}
        return files, [entry for entry in entries if entry.name not in event_names]

    def _forget(self, parts: tuple[str, ...]) -> None:
        local = self._state.directories[parts]
        for child in local.children:
            self._forget(child)
        self._clear_piece(parts)
        del self._state.directories[parts]
        run = self._state.runs.pop(parts, None)
        if run is not None:
            self._drop_run_cache(run)

    def _ensure_run(self, run_parts: tuple[str, ...]) -> RunCache:
        if run_parts not in self._state.directories:
            raise fuse_error(ENOENT)
        if run_parts not in self._state.runs:
            self._state.runs[run_parts] = RunCache(run_parts, version=self._version())
        return self._state.runs[run_parts]

    def _version(self) -> int:
        version = self._state.next_version
        self._state.next_version += 1
        return version

    def _drop_run_cache(self, run: RunCache) -> None:
        self._state.run_trees.pop(run.run_parts, None)
        self._renders.expire(run.version)

    def _change_run_version(self, run: RunCache, *, removed: bool = False) -> None:
        self._drop_run_cache(run)
        run.version = self._version()
        self._invalidate_tree(parts=None if removed else run.run_parts)

    def _refresh_run(
        self,
        run: RunCache,
        files: list[Path],
        entries: list[os.DirEntry[str]] | None,
    ) -> bool:
        """Check known files on access; discover new files only with a fresh listing."""
        now = time.time()
        local = self._state.directories[run.run_parts]
        local.siblings = None
        options = self._state.index
        had_files = bool(run.files)
        changed, base = _RunData.refresh_event_files(
            run,
            files,
            force=False,
            log=logger.info if self.log_to_screen else None,
            options=options,
        )
        if changed or entries is not None:
            local.siblings = _SiblingSnapshot.capture(run, options)
        self._refresh_sidecars(run, entries)
        run.refreshed_at = now
        if changed:
            self._publish_changes(run, base, removed=had_files and not run.files)
        return changed

    def _publish_changes(
        self, run: RunCache, before: EventFileCache | None, *, removed: bool
    ) -> None:
        """Extend the last source only when its previous tree and index agree."""
        cached = self._state.run_trees.get(run.run_parts)
        self._change_run_version(run, removed=removed)
        if cached is None or before is None or before.source_path != max(run.files):
            return
        after = run.files[before.source_path]
        tree = RunTree.append(
            cached[1],
            run,
            before,
            after,
            self._formats,
            self.step_digits,
            raw=self._raw,
        )
        if tree is not None:
            self._state.run_trees[run.run_parts] = (run.version, tree)

    def _clear_piece(self, parts: tuple[str, ...]) -> None:
        local = self._state.directories[parts]
        if local.owner is None:
            return
        pieces = self._state.directories[local.owner].pieces
        if pieces.pop(parts, ()):
            self._publish_sidecars(local.owner)
        local.owner = None

    def _publish_sidecars(self, owner: tuple[str, ...]) -> None:
        run = self._state.runs[owner]
        pieces = self._state.directories[owner].pieces
        sidecars = tuple(item for key in sorted(pieces) for item in pieces[key])
        if run.sidecars != sidecars:
            run.sidecars = sidecars
            self._change_run_version(run)

    def _refresh_sidecars(
        self, run: RunCache, entries: list[os.DirEntry[str]] | None
    ) -> None:
        parts = run.run_parts
        local = self._state.directories[parts]
        owner = next(
            (
                parts[:i]
                for i in range(len(parts), -1, -1)
                if parts[:i] in self._state.runs and self._state.runs[parts[:i]].files
            ),
            None,
        )
        if owner != local.owner:
            self._move_group_sidecars(parts, local.owner, owner)
            self._clear_piece(parts)
            local.owner = owner
        if owner is None:
            local.owned.clear()
            if entries is not None:
                self._scan_branches(parts, entries, None)
            return
        pieces = self._state.directories[owner].pieces
        if entries is not None:
            scan = SidecarScan(
                self.source.joinpath(*parts),
                self.source.joinpath(*owner),
                self._allowed,
            )
            direct = scan.collect(entries)
            local.owned = scan.owned
            files = (*direct, *self._scan_branches(parts, entries, owner))
            if files:
                pieces[parts] = files
            else:
                pieces.pop(parts, None)
        if owner == parts:
            for group, files in pieces.items():
                if group != parts or entries is None:
                    pieces[group] = SidecarScan.restat(files, self._allowed)
        self._publish_sidecars(owner)

    def _move_group_sidecars(
        self,
        parts: tuple[str, ...],
        old: tuple[str, ...] | None,
        owner: tuple[str, ...] | None,
    ) -> None:
        if old is None:
            return
        pieces = self._state.directories[old].pieces
        groups = [
            group for group in pieces if group != parts and group[: len(parts)] == parts
        ]
        for group in groups:
            items = pieces.pop(group)
            self._state.directories[group].owner = owner
            if owner is not None:
                self._state.directories[owner].pieces[group] = SidecarScan.rebase(
                    items, self.source.joinpath(*owner)
                )
        if groups:
            self._publish_sidecars(old)

    def _log(self, message: str, *args: object) -> None:
        if self.log_to_screen:
            logger.info(message, *args)

    def _cached_tree(self) -> dict[str, Any]:
        snapshot = self._state.tree
        root = snapshot.root
        if root is not None and not snapshot.dirty:
            return root
        generation = snapshot.generation
        if root is None:
            tree = self._build_tree()
        else:
            changes = {
                parts: self._visible_run(parts) for parts in tuple(snapshot.dirty)
            }
            tree = self._run_places.update(root, changes)
        snapshot.dirty.clear()
        snapshot.root = tree if generation == snapshot.generation else None
        return tree

    def _invalidate_tree(self, *, parts: tuple[str, ...] | None = None) -> None:
        """Retain the last snapshot only while its folder placement is valid."""
        self._renders.expire(None)
        snapshot = self._state.tree
        snapshot.leaves.clear()
        snapshot.generation += 1
        if (
            parts
            and snapshot.root is not None
            and not self._state.directories[parts].children
            and self._run_places.can_update(parts)
        ):
            snapshot.dirty.add(parts)
        else:
            snapshot.root = None
            snapshot.dirty.clear()
