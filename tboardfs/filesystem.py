from dataclasses import dataclass, field
from errno import EIO, EISDIR, ENOENT, ENOTDIR, EROFS
from io import FileIO
import os
from pathlib import Path
import stat
from threading import RLock
import time
from typing import Any
from collections.abc import Callable

from loguru import logger

from tboardfs.constants import (
    DEFAULT_CACHE_SIZE_MB,
    DEFAULT_DISCOVERY_SECONDS,
    DEFAULT_SCALAR_FORMATS,
)
from tboardfs.errors import fuse_error
from tboardfs.file_times import FileTimes
from tboardfs.file_status import FileIdentity
from tboardfs.indexer import _EventIndexer
from tboardfs.model import BinaryEntry, GraphEntry, RunCache, SidecarEntry
from tboardfs.open3d_asset import OPEN3D_PLUGIN, Open3DAsset
from tboardfs.paths import _DirectoryNode, _Paths
from tboardfs.run_places import RunPlaces
from tboardfs.run_tree import build_run_tree
from tboardfs.sidecar_scan import SidecarScan
from tboardfs.render_cache import RenderCache, RenderKey
from tboardfs.render_node import render_node
from tboardfs.verified_span import _FrameValidations, _VerifiedSpan


@dataclass
class _Directory:
    """Local listing deadline, browseable children, and their sidecar ownership."""

    scanned_at: float | None = None
    children: set[tuple[str, ...]] = field(default_factory=set)
    owned: set[Path] = field(default_factory=set)
    owner: tuple[str, ...] | None = None
    pieces: dict[tuple[str, ...], tuple[SidecarEntry, ...]] = field(
        default_factory=dict
    )


@dataclass
class _FsState:
    """Mutable TensorBoardFS cache state.

    :ivar runs: run caches by virtual run path
    :ivar directories: local discovery state by source directory
    :ivar tree_cache: generation and virtual tree put together from run trees
    :ivar tree_generation: current virtual tree generation
    :ivar run_trees: run version and virtual tree by run path
    :ivar next_version: next unused run data version
    :ivar open_files: content and mtime kept for each open file handle
    :ivar next_handle: number for the next open file handle
    """

    runs: dict[tuple[str, ...], RunCache] = field(default_factory=dict)
    directories: dict[tuple[str, ...], _Directory] = field(default_factory=dict)
    mountpoint: Path | None = None
    tree_cache: tuple[int, dict[str, Any]] | None = None
    tree_generation: int = 0
    run_trees: dict[tuple[str, ...], tuple[int, dict[str, Any]]] = field(
        default_factory=dict
    )
    next_version: int = 1
    frame_validations: _FrameValidations = field(default_factory=_FrameValidations)
    open_files: dict[int, tuple[bytes | _VerifiedSpan, float]] = field(
        default_factory=dict
    )
    next_handle: int = 1


class TensorBoardFS:  # pcr: noqa: PCR010
    """Read-only virtual filesystem over TensorBoard event files.

    FUSE calls the callbacks from several threads. Discovery, refresh and
    tree lookup run under ``_lock``; rendering a node runs outside it,
    because nodes and event-file caches are not changed after creation.
    ``open`` pins a source descriptor or rendered bytes until ``release``.
    Rendered bytes are cached by the data versions of the runs on the path,
    so a change in one run makes only its own files render again.

    :ivar source: source log directory
    :ivar step_digits: zero-padding width for step filenames
    :ivar refresh_age_seconds: minimum age before cache refresh
    :ivar scalar_formats: enabled scalar export formats
    :ivar log_to_screen: whether operations log status messages
    :ivar owner: user and group ids that own every file and directory
    :ivar _renders: rendered file bytes by run versions and path
    :ivar _state: mutable run discovery and tree state
    :ivar _lock: serializes every change and read of ``_state``
    :ivar _run_places: places the run trees in one tree, and keeps the run
        folders with a suffix so each is logged once
    """

    source: Path
    step_digits: int
    refresh_age_seconds: float
    scalar_formats: tuple[str, ...]
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
        scalar_formats: str | list[str] | tuple[str, ...] = DEFAULT_SCALAR_FORMATS,
        log_to_screen: bool = False,
        cache_size_mb: int = DEFAULT_CACHE_SIZE_MB,
        owner: tuple[int, int] = (0, 0),
        mountpoint: str | Path | None = None,
    ) -> None:
        self.source = Path(source)
        self.step_digits = step_digits
        self.refresh_age_seconds = refresh_age_seconds
        self.scalar_formats = (
            _Paths.normalize_formats(scalar_formats) or DEFAULT_SCALAR_FORMATS
        )
        self.log_to_screen = log_to_screen
        self.owner = owner
        self._renders = RenderCache(cache_size_mb * 1024 * 1024)
        self._state = _FsState(
            mountpoint=Path(mountpoint).resolve() if mountpoint else None
        )
        self._lock = RLock()
        self._run_places = RunPlaces()
        self._state.directories[()] = _Directory()
        self._ensure_run(())

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
        opened = self._open_file(file_handles)
        mode, nlink = stat.S_IFREG | 0o444, 1
        if opened is not None:
            content, mtime = opened
            size = content.size if isinstance(content, _VerifiedSpan) else len(content)
        else:
            node, key = self._locate(path)
            mtime = FileTimes.of(node)
            if node["type"] == "dir":
                mode, nlink, size = stat.S_IFDIR | 0o555, 2, 0
            else:
                size = self._node_size(node, key)
        uid, gid = self.owner
        return {
            "st_mode": mode,
            "st_nlink": nlink,
            "st_size": size,
            "st_blocks": (size + 511) // 512,
            "st_ctime": mtime,
            "st_mtime": mtime,
            "st_atime": mtime,
            "st_uid": uid,
            "st_gid": gid,
        }

    def readdir(self, path: str, *file_handles: object) -> list[str]:
        """Return virtual directory entries."""
        del file_handles
        with self._lock:
            node = self._lookup(path)
            if not isinstance(node, _DirectoryNode):
                raise fuse_error(ENOTDIR)
            names = node.sorted_names()
        return [".", "..", *names]

    def open(self, path: str, flags: int) -> int:
        """Open a virtual file and pin its current version for the handle."""
        del flags
        node, key = self._locate(path)
        if node["type"] == "dir":
            raise fuse_error(EISDIR)
        mtime = FileTimes.of(node)
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

    def _locate(self, path: str) -> tuple[dict[str, Any], RenderKey]:
        """Return the node at ``path`` and its generation-scoped render key.

        The key is read under the same lock as the node, so both match.
        """
        parts = _Paths.path_parts(path)
        with self._lock:
            tree = self._cached_tree()
            depth = len(parts)
            disk_parts = self._run_places.disk_parts
            while (owner := disk_parts.get(parts[:depth])) is None:
                depth -= 1
            while True:
                self._activate(owner)
                refreshed = self._cached_tree()
                if refreshed is not tree:
                    tree = refreshed
                    disk_parts = self._run_places.disk_parts
                if depth == len(parts):
                    break
                child_owner = disk_parts.get(parts[: depth + 1])
                if child_owner is None:
                    break
                owner = child_owner
                depth += 1
            node = tree
            for part in parts:
                if node["type"] != "dir" or part not in node["children"]:
                    raise FileNotFoundError(ENOENT, os.strerror(ENOENT), path)
                node = node["children"][part]
            version = None
            if node["type"] != "name_map":
                while (owner := disk_parts.get(parts[:depth])) is None:
                    depth -= 1
                version = self._state.runs[owner].version
            return node, (self._renders.scope(version), parts)

    def _build_tree(self) -> dict[str, Any]:
        tree = self._run_places.build(
            {
                run_parts: (
                    self._run_tree(self._state.runs[run_parts])
                    if run_parts in self._state.runs
                    and (
                        self._state.runs[run_parts].files
                        or (not run_parts and not self._state.directories[()].children)
                    )
                    else {"type": "dir", "children": {}}
                )
                for run_parts in self._state.directories
            }
        )
        # The run trees have their times, so only the run folders get them.
        FileTimes.stamp_dirs(tree)
        return tree

    def _run_tree(self, run: RunCache) -> dict[str, Any]:
        cached = self._state.run_trees.get(run.run_parts)
        if cached is not None and cached[0] == run.version:
            return cached[1]
        tree = build_run_tree(run, self.scalar_formats, self.step_digits)
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

    def _node_size(self, node: dict[str, Any], key: RenderKey) -> int:
        if node["type"] == "raw":
            return int(node["frame"].size)
        if node["type"] in {"binary", "graph"}:
            stored: BinaryEntry | GraphEntry = node["entry"]
            if not self._allowed(stored.source_path):
                raise fuse_error(ENOENT)
            return stored.blob.size
        # A sidecar can be large, so its size comes from the disk.
        if node["type"] == "sidecar":
            entry: SidecarEntry = node["entry"]
            if not self._allowed(entry.source_path):
                raise fuse_error(ENOENT)
            if entry.tab != OPEN3D_PLUGIN:
                return entry.source_path.stat().st_size
            asset = Open3DAsset.stat(entry.source_path)
            if asset is None:
                raise fuse_error(ENOENT)
            return asset.size
        return len(self._node_bytes(node, key))

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
            entry = node.get("entry")
            source = node.get("source") if entry is None else entry.source_path
            if source is not None and not self._allowed(source):
                raise fuse_error(ENOENT)
            data = render_node(node)
            self._renders.put(key, data)
        return data

    def _allowed(self, path: Path) -> bool:
        if self._state.mountpoint is None:
            return True
        pending = list(reversed(path.absolute().parts[1:]))
        current = Path("/")
        links = 0
        while pending:
            part = pending.pop()
            if part == "..":
                current = current.parent
                continue
            current /= part
            if current == self._state.mountpoint:
                return False
            if current.is_symlink():
                links += 1
                if links > 40:
                    return False
                target = current.readlink()
                current = Path("/") if target.is_absolute() else current.parent
                pending.extend(reversed(target.parts[int(target.is_absolute()) :]))
        return True

    def _activate(self, parts: tuple[str, ...]) -> None:
        directory = self.source.joinpath(*parts)
        run = self._ensure_run(parts)
        local = self._state.directories[parts]
        previous = local.scanned_at
        if previous is not None and self.refresh_age_seconds == float("inf"):
            return
        now = time.time()
        entries = None
        if previous is None or now - previous >= self.refresh_age_seconds:
            entries = (
                [
                    entry
                    for entry in _Paths.entries(directory)
                    if not entry.is_symlink() or self._allowed(Path(entry.path))
                ]
                if self._allowed(directory)
                else []
            )
            local.scanned_at = now
        if self._refresh_run(run, entries):
            self._log("finished parsing TensorBoard files")
        if entries is not None:
            children = {
                (*parts, name)
                for name in self._child_names(directory, entries, local.owned)
            }
            old = local.children
            if old != children:
                local.children = children
                for child in children - old:
                    self._state.directories[child] = _Directory()
                for removed in old - children:
                    self._forget(removed)
                self._invalidate_tree()

    @staticmethod
    def _child_names(
        directory: Path, entries: list[os.DirEntry[str]], owned: set[Path]
    ) -> list[str]:
        children = [
            entry
            for entry in entries
            if _Paths.is_dir(entry) and Path(entry.path) not in owned
        ]
        if not any(entry.is_symlink() for entry in children):
            return [entry.name for entry in children]
        parent = directory.resolve()
        seen = {parent, *(p.resolve() for p in directory.parents)}
        names = []
        for entry in children:
            target = (
                Path(entry.path).resolve()
                if entry.is_symlink()
                else parent / entry.name
            )
            if target in seen or parent.is_relative_to(target):
                continue
            seen.add(target)
            names.append(entry.name)
        return names

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

    def _change_run_version(self, run: RunCache) -> None:
        self._drop_run_cache(run)
        run.version = self._version()
        self._invalidate_tree()

    def _refresh_run(
        self,
        run: RunCache,
        entries: list[os.DirEntry[str]] | None,
        *,
        force: bool = False,
    ) -> bool:
        """Check known files on access; discover new files only with a fresh listing."""
        now = time.time()
        run_dir = self.source.joinpath(*run.run_parts)
        files = (
            _Paths.event_files(run_dir, entries)
            if entries is not None
            else [path for path in run.files if self._allowed(path)]
        )
        log_parse = logger.info if self.log_to_screen else None
        changed = _RunData.refresh_event_files(run, files, force=force, log=log_parse)
        self._refresh_sidecars(run, entries)
        run.refreshed_at = now
        if changed:
            self._change_run_version(run)
        return changed

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
            return
        pieces = self._state.directories[owner].pieces
        if entries is not None:
            scan = SidecarScan(
                self.source.joinpath(*parts),
                self.source.joinpath(*owner),
                self._allowed,
            )
            files = scan.collect(entries)
            local.owned = scan.owned
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
        generation = self._state.tree_generation
        cached = self._state.tree_cache
        if cached is not None and cached[0] == generation:
            return cached[1]
        # Keep the generation from before the build: a change during the
        # build must make the next lookup rebuild.
        tree = self._build_tree()
        self._state.tree_cache = (generation, tree)
        return tree

    def _invalidate_tree(self) -> None:
        self._renders.expire(None)
        self._state.tree_generation += 1
        self._state.tree_cache = None


class _RunData:
    """Refresh event-file caches and collect run entries."""

    @staticmethod
    def refresh_event_files(
        run: RunCache,
        files: list[Path],
        *,
        force: bool,
        log: Callable[[str, object], None] | None,
    ) -> bool:
        """Refresh each source independently; never publish an unreadable generation."""
        changed = False
        carry: dict[str, dict[str, Any]] = {}
        for source_path in sorted(files):
            old = run.files.get(source_path)
            try:
                changed = (
                    _RunData.refresh_event_file(
                        run,
                        source_path,
                        force=force
                        or (old is not None and old.progress.initial_metadata != carry),
                        log=log,
                        initial_metadata=carry,
                    )
                    or changed
                )
            except OSError as error:
                if not isinstance(error, FileNotFoundError):
                    logger.warning("cannot read event file {}: {}", source_path, error)
                if run.files.pop(source_path, None) is not None:
                    changed = True
            if source_path in run.files:
                carry = run.files[source_path].progress.effective_metadata
        for deleted in set(run.files) - set(files):
            if log is not None:
                log("event file removed {}", deleted)
            del run.files[deleted]
            changed = True
        return changed

    @staticmethod
    def refresh_event_file(
        run: RunCache,
        source_path: Path,
        *,
        force: bool,
        log: Callable[[str, object], None] | None,
        initial_metadata: dict[str, dict[str, Any]] | None = None,
    ) -> bool:
        """Check a known source; open and parse it only when its status changes."""
        old = run.files.get(source_path)
        if (
            not force
            and old is not None
            and FileIdentity.fingerprint(source_path.stat())
            == old.progress.checked_status
        ):
            return False
        if log is not None:
            log("parsing {}", source_path)
        run.files[source_path] = _EventIndexer.parse_event_file(
            source_path,
            ignore_truncated=True,
            base_cache=None if force else old,
            initial_metadata=initial_metadata,
        )
        return True
