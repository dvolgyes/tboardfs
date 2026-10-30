from dataclasses import dataclass, field
from errno import EISDIR, ENOENT, ENOTDIR, EOPNOTSUPP, EROFS
from fnmatch import fnmatchcase
import os
from pathlib import Path
import stat
from threading import RLock
import time
from typing import Any
from collections.abc import Callable, Hashable, Iterator

from loguru import logger

from tboardfs.constants import (
    DEFAULT_CACHE_SIZE_MB,
    DEFAULT_SCALAR_FORMATS,
    FIXED_TABS,
    TENSORBOARD_EVENT_GLOB,
)
from tboardfs.errors import fuse_error
from tboardfs.extract import _extract_graph_blob, extract_binary_blob
from tboardfs.indexer import _EventIndexer
from tboardfs.model import (
    BinaryEntry,
    EventFileCache,
    GraphEntry,
    JsonEntry,
    RunCache,
    SidecarEntry,
)
from tboardfs.npy_export import array_to_npy
from tboardfs.paths import (
    _Paths,
    find_tensorboard_files,
)
from tboardfs.plugin_files import add_typed_plugin_files
from tboardfs.projector import _ProjectorConfig
from tboardfs.render_cache import RenderCache
from tboardfs.restarts import _RestartPurge
from tboardfs.tables import (
    _TableExport,
    _TableMerge,
    export_table,
)
from tboardfs.text_table import _TextTable


def _materialize_node_bytes(node: dict[str, Any]) -> bytes:
    """Return the byte representation for one virtual file node."""
    node_type = node["type"]
    if node_type == "bytes":
        return bytes(node["data"])
    if node_type == "obj":
        return str(node["data"]).encode()
    if node_type == "npy":
        return array_to_npy(node["data"])
    if node_type == "json":
        return _TableExport.json_document(node["data"])
    if node_type in {"scalar", "table"}:
        return export_table(node["series"], node["format"])
    if node_type == "binary":
        return extract_binary_blob(node["entry"])
    if node_type == "graph":
        return _extract_graph_blob(node["entry"])
    if node_type == "sidecar":
        return bytes(node["entry"].source_path.read_bytes())
    raise fuse_error(EOPNOTSUPP)


def _build_run_virtual_tree(
    run: RunCache,
    scalar_formats: tuple[str, ...],
    step_digits: int,
) -> dict[str, Any]:
    """Return virtual tab directories and files for one TensorBoard run.

    Projector/profile sidecars come from ``run.sidecars``, which only a
    mounted log directory fills.
    """
    run_node: dict[str, Any] = {"type": "dir", "children": {}}
    for tab in FIXED_TABS:
        run_node["children"].setdefault(tab, {"type": "dir", "children": {}})
    _RunTree.add_run_files(run_node, run, scalar_formats, step_digits)
    return run_node


@dataclass
class _FsState:
    """Mutable TensorBoardFS cache state.

    :ivar runs: run caches by virtual run path
    :ivar known_run_parts: discovered run path parts
    :ivar last_event_files: event files seen during last discovery
    :ivar last_discovery_at: timestamp of the last discovery pass
    :ivar tree_cache: generation and virtual tree put together from run trees
    :ivar tree_generation: current virtual tree generation
    :ivar run_trees: run version and virtual tree by run path
    :ivar next_version: next unused run data version
    :ivar open_files: bytes and mtime kept for each open file handle
    :ivar next_handle: number for the next open file handle
    """

    runs: dict[tuple[str, ...], RunCache] = field(default_factory=dict)
    known_run_parts: set[tuple[str, ...]] = field(default_factory=set)
    last_event_files: set[Path] | None = None
    last_discovery_at: float = 0.0
    tree_cache: tuple[int, dict[str, Any]] | None = None
    tree_generation: int = 0
    run_trees: dict[tuple[str, ...], tuple[int, dict[str, Any]]] = field(
        default_factory=dict
    )
    next_version: int = 1
    open_files: dict[int, tuple[bytes, float]] = field(default_factory=dict)
    next_handle: int = 1


class TensorBoardFS:  # pcr: noqa: PCR010
    """Read-only virtual filesystem over TensorBoard event files.

    FUSE calls the callbacks from several threads. Discovery, refresh and
    tree lookup run under ``_lock``; rendering a node runs outside it,
    because nodes and event-file caches are not changed after creation.
    ``open`` keeps the rendered bytes until ``release``, so every read and
    stat through one handle sees one version while the run grows.
    Rendered bytes are cached by the data versions of the runs on the path,
    so a change in one run makes only its own files render again.

    :ivar source: source log directory
    :ivar step_digits: zero-padding width for step filenames
    :ivar refresh_age_seconds: minimum age before cache refresh
    :ivar scalar_formats: enabled scalar export formats
    :ivar log_to_screen: whether operations log status messages
    :ivar _renders: rendered file bytes by run versions and path
    :ivar _state: mutable run discovery and tree state
    :ivar _lock: serializes every change and read of ``_state``
    """

    source: Path
    step_digits: int
    refresh_age_seconds: float
    scalar_formats: tuple[str, ...]
    log_to_screen: bool
    _renders: RenderCache
    _state: "_FsState"
    _lock: RLock

    def __init__(
        self,
        source: str | Path,
        *,
        step_digits: int = 6,
        refresh_age_seconds: float = 60.0,
        scalar_formats: str | list[str] | tuple[str, ...] = DEFAULT_SCALAR_FORMATS,
        log_to_screen: bool = False,
        cache_size_mb: int = DEFAULT_CACHE_SIZE_MB,
    ) -> None:
        self.source = Path(source)
        self.step_digits = step_digits
        self.refresh_age_seconds = refresh_age_seconds
        self.scalar_formats = (
            _Paths.normalize_formats(scalar_formats) or DEFAULT_SCALAR_FORMATS
        )
        self.log_to_screen = log_to_screen
        self._renders = RenderCache(cache_size_mb * 1024 * 1024)
        self._state = _FsState()
        self._lock = RLock()
        self._discover_runs(force=True)

    def getattr(self, path: str, *file_handles: object) -> dict[str, Any]:
        """Return stat data for a virtual path, or for an open handle."""
        opened = self._open_file(file_handles)
        if opened is not None:
            data, mtime = opened
            return _FsNode.stat_dict(
                stat.S_IFREG | 0o444, 1, len(data), time.time(), mtime
            )
        try:
            node, key = self._locate(path)
        except FileNotFoundError:
            raise fuse_error(ENOENT)
        now = time.time()
        if node["type"] == "dir":
            return _FsNode.stat_dict(
                stat.S_IFDIR | 0o555, 2, 0, now, node.get("mtime", now)
            )
        data = self._node_bytes(node, key)
        return _FsNode.stat_dict(
            stat.S_IFREG | 0o444, 1, len(data), now, node.get("mtime", now)
        )

    def readdir(self, path: str, *file_handles: object) -> list[str]:
        """Return virtual directory entries."""
        del file_handles
        node = self._lookup(path)
        if node["type"] != "dir":
            raise fuse_error(ENOTDIR)
        return [".", "..", *sorted(node["children"])]

    def open(self, path: str, flags: int) -> int:
        """Open a virtual file and keep its current bytes for the handle."""
        del flags
        try:
            node, key = self._locate(path)
        except FileNotFoundError:
            raise fuse_error(ENOENT)
        if node["type"] == "dir":
            raise fuse_error(EISDIR)
        opened = (self._node_bytes(node, key), node.get("mtime", time.time()))
        with self._lock:
            handle = self._state.next_handle
            self._state.next_handle += 1
            self._state.open_files[handle] = opened
        return handle

    def release(self, path: str, handle: int) -> int:
        """Free the bytes kept for an open handle."""
        del path
        with self._lock:
            self._state.open_files.pop(handle, None)
        return 0

    def read(self, path: str, size: int, offset: int, *file_handles: object) -> bytes:
        """Read a byte range from an open handle, or from the current file."""
        opened = self._open_file(file_handles)
        if opened is not None:
            return opened[0][offset : offset + size]
        node, key = self._locate(path)
        if node["type"] == "dir":
            raise fuse_error(EISDIR)
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

    def _locate(self, path: str) -> tuple[dict[str, Any], Hashable]:
        """Return the node at ``path`` and the cache key of its bytes.

        The key names the versions of all runs that can add files at the
        path; it is read under the same lock as the node, so both match.
        """
        parts = _Paths.path_parts(path)
        with self._lock:
            self._discover_runs()
            self._refresh_runs()
            node = self._cached_tree()
            for part in parts:
                if node["type"] != "dir" or part not in node["children"]:
                    raise FileNotFoundError(path)
                node = node["children"][part]
            runs = self._state.runs
            versions = tuple(
                runs[parts[:end]].version
                for end in range(len(parts) + 1)
                if parts[:end] in runs
            )
            return node, (versions, parts)

    def _build_tree(self) -> dict[str, Any]:
        root: dict[str, Any] = {"type": "dir", "children": {}}
        for run_parts in sorted(self._state.known_run_parts):
            run_tree = self._run_tree(self._ensure_run(run_parts))
            # Copy each directory on the way: a nested run must not add its
            # entries to the cached tree of its parent run.
            node = root
            for part in run_parts:
                child = node["children"].get(part, {"type": "dir", "children": {}})
                node["children"][part] = node = {
                    **child,
                    "children": dict(child["children"]),
                }
            node["children"].update(run_tree["children"])
        return root

    def _run_tree(self, run: RunCache) -> dict[str, Any]:
        cached = self._state.run_trees.get(run.run_parts)
        if cached is not None and cached[0] == run.version:
            return cached[1]
        tree = _build_run_virtual_tree(run, self.scalar_formats, self.step_digits)
        self._state.run_trees[run.run_parts] = (run.version, tree)
        return tree

    def _open_file(
        self, file_handles: tuple[object, ...]
    ) -> tuple[bytes, float] | None:
        handle = file_handles[0] if file_handles else None
        if not isinstance(handle, int):
            return None
        with self._lock:
            return self._state.open_files.get(handle)

    def _node_bytes(self, node: dict[str, Any], key: Hashable) -> bytes:
        # Sidecars are files on disk that can change without a new run
        # version, so they are read each time.
        if node["type"] == "sidecar":
            return _materialize_node_bytes(node)
        data = self._renders.get(key)
        if data is None:
            data = _materialize_node_bytes(node)
            self._renders.put(key, data)
        return data

    def _discover_runs(self, *, force: bool = False) -> None:
        now = time.time()
        if (
            not force
            and self._state.last_event_files is not None
            and now - self._state.last_discovery_at < self.refresh_age_seconds
        ):
            return
        files = find_tensorboard_files(self.source)
        file_set = set(files)
        should_log = file_set != self._state.last_event_files
        if should_log:
            self._log("finding TensorBoard files in {}", self.source)
        self._state.known_run_parts = {
            _Paths.run_parts_for(self.source, path) for path in files
        }
        if not self._state.known_run_parts:
            self._state.known_run_parts = {()}
        for stale_run in set(self._state.runs) - self._state.known_run_parts:
            del self._state.runs[stale_run]
            self._state.run_trees.pop(stale_run, None)
        for run_parts in self._state.known_run_parts:
            self._ensure_run(run_parts)
        if should_log:
            self._log(
                "found {} event files in {} runs",
                len(files),
                len(self._state.known_run_parts),
            )
        self._state.last_event_files = file_set
        self._state.last_discovery_at = now
        if should_log:
            self._invalidate_tree()

    def _ensure_run(self, run_parts: tuple[str, ...]) -> RunCache:
        if run_parts not in self._state.known_run_parts:
            raise fuse_error(ENOENT)
        if run_parts not in self._state.runs:
            self._state.runs[run_parts] = RunCache(run_parts, version=self._version())
        return self._state.runs[run_parts]

    def _version(self) -> int:
        version = self._state.next_version
        self._state.next_version += 1
        return version

    def _refresh_run(self, run: RunCache, *, force: bool = False) -> bool:
        now = time.time()
        if not _RunData.should_refresh(run, now, self.refresh_age_seconds, force):
            return False
        run_dir = self.source.joinpath(*run.run_parts)
        files = sorted(
            path for path in run_dir.glob(TENSORBOARD_EVENT_GLOB) if path.is_file()
        )
        log_parse = logger.info if self.log_to_screen else None
        changed = _RunData.refresh_event_files(run, files, force=force, log=log_parse)
        # Sidecars can change after the last event, so they count as run data.
        sidecars = tuple(_RunData.run_sidecars(self.source, run))
        if sidecars != run.sidecars:
            run.sidecars = sidecars
            changed = True
        run.refreshed_at = now
        if changed:
            run.version = self._version()
            self._invalidate_tree()
        return changed

    def _log(self, message: str, *args: object) -> None:
        if self.log_to_screen:
            logger.info(message, *args)

    def _refresh_runs(self, *, force: bool = False) -> None:
        refreshed = False
        for run in list(self._state.runs.values()):
            refreshed = self._refresh_run(run, force=force) or refreshed
        if refreshed:
            self._log("finished parsing TensorBoard files")

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
        self._state.tree_generation += 1
        self._state.tree_cache = None


class _FsNode:
    """Create virtual filesystem nodes."""

    @staticmethod
    def stat_dict(
        mode: int, nlink: int, size: int, now: float, mtime: float
    ) -> dict[str, Any]:
        """Return a FUSE stat dictionary."""
        return {
            "st_mode": mode,
            "st_nlink": nlink,
            "st_size": size,
            "st_ctime": now,
            "st_mtime": mtime,
            "st_atime": now,
        }

    @staticmethod
    def add_file(
        root: dict[str, Any], rel_parts: tuple[str, ...], node: dict[str, Any]
    ) -> None:
        """Add one virtual file to a directory tree."""
        parent = _Paths.mkdirs(root, rel_parts[:-1])
        parent["children"][rel_parts[-1]] = node

    @staticmethod
    def table_node(series: dict[str, Any], fmt: str, mtime: float) -> dict[str, Any]:
        """Return a virtual table node."""
        return {"type": "table", "series": series, "format": fmt, "mtime": mtime}


class _RunTree:
    """Populate virtual run directories from run caches."""

    @staticmethod
    def add_run_files(
        run_node: dict[str, Any],
        run: RunCache,
        scalar_formats: tuple[str, ...],
        step_digits: int,
    ) -> None:
        """Populate one run directory with indexed virtual files."""
        _RunTree.add_scalar_files(run_node, run, scalar_formats)
        _HistogramFiles.add_files(run_node, run, scalar_formats, step_digits)
        _RunTree.add_binary_files(run_node, run, step_digits)
        consumed = add_typed_plugin_files(
            run_node,
            _RunData.run_json_entries(run),
            _RunTree.merged_scalars(run),
            step_digits,
        )
        _RunTree.add_json_files(run_node, run, step_digits, consumed)
        _RunTree.add_graph_files(run_node, run)
        _RunTree.add_sidecar_files(run_node, run)

    @staticmethod
    def add_scalar_files(
        run_node: dict[str, Any], run: RunCache, scalar_formats: tuple[str, ...]
    ) -> None:
        """Add scalar export files for one run."""
        scalars = _TableMerge.merge_series(
            cache.scalars for cache in _RunData.sorted_file_caches(run)
        )
        mtimes = _TableMerge.merge_mtimes(
            cache.scalar_mtimes for cache in _RunData.sorted_file_caches(run)
        )
        for tag, series in scalars.items():
            for fmt in scalar_formats:
                _FsNode.add_file(
                    run_node,
                    ("scalars", *_Paths.tag_parts(tag, fmt)),
                    {
                        "type": "scalar",
                        "series": series,
                        "format": fmt,
                        "mtime": mtimes.get(tag, time.time()),
                    },
                )

    @staticmethod
    def add_binary_files(
        run_node: dict[str, Any], run: RunCache, step_digits: int
    ) -> None:
        """Add lazily extracted binary files for one run."""
        for entry in _RunData.run_binaries(run):
            rel = (
                *_Paths.tag_dir_parts(entry.tag),
                f"{_Paths.step_name(entry.step, step_digits)}.{entry.extension}",
            )
            _FsNode.add_file(
                run_node,
                (entry.kind, *rel),
                {"type": "binary", "entry": entry, "mtime": entry.wall_time},
            )

    @staticmethod
    def add_json_files(
        run_node: dict[str, Any],
        run: RunCache,
        step_digits: int,
        consumed: set[int],
    ) -> None:
        """Add plugin JSON and text files for one run."""
        for entry in _RunData.run_json_entries(run):
            if id(entry) in consumed:
                continue
            tab = entry.tab if entry.tab in FIXED_TABS else "plugins"
            text_nodes = tab == "text" and _TextTable.nodes(entry)
            if text_nodes:
                step = _Paths.step_name(entry.step, step_digits)
                for suffix, node in text_nodes:
                    parts = (tab, *_Paths.tag_dir_parts(entry.tag), f"{step}.{suffix}")
                    _FsNode.add_file(run_node, parts, node)
                continue
            _FsNode.add_file(
                run_node,
                _RunTree.json_entry_parts(tab, entry, step_digits),
                {"type": "json", "data": entry.payload, "mtime": entry.wall_time},
            )

    @staticmethod
    def merged_scalars(run: RunCache) -> dict[str, dict[str, Any]]:
        """Return merged scalar series for one run."""
        return _TableMerge.merge_series(
            cache.scalars for cache in _RunData.sorted_file_caches(run)
        )

    @staticmethod
    def add_graph_files(run_node: dict[str, Any], run: RunCache) -> None:
        """Add graph protobuf files for one run.

        graph.pb is the newest graph, as in TensorBoard. A run with several
        graphs also lists each one as graph_<i>.pb in write order.
        """
        graphs = _RunData.run_graphs(run)
        if not graphs:
            return
        named = [(graphs[-1].name, graphs[-1])]
        if len(graphs) > 1:
            named += [
                (f"graph_{index}.pb", graph) for index, graph in enumerate(graphs)
            ]
        for name, graph in named:
            _FsNode.add_file(
                run_node,
                ("graphs", name),
                {"type": "graph", "entry": graph, "mtime": graph.wall_time},
            )

    @staticmethod
    def add_sidecar_files(run_node: dict[str, Any], run: RunCache) -> None:
        """Add projector/profile sidecar files for one run."""
        for sidecar in run.sidecars:
            _FsNode.add_file(
                run_node,
                (sidecar.tab, *sidecar.rel_parts),
                {
                    "type": "sidecar",
                    "entry": sidecar,
                    "mtime": sidecar.mtime_ns / 1e9,
                },
            )

    @staticmethod
    def json_entry_parts(
        tab: str, entry: JsonEntry, step_digits: int
    ) -> tuple[str, ...]:
        """Return virtual path parts for a plugin JSON entry."""
        filename = f"{_Paths.step_name(entry.step, step_digits)}.json"
        if tab == "plugins":
            plugin = _Paths.safe_component(
                str(entry.payload.get("plugin_name") or "unknown")
            )
            return (tab, plugin, *_Paths.tag_dir_parts(entry.tag), filename)
        return (tab, *_Paths.tag_dir_parts(entry.tag), filename)


class _HistogramFiles:
    """Add per-step histogram and distribution files."""

    @staticmethod
    def add_files(
        run_node: dict[str, Any],
        run: RunCache,
        scalar_formats: tuple[str, ...],
        step_digits: int,
    ) -> None:
        """Add histogram and distribution table files for one run."""
        histograms = _TableMerge.merge_series(
            cache.histograms for cache in _RunData.sorted_file_caches(run)
        )
        mtimes = _TableMerge.merge_mtimes(
            cache.histogram_mtimes for cache in _RunData.sorted_file_caches(run)
        )
        for tag, series in histograms.items():
            steps = _TableMerge.unique_steps(series)
            for fmt in scalar_formats:
                if fmt not in {"json", "tsv", "npz"}:
                    continue
                for step in steps:
                    step_series = _TableMerge.series_for_step(series, step)
                    _HistogramFiles.add_step_table(
                        run_node,
                        {
                            "tab": "histograms",
                            "tag": tag,
                            "step": step,
                            "series": step_series,
                            "fmt": fmt,
                            "mtime": mtimes.get(tag, time.time()),
                        },
                        step_digits,
                    )
                    distribution = _TableMerge.distribution_from_histogram(step_series)
                    _HistogramFiles.add_step_table(
                        run_node,
                        {
                            "tab": "distributions",
                            "tag": tag,
                            "step": step,
                            "series": distribution,
                            "fmt": fmt,
                            "mtime": mtimes.get(tag, time.time()),
                        },
                        step_digits,
                    )

    @staticmethod
    def add_step_table(
        run_node: dict[str, Any], item: dict[str, Any], step_digits: int
    ) -> None:
        """Add one per-step tabular file."""
        rel = (
            item["tab"],
            *_Paths.tag_dir_parts(item["tag"]),
            f"{_Paths.step_name(item['step'], step_digits)}.{item['fmt']}",
        )
        _FsNode.add_file(
            run_node,
            rel,
            _FsNode.table_node(item["series"], item["fmt"], item["mtime"]),
        )


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
        """Refresh event-file caches for a run.

        An unreadable file keeps its previous cache, so one bad file never
        hides the rest of the run or the mount.
        """
        changed = False
        for source_path in files:
            try:
                changed = (
                    _RunData.refresh_event_file(run, source_path, force=force, log=log)
                    or changed
                )
            except OSError as error:
                logger.warning("cannot read event file {}: {}", source_path, error)
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
    ) -> bool:
        """Parse new records of one event file; return true when it changed."""
        old = run.files.get(source_path)
        with source_path.open("rb") as handle:
            size, identity = _EventIndexer.file_identity(handle)
        if old is not None and not old.progress.continues_in(identity):
            old = None
        if not force and old is not None and size == old.progress.last_checked_size:
            return False
        if log is not None:
            log("parsing {}", source_path)
        if not force and old is not None and size >= old.progress.processed_pos:
            run.files[source_path] = _EventIndexer.parse_event_file(
                source_path,
                ignore_truncated=True,
                start_pos=old.progress.processed_pos,
                base_cache=old,
            )
        else:
            run.files[source_path] = _EventIndexer.parse_event_file(
                source_path, ignore_truncated=True
            )
        return True

    @staticmethod
    def sorted_file_caches(run: RunCache) -> list[EventFileCache]:
        """Return run file caches in source-file order, after restart purges."""
        return _RestartPurge.purge([run.files[path] for path in sorted(run.files)])

    @staticmethod
    def should_refresh(
        run: RunCache, now: float, refresh_age_seconds: float, force: bool
    ) -> bool:
        """Return true when a run cache should be refreshed."""
        return (
            force
            or not run.refreshed_at
            or now - run.refreshed_at >= refresh_age_seconds
        )

    @staticmethod
    def run_binaries(run: RunCache) -> list[BinaryEntry]:
        """Return binary entries from all event-file caches."""
        return [
            entry
            for cache in _RunData.sorted_file_caches(run)
            for entry in cache.binaries
        ]

    @staticmethod
    def run_json_entries(run: RunCache) -> list[JsonEntry]:
        """Return plugin JSON entries from all event-file caches."""
        return [
            entry
            for cache in _RunData.sorted_file_caches(run)
            for entry in cache.json_entries
        ]

    @staticmethod
    def run_graphs(run: RunCache) -> list[GraphEntry]:
        """Return graph entries from all event-file caches."""
        return [
            entry
            for cache in _RunData.sorted_file_caches(run)
            for entry in cache.graphs
        ]

    @staticmethod
    def run_sidecars(source: Path, run: RunCache) -> list[SidecarEntry]:
        """Return projector/profile sidecar entries discovered under a run."""
        if not run.files:
            return []
        run_dir = source.joinpath(*run.run_parts)
        sidecars = []
        paths = list(_RunData.sidecar_paths(source, run_dir))
        embeddings = _ProjectorConfig.references(paths)
        for path in paths:
            rel_parts = path.relative_to(run_dir).parts
            tab = _RunData.sidecar_tab(rel_parts)
            if tab is None and path in embeddings:
                config, named = embeddings[path]
                config_parts = config.relative_to(run_dir).parts
                tab, rel_parts = "projector", (*config_parts[:-1], *named)
            if tab is None:
                continue
            try:
                status = path.stat()
            except FileNotFoundError:
                continue  # removed since the directory walk
            sidecars.append(
                SidecarEntry(
                    path,
                    tab,
                    _RunData.sidecar_rel_parts(tab, rel_parts),
                    status.st_size,
                    status.st_mtime_ns,
                )
            )
        return sidecars

    @staticmethod
    def sidecar_paths(source: Path, run_dir: Path) -> Iterator[Path]:
        """Yield candidate sidecar files that belong to one run directory.

        Directories with their own event files are other runs and are pruned.
        Symlinked directories are not followed, and symlinked files must
        resolve inside the source.
        """
        root = source.resolve()
        for dir_name, dir_names, file_names in os.walk(run_dir):
            directory = Path(dir_name)
            files = sorted(name for name in file_names if (directory / name).is_file())
            events = [
                name for name in files if fnmatchcase(name, TENSORBOARD_EVENT_GLOB)
            ]
            if events and directory != run_dir:
                dir_names.clear()
                continue
            dir_names.sort()
            for name in files:
                path = directory / name
                if name not in events and path.resolve().is_relative_to(root):
                    yield path

    @staticmethod
    def sidecar_tab(rel_parts: tuple[str, ...]) -> str | None:
        """Return the virtual tab for supported sidecar files."""
        lowered = tuple(part.lower() for part in rel_parts)
        if "profile" in lowered:
            return "profile"
        if "projector" in lowered or lowered[-1] == "projector_config.pbtxt":
            return "projector"
        return None

    @staticmethod
    def sidecar_rel_parts(tab: str, rel_parts: tuple[str, ...]) -> tuple[str, ...]:
        """Trim the sidecar tab directory prefix when present."""
        lowered = tuple(part.lower() for part in rel_parts)
        if tab in lowered:
            tail = rel_parts[lowered.index(tab) + 1 :]
            if tail:
                return tail
        return rel_parts
