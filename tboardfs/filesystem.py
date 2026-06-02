from dataclasses import dataclass, field
from errno import EINVAL, EISDIR, ENOENT, ENOTDIR, EOPNOTSUPP, EROFS
import json
from pathlib import Path, PurePosixPath
import posixpath
import stat
import time
from typing import Any
from collections.abc import Callable

from loguru import logger

from tboardfs.constants import (
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
from tboardfs.paths import (
    _Paths,
    find_tensorboard_files,
)
from tboardfs.tables import (
    _TableExport,
    _TableMerge,
    export_table,
)


@dataclass
class _FsState:
    """Mutable TensorBoardFS cache state.

    :ivar runs: run caches by virtual run path
    :ivar known_run_parts: discovered run path parts
    :ivar last_event_files: event files seen during last discovery
    :ivar last_discovery_at: timestamp of the last discovery pass
    :ivar tree_cache: cached virtual tree
    :ivar tree_cache_generation: generation represented by tree_cache
    :ivar tree_generation: current virtual tree generation
    """

    runs: dict[tuple[str, ...], RunCache] = field(default_factory=dict)
    known_run_parts: set[tuple[str, ...]] = field(default_factory=set)
    last_event_files: set[Path] | None = None
    last_discovery_at: float = 0.0
    tree_cache: dict[str, Any] | None = None
    tree_cache_generation: int = -1
    tree_generation: int = 0


class TensorBoardFS:
    """Read-only virtual filesystem over TensorBoard event files.

    :ivar source: source log directory
    :ivar step_digits: zero-padding width for step filenames
    :ivar refresh_age_seconds: minimum age before cache refresh
    :ivar scalar_formats: enabled scalar export formats
    :ivar log_to_screen: whether operations log status messages
    :ivar _state: mutable run discovery and tree state
    """

    source: Path
    step_digits: int
    refresh_age_seconds: float
    scalar_formats: tuple[str, ...]
    log_to_screen: bool
    _state: "_FsState"

    def __init__(
        self,
        source: str | Path,
        *,
        step_digits: int = 6,
        refresh_age_seconds: float = 60.0,
        scalar_formats: str | list[str] | tuple[str, ...] = DEFAULT_SCALAR_FORMATS,
        log_to_screen: bool = False,
    ) -> None:
        self.source = Path(source)
        self.step_digits = step_digits
        self.refresh_age_seconds = refresh_age_seconds
        self.scalar_formats = (
            _Paths.normalize_formats(scalar_formats) or DEFAULT_SCALAR_FORMATS
        )
        self.log_to_screen = log_to_screen
        self._state = _FsState()
        self._discover_runs(force=True)

    def getattr(self, path: str, *file_handles: object) -> dict[str, Any]:
        """Return stat data for a virtual path."""
        del file_handles
        try:
            node = self._lookup(path)
        except FileNotFoundError:
            raise fuse_error(ENOENT)
        now = time.time()
        if node["type"] == "dir":
            return _FsNode.stat_dict(
                stat.S_IFDIR | 0o555, 2, 0, now, node.get("mtime", now)
            )
        data = self._node_bytes(node)
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

    def read(self, path: str, size: int, offset: int, *file_handles: object) -> bytes:
        """Read a byte range from a virtual file."""
        del file_handles
        node = self._lookup(path)
        if node["type"] == "dir":
            raise fuse_error(EISDIR)
        data = self._node_bytes(node)
        return data[offset : offset + size]

    def symlink(self, target: str, source: str) -> None:
        """Register a same-stem scalar export alias in memory."""
        link_path = _Paths.norm_virtual_path(source)
        target_path = _Paths.norm_virtual_path(
            target, base=posixpath.dirname(link_path)
        )
        run_parts, tab, rel = self._split_tab_path(link_path)
        target_run, target_tab, target_rel = self._split_tab_path(target_path)
        if tab != "scalars" or target_tab != "scalars" or target_run != run_parts:
            raise fuse_error(EINVAL)
        if not rel or not target_rel:
            raise fuse_error(EINVAL)
        if PurePosixPath(rel).parent != PurePosixPath(target_rel).parent:
            raise fuse_error(EINVAL)
        link_stem, link_ext = _Paths.split_ext(rel)
        target_stem, target_ext = _Paths.split_ext(target_rel)
        if link_stem != target_stem or not link_ext or link_ext == target_ext:
            raise fuse_error(EINVAL)
        if (
            target_ext not in self.scalar_formats
            and target_ext not in DEFAULT_SCALAR_FORMATS
        ):
            raise fuse_error(EINVAL)
        run = self._ensure_run(run_parts)
        self._refresh_run(run)
        if target_stem not in self._run_scalars(run):
            raise fuse_error(EINVAL)
        run.scalar_links.add(rel)
        self._invalidate_tree()
        self._log("registered scalar alias {}", link_path)

    def unlink(self, path: str) -> None:
        """Handle cache-control unlinks; reject all normal mutations."""
        parts = _Paths.path_parts(path)
        if not parts:
            raise fuse_error(EROFS)
        if parts[-1] == ".cache":
            self._log(
                "refresh requested {}",
                "/" + "/".join(parts[:-1]) if len(parts) > 1 else "/",
            )
            self._unlink_cache(parts)
            return
        if parts[-1] == ".in_memory":
            self._log(
                "memory cache cleared {}",
                "/" + "/".join(parts[:-1]) if len(parts) > 1 else "/",
            )
            self._unlink_in_memory(parts)
            return
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
        self._discover_runs()
        self._refresh_runs()
        tree = self._cached_tree()
        node = tree
        for part in _Paths.path_parts(path):
            if node["type"] != "dir" or part not in node["children"]:
                raise FileNotFoundError(path)
            node = node["children"][part]
        return node

    def _build_tree(self) -> dict[str, Any]:
        root: dict[str, Any] = {
            "type": "dir",
            "children": {
                ".cache": _FsNode.control_node(),
                ".in_memory": _FsNode.control_node(),
            },
        }
        for run_parts in sorted(self._state.known_run_parts):
            run = self._ensure_run(run_parts)
            run_node = _Paths.mkdirs(root, run_parts)
            run_node["children"].setdefault(".cache", _FsNode.control_node())
            run_node["children"].setdefault(".in_memory", _FsNode.control_node())
            for tab in FIXED_TABS:
                run_node["children"].setdefault(tab, {"type": "dir", "children": {}})
            _RunTree.add_run_files(run_node, run, self.scalar_formats, self.step_digits)
        return root

    def _node_bytes(self, node: dict[str, Any]) -> bytes:
        node_type = node["type"]
        if node_type == "bytes":
            return bytes(node["data"])
        if node_type == "json":
            return (
                json.dumps(node["data"], default=_TableExport.json_default, indent=2)
                + "\n"
            ).encode()
        if node_type in {"scalar", "table"}:
            return export_table(node["series"], node["format"])
        if node_type == "binary":
            return extract_binary_blob(node["entry"])
        if node_type == "graph":
            return _extract_graph_blob(node["entry"])
        if node_type == "sidecar":
            return bytes(node["entry"].source_path.read_bytes())
        if node_type == "control":
            return b""
        raise fuse_error(EOPNOTSUPP)

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
        for run_parts in self._state.known_run_parts:
            self._state.runs.setdefault(run_parts, RunCache(run_parts))
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
        return self._state.runs.setdefault(run_parts, RunCache(run_parts))

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
        run.refreshed_at = now
        if changed:
            self._invalidate_tree()
        return changed

    def _run_scalars(self, run: RunCache) -> dict[str, dict[str, Any]]:
        return _TableMerge.merge_series(
            cache.scalars for cache in _RunData.sorted_file_caches(run)
        )

    def _split_tab_path(self, path: str) -> tuple[tuple[str, ...], str, str]:
        parts = _Paths.path_parts(path)
        for index, part in enumerate(parts):
            if part in FIXED_TABS:
                rel = posixpath.join(*parts[index + 1 :]) if parts[index + 1 :] else ""
                return tuple(parts[:index]), part, rel
        raise fuse_error(EINVAL)

    def _unlink_cache(self, parts: tuple[str, ...]) -> None:
        if len(parts) == 1:
            self._state.runs.clear()
            self._discover_runs(force=True)
            self._refresh_runs(force=True)
            self._invalidate_tree()
            return
        run = self._ensure_run(tuple(parts[:-1]))
        run.files.clear()
        self._refresh_run(run, force=True)

    def _unlink_in_memory(self, parts: tuple[str, ...]) -> None:
        if len(parts) == 1:
            links = {
                key: set(run.scalar_links) for key, run in self._state.runs.items()
            }
            self._state.runs.clear()
            self._discover_runs(force=True)
            for key, value in links.items():
                self._state.runs.setdefault(key, RunCache(key)).scalar_links = value
            self._invalidate_tree()
            return
        run = self._ensure_run(tuple(parts[:-1]))
        self._state.runs[run.run_parts] = RunCache(
            run.run_parts, scalar_links=set(run.scalar_links)
        )
        self._invalidate_tree()

    def _log(self, message: str, *args: object) -> None:
        if self.log_to_screen:
            logger.info(message, *args)

    def _refresh_runs(self, *, force: bool = False) -> None:
        for run in list(self._state.runs.values()):
            self._refresh_run(run, force=force)

    def _cached_tree(self) -> dict[str, Any]:
        if (
            self._state.tree_cache is not None
            and self._state.tree_cache_generation == self._state.tree_generation
        ):
            return self._state.tree_cache
        self._state.tree_cache = self._build_tree()
        self._state.tree_cache_generation = self._state.tree_generation
        return self._state.tree_cache

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
    def control_node() -> dict[str, Any]:
        """Return a virtual control-file node."""
        return {"type": "control", "mtime": time.time()}

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
        _RunTree.add_histogram_files(run_node, run, scalar_formats)
        _RunTree.add_binary_files(run_node, run, step_digits)
        _RunTree.add_json_files(run_node, run, step_digits)
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
        for rel in run.scalar_links:
            stem, fmt = _Paths.split_ext(rel)
            if stem in scalars:
                _FsNode.add_file(
                    run_node,
                    ("scalars", *_Paths.path_parts(rel)),
                    {
                        "type": "scalar",
                        "series": scalars[stem],
                        "format": fmt,
                        "mtime": mtimes.get(stem, time.time()),
                    },
                )

    @staticmethod
    def add_histogram_files(
        run_node: dict[str, Any], run: RunCache, scalar_formats: tuple[str, ...]
    ) -> None:
        """Add histogram and distribution table files for one run."""
        histograms = _TableMerge.merge_series(
            cache.histograms for cache in _RunData.sorted_file_caches(run)
        )
        mtimes = _TableMerge.merge_mtimes(
            cache.histogram_mtimes for cache in _RunData.sorted_file_caches(run)
        )
        for tag, series in histograms.items():
            for fmt in scalar_formats:
                if fmt not in {"json", "tsv", "npz"}:
                    continue
                _FsNode.add_file(
                    run_node,
                    ("histograms", *_Paths.tag_parts(tag, fmt)),
                    _FsNode.table_node(series, fmt, mtimes.get(tag, time.time())),
                )
                _FsNode.add_file(
                    run_node,
                    ("distributions", *_Paths.tag_parts(tag, fmt)),
                    _FsNode.table_node(
                        _TableMerge.distribution_from_histogram(series),
                        fmt,
                        mtimes.get(tag, time.time()),
                    ),
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
        run_node: dict[str, Any], run: RunCache, step_digits: int
    ) -> None:
        """Add plugin JSON and text files for one run."""
        for entry in _RunData.run_json_entries(run):
            tab = entry.tab if entry.tab in FIXED_TABS else "plugins"
            text = _RunTree.text_from_json_entry(entry)
            if tab == "text" and text is not None:
                _FsNode.add_file(
                    run_node,
                    (
                        tab,
                        *_Paths.tag_dir_parts(entry.tag),
                        f"{_Paths.step_name(entry.step, step_digits)}.txt",
                    ),
                    {"type": "bytes", "data": text.encode(), "mtime": entry.wall_time},
                )
                continue
            _FsNode.add_file(
                run_node,
                _RunTree.json_entry_parts(tab, entry, step_digits),
                {"type": "json", "data": entry.payload, "mtime": entry.wall_time},
            )

    @staticmethod
    def add_graph_files(run_node: dict[str, Any], run: RunCache) -> None:
        """Add graph protobuf files for one run."""
        for graph in _RunData.run_graphs(run):
            _FsNode.add_file(
                run_node,
                ("graphs", graph.name),
                {"type": "graph", "entry": graph, "mtime": graph.wall_time},
            )

    @staticmethod
    def add_sidecar_files(run_node: dict[str, Any], run: RunCache) -> None:
        """Add projector/profile sidecar files for one run."""
        for sidecar in _RunData.run_sidecars(run):
            _FsNode.add_file(
                run_node,
                (sidecar.tab, *sidecar.rel_parts),
                {
                    "type": "sidecar",
                    "entry": sidecar,
                    "mtime": sidecar.source_path.stat().st_mtime,
                },
            )

    @staticmethod
    def text_from_json_entry(entry: JsonEntry) -> str | None:
        """Return text content from a text plugin JSON entry."""
        tensor = entry.payload.get("tensor") or {}
        values = tensor.get("string_val")
        if values:
            return "\n".join(str(value) for value in values)
        return None

    @staticmethod
    def json_entry_parts(
        tab: str, entry: JsonEntry, step_digits: int
    ) -> tuple[str, ...]:
        """Return virtual path parts for a plugin JSON entry."""
        filename = f"{_Paths.step_name(entry.step, step_digits)}.json"
        if tab == "plugins":
            plugin = str(entry.payload.get("plugin_name") or "unknown")
            return (tab, plugin, *_Paths.tag_dir_parts(entry.tag), filename)
        return (tab, *_Paths.tag_dir_parts(entry.tag), filename)


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
        """Refresh event-file caches for a run."""
        changed = False
        for source_path in files:
            old = run.files.get(source_path)
            size = source_path.stat().st_size
            if not force and old is not None and size == old.last_checked_size:
                continue
            if log is not None:
                log("parsing {}", source_path)
            changed = True
            if not force and old is not None and size >= old.processed_pos:
                run.files[source_path] = _EventIndexer.parse_event_file(
                    source_path,
                    ignore_truncated=True,
                    start_pos=old.processed_pos,
                    base_cache=old,
                )
            else:
                run.files[source_path] = _EventIndexer.parse_event_file(
                    source_path, ignore_truncated=True
                )
        for deleted in set(run.files) - set(files):
            if log is not None:
                log("event file removed {}", deleted)
            del run.files[deleted]
            changed = True
        return changed

    @staticmethod
    def sorted_file_caches(run: RunCache) -> list[EventFileCache]:
        """Return run file caches in deterministic source-file order."""
        return [run.files[path] for path in sorted(run.files)]

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
    def run_sidecars(run: RunCache) -> list[SidecarEntry]:
        """Return projector/profile sidecar entries discovered under a run."""
        run_dir = (
            run.files[next(iter(run.files))].source_path.parent if run.files else Path()
        )
        if not run_dir.exists():
            return []
        sidecars = []
        for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
            if path.match(TENSORBOARD_EVENT_GLOB):
                continue
            rel_parts = path.relative_to(run_dir).parts
            tab = _RunData.sidecar_tab(rel_parts)
            if tab is not None:
                sidecars.append(
                    SidecarEntry(path, tab, _RunData.sidecar_rel_parts(tab, rel_parts))
                )
        return sidecars

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
