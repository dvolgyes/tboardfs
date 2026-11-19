"""Build the virtual tab directories and files of one TensorBoard run."""

from functools import partial
from typing import Any

from tboardfs.constants import FIXED_TABS
from tboardfs.file_times import FileTimes
from tboardfs.graph_files import add_graph_files
from tboardfs.histogram_files import add_histogram_files
from tboardfs.model import EventFileCache, JsonEntry, RunCache
from tboardfs.open3d_asset import OPEN3D_ASSETS, OPEN3D_PLUGIN
from tboardfs.open3d_files import Open3DFiles
from tboardfs.paths import _DirectoryNode, _Paths
from tboardfs.raw_frames import RawFrames
from tboardfs.plugin_files import add_typed_plugin_files
from tboardfs.restarts import _RestartPurge
from tboardfs.run_files import RunFiles
from tboardfs.skip_failure import skip_failure
from tboardfs.tables import _TableMerge
from tboardfs.text_table import _TextTable


def build_run_tree(
    run: RunCache,
    scalar_formats: tuple[str, ...],
    step_digits: int,
) -> dict[str, Any]:
    """Return virtual tab directories and files for one TensorBoard run.

    Directory sources include sidecars; a single event file has none.
    """
    run_node = _DirectoryNode()
    for tab in FIXED_TABS:
        run_node["children"][tab] = _DirectoryNode()
    _RunTree.add_run_files(run_node, run, scalar_formats, step_digits)
    FileTimes.stamp_dirs(run_node)
    return run_node


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
        caches = _RunTree.sorted_file_caches(run)
        scalars = _RunTree.merged_scalars(caches)
        files = RunFiles()
        _RunTree.add_scalar_files(files, caches, scalars, scalar_formats)
        epochs = scalars.get("epoch")
        add_histogram_files(files, caches, epochs, scalar_formats, step_digits)
        _RunTree.add_binary_files(files, caches, step_digits)
        entries = [entry for cache in caches for entry in cache.json_entries]
        consumed = add_typed_plugin_files(files, entries, scalars, step_digits)
        Open3DFiles.add(files, entries, run.sidecars, step_digits)
        _RunTree.add_json_files(files, entries, step_digits, consumed)
        add_graph_files(files, caches, step_digits)
        _RunTree.add_sidecar_files(files, run)
        RawFrames.add(run_node["children"]["raw"], run, files.name_map)
        files.fill(run_node, run)

    @staticmethod
    def add_scalar_files(
        files: RunFiles,
        caches: list[EventFileCache],
        scalars: dict[str, dict[str, Any]],
        scalar_formats: tuple[str, ...],
    ) -> None:
        """Add the merged scalar tables of one run as export files."""
        mtimes = _TableMerge.merge_mtimes(cache.scalar_mtimes for cache in caches)
        for tag, series in scalars.items():
            for fmt in scalar_formats:
                files.add_named(
                    ("scalars",),
                    tag,
                    fmt,
                    {
                        "type": "scalar",
                        "series": series,
                        "format": fmt,
                        "mtime": mtimes.get(tag, 0.0),
                    },
                )

    @staticmethod
    def add_binary_files(
        files: RunFiles, caches: list[EventFileCache], step_digits: int
    ) -> None:
        """Add lazily extracted binary files for one run."""
        for entry in (entry for cache in caches for entry in cache.binaries):
            files.add(
                (entry.kind,),
                entry.tag,
                f"{_Paths.step_name(entry.step, step_digits)}{entry.suffix}"
                f".{entry.extension}",
                {"type": "binary", "entry": entry, "mtime": entry.wall_time},
            )

    @staticmethod
    def add_json_files(
        files: RunFiles,
        entries: list[JsonEntry],
        step_digits: int,
        consumed: set[int],
    ) -> None:
        """Add plugin JSON and text files for one run."""
        for entry in entries:
            if id(entry) in consumed:
                continue
            tab = entry.tab if entry.tab in FIXED_TABS else "plugins"
            text_nodes = (
                skip_failure(
                    f"text {entry.tag!r} at step {entry.step}",
                    partial(_TextTable.nodes, entry),
                )
                if tab == "text"
                else None
            )
            step = _Paths.step_name(entry.step, step_digits)
            if text_nodes:
                for suffix, node in text_nodes:
                    files.add((tab,), entry.tag, f"{step}.{suffix}", node)
                continue
            files.add(
                _RunTree.json_entry_root(tab, entry, files.name_map),
                entry.tag,
                f"{step}.json",
                {"type": "json", "data": entry.payload, "mtime": entry.wall_time},
            )

    @staticmethod
    def merged_scalars(caches: list[EventFileCache]) -> dict[str, dict[str, Any]]:
        """Return merged scalar series for one run, with the epoch of each step."""
        scalars = _TableMerge.merge_run_tables(caches)
        return _TableMerge.sort_scalars(
            _TableMerge.step_epochs(scalars, scalars.get("epoch"))
        )

    @staticmethod
    def add_sidecar_files(files: RunFiles, run: RunCache) -> None:
        """Add sidecar files for one run under their corresponding plugin views."""
        for sidecar in run.sidecars:
            parts = (
                _Paths.safe_component(p, name_map=files.name_map)
                for p in sidecar.rel_parts
            )
            root = OPEN3D_ASSETS if sidecar.tab == OPEN3D_PLUGIN else (sidecar.tab,)
            files.add_fixed(
                (*root, *parts),
                {
                    "type": "sidecar",
                    "entry": sidecar,
                    "mtime": sidecar.mtime_ns / 1e9,
                },
                identity=str(sidecar.source_path),
            )

    @staticmethod
    def json_entry_root(
        tab: str, entry: JsonEntry, name_map: dict[str, str]
    ) -> tuple[str, ...]:
        """Return the directory of the tag directories of a plugin JSON entry."""
        if tab == "plugins":
            plugin = _Paths.safe_component(
                str(entry.payload.get("plugin_name", "")), name_map=name_map
            )
            return (tab, plugin)
        return (tab,)

    @staticmethod
    def sorted_file_caches(run: RunCache) -> list[EventFileCache]:
        """Return run file caches in source-file order, after restart purges."""
        return _RestartPurge.purge([run.files[path] for path in sorted(run.files)])
