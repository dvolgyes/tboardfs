"""Build the virtual tab directories and files of one TensorBoard run."""

from functools import partial
from typing import Any

from tboardfs.constants import FIXED_TABS
from tboardfs.file_times import FileTimes
from tboardfs.graph_files import add_graph_files
from tboardfs.histogram_files import add_histogram_files
from tboardfs.model import BinaryEntry, EventFileCache, JsonEntry, RunCache
from tboardfs.paths import _Paths
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
    run_node: dict[str, Any] = {"type": "dir", "children": {}}
    for tab in FIXED_TABS:
        run_node["children"].setdefault(tab, {"type": "dir", "children": {}})
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
        scalars = _RunTree.merged_scalars(run)
        files = RunFiles()
        _RunTree.add_scalar_files(files, run, scalars, scalar_formats)
        epochs = scalars.get("epoch")
        add_histogram_files(
            files, _RunTree.sorted_file_caches(run), epochs, scalar_formats, step_digits
        )
        _RunTree.add_binary_files(files, run, step_digits)
        consumed = add_typed_plugin_files(
            files, _RunTree.run_json_entries(run), scalars, step_digits
        )
        _RunTree.add_json_files(files, run, step_digits, consumed)
        add_graph_files(files, _RunTree.sorted_file_caches(run), step_digits)
        _RunTree.add_sidecar_files(files, run)
        RawFrames.add(files, run)
        files.fill(run_node, run)
        raw = run_node["children"]["raw"]
        raw["view"] = "raw"
        for event in raw["children"].values():
            event["view"] = "raw"

    @staticmethod
    def add_scalar_files(
        files: RunFiles,
        run: RunCache,
        scalars: dict[str, dict[str, Any]],
        scalar_formats: tuple[str, ...],
    ) -> None:
        """Add the merged scalar tables of one run as export files."""
        mtimes = _TableMerge.merge_mtimes(
            cache.scalar_mtimes for cache in _RunTree.sorted_file_caches(run)
        )
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
    def add_binary_files(files: RunFiles, run: RunCache, step_digits: int) -> None:
        """Add lazily extracted binary files for one run."""
        for entry in _RunTree.run_binaries(run):
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
        run: RunCache,
        step_digits: int,
        consumed: set[int],
    ) -> None:
        """Add plugin JSON and text files for one run."""
        for entry in _RunTree.run_json_entries(run):
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
    def merged_scalars(run: RunCache) -> dict[str, dict[str, Any]]:
        """Return merged scalar series for one run, with the epoch of each step."""
        scalars = _TableMerge.merge_run_tables(_RunTree.sorted_file_caches(run))
        return _TableMerge.sort_scalars(
            _TableMerge.step_epochs(scalars, scalars.get("epoch"))
        )

    @staticmethod
    def add_sidecar_files(files: RunFiles, run: RunCache) -> None:
        """Add projector, profile and hparams sidecar files for one run."""
        for sidecar in run.sidecars:
            parts = (
                _Paths.safe_component(p, name_map=files.name_map)
                for p in sidecar.rel_parts
            )
            files.add_fixed(
                (sidecar.tab, *parts),
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

    @staticmethod
    def run_binaries(run: RunCache) -> list[BinaryEntry]:
        """Return binary entries from all event-file caches."""
        return [
            entry
            for cache in _RunTree.sorted_file_caches(run)
            for entry in cache.binaries
        ]

    @staticmethod
    def run_json_entries(run: RunCache) -> list[JsonEntry]:
        """Return plugin JSON entries from all event-file caches."""
        return [
            entry
            for cache in _RunTree.sorted_file_caches(run)
            for entry in cache.json_entries
        ]
