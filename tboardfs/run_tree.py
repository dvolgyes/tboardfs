"""Build the virtual tab directories and files of one TensorBoard run."""

from functools import partial
from typing import Any

from tboardfs.constants import FIXED_TABS
from tboardfs.epoch_breakdown import _EpochBreakdown
from tboardfs.epochs import _Epochs
from tboardfs.export_formats import _ExportFormats
from tboardfs.file_times import FileTimes
from tboardfs.graph_files import GraphFiles
from tboardfs.histogram_files import add_histogram_files
from tboardfs.leaf_spec import LeafSpec
from tboardfs.model import EventFileCache, JsonEntry, RunCache
from tboardfs.open3d_asset import OPEN3D_ASSETS, OPEN3D_PLUGIN
from tboardfs.open3d_files import Open3DFiles
from tboardfs.paths import _DirectoryNode, _Paths
from tboardfs.plugin_files import _TypedPluginFiles, add_typed_plugin_files
from tboardfs.raw_frames import RawFrames
from tboardfs.restarts import _RestartPurge
from tboardfs.run_files import RunFiles
from tboardfs.skip_failure import skip_failure
from tboardfs.tables import _TableMerge
from tboardfs.text_table import _TextTable


class RunTree:
    """Build and update immutable run trees."""

    @staticmethod
    def build(
        run: RunCache,
        formats: _ExportFormats,
        step_digits: int,
        *,
        raw: bool = True,
        epoch_breakdown: bool = False,
    ) -> dict[str, Any]:
        """Build a complete run tree, including sidecars when present."""
        run_node = _DirectoryNode()
        _Projection.add_run_files(
            run_node,
            run,
            formats,
            step_digits,
            raw=raw,
            epoch_breakdown=epoch_breakdown,
        )
        FileTimes.stamp_dirs(run_node)
        return run_node

    @staticmethod
    def append(
        tree: dict[str, Any],
        run: RunCache,
        before: EventFileCache,
        after: EventFileCache,
        formats: _ExportFormats,
        step_digits: int,
        *,
        raw: bool = True,
    ) -> dict[str, Any] | None:
        """Project a verified final-file append, or request a cold build.

        The caller proves that earlier file caches and sidecars are unchanged.
        """
        if before.progress.processed_pos == after.progress.processed_pos:
            return tree

        entries = after.json_entries[len(before.json_entries) :]
        if len(before.graphs) != len(after.graphs) or any(
            _Projection.is_nonlocal_plugin(entry) for entry in entries
        ):
            return None
        binaries = after.binaries[len(before.binaries) :]
        scalar_tags = _Projection.changed_tags(before.scalars, after.scalars)
        hist_tags = _Projection.changed_tags(before.histograms, after.histograms)
        typed_change = bool(scalar_tags or hist_tags or binaries or entries)
        files = RunFiles()
        if (typed_change or len(before.restarts) != len(after.restarts)) and not (
            _Projection.project_tables(
                files,
                tree,
                run,
                before,
                scalar_tags,
                hist_tags,
                formats,
                step_digits,
            )
        ):
            return None
        files.add_binaries((binaries,), step_digits)
        consumed = add_typed_plugin_files(
            files, entries, {}, step_digits, pr_curve_npy=formats.pr_curve_npy
        )
        _Projection.add_json_files(files, entries, step_digits, consumed)
        GraphFiles.add_summaries(files, binaries, step_digits)

        raw_tab = (
            RawFrames.extend(tree["children"].get("raw"), after, files.name_map)
            if raw
            else None
        )
        return files.patch(tree, run, raw=raw_tab)


class _Projection:
    """Select and render virtual leaves from indexed run contents."""

    @staticmethod
    def is_nonlocal_plugin(entry: JsonEntry) -> bool:
        """Identify producers that need a run-wide aggregate or path change."""
        return (
            entry.tab in {"hparams", "custom_scalars", "meshes"}
            or entry.payload.get("plugin_name") == OPEN3D_PLUGIN
        )

    @staticmethod
    def project_tables(
        files: RunFiles,
        tree: dict[str, Any],
        run: RunCache,
        before: EventFileCache,
        scalar_tags: set[str],
        hist_tags: set[str],
        formats: _ExportFormats,
        step_digits: int,
    ) -> bool:
        """Project dirty tables only when parser chunks match the unpurged view."""
        caches = [run.files[path] for path in sorted(run.files)]
        if _RestartPurge.starts(caches):
            return False
        if not scalar_tags and not hist_tags:
            return True
        if not _Projection.histogram_names_are_stable(caches, before, hist_tags):
            return False
        epochs = _Epochs.from_caches(caches)
        align_all = any(map(_Epochs.is_marker, scalar_tags))
        if align_all:
            scalar_tags = {tag for cache in caches for tag in cache.scalars}
        if not _Projection.project_scalars(
            files, tree, caches, scalar_tags, formats.scalars, epochs
        ):
            return False
        dirty = (
            {tag for cache in caches for tag in cache.histograms}
            if align_all
            else hist_tags
        )
        return not dirty or add_histogram_files(
            files, caches, epochs, formats.histograms, step_digits, tags=dirty
        )

    @staticmethod
    def project_scalars(
        files: RunFiles,
        tree: dict[str, Any],
        caches: list[EventFileCache],
        tags: set[str],
        scalar_formats: tuple[str, ...],
        epochs: _Epochs,
    ) -> bool:
        """Realign scalar tags on an epoch append and update HParams metrics."""
        if not tags:
            return True
        scalars = _Projection.merged_scalars(caches, tags=tags, epochs=epochs)
        _Projection.add_scalar_files(files, caches, scalars, scalar_formats, tags=tags)
        if "hparams" not in tree["children"]:
            return True
        return _Projection.project_hparams(files, caches, scalars, epochs)

    @staticmethod
    def project_hparams(
        files: RunFiles,
        caches: list[EventFileCache],
        scalars: dict[str, dict[str, Any]],
        epochs: _Epochs,
    ) -> bool:
        """Regenerate only HParams from its original entries and all metrics."""
        entries = [
            entry
            for cache in caches
            for entry in cache.json_entries
            if entry.tab == "hparams"
        ]
        if not entries:
            return True
        missing = {tag for cache in caches for tag in cache.scalars} - scalars.keys()
        metrics = {
            **scalars,
            **_Projection.merged_scalars(caches, tags=missing, epochs=epochs),
        }
        consumed: set[int] = set()
        skip_failure(
            "the hparams file",
            partial(_TypedPluginFiles.add_hparams, files, entries, metrics, consumed),
        )
        return bool(consumed)

    @staticmethod
    def histogram_names_are_stable(
        caches: list[EventFileCache], before: EventFileCache, tags: set[str]
    ) -> bool:
        """Reject dtype promotion that could round and rename a historical step."""
        after = caches[-1]
        return all(
            cache.histograms[tag]["step"].dtype == after.histograms[tag]["step"].dtype
            for cache in (*caches[:-1], before)
            for tag in tags
            if tag in cache.histograms
        )

    @staticmethod
    def changed_tags(
        before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
    ) -> set[str]:
        """Find appended tables; the parser shares unchanged column arrays."""
        return {
            tag
            for tag, columns in after.items()
            if tag not in before or columns["step"] is not before[tag]["step"]
        }

    @staticmethod
    def add_run_files(
        run_node: dict[str, Any],
        run: RunCache,
        formats: _ExportFormats,
        step_digits: int,
        *,
        raw: bool,
        epoch_breakdown: bool = False,
    ) -> None:
        """Populate one run directory with indexed virtual files."""
        caches = _Projection.sorted_file_caches(run)
        epochs = _Epochs.from_caches(caches)
        breakdown = (
            _EpochBreakdown(epochs) if epoch_breakdown and epochs.has_markers else None
        )
        scalars = _Projection.merged_scalars(caches, epochs=epochs, breakdown=breakdown)
        files = RunFiles(epoch_breakdown=breakdown)
        _Projection.add_scalar_files(files, caches, scalars, formats.scalars)
        _ = add_histogram_files(files, caches, epochs, formats.histograms, step_digits)
        files.add_binaries((cache.binaries for cache in caches), step_digits)
        entries = [entry for cache in caches for entry in cache.json_entries]
        consumed = add_typed_plugin_files(
            files, entries, scalars, step_digits, pr_curve_npy=formats.pr_curve_npy
        )
        Open3DFiles.add(files, entries, run.sidecars, step_digits)
        _Projection.add_json_files(files, entries, step_digits, consumed)
        GraphFiles.add(files, caches, step_digits)
        _Projection.add_sidecar_files(files, run)
        if raw:
            raw_tab = _DirectoryNode()
            RawFrames.add(raw_tab, run, files.name_map)
            if raw_tab["children"]:
                run_node["children"]["raw"] = raw_tab
        files.fill(run_node, run)
        if breakdown is not None:
            breakdown.project(run_node)

    @staticmethod
    def add_scalar_files(
        files: RunFiles,
        caches: list[EventFileCache],
        scalars: dict[str, dict[str, Any]],
        scalar_formats: tuple[str, ...],
        *,
        tags: set[str] | None = None,
    ) -> None:
        """Add the merged scalar tables of one run as export files."""
        mtimes = _TableMerge.merge_mtimes(
            (cache.scalar_mtimes for cache in caches), tags=tags
        )
        for tag, series in scalars.items():
            for fmt in scalar_formats:
                files.add_named(
                    ("scalars",),
                    tag,
                    fmt,
                    LeafSpec(
                        "scalar", mtimes.get(tag, 0.0), "series", series, format=fmt
                    ),
                    producer="scalar",
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
                    files.add(
                        (tab,), entry.tag, f"{step}.{suffix}", node, step=entry.step
                    )
                continue
            files.add(
                _Projection.json_entry_root(tab, entry, files.name_map),
                entry.tag,
                f"{step}.json",
                LeafSpec("json", entry.wall_time, "data", entry.payload),
                producer="fallback" if tab in {"tensors", "pr_curves"} else "",
                step=entry.step,
            )

    @staticmethod
    def merged_scalars(
        caches: list[EventFileCache],
        *,
        epochs: _Epochs,
        tags: set[str] | None = None,
        breakdown: _EpochBreakdown | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Return merged scalar series for one run, with the epoch of each step."""
        scalars = _TableMerge.sort_scalars(
            _TableMerge.merge_run_tables(caches, tags=tags)
        )
        record = breakdown.record_scalars if breakdown is not None else None
        return epochs.align(scalars, record=record)

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
                LeafSpec("sidecar", sidecar.mtime_ns / 1e9, "entry", sidecar),
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
