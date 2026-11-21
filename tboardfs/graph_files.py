"""Add the graph files of a run to its virtual directory."""

from collections.abc import Iterable, Sequence

from tboardfs.leaf_spec import LeafSpec
from tboardfs.model import (
    BinaryEntry,
    EventFileCache,
    GraphEntry,
    GraphSummaryEntry,
    MetaGraphEntry,
)
from tboardfs.model.packed_binaries import PackedBinaries
from tboardfs.paths import _Paths
from tboardfs.run_files import RunFiles


class GraphFiles:
    """Add written graphs and independent graph summaries to a run."""

    @staticmethod
    def add(files: RunFiles, caches: list[EventFileCache], step_digits: int) -> None:
        """Add the newest graph of each kind, numbered copies, and summaries.

        Without a graph_def, graph.pb uses the newest readable TF1 meta graph.
        A TF2 summary may have GraphDef and RunMetadata profile companions.
        """
        entries = [entry for cache in caches for entry in cache.graphs]
        graphs = [entry for entry in entries if not isinstance(entry, MetaGraphEntry)]
        meta_graphs = [entry for entry in entries if isinstance(entry, MetaGraphEntry)]
        _add_written_graphs(files, "graph", graphs)
        _add_written_graphs(files, "meta_graph", meta_graphs)
        if not graphs and meta_graphs and meta_graphs[-1].readable:
            files.add_fixed(
                ("graphs", "graph.pb"),
                LeafSpec(
                    "meta_graph_graph",
                    meta_graphs[-1].wall_time,
                    "entry",
                    meta_graphs[-1],
                ),
            )
        for cache in caches:
            GraphFiles.add_summaries(files, cache.binaries, step_digits)

    @staticmethod
    def add_summaries(
        files: RunFiles, entries: Iterable[BinaryEntry], step_digits: int
    ) -> None:
        """Add companions for indexed TF2 graph summaries."""
        if isinstance(entries, PackedBinaries):
            for chunk in entries.chunks():
                if isinstance(chunk, list):
                    GraphFiles.add_summaries(files, chunk, step_digits)
                else:
                    for summary in chunk.summaries():
                        _add_summary_files(files, summary, step_digits)
            return
        summaries = (entry for entry in entries if isinstance(entry, GraphSummaryEntry))
        for entry in summaries:
            _add_summary_files(files, entry, step_digits)


def _add_written_graphs(
    files: RunFiles,
    prefix: str,
    written: Sequence[GraphEntry],
) -> None:
    """Add <prefix>.pb for the newest graph, and <prefix>_<i>.pb for each of several."""
    named: list[tuple[str, GraphEntry]] = (
        [(f"{prefix}.pb", written[-1])] if written else []
    )
    if len(written) > 1:
        named += [
            (f"{prefix}_{index}.pb", graph) for index, graph in enumerate(written)
        ]
    for name, graph in named:
        files.add_fixed(
            ("graphs", name),
            LeafSpec("graph", graph.wall_time, "entry", graph),
        )


def _add_summary_files(
    files: RunFiles, entry: GraphSummaryEntry, step_digits: int
) -> None:
    """Add the GraphDef and profile files of one TF2 graph summary."""
    step = _Paths.step_name(entry.step, step_digits)
    if entry.converts:
        name = "conceptual.pb" if entry.extension == "json" else "pb"
        files.add(
            ("graphs",),
            entry.tag,
            f"{step}.{name}",
            LeafSpec("graph_def", entry.wall_time, "entry", entry),
        )
    if entry.profiles:
        files.add(
            ("graphs",),
            entry.tag,
            f"{step}.run_metadata.pbtxt",
            LeafSpec("run_metadata_text", entry.wall_time, "entry", entry),
        )
