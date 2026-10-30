"""Add the graph files of a run to its virtual directory."""

from collections.abc import Sequence

from tboardfs.model import (
    EventFileCache,
    GraphEntry,
    GraphSummaryEntry,
    MetaGraphEntry,
)
from tboardfs.paths import _Paths
from tboardfs.run_files import RunFiles


def add_graph_files(
    files: RunFiles,
    caches: list[EventFileCache],
    step_digits: int,
) -> None:
    """Add graph protobuf files for one run.

    graph.pb is the graph that TensorBoard shows: the newest graph_def, or
    without one, the graph of the newest TF1 meta graph. A run with several
    graph_defs also lists each one as graph_<i>.pb in write order.
    meta_graph.pb is the newest meta graph, with meta_graph_<i>.pb in the same
    way. A TF2 graph summary is stored as <step>.json (Keras config) or
    <step>.run_metadata.pb; beside it, <step>.conceptual.pb or <step>.pb is
    the GraphDef that TensorBoard shows for it, when the summary converts, and
    <step>.run_metadata.pbtxt is the profile text of a RunMetadata.
    """
    entries = [entry for cache in caches for entry in cache.graphs]
    graphs = [entry for entry in entries if not isinstance(entry, MetaGraphEntry)]
    meta_graphs = [entry for entry in entries if isinstance(entry, MetaGraphEntry)]
    _add_written_graphs(files, "graph", graphs)
    _add_written_graphs(files, "meta_graph", meta_graphs)
    if not graphs and meta_graphs and meta_graphs[-1].readable:
        files.add_fixed(
            ("graphs", "graph.pb"),
            {
                "type": "meta_graph_graph",
                "entry": meta_graphs[-1],
                "mtime": meta_graphs[-1].wall_time,
            },
        )
    for cache in caches:
        for entry in cache.binaries:
            if isinstance(entry, GraphSummaryEntry):
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
            {"type": "graph", "entry": graph, "mtime": graph.wall_time},
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
            {"type": "graph_def", "entry": entry, "mtime": entry.wall_time},
        )
    if entry.profiles:
        files.add(
            ("graphs",),
            entry.tag,
            f"{step}.run_metadata.pbtxt",
            {"type": "run_metadata_text", "entry": entry, "mtime": entry.wall_time},
        )
