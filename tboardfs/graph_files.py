"""Add the graph files of a run to its virtual directory."""

from typing import Any

from tboardfs.model import EventFileCache
from tboardfs.paths import _Paths


def add_graph_files(
    run_node: dict[str, Any], caches: list[EventFileCache], step_digits: int
) -> None:
    """Add graph protobuf files for one run.

    graph.pb is the newest graph, as in TensorBoard. A run with several
    graphs also lists each one as graph_<i>.pb in write order. A TF2 graph
    summary is stored as <step>.json (Keras config) or <step>.run_metadata.pb;
    beside it, <step>.conceptual.pb or <step>.pb is the GraphDef that
    TensorBoard shows for it.
    """
    graphs = [entry for cache in caches for entry in cache.graphs]
    named = [(graphs[-1].name, graphs[-1])] if graphs else []
    if len(graphs) > 1:
        named += [(f"graph_{index}.pb", graph) for index, graph in enumerate(graphs)]
    for name, graph in named:
        _add_file(
            run_node,
            ("graphs", name),
            {"type": "graph", "entry": graph, "mtime": graph.wall_time},
        )
    for cache in caches:
        for entry in cache.binaries:
            if entry.kind != "graphs":
                continue
            step = _Paths.step_name(entry.step, step_digits)
            name = "conceptual.pb" if entry.extension == "json" else "pb"
            _add_file(
                run_node,
                ("graphs", *_Paths.tag_dir_parts(entry.tag), f"{step}.{name}"),
                {"type": "graph_def", "entry": entry, "mtime": entry.wall_time},
            )


def _add_file(
    root: dict[str, Any], rel_parts: tuple[str, ...], node: dict[str, Any]
) -> None:
    """Add one virtual file to a directory tree."""
    _Paths.mkdirs(root, rel_parts[:-1])["children"][rel_parts[-1]] = node
