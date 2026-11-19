"""Read the graph that a TF1 MetaGraphDef holds."""

from typing import Any, cast

from google.protobuf.message import DecodeError

from tboardfs.model import GraphEntry
from tboardfs.proto import compat_pb2
from tboardfs.skip_failure import _warn_skipped


def meta_graph_graph_def(entry: GraphEntry, blob: bytes) -> bytes | None:
    """Return the serialized GraphDef of a serialized MetaGraphDef.

    The graph is read as the TensorBoard 2.21 event_accumulator reads it: the
    MetaGraphDef is parsed and its graph_def serialized again, here with
    sorted maps so that the bytes do not change between mounts. A MetaGraphDef
    without a graph gives an empty GraphDef. A MetaGraphDef that TensorBoard
    cannot parse gives None and a warning.
    """
    meta_graph = cast(Any, getattr(compat_pb2, "MetaGraphDef")())
    try:
        meta_graph.ParseFromString(blob)
    except DecodeError as error:
        _warn_skipped(
            "meta graph at offset {} in {} has no graph file: {}",
            entry.frame.payload_offset,
            entry.source_path,
            error,
        )
        return None
    data: bytes = meta_graph.graph_def.SerializeToString(deterministic=True)
    return data
