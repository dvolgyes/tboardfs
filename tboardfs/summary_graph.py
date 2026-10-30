"""Render the GraphDef that TensorBoard shows for a TF2 graph summary."""

from errno import EIO
import json
from typing import Any, cast

from google.protobuf.message import DecodeError, Message
from loguru import logger

from tboardfs.errors import fuse_error
from tboardfs.extract import extract_binary_blob
from tboardfs.keras_graph import keras_model_to_graph_def
from tboardfs.model import BinaryEntry
from tboardfs.proto import graph_pb2


def summary_graph_def(entry: BinaryEntry) -> bytes:
    """Return the serialized GraphDef of a Keras or RunMetadata graph summary.

    A Keras config (a .json entry) gives the conceptual graph. A RunMetadata
    gives the op graph: the pre-optimization graphs of its functions, merged
    as TensorBoard 2.21 merges them.

    Raises:
        FuseOSError: EIO when the summary cannot be converted.
    """
    blob = extract_binary_blob(entry)
    try:
        if entry.extension == "json":
            graph = keras_model_to_graph_def(json.loads(blob))
        else:
            run_metadata = cast(Any, getattr(graph_pb2, "RunMetadata")())
            run_metadata.ParseFromString(blob)
            graph = _OpGraph.merge(
                [
                    function.pre_optimization_graph
                    for function in run_metadata.function_graphs
                ]
            )
    except (DecodeError, ValueError) as error:
        logger.warning(
            "cannot convert graph {} at step {} in {}: {}",
            entry.tag,
            entry.step,
            entry.source_path,
            error,
        )
        raise fuse_error(EIO) from error
    data: bytes = graph.SerializeToString(deterministic=True)
    return data


class _OpGraph:
    """Merge function graphs as TensorBoard's graph_util.merge_graph_defs."""

    @staticmethod
    def merge(graphs: list[Any]) -> Message:
        """Return one graph, or graph_<i>/ prefixed copies of several graphs.

        Raises:
            ValueError: The graphs have different producer versions.
        """
        if len(graphs) == 1:
            return cast(Message, graphs[0])
        merged = cast(Any, getattr(graph_pb2, "GraphDef")())
        if graphs and graphs[0].versions.producer:
            merged.versions.CopyFrom(graphs[0].versions)
        for index, graph in enumerate(graphs, start=1):
            if graph.versions.producer != merged.versions.producer:
                raise ValueError("Cannot combine GraphDefs of different versions.")
            _OpGraph.add_prefixed(f"graph_{index}", graph, merged)
        return cast(Message, merged)

    @staticmethod
    def add_prefixed(prefix: str, source: Message, target: Message) -> None:
        """Add the nodes and functions of a graph with prefixed names."""
        graph, merged = cast(Any, source), cast(Any, target)
        for node in graph.node:
            copy = merged.node.add()
            copy.CopyFrom(node)
            copy.name = f"{prefix}/{node.name}"
            copy.input[:] = [f"{prefix}/{name}" for name in node.input]
            if copy.op == "PartitionedCall":
                # TensorBoard reads attr["f"], which adds it when it is missing.
                function = copy.attr["f"].func
                function.name = f"{prefix}_{function.name}"
        for function_def in graph.library.function:
            copy_function = merged.library.function.add()
            copy_function.CopyFrom(function_def)
            copy_function.signature.name = f"{prefix}_{function_def.signature.name}"
        for gradient in graph.library.gradient:
            copy_gradient = merged.library.gradient.add()
            copy_gradient.CopyFrom(gradient)
            copy_gradient.function_name = f"{prefix}_{gradient.function_name}"
            copy_gradient.gradient_func = f"{prefix}_{gradient.gradient_func}"
