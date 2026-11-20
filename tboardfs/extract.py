from errno import EIO
from typing import Any

from google.protobuf.message import DecodeError

from tboardfs.classify import _Classifier
from tboardfs.errors import fuse_error
from tboardfs.indexed_frame import IndexedFrame
from tboardfs.model import BinaryEntry, GraphEntry, MetaGraphEntry
from tboardfs.summary import parse_event


def extract_binary_blob(entry: BinaryEntry) -> bytes:
    """Read one indexed binary blob from its source event record."""
    payload = IndexedFrame.read(entry.source_path, entry.frame).payload
    if (span := entry.blob.span) is not None:
        offset, size = span
        return payload[offset : offset + size]
    event = _source_event(payload)
    try:
        value = event["values"][entry.value_ordinal]
        if value["tag"] == entry.tag:
            if entry.part is not None:
                return bytes(value["tensor"]["string_val"][entry.part])
            blob = _Classifier.binary_blob(value)
            if blob is not None:
                return blob
    except (IndexError, KeyError) as error:
        raise fuse_error(EIO) from error
    raise fuse_error(EIO)


def _extract_graph_blob(entry: GraphEntry) -> bytes:
    """Read one indexed graph or meta graph payload from its source event record."""
    payload = IndexedFrame.read(entry.source_path, entry.frame).payload
    if (span := entry.blob.span) is not None:
        offset, size = span
        return payload[offset : offset + size]
    event = _source_event(payload)
    field = "meta_graph_def" if isinstance(entry, MetaGraphEntry) else "graph_def"
    graph = event.get(field)
    if graph is None:
        raise fuse_error(EIO)  # the event file changed after indexing
    return bytes(graph)


def _source_event(payload: bytes) -> dict[str, Any]:
    """Parse the verified event payload, or report EIO if it cannot be decoded."""
    try:
        return parse_event(payload)
    except DecodeError:
        raise fuse_error(EIO) from None
