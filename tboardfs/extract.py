from errno import EIO
from typing import Any

from google.protobuf.message import DecodeError

from tboardfs.classify import _Classifier
from tboardfs.errors import fuse_error
from tboardfs.model import BinaryEntry, GraphEntry, MetaGraphEntry
from tboardfs.summary import parse_event
from tboardfs.tensor_summary import _TensorSummary


def extract_binary_blob(entry: BinaryEntry) -> bytes:
    """Read one indexed binary blob from its source event record."""
    event = _source_event(entry)
    for value in event.get("values", []):
        if entry.part is not None:
            if value.get("tag") == entry.tag and _TensorSummary.media_parts(value):
                return bytes(value["tensor"]["string_val"][entry.part])
            continue
        if (
            value.get("tag") == entry.tag
            and _Classifier.binary_kind(value) == entry.kind
        ):
            blob = _Classifier.binary_blob(value)
            if blob is not None:
                return blob
    raise fuse_error(EIO)  # the event file changed after indexing


def _extract_graph_blob(entry: GraphEntry) -> bytes:
    """Read one indexed graph or meta graph payload from its source event record."""
    event = _source_event(entry)
    field = "meta_graph_def" if isinstance(entry, MetaGraphEntry) else "graph_def"
    graph = event.get(field)
    if graph is None:
        raise fuse_error(EIO)  # the event file changed after indexing
    return bytes(graph)


def _source_event(entry: BinaryEntry | GraphEntry) -> dict[str, Any]:
    """Parse the indexed record of an entry, or raise EIO when it is gone.

    The event file can change in place after it was indexed, so the bytes at
    the indexed offset may be short or may not be an Event any more.
    """
    with entry.source_path.open("rb") as handle:
        handle.seek(entry.payload_offset)
        payload = handle.read(entry.payload_size)
    if len(payload) != entry.payload_size:
        raise fuse_error(EIO)
    try:
        return parse_event(payload)
    except DecodeError:
        raise fuse_error(EIO) from None
