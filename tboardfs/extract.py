from errno import ENOENT
from typing import Any

from tboardfs.classify import _Classifier
from tboardfs.errors import fuse_error
from tboardfs.model import BinaryEntry, GraphEntry, MetaGraphEntry
from tboardfs.summary import parse_event
from tboardfs.tensor_summary import _TensorSummary


def extract_binary_blob(entry: BinaryEntry) -> bytes:
    """Read one indexed binary blob from its source event record."""
    with entry.source_path.open("rb") as handle:
        handle.seek(entry.payload_offset)
        payload = handle.read(entry.payload_size)
    event = parse_event(payload)
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
    raise fuse_error(ENOENT)


def _extract_graph_blob(entry: GraphEntry) -> bytes:
    """Read one indexed graph or meta graph payload from its source event record."""
    with entry.source_path.open("rb") as handle:
        handle.seek(entry.payload_offset)
        payload = handle.read(entry.payload_size)
    event: dict[str, Any] = parse_event(payload)
    field = "meta_graph_def" if isinstance(entry, MetaGraphEntry) else "graph_def"
    graph = event.get(field)
    if graph is None:
        raise fuse_error(ENOENT)
    return bytes(graph)
