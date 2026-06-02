from errno import ENOENT
from typing import Any

from tboardfs.classify import _Classifier
from tboardfs.errors import fuse_error
from tboardfs.model import BinaryEntry, GraphEntry
from tboardfs.summary import parse_event


def extract_binary_blob(entry: BinaryEntry) -> bytes:
    """Read one indexed binary blob from its source event record."""
    with entry.source_path.open("rb") as handle:
        handle.seek(entry.payload_offset)
        payload = handle.read(entry.payload_size)
    event = parse_event(payload)
    for value in event.get("values", []):
        if (
            value.get("tag") == entry.tag
            and _Classifier.binary_kind(value) == entry.kind
        ):
            blob = _Classifier.binary_blob(value)
            if blob is not None:
                return blob
    raise fuse_error(ENOENT)


def _extract_graph_blob(entry: GraphEntry) -> bytes:
    """Read one indexed graph payload from its source event record."""
    with entry.source_path.open("rb") as handle:
        handle.seek(entry.payload_offset)
        payload = handle.read(entry.payload_size)
    event: dict[str, Any] = parse_event(payload)
    graph = event.get("graph_def")
    if graph is None:
        raise fuse_error(ENOENT)
    return bytes(graph)
