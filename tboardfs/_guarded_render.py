"""Render mounted cache misses through the source guard."""

from collections.abc import Callable
from errno import ENOENT
from pathlib import Path
from typing import Any

from tboardfs._directory import _GuardPlans, _OpenHow
from tboardfs.errors import fuse_error
from tboardfs.extract import _IndexedBlobDecoder
from tboardfs.indexed_frame import IndexedFrame
from tboardfs.model import BinaryEntry
from tboardfs.node_source import _NodeSource
from tboardfs.render_node import render_node


def _render_miss(
    node: dict[str, Any],
    allowed: Callable[[Path], bool],
    plans: _GuardPlans | None,
    mountpoint: Path | None,
) -> bytes:
    """Try a guarded frame read before falling back to ordinary rendering."""
    source = _NodeSource.path(node)
    if source is None:
        return render_node(node)
    data = _guarded_frame_bytes(node, source, plans, mountpoint)
    if data is not None:
        return data
    if not allowed(source):
        raise fuse_error(ENOENT)
    return render_node(node)


def _guarded_frame_bytes(
    node: dict[str, Any],
    source: Path,
    plans: _GuardPlans | None,
    mountpoint: Path | None,
) -> bytes | None:
    """Read one small frame through the descriptor that passed the path guard."""
    raw = node["type"] == "raw"
    entry: BinaryEntry | None = node["entry"] if node["type"] == "binary" else None
    blob = entry.blob if entry is not None else None
    if raw:
        frame = node["frame"]
    elif blob is not None and blob.span is None:
        frame = blob.frame
    else:
        return None
    if not IndexedFrame._is_small_frame(frame):
        return None
    fd = _OpenHow.open_source(source, plans, mountpoint)
    if fd is None:
        return None
    payload = IndexedFrame._read_small(source, frame, raw=raw, fd=fd)
    return payload if entry is None else _IndexedBlobDecoder.binary(entry, payload)
