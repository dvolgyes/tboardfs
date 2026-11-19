"""Indexed source evidence for one verbatim binary payload."""

from dataclasses import dataclass

from tboardfs.frame_ref import FrameRef
from tboardfs.payload_spans import Span


@dataclass(frozen=True, slots=True)
class IndexedBlob:
    """A blob's frame, length, and optional range verification evidence."""

    frame: FrameRef
    size: int
    span: Span | None = None
    range_checksums: bytes = b""


@dataclass
class _IndexedBlobEntry:
    """Store indexed evidence and expose its source frame."""

    blob: IndexedBlob

    @property
    def frame(self) -> FrameRef:
        """Return the source frame of the indexed payload."""
        return self.blob.frame
