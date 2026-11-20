"""Indexed source evidence for one verbatim binary payload."""

from dataclasses import dataclass
from typing import ClassVar, NamedTuple, Protocol, dataclass_transform

from tboardfs.file_status import FileIdentity
from tboardfs.frame_ref import FrameRef
from tboardfs.payload_spans import Span


@dataclass(frozen=True, slots=True)
class IndexedBlob:
    """A blob's frame, length, and optional range verification evidence."""

    frame: FrameRef
    size: int
    span: Span | None = None
    range_checksums: bytes = b""


class _FrameItem(NamedTuple):
    """Keep a binary item's size and verified frame coordinates without its payload."""

    identity: FileIdentity
    fingerprint: tuple[int | None, ...]
    offset: int
    frame_size: int
    checksum: int
    size: int

    def resolve(self) -> IndexedBlob:
        """Materialize evidence for an item that requires a whole-frame read."""
        frame = FrameRef(
            self.identity, self.offset, self.frame_size, self.checksum, self.fingerprint
        )
        return IndexedBlob(frame, self.size)


class _BlobOwner(Protocol):
    def blob_at(self, index: int) -> IndexedBlob:
        """Return the unpacked evidence for a row."""
        ...

    def size_at(self, index: int) -> int:
        """Return the validated byte count for a row."""
        ...


@dataclass(frozen=True, slots=True)
class _BlobRef:
    """Locate indexed evidence in shared row storage."""

    owner: _BlobOwner
    index: int

    @property
    def size(self) -> int:
        """Return the byte count without unpacking the row."""
        return self.owner.size_at(self.index)

    def resolve(self) -> IndexedBlob:
        """Unpack this row's validated frame evidence."""
        return self.owner.blob_at(self.index)


type _Evidence = IndexedBlob | _FrameItem | _BlobRef


class _BlobField:
    """Require the constructor field and resolve a stored locator on first access."""

    def __get__(
        self,
        entry: "_IndexedBlobEntry | None",
        _owner: type["_IndexedBlobEntry"],
    ) -> IndexedBlob:
        if entry is None:
            raise AttributeError("blob")
        blob = entry._blob
        if not isinstance(blob, IndexedBlob):
            blob = blob.resolve()
            entry._blob = blob
        return blob

    def __set__(self, entry: "_IndexedBlobEntry", blob: _Evidence) -> None:
        entry._blob = blob


class _BlobStorage:
    """Keep the backing slot outside the dataclass fields."""

    __slots__: ClassVar[tuple[str, ...]] = ("_blob",)

    _blob: _Evidence


@dataclass_transform(field_specifiers=(_BlobField,))
def _entry_dataclass[T](cls: type[T]) -> type[T]:
    """Declare the descriptor as a required field for both dataclasses and type checkers."""
    return dataclass(cls)


@_entry_dataclass
class _IndexedBlobEntry(_BlobStorage):
    """Store indexed evidence and expose its source frame."""

    blob: _BlobField = _BlobField()

    @property
    def size(self) -> int:
        """Return the validated byte count without unpacking frame evidence."""
        return self._blob.size

    @property
    def frame(self) -> FrameRef:
        """Return the source frame of the indexed payload."""
        return self.blob.frame
