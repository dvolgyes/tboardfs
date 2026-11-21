"""Keep binary rows packed until a caller needs an entry."""

from bisect import bisect_right
from collections.abc import Iterator, Sequence
from operator import index as to_index
from types import NotImplementedType
from typing import ClassVar, Self, overload

from tboardfs.model.binary_entry import BinaryEntry
from tboardfs.model.packed_source import PackedSource

type BinaryChunk = PackedSource | list[BinaryEntry]
type BinaryEntries = list[BinaryEntry] | PackedBinaries


class PackedBinaries(Sequence[BinaryEntry]):
    """An ordered sequence with packed sources and an optional eager tail."""

    __slots__: ClassVar[tuple[str, ...]] = ("_chunks", "_ends", "_tail")
    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    _chunks: list[BinaryChunk]
    _ends: list[int]
    _tail: list[BinaryEntry] | None

    def __init__(self, source: BinaryChunk) -> None:
        self._chunks = []
        self._ends = []
        self._tail = None
        self._add_chunk(source)

    def _add_chunk(self, chunk: BinaryChunk) -> None:
        if chunk:
            self._ends.append(len(self) + len(chunk))
            self._chunks.append(chunk)

    def chunks(self) -> Iterator[BinaryChunk]:
        """Iterate sources in event order without resolving their rows."""
        return iter(self._chunks)

    def copy(self) -> Self:
        """Share historical chunks and isolate the appendable tail."""
        result = type(self)([])
        result._chunks = self._chunks.copy()
        result._ends = self._ends.copy()
        if self._tail is not None:
            tail = self._tail.copy()
            result._chunks[-1] = tail
            result._tail = tail
        return result

    def append(self, entry: BinaryEntry) -> None:
        """Append a newly parsed entry without unpacking earlier rows."""
        if self._tail is None:
            self._tail = [entry]
            self._add_chunk(self._tail)
        else:
            self._tail.append(entry)
            self._ends[-1] += 1

    def extend(self, entries: BinaryEntries) -> None:
        """Join an indexed suffix without sharing a mutable tail."""
        if isinstance(entries, PackedBinaries):
            if entries is self:
                entries = entries.copy()
            self._tail = None
            for chunk in entries.chunks():
                if isinstance(chunk, list) and chunk is entries._tail:
                    self._add_chunk(chunk.copy())
                else:
                    self._add_chunk(chunk)
        elif entries:
            self._tail = None
            self._add_chunk(entries.copy())

    def __len__(self) -> int:
        return self._ends[-1] if self._ends else 0

    def __iter__(self) -> Iterator[BinaryEntry]:
        for chunk in self._chunks:
            if isinstance(chunk, list):
                yield from chunk
            else:
                yield from (chunk.entry_at(index) for index in range(len(chunk)))

    @overload
    def __getitem__(self, index: int) -> BinaryEntry: ...

    @overload
    def __getitem__(self, index: slice) -> list[BinaryEntry]: ...

    def __getitem__(self, index: int | slice) -> BinaryEntry | list[BinaryEntry]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        position = to_index(index)
        if position < 0:
            position += len(self)
        if not 0 <= position < len(self):
            raise IndexError("binary row index out of range")
        chunk_index = bisect_right(self._ends, position)
        if chunk_index:
            position -= self._ends[chunk_index - 1]
        chunk = self._chunks[chunk_index]
        return chunk[position] if isinstance(chunk, list) else chunk.entry_at(position)

    def __eq__(self, other: object) -> bool | NotImplementedType:
        if other is self:
            return True
        if not isinstance(other, (list, PackedBinaries)):
            return NotImplemented
        return len(self) == len(other) and all(
            left == right for left, right in zip(self, other, strict=True)
        )

    @staticmethod
    def merge(prefix: BinaryEntries, suffix: BinaryEntries) -> BinaryEntries:
        """Extend an owned prefix without expanding any packed sources."""
        if isinstance(prefix, list) and isinstance(suffix, list):
            prefix.extend(suffix)
        else:
            if isinstance(prefix, list):
                prefix = PackedBinaries(prefix)
            prefix.extend(suffix)
        return prefix
