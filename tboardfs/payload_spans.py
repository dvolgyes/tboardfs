"""Find candidate protobuf byte ranges without interpreting summary values."""

from collections.abc import Iterator
from functools import cached_property
from typing import Self

Span = tuple[int, int]
_MIN_FRAME_SIZE = 4096


class _UnsupportedWireError(ValueError):
    """The wire fields cannot be indexed without protobuf interpretation."""


def _varint(data: bytes, pos: int, end: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 64, 7):
        if pos >= end:
            break
        byte = data[pos]
        pos += 1
        value |= (byte & 127) << shift
        if byte < 128 and value < 1 << 64:
            return value, pos
    raise _UnsupportedWireError


def _byte_fields(data: bytes, parent: Span) -> Iterator[tuple[int, Span]]:
    pos, size = parent
    end = pos + size
    while pos < end:
        key, pos = _varint(data, pos, end)
        number, wire = divmod(key, 8)
        if not 0 < number < 1 << 29:
            raise _UnsupportedWireError
        if wire == 0:
            _, pos = _varint(data, pos, end)
            continue
        if wire == 2:
            size, pos = _varint(data, pos, end)
        elif wire == 1:
            size = 8
        elif wire == 5:
            size = 4
        else:
            raise _UnsupportedWireError
        if size > end - pos:
            raise _UnsupportedWireError
        if wire == 2:
            yield number, (pos, size)
        pos += size


class PayloadSpans:
    """Index byte field boundaries on demand; protobuf supplies the actual bytes."""

    payload: bytes

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    @classmethod
    def for_payload(cls, payload: bytes) -> Self | None:
        """Skip extent indexing for small frames."""
        return cls(payload) if len(payload) >= _MIN_FRAME_SIZE else None

    def _fields(self, parent: Span, field: int) -> list[Span]:
        try:
            return [
                span
                for number, span in _byte_fields(self.payload, parent)
                if number == field
            ]
        except _UnsupportedWireError:
            return []

    def _only(self, parent: Span, field: int) -> Span | None:
        found = self._fields(parent, field)
        return found[0] if len(found) == 1 else None

    @cached_property
    def _values(self) -> list[Span]:
        summary = self._only((0, len(self.payload)), 5)
        return self._fields(summary, 1) if summary is not None else []

    def value_fields(self, ordinal: int, field: int, nested_field: int) -> list[Span]:
        """Find a legacy media or tensor field within the selected value."""
        if ordinal >= len(self._values):
            return []
        message = self._only(self._values[ordinal], field)
        return self._fields(message, nested_field) if message is not None else []

    def graph(self, field: int, blob: bytes) -> Span | None:
        """Return the verified boundary of an Event graph field."""
        return self.match(self._only((0, len(self.payload)), field), blob)

    def match(self, span: Span | None, blob: bytes) -> Span | None:
        """Accept a candidate only if it contains exactly the parsed bytes."""
        if span is not None:
            pos, size = span
            if size == len(blob) and self.payload.startswith(blob, pos, pos + size):
                return span
        return None
