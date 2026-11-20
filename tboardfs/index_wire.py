"""Bounded, data-only primitives for the persistent index body."""

import struct
from io import BytesIO
from typing import Any, cast

import numpy as np

from tboardfs.index_utf8 import encode_utf8

_LENGTH = struct.Struct("<Q")
_SIGNED = struct.Struct("<q")
_FLOAT = struct.Struct("<d")
_MAX_DEPTH = 32
_MAX_VALUE_NODES = 4_000_000


class _IndexValueError(ValueError):
    """A cache value does not conform to the persistent index format."""


class _IndexWriter:
    """Write fixed-width lengths and explicitly tagged primitive values."""

    _stream: BytesIO
    remaining: int
    _nodes: int

    def __init__(self, limit: int) -> None:
        """Create an empty wire body with a hard byte budget."""
        self._stream = BytesIO()
        self.remaining = limit
        self._nodes = _MAX_VALUE_NODES

    def ensure(self, size: int) -> None:
        """Reject an oversized write before any temporary allocation."""
        if not 0 <= size <= self.remaining:
            raise ValueError("index body exceeds byte budget")

    def write(self, data: bytes | memoryview) -> None:
        """Check all bytes at the serialization sink."""
        size = len(data)
        self.ensure(size)
        self._stream.write(data)
        self.remaining -= size

    def finish(self) -> bytes:
        """Return the accumulated bounded body."""
        return self._stream.getvalue()

    def count(self, value: int) -> None:
        """Write a bounded unsigned count."""
        if not 0 <= value <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("index count exceeds 64 bits")
        self.write(_LENGTH.pack(value))

    def block(self, value: bytes | memoryview) -> None:
        """Write a length followed by uninterpreted bytes."""
        self.ensure(_LENGTH.size + len(value))
        self.count(len(value))
        self.write(value)

    def value(self, value: object, *, depth: int = 0) -> None:
        """Write a supported primitive or a bounded nested container."""
        self._nodes -= 1
        if self._nodes < 0:
            raise ValueError("too many index value nodes")
        if depth >= _MAX_DEPTH:
            raise ValueError("index value nesting is too deep")
        if isinstance(value, np.generic):
            value = value.item()
        if not self.atom(value):
            self.container(value, depth)

    def atom(self, value: object) -> bool:
        """Write an atomic value, or report that it is a container."""
        if value is None or type(value) is bool:
            self.write({None: b"n", False: b"f", True: b"t"}[value])
        elif type(value) is int:
            self._integer(value)
        elif type(value) is float:
            self.ensure(1 + _FLOAT.size)
            self.write(b"r")
            self.write(_FLOAT.pack(value))
        elif type(value) is str:
            self._text(value)
        elif type(value) is bytes:
            self.ensure(1 + _LENGTH.size + len(value))
            self.write(b"b")
            self.block(value)
        else:
            return False
        return True

    def _text(self, value: str) -> None:
        """Check UTF-8 size before creating a temporary encoded string."""
        encoded = encode_utf8(value, self.remaining - 1 - _LENGTH.size)
        self.write(b"s")
        self.block(encoded)

    def _integer(self, value: int) -> None:
        """Preserve both signed and unsigned source integers."""
        if -(1 << 63) <= value < 0:
            self.write(b"i")
            self.write(_SIGNED.pack(value))
        elif 0 <= value < 1 << 64:
            self.write(b"u")
            self.count(value)
        else:
            raise ValueError("index integer exceeds 64 bits")

    def container(self, value: object, depth: int) -> None:
        """Write a dictionary, tuple, or list without implicit conversions."""
        if isinstance(value, dict):
            self.ensure(1 + _LENGTH.size + 2 * len(value))
            if len(value) > self._nodes // 2:
                raise ValueError("too many index value nodes")
            self.write(b"d")
            self.count(len(value))
            for key, item in value.items():
                _check_key(key)
                self.value(key, depth=depth + 1)
                self.value(item, depth=depth + 1)
        elif isinstance(value, (list, tuple)):
            self.ensure(1 + _LENGTH.size + len(value))
            if len(value) > self._nodes:
                raise ValueError("too many index value nodes")
            self.write(b"l" if isinstance(value, list) else b"p")
            self.count(len(value))
            for item in value:
                self.value(item, depth=depth + 1)
        else:
            raise _IndexValueError(f"unsupported index value: {type(value).__name__}")


class _IndexReader:
    """Consume each body byte at most once; never allocate from a claimed size."""

    remaining: memoryview
    _nodes: int

    def __init__(self, body: bytes) -> None:
        """Start consuming an existing body without a whole-body copy."""
        self.remaining = memoryview(body)
        self._nodes = _MAX_VALUE_NODES

    def take(self, size: int) -> memoryview:
        """Read exactly the requested bytes, or reject the body."""
        if not 0 <= size <= len(self.remaining):
            raise ValueError("truncated index body")
        result = self.remaining[:size]
        self.remaining = self.remaining[size:]
        return result

    def count(self) -> int:
        """Read one fixed-width unsigned count."""
        (value,) = _LENGTH.unpack(self.take(_LENGTH.size))
        return int(value)

    def items(self, *, width: int = 1) -> int:
        """Read a count with enough remaining bytes for its minimum width."""
        count = self.count()
        if count > len(self.remaining) // width:
            raise ValueError("index count exceeds remaining data")
        return count

    def _value_items(self, width: int) -> int:
        """Bound a generic container by the remaining value nodes."""
        count = self.items(width=width)
        if count > self._nodes // width:
            raise ValueError("too many index value nodes")
        return count

    def block(self) -> memoryview:
        """Read a length-prefixed byte range."""
        return self.take(self.count())

    def value(self, *, depth: int = 0) -> object:
        """Read one supported value with bounded nesting."""
        self._nodes -= 1
        if self._nodes < 0:
            raise ValueError("too many index value nodes")
        if depth >= _MAX_DEPTH:
            raise ValueError("index value nesting is too deep")
        tag = bytes(self.take(1))
        if tag in (b"l", b"p"):
            items = [self.value(depth=depth + 1) for _ in range(self._value_items(1))]
            return items if tag == b"l" else tuple(items)
        if tag == b"d":
            return self.mapping(depth)
        return self.atom(tag)

    def mapping(self, depth: int) -> dict[Any, Any]:
        """Read unique, scalar dictionary keys in wire order."""
        result: dict[Any, Any] = {}
        for _ in range(self._value_items(2)):
            key = self.value(depth=depth + 1)
            _check_key(key)
            if key in result:
                raise ValueError("duplicate index key")
            result[key] = self.value(depth=depth + 1)
        return result

    def atom(self, tag: bytes) -> object:
        """Decode a known scalar tag without any executable interpretation."""
        if tag in (b"n", b"f", b"t"):
            return {b"n": None, b"f": False, b"t": True}[tag]
        if tag == b"r":
            return _FLOAT.unpack(self.take(_FLOAT.size))[0]
        if tag == b"i":
            return _SIGNED.unpack(self.take(_SIGNED.size))[0]
        if tag == b"u":
            return self.count()
        if tag == b"s":
            return self.block().tobytes().decode()
        if tag == b"b":
            return self.block().tobytes()
        raise ValueError("invalid index value tag")


def _check_key(key: object) -> None:
    """Allow only string and integer keys on the wire."""
    if type(key) not in (str, int):
        raise ValueError("unsupported index mapping key")


def exact_value(left: object, right: object) -> bool:
    """Compare metadata without equating ints, floats, bools or signed zeros."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return _exact_mapping(left, cast(dict[Any, Any], right))
    if isinstance(left, (list, tuple)):
        sequence = cast(list[Any] | tuple[Any, ...], right)
        return len(left) == len(sequence) and all(
            exact_value(a, b) for a, b in zip(left, sequence, strict=True)
        )
    if isinstance(left, float):
        return _FLOAT.pack(left) == _FLOAT.pack(cast(float, right))
    return bool(left == right)


def _exact_mapping(left: dict[Any, Any], right: dict[Any, Any]) -> bool:
    """Compare mapping values and reject implicitly equal non-wire keys."""
    return left.keys() == right.keys() and all(
        type(key) in (str, int) and exact_value(left[key], value)
        for key, value in right.items()
    )
