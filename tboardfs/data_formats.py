"""Recognize scientific files, compression wrappers and conservative mesh subsets."""

import json
import math
import re
import struct
import zlib
from types import MappingProxyType

_OBJ_NUMBER = re.compile(rb"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[Ee][+-]?[0-9]+)?")
_OBJ_INDEX = re.compile(rb"-?[1-9][0-9]{0,8}")
_TENSOR_BITS = MappingProxyType(
    {
        "BOOL": 8,
        "U8": 8,
        "I8": 8,
        "F8_E4M3": 8,
        "F8_E5M2": 8,
        "I16": 16,
        "U16": 16,
        "F16": 16,
        "BF16": 16,
        "I32": 32,
        "U32": 32,
        "F32": 32,
        "I64": 64,
        "U64": 64,
        "F64": 64,
    }
)
_STL_FACET = struct.Struct("<12fH")


class _Scientific:
    @staticmethod
    def mat(blob: bytes) -> bool:
        """Match a Level 5 header and its first matrix element."""
        if len(blob) < 136 or not blob.startswith(b"MATLAB 5.0 MAT-file"):
            return False
        if blob[124:128] == b"\x00\x01IM":
            order = "<"
        elif blob[124:128] == b"\x01\x00MI":
            order = ">"
        else:
            return False
        kind, size = struct.unpack_from(f"{order}II", blob, 128)
        return (kind == 14 and 16 <= size <= len(blob) - 136 and size % 8 == 0) or (
            kind == 15 and 8 <= size <= len(blob) - 136 and blob[136] == 0x78
        )

    @staticmethod
    def sqlite(blob: bytes) -> bool:
        """Check database page framing and the schema B-tree page type."""
        if len(blob) < 512 or not blob.startswith(b"SQLite format 3\0"):
            return False
        page_size = int.from_bytes(blob[16:18], "big")
        if page_size == 1:
            page_size = 65536
        return (
            512 <= page_size <= 65536
            and page_size & (page_size - 1) == 0
            and len(blob) % page_size == 0
            and blob[18] in (1, 2)
            and blob[19] in (1, 2)
            and blob[20] <= page_size - 480
            and blob[21:24] == b"\x40\x20\x20"
            and blob[100] in (5, 13)
        )


class _NetCDF:
    blob: bytes
    width: int
    offset_width: int
    pos: int
    limit: int
    types: tuple[int, ...]

    def __init__(self, blob: bytes) -> None:
        self.blob = blob
        version = blob[3]
        self.width = 8 if version == 5 else 4
        self.offset_width = 4 if version == 1 else 8
        self.types = (
            (0, 1, 1, 2, 4, 4, 8, 1, 2, 4, 8, 8)
            if version == 5
            else (0, 1, 1, 2, 4, 4, 8)
        )
        self.pos = 4 + self.width
        self.limit = min(len(blob), 65536)

    def advance(self, size: int) -> int:
        """Reserve a bounded range of header bytes and return its start."""
        start = self.pos
        self.pos += size
        if self.pos > self.limit:
            raise ValueError("CDF header exceeds bounds")
        return start

    def number(self, width: int) -> int:
        """Consume one big-endian integer."""
        start = self.advance(width)
        return int.from_bytes(self.blob[start : self.pos], "big")

    def count(self, expected: int) -> int:
        """Read a list tag and bound the count by the remaining header."""
        tag, count = self.number(4), self.number(self.width)
        if tag == count == 0:
            return 0
        if tag != expected or not 0 < count <= (self.limit - self.pos) // (
            2 * self.width
        ):
            raise ValueError("invalid CDF list")
        return count

    def name(self) -> None:
        """Skip a nonempty padded name."""
        size = self.number(self.width)
        start = self.advance((size + 3) & ~3)
        if not size or b"\0" in self.blob[start : start + size]:
            raise ValueError("invalid CDF name")

    def type_size(self) -> int:
        """Read a supported type tag and return its external byte size."""
        kind = self.number(4)
        if not 0 < kind < len(self.types):
            raise ValueError("unknown CDF type")
        return self.types[kind]

    def attributes(self) -> None:
        """Skip an attribute list without loading its values."""
        for _ in range(self.count(12)):
            self.name()
            size = self.type_size() * self.number(self.width)
            self.advance((size + 3) & ~3)

    def shape(self, dimensions: list[int]) -> tuple[int, bool]:
        """Read dimension IDs and cap the element count above the file size."""
        rank = self.number(self.width)
        elements, record = 1, False
        for axis in range(rank):
            index = self.number(self.width)
            if index >= len(dimensions):
                raise ValueError("unknown CDF dimension")
            length = dimensions[index]
            if length:
                elements = min(elements * length, len(self.blob) + 1)
            elif axis == 0:
                record = True
            else:
                raise ValueError("CDF unlimited dimension must be first")
        return elements, record

    def variable(self, dimensions: list[int], records: int) -> int:
        """Check the fixed extent or first nonempty record slab without reading data."""
        self.name()
        elements, record = self.shape(dimensions)
        self.attributes()
        data_size = self.type_size() * elements
        advertised = self.number(self.width)
        begin = self.number(self.offset_width)
        if record:
            if records and begin + data_size > len(self.blob):
                raise ValueError("truncated CDF record data")
        else:
            padded = (data_size + 3) & ~3
            expected = min(padded, 0xFFFFFFFF) if self.width == 4 else padded
            if advertised != expected or begin + padded > len(self.blob):
                raise ValueError("invalid CDF fixed variable extent")
        return begin

    def header(self) -> bool:
        """Read the header and ensure each data offset comes after it."""
        records = int.from_bytes(self.blob[4 : self.pos], "big")
        dimensions = []
        for _ in range(self.count(10)):
            self.name()
            dimensions.append(self.number(self.width))
        self.attributes()
        first_data = (1 << (8 * self.offset_width)) - 1
        for _ in range(self.count(11)):
            begin = self.variable(dimensions, records)
            if begin % 4:
                return False
            first_data = min(first_data, begin)
        return self.pos <= first_data

    @classmethod
    def matches(cls, blob: bytes) -> bool:
        """Recognize classic, 64-bit offset, and 64-bit data NetCDF."""
        if len(blob) < 32 or blob[3] not in (1, 2, 5):
            return False
        try:
            return cls(blob).header()
        except ValueError:
            return False


class _SafeTensor:
    @staticmethod
    def unique(items: list[tuple[str, object]]) -> dict[str, object]:
        """Reject ambiguous repeated keys at any level of the header."""
        result = dict(items)
        if len(items) != len(result):
            raise ValueError("duplicate JSON member")
        return result

    @staticmethod
    def extent(entry: object, available: int) -> tuple[int, int] | None:
        """Validate one dtype, shape, and byte extent against the buffer."""
        if not isinstance(entry, dict) or entry.keys() != {
            "dtype",
            "shape",
            "data_offsets",
        }:
            return None
        dtype, shape, offsets = (entry[k] for k in ("dtype", "shape", "data_offsets"))
        if not isinstance(dtype, str) or dtype not in _TENSOR_BITS:
            return None
        if not isinstance(shape, list) or len(shape) > 64:
            return None
        if any(type(dim) is not int or not 0 <= dim <= (1 << 63) - 1 for dim in shape):
            return None
        if not isinstance(offsets, list) or len(offsets) != 2:
            return None
        start, end = offsets
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start <= end <= available
        ):
            return None
        expected = _TENSOR_BITS[dtype] // 8
        if 0 in shape:
            expected = 0
        else:
            for dim in shape:
                expected *= dim
                if expected > end - start:
                    return None
        return (start, end) if expected == end - start else None

    @classmethod
    def matches(cls, blob: bytes) -> bool:
        """Parse a bounded Safetensors header and validate its entire layout."""
        if len(blob) < 10 or blob[8] != 123:
            return False
        size = int.from_bytes(blob[:8], "little")
        if not 2 <= size <= min(65536, len(blob) - 8):
            return False
        try:
            header: object = json.loads(
                blob[8 : 8 + size].decode("utf-8"), object_pairs_hook=cls.unique
            )
        except (ValueError, RecursionError):
            return False
        if not isinstance(header, dict):
            return False
        metadata: object = header.pop("__metadata__", {})
        if not isinstance(metadata, dict) or not all(
            isinstance(v, str) for v in metadata.values()
        ):
            return False
        return cls.covers(header, len(blob) - 8 - size)

    @classmethod
    def covers(cls, header: dict[str, object], available: int) -> bool:
        """Require ordered extents to cover every byte without overlap or gaps."""
        extents = []
        for entry in header.values():
            extent = cls.extent(entry, available)
            if extent is None:
                return False
            extents.append(extent)
        if not extents:
            return False
        pos = 0
        for start, end in sorted(extents):
            if start != pos:
                return False
            pos = end
        return pos == available


class _Mesh:
    @staticmethod
    def stl(blob: bytes) -> bool:
        """Check binary STL framing and bound the number of facets inspected."""
        if len(blob) < 134 or (len(blob) - 84) % _STL_FACET.size:
            return False
        count = struct.unpack_from("<I", blob, 80)[0]
        if not 0 < count <= 65536 or len(blob) != 84 + count * _STL_FACET.size:
            return False
        return _Mesh.facets(memoryview(blob)[84:])

    @staticmethod
    def facets(data: memoryview) -> bool:
        """Require finite facets with at least one nondegenerate triangle."""
        nondegenerate = False
        for facet in _STL_FACET.iter_unpack(data):
            coords = facet[:12]
            if not all(map(math.isfinite, coords)):
                return False
            _, _, _, ax, ay, az, bx, by, bz, cx, cy, cz = coords
            ux, uy, uz = bx - ax, by - ay, bz - az
            vx, vy, vz = cx - ax, cy - ay, cz - az
            nondegenerate |= bool(
                uy * vz - uz * vy or uz * vx - ux * vz or ux * vy - uy * vx
            )
        return nondegenerate

    @classmethod
    def obj(cls, blob: bytes) -> bool:
        """Recognize only small plain OBJ files with vertices and faces."""
        if len(blob) > 262144 or not blob.isascii() or b"\0" in blob:
            return False
        vertices = faces = 0
        for raw in blob.splitlines():
            line = raw.partition(b"#")[0].strip()
            if not line:
                continue
            if len(line) > 1024:
                return False
            kind, *fields = line.split()
            if kind == b"v" and cls.vertex(fields):
                vertices += 1
            elif kind == b"f" and cls.face(fields, vertices):
                faces += 1
            else:
                return False
        return bool(faces)

    @staticmethod
    def vertex(fields: list[bytes]) -> bool:
        """Require three finite decimal coordinates."""
        return len(fields) == 3 and all(
            _OBJ_NUMBER.fullmatch(field) and math.isfinite(float(field))
            for field in fields
        )

    @staticmethod
    def face(fields: list[bytes], vertices: int) -> bool:
        """Require distinct, already defined vertex references."""
        if len(fields) < 3 or not all(map(_OBJ_INDEX.fullmatch, fields)):
            return False
        indices = {
            value if value > 0 else vertices + value + 1 for value in map(int, fields)
        }
        return (
            len(indices) == len(fields)
            and min(indices) >= 1
            and max(indices) <= vertices
        )


class _Compression:
    @staticmethod
    def gzip(blob: bytes) -> bool:
        """Check gzip header framing without expanding the deflate stream."""
        if len(blob) < 20 or blob[3] & 0xE0:
            return False
        remaining = len(blob) - 8
        flags, pos = blob[3], 10
        if flags & 4:
            if pos + 2 > remaining:
                return False
            pos += 2 + int.from_bytes(blob[pos : pos + 2], "little")
        for flag in (8, 16):
            if flags & flag:
                end = blob.find(b"\0", pos, min(remaining, 65536))
                if end < 0:
                    return False
                pos = end + 1
        if flags & 2:
            pos += 2
        return pos + 2 <= remaining and blob[pos] & 6 != 6

    @staticmethod
    def bzip(blob: bytes) -> bool:
        """Check the bzip2 level and first block or end marker."""
        return (
            len(blob) >= 14
            and blob[3] in b"123456789"
            and blob[4:10]
            in (
                b"1AY&SY",
                b"\x17rE8P\x90",
            )
        )

    @staticmethod
    def xz(blob: bytes) -> bool:
        """Check XZ header and footer framing without decoding the stream."""
        end = len(blob)
        while end >= max(32, len(blob) - 65536) and blob[end - 4 : end] == b"\0" * 4:
            end -= 4
        if end < 32 or end % 4 or blob[end - 2 : end] != b"YZ":
            return False
        flags = blob[6:8]
        footer = blob[end - 8 : end - 2]
        return (
            flags[0] == 0
            and flags[1] < 16
            and flags == footer[4:]
            and zlib.crc32(flags) == int.from_bytes(blob[8:12], "little")
            and zlib.crc32(footer) == int.from_bytes(blob[end - 12 : end - 8], "little")
            and (int.from_bytes(footer[:4], "little") + 1) * 4 <= end - 24
        )


_PREFIXES = (
    (b"MATLAB 5.0 MAT-file", _Scientific.mat, "mat"),
    (b"SQLite format 3\0", _Scientific.sqlite, "sqlite"),
    (b"CDF", _NetCDF.matches, "nc"),
    (b"\x1f\x8b\x08", _Compression.gzip, "gz"),
    (b"BZh", _Compression.bzip, "bz2"),
    (b"\xfd7zXZ\0", _Compression.xz, "xz"),
)


def data_extension(blob: bytes) -> str | None:
    """Infer a file suffix without opening containers or decoding their payloads."""
    for prefix, matches, extension in _PREFIXES:
        if blob.startswith(prefix):
            return extension if matches(blob) else None
    if _SafeTensor.matches(blob):
        return "safetensors"
    if _Mesh.stl(blob):
        return "stl"
    return "obj" if _Mesh.obj(blob) else None
