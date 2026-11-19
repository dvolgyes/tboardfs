"""Bounded structural recognition for selected scientific binary containers."""

import re
import struct

_PLY_FORMAT = re.compile(
    rb"ply\r?\nformat (?:ascii|binary_(?:little|big)_endian) 1\.0\r?\n"
)
_PLY_ELEMENT = re.compile(rb"(?m)^element \w+ [0-9]+\r?\nproperty [^\r\n]+\r?\n")
_PLY_END = re.compile(rb"(?m)^end_header\r?\n")
_PDF_HEADER = re.compile(rb"%PDF-(?:1\.[0-7]|2\.0)[\r\n]")
_PDF_END = re.compile(rb"startxref\s+([0-9]{1,20})\s+%%EOF\s*\Z")
_HDF = b"\x89HDF\r\n\x1a\n"


class _Framing:
    """Read only bounded framing fields, not scientific payloads."""

    @staticmethod
    def ply(blob: bytes) -> bool:
        """Check the format declaration and schema in the first 16 KiB."""
        if not _PLY_FORMAT.match(blob):
            return False
        end = _PLY_END.search(blob, 0, 16384)
        return end is not None and _PLY_ELEMENT.search(blob, 0, end.start()) is not None

    @staticmethod
    def glb(blob: bytes) -> bool:
        """Recognize the v2 container and its first aligned JSON chunk."""
        if len(blob) < 24 or blob[16:20] != b"JSON":
            return False
        version, total, size = struct.unpack_from("<III", blob, 4)
        return (
            version == 2
            and total == len(blob)
            and total % 4 == 0
            and 4 <= size <= total - 20
            and size % 4 == 0
            and blob[20 : min(20 + size, 84)].lstrip().startswith(b"{")
        )

    @staticmethod
    def hdf(blob: bytes) -> bool:
        """Probe offset zero and legal powers of two, without reading a user block."""
        offset = 0
        while offset + 24 <= len(blob):
            if blob.startswith(_HDF, offset):
                version = blob[offset + 8]
                widths = offset + (13 if version < 2 else 9)
                sizes = blob[widths : widths + 2]
                if version <= 3 and all(n in (2, 4, 8, 16) for n in sizes):
                    return True
            offset = 512 if offset == 0 else offset * 2
        return False

    @staticmethod
    def pdf(blob: bytes) -> bool:
        """Require a version line and a final cross-reference pointer."""
        if not _PDF_HEADER.match(blob):
            return False
        end = _PDF_END.search(blob, max(0, len(blob) - 1024))
        return end is not None and 0 < int(end[1]) < end.start()

    @staticmethod
    def parquet(blob: bytes) -> bool:
        """Check both file markers and the compact Thrift metadata envelope."""
        if not blob.endswith(b"PAR1"):
            return False
        size = int.from_bytes(blob[-8:-4], "little")
        start = len(blob) - 8 - size
        return start >= 4 and size >= 3 and blob[start] == 0x15 and blob[-9] == 0

    @staticmethod
    def arrow(blob: bytes) -> bool:
        """Check file padding, trailer size and the aligned FlatBuffer root."""
        if not blob.endswith(b"ARROW1"):
            return False
        size = int.from_bytes(blob[-10:-6], "little")
        start = len(blob) - 10 - size
        if start < 8 or start % 8 or size < 8:
            return False
        root = int.from_bytes(blob[start : start + 4], "little")
        return 4 <= root <= size - 4 and root % 4 == 0


def native_extension(blob: bytes) -> str | None:
    """Return a scientific file suffix only when its framing matches."""
    if blob.startswith(b"glTF") and _Framing.glb(blob):
        return "glb"
    if blob.startswith(b"ply") and _Framing.ply(blob):
        return "ply"
    if blob.startswith(b"%PDF-") and _Framing.pdf(blob):
        return "pdf"
    if blob.startswith(b"PAR1") and _Framing.parquet(blob):
        return "parquet"
    if blob.startswith(b"ARROW1\0\0") and _Framing.arrow(blob):
        return "arrow"
    return "h5" if _Framing.hdf(blob) else None
