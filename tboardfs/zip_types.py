"""Classify a bounded peek of ZIP-based embedded formats."""

import ast
import struct
import zlib
from dataclasses import dataclass
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile, ZipInfo

from tboardfs.opc import office_extension

_MAX_ENTRIES = 128
_MAX_DIRECTORY = 1 << 16
_MAX_NPY_INPUT = 8192
_MAX_NPY_HEADER = 4096
_EOCD = b"PK\x05\x06"


def zip_extension(blob: bytes) -> str:
    """Return a conservative extension inferred from ZIP package contents."""
    entries = _directory_entries(blob)
    if entries is None:
        return "bin"
    if entries == -1:
        return "zip"
    try:
        with ZipFile(BytesIO(blob)) as archive:
            return _Package(archive, blob).classify(entries)
    except NotImplementedError:
        return "zip"
    except (BadZipFile, EOFError, UnicodeError, zlib.error):
        return "bin"


def _directory_entries(blob: bytes) -> int | None:
    """Validate central directory bounds before zipfile can read or redirect it."""
    end = len(blob) - 22
    if not (blob.startswith(_EOCD, end) and blob.endswith(b"\0\0")):
        end = blob.rfind(_EOCD, max(0, end - 65535))
    if end < 0 or end + 22 > len(blob):
        return None
    disk, cd_disk, n_disk, count, size, offset, comment = struct.unpack_from(
        "<4H2IH", blob, end + 4
    )
    if end + 22 + comment != len(blob) or disk or cd_disk or count != n_disk:
        return None
    if end >= 20 and blob.startswith(b"PK\x06\x07", end - 20):
        return _zip64_entries(blob, end - 20)
    return _Package.directory_count(blob, end, offset, size, count)


def _zip64_entries(blob: bytes, locator: int) -> int | None:
    """Inspect only fixed trailers whose location agrees with zipfile's end-relative lookup."""
    disk, offset, disks = struct.unpack_from("<IQI", blob, locator + 4)
    if disk or disks != 1 or offset + 56 > locator:
        return None
    if not blob.startswith(b"PK\x06\x06", offset):
        return None
    size = struct.unpack_from("<Q", blob, offset + 4)[0]
    disk, cd_disk, n_disk, count, cd_size, cd_offset = struct.unpack_from(
        "<II4Q", blob, offset + 16
    )
    if disk or cd_disk or n_disk != count:
        return None
    if offset + 12 + size != locator or cd_offset + cd_size != offset:
        return None
    entries = _Package.directory_count(blob, offset, cd_offset, cd_size, count)
    if entries is None:
        return None
    classic_count, classic_size, classic_offset = struct.unpack_from(
        "<HII", blob, locator + 30
    )
    if (
        classic_count not in (count, 0xFFFF)
        or classic_size not in (cd_size, 0xFFFFFFFF)
        or classic_offset not in (cd_offset, 0xFFFFFFFF)
    ):
        return -1
    return entries if size == 44 else -1


@dataclass(slots=True)
class _Package:
    """Inspect an archive only after the directory has passed the size budget."""

    archive: ZipFile
    blob: bytes

    @staticmethod
    def directory_count(
        blob: bytes, end: int, offset: int, size: int, count: int
    ) -> int | None:
        """Bound the actual record count, not just the count in the trailer."""
        if offset + size != end or count * 46 > size:
            return None
        if count == 0:
            return 0 if size == offset == 0 else None
        if not blob.startswith(b"PK\x01\x02", offset) or not blob.startswith(
            b"PK\x03\x04"
        ):
            return None
        if size > _MAX_DIRECTORY or count > _MAX_ENTRIES:
            return -1
        for _ in range(count):
            if offset + 46 > end or not blob.startswith(b"PK\x01\x02", offset):
                return None
            offset += 46 + sum(struct.unpack_from("<3H", blob, offset + 28))
        return count if offset == end else None

    def classify(self, count: int) -> str:
        """Check local framing and resolve a single supported embedded type."""
        infos = self.archive.infolist()
        if len(infos) != count or any(not self._local_header(info) for info in infos):
            return "bin"
        if not all(self._inspectable(info) for info in infos):
            return "zip"
        names = set(self.archive.namelist())
        if not names or len(names) != count:
            return "zip"
        for info in infos:
            self.archive.open(info).close()
        if all(name.endswith(".npy") for name in names):
            return "npz" if all(self._npy_header(info) for info in infos) else "zip"
        return office_extension(self.archive, names)

    @staticmethod
    def _inspectable(info: ZipInfo) -> bool:
        return (
            not info.flag_bits & 0x2041
            and info.compress_type in (ZIP_STORED, ZIP_DEFLATED)
            and info.orig_filename == info.filename
        )

    def _local_header(self, info: ZipInfo) -> bool:
        start = info.header_offset
        if start + 30 > len(self.blob) or not self.blob.startswith(
            b"PK\x03\x04", start
        ):
            return False
        flags, method = struct.unpack_from("<HH", self.blob, start + 6)
        if flags != info.flag_bits or method != info.compress_type:
            return False
        if flags & 8:
            return True
        crc, compressed, size = struct.unpack_from("<III", self.blob, start + 14)
        return (
            crc == info.CRC
            and compressed in (info.compress_size, 0xFFFFFFFF)
            and size in (info.file_size, 0xFFFFFFFF)
        )

    def _npy_prefix(self, info: ZipInfo) -> bytes:
        limit = min(info.file_size, 12 + _MAX_NPY_HEADER)
        if limit == 0:
            return b""
        start = info.header_offset + 30
        start += sum(struct.unpack_from("<HH", self.blob, start - 4))
        if info.compress_type == ZIP_STORED:
            return self.blob[start : start + min(limit, info.compress_size)]
        stop = start + min(info.compress_size, _MAX_NPY_INPUT)
        return zlib.decompressobj(-15).decompress(
            memoryview(self.blob)[start:stop], limit
        )

    def _npy_header(self, info: ZipInfo) -> bool:
        prefix = self._npy_prefix(info)
        if prefix[:6] != b"\x93NUMPY" or prefix[6:8] not in (b"\1\0", b"\2\0", b"\3\0"):
            return False
        width = 2 if prefix[6] == 1 else 4
        start = 8 + width
        length = int.from_bytes(prefix[8:start], "little")
        if not 0 < length <= _MAX_NPY_HEADER or (start + length) % 16:
            return False
        header = prefix[start : start + length]
        if len(header) != length or not header.endswith(b"\n"):
            return False
        encoding = "utf-8" if prefix[6] == 3 else "latin1"
        return self._array_fields(header, encoding)

    @staticmethod
    def _array_fields(header: bytes, encoding: str) -> bool:
        try:
            fields = ast.literal_eval(header.decode(encoding))
        except (SyntaxError, ValueError, UnicodeError, RecursionError, TypeError):
            return False
        if not isinstance(fields, dict) or fields.keys() != {
            "shape",
            "fortran_order",
            "descr",
        }:
            return False
        shape = fields["shape"]
        return (
            isinstance(fields["fortran_order"], bool)
            and isinstance(fields["descr"], (str, list))
            and isinstance(shape, tuple)
            and all(type(dim) is int and dim >= 0 for dim in shape)
        )
