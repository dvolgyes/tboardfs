"""Serialize small per-step NPZ tables from NumPy-generated layouts."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from struct import Struct
from zipfile import ZIP_STORED, BadZipFile, ZipFile
from zlib import crc32

import numpy as np

from tboardfs._step_schema import schema

_MAX_LAYOUTS = 32

_END_SIZE, _DIRECTORY_SIZE, _LOCAL_SIZE = 22, 46, 30
_CRC = Struct("<I")
_LOCAL_CRC_OFFSET, _DIRECTORY_CRC_OFFSET = 14, 16


@dataclass(frozen=True, slots=True)
class _Member:
    """Locate one member's array bytes and CRC fields in the template."""

    payload: int
    local_crc: int
    directory_crc: int
    header_crc: int


@dataclass(frozen=True, slots=True)
class _Layout:
    """Keep only a zero-filled archive and its immutable patch positions."""

    template: bytes
    members: tuple[_Member, ...]

    def encode(self, series: dict[str, np.ndarray], rows: int) -> bytes:
        """Copy the raw array bytes and update both CRC fields per member."""
        result = bytearray(self.template)
        size = rows * 8
        for column, member in zip(series.values(), self.members, strict=True):
            payload = column.data.cast("B")
            result[member.payload : member.payload + size] = payload
            checksum = crc32(payload, member.header_crc)
            _CRC.pack_into(result, member.local_crc, checksum)
            _CRC.pack_into(result, member.directory_crc, checksum)
        return bytes(result)


def _stock_npz(columns: dict[str, np.ndarray]) -> bytes:
    """Serialize a layout template or a probe with the standard writer."""
    with BytesIO() as buffer:
        np.savez(buffer, **columns)
        return buffer.getvalue()


def _build_layout(
    names: tuple[str, ...], rows: int, step_dtype: np.dtype
) -> _Layout | None:
    """Locate patches in a stored archive and verify them against stock NumPy."""
    columns = dict.fromkeys(names, np.zeros(rows, dtype=np.float64))
    columns["step"] = np.zeros(rows, dtype=step_dtype)
    template = _stock_npz(columns)
    try:
        with ZipFile(BytesIO(template)) as archive:
            entries = archive.infolist()
            if archive.comment or tuple(e.filename for e in entries) != tuple(
                name + ".npy" for name in names
            ):
                return None
    except (BadZipFile, NotImplementedError):
        return None

    if template[-_END_SIZE : -_END_SIZE + 4] != b"PK\x05\x06":
        return None
    sizes = [
        _DIRECTORY_SIZE + len(e.filename) + len(e.extra) + len(e.comment)
        for e in entries
    ]
    directory = len(template) - _END_SIZE - sum(sizes)
    ends = [*(e.header_offset for e in entries[1:]), directory]
    members = []
    for entry, stop, size in zip(entries, ends, sizes, strict=True):
        local = entry.header_offset
        start = stop - entry.file_size
        payload = stop - rows * 8
        if (
            entry.flag_bits
            or entry.compress_type != ZIP_STORED
            or entry.compress_size != entry.file_size
            or not 0 <= local <= start - _LOCAL_SIZE
            or start >= payload
            or template[local : local + 4] != b"PK\x03\x04"
            or template[directory : directory + 4] != b"PK\x01\x02"
        ):
            return None
        members.append(
            _Member(
                payload,
                local + _LOCAL_CRC_OFFSET,
                directory + _DIRECTORY_CRC_OFFSET,
                crc32(template[start:payload]),
            )
        )
        directory += size

    layout = _Layout(template, tuple(members))
    probe = {
        name: np.arange(i * rows + 1, (i + 1) * rows + 1, dtype=column.dtype)
        for i, (name, column) in enumerate(columns.items())
    }
    return layout if layout.encode(probe, rows) == _stock_npz(probe) else None


class StepArchives:
    """Share a bounded layout cache across the per-step exports of one run."""

    _layout: Callable[[tuple[str, ...], int, np.dtype], _Layout | None]

    def __init__(self) -> None:
        self._layout = lru_cache(maxsize=_MAX_LAYOUTS)(_build_layout)

    def encode(self, series: dict[str, np.ndarray]) -> bytes | None:
        """Return exact NPZ bytes, or None to request standard serialization."""
        names = schema(series)
        if names is None:
            return None
        step = series["step"]
        rows = len(step)
        layout = self._layout(names, rows, step.dtype)
        return None if layout is None else layout.encode(series, rows)
