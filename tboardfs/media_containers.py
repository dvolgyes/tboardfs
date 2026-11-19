"""Identify media containers not reliably distinguished by MIME."""

from math import isfinite
import re
from struct import unpack_from


_HDR_RESOLUTION = re.compile(rb"[+-]([XY]) [1-9][0-9]* [+-]([XY]) [1-9][0-9]*")
_MPEG_PACK = b"\x00\x00\x01\xba"
_HDR_SIGNATURES = (b"#?RADIANCE\n", b"#?RGBE\n", b"#?RADIANCE\r\n", b"#?RGBE\r\n")
_CAF_HEADER = b"caff\x00\x01\x00\x00desc\0\0\0\0\0\0\0\x20"


class _MediaHeaders:
    """Inspect only bounded headers, without decoding media."""

    @staticmethod
    def qoi(blob: bytes) -> bool:
        """Require dimensions, supported channel flags, and the end marker."""
        return (
            len(blob) >= 23
            and blob[4:8] != b"\0" * 4
            and blob[8:12] != b"\0" * 4
            and blob[12] in (3, 4)
            and blob[13] in (0, 1)
            and blob.endswith(b"\0\0\0\0\0\0\0\1")
        )

    @staticmethod
    def radiance(blob: bytes) -> bool:
        """Require the RGBE or XYZE encoding and two distinct resolution axes."""
        header, separator, raster = (
            blob[:4096].replace(b"\r\n", b"\n").partition(b"\n\n")
        )
        if not separator or not {
            b"FORMAT=32-bit_rle_rgbe",
            b"FORMAT=32-bit_rle_xyze",
        }.intersection(header.split(b"\n")):
            return False
        size, newline, pixels = raster.partition(b"\n")
        resolution = _HDR_RESOLUTION.fullmatch(size)
        return bool(
            newline and pixels and resolution and resolution[1] != resolution[2]
        )

    @staticmethod
    def pack_end(blob: bytes) -> int:
        """Read the versioned MPEG pack header, including MPEG-2 stuffing."""
        if len(blob) >= 14 and blob[4] & 0xC4 == 0x44:
            return _MediaHeaders.pack_end_v2(blob)
        if (
            len(blob) >= 12
            and blob[4] & 0xF1 == 0x21
            and all(blob[i] & mask for i, mask in ((6, 1), (8, 1), (9, 128), (11, 1)))
        ):
            return 12
        return 0

    @staticmethod
    def pack_end_v2(blob: bytes) -> int:
        """Require MPEG-2 marker bits and complete stuffing bytes."""
        if any(
            blob[i] & mask != mask
            for i, mask in ((6, 4), (8, 4), (9, 1), (12, 3), (13, 248))
        ):
            return 0
        end = 14 + (blob[13] & 7)
        return end if blob[14:end] == b"\xff" * (end - 14) else 0

    @staticmethod
    def program_stream(blob: bytes) -> bool:
        """Require a pack followed by a system header or multiplexed packet."""
        end = _MediaHeaders.pack_end(blob)
        return (
            0 < end <= len(blob) - 6
            and blob.startswith(b"\x00\x00\x01", end)
            and blob[end + 3] >= 0xBB
            and int.from_bytes(blob[end + 4 : end + 6], "big") <= len(blob) - end - 6
        )

    @staticmethod
    def transport_packet(blob: bytes, start: int) -> bool:
        """Check the sync byte and payload/adaptation layout of one packet."""
        if blob[start] != 0x47:
            return False
        control = blob[start + 3] & 0x30
        if control == 0x10:
            return True
        if control == 0x20:
            return blob[start + 4] == 183
        return control == 0x30 and blob[start + 4] <= 182

    @staticmethod
    def caf(blob: bytes) -> bool:
        """Require the mandatory stream description and basic audio attributes."""
        if len(blob) < 52 or not blob.startswith(_CAF_HEADER):
            return False
        rate, encoding, channels = unpack_from(">d4s12xI", blob, 20)
        return bool(isfinite(rate) and rate > 0 and encoding != b"\0" * 4 and channels)

    @staticmethod
    def transport_stream(blob: bytes) -> bool:
        """Confirm five packet headers at the TS, M2TS, or parity-packet stride."""
        for stride, offset in ((188, 0), (192, 4), (204, 0)):
            if len(blob) >= 5 * stride and all(
                _MediaHeaders.transport_packet(blob, start)
                for start in range(offset, offset + 5 * stride, stride)
            ):
                return True
        return False


def media_extension(blob: bytes) -> str | None:
    """Return an extension for a recognized ambiguous media container."""
    if blob.startswith(b"qoif") and _MediaHeaders.qoi(blob):
        return "qoi"
    if blob.startswith(_HDR_SIGNATURES) and _MediaHeaders.radiance(blob):
        return "hdr"
    if blob.startswith(b"caff") and _MediaHeaders.caf(blob):
        return "caf"
    if blob.startswith(_MPEG_PACK) and _MediaHeaders.program_stream(blob):
        return "mpg"
    if (blob.startswith(b"G") or blob[4:5] == b"G") and _MediaHeaders.transport_stream(
        blob
    ):
        return "ts"
    return None
