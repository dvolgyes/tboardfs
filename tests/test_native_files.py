"""Check scientific file inference without loading third-party producers."""

from pathlib import Path
import struct
from zlib import compress, crc32

import pytest

from tboardfs.classify import detect_extension

_DATA = Path(__file__).parent / "data" / "file-types" / "scientific"


@pytest.mark.parametrize(
    "name",
    [
        "terrain.glb",
        "ascii.ply",
        "little.ply",
        "big.ply",
        "survey.pdf",
        "profile.h5",
        "user-block.h5",
        "stations.parquet",
        "stations.arrow",
    ],
)
def test_scientific_file(name: str) -> None:
    """Real producer files keep their native suffixes."""
    assert detect_extension((_DATA / name).read_bytes()) == name.split(".")[-1]


def test_ply_ascii_header() -> None:
    """An ordinary CRLF-terminated text point cloud is still a PLY file."""
    blob = (
        b"ply\r\nformat ascii 1.0\r\nelement vertex 1\r\n"
        b"property float x\r\nend_header\r\n2.5\r\n"
    )
    assert detect_extension(blob) == "ply"


def test_non_native_fragments() -> None:
    """Headers, streams, prose and unframed data are not scientific files."""
    assert detect_extension(b'{"asset":{"version":"2.0"}}') == "json"
    for name in ("terrain.glb", "stations.arrow", "stations.parquet"):
        blob = (_DATA / name).read_bytes()
        assert detect_extension(blob[:-1]) != name.split(".")[-1]
    assert detect_extension((_DATA / "stations.stream").read_bytes()) != "arrow"
    assert detect_extension(b"ply\nplease send the report\nend_header\n") != "ply"
    assert detect_extension(b"%PDF-1.7\n") != "pdf"
    assert detect_extension(b"PAR1PAR1") != "parquet"
    for name, trailer in (("stations.parquet", 8), ("stations.arrow", 10)):
        blob = (_DATA / name).read_bytes()
        impossible = blob[:-trailer] + b"\xff" * 4 + blob[4 - trailer :]
        assert detect_extension(impossible) != name.split(".")[-1]
    hdf = (_DATA / "profile.h5").read_bytes()
    assert detect_extension(b"\0" + hdf) != "h5"
    assert detect_extension(hdf[:9]) != "h5"


def _png_chunk(kind: bytes, body: bytes) -> bytes:
    return (
        struct.pack(">I4s", len(body), kind)
        + body
        + struct.pack(">I", crc32(body, crc32(kind)))
    )


_PNG_START = (
    b"\x89PNG\r\n\x1a\n"
    + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
    + _png_chunk(b"IDAT", compress(b"\x00\x00"))
)
_PNG_END = _png_chunk(b"IEND", b"")
_ANCILLARY_START = len(_PNG_START) + 8


@pytest.mark.parametrize(
    ("offset", "suffix"), [(512, "h5"), (2048, "h5"), (513, "png")]
)
def test_png_hdf_user_block(offset: int, suffix: str) -> None:
    """HDF5 inside a valid PNG chunk takes precedence at legal user-block offsets."""
    hdf = (_DATA / "profile.h5").read_bytes()
    ancillary = bytes(offset - _ANCILLARY_START) + hdf
    blob = _PNG_START + _png_chunk(b"vpAg", ancillary) + _PNG_END
    assert detect_extension(blob) == suffix


@pytest.mark.parametrize(("count", "suffix"), [(2, "stl"), (3, "png")])
def test_png_binary_stl_header(count: int, suffix: str) -> None:
    """Binary STL permits a PNG header but still requires exact facet framing."""
    ancillary = bytearray(184 - len(_PNG_END) - 4 - _ANCILLARY_START)
    struct.pack_into("<I", ancillary, 80 - _ANCILLARY_START, count)
    struct.pack_into("<9f", ancillary, 96 - _ANCILLARY_START, 0, 0, 0, 1, 0, 0, 0, 1, 0)
    blob = _PNG_START + _png_chunk(b"vpAg", bytes(ancillary)) + _PNG_END
    assert detect_extension(blob) == suffix
