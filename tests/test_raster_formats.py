"""Verify that native raster encodings survive export without image libraries."""

from hashlib import sha256
import struct

import pytest

from tboardfs import TensorBoardFS
from test_tf_producer_cases import CORPUS, _events


@pytest.fixture(scope="module")
def fs() -> TensorBoardFS:
    return TensorBoardFS(CORPUS / "rasters")


@pytest.fixture(scope="module")
def payloads() -> dict[str, bytes]:
    return {
        v.tag: v.image.encoded_image_string
        for e in _events("rasters")
        for v in e.summary.value
    }


@pytest.mark.parametrize(
    ("tag", "depth", "color", "width"),
    [
        ("gray-1", 1, 0, 2),
        ("gray-2", 2, 0, 4),
        ("gray-4", 4, 0, 16),
        ("gray-8", 8, 0, 4),
        ("gray-16", 16, 0, 4),
        ("rgb-16", 16, 2, 3),
        ("palette-2", 2, 3, 4),
        ("gray-alpha-16", 16, 4, 2),
        ("rgba-16", 16, 6, 2),
        ("palette-1", 1, 3, 2),
        ("palette-4", 4, 3, 4),
        ("palette-8", 8, 3, 4),
        ("rgb-8", 8, 2, 3),
        ("gray-alpha-8", 8, 4, 2),
        ("rgba-8", 8, 6, 2),
    ],
)
def test_png_color_and_depth(
    fs: TensorBoardFS,
    payloads: dict[str, bytes],
    tag: str,
    depth: int,
    color: int,
    width: int,
) -> None:
    blob = fs.read(f"/images/png-{tag}/000001.png", 65536, 0)
    assert blob == payloads[f"png-{tag}"]
    assert blob.startswith(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
    assert struct.unpack_from(">IIBB", blob, 16) == (width, 1, depth, color)
    if color == 3:
        offset = 8
        while blob[offset + 4 : offset + 8] != b"PLTE":
            length = struct.unpack_from(">I", blob, offset)[0]
            offset += length + 12
        length = struct.unpack_from(">I", blob, offset)[0]
        assert length == width * 3
        colors = {blob[i : i + 3] for i in range(offset + 8, offset + 8 + length, 3)}
        expected = {b"\xff\x00\x00", b"\x00\xff\x00"}
        if depth > 1:
            expected.update((b"\x00\x00\xff", b"\xff\xff\x00"))
        assert colors == expected


@pytest.mark.parametrize(
    ("tag", "marker", "components"),
    [
        ("gray-baseline", 0xC0, 1),
        ("rgb-baseline", 0xC0, 3),
        ("cmyk-baseline", 0xC0, 4),
        ("rgb-progressive", 0xC2, 3),
        ("gray-progressive", 0xC2, 1),
        ("cmyk-progressive", 0xC2, 4),
    ],
)
def test_jpeg_frame_encoding(
    fs: TensorBoardFS,
    payloads: dict[str, bytes],
    tag: str,
    marker: int,
    components: int,
) -> None:
    blob = fs.read(f"/images/jpeg-{tag}/000001.jpg", 65536, 0)
    assert blob == payloads[f"jpeg-{tag}"]
    assert blob[:2] == b"\xff\xd8"
    pos = 2
    while blob[pos + 1] not in (0xC0, 0xC2):
        assert blob[pos] == 0xFF and blob[pos + 1] != 0xDA
        pos += 2 + struct.unpack_from(">H", blob, pos + 2)[0]
    assert struct.unpack_from(">BBHBHHB", blob, pos) == (
        0xFF,
        marker,
        8 + 3 * components,
        8,
        8,
        16,
        components,
    )
    if components == 4:
        assert blob[pos + 10 : pos + 22 : 3] == b"CMYK"


@pytest.mark.parametrize(
    ("tag", "tab", "suffix", "digest"),
    [
        (
            "gif",
            "videos",
            "gif",
            "c49cf6914122e3b10c6b3a1b9e0b1f4089bc2dd7cee1a551132f6f5902c3ab14",
        ),
        (
            "apng",
            "images",
            "png",
            "88f98be9654d70f2c50da1687f9c867bbcfca523b84f2cadd91ef8a2d55c3a70",
        ),
        (
            "webp",
            "images",
            "webp",
            "c3634d8c97b054b362deaac33ac4e5ae91248cd660870fa148f8d01e740652b8",
        ),
    ],
)
def test_two_timed_frames_survive_export(
    fs: TensorBoardFS,
    payloads: dict[str, bytes],
    tag: str,
    tab: str,
    suffix: str,
    digest: str,
) -> None:
    name = f"animated-{tag}"
    blob = fs.read(f"/{tab}/{name}/000001.{suffix}", 65536, 0)
    assert blob == payloads[name]
    assert sha256(blob).hexdigest() == digest
    assert name not in fs.readdir("/images" if tab == "videos" else "/videos")
    if tag == "gif":
        assert blob.startswith(b"GIF89a")
        assert [
            int.from_bytes(block[1:3], "little")
            for block in blob.split(b"\x21\xf9\x04")[1:]
        ] == [12, 24]
    elif tag == "apng":
        assert b"acTL\x00\x00\x00\x02" in blob
        assert blob.count(b"fcTL") == 2
    else:
        assert blob[:4] == b"RIFF" and blob[8:12] == b"WEBP"
        assert blob.count(b"ANMF") == 2
