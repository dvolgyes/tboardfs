"""Export native media unchanged and distinguish ambiguous container signatures."""

import json
from hashlib import sha256

import pytest
from test_tf_producer_cases import CORPUS, _events

from tboardfs import TensorBoardFS
from tboardfs.media_containers import media_extension

_CASES = "media-containers"


def _payloads() -> dict[str, bytes]:
    return {
        value.tag: (
            value.image.encoded_image_string
            or value.audio.encoded_audio_string
            or value.tensor.tensor_content
        )
        for event in _events(_CASES)
        for value in event.summary.value
    }


@pytest.mark.parametrize(
    ("tab", "tag", "extension", "signature"),
    [
        ("images", "qoi", "qoi", b"qoif"),
        ("images", "ico-bmp", "ico", b"\x00\x00\x01\x00"),
        ("images", "ico-png", "ico", b"\x00\x00\x01\x00"),
        ("images", "hdr", "hdr", b"#?RADIANCE\n"),
        ("images", "svg", "svg", b"<?xml"),
        ("audio", "caf", "caf", b"caff"),
        ("audio", "ogg", "ogg", b"OggS"),
        ("audio", "webm", "webm", b"\x1a\x45\xdf\xa3"),
        ("videos", "program-v1", "mpg", b"\x00\x00\x01\xba"),
        ("videos", "program-v2", "mpg", b"\x00\x00\x01\xba"),
        ("videos", "transport", "ts", b"G"),
        ("videos", "windows-media", "asf", b"\x30\x26\xb2\x75\x8e\x66\xcf\x11"),
        ("videos", "elementary", "bin", b"\x00\x00\x01\xb3"),
    ],
)
def test_media_container_exports(
    tab: str, tag: str, extension: str, signature: bytes
) -> None:
    """Keep container names rather than embedded image or codec names."""
    root = CORPUS / _CASES
    payload = _payloads()[tag]
    assert payload.startswith(signature)
    assert (
        sha256(payload).hexdigest()
        == json.loads((root / "sha256.json").read_text())[tag]
    )
    fs = TensorBoardFS(root)
    name = f"000001.{extension}"
    directory = f"/{tab}/{tag}"
    assert fs.readdir(directory)[2:] == [name]
    assert fs.read(f"{directory}/{name}", len(payload) + 1, 0) == payload


def test_icon_backings_and_opus_controls() -> None:
    """Exercise genuinely different icon backings and both Opus containers."""
    payloads = _payloads()
    assert payloads["ico-png"][22:30] == b"\x89PNG\r\n\x1a\n"
    assert payloads["ico-bmp"][22:26] == b"\x28\0\0\0"
    assert b"OpusHead" in payloads["ogg"]
    assert b"A_OPUS" in payloads["webm"]


def test_media_rejects_incomplete_or_ambiguous_headers() -> None:
    """Do not infer a format from coincidental sync or truncated magic alone."""
    payloads = _payloads()
    for blob in (
        payloads["qoi"][:-1],
        payloads["caf"][:51],
        payloads["hdr"].replace(b"-Y 32 +X 48", b"-Y 32 +Y 48", 1),
        payloads["program-v2"][:14],
        payloads["program-v1"][:12],
        payloads["elementary"],
        payloads["transport"][: 5 * 188 - 1],
        (b"G" + b"\0" * 187) * 5,
        (b"G\0\x10\x30\xff" + b"\0" * 183) * 5,
    ):
        assert media_extension(blob) is None


def test_transport_packet_strides() -> None:
    """Accept repeated packet headers with timestamps or Reed–Solomon trailers."""
    ts = _payloads()["transport"]
    packets = [ts[i : i + 188] for i in range(0, 5 * 188, 188)]
    assert media_extension(b"".join(b"\0" * 4 + packet for packet in packets)) == "ts"
    assert media_extension(b"".join(packet + b"\0" * 16 for packet in packets)) == "ts"
