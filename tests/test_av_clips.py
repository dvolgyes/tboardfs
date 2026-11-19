"""Export verified moving clips with both audio and video streams intact."""

from hashlib import sha256
import json
from pathlib import Path

import pytest

from tboardfs import TensorBoardFS


ROOT = Path(__file__).parent / "data" / "producer-cases" / "av-clips"


@pytest.mark.parametrize(
    ("suffix", "markers"),
    [
        ("mp4", (b"ftypisom", b"avc1", b"mp4a")),
        ("mov", (b"ftypqt  ", b"avc1", b"mp4a")),
        ("avi", (b"RIFF", b"AVI ", b"auds", b"vids")),
        ("webm", (b"webm", b"V_VP9", b"A_OPUS")),
        ("mkv", (b"matroska", b"V_MPEG4/ISO/AVC", b"A_FLAC")),
        ("ogg", (b"OggS", b"\x80theora", b"\x01vorbis")),
    ],
)
def test_export_matches_decoded_audiovisual_clip(
    suffix: str, markers: tuple[bytes, ...]
) -> None:
    fs = TensorBoardFS(ROOT)
    path = f"/videos/{suffix}/000008.{suffix}"
    size = fs.getattr(path)["st_size"]
    assert 0 < size < 500 * 1024
    blob = fs.read(path, size, 0)
    oracle = json.loads((ROOT / "decoded.json").read_bytes())[suffix]
    assert sha256(blob).hexdigest() == oracle["sha256"]
    assert all(marker in blob for marker in markers)
    assert oracle["centers"] == pytest.approx([9.5 + 11 * i for i in range(8)], abs=1)
    assert oracle["pitch_hz"] == pytest.approx([440] * 4 + [880] * 4, abs=2)
    allowance = {"mp4": 1024, "mov": 1024, "ogg": 128}.get(suffix, 0)
    assert abs(oracle["samples"] - 48000) <= allowance
