"""Check real audio subtypes and byte-exact exports without writer dependencies."""

import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import magic
import numpy as np
import pytest
from test_tf_producer_cases import _events

from tboardfs import TensorBoardFS
from tboardfs.classify import detect_extension
from tboardfs.proto import compat_pb2

RUN = Path(__file__).parent / "data/producer-cases/audio-variants"
CASES = (
    ("aac-adts", "aac", "audio/aac"),
    ("aac-m4a", "m4a", "audio/mp4"),
    ("alac-m4a", "m4a", "audio/mp4"),
    ("float-wav", "wav", "audio/wav"),
    ("pcm-aiff", "aiff", "audio/aiff"),
)


@pytest.fixture(scope="module")
def audio() -> dict[str, compat_pb2.Summary.Audio]:
    return {v.tag: v.audio for e in _events("audio-variants") for v in e.summary.value}


@pytest.fixture(scope="module")
def fs() -> TensorBoardFS:
    return TensorBoardFS(RUN)


@pytest.mark.parametrize(("tag", "suffix", "mime"), CASES)
def test_audio_exports(
    fs: TensorBoardFS, audio: dict, tag: str, suffix: str, mime: str
) -> None:
    clip = audio[tag]
    assert clip.content_type == mime
    assert (clip.sample_rate, clip.length_frames, clip.num_channels) == (24000, 6000, 1)
    blob = clip.encoded_audio_string
    assert detect_extension(blob) == suffix
    assert fs.readdir(f"/audio/{tag}")[2:] == [f"000001.{suffix}"]
    assert fs.read(f"/audio/{tag}/000001.{suffix}", len(blob) + 1, 0) == blob


def test_m4a_brand_avoids_mime_without_overriding_image_brands(audio: dict) -> None:
    with patch.object(magic, "from_buffer", wraps=magic.from_buffer) as detect:
        for tag in ("aac-m4a", "alac-m4a"):
            clip = audio[tag].encoded_audio_string
            assert detect_extension(clip) == "m4a"
        assert detect_extension(clip[:16] + b"avif" + clip[20:]) == "avif"
        assert detect_extension(clip[:12]) == "bin"
        detect.assert_not_called()


@pytest.mark.parametrize(("tag", "suffix", "mime"), CASES)
def test_tone_decodes(audio: dict, tag: str, suffix: str, mime: str) -> None:
    decoder = shutil.which("ffmpeg")
    if decoder is None:
        pytest.skip("system FFmpeg not installed")
    raw = subprocess.run(
        [decoder, "-v", "error", "-i", "pipe:0", "-f", "f32le", "pipe:1"],
        input=audio[tag].encoded_audio_string,
        capture_output=True,
        check=True,
    ).stdout
    samples = np.frombuffer(raw, dtype="<f4")
    assert 6000 <= len(samples) < 8048
    middle = samples[2000:4000]
    assert 0.2 < np.max(np.abs(middle)) < 0.55
    crossings = np.count_nonzero((middle[:-1] <= 0) & (middle[1:] > 0))
    assert crossings * 24000 / len(middle) == pytest.approx(440, abs=8)
