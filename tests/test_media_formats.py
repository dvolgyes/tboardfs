"""Check exact media exports against payloads written by TensorBoard's writer."""

from io import BytesIO
from pathlib import Path
import struct
import wave

import pytest

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2

from test_tf_producer_cases import _events


RUN = Path(__file__).parent / "data" / "producer-cases" / "media"


@pytest.fixture(scope="module")
def fs() -> TensorBoardFS:
    return TensorBoardFS(RUN)


@pytest.fixture(scope="module")
def values() -> dict[str, compat_pb2.Summary.Value]:
    return {v.tag: v for e in _events("media") for v in e.summary.value}


@pytest.mark.parametrize(
    ("tag", "tab", "suffix"),
    [
        ("png", "images", "png"),
        ("jpeg", "images", "jpg"),
        ("webp", "images", "webp"),
        ("gif", "videos", "gif"),
        ("animated", "videos", "gif"),
        ("bmp", "images", "bmp"),
        ("tiff", "images", "bin"),
        ("avif", "images", "bin"),
    ],
)
def test_legacy_images(
    fs: TensorBoardFS, values: dict, tag: str, tab: str, suffix: str
) -> None:
    image = values[tag].image
    assert (image.width, image.height, image.colorspace) == (16, 8, 3)
    blob = fs.read(f"/{tab}/{tag}/000001.{suffix}", 65536, 0)
    assert blob == image.encoded_image_string


def test_image_tensor_keeps_png_dimensions_channels_and_order(
    fs: TensorBoardFS, values: dict
) -> None:
    value = values["channels"]
    assert value.metadata.plugin_data.plugin_name == "images"
    assert [dim.size for dim in value.tensor.tensor_shape.dim] == [6]
    width, height, *images = value.tensor.string_val
    assert (width, height) == (b"16", b"8")
    assert fs.readdir("/images/channels")[2:] == [f"000001_{i}.png" for i in range(4)]
    for i, color_type in enumerate([0, 4, 2, 6]):
        blob = fs.read(f"/images/channels/000001_{i}.png", 65536, 0)
        assert blob == images[i]
        assert blob[:8] == b"\x89PNG\r\n\x1a\n"
        assert struct.unpack_from(">IIBB", blob, 16) == (16, 8, 8, color_type)


def test_animated_gif_has_same_route_in_image_tensor_and_legacy_field(
    fs: TensorBoardFS, values: dict
) -> None:
    clip = values["tensor-gif"]
    assert clip.metadata.plugin_data.plugin_name == "images"
    assert list(clip.tensor.string_val) == [
        b"16",
        b"8",
        values["animated"].image.encoded_image_string,
    ]
    assert (
        fs.read("/videos/tensor-gif/000001.gif", 65536, 0) == clip.tensor.string_val[2]
    )
    assert "tensor-gif" not in fs.readdir("/images")


@pytest.mark.parametrize(
    ("channels", "width", "rate"),
    [(1, 1, 8000), (2, 2, 16000), (2, 3, 44100), (2, 4, 48000)],
)
def test_wav_headers_and_interleaved_pcm(
    fs: TensorBoardFS, values: dict, channels: int, width: int, rate: int
) -> None:
    tag = f"pcm{8 * width}"
    audio = values[tag].audio
    blob = fs.read(f"/audio/{tag}/000001.wav", 65536, 0)
    metadata = (audio.num_channels, audio.sample_rate, audio.length_frames)
    assert metadata == (channels, rate, 3)
    assert audio.content_type == "audio/wav"
    assert blob == audio.encoded_audio_string
    with wave.open(BytesIO(blob)) as wav:
        assert wav.getparams()[:4] == (channels, width, rate, 3)
        assert wav.readframes(3) == bytes(range(3 * channels * width))


@pytest.mark.parametrize(
    ("tag", "mime", "suffix"),
    [
        ("audio-flac", "audio/flac", "flac"),
        ("audio-mp3", "audio/mpeg", "mp3"),
        ("audio-ogg", "audio/ogg", "ogg"),
        ("audio-webm", "audio/webm", "webm"),
        ("riff-midi", "audio/mid", "bin"),
    ],
)
def test_custom_encoded_audio(
    fs: TensorBoardFS, values: dict, tag: str, mime: str, suffix: str
) -> None:
    audio = values[tag].audio
    assert audio.content_type == mime
    blob = fs.read(f"/audio/{tag}/000001.{suffix}", 65536, 0)
    assert blob == audio.encoded_audio_string


@pytest.mark.parametrize("tag", ["mp4", "mov", "avi", "webm", "mkv", "ogg"])
def test_custom_video_plugin_keeps_native_container(
    fs: TensorBoardFS, values: dict, tag: str
) -> None:
    value = values[tag]
    assert value.metadata.plugin_data.plugin_name == "video"
    blob = fs.read(f"/videos/{tag}/000001.{tag}", 65536, 0)
    assert blob == value.tensor.tensor_content


def test_rank_one_audio_exports_every_unlabelled_clip(
    fs: TensorBoardFS, values: dict
) -> None:
    batch = values["unlabelled"]
    assert batch.metadata.plugin_data.plugin_name == "audio"
    assert [dim.size for dim in batch.tensor.tensor_shape.dim] == [4]
    clips = [
        values[f"pcm{bits}"].audio.encoded_audio_string for bits in (8, 16, 24, 32)
    ]
    assert list(batch.tensor.string_val) == clips
    assert [
        fs.read(f"/audio/unlabelled/{name}", 65536, 0)
        for name in fs.readdir("/audio/unlabelled")[2:]
    ] == clips
