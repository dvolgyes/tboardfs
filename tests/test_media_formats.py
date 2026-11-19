"""Check exact media exports against payloads written by TensorBoard's writer."""

from io import BytesIO
from pathlib import Path
import struct
import wave

import numpy as np
import pytest

from tboardfs import TensorBoardFS
from tboardfs.classify import detect_extension
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
        ("tiff", "images", "tiff"),
        ("avif", "images", "avif"),
    ],
)
def test_legacy_images(
    fs: TensorBoardFS, values: dict, tag: str, tab: str, suffix: str
) -> None:
    image = values[tag].image
    assert (image.width, image.height, image.colorspace) == (16, 8, 3)
    blob = fs.read(f"/{tab}/{tag}/000001.{suffix}", 65536, 0)
    assert blob == image.encoded_image_string


def test_avif_compatible_brand_with_generic_heif_major_brand(values: dict) -> None:
    image = values["avif"].image.encoded_image_string
    size = int.from_bytes(image[:4], "big")
    brands = [image[n : n + 4] for n in range(16, size, 4)]
    assert b"avif" in brands and b"mif1" in brands
    generic = image[:8] + b"mif1" + image[12:]
    assert detect_extension(generic) == "avif"


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
    blob = fs.read(f"/audio/{tag}/000001.wav", len(audio.encoded_audio_string) + 1, 0)
    metadata = (audio.num_channels, audio.sample_rate, audio.length_frames)
    assert metadata == (channels, rate, rate // 4)
    assert audio.content_type == "audio/wav"
    assert blob == audio.encoded_audio_string
    with wave.open(BytesIO(blob)) as wav:
        assert wav.getparams()[:4] == (channels, width, rate, rate // 4)
        pcm = wav.readframes(rate // 4)
    samples = np.array(
        [
            int.from_bytes(pcm[i : i + width], "little", signed=width > 1)
            for i in range(0, len(pcm), width)
        ]
    ).reshape(-1, channels)
    if width == 1:
        samples -= 128
    for index, hz in enumerate((440, 660)[:channels]):
        signal = samples[:, index] / (1 << (width * 8 - 1))
        assert np.mean(signal**2) == pytest.approx(0.125, abs=0.001)
        spectrum = np.abs(np.fft.rfft(signal))
        assert np.argmax(spectrum) * rate / len(signal) == hz


@pytest.mark.parametrize(
    ("tag", "mime", "suffix"),
    [
        ("audio-flac", "audio/flac", "flac"),
        ("audio-mp3", "audio/mpeg", "mp3"),
        ("audio-ogg", "audio/ogg", "ogg"),
        ("audio-webm", "audio/webm", "webm"),
        ("riff-midi", "audio/mid", "rmi"),
    ],
)
def test_custom_encoded_audio(
    fs: TensorBoardFS, values: dict, tag: str, mime: str, suffix: str
) -> None:
    audio = values[tag].audio
    assert audio.content_type == mime
    blob = fs.read(
        f"/audio/{tag}/000001.{suffix}", len(audio.encoded_audio_string) + 1, 0
    )
    assert blob == audio.encoded_audio_string
    if tag != "riff-midi":
        assert (audio.sample_rate, audio.num_channels, audio.length_frames) == (
            16000,
            1,
            4000,
        )
    else:
        assert blob[8:16] == b"RMIDdata"
        assert blob[20:24] == b"MThd"
        assert blob[32:34] == b"\x00\x60"
        assert blob[34:42] == b"MTrk\x00\x00\x00\x0c"
        assert blob[42:] == bytes.fromhex("00 90 45 60 30 80 45 00 00 ff 2f 00")


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
    for index, clip in enumerate(clips):
        assert (
            fs.read(f"/audio/unlabelled/000001_{index}.wav", len(clip) + 1, 0) == clip
        )
