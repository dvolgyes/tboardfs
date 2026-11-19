"""Compare real, low-level event writers' filesystem exports to codec output."""

from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import struct
import wave

import pytest

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2
from tboardfs.tfrecord import iter_tfrecords
from test_tf_producer_cases import CORPUS, _events


@pytest.fixture(
    scope="module",
    params=["tb115", "tb200", "tf1", "tf2", "tb221", "tbx", "torch", "lightning"],
)
def writer(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[TensorBoardFS, dict]:
    """Reassemble the producer's exact file, validating its complete hash and CRCs."""
    name = request.param
    run = f"encoded-{name}"
    canonical = _events("encoded-tb115")
    if name == "tb115":
        events, directory = canonical, CORPUS / run
    else:
        directory = tmp_path_factory.mktemp(run)
        envelope = json.loads((CORPUS / "encoded-envelopes.json").read_text())[name]
        (summary,) = (e.summary for e in canonical if e.HasField("summary"))
        data = (
            bytes.fromhex(envelope["prefix"])
            + summary.SerializeToString()
            + bytes.fromhex(envelope["suffix"])
        )
        assert sha256(data).hexdigest() == envelope["sha256"]
        path: Path = directory / f"events.out.tfevents.{run}"
        path.write_bytes(data)
        with path.open("rb") as stream:
            events = [
                compat_pb2.Event.FromString(frame.payload)
                for frame in iter_tfrecords(stream)
            ]
    values = {v.tag: v for event in events for v in event.summary.value}
    return TensorBoardFS(directory), values


@pytest.fixture(scope="module")
def reference() -> dict[str, bytes]:
    """Original sources: Pillow-verified animations and FFmpeg-decoded films."""
    samples = {}
    for run in ("media", "rasters"):
        for event in _events(run):
            for value in event.summary.value:
                if value.HasField("image"):
                    samples[value.tag] = value.image.encoded_image_string
                elif value.HasField("audio"):
                    samples[value.tag] = value.audio.encoded_audio_string
                elif value.metadata.plugin_data.plugin_name == "video":
                    samples[f"video-{value.tag}"] = value.tensor.tensor_content
    return samples


@pytest.mark.parametrize(
    ("tag", "tab", "suffix"),
    [
        ("png", "images", "png"),
        ("jpeg", "images", "jpg"),
        ("webp", "images", "webp"),
        ("bmp", "images", "bmp"),
        ("tiff", "images", "tiff"),
        ("avif", "images", "avif"),
        ("animated-gif", "videos", "gif"),
        ("animated-apng", "images", "png"),
        ("animated-webp", "images", "webp"),
    ],
)
def test_image_field_and_dimension_prefixed_tensor(writer, reference, tag, tab, suffix):
    fs, values = writer
    legacy = values[f"legacy-{tag}"].image
    assert (legacy.width, legacy.height, legacy.colorspace) == (16, 8, 3)
    plugin = values[f"plugin-{tag}"]
    assert plugin.tensor.dtype == compat_pb2.DT_STRING
    assert plugin.metadata.plugin_data.plugin_name == "images"
    assert [d.size for d in plugin.tensor.tensor_shape.dim] == [3]
    assert list(plugin.tensor.string_val[:2]) == [b"16", b"8"]
    for layout in ("legacy", "plugin"):
        stem = f"{layout}-{tag}"
        assert fs.read(f"/{tab}/{stem}/000007.{suffix}", 65536, 0) == reference[tag]
        assert stem not in fs.readdir("/images" if tab == "videos" else "/videos")


@pytest.mark.parametrize(
    ("ext", "mime"),
    [
        ("wav", "audio/wav"),
        ("mp3", "audio/mpeg"),
        ("flac", "audio/flac"),
        ("ogg", "audio/ogg"),
        ("webm", "audio/webm"),
    ],
)
def test_audio_field_and_labelled_tensor_keep_playable_tone(writer, ext, mime):
    """Compare exported clips to independently decoded encoder outputs."""
    fs, values = writer
    legacy = values[f"legacy-tone-{ext}"].audio
    assert (
        legacy.content_type,
        legacy.sample_rate,
        legacy.num_channels,
        legacy.length_frames,
    ) == (mime, 16000, 1, 4000)
    plugin = values[f"plugin-tone-{ext}"]
    assert plugin.metadata.plugin_data.plugin_name == "audio"
    assert plugin.tensor.dtype == compat_pb2.DT_STRING
    assert [d.size for d in plugin.tensor.tensor_shape.dim] == [1, 2]
    assert plugin.tensor.string_val[1] == b"Quarter-second A4"
    hashes = json.loads((CORPUS / "encoded-tone-sha256.json").read_text())
    for layout in ("legacy", "plugin"):
        directory = f"/audio/{layout}-tone-{ext}"
        assert fs.readdir(directory)[2:] == [f"000007.{ext}"]
        data = fs.read(f"{directory}/000007.{ext}", 65536, 0)
        assert sha256(data).hexdigest() == hashes[ext]
        if ext == "wav":
            with wave.open(BytesIO(data)) as audio:
                assert audio.getparams()[:4] == (1, 2, 16000, 4000)
                samples = struct.unpack("<4000h", audio.readframes(4000))
            expected = [
                round(12000 * math.sin(math.tau * 440 * i / 16000)) for i in range(4000)
            ]
            assert list(samples) == expected


@pytest.mark.parametrize("ext", ["mp4", "mov", "avi", "webm", "mkv", "ogg"])
@pytest.mark.parametrize("layout", ["scalar", "vector", "bytes"])
def test_custom_video_layouts_preserve_both_decoded_frames(
    writer, reference, ext, layout
):
    fs, values = writer
    value = values[f"{layout}-{ext}"]
    assert value.metadata.plugin_data.plugin_name == (
        "videos" if layout == "vector" else "video"
    )
    payload = reference[f"video-{ext}"]
    tensor = value.tensor
    assert tensor.dtype == (
        compat_pb2.DT_UINT8 if layout == "bytes" else compat_pb2.DT_STRING
    )
    assert [d.size for d in tensor.tensor_shape.dim] == {
        "scalar": [],
        "vector": [1],
        "bytes": [len(payload)],
    }[layout]
    assert fs.read(f"/videos/{layout}-{ext}/000007.{ext}", 65536, 0) == payload


def test_wrapped_midi_keeps_riff_container(writer, reference):
    fs, _ = writer
    assert fs.read("/audio/riff-midi/000007.rmi", 65536, 0) == reference["riff-midi"]
