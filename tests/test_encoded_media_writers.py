"""Compare real, low-level event writers' filesystem exports to codec output."""

import json
import math
import struct
import wave
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pytest
from test_tf_producer_cases import CORPUS, _events

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2
from tboardfs.tfrecord import iter_tfrecords

_WRITER_VERSIONS = {
    "tb115": {"tensorboard": "1.15.0"},
    "tb200": {"tensorboard": "2.0.0"},
    "tf1": {"tensorflow-cpu": "2.21.0"},
    "tf2": {"tensorflow-cpu": "2.21.0"},
    "tb221": {"tensorboard": "2.21.0"},
    "tbx": {"tensorboardX": "2.6.5"},
    "torch": {"torch": "2.12.0+cpu", "tensorboard": "2.21.0"},
    "lightning": {
        "lightning": "2.6.6",
        "torch": "2.12.0+cpu",
        "tensorboard": "2.21.0",
    },
}
_RECORD_WRITER = "tensorboard.summary.writer.record_writer.RecordWriter.write"
_TORCH_WRITER = "torch.utils.tensorboard.writer.FileWriter.add_summary"
_WRITER_APIS = {
    "tb115": _RECORD_WRITER,
    "tb200": _RECORD_WRITER,
    "tf1": "tensorflow.python.summary.writer.writer.FileWriter.add_event",
    "tf2": "tensorflow.python.ops.summary_ops_v2.write_raw_pb",
    "tb221": "tensorboard.summary.writer.event_file_writer.EventFileWriter.add_event",
    "tbx": "tensorboardX.writer.FileWriter.add_summary",
    "torch": _TORCH_WRITER,
    "lightning": (
        "lightning.pytorch.loggers.tensorboard.TensorBoardLogger -> " + _TORCH_WRITER
    ),
}

_EXTENDED_ROUTES = {
    "media/media": {
        "gif": ("videos", "gif"),
        "channels": ("images", "png"),
        "unlabelled": ("audio", "wav"),
        **{f"pcm{width}": ("audio", "wav") for width in (8, 16, 24, 32)},
    },
    "rasters/rasters": {
        **{
            f"png-{color}-{depth}": ("images", "png")
            for color, depths in (
                ("gray", (1, 2, 4, 8, 16)),
                ("palette", (1, 2, 4, 8)),
                ("rgb", (8, 16)),
                ("rgba", (8, 16)),
                ("gray-alpha", (8, 16)),
            )
            for depth in depths
        },
        **{
            f"jpeg-{color}-{mode}": ("images", "jpg")
            for color in ("gray", "rgb", "cmyk")
            for mode in ("baseline", "progressive")
        },
    },
    "rasters/moving-scenes": {
        "moving-gif": ("videos", "gif"),
        "moving-apng": ("images", "png"),
        "moving-webp-lossless": ("images", "webp"),
        "moving-webp-lossy": ("images", "webp"),
    },
    "audio-variants/audio-variants": {
        tag: ("audio", suffix)
        for tag, suffix in (
            ("aac-adts", "aac"),
            ("aac-m4a", "m4a"),
            ("alac-m4a", "m4a"),
            ("float-wav", "wav"),
            ("pcm-aiff", "aiff"),
        )
    },
    "av-clips/av-clips": {
        suffix: ("videos", suffix)
        for suffix in ("mp4", "mov", "avi", "webm", "mkv", "ogg")
    },
    "geotiff/geotiff": {
        f"{kind}-{layout}": ("images", "tiff")
        for kind in ("classic", "bigtiff")
        for layout in ("legacy", "tensor")
    },
    "heic/heic": {
        f"{suffix}-{layout}": ("images", suffix)
        for suffix in ("heic", "avif")
        for layout in ("image", "tensor")
    },
    "wave64/wave64": {
        f"{layout}-{kind}": ("audio", "wav")
        for layout in ("legacy", "tf2")
        for kind in ("rf64", "bw64")
    },
    "native-media/native-media": {
        f"{layout}-{kind}": ("audio" if suffix in {"mp2", "mp3"} else "images", suffix)
        for layout in ("legacy", "tf2")
        for kind, suffix in (
            ("jp2", "jp2"),
            ("j2k", "j2k"),
            ("jxl-codestream", "jxl"),
            ("jxl-container", "jxl"),
            ("exr", "exr"),
            ("mp2", "mp2"),
            ("mp3", "mp3"),
        )
    },
    "media-containers/media-containers": {
        "qoi": ("images", "qoi"),
        "ico-bmp": ("images", "ico"),
        "ico-png": ("images", "ico"),
        "hdr": ("images", "hdr"),
        "svg": ("images", "svg"),
        "caf": ("audio", "caf"),
        "ogg": ("audio", "ogg"),
        "webm": ("audio", "webm"),
        "program-v1": ("videos", "mpg"),
        "program-v2": ("videos", "mpg"),
        "transport": ("videos", "ts"),
        "windows-media": ("videos", "asf"),
        "elementary": ("videos", "bin"),
    },
}


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


@pytest.fixture(scope="module")
def extended_summary() -> compat_pb2.Summary:
    """Reconstruct only the shared payload, from the original CRC-checked runs."""
    summary = compat_pb2.Summary()
    for key, routes in _EXTENDED_ROUTES.items():
        run, name = key.split("/")
        path = CORPUS / run / f"events.out.tfevents.{name}"
        with path.open("rb") as stream:
            values = {
                value.tag: value
                for frame in iter_tfrecords(stream)
                for value in compat_pb2.Event.FromString(frame.payload).summary.value
                if value.tag in routes
            }
        assert values.keys() == routes.keys()
        for tag, original in values.items():
            value = summary.value.add()
            value.CopyFrom(original)
            value.tag = f"{run}-{tag}"
    return summary


def _assert_writer_provenance(
    name: str, header: compat_pb2.Event, event: compat_pb2.Event
) -> None:
    """Check native file headers and the eager writer's own event timestamp."""
    assert header.file_version == "brain.Event:2"
    assert header.wall_time > 0
    if name in {"tf1", "tf2"}:
        assert header.source_metadata.writer == "tensorflow.core.util.events_writer"
    elif name in {"tb221", "torch", "lightning"}:
        assert (
            header.source_metadata.writer
            == "tensorboard.summary.writer.event_file_writer"
        )
    else:
        assert not header.HasField("source_metadata")
    if name == "tf2":
        assert event.wall_time > 1
    else:
        assert event.wall_time == 1


@pytest.fixture(scope="module", params=list(_WRITER_VERSIONS))
def extended_writer(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    extended_summary: compat_pb2.Summary,
) -> TensorBoardFS:
    """Restore the exact native TFRecord and verify its independent writer attestation."""
    name = request.param
    manifest = json.loads((CORPUS / "encoded-extended.json").read_text())
    assert manifest["writers"].keys() == _WRITER_VERSIONS.keys()
    assert len({writer["sha256"] for writer in manifest["writers"].values()}) == len(
        _WRITER_VERSIONS
    )
    envelope = manifest["writers"][name]
    assert envelope["versions"] == _WRITER_VERSIONS[name]
    assert envelope["api"] == _WRITER_APIS[name]
    summary = extended_summary.SerializeToString()
    assert len(summary) < 1024**2
    assert sha256(summary).hexdigest() == manifest["summary_sha256"]
    prefix = bytes.fromhex(envelope["prefix"])
    suffix = bytes.fromhex(envelope["suffix"])
    assert len(prefix) + len(suffix) <= 4096
    data = prefix + summary + suffix
    assert sha256(data).hexdigest() == envelope["sha256"]
    directory = tmp_path_factory.mktemp(f"extended-{name}")
    path = directory / f"events.out.tfevents.extended-{name}"
    path.write_bytes(data)
    with path.open("rb") as stream:
        header, event = (
            compat_pb2.Event.FromString(frame.payload)
            for frame in iter_tfrecords(stream)
        )
    _assert_writer_provenance(name, header, event)
    assert event.step == 7
    assert event.summary == extended_summary
    return TensorBoardFS(directory)


@pytest.mark.parametrize("key", list(_EXTENDED_ROUTES))
def test_extended_encoded_media_original_bytes_and_routes(
    extended_writer: TensorBoardFS, extended_summary: compat_pb2.Summary, key: str
) -> None:
    """Test exact exports without high-level codec claims or classifier-derived names."""
    run, _ = key.split("/")
    sources = {value.tag: value for value in extended_summary.value}
    for tag, (tab, suffix) in _EXTENDED_ROUTES[key].items():
        stem = f"{run}-{tag}"
        source = sources[stem]
        if source.HasField("image"):
            payloads = [source.image.encoded_image_string]
        elif source.HasField("audio"):
            payloads = [source.audio.encoded_audio_string]
        elif source.tensor.tensor_content:
            payloads = [source.tensor.tensor_content]
        elif tab == "audio":
            payloads = list(source.tensor.string_val)
            if tag != "unlabelled":
                payloads = payloads[::2]
        else:
            payloads = list(source.tensor.string_val[2:])
        assert len(payloads) == (4 if tag in {"channels", "unlabelled"} else 1)
        names = [
            f"000007{f'_{i}' if len(payloads) > 1 else ''}.{suffix}"
            for i in range(len(payloads))
        ]
        directory = f"/{tab}/{stem}"
        assert extended_writer.readdir(directory)[2:] == names
        for filename, payload in zip(names, payloads, strict=True):
            assert payload
            assert (
                extended_writer.read(f"{directory}/{filename}", len(payload) + 1, 0)
                == payload
            )
