"""Check the small real-writer portion of the scale corpus without generating 1 GiB."""

import json
import struct
from base64 import b64decode
from collections.abc import Iterator
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from scripts.scale_breadth import captured_frames
from tboardfs import TensorBoardFS
from tboardfs.crc32c import masked_crc32c
from tboardfs.proto import compat_pb2
from tboardfs.tfrecord import iter_tfrecords

_EVENT = "events.out.tfevents.breadth"
_CASES = Path(__file__).parent / "data" / "producer-cases"


def _fixture_events(relative: str) -> Iterator[compat_pb2.Event]:
    """Read the stored frame payloads for a golden fixture."""
    with (_CASES / relative).open("rb") as stream:
        for frame in iter_tfrecords(stream):
            yield compat_pb2.Event.FromString(frame.payload)


class _Probe:
    """Read small mounted files against fixed payload or semantic expectations."""

    fs: TensorBoardFS

    def __init__(self, fs: TensorBoardFS) -> None:
        self.fs = fs

    def read(self, path: str) -> bytes:
        """Read a single small virtual file, including any unexpected trailing bytes."""
        return self.fs.read(path, 1 << 20, 0)

    def data(self, path: str) -> dict | list:
        """Decode the JSON representation of a captured value."""
        return json.loads(self.read(path))

    def tensors(self) -> None:
        """Check both encodings of every supported numeric dtype and boolean."""
        complex_values = [1 + 2j, -3.5 - 0.5j]
        expected = {
            1: ("<f4", [1.25, -2.5]),
            2: ("<f8", [1 + 2**-50, -2.5]),
            3: ("<i4", [-(2**31), 17]),
            4: ("u1", [0, 255]),
            5: ("<i2", [-32768, 7]),
            6: ("i1", [-128, 127]),
            8: ("<c8", complex_values),
            9: ("<i8", [-(2**63), 17]),
            10: ("?", [False, True]),
            11: ("i1", [-128, 127]),
            12: ("u1", [0, 255]),
            13: ("<i4", [-(2**31), 2**31 - 1]),
            14: ("<f4", [1.5, -2]),
            15: ("<i2", [-32768, 32767]),
            16: ("<u2", [0, 65535]),
            17: ("<u2", [65535, 1]),
            18: ("<c16", complex_values),
            19: ("<f2", [1.5, -2, 65504]),
            22: ("<u4", [0, 2**32 - 1]),
            23: ("<u8", [0, 2**64 - 1]),
        }
        for dtype, (storage, values) in expected.items():
            for mode in ("fields", "content"):
                path = f"/tensors/{mode}-{dtype}/000003.npy"
                array = np.load(BytesIO(self.read(path)), allow_pickle=False)
                np.testing.assert_array_equal(
                    array, np.array(values, dtype=storage), strict=True
                )
        path = "/tensors/complex64/000004.npy"
        array = np.load(BytesIO(self.read(path)), allow_pickle=False)
        np.testing.assert_array_equal(
            array, np.array(complex_values, dtype="<c8"), strict=True
        )

    def summaries(self) -> None:
        """Check exact scalar, histogram, PR, chart, HParams, and text semantics."""
        (row,) = self.data("/scalars/precise.json")
        assert (row["step"], row["value"]) == (4, 16_777_219)
        text = self.data("/text/binary-strings/000004.json")
        assert (text["shape"], text["values"]) == ([3], ["", "A\0B", "\ufffd\ufffd"])
        for step, counts in [(12, [1] * 4), (14, [2, 2])]:
            bins = self.data(f"/histograms/weights/{step:06d}.json")
            assert [b["count"] for b in bins if b["count"]] == counts
            rows = self.data(f"/distributions/weights/{step:06d}.json")
            extrema = [(r["percentile"], r["value"]) for r in (rows[0], rows[-1])]
            assert extrema == [(0, -3), (100, 3)]
        pr = self.data("/pr_curves/classifier/pr_curves/000012.json")
        assert [(r["true_positives"], r["false_positives"]) for r in pr] == [
            (3, 4),
            (3, 1),
            (2, 1),
            (2, 0),
            (0, 0),
        ]
        layout = self.data("/custom_scalars/layout.json")
        assert layout["category"] == [
            {
                "title": "signal",
                "chart": [
                    {"title": "power", "multiline": {"tag": ["mean_square"]}},
                ],
            }
        ]
        hparams = self.data("/hparams/hparams.json")
        (rate,) = hparams["experiment"]["hparam_infos"]
        assert (rate["name"], rate["domain_discrete"]) == ("rate", [0.25, 0.5])
        assert hparams["session_start"]["hparams"] == {"rate": 0.5}
        assert hparams["metrics"] == {"mean_square": 5.0}
        generic = self.data("/plugins/example_array/points/000002.json")
        assert b64decode(generic["plugin_content"]["base64"]) == b"\x00\xffpayload"
        tensor = generic["tensor"]
        assert tensor["dtype"] == 3
        assert b64decode(tensor["tensor_content"]["base64"]) == struct.pack(
            "<ii", 7, -3
        )

    def geometry_and_graph(self) -> None:
        """Check batched geometry and exact binary graph exports."""
        path = "/meshes/terrain/000012.npz"
        with np.load(BytesIO(self.read(path)), allow_pickle=False) as terrain:
            assert set(terrain.files) == {"vertices", "faces", "colors"}
            assert terrain["vertices"].shape == (2, 117, 3)
            np.testing.assert_array_equal(
                terrain["vertices"][:, 0], [[-1.5, -1, 1], [5, -1.5, 1.375]]
            )
            np.testing.assert_array_equal(
                terrain["colors"][:, 0], [[0, 0, 32], [32, 0, 0]]
            )
            np.testing.assert_array_equal(
                terrain["faces"][:, 0], [[0, 1, 14], [0, 1, 14]]
            )
        manifest = self.data("/plugins/Open3D/detections/000005.manifest.json")
        assert (manifest["filename"], manifest["version"]) == ("detections.msgpack", 14)
        (entry,) = manifest["start_size"]
        assert entry["start"] == 0
        assert (entry["aux_start"], entry["aux_size"]) == (entry["size"], 18)
        for event in _fixture_events("tf115-native/events.out.tfevents.tf115-native"):
            if event.HasField("graph_def"):
                assert self.read("/graphs/graph.pb") == event.graph_def
            if event.HasField("meta_graph_def"):
                assert self.read("/graphs/meta_graph.pb") == event.meta_graph_def
            if event.HasField("tagged_run_metadata"):
                expected = event.tagged_run_metadata.run_metadata
                assert self.read("/graphs/signal/000012.run_metadata.pb") == expected
        for source, index, path in (
            (
                "tf200-native/events.out.tfevents.tf200-native",
                15,
                "/graphs/signal/000014.run_metadata.pb",
            ),
            (
                "historical-variants/tf200/events.out.tfevents.function",
                1,
                "/graphs/squared_function/000007.run_metadata.pb",
            ),
            (
                "historical-variants/tf200/events.out.tfevents.dashboard",
                1,
                "/graphs/model/000007.json",
            ),
        ):
            event = next(
                event for i, event in enumerate(_fixture_events(source)) if i == index
            )
            (value,) = event.summary.value
            assert self.read(path) == value.tensor.string_val[0]

    def media(self) -> None:
        """Compare media containers byte-for-byte with their producer payloads."""
        images = {
            "png": ("images", "png"),
            "jpeg": ("images", "jpg"),
            "webp": ("images", "webp"),
            "bmp": ("images", "bmp"),
            "tiff": ("images", "tiff"),
            "avif": ("images", "avif"),
            "animated-gif": ("videos", "gif"),
            "animated-apng": ("images", "png"),
            "animated-webp": ("images", "webp"),
        }
        audio_hashes = json.loads((_CASES / "encoded-tone-sha256.json").read_text())
        for event in _fixture_events("encoded-tb115/events.out.tfevents.encoded-tb115"):
            for value in event.summary.value:
                layout, kind = value.tag.split("-", 1)
                if kind in images:
                    family, suffix = images[kind]
                    payload = (
                        value.image.encoded_image_string
                        if layout == "legacy"
                        else value.tensor.string_val[2]
                    )
                elif kind.startswith("tone-"):
                    family, suffix = "audio", kind.removeprefix("tone-")
                    path = f"/{family}/{value.tag}/000007.{suffix}"
                    assert (
                        sha256(self.read(path)).hexdigest() == audio_hashes[suffix]
                    ), path
                    continue
                elif layout in {"scalar", "vector", "bytes"}:
                    family, suffix = "videos", kind
                    payload = (
                        value.tensor.tensor_content
                        if layout == "bytes"
                        else value.tensor.string_val[0]
                    )
                else:
                    assert value.tag == "riff-midi"
                    family, suffix = "audio", "rmi"
                    payload = value.audio.encoded_audio_string
                path = f"/{family}/{value.tag}/000007.{suffix}"
                assert self.read(path) == payload, path

    def extended_media(self) -> None:
        """Check independent checksums of less common standalone media containers."""
        routes = {
            "caf": ("audio", "caf"),
            "elementary": ("videos", "bin"),
            "hdr": ("images", "hdr"),
            "ico-bmp": ("images", "ico"),
            "ico-png": ("images", "ico"),
            "ogg": ("audio", "ogg"),
            "program-v1": ("videos", "mpg"),
            "program-v2": ("videos", "mpg"),
            "qoi": ("images", "qoi"),
            "svg": ("images", "svg"),
            "transport": ("videos", "ts"),
            "webm": ("audio", "webm"),
            "windows-media": ("videos", "asf"),
        }
        hashes = json.loads((_CASES / "media-containers/sha256.json").read_text())
        assert hashes.keys() == routes.keys()
        for tag, (family, suffix) in routes.items():
            path = f"/{family}/{tag}/000001.{suffix}"
            assert sha256(self.read(path)).hexdigest() == hashes[tag], path


def check_breadth(fs: TensorBoardFS) -> None:
    """Check representative typed exports outside benchmark timing."""
    probe = _Probe(fs)
    probe.tensors()
    probe.summaries()
    probe.geometry_and_graph()
    probe.media()
    probe.extended_media()


def test_native_sidecars_in_producer_layouts() -> None:
    """Exercise native external assets in their original producer layouts."""
    probe = _Probe(TensorBoardFS(_CASES / "open3d-advanced"))
    manifest = probe.data("/plugins/Open3D/detections/000005.manifest.json")
    path = "/plugins/Open3D/~assets/detections.msgpack"
    assert manifest["asset"] == path[1:]
    asset = probe.read(path)
    original = _CASES / "open3d-advanced/plugins/Open3D/detections.msgpack"
    assert asset == original.read_bytes()
    (entry,) = manifest["start_size"]
    assert masked_crc32c(asset[: entry["size"]]) == entry["masked_crc32c"]
    assert asset[entry["aux_start"] :] == (
        b"\x0a\x07\x08\x02\x15"
        + struct.pack("<f", 0.875)
        + b"\x0a\x07\x08\x07\x15"
        + struct.pack("<f", 0.625)
    )
    projector = _Probe(TensorBoardFS(_CASES / "lightning-matrix"))
    assert projector.read("/projector/00004/labels/metadata.tsv") == b"oak\nash\n"
    sprite = projector.read("/projector/00004/sprite/sprite.png")
    assert sha256(sprite).hexdigest() == (
        "44b9f96054bbcae63c3727eb89508bf3a9c146c2a8e122d7416c3c42c51691bd"
    )
    config = projector.read("/projector/projector_config.pbtxt")
    assert b'tensor_name: "labels:00004"' in config
    trial = projector.data("/trial/hparams/hparams.json")
    assert trial["session_start"]["hparams"] == {"optimizer": "sgd"}
    assert trial["session_end"]["status"] == "STATUS_SUCCESS"
    profile = _Probe(TensorBoardFS(_CASES / "tf200-native"))
    trace = profile.read("/profile/signal/local.trace")
    assert (
        trace
        == (_CASES / "tf200-native/plugins/profile/signal/local.trace").read_bytes()
    )
    assert b"Square" in trace and b"Sum" in trace
    auxiliary = _Probe(TensorBoardFS(_CASES / "auxiliary", raw=True))
    assert auxiliary.read("/projector/projector_config.pbtxt") == (
        b'model_checkpoint_path: "weights"\nembeddings {\n  tensor_name: "vectors"\n}\n'
    )
    assert auxiliary.fs.readdir("/projector")[2:] == ["projector_config.pbtxt"]
    assert auxiliary.fs.readdir("/raw")[2:] == ["events.out.tfevents.auxiliary"]
    legacy = _Probe(TensorBoardFS(_CASES / "legacy-tool-assets"))
    assert "plugins" not in legacy.fs.readdir("/")
    root = legacy.data("/profile/schema-only/host.op_profile.json")["byProgram"]
    operations = [
        (node["xla"]["op"], node["metrics"]["rawTime"]) for node in root["children"]
    ]
    assert operations == [("dot", 1200), ("add", 800)]


def test_captured_breadth_exercises_mount_reads(tmp_path: Path) -> None:
    """Read exact typed and raw bytes from the merged event file."""
    source = tmp_path / _EVENT
    source.write_bytes(b"".join(frame for _, frame in captured_frames()))
    fs = TensorBoardFS(tmp_path, raw=True)
    check_breadth(fs)
    with source.open("rb") as stream:
        frames = list(iter_tfrecords(stream))
    assert len(fs.readdir(f"/raw/{_EVENT}")) == len(frames) + 2
    for frame in frames:
        name = f"/raw/{_EVENT}/{frame.record_offset:020d}.tfrecord"
        assert fs.read(name, frame.total_size + 1, 0) == (
            frame.header + frame.payload + frame.footer
        )
    assert "loss.json" not in fs.readdir("/scalars")


@pytest.fixture
def captured_events() -> list[compat_pb2.Event]:
    """Generate the deterministic captured frames twice and decode them once."""
    frames = list(captured_frames())
    assert frames == list(captured_frames())
    stream = BytesIO(b"".join(frame for _, frame in frames))
    payloads = [frame.payload for frame in iter_tfrecords(stream)]
    assert payloads.count(b"\xa0\x06\x07") == 1
    return [compat_pb2.Event.FromString(payload) for payload in payloads]


def test_captured_frame_families(captured_events: list[compat_pb2.Event]) -> None:
    """Keep every captured event, value, and plugin family in the mix."""
    kinds = {event.WhichOneof("what") for event in captured_events}
    assert kinds == {
        None,
        "graph_def",
        "file_version",
        "log_message",
        "meta_graph_def",
        "session_log",
        "summary",
        "tagged_run_metadata",
    }
    values = [value for event in captured_events for value in event.summary.value]
    assert {value.WhichOneof("value") for value in values} == {
        None,
        "audio",
        "histo",
        "image",
        "obsolete_old_style_histogram",
        "simple_value",
        "tensor",
    }
    plugins = {value.metadata.plugin_data.plugin_name for value in values}
    assert plugins == {
        "",
        "Open3D",
        "audio",
        "custom_scalars",
        "debugger",
        "example_array",
        "graph_keras_model",
        "graph_run_metadata",
        "graph_run_metadata_graph",
        "histograms",
        "hparams",
        "images",
        "mesh",
        "pr_curves",
        "scalars",
        "tensor",
        "tensors",
        "text",
        "unrecognized",
        "video",
        "videos",
    }


def test_captured_frame_order_and_legacy_payloads(
    captured_events: list[compat_pb2.Event],
) -> None:
    """Retain captured steps and legacy payloads without restart markers."""
    steps = [event.step for event in captured_events]
    assert steps == sorted(steps)
    assert (steps[0], steps[-1]) == (0, 14)
    messages = [
        event.log_message.message
        for event in captured_events
        if event.HasField("log_message")
    ]
    assert messages == ["retained warning"]
    session_logs = [
        event.session_log for event in captured_events if event.HasField("session_log")
    ]
    assert [(log.status, log.checkpoint_path, log.msg) for log in session_logs] == [
        (3, "checkpoint-1", ""),
        (2, "", "paused"),
    ]
    obsolete = [
        (value.tag, value.obsolete_old_style_histogram)
        for event in captured_events
        for value in event.summary.value
        if value.HasField("obsolete_old_style_histogram")
    ]
    assert obsolete == [("historical", b"\x00\xffopaque")]


def test_header_preserves_writer_provenance() -> None:
    """The mix starts with the original writer header, not synthetic metadata."""
    step, header = next(captured_frames())
    assert step == 0
    (frame,) = iter_tfrecords(BytesIO(header))
    event = compat_pb2.Event.FromString(frame.payload)
    assert event.file_version == "brain.Event:2"
    assert event.source_metadata.writer == (
        "tensorboard.summary.writer.event_file_writer"
    )
