"""TF2 tensor summaries are unpacked into the files that TF1 summaries give."""

import base64
import json
from pathlib import Path

from loguru import logger
import numpy as np
import pytest

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2

from test_incremental_parse import _files, _frame
from test_tf_producer_cases import _events

ROOT = Path(__file__).resolve().parent.parent

DT_FLOAT, DT_DOUBLE, DT_STRING = 1, 2, 7
PNG = b"\x89PNG\r\n\x1a\n"
RAW = "/raw/events.out.tfevents.host/00000000000000000000.tfrecord"


def _value(event: compat_pb2.Event, tag: str, plugin: str, shape: list[int]) -> object:
    value = event.summary.value.add(tag=tag)
    value.metadata.plugin_data.plugin_name = plugin
    value.tensor.tensor_shape.dim.extend(
        compat_pb2.TensorShapeProto.Dim(size=s) for s in shape
    )
    return value.tensor


def _log(tmp_path: Path, *events: compat_pb2.Event) -> TensorBoardFS:
    (tmp_path / "events.out.tfevents.host").write_bytes(b"".join(map(_frame, events)))
    return TensorBoardFS(tmp_path)


def _images(step: int, *blobs: bytes) -> compat_pb2.Event:
    event = compat_pb2.Event(wall_time=1.0, step=step)
    tensor = _value(event, "img", "images", [len(blobs) + 2])
    tensor.dtype = DT_STRING
    tensor.string_val.extend([b"4", b"2", *blobs])
    return event


def test_an_image_summary_is_an_image_file(tmp_path: Path) -> None:
    """One sample is <step>.png, as a TF1 image summary is."""
    fs = _log(tmp_path, _images(1, PNG + b"a"))

    assert sorted(_files(fs)) == ["/images/img/000001.png", RAW]
    assert fs.read("/images/img/000001.png", 100, 0) == PNG + b"a"


@pytest.mark.parametrize(
    ("shape", "strings", "unknown_rank"),
    [
        ([3], [b"4", b"2"], False),
        ([4], [b"4", b"2", PNG], False),
        ([2], [b"4", b"2", PNG], False),
        ([1, 3], [b"4", b"2", PNG], False),
        ([1], [b"4"], False),
        ([3], [b"4", b"2", PNG], True),
    ],
)
def test_malformed_image_keeps_packed_and_raw(
    tmp_path: Path, shape: list[int], strings: list[bytes], unknown_rank: bool
) -> None:
    event = _images(1, PNG + b"healthy")
    tensor = _value(event, "bad", "images", shape)
    tensor.dtype = DT_STRING
    tensor.tensor_shape.unknown_rank = unknown_rank
    tensor.string_val.extend(strings)
    fs = _log(tmp_path, event)
    packed = "/plugins/images/bad/000001.json"
    healthy = "/images/img/000001.png"
    assert set(_files(fs)) == {packed, healthy, RAW}
    payload = json.loads(fs.read(packed, 65536, 0))["tensor"]
    assert payload["shape"] == shape
    assert payload.get("unknown_rank", False) == unknown_rank
    assert [
        item.encode() if isinstance(item, str) else base64.b64decode(item["base64"])
        for item in payload["string_val"]
    ] == strings
    assert fs.read(healthy, 65536, 0) == PNG + b"healthy"
    assert fs.read(RAW, 65536, 0) == _frame(event)


def test_zero_image_vector_needs_no_packed_fallback(tmp_path: Path) -> None:
    event = _images(1)
    fs = _log(tmp_path, event)
    assert set(_files(fs)) == {RAW}
    assert fs.read(RAW, 65536, 0) == _frame(event)


def test_gif_routes_by_summary_kind(tmp_path: Path) -> None:
    gif = next(
        v.image.encoded_image_string
        for e in _events("media")
        for v in e.summary.value
        if v.tag == "animated"
    )
    event = _images(1, gif)
    event.summary.value.add(tag="legacy").audio.encoded_audio_string = gif
    audio = _value(event, "tensor", "audio", [1, 2])
    audio.dtype = DT_STRING
    audio.string_val.extend([gif, b"clip"])
    fs = _log(tmp_path, event)
    media = {
        "/videos/img/000001.gif",
        "/audio/legacy/000001.gif",
        "/audio/tensor/000001.gif",
    }
    assert set(_files(fs)) == media | {RAW}
    for path in media:
        assert fs.read(path, len(gif), 0) == gif


def test_omitted_metadata_across_append_and_rotation(tmp_path: Path) -> None:
    first = tmp_path / "events.out.tfevents.1"
    rotated = tmp_path / "events.out.tfevents.2"
    blob = b"GIF89a" + bytes(32)
    first.write_bytes(_frame(_images(1, blob)))
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    assert fs.read("/videos/img/000001.gif", 100, 0) == blob
    for step, path in ((2, first), (3, rotated)):
        next_event = _images(step, blob + bytes([step]))
        next_event.summary.value[0].ClearField("metadata")
        with path.open("ab") as handle:
            handle.write(_frame(next_event))
        assert fs.read(f"/videos/img/{step:06d}.gif", 100, 0) == blob + bytes([step])
    replacement = _images(1, blob)
    replacement.summary.value[0].metadata.plugin_data.plugin_name = "text"
    temp = tmp_path / "replacement"
    temp.write_bytes(_frame(replacement))
    temp.replace(first)
    assert fs.read("/text/img/000003.json", 1 << 20, 0)
    assert "/videos/img/000003.gif" not in _files(fs)


def test_duplicate_tag_media_values_keep_their_own_bytes(tmp_path: Path) -> None:
    event = compat_pb2.Event(step=1)
    jpeg = b"\xff\xd8\xff" + bytes(24)
    for blob in (PNG + b"a", jpeg, PNG + b"b"):
        event.summary.value.add(tag="img").image.encoded_image_string = blob
    fs = _log(tmp_path, event)
    assert fs.read("/images/img/000001.png", 100, 0) == PNG + b"a"
    assert fs.read("/images/img/000001.jpg", 100, 0) == jpeg
    assert fs.read("/images/img/000001_v2.png", 100, 0) == PNG + b"b"


def test_malformed_singleton_is_not_a_scalar(tmp_path: Path) -> None:
    event = compat_pb2.Event(step=1)
    value = event.summary.value.add(tag="broken")
    value.tensor.tensor_shape.dim.add(size=1)
    value.tensor.dtype = DT_FLOAT
    value.tensor.float_val.extend([1.0, 2.0])
    event.summary.value.add(tag="healthy").simple_value = 9.0
    fs = _log(tmp_path, event)
    assert "broken.json" not in fs.readdir("/scalars")
    assert json.loads(fs.read("/scalars/healthy.json", 1000, 0))[0]["value"] == 9


def test_large_string_tensor_exports_each_actual_item(tmp_path: Path) -> None:
    event = compat_pb2.Event(step=1)
    tensor = event.summary.value.add(tag="blobs").tensor
    tensor.dtype = DT_STRING
    tensor.tensor_shape.dim.add(size=2)
    blob = PNG + bytes(2048)
    tensor.string_val.extend([b"short", blob])
    fs = _log(tmp_path, event)
    assert fs.read("/tensors/blobs/000001_1.png", 4096, 0) == blob
    short_path = next(p for p in _files(fs) if p.startswith("/tensors/blobs/000001_0."))
    assert fs.read(short_path, 100, 0) == b"short"


def test_each_image_sample_is_a_file(tmp_path: Path) -> None:
    """Samples of one step are <step>_<i>, with the same width of i."""
    blobs = [PNG + bytes([index]) for index in range(11)]
    fs = _log(tmp_path, _images(1, *blobs))

    files = sorted(_files(fs))

    images = [f"/images/img/000001_{index:02d}.png" for index in range(11)]
    assert files == [*images, RAW]
    assert [fs.read(path, 100, 0) for path in images] == blobs


def test_an_audio_summary_is_an_audio_file_per_sample(tmp_path: Path) -> None:
    """Rows of a [k, 2] audio tensor are encoded audio and a label."""
    event = compat_pb2.Event(wall_time=1.0, step=2)
    tensor = _value(event, "aud", "audio", [2, 2])
    tensor.dtype = DT_STRING
    wave = b"RIFF\x00\x00\x00\x00WAVEfmt "
    tensor.string_val.extend([wave + b"0", b"first", wave + b"1", b"second"])
    fs = _log(tmp_path, event)

    assert sorted(_files(fs)) == [
        "/audio/aud/000002_0.wav",
        "/audio/aud/000002_1.wav",
        RAW,
    ]
    assert fs.read("/audio/aud/000002_1.wav", 100, 0) == wave + b"1"


def test_a_histogram_summary_is_a_histogram_table(tmp_path: Path) -> None:
    """Rows of a [k, 3] tensor are bucket left, right and count."""
    event = compat_pb2.Event(wall_time=1.0, step=1)
    tensor = _value(event, "h", "histograms", [2, 3])
    tensor.dtype = DT_DOUBLE
    tensor.tensor_content = np.asarray([0, 1, 3, 1, 2, 5], dtype="<f8").tobytes()
    fs = _log(tmp_path, event)

    rows = json.loads(fs.read("/histograms/h/000001.json", 1 << 20, 0))

    assert [(r["bucket_left"], r["bucket_right"], r["count"]) for r in rows] == [
        (0.0, 1.0, 3.0),
        (1.0, 2.0, 5.0),
    ]
    assert "/distributions/h/000001.json" in _files(fs)
    assert not any(path.startswith("/plugins") for path in _files(fs))


def test_a_scalar_summary_has_only_the_scalar_table(tmp_path: Path) -> None:
    """A TF2 scalar adds a table row, and no file per step, as in TF1."""
    event = compat_pb2.Event(wall_time=1.0, step=1)
    tensor = _value(event, "loss", "scalars", [])
    tensor.dtype = DT_FLOAT
    tensor.float_val.append(0.5)
    fs = _log(tmp_path, event)

    assert sorted(_files(fs)) == [
        RAW,
        "/scalars/loss.json",
        "/scalars/loss.npz",
        "/scalars/loss.tsv",
    ]


def test_a_non_scalar_scalars_record_keeps_its_json(tmp_path: Path) -> None:
    """A scalars record that holds no scalar is not lost."""
    event = compat_pb2.Event(wall_time=1.0, step=1)
    tensor = _value(event, "loss", "scalars", [2])
    tensor.dtype = DT_FLOAT
    tensor.float_val.extend([0.5, 0.25])
    fs = _log(tmp_path, event)

    assert sorted(_files(fs)) == [RAW, "/scalars/loss/000001.json"]


def test_tensorflow_2_summaries_are_unpacked() -> None:
    """The tf.summary fixture gives the TF1 files for each summary kind."""
    fs = TensorBoardFS(ROOT / "test-logs/tf2/summary", step_digits=3)
    files = set(_files(fs))

    assert {f"/images/sample/002_{index}.png" for index in range(2)} <= files
    assert fs.read("/images/sample/002_1.png", 8, 0) == PNG
    assert fs.read("/audio/tone/003.wav", 4, 0) == b"RIFF"
    rows = json.loads(fs.read("/histograms/weights/001.json", 1 << 20, 0))
    assert sum(row["count"] for row in rows) == 200
    assert "/distributions/constant/003.json" in files
    assert not any(path.startswith("/plugins/") for path in files)


def _tf1_histogram(
    limits: list[float], counts: list[float], *, low: float | None, high: float
) -> compat_pb2.Event:
    event = compat_pb2.Event(wall_time=1.0, step=1)
    value = event.summary.value.add(tag="h")
    if low is not None:
        value.histo.min = low
    value.histo.max = high
    value.histo.bucket_limit.extend(limits)
    value.histo.bucket.extend(counts)
    return event


def _buckets(fs: TensorBoardFS) -> list[tuple[float, float, float]]:
    rows = json.loads(fs.read("/histograms/h/000001.json", 1 << 20, 0))
    return [(r["bucket_left"], r["bucket_right"], r["count"]) for r in rows]


def test_a_tf1_histogram_spans_its_minimum_to_its_maximum(tmp_path: Path) -> None:
    """Buckets run from min to max, as TensorBoard migrates TF1 histograms."""
    event = _tf1_histogram([0.0, 1.0, 1.25], [2.0, 3.0, 1.0], low=-1.0, high=1.125)
    fs = _log(tmp_path, event)

    assert _buckets(fs) == [(-1.0, 0.0, 2.0), (0.0, 1.0, 3.0), (1.0, 1.125, 1.0)]


def test_a_tf1_histogram_without_min_starts_at_zero(tmp_path: Path) -> None:
    """Proto3 writers leave out a minimum of 0, as TensorBoard reads it."""
    fs = _log(tmp_path, _tf1_histogram([1.0, 2.0], [2.0, 3.0], low=None, high=2.0))

    assert _buckets(fs) == [(0.0, 1.0, 2.0), (1.0, 2.0, 3.0)]


def test_empty_tf1_end_buckets_are_dropped(tmp_path: Path) -> None:
    """Empty buckets at the two ends are not shown, as in TensorBoard."""
    event = _tf1_histogram(
        [-1e300, 0.0, 1.0, 2.0, 1.7e308], [0.0, 2.0, 0.0, 3.0, 0.0], low=-0.5, high=1.5
    )
    fs = _log(tmp_path, event)

    assert _buckets(fs) == [(-0.5, 0.0, 2.0), (0.0, 1.0, 0.0), (1.0, 1.5, 3.0)]


def test_an_empty_tf1_histogram_has_no_buckets(tmp_path: Path) -> None:
    """A histogram of no values has no bucket rows, so no file and no warning."""
    warnings: list[str] = []
    sink = logger.add(warnings.append, level="WARNING", format="{message}")
    try:
        fs = _log(tmp_path, _tf1_histogram([1.0, 2.0], [0.0, 0.0], low=0.0, high=0.0))

        assert list(_files(fs)) == [RAW]
    finally:
        logger.remove(sink)
    assert warnings == []


def test_a_large_image_is_only_an_image(tmp_path: Path) -> None:
    """An image over the large-tensor size is not also a tensor file."""
    blob = PNG + bytes(4096)
    fs = _log(tmp_path, _images(1, blob))

    assert sorted(_files(fs)) == ["/images/img/000001.png", RAW]
    assert fs.read("/images/img/000001.png", 1 << 20, 0) == blob


def test_a_malformed_audio_tensor_is_left_packed(tmp_path: Path) -> None:
    """A tensor missing one unlabelled clip stays packed."""
    event = compat_pb2.Event(wall_time=1.0, step=1)
    tensor = _value(event, "aud", "audio", [4])
    tensor.dtype = DT_STRING
    tensor.string_val.extend([b"RIFF0", b"RIFF1", b"RIFF2"])
    fs = _log(tmp_path, event)

    assert sorted(_files(fs)) == ["/plugins/audio/aud/000001.json", RAW]


def test_a_malformed_histogram_tensor_is_left_packed(tmp_path: Path) -> None:
    """A tensor with rows of two values is not a histogram."""
    event = compat_pb2.Event(wall_time=1.0, step=1)
    tensor = _value(event, "h", "histograms", [2, 2])
    tensor.dtype = DT_DOUBLE
    tensor.tensor_content = np.asarray([0, 1, 1, 2], dtype="<f8").tobytes()
    fs = _log(tmp_path, event)

    assert sorted(_files(fs)) == ["/histograms/h/000001.json", RAW]
    assert (
        "bucket_left" not in fs.read("/histograms/h/000001.json", 1 << 20, 0).decode()
    )


def test_a_tf1_histogram_with_too_few_limits_is_read(tmp_path: Path) -> None:
    """A malformed histogram pairs counts with the edges it has: min, limits, max."""
    event = _tf1_histogram([1.0], [2.0, 3.0, 4.0], low=0.0, high=3.0)
    fs = _log(tmp_path, event)

    assert _buckets(fs) == [(0.0, 1.0, 2.0), (1.0, 3.0, 3.0)]
