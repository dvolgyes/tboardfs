"""Read the compact TensorFlow writer corpus without importing TensorFlow."""

import json
import wave
from io import BytesIO
from pathlib import Path

import numpy as np

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2
from tboardfs.tfrecord import iter_tfrecords

CORPUS = Path(__file__).parent / "data" / "producer-cases"


def _events(producer: str) -> list[compat_pb2.Event]:
    path = CORPUS / producer / f"events.out.tfevents.{producer}"
    with path.open("rb") as handle:
        return [
            compat_pb2.Event.FromString(frame.payload)
            for frame in iter_tfrecords(handle)
        ]


def test_tf1_repeated_images_and_omitted_step() -> None:
    events = [event for event in _events("tf1") if event.summary.value]
    omitted, repeated, *_ = events
    assert "step" not in {field.name for field, _ in omitted.ListFields()}
    assert (repeated.step, repeated.wall_time) == (7, 10.0)
    values = repeated.summary.value
    assert [v.tag for v in values] == ["same-tag"] * 3
    encoded = [v.image.encoded_image_string for v in values]
    assert len(set(encoded)) == 3
    fs = TensorBoardFS(CORPUS / "tf1")
    assert fs.read("/images/omitted-step/000000.png", 4096, 0) == encoded[0]
    assert [
        fs.read(f"/images/same-tag/{name}", 4096, 0)
        for name in (
            "000007.png",
            "000007.jpg",
            "000007_v2.png",
        )
    ] == encoded


def test_tf1_coincident_histograms() -> None:
    events = [
        e for e in _events("tf1") if any(v.tag == "histogram" for v in e.summary.value)
    ]
    assert [(e.step, e.wall_time) for e in events] == [(7, 10.0)] * 2
    assert [list(e.summary.value[0].histo.bucket_limit) for e in events] == [
        [-1.0, 1.0, 3.0],
        [15.0, 25.0, 31.0],
    ]
    fs = TensorBoardFS(CORPUS / "tf1")
    rows = json.loads(fs.read("/histograms/histogram/000007.json", 8192, 0))
    assert [(r["bucket_left"], r["bucket_right"], r["count"]) for r in rows] == [
        (-2, -1, 1),
        (-1, 1, 1),
        (1, 2, 1),
        (10, 15, 1),
        (15, 25, 1),
        (25, 30, 1),
    ]


def test_tf2_batched_media_and_metadata_reuse() -> None:
    fs = TensorBoardFS(CORPUS / "tf2")
    tiles = [
        (e.step, v) for e in _events("tf2") for v in e.summary.value if v.tag == "tiles"
    ]
    assert [(step, v.metadata.plugin_data.plugin_name) for step, v in tiles] == [
        (2, "images"),
        (3, ""),
    ]
    first, second = tiles[0][1].tensor.string_val[2:]
    assert first != second
    assert [
        fs.read(f"/images/tiles/{name}.png", 4096, 0)
        for name in (
            "000002_0",
            "000002_1",
            "000003",
        )
    ] == [first, second, second]
    for item, (frequency, amplitude) in enumerate([(250, 0.5), (500, 0.75)]):
        blob = fs.read(f"/audio/tone/000002_{item}.wav", 4096, 0)
        with wave.open(BytesIO(blob)) as wav:
            assert (
                wav.getnchannels(),
                wav.getframerate(),
                wav.getsampwidth(),
                wav.getnframes(),
            ) == (1, 8000, 2, 2000)
            pcm = np.frombuffer(wav.readframes(2000), dtype="<i2")
        expected = (
            amplitude * 32768 * np.sin(np.arange(2000) * (2 * np.pi * frequency / 8000))
        )
        np.testing.assert_allclose(pcm, expected, atol=0.51, rtol=0)
        quarter_period = 8000 // (4 * frequency)
        assert pcm[[0, quarter_period, 3 * quarter_period]].tolist() == [
            0,
            int(amplitude * 32768),
            -int(amplitude * 32768),
        ]


def test_tf2_exact_steps_matrix_and_histogram() -> None:
    fs = TensorBoardFS(CORPUS / "tf2")
    big = 2**53
    rows = json.loads(fs.read("/scalars/large-step.json", 4096, 0))
    assert [(r["step"], r["value"]) for r in rows] == [(big, -1.25), (big + 1, 3.5)]
    matrix = np.load(BytesIO(fs.read(f"/tensors/matrix/{big + 1}.npy", 4096, 0)))
    assert matrix.dtype == np.int64
    np.testing.assert_array_equal(matrix, [[big + 1, -7, 0], [12, big - 1, 4]])
    rows = json.loads(fs.read("/histograms/spread/000002.json", 4096, 0))
    assert [(r["bucket_left"], r["bucket_right"], r["count"]) for r in rows] == [
        (0, 1, 2),
        (1, 2, 2),
    ]
