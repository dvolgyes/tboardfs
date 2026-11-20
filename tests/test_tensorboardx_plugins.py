"""Read a TensorBoardX stereo recording and its associated model run offline."""

import json
import wave
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from test_proto3_reference import _fields, _values

from tboardfs import TensorBoardFS
from tboardfs.tfrecord import iter_tfrecords

RUN = Path(__file__).parent / "data/producer-cases/tensorboardx-plugins"


@pytest.fixture(scope="module")
def fs() -> TensorBoardFS:
    return TensorBoardFS(RUN)


def test_stereo_recording_has_two_audible_pitches(fs: TensorBoardFS) -> None:
    with wave.open(BytesIO(fs.read("/audio/interval/000003.wav", 65536, 0))) as wav:
        assert (
            wav.getsampwidth(),
            wav.getnchannels(),
            wav.getframerate(),
            wav.getnframes(),
        ) == (2, 2, 16000, 4000)
        samples = (
            np.frombuffer(wav.readframes(4000), dtype="<i2").reshape(-1, 2) / 32768
        )
    np.testing.assert_allclose(
        np.sqrt(np.mean(samples**2, axis=0)), [0.344, 0.344], atol=0.003
    )
    peaks = np.argmax(abs(np.fft.rfft(samples, n=16000, axis=0)), axis=0)
    np.testing.assert_array_equal(peaks, [220, 330])


def test_projector_links_vectors_labels_and_sprite(fs: TensorBoardFS) -> None:
    root = "/projector/00003/notes"
    config = fs.read("/projector/projector_config.pbtxt", 4096, 0).decode()
    for filename in ("metadata.tsv", "tensors.tsv", "sprite.png"):
        assert f'"00003/notes/{filename}"' in config
    values = np.loadtxt(BytesIO(fs.read(f"{root}/tensors.tsv", 4096, 0)))
    np.testing.assert_array_equal(
        values, [[220, 440], [330, 660], [440, 880], [660, 1320]]
    )
    assert fs.read(f"{root}/metadata.tsv", 4096, 0).decode().splitlines() == [
        "note\tHz",
        "A3\t220",
        "E4\t330",
        "A4\t440",
        "E5\t660",
    ]
    assert (
        fs.read(f"{root}/sprite.png", 65536, 0)
        == (RUN / "00003/notes/sprite.png").read_bytes()
    )


def test_trial_metrics_and_margin_layout(fs: TensorBoardFS) -> None:
    trial = json.loads(fs.read("/trial/hparams/hparams.json", 8192, 0))
    assert trial["session_start"]["hparams"] == {
        "model": "pitch",
        "lr": 0.125,
        "normalize": True,
    }
    assert trial["session_end"]["status"] == "STATUS_SUCCESS"
    assert trial["metrics"] == {"accuracy": 0.875}
    (chart,) = json.loads(fs.read("/custom_scalars/layout.json", 8192, 0))["category"]
    assert chart["title"] == "Validation"
    assert chart["chart"] == [
        {
            "title": "accuracy",
            "margin": {"series": [{"value": "score", "lower": "low", "upper": "high"}]},
        },
        {"title": "split", "multiline": {"tag": ["train", "eval"]}},
    ]
    for tag, expected in (("low", 0.75), ("score", 0.875), ("high", 1.0)):
        (row,) = json.loads(fs.read(f"/scalars/{tag}.json", 4096, 0))
        assert (row["step"], row["value"]) == (3, expected)


@pytest.mark.parametrize("mode", ["fields", "packed"])
@pytest.mark.parametrize(
    ("name", "dtype", "expected"),
    [
        ("FLOAT", "float32", [0.125, -0.75]),
        ("DOUBLE", "float64", [1 + 2**-48, -0.5]),
        ("HALF", "float16", [1.5, -2]),
        ("BFLOAT16", "float32", [1.5, -2]),
        ("COMPLEX64", "complex64", [1 + 2j, -1 + 0.5j]),
        ("COMPLEX128", "complex128", [1 + 2j, -1 + 0.5j]),
        ("BOOL", "bool", [False, True]),
        ("INT8", "int8", [-12, 97]),
        ("UINT8", "uint8", [3, 253]),
        ("INT16", "int16", [-8000, 12000]),
        ("UINT16", "uint16", [8000, 48000]),
        ("INT32", "int32", [-123456, 654321]),
        ("INT64", "int64", [2**53 + 1, -7]),
        ("QINT8", "int8", [-12, 97]),
        ("QUINT8", "uint8", [3, 253]),
        ("QINT16", "int16", [-8000, 12000]),
        ("QUINT16", "uint16", [8000, 48000]),
        ("QINT32", "int32", [-123456, 654321]),
    ],
)
def test_file_writer_tensor_values(
    fs: TensorBoardFS, mode: str, name: str, dtype: str, expected: list
) -> None:
    array = np.load(BytesIO(fs.read(f"/tensors/{mode}/DT_{name}/000003.npy", 65536, 0)))
    np.testing.assert_array_equal(array, np.array(expected, dtype), strict=True)


def test_string_values_and_opaque_resource(fs: TensorBoardFS) -> None:
    rows = json.loads(fs.read("/text/notes/000003.json", 8192, 0))
    assert rows["values"] == ["A3", "ré"]
    (source,) = RUN.glob("events.out.tfevents.*")
    path = f"/raw/{source.name}"
    data = b"".join(
        fs.read(f"{path}/{name}", 1 << 20, 0) for name in sorted(fs.readdir(path)[2:])
    )
    assert data == source.read_bytes()
    values = (
        _values(frame.payload)
        for frame in iter_tfrecords(BytesIO(data))
        if _fields(frame.payload, 5)
    )
    (value,) = (record["instrument"] for record in values if "instrument" in record)
    (proto,) = _fields(value, 8)
    (resource,) = _fields(proto, 14)
    assert _fields(resource, 2) == [b"synth"] and _fields(resource, 3) == [
        b"oscillator"
    ]
