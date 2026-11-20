"""A large tensor has a raw file under /tensors only when it is a tensor summary.

The tab of a histogram, PR curve, mesh, or text summary already shows its
data, so a raw copy in /tensors is a duplicate in the wrong tab. A tensor
without a plugin name keeps its raw file.
"""

import json
import wave
from collections.abc import Callable
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from test_incremental_parse import _files, _frame

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2, mesh_plugin_data_pb2

DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_STRING = 1, 2, 4, 7
Fill = Callable[[compat_pb2.Summary.Value], None]


def _histogram(value: compat_pb2.Summary.Value) -> None:
    buckets = np.zeros((100, 3))
    buckets[:, 0] = np.arange(100)
    buckets[:, 1] = np.arange(1, 101)
    buckets[:, 2] = 1
    _tensor(value, DT_DOUBLE, buckets)


def _pr_curve(value: compat_pb2.Summary.Value) -> None:
    _tensor(value, DT_FLOAT, np.ones((6, 201), np.float32))


def _mesh(value: compat_pb2.Summary.Value) -> None:
    value.metadata.plugin_data.content = mesh_plugin_data_pb2.MeshPluginData(
        name="cloud", content_type=1, shape=[1, 200, 3]
    ).SerializeToString()
    _tensor(value, DT_FLOAT, np.ones((1, 200, 3), np.float32))


def _text(value: compat_pb2.Summary.Value) -> None:
    value.tensor.dtype = DT_STRING
    value.tensor.string_val.append(b"x" * 2000)


def _numbers(value: compat_pb2.Summary.Value) -> None:
    _tensor(value, DT_FLOAT, np.ones(400, np.float32))


def _tensor(value: compat_pb2.Summary.Value, dtype: int, data: np.ndarray) -> None:
    value.tensor.dtype = dtype
    value.tensor.tensor_shape.dim.extend(
        compat_pb2.TensorShapeProto.Dim(size=s) for s in data.shape
    )
    value.tensor.tensor_content = data.tobytes()


def _run(tmp_path: Path, plugin: str, fill: Fill) -> list[str]:
    event = compat_pb2.Event(step=1, wall_time=1.0)
    value = event.summary.value.add(tag="big")
    value.metadata.plugin_data.plugin_name = plugin
    fill(value)
    (tmp_path / "events.out.tfevents.1.host").write_bytes(_frame(event))
    return list(_files(TensorBoardFS(tmp_path)))


@pytest.mark.parametrize(
    ("plugin", "fill", "tab"),
    [
        ("histograms", _histogram, "/histograms/"),
        ("pr_curves", _pr_curve, "/pr_curves/"),
        ("mesh", _mesh, "/meshes/"),
        ("text", _text, "/text/"),
    ],
)
def test_a_large_plugin_tensor_has_no_raw_copy(
    tmp_path: Path, plugin: str, fill: Fill, tab: str
) -> None:
    files = _run(tmp_path, plugin, fill)

    assert any(path.startswith(tab) for path in files)
    assert [path for path in files if path.startswith("/tensors/")] == []


@pytest.mark.parametrize("plugin", ["tensors", "", "someone_elses_plugin"])
def test_a_large_tensor_summary_keeps_its_raw_file(tmp_path: Path, plugin: str) -> None:
    """A tensor without a known tab is a tensor summary."""
    files = _run(tmp_path, plugin, _numbers)

    assert "/tensors/big/000001.bin" in files


def test_an_unnamed_string_tensor_keeps_its_raw_file(tmp_path: Path) -> None:
    """Its strings can be a native file, which the raw file shows."""
    files = _run(tmp_path, "", _text)

    assert "/text/big/000001.txt" in files
    assert "/tensors/big/000001.txt" in files


def test_the_fixture_point_clouds_have_no_raw_copy() -> None:
    fs = TensorBoardFS(Path(__file__).resolve().parent.parent / "test-logs/meshes")

    assert [path for path in _files(fs) if path.startswith("/tensors/")] == []


def _wave_payload(size: int) -> bytes:
    """Make a mono, 400 Hz square wave with an exact total file size."""
    frames = size - 44
    cycle = bytes([96] * 10 + [160] * 10)
    samples = (cycle * ((frames + len(cycle) - 1) // len(cycle)))[:frames]
    with BytesIO() as out:
        with wave.open(out, "wb") as audio:
            audio.setparams((1, 1, 8000, 0, "NONE", "not compressed"))
            audio.writeframes(samples)
        payload = out.getvalue()
    assert len(payload) == size
    return payload


def _packed_tensor(
    payload: bytes, *shape: int, dtype: int = DT_FLOAT
) -> compat_pb2.TensorProto:
    tensor = compat_pb2.TensorProto(dtype=dtype, tensor_content=payload)
    for size in shape:
        tensor.tensor_shape.dim.add(size=size)
    return tensor


@pytest.mark.parametrize("reverse", [False, True], ids=["float-first", "float-last"])
def test_packed_float_raw_files_skip_embedded_detection(
    tmp_path: Path, reverse: bool
) -> None:
    """Identical bytes cannot share a file-type decision across tensor types."""
    wav = _wave_payload(1024)
    float_tensor = _packed_tensor(wav, 16, 16)
    unknown_rank = _packed_tensor(wav, 16, 16)
    unknown_rank.tensor_shape.unknown_rank = True
    repeated = _packed_tensor(wav, 16, 16)
    repeated.float_val.append(1.0)
    wrong_value_field = _packed_tensor(wav, 16, 16)
    wrong_value_field.int_val.append(1)
    conflicting_string = _packed_tensor(wav, 16, 16)
    conflicting_string.string_val.append(b"not the packed payload")
    variant = _packed_tensor(wav, 16, 16)
    variant.variant_val.add(type_name="extra")
    float8 = _packed_tensor(wav, 16, 16)
    float8.float8_val = b"extra"
    uint8 = _packed_tensor(wav, len(wav), dtype=DT_UINT8)
    double = _packed_tensor(wav, 8, 16, dtype=DT_DOUBLE)
    string = compat_pb2.TensorProto(dtype=DT_STRING, string_val=[wav])
    cases = [
        ("packed", float_tensor, "bin"),
        ("max_rank", _packed_tensor(wav, 256, *([1] * 253)), "bin"),
        ("excess_rank", _packed_tensor(wav, 256, *([1] * 254)), "wav"),
        ("uint8", uint8, "wav"),
        ("size_mismatch", _packed_tensor(wav, 16, 15), "wav"),
        ("unknown_rank", unknown_rank, "wav"),
        ("missing_shape", _packed_tensor(wav), "wav"),
        ("negative_dims", _packed_tensor(wav, -16, -16), "wav"),
        ("repeated", repeated, "wav"),
        ("wrong_value_field", wrong_value_field, "wav"),
        ("conflicting_string", conflicting_string, "wav"),
        ("variant", variant, "wav"),
        ("float8", float8, "wav"),
        ("double", double, "wav"),
        ("string", string, "wav"),
        ("small", _packed_tensor(_wave_payload(1020), 15, 17), None),
    ]
    event = compat_pb2.Event(step=1, wall_time=1.0)
    for tag, tensor, _ in reversed(cases) if reverse else cases:
        value = event.summary.value.add(tag=tag, tensor=tensor)
        value.metadata.plugin_data.plugin_name = "tensors"
    event.summary.value.add(tag="explicit_audio").audio.encoded_audio_string = wav
    (tmp_path / "events.out.tfevents.1.host").write_bytes(_frame(event))
    fs = TensorBoardFS(tmp_path)

    for tag, _, suffix in cases:
        directory = f"/tensors/{tag}"
        raw_files = {
            name
            for name in fs.readdir(directory)
            if name not in {".", "..", "000001.json", "000001.npy"}
        }
        if suffix is None:
            assert raw_files == set()
        else:
            name = f"000001.{suffix}"
            assert raw_files == {name}
            assert fs.read(f"{directory}/{name}", len(wav) + 1, 0) == wav
    assert fs.read("/audio/explicit_audio/000001.wav", len(wav) + 1, 0) == wav

    with BytesIO(fs.read("/tensors/packed/000001.npy", 4096, 0)) as encoded:
        floats = np.load(encoded, allow_pickle=False)
    assert floats.dtype == np.dtype("<f4") and floats.shape == (16, 16)
    assert floats.tobytes() == wav
    metadata = json.loads(fs.read("/tensors/packed/000001.json", 65536, 0))
    assert metadata == {
        "dtype": "float32",
        "shape": [16, 16],
        "values": floats.tolist(),
    }
