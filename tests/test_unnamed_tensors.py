"""A numeric tensor without a plugin name is a tensor file, or a scalar."""

import io
from pathlib import Path

import numpy as np
from test_incremental_parse import _files as _all_files
from test_incremental_parse import _frame

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2

DT_FLOAT = 1


def _files(fs: TensorBoardFS) -> list[str]:
    return [path for path in _all_files(fs) if path != "/llms.txt"]


def _log(tmp_path: Path, values: list[float], shape: list[int]) -> TensorBoardFS:
    event = compat_pb2.Event(wall_time=1.0, step=3)
    value = event.summary.value.add(tag="w")
    value.tensor.dtype = DT_FLOAT
    value.tensor.tensor_shape.dim.extend(
        compat_pb2.TensorShapeProto.Dim(size=s) for s in shape
    )
    value.tensor.tensor_content = np.asarray(values, dtype="<f4").tobytes()
    (tmp_path / "events.out.tfevents.host").write_bytes(_frame(event))
    return TensorBoardFS(tmp_path)


def test_a_vector_without_plugin_name_is_a_tensor_file(tmp_path: Path) -> None:
    """The tensor keeps its values and shape instead of being dropped."""
    fs = _log(tmp_path, [1.0, 2.0, 3.0, 4.0], [2, 2])

    data = np.load(io.BytesIO(fs.read("/tensors/w/000003.npy", 1 << 20, 0)))

    assert sorted(_files(fs)) == [
        "/raw/events.out.tfevents.host/00000000000000000000.tfrecord",
        "/tensors/w/000003.json",
        "/tensors/w/000003.npy",
    ]
    np.testing.assert_array_equal(data, np.asarray([[1, 2], [3, 4]], dtype="<f4"))


def test_a_scalar_without_plugin_name_stays_a_scalar(tmp_path: Path) -> None:
    """A one-value tensor is a scalar row, not also a tensor file."""
    fs = _log(tmp_path, [0.5], [])

    assert not any(path.startswith("/tensors") for path in _files(fs))
    assert "/scalars/w.json" in _files(fs)


def test_a_vector_of_repeated_values_is_not_a_scalar(tmp_path: Path) -> None:
    """A vector in float_val, not tensor_content, is a tensor file too."""
    event = compat_pb2.Event(wall_time=1.0, step=3)
    value = event.summary.value.add(tag="w")
    value.tensor.dtype = DT_FLOAT
    value.tensor.tensor_shape.dim.add(size=3)
    value.tensor.float_val.extend([1.0, 2.0, 3.0])
    (tmp_path / "events.out.tfevents.host").write_bytes(_frame(event))
    fs = TensorBoardFS(tmp_path)

    data = np.load(io.BytesIO(fs.read("/tensors/w/000003.npy", 1 << 20, 0)))

    assert not any(path.startswith("/scalars") for path in _files(fs))
    np.testing.assert_array_equal(data, np.asarray([1, 2, 3], dtype="<f4"))
