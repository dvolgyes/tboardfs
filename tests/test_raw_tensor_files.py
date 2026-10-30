"""A large tensor has a raw file under /tensors only when it is a tensor summary.

The tab of a histogram, PR curve, mesh, or text summary already shows its
data, so a raw copy in /tensors is a duplicate in the wrong tab. A tensor
without a plugin name keeps its raw file.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2, mesh_plugin_data_pb2

from test_incremental_parse import _files, _frame

DT_FLOAT, DT_DOUBLE, DT_STRING = 1, 2, 7
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
