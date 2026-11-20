"""Mesh files are grouped by mesh tag, so meshes of one name in two scopes stay apart.

TensorBoard groups mesh tensors by MeshPluginData.name, which has no name
scope; tboardfs groups them by the tag without its _VERTEX, _FACE or _COLOR
suffix, which has the scope.
"""

import io
import json
from pathlib import Path

import numpy as np
import pytest
from loguru import logger
from test_incremental_parse import _files, _frame

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2, mesh_plugin_data_pb2

DT_FLOAT = 1
DT_UINT8 = 4
VERTEX, FACE, COLOR = 1, 2, 3


def _add(
    event: compat_pb2.Event, tag: str, name: str, content_type: int, data: np.ndarray
) -> None:
    value = event.summary.value.add(tag=tag)
    value.metadata.plugin_data.plugin_name = "mesh"
    value.metadata.plugin_data.content = mesh_plugin_data_pb2.MeshPluginData(
        name=name, content_type=content_type, shape=data.shape
    ).SerializeToString()
    value.tensor.dtype = DT_UINT8 if data.dtype == np.uint8 else DT_FLOAT
    value.tensor.tensor_shape.dim.extend(
        compat_pb2.TensorShapeProto.Dim(size=s) for s in data.shape
    )
    value.tensor.tensor_content = data.tobytes()


def _points(value: float) -> np.ndarray:
    return np.full((1, 2, 3), value, dtype="<f4")


def _fs(tmp_path: Path, event: compat_pb2.Event) -> TensorBoardFS:
    (tmp_path / "events.out.tfevents.host").write_bytes(_frame(event))
    return TensorBoardFS(tmp_path)


def _npz(fs: TensorBoardFS, path: str) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(fs.read(path, 1 << 20, 0))) as data:
        return dict(data)


def test_meshes_of_one_name_in_two_scopes_stay_apart(tmp_path: Path) -> None:
    """The vertices and colors of train/cloud do not mix with val/cloud."""
    event = compat_pb2.Event(wall_time=1.0, step=1)
    _add(event, "train/cloud_VERTEX", "cloud", VERTEX, _points(1.0))
    _add(event, "train/cloud_COLOR", "cloud", COLOR, np.zeros((1, 2, 3), "u1"))
    _add(event, "val/cloud_VERTEX", "cloud", VERTEX, _points(2.0))
    fs = _fs(tmp_path, event)

    train = _npz(fs, "/meshes/train/cloud/000001.npz")
    val = _npz(fs, "/meshes/val/cloud/000001.npz")
    assert sorted(train) == ["colors", "vertices"]
    assert train["vertices"].ravel().tolist() == [1.0] * 6
    assert sorted(val) == ["vertices"]
    assert val["vertices"].ravel().tolist() == [2.0] * 6
    assert not [path for path in _files(fs) if path.startswith("/meshes/cloud/")]
    metadata = json.loads(fs.read("/meshes/val/cloud/000001.json", 1 << 20, 0))
    assert metadata["name"] == "cloud"


def test_a_tag_without_a_type_suffix_uses_the_mesh_name(tmp_path: Path) -> None:
    """Tags that do not end in _VERTEX or _FACE are grouped by the mesh name."""
    event = compat_pb2.Event(wall_time=1.0, step=1)
    _add(event, "box_vertices", "box", VERTEX, _points(1.0))
    _add(event, "box_faces", "box", FACE, np.zeros((1, 1, 3), "<f4"))
    _add(event, "plain", "", VERTEX, _points(3.0))
    fs = _fs(tmp_path, event)

    assert sorted(_npz(fs, "/meshes/box/000001.npz")) == ["faces", "vertices"]
    assert sorted(_npz(fs, "/meshes/plain/000001.npz")) == ["vertices"]


@pytest.fixture
def warnings() -> list[str]:
    """Return the warning messages that tboardfs logs during the test."""
    messages: list[str] = []
    sink = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages  # type: ignore[misc]
    logger.remove(sink)


def test_mesh_tags_with_the_same_path_keep_their_own_files(
    tmp_path: Path, warnings: list[str]
) -> None:
    """The empty component separates the two mesh directories without a rename."""
    event = compat_pb2.Event(wall_time=1.0, step=1)
    _add(event, "a/b_VERTEX", "b", VERTEX, _points(1.0))
    _add(event, "a//b_VERTEX", "b", VERTEX, _points(2.0))
    fs = _fs(tmp_path, event)

    plain = _npz(fs, "/meshes/a/b/000001.npz")["vertices"]
    doubled = _npz(fs, "/meshes/a/~empty/b/000001.npz")["vertices"]
    assert plain.ravel().tolist() == [1.0] * 6
    assert doubled.ravel().tolist() == [2.0] * 6
    assert not warnings
