"""OBJ exports keep every batch element, vertex colors, and point clouds."""

from pathlib import Path

import numpy as np
import pytest
from test_incremental_parse import _files, _frame

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2, mesh_plugin_data_pb2

DT_FLOAT = 1
DT_INT32 = 3
DT_UINT8 = 4
VERTEX, FACE, COLOR = 1, 2, 3

TRIANGLES = np.array(
    [
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        [[0, 0, 1], [1, 0, 1], [0, 1, 1]],
    ],
    dtype="<f4",
)
FACES = np.array([[[0, 1, 2]], [[2, 1, 0]]], dtype="<i4")
COLORS = np.array(
    [
        [[255, 0, 0], [0, 51, 0], [0, 0, 255]],
        [[0, 0, 0], [51, 51, 51], [255, 255, 255]],
    ],
    dtype="u1",
)
DTYPES = {VERTEX: DT_FLOAT, FACE: DT_INT32, COLOR: DT_UINT8}


def _mesh(tmp_path: Path, components: dict[int, np.ndarray]) -> TensorBoardFS:
    event = compat_pb2.Event(wall_time=1.0, step=5)
    for content_type, data in components.items():
        value = event.summary.value.add(tag=f"shape_{content_type}")
        value.metadata.plugin_data.plugin_name = "mesh"
        value.metadata.plugin_data.content = mesh_plugin_data_pb2.MeshPluginData(
            name="shape", content_type=content_type, shape=data.shape
        ).SerializeToString()
        value.tensor.dtype = DTYPES[content_type]
        value.tensor.tensor_shape.dim.extend(
            compat_pb2.TensorShapeProto.Dim(size=s) for s in data.shape
        )
        value.tensor.tensor_content = data.tobytes()
    (tmp_path / "events.out.tfevents.host").write_bytes(_frame(event))
    return TensorBoardFS(tmp_path)


def _obj(fs: TensorBoardFS) -> list[str]:
    text = fs.read("/meshes/shape/000005.obj", 1 << 20, 0).decode()
    return text.splitlines()[1:]


def test_one_mesh_has_no_object_names(tmp_path: Path) -> None:
    """A batch of one gives plain vertex and face lines."""
    fs = _mesh(tmp_path, {VERTEX: TRIANGLES[:1], FACE: FACES[:1]})

    assert _obj(fs) == ["v 0.0 0.0 0.0", "v 1.0 0.0 0.0", "v 0.0 1.0 0.0", "f 1 2 3"]


def test_each_batch_element_is_one_object(tmp_path: Path) -> None:
    """Face indices of later objects count the vertices of earlier ones."""
    fs = _mesh(tmp_path, {VERTEX: TRIANGLES, FACE: FACES})

    assert _obj(fs) == [
        "o mesh_0",
        "v 0.0 0.0 0.0",
        "v 1.0 0.0 0.0",
        "v 0.0 1.0 0.0",
        "f 1 2 3",
        "o mesh_1",
        "v 0.0 0.0 1.0",
        "v 1.0 0.0 1.0",
        "v 0.0 1.0 1.0",
        "f 6 5 4",
    ]


def test_vertex_colors_follow_the_coordinates(tmp_path: Path) -> None:
    """Colors 0..255 become r g b values in [0, 1] after x y z."""
    fs = _mesh(tmp_path, {VERTEX: TRIANGLES, FACE: FACES, COLOR: COLORS})
    vertices = [line for line in _obj(fs) if line.startswith("v ")]

    assert vertices == [
        "v 0.0 0.0 0.0 1 0 0",
        "v 1.0 0.0 0.0 0 0.2 0",
        "v 0.0 1.0 0.0 0 0 1",
        "v 0.0 0.0 1.0 0 0 0",
        "v 1.0 0.0 1.0 0.2 0.2 0.2",
        "v 0.0 1.0 1.0 1 1 1",
    ]


def test_a_point_cloud_has_only_vertices(tmp_path: Path) -> None:
    """Vertices without faces are a valid OBJ point cloud."""
    fs = _mesh(tmp_path, {VERTEX: TRIANGLES[:1], COLOR: COLORS[:1]})

    assert _obj(fs) == [
        "v 0.0 0.0 0.0 1 0 0",
        "v 1.0 0.0 0.0 0 0.2 0",
        "v 0.0 1.0 0.0 0 0 1",
    ]


def test_colors_of_another_shape_are_left_out(tmp_path: Path) -> None:
    """Colors that do not match the vertices do not corrupt the OBJ."""
    fs = _mesh(tmp_path, {VERTEX: TRIANGLES[:1], COLOR: COLORS[:1, :2]})

    assert _obj(fs) == ["v 0.0 0.0 0.0", "v 1.0 0.0 0.0", "v 0.0 1.0 0.0"]
    assert "/meshes/shape/000005.npz" in _files(fs)


def test_colors_that_are_not_rgb_bytes_are_left_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A color above 255 is not an OBJ color above 1."""
    monkeypatch.setitem(DTYPES, COLOR, DT_INT32)
    fs = _mesh(tmp_path, {VERTEX: TRIANGLES[:1], COLOR: COLORS[:1].astype("<i4") + 45})

    assert _obj(fs) == ["v 0.0 0.0 0.0", "v 1.0 0.0 0.0", "v 0.0 1.0 0.0"]


def test_a_shorter_face_batch_leaves_later_objects_without_faces(
    tmp_path: Path,
) -> None:
    """Faces pair with the vertices of the same batch index only."""
    fs = _mesh(tmp_path, {VERTEX: TRIANGLES, FACE: FACES[:1]})

    assert [line for line in _obj(fs) if line.startswith(("o", "f"))] == [
        "o mesh_0",
        "f 1 2 3",
        "o mesh_1",
    ]


def test_unbatched_vertices_are_one_mesh(tmp_path: Path) -> None:
    """[N, 3] vertices without a batch axis are read as a batch of one."""
    fs = _mesh(tmp_path, {VERTEX: TRIANGLES[0], FACE: FACES[0]})

    assert _obj(fs) == ["v 0.0 0.0 0.0", "v 1.0 0.0 0.0", "v 0.0 1.0 0.0", "f 1 2 3"]


def test_vertices_that_are_not_3d_have_no_obj(tmp_path: Path) -> None:
    """2-D points cannot be OBJ vertices; the NPZ file still has them."""
    fs = _mesh(tmp_path, {VERTEX: TRIANGLES[:, :, :2].copy()})

    assert "/meshes/shape/000005.obj" not in _files(fs)
    assert "/meshes/shape/000005.npz" in _files(fs)
