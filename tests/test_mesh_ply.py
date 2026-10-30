"""PLY exports read back as the vertices, faces, and colors of the mesh."""

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2, mesh_plugin_data_pb2

from test_incremental_parse import _files, _frame
from test_mesh_obj import COLOR, COLORS, FACE, FACES, TRIANGLES, VERTEX

DTYPES = {"float32": 1, "float64": 2, "int32": 3, "uint8": 4}
PLY = "/meshes/shape/000005.ply"


def _mesh(tmp_path: Path, components: dict[int, np.ndarray]) -> TensorBoardFS:
    event = compat_pb2.Event(wall_time=1.0, step=5)
    for content_type, data in components.items():
        value = event.summary.value.add(tag=f"shape_{content_type}")
        value.metadata.plugin_data.plugin_name = "mesh"
        value.metadata.plugin_data.content = mesh_plugin_data_pb2.MeshPluginData(
            name="shape", content_type=content_type, shape=data.shape
        ).SerializeToString()
        value.tensor.dtype = DTYPES[data.dtype.name]
        value.tensor.tensor_shape.dim.extend(
            compat_pb2.TensorShapeProto.Dim(size=s) for s in data.shape
        )
        value.tensor.tensor_content = data.astype(
            data.dtype.newbyteorder("<")
        ).tobytes()
    (tmp_path / "events.out.tfevents.host").write_bytes(_frame(event))
    return TensorBoardFS(tmp_path)


def _ply(fs: TensorBoardFS, path: str = PLY) -> PlyData:
    return PlyData.read(BytesIO(fs.read(path, 1 << 24, 0)))


def _vertices(ply: PlyData) -> np.ndarray:
    vertex = ply["vertex"].data
    return np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)


def _colors(ply: PlyData) -> np.ndarray:
    vertex = ply["vertex"].data
    return np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=-1)


def _faces(ply: PlyData) -> np.ndarray:
    return np.stack(list(ply["face"].data["vertex_indices"])).astype(np.int64)


def _elements(ply: PlyData) -> list[str]:
    return [element.name for element in ply.elements]


def _properties(ply: PlyData) -> list[str]:
    return [prop.name for prop in ply["vertex"].properties]


def test_a_colored_mesh_reads_back_unchanged(tmp_path: Path) -> None:
    fs = _mesh(tmp_path, {VERTEX: TRIANGLES[:1], FACE: FACES[:1], COLOR: COLORS[:1]})
    ply = _ply(fs)

    assert ply.text is False
    assert _vertices(ply).dtype == np.float32
    assert np.array_equal(_vertices(ply), TRIANGLES[0])
    assert np.array_equal(_faces(ply), FACES[0])
    assert _colors(ply).dtype == np.uint8
    assert np.array_equal(_colors(ply), COLORS[0])


def test_a_mesh_without_colors_has_no_color_properties(tmp_path: Path) -> None:
    fs = _mesh(tmp_path, {VERTEX: TRIANGLES[:1], FACE: FACES[:1]})
    ply = _ply(fs)

    assert _properties(ply) == ["x", "y", "z"]
    assert np.array_equal(_vertices(ply), TRIANGLES[0])
    assert np.array_equal(_faces(ply), FACES[0])


def test_a_colored_point_cloud_has_no_faces(tmp_path: Path) -> None:
    fs = _mesh(tmp_path, {VERTEX: TRIANGLES[:1], COLOR: COLORS[:1]})
    ply = _ply(fs)

    assert _elements(ply) == ["vertex"]
    assert np.array_equal(_vertices(ply), TRIANGLES[0])
    assert np.array_equal(_colors(ply), COLORS[0])


def test_a_point_cloud_has_only_coordinates(tmp_path: Path) -> None:
    fs = _mesh(tmp_path, {VERTEX: TRIANGLES[:1]})
    ply = _ply(fs)

    assert _elements(ply) == ["vertex"]
    assert _properties(ply) == ["x", "y", "z"]
    assert np.array_equal(_vertices(ply), TRIANGLES[0])


def test_a_batch_is_one_mesh_with_offset_faces(tmp_path: Path) -> None:
    """Face indices of a later batch element count the vertices before it."""
    fs = _mesh(tmp_path, {VERTEX: TRIANGLES, FACE: FACES, COLOR: COLORS})
    ply = _ply(fs)

    assert np.array_equal(_vertices(ply), TRIANGLES.reshape(-1, 3))
    assert np.array_equal(_faces(ply), [[0, 1, 2], [5, 4, 3]])
    assert np.array_equal(_colors(ply), COLORS.reshape(-1, 3))
    assert [c for c in ply.comments if c.startswith("mesh_")] == [
        "mesh_0 vertices 0 3",
        "mesh_1 vertices 3 3",
    ]


def test_a_shorter_face_batch_gives_faces_to_the_first_elements(
    tmp_path: Path,
) -> None:
    fs = _mesh(tmp_path, {VERTEX: TRIANGLES, FACE: FACES[:1]})

    assert np.array_equal(_faces(_ply(fs)), FACES[0])


def test_faces_of_a_larger_batch_than_the_vertices_are_left_out(
    tmp_path: Path,
) -> None:
    """Faces of a batch element without vertices would index past the end."""
    fs = _mesh(tmp_path, {VERTEX: TRIANGLES[:1], FACE: FACES})

    assert np.array_equal(_faces(_ply(fs)), FACES[0])


def test_double_vertices_stay_double(tmp_path: Path) -> None:
    """Float64 coordinates that float32 cannot hold read back exactly."""
    vertices = TRIANGLES[:1].astype(np.float64) + 1e-12
    fs = _mesh(tmp_path, {VERTEX: vertices, FACE: FACES[:1]})
    ply = _ply(fs)

    assert _vertices(ply).dtype == np.float64
    assert np.array_equal(_vertices(ply), vertices[0])


@pytest.mark.parametrize(
    "colors",
    [
        pytest.param(COLORS[:1, :2], id="another shape"),
        pytest.param(COLORS[:1].astype(np.float32) / 255, id="not whole numbers"),
        pytest.param(COLORS[:1].astype(np.int32) + 1, id="above 255"),
        pytest.param(COLORS[:1].astype(np.int32) - 1, id="below 0"),
    ],
)
def test_colors_that_are_not_rgb_bytes_are_left_out(
    tmp_path: Path, colors: np.ndarray
) -> None:
    """The NPZ file keeps such colors; the PLY file keeps the geometry."""
    fs = _mesh(tmp_path, {VERTEX: TRIANGLES[:1], FACE: FACES[:1], COLOR: colors})
    ply = _ply(fs)

    assert _properties(ply) == ["x", "y", "z"]
    assert np.array_equal(_faces(ply), FACES[0])


def test_faces_that_are_not_vertex_indices_are_left_out(tmp_path: Path) -> None:
    fs = _mesh(tmp_path, {VERTEX: TRIANGLES[:1], FACE: FACES[:1] + 1})

    assert _elements(_ply(fs)) == ["vertex"]


def test_vertices_that_are_not_3d_have_no_ply(tmp_path: Path) -> None:
    fs = _mesh(tmp_path, {VERTEX: TRIANGLES[:, :, :2].copy()})

    assert PLY not in _files(fs)
    assert "/meshes/shape/000005.npz" in _files(fs)


def test_a_mesh_without_vertices_has_no_files(tmp_path: Path) -> None:
    """Faces and colors alone are not a mesh."""
    fs = _mesh(tmp_path, {FACE: FACES[:1], COLOR: COLORS[:1]})

    assert [path for path in _files(fs) if path.startswith("/meshes/")] == []


def _fixtures() -> TensorBoardFS:
    return TensorBoardFS(Path(__file__).resolve().parent.parent / "test-logs")


def test_each_fixture_mesh_has_a_ply_of_its_npz_arrays() -> None:
    """The fixtures come from TensorBoard and tensorboardX writers."""
    fs = _fixtures()
    plys = [path for path in _files(fs) if path.endswith(".ply")]

    assert len(plys) >= 11
    for path in plys:
        arrays = np.load(BytesIO(fs.read(path[:-4] + ".npz", 1 << 24, 0)))
        ply = _ply(fs, path)
        vertices = arrays["vertices"]
        assert np.array_equal(_vertices(ply), vertices.reshape(-1, 3))
        if "faces" in arrays:
            offsets = np.arange(len(vertices))[:, None, None] * vertices.shape[1]
            faces = (arrays["faces"] + offsets).reshape(-1, 3)
            assert np.array_equal(_faces(ply), faces)
        else:
            assert _elements(ply) == ["vertex"]
        if "colors" in arrays:
            assert np.array_equal(_colors(ply), arrays["colors"].reshape(-1, 3))
        else:
            assert _properties(ply) == ["x", "y", "z"]


TETRAHEDRON = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
TETRAHEDRON_FACES = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
CORNER_COLORS = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]


def test_the_tensorboard_meshes_read_back_as_written() -> None:
    """scripts/generate_mesh_logs.py writes these values."""
    fs = _fixtures()
    colored = _ply(fs, "/meshes/meshes/colored_mesh/000001.ply")
    plain = _ply(fs, "/meshes/meshes/mesh/000001.ply")
    batch = _ply(fs, "/meshes/meshes/batch/000001.ply")

    assert np.array_equal(_vertices(colored), TETRAHEDRON)
    assert np.array_equal(_faces(colored), TETRAHEDRON_FACES)
    assert np.array_equal(_colors(colored), CORNER_COLORS)
    assert _properties(plain) == ["x", "y", "z"]
    assert np.array_equal(_faces(plain), TETRAHEDRON_FACES)
    assert np.array_equal(
        _vertices(batch), TETRAHEDRON + (np.array(TETRAHEDRON) + 2).tolist()
    )
    assert np.array_equal(
        _faces(batch),
        TETRAHEDRON_FACES + (np.array(TETRAHEDRON_FACES)[:, ::-1] + 4).tolist(),
    )
    assert np.array_equal(_colors(batch), CORNER_COLORS + CORNER_COLORS[::-1])


def test_the_tensorboard_point_clouds_have_no_faces() -> None:
    fs = _fixtures()
    colored = _ply(fs, "/meshes/meshes/colored_cloud/000001.ply")
    plain = _ply(fs, "/meshes/meshes/cloud/000001.ply")

    assert _elements(colored) == _elements(plain) == ["vertex"]
    assert len(_vertices(colored)) == 200
    assert np.array_equal(_vertices(colored), _vertices(plain))
    assert _properties(plain) == ["x", "y", "z"]
    assert len(np.unique(_colors(colored), axis=0)) > 100
