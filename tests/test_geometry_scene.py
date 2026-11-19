"""Read two posed surfaces and two sensor sweeps without TensorFlow or TensorBoard."""

from collections import Counter
from io import BytesIO
import math

import numpy as np
from plyfile import PlyData
import pytest

from tboardfs import TensorBoardFS
from test_tf_producer_cases import CORPUS


@pytest.fixture(scope="module")
def fs() -> TensorBoardFS:
    return TensorBoardFS(CORPUS / "geometry-scene")


def _terrain(pose: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points, rgb, triangles = [], [], []
    for row in range(9):
        for col in range(13):
            x, y = (col - 6) / 4, (row - 4) / 4
            height = y * (x + 0.5)
            points.append((4 - y, x, height + 0.375) if pose else (x, y, height))
            color = (16 * col, 28 * row, 32 + 4 * row + col)
            rgb.append(color[::-1] if pose else color)
            if row < 8 and col < 12:
                v = row * 13 + col
                triangles.extend(((v, v + 1, v + 14), (v, v + 14, v + 13)))
    return np.array(points), np.array(rgb), np.array(triangles)


def _scan(pose: int) -> np.ndarray:
    points = []
    for ring in range(3):
        for bearing in range(16):
            angle = math.tau * bearing / 16
            radius = 2 + ring / 4 + 0.5 * math.cos(2 * angle) + pose / 8
            points.append(
                (
                    8 * pose + radius * math.cos(angle),
                    radius * math.sin(angle) - pose / 2,
                    radius * (ring - 1) / 8,
                )
            )
    return np.array(points)


@pytest.mark.parametrize("tag", ["terrain", "range-scan"])
def test_component_arrays(fs: TensorBoardFS, tag: str) -> None:
    with np.load(BytesIO(fs.read(f"/meshes/{tag}/000012.npz", 65536, 0))) as arrays:
        assert set(arrays.files) == (
            {"vertices", "colors", "faces"} if tag == "terrain" else {"vertices"}
        )
        assert arrays["vertices"].dtype == np.float32
        assert arrays["vertices"].shape == (2, 117 if tag == "terrain" else 48, 3)
        for pose in range(2):
            if tag == "terrain":
                points, colors, triangles = _terrain(pose)
                np.testing.assert_array_equal(
                    arrays["colors"][pose], colors.astype(np.uint8), strict=True
                )
                np.testing.assert_array_equal(
                    arrays["faces"][pose], triangles.astype(np.int32), strict=True
                )
            else:
                points = _scan(pose)
            np.testing.assert_allclose(arrays["vertices"][pose], points, atol=1e-6)


@pytest.mark.parametrize("tag", ["terrain", "range-scan"])
def test_obj_objects_keep_geometry_and_index_origin(
    fs: TensorBoardFS, tag: str
) -> None:
    lines = fs.read(f"/meshes/{tag}/000012.obj", 65536, 0).decode().splitlines()
    vertices: list[list[list[float]]] = []
    faces: list[list[list[int]]] = []
    for line in lines:
        op, *fields = line.split()
        if op == "o":
            vertices.append([])
            faces.append([])
        elif op == "v":
            vertices[-1].append([float(n) for n in fields])
        elif op == "f":
            faces[-1].append([int(n) for n in fields])
    assert len(vertices) == 2
    for pose, batch in enumerate(vertices):
        if tag == "terrain":
            points, colors, triangles = _terrain(pose)
            np.testing.assert_allclose(np.array(batch)[:, :3], points)
            np.testing.assert_allclose(np.array(batch)[:, 3:], colors / 255, atol=1e-6)
            np.testing.assert_array_equal(faces[pose], triangles + 1 + pose * 117)
        else:
            np.testing.assert_allclose(batch, _scan(pose), atol=1e-6)
            assert faces[pose] == []


@pytest.mark.parametrize("tag", ["terrain", "range-scan"])
def test_ply_topology_and_boundaries(fs: TensorBoardFS, tag: str) -> None:
    ply = PlyData.read(BytesIO(fs.read(f"/meshes/{tag}/000012.ply", 65536, 0)))
    xyz = np.column_stack([ply["vertex"][dim] for dim in "xyz"])
    if tag == "range-scan":
        assert [e.name for e in ply.elements] == ["vertex"]
        assert ply["vertex"].data.dtype.names == ("x", "y", "z")
        np.testing.assert_allclose(xyz, np.vstack([_scan(0), _scan(1)]), atol=1e-6)
        return
    assert xyz.shape == (234, 3)
    rgb = np.column_stack([ply["vertex"][c] for c in ("red", "green", "blue")])
    all_faces = np.stack(ply["face"]["vertex_indices"])
    assert all_faces.shape == (384, 3)
    for pose in range(2):
        points, colors, triangles = _terrain(pose)
        start = pose * 117
        np.testing.assert_array_equal(xyz[start : start + 117], points)
        np.testing.assert_array_equal(rgb[start : start + 117], colors)
        local = all_faces[pose * 192 : (pose + 1) * 192] - start
        np.testing.assert_array_equal(local, triangles)
        edges = Counter(
            tuple(sorted((int(a), int(b))))
            for triangle in local
            for a, b in zip(triangle, np.roll(triangle, -1), strict=True)
        )
        assert Counter(edges.values()) == {1: 40, 2: 268}
