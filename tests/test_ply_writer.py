"""Tests for PLY file format writer."""

import pytest
import tempfile
import struct
from pathlib import Path
import numpy as np

from tboardfs.core.ply_writer import (
    PLYWriter,
    write_mesh_as_ply,
    write_point_cloud_as_ply,
)
from tboardfs.core.data_types import MeshData, PointCloudData


class TestPLYWriter:
    """Test PLYWriter class."""

    def test_init_binary_format(self):
        """Test initialization with binary format."""
        writer = PLYWriter("binary")
        assert writer.format_type == "binary"

    def test_init_text_format(self):
        """Test initialization with text format."""
        writer = PLYWriter("text")
        assert writer.format_type == "text"

    def test_init_invalid_format(self):
        """Test initialization with invalid format."""
        with pytest.raises(ValueError) as exc_info:
            PLYWriter("invalid")

        assert "Invalid PLY format" in str(exc_info.value)
        assert "Must be 'binary' or 'text'" in str(exc_info.value)

    def test_validate_mesh_data_valid_point_cloud(self):
        """Test validation of valid point cloud data."""
        vertices = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
        mesh_data = MeshData(step=0, vertices=vertices, wall_time=0.0)

        writer = PLYWriter()
        writer._validate_mesh_data(mesh_data)  # Should not raise

    def test_validate_mesh_data_valid_mesh_with_faces(self):
        """Test validation of valid mesh data with faces."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh_data = MeshData(step=0, vertices=vertices, faces=faces, wall_time=0.0)

        writer = PLYWriter()
        writer._validate_mesh_data(mesh_data)  # Should not raise

    def test_validate_mesh_data_valid_mesh_with_colors(self):
        """Test validation of valid mesh data with colors."""
        vertices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        mesh_data = MeshData(step=0, vertices=vertices, colors=colors, wall_time=0.0)

        writer = PLYWriter()
        writer._validate_mesh_data(mesh_data)  # Should not raise

    def test_validate_mesh_data_empty_vertices(self):
        """Test validation with empty vertices."""
        vertices = np.array([], dtype=np.float32).reshape(0, 3)
        mesh_data = MeshData(step=0, vertices=vertices, wall_time=0.0)

        writer = PLYWriter()
        with pytest.raises(ValueError) as exc_info:
            writer._validate_mesh_data(mesh_data)

        assert "must contain at least one vertex" in str(exc_info.value)

    def test_validate_mesh_data_none_vertices(self):
        """Test validation with None vertices."""
        mesh_data = MeshData(step=0, vertices=None, wall_time=0.0)

        writer = PLYWriter()
        with pytest.raises(ValueError) as exc_info:
            writer._validate_mesh_data(mesh_data)

        assert "must contain at least one vertex" in str(exc_info.value)

    def test_validate_mesh_data_wrong_vertex_dimensions(self):
        """Test validation with wrong vertex dimensions."""
        vertices = np.array([[0, 0], [1, 1]], dtype=np.float32)  # Only 2D
        mesh_data = MeshData(step=0, vertices=vertices, wall_time=0.0)

        writer = PLYWriter()
        with pytest.raises(ValueError) as exc_info:
            writer._validate_mesh_data(mesh_data)

        assert "Vertices must have 3 coordinates (XYZ)" in str(exc_info.value)

    def test_validate_mesh_data_wrong_face_dimensions(self):
        """Test validation with wrong face dimensions."""
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32
        )
        faces = np.array([[0, 1, 2, 3]], dtype=np.int32)  # Quad instead of triangle
        mesh_data = MeshData(step=0, vertices=vertices, faces=faces, wall_time=0.0)

        writer = PLYWriter()
        writer._validate_mesh_data(mesh_data)

        # Should convert to point cloud (faces set to None)
        assert mesh_data.faces is None

    def test_validate_mesh_data_invalid_face_indices(self):
        """Test validation with invalid face indices."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 5]], dtype=np.int32)  # Index 5 doesn't exist
        mesh_data = MeshData(step=0, vertices=vertices, faces=faces, wall_time=0.0)

        writer = PLYWriter()
        writer._validate_mesh_data(mesh_data)

        # Should convert to point cloud (faces set to None)
        assert mesh_data.faces is None

    def test_validate_mesh_data_negative_face_indices(self):
        """Test validation with negative face indices."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[-1, 1, 2]], dtype=np.int32)  # Negative index
        mesh_data = MeshData(step=0, vertices=vertices, faces=faces, wall_time=0.0)

        writer = PLYWriter()
        writer._validate_mesh_data(mesh_data)

        # Should convert to point cloud (faces set to None)
        assert mesh_data.faces is None

    def test_validate_mesh_data_color_count_mismatch(self):
        """Test validation with mismatched color count."""
        vertices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        colors = np.array(
            [[1.0, 0.0, 0.0]], dtype=np.float32
        )  # Only 1 color for 2 vertices
        mesh_data = MeshData(step=0, vertices=vertices, colors=colors, wall_time=0.0)

        writer = PLYWriter()
        with pytest.raises(ValueError) as exc_info:
            writer._validate_mesh_data(mesh_data)

        assert "Color count" in str(exc_info.value)
        assert "must match vertex count" in str(exc_info.value)

    def test_validate_mesh_data_wrong_color_dimensions(self):
        """Test validation with wrong color dimensions."""
        vertices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        colors = np.array(
            [[1.0, 0.0], [0.0, 1.0]], dtype=np.float32
        )  # Only 2 components
        mesh_data = MeshData(step=0, vertices=vertices, colors=colors, wall_time=0.0)

        writer = PLYWriter()
        with pytest.raises(ValueError) as exc_info:
            writer._validate_mesh_data(mesh_data)

        assert "Colors must have 3 components (RGB)" in str(exc_info.value)

    def test_write_binary_point_cloud(self):
        """Test writing point cloud in binary format."""
        vertices = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
        mesh_data = MeshData(step=0, vertices=vertices, wall_time=0.0)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            writer = PLYWriter("binary")
            writer.write_mesh(mesh_data, tmp_path)

            # Verify file was created and has content
            assert tmp_path.exists()
            assert tmp_path.stat().st_size > 0

            # Read and verify header
            with tmp_path.open("rb") as f:
                header_content = f.read(1000).decode("ascii", errors="ignore")
                assert "ply" in header_content
                assert "format binary_little_endian 1.0" in header_content
                assert "element vertex 3" in header_content
                assert "property float x" in header_content
                assert "property float y" in header_content
                assert "property float z" in header_content
                assert "end_header" in header_content

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_write_text_point_cloud(self):
        """Test writing point cloud in text format."""
        vertices = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
        mesh_data = MeshData(step=0, vertices=vertices, wall_time=0.0)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            writer = PLYWriter("text")
            writer.write_mesh(mesh_data, tmp_path)

            # Verify file was created and has content
            assert tmp_path.exists()
            assert tmp_path.stat().st_size > 0

            # Read and verify content
            content = tmp_path.read_text()
            lines = content.strip().split("\n")

            assert lines[0] == "ply"
            assert "format ascii 1.0" in content
            assert "element vertex 3" in content
            assert "property float x" in content
            assert "end_header" in content

            # Check vertex data
            assert "0.0 0.0 0.0" in content
            assert "1.0 1.0 1.0" in content
            assert "2.0 2.0 2.0" in content

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_write_binary_mesh_with_faces(self):
        """Test writing mesh with faces in binary format."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh_data = MeshData(step=0, vertices=vertices, faces=faces, wall_time=0.0)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            writer = PLYWriter("binary")
            writer.write_mesh(mesh_data, tmp_path)

            # Verify file was created and has content
            assert tmp_path.exists()
            assert tmp_path.stat().st_size > 0

            # Read and verify header
            with tmp_path.open("rb") as f:
                header_content = f.read(1000).decode("ascii", errors="ignore")
                assert "ply" in header_content
                assert "element vertex 3" in header_content
                assert "element face 1" in header_content
                assert "property list uchar int vertex_indices" in header_content

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_write_text_mesh_with_faces(self):
        """Test writing mesh with faces in text format."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh_data = MeshData(step=0, vertices=vertices, faces=faces, wall_time=0.0)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            writer = PLYWriter("text")
            writer.write_mesh(mesh_data, tmp_path)

            # Verify file was created and has content
            assert tmp_path.exists()
            assert tmp_path.stat().st_size > 0

            # Read and verify content
            content = tmp_path.read_text()

            assert "element vertex 3" in content
            assert "element face 1" in content
            assert "property list uchar int vertex_indices" in content

            # Check vertex data
            assert "0.0 0.0 0.0" in content
            assert "1.0 0.0 0.0" in content
            assert "0.0 1.0 0.0" in content

            # Check face data
            assert "3 0 1 2" in content

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_write_binary_mesh_with_colors(self):
        """Test writing mesh with colors in binary format."""
        vertices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        mesh_data = MeshData(step=0, vertices=vertices, colors=colors, wall_time=0.0)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            writer = PLYWriter("binary")
            writer.write_mesh(mesh_data, tmp_path)

            # Verify file was created and has content
            assert tmp_path.exists()
            assert tmp_path.stat().st_size > 0

            # Read and verify header
            with tmp_path.open("rb") as f:
                header_content = f.read(1000).decode("ascii", errors="ignore")
                assert "property uchar red" in header_content
                assert "property uchar green" in header_content
                assert "property uchar blue" in header_content

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_write_text_mesh_with_colors(self):
        """Test writing mesh with colors in text format."""
        vertices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        mesh_data = MeshData(step=0, vertices=vertices, colors=colors, wall_time=0.0)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            writer = PLYWriter("text")
            writer.write_mesh(mesh_data, tmp_path)

            # Verify file was created and has content
            assert tmp_path.exists()
            assert tmp_path.stat().st_size > 0

            # Read and verify content
            content = tmp_path.read_text()

            assert "property uchar red" in content
            assert "property uchar green" in content
            assert "property uchar blue" in content

            # Check vertex data with colors (RGB values converted to 0-255 range)
            assert "0.0 0.0 0.0 255 0 0" in content  # Red vertex
            assert "1.0 1.0 1.0 0 255 0" in content  # Green vertex

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_write_point_cloud_data(self):
        """Test writing PointCloudData object."""
        vertices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        point_data = PointCloudData(
            step=0, vertices=vertices, colors=colors, wall_time=0.0
        )

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            writer = PLYWriter("text")
            writer.write_point_cloud(point_data, tmp_path)

            # Verify file was created and has content
            assert tmp_path.exists()
            assert tmp_path.stat().st_size > 0

            # Read and verify content
            content = tmp_path.read_text()
            assert "element vertex 2" in content
            assert "property uchar red" in content

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_color_clamping(self):
        """Test that colors are properly clamped to 0-255 range."""
        vertices = np.array([[0, 0, 0]], dtype=np.float32)
        colors = np.array([[2.0, -0.5, 0.5]], dtype=np.float32)  # Out of 0-1 range
        mesh_data = MeshData(step=0, vertices=vertices, colors=colors, wall_time=0.0)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            writer = PLYWriter("text")
            writer.write_mesh(mesh_data, tmp_path)

            content = tmp_path.read_text()

            # Colors should be clamped: 2.0 -> 255, -0.5 -> 0, 0.5 -> 127
            assert "0.0 0.0 0.0 255 0 127" in content

        finally:
            if tmp_path.exists():
                tmp_path.unlink()


class TestConvenienceFunctions:
    """Test convenience functions for PLY writing."""

    def test_write_mesh_as_ply_binary(self):
        """Test write_mesh_as_ply convenience function with binary format."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh_data = MeshData(step=0, vertices=vertices, faces=faces, wall_time=0.0)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            write_mesh_as_ply(mesh_data, tmp_path, "binary")

            assert tmp_path.exists()
            assert tmp_path.stat().st_size > 0

            # Verify it's binary format
            with tmp_path.open("rb") as f:
                header_content = f.read(1000).decode("ascii", errors="ignore")
                assert "format binary_little_endian 1.0" in header_content

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_write_mesh_as_ply_text(self):
        """Test write_mesh_as_ply convenience function with text format."""
        vertices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        mesh_data = MeshData(step=0, vertices=vertices, wall_time=0.0)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            write_mesh_as_ply(mesh_data, tmp_path, "text")

            assert tmp_path.exists()
            assert tmp_path.stat().st_size > 0

            # Verify it's text format
            content = tmp_path.read_text()
            assert "format ascii 1.0" in content
            assert "0.0 0.0 0.0" in content
            assert "1.0 1.0 1.0" in content

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_write_mesh_as_ply_default_format(self):
        """Test write_mesh_as_ply convenience function with default format."""
        vertices = np.array([[0, 0, 0]], dtype=np.float32)
        mesh_data = MeshData(step=0, vertices=vertices, wall_time=0.0)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            write_mesh_as_ply(mesh_data, tmp_path)  # Default should be binary

            assert tmp_path.exists()
            assert tmp_path.stat().st_size > 0

            # Verify it's binary format (default)
            with tmp_path.open("rb") as f:
                header_content = f.read(1000).decode("ascii", errors="ignore")
                assert "format binary_little_endian 1.0" in header_content

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_write_point_cloud_as_ply_binary(self):
        """Test write_point_cloud_as_ply convenience function with binary format."""
        vertices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        point_data = PointCloudData(
            step=0, vertices=vertices, colors=colors, wall_time=0.0
        )

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            write_point_cloud_as_ply(point_data, tmp_path, "binary")

            assert tmp_path.exists()
            assert tmp_path.stat().st_size > 0

            # Verify it's binary format
            with tmp_path.open("rb") as f:
                header_content = f.read(1000).decode("ascii", errors="ignore")
                assert "format binary_little_endian 1.0" in header_content
                assert "property uchar red" in header_content

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_write_point_cloud_as_ply_text(self):
        """Test write_point_cloud_as_ply convenience function with text format."""
        vertices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        point_data = PointCloudData(step=0, vertices=vertices, wall_time=0.0)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            write_point_cloud_as_ply(point_data, tmp_path, "text")

            assert tmp_path.exists()
            assert tmp_path.stat().st_size > 0

            # Verify it's text format
            content = tmp_path.read_text()
            assert "format ascii 1.0" in content
            assert "0.0 0.0 0.0" in content
            assert "1.0 1.0 1.0" in content

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_write_point_cloud_as_ply_default_format(self):
        """Test write_point_cloud_as_ply convenience function with default format."""
        vertices = np.array([[0, 0, 0]], dtype=np.float32)
        point_data = PointCloudData(step=0, vertices=vertices, wall_time=0.0)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            write_point_cloud_as_ply(point_data, tmp_path)  # Default should be binary

            assert tmp_path.exists()
            assert tmp_path.stat().st_size > 0

            # Verify it's binary format (default)
            with tmp_path.open("rb") as f:
                header_content = f.read(1000).decode("ascii", errors="ignore")
                assert "format binary_little_endian 1.0" in header_content

        finally:
            if tmp_path.exists():
                tmp_path.unlink()


class TestBinaryFormatDetails:
    """Test specific binary format encoding details."""

    def test_binary_vertex_encoding(self):
        """Test that binary vertices are encoded correctly."""
        vertices = np.array([[1.5, 2.5, 3.5]], dtype=np.float32)
        mesh_data = MeshData(step=0, vertices=vertices, wall_time=0.0)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            writer = PLYWriter("binary")
            writer.write_mesh(mesh_data, tmp_path)

            # Read binary data and check vertex encoding
            with tmp_path.open("rb") as f:
                # Skip header
                content = f.read()
                header_end = content.find(b"end_header\n") + len(b"end_header\n")
                binary_data = content[header_end:]

                # Unpack the vertex data (3 floats in little-endian format)
                x, y, z = struct.unpack("<fff", binary_data[:12])
                assert abs(x - 1.5) < 1e-6
                assert abs(y - 2.5) < 1e-6
                assert abs(z - 3.5) < 1e-6

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_binary_color_encoding(self):
        """Test that binary colors are encoded correctly."""
        vertices = np.array([[0, 0, 0]], dtype=np.float32)
        colors = np.array(
            [[0.5, 1.0, 0.0]], dtype=np.float32
        )  # 50% red, 100% green, 0% blue
        mesh_data = MeshData(step=0, vertices=vertices, colors=colors, wall_time=0.0)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            writer = PLYWriter("binary")
            writer.write_mesh(mesh_data, tmp_path)

            # Read binary data and check color encoding
            with tmp_path.open("rb") as f:
                content = f.read()
                header_end = content.find(b"end_header\n") + len(b"end_header\n")
                binary_data = content[header_end:]

                # Unpack vertex (3 floats) + color (3 bytes)
                x, y, z = struct.unpack("<fff", binary_data[:12])
                r, g, b = struct.unpack("<BBB", binary_data[12:15])

                assert abs(x - 0.0) < 1e-6
                assert abs(y - 0.0) < 1e-6
                assert abs(z - 0.0) < 1e-6
                assert r == 127  # 0.5 * 255 = 127.5 -> 127
                assert g == 255  # 1.0 * 255 = 255
                assert b == 0  # 0.0 * 255 = 0

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_binary_face_encoding(self):
        """Test that binary faces are encoded correctly."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh_data = MeshData(step=0, vertices=vertices, faces=faces, wall_time=0.0)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            writer = PLYWriter("binary")
            writer.write_mesh(mesh_data, tmp_path)

            # Read binary data and check face encoding
            with tmp_path.open("rb") as f:
                content = f.read()
                header_end = content.find(b"end_header\n") + len(b"end_header\n")
                binary_data = content[header_end:]

                # Skip vertices (3 vertices * 3 floats * 4 bytes = 36 bytes)
                face_data = binary_data[36:]

                # Unpack face data: 1 byte for vertex count + 3 ints for indices
                vertex_count = struct.unpack("<B", face_data[:1])[0]
                v1, v2, v3 = struct.unpack("<III", face_data[1:13])

                assert vertex_count == 3
                assert v1 == 0
                assert v2 == 1
                assert v3 == 2

        finally:
            if tmp_path.exists():
                tmp_path.unlink()
