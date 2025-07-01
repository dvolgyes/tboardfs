"""PLY file format writer for 3D meshes and point clouds.

Supports both binary and text PLY formats with comprehensive coverage of:
- Point clouds (vertices only)
- Meshes (vertices + faces)
- Colors (RGB per vertex)
- All corner cases and validation
"""

import struct
from pathlib import Path
from typing import BinaryIO, TextIO
import numpy as np
from loguru import logger

from .data_types import MeshData, PointCloudData


class PLYWriter:
    """Comprehensive PLY file format writer."""

    def __init__(self, format_type: str = "binary"):
        """Initialize PLY writer.

        Args:
            format_type: "binary" or "text" format
        """
        if format_type not in ("binary", "text"):
            raise ValueError(
                f"Invalid PLY format: {format_type}. Must be 'binary' or 'text'"
            )
        self.format_type = format_type

    def write_mesh(self, mesh_data: MeshData, output_path: Path) -> None:
        """Write mesh data to PLY file.

        Args:
            mesh_data: MeshData object containing vertices, faces, and optional colors
            output_path: Path where to save the PLY file
        """
        self._validate_mesh_data(mesh_data)

        logger.debug(f"Writing mesh to PLY ({self.format_type}): {output_path}")
        logger.debug(
            f"Mesh stats: {mesh_data.num_vertices} vertices, {mesh_data.num_faces} faces, colors: {mesh_data.has_colors}"
        )

        if self.format_type == "binary":
            self._write_binary_mesh(mesh_data, output_path)
        else:
            self._write_text_mesh(mesh_data, output_path)

    def write_point_cloud(self, point_data: PointCloudData, output_path: Path) -> None:
        """Write point cloud data to PLY file.

        Args:
            point_data: PointCloudData object containing vertices and optional colors
            output_path: Path where to save the PLY file
        """
        # Convert to MeshData for unified processing
        mesh_data = point_data.to_mesh_data()
        self.write_mesh(mesh_data, output_path)

    def _validate_mesh_data(self, mesh_data: MeshData) -> None:
        """Validate mesh data before writing."""
        if mesh_data.vertices is None or len(mesh_data.vertices) == 0:
            raise ValueError("Mesh data must contain at least one vertex")

        if mesh_data.vertices.shape[1] != 3:
            raise ValueError(
                f"Vertices must have 3 coordinates (XYZ), got shape {mesh_data.vertices.shape}"
            )

        # Validate faces if present and fix if needed
        if mesh_data.faces is not None and len(mesh_data.faces) > 0:
            if mesh_data.faces.shape[1] != 3:
                logger.warning(
                    f"Faces must be triangles (3 indices), got shape {mesh_data.faces.shape}. Treating as point cloud."
                )
                mesh_data.faces = None
            else:
                # Check face indices are valid
                max_vertex_idx = len(mesh_data.vertices) - 1
                if (
                    np.max(mesh_data.faces) > max_vertex_idx
                    or np.min(mesh_data.faces) < 0
                ):
                    logger.warning(
                        f"Invalid face indices detected (range: {np.min(mesh_data.faces)} to {np.max(mesh_data.faces)}, "
                        f"valid range: 0 to {max_vertex_idx}). Treating as point cloud instead of mesh."
                    )
                    mesh_data.faces = None

        # Validate colors if present
        if mesh_data.colors is not None and len(mesh_data.colors) > 0:
            if mesh_data.colors.shape[0] != mesh_data.vertices.shape[0]:
                raise ValueError(
                    f"Color count ({mesh_data.colors.shape[0]}) must match vertex count ({mesh_data.vertices.shape[0]})"
                )

            if mesh_data.colors.shape[1] != 3:
                raise ValueError(
                    f"Colors must have 3 components (RGB), got shape {mesh_data.colors.shape}"
                )

    def _write_binary_mesh(self, mesh_data: MeshData, output_path: Path) -> None:
        """Write mesh data in binary PLY format."""
        with output_path.open("wb") as f:
            # Write header
            self._write_binary_header(f, mesh_data)

            # Write vertex data
            self._write_binary_vertices(f, mesh_data)

            # Write face data if present
            if not mesh_data.is_point_cloud:
                self._write_binary_faces(f, mesh_data)

    def _write_text_mesh(self, mesh_data: MeshData, output_path: Path) -> None:
        """Write mesh data in text PLY format."""
        with output_path.open("w", encoding="utf-8") as f:
            # Write header
            self._write_text_header(f, mesh_data)

            # Write vertex data
            self._write_text_vertices(f, mesh_data)

            # Write face data if present
            if not mesh_data.is_point_cloud:
                self._write_text_faces(f, mesh_data)

    def _write_binary_header(self, f: BinaryIO, mesh_data: MeshData) -> None:
        """Write binary PLY header."""
        header_lines = self._get_header_lines(mesh_data, "binary_little_endian")
        for line in header_lines:
            f.write(line.encode("ascii"))

    def _write_text_header(self, f: TextIO, mesh_data: MeshData) -> None:
        """Write text PLY header."""
        header_lines = self._get_header_lines(mesh_data, "ascii")
        for line in header_lines:
            f.write(line)

    def _get_header_lines(self, mesh_data: MeshData, format_str: str) -> list[str]:
        """Generate PLY header lines."""
        lines = []
        lines.append("ply\n")
        lines.append(f"format {format_str} 1.0\n")
        lines.append("comment Generated by tboardfs\n")
        lines.append(f"element vertex {mesh_data.num_vertices}\n")
        lines.append("property float x\n")
        lines.append("property float y\n")
        lines.append("property float z\n")

        # Add color properties if colors are present
        if mesh_data.has_colors:
            lines.append("property uchar red\n")
            lines.append("property uchar green\n")
            lines.append("property uchar blue\n")

        # Add face element if this is a mesh (not just point cloud)
        if not mesh_data.is_point_cloud:
            lines.append(f"element face {mesh_data.num_faces}\n")
            lines.append("property list uchar int vertex_indices\n")

        lines.append("end_header\n")
        return lines

    def _write_binary_vertices(self, f: BinaryIO, mesh_data: MeshData) -> None:
        """Write vertices in binary format."""
        for i in range(mesh_data.num_vertices):
            # Write XYZ coordinates
            x, y, z = mesh_data.vertices[i]
            f.write(struct.pack("<fff", float(x), float(y), float(z)))

            # Write RGB colors if present
            if mesh_data.has_colors and mesh_data.colors is not None:
                r, g, b = mesh_data.colors[i]
                # Convert from 0-1 range to 0-255 range
                r_byte = min(255, max(0, int(r * 255)))
                g_byte = min(255, max(0, int(g * 255)))
                b_byte = min(255, max(0, int(b * 255)))
                f.write(struct.pack("<BBB", r_byte, g_byte, b_byte))

    def _write_text_vertices(self, f: TextIO, mesh_data: MeshData) -> None:
        """Write vertices in text format."""
        for i in range(mesh_data.num_vertices):
            # Write XYZ coordinates
            x, y, z = mesh_data.vertices[i]
            line = f"{float(x)} {float(y)} {float(z)}"

            # Write RGB colors if present
            if mesh_data.has_colors and mesh_data.colors is not None:
                r, g, b = mesh_data.colors[i]
                # Convert from 0-1 range to 0-255 range
                r_byte = min(255, max(0, int(r * 255)))
                g_byte = min(255, max(0, int(g * 255)))
                b_byte = min(255, max(0, int(b * 255)))
                line += f" {r_byte} {g_byte} {b_byte}"

            f.write(line + "\n")

    def _write_binary_faces(self, f: BinaryIO, mesh_data: MeshData) -> None:
        """Write faces in binary format."""
        for i in range(mesh_data.num_faces):
            # Write number of vertices per face (always 3 for triangles)
            f.write(struct.pack("<B", 3))

            # Write vertex indices
            if mesh_data.faces is not None:
                v1, v2, v3 = mesh_data.faces[i]
                f.write(struct.pack("<III", int(v1), int(v2), int(v3)))

    def _write_text_faces(self, f: TextIO, mesh_data: MeshData) -> None:
        """Write faces in text format."""
        for i in range(mesh_data.num_faces):
            # Write number of vertices per face (always 3 for triangles)
            if mesh_data.faces is not None:
                v1, v2, v3 = mesh_data.faces[i]
                f.write(f"3 {int(v1)} {int(v2)} {int(v3)}\n")


def write_mesh_as_ply(
    mesh_data: MeshData, output_path: Path, format_type: str = "binary"
) -> None:
    """Convenience function to write mesh data as PLY file.

    Args:
        mesh_data: MeshData object to write
        output_path: Path where to save the PLY file
        format_type: "binary" or "text" format
    """
    writer = PLYWriter(format_type)
    writer.write_mesh(mesh_data, output_path)


def write_point_cloud_as_ply(
    point_data: PointCloudData, output_path: Path, format_type: str = "binary"
) -> None:
    """Convenience function to write point cloud data as PLY file.

    Args:
        point_data: PointCloudData object to write
        output_path: Path where to save the PLY file
        format_type: "binary" or "text" format
    """
    writer = PLYWriter(format_type)
    writer.write_point_cloud(point_data, output_path)
