"""Write a TensorBoard mesh or point cloud as a binary PLY file."""

import numpy as np

_COLORS = ("red", "green", "blue")


def ply_bytes(
    vertices: np.ndarray, faces: np.ndarray | None, colors: np.ndarray | None
) -> bytes:
    """Return all batch elements as one little-endian binary PLY mesh.

    Float32 and smaller coordinates are PLY floats; other numbers are
    doubles, so the coordinates read back unchanged. The faces of a later
    batch element are offset by the vertices before it, and a comment keeps
    the vertex range of each element of a larger batch.

    Args:
        vertices: [B, N, 3] coordinates.
        faces: [B', F, 3] vertex indices of the first B' batch elements, or
            None for a point cloud.
        colors: [B, N, 3] whole numbers from 0 to 255, or None.
    """
    batch, count, _ = vertices.shape
    header = [
        "ply",
        "format binary_little_endian 1.0",
        "comment TensorBoard mesh export",
    ]
    if batch > 1:
        header += [
            f"comment mesh_{i} vertices {i * count} {count}" for i in range(batch)
        ]
    vertex_header, vertex = _PlyElements.vertices(vertices, colors)
    header += vertex_header
    body = [vertex]
    if faces is not None:
        face_header, face = _PlyElements.faces(faces[:batch], count)
        header += face_header
        body.append(face)
    header.append("end_header\n")
    return "\n".join(header).encode() + b"".join(body)


class _PlyElements:
    """Build the header lines and binary records of one PLY element."""

    @staticmethod
    def vertices(
        vertices: np.ndarray, colors: np.ndarray | None
    ) -> tuple[list[str], bytes]:
        """Return the vertex element of all batch elements."""
        small = vertices.dtype.kind == "f" and vertices.dtype.itemsize <= 4
        coordinate, name = ("<f4", "float") if small else ("<f8", "double")
        columns = {axis: (coordinate, vertices[..., i]) for i, axis in enumerate("xyz")}
        if colors is not None:
            columns |= {c: ("u1", colors[..., i]) for i, c in enumerate(_COLORS)}
        records = np.empty(
            vertices.shape[0] * vertices.shape[1],
            dtype=[(key, dtype) for key, (dtype, _) in columns.items()],
        )
        for key, (_, values) in columns.items():
            records[key] = values.reshape(-1)
        header = [f"element vertex {len(records)}"]
        header += [f"property {name} {axis}" for axis in "xyz"]
        header += [f"property uchar {c}" for c in _COLORS if c in columns]
        return header, records.tobytes()

    @staticmethod
    def faces(faces: np.ndarray, count: int) -> tuple[list[str], bytes]:
        """Return the triangles of each batch element, offset by earlier vertices."""
        offsets = np.arange(len(faces))[:, None, None] * count
        indices = (faces + offsets).reshape(-1, 3)
        records = np.empty(len(indices), dtype=[("count", "u1"), ("indices", "<i4", 3)])
        records["count"] = 3
        records["indices"] = indices
        header = [
            f"element face {len(records)}",
            "property list uchar int vertex_indices",
        ]
        return header, records.tobytes()
