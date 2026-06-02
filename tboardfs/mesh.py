from typing import Any

import numpy as np


def box_mesh() -> dict[str, Any]:
    """Return a compact CC0-inspired vertex-color box mesh fixture."""
    vertices = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [1, 2, 3],
            [1, 3, 4],
            [5, 8, 7],
            [5, 7, 6],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 8],
            [3, 8, 4],
            [4, 8, 5],
            [4, 5, 1],
        ],
        dtype=np.int32,
    )
    colors = np.asarray(
        [
            [255, 0, 0],
            [255, 128, 0],
            [255, 255, 0],
            [0, 255, 0],
            [0, 255, 255],
            [0, 0, 255],
            [128, 0, 255],
            [255, 0, 255],
        ],
        dtype=np.uint8,
    )
    obj = "\n".join(
        [
            "# Box Vertex Colors fixture based on Khronos glTF Sample Assets CC0",
            *[f"v {x} {y} {z}" for x, y, z in vertices],
            *[f"f {a} {b} {c}" for a, b, c in faces],
            "",
        ]
    )
    return {
        "vertices": vertices,
        "faces": faces,
        "colors": colors,
        "obj": obj,
        "metadata": {
            "name": "Box Vertex Colors",
            "source": "Khronos glTF Sample Assets",
            "license": "CC0-1.0",
        },
    }
