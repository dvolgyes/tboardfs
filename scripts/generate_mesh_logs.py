#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "numpy>=2.1,<3.0",
#   "tensorboard>=2.20.0,<3.0",
#   "tensorflow>=2.21.0,<3.0",
# ]
# ///
"""Generate small mesh and point cloud fixture logs under test-logs/meshes.

TensorBoard's mesh summary writes each mesh: with and without faces, with
and without vertex colors, and a batch of two colored meshes.
"""

from pathlib import Path
import shutil

from tensorboard.plugins.mesh import summary_v2 as mesh_summary
import numpy as np
import tensorflow as tf

ROOT = Path("test-logs") / "meshes"
TETRAHEDRON = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
FACES = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)


def main() -> None:
    """Write one run of meshes and point clouds, replacing an earlier one."""
    if ROOT.exists():
        shutil.rmtree(ROOT)
    rng = np.random.default_rng(0)
    points = rng.normal(size=(1, 200, 3)).astype(np.float32)
    point_colors = rng.integers(0, 256, size=(1, 200, 3), dtype=np.uint8)
    corner_colors = np.array(
        [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]], dtype=np.uint8
    )
    meshes = {
        "colored_mesh": (TETRAHEDRON[None], FACES[None], corner_colors[None]),
        "mesh": (TETRAHEDRON[None], FACES[None], None),
        "colored_cloud": (points, None, point_colors),
        "cloud": (points, None, None),
        "batch": (
            np.stack([TETRAHEDRON, TETRAHEDRON + 2]),
            np.stack([FACES, FACES[:, ::-1]]),
            np.stack([corner_colors, corner_colors[::-1]]),
        ),
    }
    writer = tf.summary.create_file_writer(str(ROOT))
    with writer.as_default():
        for name, (vertices, faces, colors) in meshes.items():
            mesh_summary.mesh(
                name, vertices=vertices, faces=faces, colors=colors, step=1
            )
    writer.close()


if __name__ == "__main__":
    main()
