"""Mesh data exporter for TensorBoard events."""

from pathlib import Path
from typing import Any
import numpy as np
from tensorboard.compat.proto import event_pb2
from tensorboard.util import tensor_util
from loguru import logger

from .base_exporter import BaseExporter
from ..core.data_types import MeshData


class MeshExporter(BaseExporter):
    """Export mesh data from TensorBoard events."""

    def __init__(self, output_path: Path, digits: int = 6, ply_format: str = "binary"):
        """Initialize mesh exporter.

        Args:
            output_path: Base output directory
            digits: Number of digits for step padding
            ply_format: PLY format ("binary" or "ascii")
        """
        super().__init__(output_path, digits)
        self.ply_format = ply_format
        self.mesh_cache: dict[str, dict[int, dict[str, Any]]] = {}

    def save_data(self, event: event_pb2.Event, value: Any, **kwargs: Any) -> None:
        """Save mesh data from a TensorBoard event.

        Args:
            event: TensorBoard event containing the mesh data
            value: Summary value containing mesh tensor information
        """
        ply_format = kwargs.get("ply_format", self.ply_format)

        tag = value.tag

        # Determine base tag and component type
        base_tag = tag.rstrip("_VERTEX").rstrip("_FACE").rstrip("_COLOR")
        component_type = None

        if tag.endswith("_VERTEX"):
            component_type = "vertices"
        elif tag.endswith("_FACE"):
            component_type = "faces"
        elif tag.endswith("_COLOR"):
            component_type = "colors"
        else:
            # If tag doesn't follow the expected pattern, skip
            logger.debug(f"Skipping mesh tag with unexpected format: {tag}")
            return

        # Initialize cache for this base tag if needed
        if base_tag not in self.mesh_cache:
            self.mesh_cache[base_tag] = {}

        step = event.step
        if step not in self.mesh_cache[base_tag]:
            self.mesh_cache[base_tag][step] = {}

        # Extract and store tensor data
        if value.HasField("tensor"):
            try:
                arr = tensor_util.make_ndarray(value.tensor)

                # Remove batch dimension if present
                if arr.ndim == 3 and arr.shape[0] == 1:
                    arr = arr[0]

                self.mesh_cache[base_tag][step][component_type] = arr
                self.mesh_cache[base_tag][step]["wall_time"] = event.wall_time

                # Check if we have enough components to create a mesh
                components = self.mesh_cache[base_tag][step]
                if "vertices" in components:
                    self._save_mesh_if_complete(base_tag, step, components, ply_format)

            except Exception as e:
                logger.error(f"Error processing mesh tensor for tag {tag}: {e}")

    def _save_mesh_if_complete(
        self, base_tag: str, step: int, components: dict[str, Any], ply_format: str
    ) -> None:
        """Save mesh data if we have complete components."""
        from tboardfs.core.ply_writer import write_mesh_as_ply

        # Create and save mesh data
        vertices = components["vertices"]
        faces = components.get("faces", None)
        colors = components.get("colors", None)
        wall_time = float(components.get("wall_time", 0.0))

        # Validate vertex data
        if vertices.shape[1] == 3:  # XYZ coordinates
            # Pre-validate faces to avoid PLY writer errors
            valid_faces = faces
            if faces is not None and len(faces) > 0:
                max_vertex_idx = len(vertices) - 1
                if np.max(faces) > max_vertex_idx or np.min(faces) < 0:
                    logger.warning(
                        f"Invalid face indices in {base_tag} step {step} "
                        f"(range: {np.min(faces)} to {np.max(faces)}, valid: 0 to {max_vertex_idx}). "
                        f"Saving as point cloud instead."
                    )
                    valid_faces = None

            mesh_data = MeshData(
                step=step,
                vertices=vertices,
                faces=valid_faces,
                colors=colors,
                wall_time=wall_time,
            )

            # Create output directory and file
            safe_tag = self._sanitize_tag(base_tag)
            tag_dir = self.output_path / "meshes" / safe_tag
            self._ensure_directory_exists(tag_dir)

            padded_step = self._format_step(step)
            ply_file = tag_dir / f"{padded_step}.ply"

            # Write PLY file
            try:
                write_mesh_as_ply(mesh_data, ply_file, ply_format)
                mesh_type = "point cloud" if mesh_data.is_point_cloud else "mesh"
                logger.debug(f"Saved {mesh_type} data to {ply_file}")
            except Exception as ply_error:
                logger.error(f"Failed to write PLY file {ply_file}: {ply_error}")

    def finalize(self) -> None:
        """Finalize mesh processing (called after all events processed)."""
        # Any remaining mesh components that might not have been complete
        # could be processed here if needed
        logger.debug(f"Mesh cache contains {len(self.mesh_cache)} base tags")
