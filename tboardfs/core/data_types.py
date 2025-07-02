"""Shared data types for TensorBoard parsing."""

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class ScalarData:
    """Container for scalar data points."""

    step: int
    value: float
    wall_time: float


@dataclass
class ImageData:
    """Container for image data."""

    step: int
    encoded_image_string: bytes
    width: int
    height: int
    wall_time: float


@dataclass
class VideoData:
    """Container for video data (stored as GIF in TensorBoard)."""

    step: int
    encoded_video_string: bytes  # GIF or other video format
    width: int
    height: int
    wall_time: float


@dataclass
class HistogramData:
    """Container for histogram data."""

    step: int
    min: float
    max: float
    num: int
    sum: float
    sum_squares: float
    bucket_limit: list[float]
    bucket: list[int]
    wall_time: float


@dataclass
class AudioData:
    """Container for audio data."""

    step: int
    encoded_audio_string: bytes
    content_type: str
    sample_rate: float
    length_frames: int
    wall_time: float


@dataclass
class TextData:
    """Container for text data."""

    step: int
    text: str
    wall_time: float


@dataclass
class MeshData:
    """Container for 3D mesh data with vertices, faces, and optional colors."""

    step: int
    vertices: np.ndarray  # Shape (N, 3) - XYZ coordinates
    faces: np.ndarray | None = (
        None  # Shape (M, 3) - triangle indices, None for point clouds
    )
    colors: np.ndarray | None = None  # Shape (N, 3) - RGB colors per vertex (0-1 range)
    wall_time: float = 0.0

    @property
    def is_point_cloud(self) -> bool:
        """Check if this is a point cloud (no faces) or a mesh."""
        return self.faces is None or len(self.faces) == 0

    @property
    def has_colors(self) -> bool:
        """Check if color data is available."""
        return self.colors is not None and len(self.colors) > 0

    @property
    def num_vertices(self) -> int:
        """Get number of vertices."""
        return len(self.vertices) if self.vertices is not None else 0

    @property
    def num_faces(self) -> int:
        """Get number of faces."""
        return len(self.faces) if self.faces is not None else 0


@dataclass
class PointCloudData:
    """Container for point cloud data (alias for MeshData without faces)."""

    step: int
    vertices: np.ndarray  # Shape (N, 3) - XYZ coordinates
    colors: np.ndarray | None = None  # Shape (N, 3) - RGB colors per vertex (0-1 range)
    wall_time: float = 0.0

    @property
    def has_colors(self) -> bool:
        """Check if color data is available."""
        return self.colors is not None and len(self.colors) > 0

    @property
    def num_vertices(self) -> int:
        """Get number of vertices."""
        return len(self.vertices) if self.vertices is not None else 0

    def to_mesh_data(self) -> MeshData:
        """Convert to MeshData format for unified processing."""
        return MeshData(
            step=self.step,
            vertices=self.vertices,
            faces=None,
            colors=self.colors,
            wall_time=self.wall_time,
        )


@dataclass
class HyperparameterData:
    """Container for hyperparameter data."""

    step: int
    hparams: dict[str, Any]  # parameter_name -> value
    model_uri: str | None = None
    monitor_url: str | None = None
    group_name: str | None = None
    wall_time: float = 0.0


@dataclass
class PRCurveData:
    """Container for Precision-Recall curve data."""

    step: int
    precision: np.ndarray  # Precision values at different thresholds
    recall: np.ndarray  # Recall values at different thresholds
    thresholds: np.ndarray  # Threshold values
    wall_time: float = 0.0

    @property
    def num_points(self) -> int:
        """Get number of points in the PR curve."""
        return len(self.precision) if self.precision is not None else 0
