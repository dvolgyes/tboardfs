"""Constants and configuration values for tboardfs.

This module centralizes all magic numbers, default values, and constant definitions
used throughout the codebase to improve maintainability and reduce duplication.
"""

from enum import IntEnum
from typing import ClassVar


class TensorFlowDTypes(IntEnum):
    """TensorFlow data type constants."""

    DT_INVALID = 0
    DT_FLOAT = 1
    DT_DOUBLE = 2
    DT_INT32 = 3
    DT_UINT8 = 4
    DT_INT16 = 5
    DT_INT8 = 6
    DT_STRING = 7
    DT_COMPLEX64 = 8
    DT_INT64 = 9
    DT_BOOL = 10
    DT_QINT8 = 11
    DT_QUINT8 = 12
    DT_QINT32 = 13
    DT_BFLOAT16 = 14
    DT_QINT16 = 15
    DT_QUINT16 = 16
    DT_UINT16 = 17
    DT_COMPLEX128 = 18
    DT_HALF = 19
    DT_RESOURCE = 20
    DT_VARIANT = 21
    DT_UINT32 = 22
    DT_UINT64 = 23


class ImageFormats:
    """Image format constants and validation."""

    # Supported channel counts for image detection
    VALID_CHANNELS: ClassVar[frozenset[int]] = frozenset(
        [1, 3, 4]
    )  # Grayscale, RGB, RGBA

    # Common image formats
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    GIF = "gif"

    # Default image settings
    DEFAULT_FORMAT = PNG
    DEFAULT_QUALITY = 90

    # Image dimension constraints
    MIN_IMAGE_SIZE = 1
    MAX_REASONABLE_IMAGE_SIZE = 10000  # pixels


class AudioFormats:
    """Audio format constants."""

    WAV = "wav"
    MP3 = "mp3"

    DEFAULT_FORMAT = WAV
    DEFAULT_SAMPLE_RATE = 22050.0

    # Audio constraints
    MIN_SAMPLE_RATE = 8000
    MAX_SAMPLE_RATE = 96000


class PLYFormats:
    """PLY (Polygon File Format) constants."""

    BINARY = "binary"
    TEXT = "text"
    ASCII = "ascii"  # Alias for text

    DEFAULT_FORMAT = BINARY


class TensorBoardConstants:
    """TensorBoard-specific constants."""

    # Event file patterns
    EVENT_FILE_PREFIX = "tfevents"

    # Step formatting
    DEFAULT_STEP_DIGITS = 6

    # PR Curve specific constants
    PR_CURVE_TENSOR_SHAPE = (
        6,
    )  # (threshold, precision, recall, tp, fp, tn, fn counts)
    PR_CURVE_REQUIRED_COMPONENTS = 6

    # Mesh component requirements
    MESH_MIN_VERTICES = 3
    MESH_VERTEX_DIMS = 3  # x, y, z coordinates

    # Histogram bucket constraints
    MIN_HISTOGRAM_BUCKETS = 1
    MAX_HISTOGRAM_BUCKETS = 10000

    # Text encoding
    DEFAULT_TEXT_ENCODING = "utf-8"


class FileSystemConstants:
    """File system and path constants."""

    # Path sanitization
    UNSAFE_PATH_CHARS = frozenset(["/", "\\", ":", "*", "?", '"', "<", ">", "|"])
    PATH_REPLACEMENT_CHAR = "_"

    # Directory creation
    DEFAULT_DIR_PERMISSIONS = 0o755

    # File extensions
    YAML_EXT = "yaml"
    CSV_EXT = "csv"
    NPZ_EXT = "npz"
    TXT_EXT = "txt"
    PLY_EXT = "ply"


class DataTypeShapes:
    """Shape validation constants for different data types."""

    # Image tensor shapes (H, W, C) or (N, H, W, C)
    IMAGE_MIN_DIMS = 2
    IMAGE_MAX_DIMS = 4

    # Video tensor shapes (N, H, W, C) or (T, H, W, C)
    VIDEO_MIN_DIMS = 4
    VIDEO_MAX_DIMS = 4

    # Audio tensor shapes
    AUDIO_MIN_DIMS = 1
    AUDIO_MAX_DIMS = 2

    # Scalar - single values
    SCALAR_SHAPE = ()

    # Histogram - variable bucket counts
    HISTOGRAM_MIN_DIMS = 1

    # Mesh components
    MESH_VERTEX_SHAPE_2D = 2  # (N, 2) for 2D points
    MESH_VERTEX_SHAPE_3D = 3  # (N, 3) for 3D points
    MESH_FACE_SHAPE = 3  # (N, 3) for triangular faces
    MESH_COLOR_DIMS = [3, 4]  # RGB or RGBA


class ExportSettings:
    """Default export configuration settings."""

    # Progress display
    SHOW_PROGRESS_DEFAULT = False

    # Output organization
    CREATE_STEP_DIRECTORIES = True
    AGGREGATE_SCALARS = True
    AGGREGATE_HISTOGRAMS = True

    # File naming
    USE_ZERO_PADDING = True
    SANITIZE_TAGS = True

    # Memory management
    MAX_EVENTS_IN_MEMORY = 10000
    CHUNK_SIZE_BYTES = 1024 * 1024  # 1MB chunks

    # Export formats
    EXPORT_FORMATS = {
        "scalar": ["txt", "csv"],
        "histogram": ["txt", "csv", "npz"],
        "image": ["png", "jpg"],
        "audio": ["wav", "mp3"],
        "video": ["gif", "mp4"],
        "text": ["txt"],
        "mesh": ["ply"],
        "hyperparameter": ["yaml", "json"],
        "pr_curve": ["csv", "npz"],
    }


class ValidationLimits:
    """Validation limits for data processing."""

    # String lengths
    MAX_TAG_LENGTH = 1024
    MAX_FILENAME_LENGTH = 255

    # Numeric limits
    MAX_STEP_VALUE = 2**63 - 1  # Maximum int64
    MIN_STEP_VALUE = 0

    # Memory limits
    MAX_TENSOR_SIZE_BYTES = 100 * 1024 * 1024  # 100MB
    MAX_STRING_LENGTH = 10 * 1024 * 1024  # 10MB

    # Collection limits
    MAX_TAGS_PER_TYPE = 10000
    MAX_STEPS_PER_TAG = 100000


# Convenience constants for backward compatibility
DT_STRING = TensorFlowDTypes.DT_STRING
DEFAULT_DIGITS = TensorBoardConstants.DEFAULT_STEP_DIGITS
DEFAULT_IMAGE_QUALITY = ImageFormats.DEFAULT_QUALITY
DEFAULT_AUDIO_FORMAT = AudioFormats.DEFAULT_FORMAT
DEFAULT_PLY_FORMAT = PLYFormats.DEFAULT_FORMAT
