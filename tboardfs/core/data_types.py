"""Shared data types for TensorBoard parsing."""

from dataclasses import dataclass


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
