"""Data iterator factory for TensorBoard event processing.

This module provides a unified factory for creating specialized data iterators,
eliminating code duplication across different data type processors while
maintaining type safety and performance.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, TYPE_CHECKING
import numpy as np

from loguru import logger
from tensorboard.util import tensor_util
from tensorboard.compat.proto import event_pb2

from tboardfs.core.data_types import (
    ScalarData,
    ImageData,
    VideoData,
    HistogramData,
    AudioData,
    TextData,
    MeshData,
    HyperparameterData,
    PRCurveData,
)
from tboardfs.core.data_detector import TensorDataDetector
from tboardfs.core.image_processor import ImageProcessor
from tboardfs.core.constants import TensorFlowDTypes, AudioFormats

if TYPE_CHECKING:
    from tboardfs.efficient_parser import EfficientTensorBoardParser

# Import hyperparameter protobuf definitions
try:
    from tensorboard.plugins.hparams import plugin_data_pb2 as hparams_pb2
    from google.protobuf.struct_pb2 import Value as protobuf_Value

    HPARAMS_AVAILABLE = True
except ImportError:
    HPARAMS_AVAILABLE = False
    hparams_pb2 = None
    protobuf_Value = Any  # type: ignore


class BaseDataIterator(ABC):
    """Base class for all data iterators."""

    def __init__(self, parser: "EfficientTensorBoardParser") -> None:
        """Initialize iterator with parser instance.

        Args:
            parser: The TensorBoard parser instance providing data access.
        """
        self.parser = parser

    @abstractmethod
    def iterate(self, tag: str) -> Iterator[Any]:
        """Iterate over data for given tag.

        Args:
            tag: The tag to iterate data for.

        Returns:
            Iterator over data items of specific type.
        """
        pass

    def _iterate_matching_values(
        self, tag: str
    ) -> Iterator[tuple[event_pb2.Event, Any]]:
        """Helper method to iterate over matching tag values.

        Args:
            tag: Tag to match against.

        Yields:
            Tuples of (event, summary_value) for matching tags.
        """
        for event in self.parser._iterate_events():
            if event.HasField("summary"):
                for value in event.summary.value:
                    if value.tag == tag:
                        yield event, value


class ScalarDataIterator(BaseDataIterator):
    """Iterator for scalar data extraction."""

    def iterate(self, tag: str) -> Iterator[ScalarData]:
        """Iterate over scalar data for a given tag."""
        for event, value in self._iterate_matching_values(tag):
            scalar_val = None
            logger.debug(f"Processing scalar tag: {tag}")
            logger.debug(f"  Has simple_value: {value.HasField('simple_value')}")
            logger.debug(f"  Has tensor: {value.HasField('tensor')}")

            if value.HasField("simple_value"):
                scalar_val = value.simple_value
                logger.debug(f"  Extracted from simple_value: {scalar_val}")
            elif value.HasField("tensor"):
                # Attempt to extract scalar from tensor
                try:
                    arr = tensor_util.make_ndarray(value.tensor)
                    logger.debug(f"  Tensor dtype: {value.tensor.dtype}")
                    logger.debug(f"  Tensor shape: {arr.shape}")
                    if arr.size == 1:
                        scalar_val = float(arr.item())
                        logger.debug(f"  Extracted from tensor: {scalar_val}")
                    else:
                        logger.warning(
                            f"Tensor for scalar tag '{tag}' has more than one element (size={arr.size}). Skipping."
                        )
                except Exception as e:
                    logger.warning(
                        f"Could not extract scalar from tensor for tag '{tag}': {e}"
                    )

            if scalar_val is not None:
                yield ScalarData(
                    step=event.step,
                    value=scalar_val,
                    wall_time=event.wall_time,
                )


class ImageDataIterator(BaseDataIterator):
    """Iterator for image data extraction."""

    def iterate(self, tag: str) -> Iterator[ImageData]:
        """Iterate over image data for a given tag."""
        for event, value in self._iterate_matching_values(tag):
            if value.HasField("image"):
                yield ImageData(
                    step=event.step,
                    encoded_image_string=value.image.encoded_image_string,
                    width=value.image.width,
                    height=value.image.height,
                    wall_time=event.wall_time,
                )
            elif value.HasField("tensor"):
                if TensorDataDetector.is_image_tensor(value.tensor, tag):
                    decoded_image = ImageProcessor.decode_image_from_tensor(
                        value.tensor
                    )
                    if decoded_image:
                        yield ImageData(
                            step=event.step,
                            encoded_image_string=decoded_image,
                            width=0,  # Width/height might not be easily available
                            height=0,
                            wall_time=event.wall_time,
                        )
                elif value.tensor.dtype == 7:
                    try:
                        arr = tensor_util.make_ndarray(value.tensor)
                        for item in arr:
                            if isinstance(item, bytes):
                                ext = ImageProcessor.get_image_extension(item, tag)
                                if ext != "bin":
                                    yield ImageData(
                                        step=event.step,
                                        encoded_image_string=item,
                                        width=0,
                                        height=0,
                                        wall_time=event.wall_time,
                                    )
                    except Exception as e:
                        logger.warning(
                            f"Could not decode string tensor for tag '{tag}': {e}"
                        )


class VideoDataIterator(BaseDataIterator):
    """Iterator for video data extraction."""

    def iterate(self, tag: str) -> Iterator[VideoData]:
        """Iterate over video data for a given tag."""
        for event, value in self._iterate_matching_values(tag):
            if value.HasField("image"):
                # Video data is stored as images (GIF) in TensorBoard
                if TensorDataDetector.is_video_data(
                    value.image.encoded_image_string, tag
                ):
                    yield VideoData(
                        step=event.step,
                        encoded_video_string=value.image.encoded_image_string,
                        width=value.image.width,
                        height=value.image.height,
                        wall_time=event.wall_time,
                    )


class HistogramDataIterator(BaseDataIterator):
    """Iterator for histogram data extraction."""

    def iterate(self, tag: str) -> Iterator[HistogramData]:
        """Iterate over histogram data for a given tag."""
        for event, value in self._iterate_matching_values(tag):
            if value.HasField("histo"):
                hist = value.histo
                yield HistogramData(
                    step=event.step,
                    min=hist.min,
                    max=hist.max,
                    num=hist.num,
                    sum=hist.sum,
                    sum_squares=hist.sum_squares,
                    bucket_limit=list(hist.bucket_limit),
                    bucket=list(hist.bucket),
                    wall_time=event.wall_time,
                )


class AudioDataIterator(BaseDataIterator):
    """Iterator for audio data extraction."""

    def iterate(self, tag: str) -> Iterator[AudioData]:
        """Iterate over audio data for a given tag."""
        for event, value in self._iterate_matching_values(tag):
            # Check for direct audio field first
            if value.HasField("audio"):
                audio = value.audio
                yield AudioData(
                    step=event.step,
                    encoded_audio_string=audio.encoded_audio_string,
                    content_type=audio.content_type,
                    sample_rate=audio.sample_rate,
                    length_frames=audio.length_frames,
                    wall_time=event.wall_time,
                )
            # Check for audio data stored as tensor with plugin metadata
            elif (
                value.HasField("metadata")
                and value.metadata.plugin_data.plugin_name == "audio"
            ):
                # Audio data may be stored in tensor format
                if value.HasField("tensor") and value.tensor.dtype == 7:
                    try:
                        # Extract audio data from tensor (dtype 7 = DT_STRING)
                        arr = tensor_util.make_ndarray(value.tensor)
                        if arr.size > 0:
                            # For string tensors containing audio data
                            audio_data = arr.item() if arr.ndim == 0 else arr[0]
                            if isinstance(audio_data, bytes):
                                audio_bytes = audio_data
                            else:
                                # If it's a string, encode it to bytes
                                audio_bytes = str(audio_data).encode("utf-8")

                            yield AudioData(
                                step=event.step,
                                encoded_audio_string=audio_bytes,
                                content_type="audio/wav",  # Default assumption
                                sample_rate=AudioFormats.DEFAULT_SAMPLE_RATE,
                                length_frames=len(audio_bytes),
                                wall_time=event.wall_time,
                            )
                    except Exception as e:
                        logger.debug(
                            f"Failed to extract audio from tensor for tag {tag}: {e}"
                        )


class TextDataIterator(BaseDataIterator):
    """Iterator for text data extraction."""

    def iterate(self, tag: str) -> Iterator[TextData]:
        """Iterate over text data for a given tag."""
        for event, value in self._iterate_matching_values(tag):
            if (
                value.HasField("tensor")
                and value.tensor.dtype == TensorFlowDTypes.DT_STRING
            ):
                # Decode text from tensor
                text_value = tensor_util.make_ndarray(value.tensor)
                if text_value.size > 0:
                    text = (
                        text_value.item()
                        if text_value.ndim == 0
                        else str(text_value[0])
                    )
                    if isinstance(text, bytes):
                        text = text.decode("utf-8", errors="replace")
                    yield TextData(
                        step=event.step,
                        text=text,
                        wall_time=event.wall_time,
                    )


class MeshDataIterator(BaseDataIterator):
    """Iterator for mesh data extraction."""

    def iterate(self, tag: str) -> Iterator[MeshData]:
        """Iterate over mesh data for a given tag.

        TensorBoard mesh plugin stores 3D data as tensors with plugin_name="mesh".
        Mesh data consists of VERTEX, FACE, and COLOR components stored as separate tags.
        """
        # Collect mesh components by finding related tags
        base_tag = tag.rstrip("_VERTEX").rstrip("_FACE").rstrip("_COLOR")

        # Storage for mesh components by step
        mesh_components: dict[int, dict[str, np.ndarray]] = {}

        # Iterate through events to collect all mesh components
        for event in self.parser._iterate_events():
            if event.HasField("summary"):
                for value in event.summary.value:
                    value_tag = value.tag

                    # Check if this is a mesh component for our base tag
                    if value_tag.startswith(base_tag):
                        if value.HasField("tensor"):
                            try:
                                # Extract tensor data
                                arr = tensor_util.make_ndarray(value.tensor)

                                # Determine component type
                                component_type = None
                                if value_tag.endswith("_VERTEX"):
                                    component_type = "vertices"
                                elif value_tag.endswith("_FACE"):
                                    component_type = "faces"
                                elif value_tag.endswith("_COLOR"):
                                    component_type = "colors"

                                if component_type:
                                    step = event.step
                                    if step not in mesh_components:
                                        mesh_components[step] = {}

                                    # Reshape array for consistency (remove batch dimension if present)
                                    if arr.ndim == 3 and arr.shape[0] == 1:
                                        arr = arr[0]  # Remove batch dimension

                                    mesh_components[step][component_type] = arr
                                    mesh_components[step]["wall_time"] = event.wall_time

                            except Exception as e:
                                logger.debug(
                                    f"Error processing mesh tensor for tag {value_tag}: {e}"
                                )
                                continue

        # Yield complete mesh data for each step
        for step in sorted(mesh_components.keys()):
            components = mesh_components[step]

            # Verify we have at least vertices
            if "vertices" in components:
                vertices = components["vertices"]
                faces = components.get("faces", None)
                colors = components.get("colors", None)
                wall_time = float(components.get("wall_time", 0.0))

                # Validate vertex data
                if vertices.shape[1] == 3:  # XYZ coordinates
                    yield MeshData(
                        step=step,
                        vertices=vertices,
                        faces=faces,
                        colors=colors,
                        wall_time=wall_time,
                    )


class HyperparameterDataIterator(BaseDataIterator):
    """Iterator for hyperparameter data extraction."""

    def iterate(self, tag: str) -> Iterator[HyperparameterData]:
        """Iterate over hyperparameter data for a given tag.

        TensorBoard hyperparameters are stored with plugin_name="hparams".
        The actual data is serialized in the metadata content field.
        """
        if not HPARAMS_AVAILABLE:
            logger.warning(
                "TensorBoard hparams plugin not available. Cannot parse hyperparameters."
            )
            return

        for event, value in self._iterate_matching_values(tag):
            if value.HasField("metadata"):
                plugin_name = value.metadata.plugin_data.plugin_name
                if plugin_name == "hparams":
                    try:
                        # Parse the hyperparameter data from metadata
                        plugin_data = hparams_pb2.HParamsPluginData.FromString(
                            value.metadata.plugin_data.content
                        )

                        if plugin_data.HasField("session_start_info"):
                            session_info = plugin_data.session_start_info

                            # Extract hyperparameters from protobuf Value objects
                            hparams = {}
                            for param_name, param_value in session_info.hparams.items():
                                hparams[param_name] = self._extract_protobuf_value(
                                    param_value
                                )

                            yield HyperparameterData(
                                step=event.step,
                                hparams=hparams,
                                model_uri=session_info.model_uri
                                if session_info.model_uri
                                else None,
                                monitor_url=session_info.monitor_url
                                if session_info.monitor_url
                                else None,
                                group_name=session_info.group_name
                                if session_info.group_name
                                else None,
                                wall_time=event.wall_time,
                            )

                    except Exception as e:
                        logger.warning(
                            f"Failed to parse hyperparameter data for tag '{tag}': {e}"
                        )
                        continue

    def _extract_protobuf_value(self, value: protobuf_Value) -> Any:
        """Extract value from google.protobuf.Value object."""
        if value.HasField("bool_value"):
            return value.bool_value
        elif value.HasField("number_value"):
            return value.number_value
        elif value.HasField("string_value"):
            return value.string_value
        elif value.HasField("list_value"):
            return [self._extract_protobuf_value(v) for v in value.list_value.values]
        elif value.HasField("struct_value"):
            return {
                k: self._extract_protobuf_value(v)
                for k, v in value.struct_value.fields.items()
            }
        else:
            return None


class PRCurveDataIterator(BaseDataIterator):
    """Iterator for PR curve data extraction."""

    def iterate(self, tag: str) -> Iterator[PRCurveData]:
        """Iterate over PR curve data for a given tag."""
        for event, value in self._iterate_matching_values(tag):
            if value.HasField("tensor"):
                if TensorDataDetector.is_pr_curve_tensor(value.tensor, tag):
                    try:
                        # Extract tensor data
                        arr = tensor_util.make_ndarray(value.tensor)
                        logger.debug(
                            f"Processing PR curve data for tag '{tag}', shape: {arr.shape}"
                        )

                        # Based on TensorBoard format, PR curves are [6, N] where:
                        # Row 4 = precision values, Row 5 = recall values
                        # Generate thresholds as linspace from 0 to 1
                        num_thresholds = arr.shape[1]
                        thresholds = np.linspace(0.0, 1.0, num_thresholds)
                        precision = arr[4, :]  # Row 4 contains precision
                        recall = arr[5, :]  # Row 5 contains recall

                        yield PRCurveData(
                            step=event.step,
                            precision=precision,
                            recall=recall,
                            thresholds=thresholds,
                            wall_time=event.wall_time,
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to extract PR curve data for tag '{tag}': {e}"
                        )


class DataIteratorFactory:
    """Factory for creating specialized data iterators."""

    def __init__(self, parser: "EfficientTensorBoardParser") -> None:
        """Initialize factory with parser instance.

        Args:
            parser: The TensorBoard parser instance.
        """
        self.parser = parser
        self._iterators = {
            "scalar": ScalarDataIterator(parser),
            "image": ImageDataIterator(parser),
            "video": VideoDataIterator(parser),
            "histogram": HistogramDataIterator(parser),
            "audio": AudioDataIterator(parser),
            "text": TextDataIterator(parser),
            "mesh": MeshDataIterator(parser),
            "hyperparameter": HyperparameterDataIterator(parser),
            "pr_curve": PRCurveDataIterator(parser),
        }

    def get_iterator(self, data_type: str) -> BaseDataIterator:
        """Get iterator for specific data type.

        Args:
            data_type: Type of data iterator to create.

        Returns:
            Specialized data iterator instance.

        Raises:
            ValueError: If data_type is not supported.
        """
        if data_type not in self._iterators:
            raise ValueError(
                f"Unsupported data type: {data_type}. "
                f"Supported types: {list(self._iterators.keys())}"
            )
        return self._iterators[data_type]

    def iterate_scalar_data(self, tag: str) -> Iterator[ScalarData]:
        """Convenience method for scalar data iteration."""
        return self._iterators["scalar"].iterate(tag)

    def iterate_image_data(self, tag: str) -> Iterator[ImageData]:
        """Convenience method for image data iteration."""
        return self._iterators["image"].iterate(tag)

    def iterate_video_data(self, tag: str) -> Iterator[VideoData]:
        """Convenience method for video data iteration."""
        return self._iterators["video"].iterate(tag)

    def iterate_histogram_data(self, tag: str) -> Iterator[HistogramData]:
        """Convenience method for histogram data iteration."""
        return self._iterators["histogram"].iterate(tag)

    def iterate_audio_data(self, tag: str) -> Iterator[AudioData]:
        """Convenience method for audio data iteration."""
        return self._iterators["audio"].iterate(tag)

    def iterate_text_data(self, tag: str) -> Iterator[TextData]:
        """Convenience method for text data iteration."""
        return self._iterators["text"].iterate(tag)

    def iterate_mesh_data(self, tag: str) -> Iterator[MeshData]:
        """Convenience method for mesh data iteration."""
        return self._iterators["mesh"].iterate(tag)

    def iterate_hyperparameter_data(self, tag: str) -> Iterator[HyperparameterData]:
        """Convenience method for hyperparameter data iteration."""
        return self._iterators["hyperparameter"].iterate(tag)

    def iterate_pr_curve_data(self, tag: str) -> Iterator[PRCurveData]:
        """Convenience method for PR curve data iteration."""
        return self._iterators["pr_curve"].iterate(tag)
