"""Efficient TensorBoard event file parser using iterators.

This parser uses EventFileLoader directly to iterate over events without
loading everything into memory, making it much more efficient for large files.
"""

from pathlib import Path
from collections.abc import Iterator
from typing import Any
import numpy as np

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.util import tensor_util
from tensorboard.compat.proto import event_pb2
from tqdm import tqdm
import sys
from loguru import logger
import magic
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
from tboardfs.core.export_pipeline import ExportPipeline, ExportConfig
from tboardfs.core.data_source import DataSource, FileDataSource

# Import pydub for audio format conversion
try:
    from pydub import AudioSegment

    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    AudioSegment = None
    logger.warning("pydub not available. Audio format conversion disabled.")

# Import hyperparameter protobuf definitions
try:
    from tensorboard.plugins.hparams import plugin_data_pb2 as hparams_pb2
    from google.protobuf.struct_pb2 import Value as protobuf_Value

    HPARAMS_AVAILABLE = True
except ImportError:
    HPARAMS_AVAILABLE = False
    hparams_pb2 = None
    protobuf_Value = Any  # type: ignore
    logger.warning(
        "TensorBoard hparams plugin not available. Hyperparameter parsing disabled."
    )


class EfficientTensorBoardParser:
    """Efficient parser for TensorBoard event files using iterators.

    This parser supports TensorFlow v2 event file format where all data types
    (scalars, images, histograms, audio) are stored as tensors. Text data is
    identified by checking tensor dtype == 7 (DT_STRING).

    This implementation uses EventFileLoader for efficient streaming instead of
    EventAccumulator which loads everything into memory.
    """

    def __init__(
        self,
        event_file_path: str | DataSource | None = None,
        show_progress: bool = False,
        data_source: DataSource | None = None,
    ):
        """Initialize parser with event file path or DataSource.

        Args:
            event_file_path: Path to event file (deprecated, use data_source instead)
            show_progress: Whether to show progress bars
            data_source: DataSource instance for flexible data access
        """
        # Handle backward compatibility
        if data_source is not None:
            self.data_source = data_source
            # Keep event_file_path for backward compatibility
            self.event_file_path = data_source.get_identifier()
        elif event_file_path is not None:
            if isinstance(event_file_path, DataSource):
                # Handle case where DataSource is passed as first argument
                self.data_source = event_file_path
                self.event_file_path = event_file_path.get_identifier()
            else:
                # Traditional file path
                self.data_source = FileDataSource(event_file_path)
                self.event_file_path = str(event_file_path)
        else:
            raise ValueError("Either event_file_path or data_source must be provided")

        self.show_progress = show_progress

        # Cache for tags to avoid re-scanning
        self._tags_cache: dict[str, list[str]] | None = None
        self._detailed_tags: dict[str, list[str]] | None = None
        self._event_count: int | None = None

        # Log data source information
        source_info = self.data_source.get_source_info()
        logger.debug(
            f"Initializing EfficientTensorBoardParser for {source_info.get('type', 'unknown')} source: "
            f"{self.data_source.get_identifier()}"
        )

        size_mb = source_info.get("size_mb", 0)
        if size_mb > 100:
            logger.info(
                f"Large data source detected ({size_mb:.1f} MB). Processing will be memory efficient."
            )

        # Validate data source availability
        if not self.data_source.is_available():
            raise FileNotFoundError(
                f"Data source not available: {self.data_source.get_identifier()}"
            )

    def _create_loader(self) -> EventFileLoader:
        """Create a new EventFileLoader instance (deprecated, use data_source)."""
        # Fallback for backward compatibility - only works with FileDataSource
        if isinstance(self.data_source, FileDataSource):
            return EventFileLoader(str(self.data_source.file_path))
        else:
            raise RuntimeError(
                "_create_loader() only supports FileDataSource. Use _iterate_events() instead."
            )

    def _iterate_events(self) -> Iterator[event_pb2.Event]:
        """Iterate over all events using the data source."""
        if not self.data_source.is_available():
            raise FileNotFoundError(
                f"Data source not available: {self.data_source.get_identifier()}"
            )

        yield from self.data_source.get_event_iterator()

    def _scan_tags(self) -> dict[str, list[str]]:
        """Scan the file once to build a directory of all tags by type."""
        if self._tags_cache is not None:
            return self._tags_cache

        logger.debug("Scanning file for tags...")
        tags: dict[str, set[str]] = {
            "scalars": set(),
            "images": set(),
            "videos": set(),
            "histograms": set(),
            "tensors": set(),
            "audio": set(),
            "text": set(),
            "meshes": set(),
            "hyperparameters": set(),
            "pr_curves": set(),
        }

        event_count = 0
        iterator: Iterator[event_pb2.Event] = self._iterate_events()

        if self.show_progress:
            iterator = tqdm(iterator, desc="Scanning for tags", unit=" events")  # type: ignore[assignment]

        for event in iterator:
            event_count += 1

            # Check for summary data
            if event.HasField("summary"):
                for value in event.summary.value:
                    tag = value.tag

                    # Check metadata for plugin type
                    if value.HasField("metadata"):
                        plugin_name = value.metadata.plugin_data.plugin_name

                        if plugin_name == "scalars":
                            tags["scalars"].add(tag)
                        elif plugin_name == "images":
                            tags["images"].add(tag)
                        elif plugin_name == "histograms":
                            tags["histograms"].add(tag)
                        elif plugin_name == "audio":
                            tags["audio"].add(tag)
                        elif plugin_name == "text":
                            tags["text"].add(tag)
                        elif plugin_name == "mesh":
                            tags["meshes"].add(tag)
                        elif plugin_name == "hparams":
                            tags["hyperparameters"].add(tag)
                        elif plugin_name == "pr_curves":
                            tags["pr_curves"].add(tag)
                        else:
                            # Default to tensors
                            tags["tensors"].add(tag)
                    else:
                        # Legacy format detection based on field presence
                        if value.HasField("simple_value"):
                            tags["scalars"].add(tag)
                        elif value.HasField("image"):
                            # Check if this image is actually a video (GIF)
                            if TensorDataDetector.is_video_data(
                                value.image.encoded_image_string, tag
                            ):
                                tags["videos"].add(tag)
                            else:
                                tags["images"].add(tag)
                        elif value.HasField("histo"):
                            tags["histograms"].add(tag)
                        elif value.HasField("audio"):
                            tags["audio"].add(tag)
                        elif value.HasField("tensor"):
                            # Attempt to make ndarray to check size for scalar
                            try:
                                arr = tensor_util.make_ndarray(value.tensor)
                                if arr.size == 1:
                                    tags["scalars"].add(tag)
                                elif (
                                    value.tensor.dtype == 7
                                ):  # Check if it's text (DT_STRING = 7)
                                    tags["text"].add(tag)
                                elif TensorDataDetector.is_pr_curve_tensor(
                                    value.tensor, tag
                                ):  # Check if it's a PR curve
                                    tags["pr_curves"].add(tag)
                                elif TensorDataDetector.is_image_tensor(
                                    value.tensor, tag
                                ):  # Check if it's an image
                                    tags["images"].add(tag)
                                else:
                                    tags["tensors"].add(tag)
                            except Exception as e:
                                logger.debug(
                                    f"Could not make ndarray for tensor tag {tag}: {e}. Treating as generic tensor."
                                )
                                tags["tensors"].add(tag)

        # Store the specific categorization for internal use (correct categorization)
        self._detailed_tags = {k: sorted(v) for k, v in tags.items()}

        # For backward compatibility with v2 format: EventAccumulator puts everything in tensors
        # and leaves other categories empty. We'll match this behavior for list_* methods.
        all_data_tags = set()
        all_data_tags.update(tags["scalars"])
        all_data_tags.update(tags["images"])
        all_data_tags.update(tags["videos"])
        all_data_tags.update(tags["histograms"])
        all_data_tags.update(tags["audio"])
        all_data_tags.update(tags["text"])
        all_data_tags.update(tags["meshes"])
        all_data_tags.update(tags["hyperparameters"])
        all_data_tags.update(tags["pr_curves"])
        all_data_tags.update(tags["tensors"])

        # For external API, mostly match EventAccumulator behavior but keep text separate
        # since the original parser did identify text tags by scanning tensors
        self._tags_cache = {
            "scalars": self._detailed_tags["scalars"],
            "images": self._detailed_tags["images"],
            "videos": self._detailed_tags["videos"],
            "histograms": self._detailed_tags["histograms"],
            "audio": self._detailed_tags["audio"],
            "text": self._detailed_tags["text"],  # Keep text tags identifiable
            "meshes": self._detailed_tags["meshes"],
            "hyperparameters": self._detailed_tags["hyperparameters"],
            "pr_curves": self._detailed_tags["pr_curves"],
            "tensors": self._detailed_tags["tensors"],
        }

        self._event_count = event_count

        logger.debug(f"Tag scan complete. Found {event_count} events")
        logger.debug(
            f"Tags found - scalars: {len(self._tags_cache['scalars'])}, "
            f"images: {len(self._tags_cache['images'])}, "
            f"videos: {len(self._tags_cache['videos'])}, "
            f"histograms: {len(self._tags_cache['histograms'])}, "
            f"audio: {len(self._tags_cache['audio'])}, "
            f"text: {len(self._tags_cache['text'])}, "
            f"meshes: {len(self._tags_cache['meshes'])}, "
            f"hyperparameters: {len(self._tags_cache['hyperparameters'])}, "
            f"pr_curves: {len(self._tags_cache['pr_curves'])}, "
            f"tensors: {len(self._tags_cache['tensors'])}"
        )

        return self._tags_cache

    def list_scalars(self) -> list[str]:
        """List all scalar tags in the event file."""
        return self._scan_tags()["scalars"]

    def list_images(self) -> list[str]:
        """List all image tags in the event file."""
        return self._scan_tags()["images"]

    def list_videos(self) -> list[str]:
        """List all video tags in the event file."""
        return self._scan_tags()["videos"]

    def list_histograms(self) -> list[str]:
        """List all histogram tags in the event file."""
        return self._scan_tags()["histograms"]

    def list_tensors(self) -> list[str]:
        """List all tensor tags in the event file."""
        return self._scan_tags()["tensors"]

    def list_audio(self) -> list[str]:
        """List all audio tags in the event file."""
        return self._scan_tags()["audio"]

    def list_text(self) -> list[str]:
        """List all text tags in the event file."""
        return self._scan_tags()["text"]

    def list_meshes(self) -> list[str]:
        """List all mesh tags in the event file."""
        return self._scan_tags()["meshes"]

    def list_hyperparameters(self) -> list[str]:
        """List all hyperparameter tags in the event file."""
        return self._scan_tags()["hyperparameters"]

    def list_pr_curves(self) -> list[str]:
        """List all PR curve tags in the event file."""
        return self._scan_tags()["pr_curves"]

    def list_all_content(self) -> dict[str, list[str]]:
        """List all content organized by type."""
        return self._scan_tags()

    def iterate_scalar_data(self, tag: str) -> Iterator[ScalarData]:
        """Iterate over scalar data for a given tag."""
        for event in self._iterate_events():
            if event.HasField("summary"):
                for value in event.summary.value:
                    if value.tag == tag:
                        scalar_val = None
                        logger.debug(f"Processing scalar tag: {tag}")
                        logger.debug(
                            f"  Has simple_value: {value.HasField('simple_value')}"
                        )
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
                                    logger.debug(
                                        f"  Extracted from tensor: {scalar_val}"
                                    )
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

    def iterate_image_data(self, tag: str) -> Iterator[ImageData]:
        """Iterate over image data for a given tag."""
        for event in self._iterate_events():
            if event.HasField("summary"):
                for value in event.summary.value:
                    if value.tag == tag:
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
                                decoded_image = self._decode_image_from_tensor(
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
                                            ext = self.get_image_extension(item, tag)
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

    def _decode_image_from_tensor(self, tensor_proto: Any) -> bytes | None:
        """Decode an image from a tensor_proto."""
        try:
            import numpy as np
            from PIL import Image
            import io

            arr = tensor_util.make_ndarray(tensor_proto)
            logger.debug(f"Decoding image tensor: shape={arr.shape}, dtype={arr.dtype}")

            # Squeeze batch dimension if present
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr.squeeze(0)

            # Handle channel-first format (C, H, W) -> (H, W, C)
            if arr.ndim == 3 and arr.shape[0] in [1, 3, 4]:
                arr = np.transpose(arr, (1, 2, 0))

            # Handle grayscale with no channel dim
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=-1)

            # Normalize to 0-255
            if arr.dtype == np.float32 or arr.dtype == np.float64:
                logger.debug("Normalizing float tensor to uint8")
                arr = (arr * 255).astype(np.uint8)

            # Ensure it is uint8
            if arr.dtype != np.uint8:
                logger.warning(f"Unsupported image tensor dtype: {arr.dtype}")
                return None

            # Handle single-channel (grayscale)
            if arr.shape[-1] == 1:
                arr = arr.squeeze(axis=-1)

            logger.debug(f"Final array shape for Pillow: {arr.shape}")
            # Convert to image
            img = Image.fromarray(arr)

            # Save to bytes buffer
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            logger.debug("Successfully decoded tensor to PNG")
            return buf.getvalue()

        except Exception as e:
            logger.error(f"Failed to decode image from tensor: {e}")
            return None

    def iterate_video_data(self, tag: str) -> Iterator[VideoData]:
        """Iterate over video data for a given tag."""
        for event in self._iterate_events():
            if event.HasField("summary"):
                for value in event.summary.value:
                    if value.tag == tag:
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

    def iterate_histogram_data(self, tag: str) -> Iterator[HistogramData]:
        """Iterate over histogram data for a given tag."""
        for event in self._iterate_events():
            if event.HasField("summary"):
                for value in event.summary.value:
                    if value.tag == tag and value.HasField("histo"):
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

    def iterate_audio_data(self, tag: str) -> Iterator[AudioData]:
        """Iterate over audio data for a given tag."""
        for event in self._iterate_events():
            if event.HasField("summary"):
                for value in event.summary.value:
                    if value.tag == tag:
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
                                        audio_data = (
                                            arr.item() if arr.ndim == 0 else arr[0]
                                        )
                                        if isinstance(audio_data, bytes):
                                            audio_bytes = audio_data
                                        else:
                                            # If it's a string, encode it to bytes
                                            audio_bytes = str(audio_data).encode(
                                                "utf-8"
                                            )

                                        yield AudioData(
                                            step=event.step,
                                            encoded_audio_string=audio_bytes,
                                            content_type="audio/wav",  # Default assumption
                                            sample_rate=22050.0,  # Default assumption
                                            length_frames=len(audio_bytes),
                                            wall_time=event.wall_time,
                                        )
                                except Exception as e:
                                    logger.debug(
                                        f"Failed to extract audio from tensor for tag {tag}: {e}"
                                    )

    def iterate_text_data(self, tag: str) -> Iterator[TextData]:
        """Iterate over text data for a given tag."""
        for event in self._iterate_events():
            if event.HasField("summary"):
                for value in event.summary.value:
                    if (
                        value.tag == tag
                        and value.HasField("tensor")
                        and value.tensor.dtype == 7
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

    def iterate_mesh_data(self, tag: str) -> Iterator[MeshData]:
        """Iterate over mesh data for a given tag.

        TensorBoard mesh plugin stores 3D data as tensors with plugin_name="mesh".
        Mesh data consists of VERTEX, FACE, and COLOR components stored as separate tags.
        """
        # Collect mesh components by finding related tags
        base_tag = tag.rstrip("_VERTEX").rstrip("_FACE").rstrip("_COLOR")

        # Storage for mesh components by step
        mesh_components: dict[int, dict[str, np.ndarray]] = {}

        # Iterate through events to collect all mesh components
        for event in self._iterate_events():
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

    def iterate_hyperparameter_data(self, tag: str) -> Iterator[HyperparameterData]:
        """Iterate over hyperparameter data for a given tag.

        TensorBoard hyperparameters are stored with plugin_name="hparams".
        The actual data is serialized in the metadata content field.
        """
        if not HPARAMS_AVAILABLE:
            logger.warning(
                "TensorBoard hparams plugin not available. Cannot parse hyperparameters."
            )
            return

        for event in self._iterate_events():
            if event.HasField("summary"):
                for value in event.summary.value:
                    if value.tag == tag and value.HasField("metadata"):
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
                                    for (
                                        param_name,
                                        param_value,
                                    ) in session_info.hparams.items():
                                        hparams[param_name] = (
                                            self._extract_protobuf_value(param_value)
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

    def iterate_pr_curve_data(self, tag: str) -> Iterator[PRCurveData]:
        """Iterate over PR curve data for a given tag."""
        for event in self._iterate_events():
            if event.HasField("summary"):
                for value in event.summary.value:
                    if value.tag == tag and value.HasField("tensor"):
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

    def get_image_extension(self, image_bytes: bytes, tag: str = "unknown") -> str:
        """Determine image extension from bytes using python-magic."""
        # Use python-magic to detect the actual file type
        mime_type = magic.from_buffer(image_bytes, mime=True)
        logger.debug(
            f"MIME type for image bytes (tag='{tag}', len={len(image_bytes)}): {mime_type}"
        )

        # Map MIME types to extensions
        mime_to_ext = {
            "image/png": "png",
            "image/jpeg": "jpg",
            "image/jpg": "jpg",
            "image/gif": "gif",
            "image/bmp": "bmp",
            "image/tiff": "tiff",
            "image/webp": "webp",
            "image/svg+xml": "svg",
        }

        ext = mime_to_ext.get(mime_type, "bin")
        if ext == "bin":
            # This is expected for non-image string tensors, so log at debug level.
            logger.debug(
                f"Could not determine image type from MIME type '{mime_type}'. Returning .bin"
            )
        return ext

    def get_audio_extension(self, content_type: str) -> str:
        """Determine audio extension from content type."""
        if "wav" in content_type:
            return "wav"
        elif "mp3" in content_type:
            return "mp3"
        elif "ogg" in content_type:
            return "ogg"
        else:
            return "audio"

    def _save_unified_histograms(
        self,
        histogram_data_buffers: dict[str, list[tuple[int, dict, float]]],
        output_path: Path,
    ) -> None:
        """Save histogram data in unified CSV + NPZ formats."""
        histograms_dir = output_path / "histograms"

        for tag, data_points in histogram_data_buffers.items():
            if not data_points:
                continue

            # Create histograms directory only when we have data to save
            histograms_dir.mkdir(parents=True, exist_ok=True)

            safe_tag = tag.replace("/", "_")

            # Sort by step
            data_points.sort(key=lambda x: x[0])

            # Check histogram format (legacy vs tensor)
            first_format = data_points[0][1].get("format", "unknown")

            if first_format == "legacy_histo":
                # Save legacy histogram format
                self._save_legacy_histogram_unified(
                    tag, safe_tag, data_points, histograms_dir
                )
            elif first_format == "tensor":
                # Save tensor histogram format
                self._save_tensor_histogram_unified(
                    tag, safe_tag, data_points, histograms_dir
                )
            else:
                logger.warning(
                    f"Unknown histogram format for tag '{tag}': {first_format}"
                )

    def _save_legacy_histogram_unified(
        self, tag: str, safe_tag: str, data_points: list, histograms_dir: Path
    ) -> None:
        """Save legacy histogram data in unified formats."""
        # Prepare data for CSV
        csv_rows = []
        npz_data: dict[str, list[Any]] = {
            "step": [],
            "wall_time": [],
            "min": [],
            "max": [],
            "num": [],
            "sum": [],
            "sum_squares": [],
            "bucket_limits": [],
            "bucket_counts": [],
        }

        for step, hist_data, wall_time in data_points:
            # CSV row
            bucket_limits = hist_data["bucket_limit"]
            bucket_counts = hist_data["bucket"]

            csv_rows.append(
                {
                    "step": step,
                    "wall_time": wall_time,
                    "min": hist_data["min"],
                    "max": hist_data["max"],
                    "num": hist_data["num"],
                    "sum": hist_data["sum"],
                    "sum_squares": hist_data["sum_squares"],
                    "num_buckets": len(bucket_limits),
                    "bucket_limits_str": "|".join(map(str, bucket_limits)),
                    "bucket_counts_str": "|".join(map(str, bucket_counts)),
                }
            )

            # NPZ arrays
            npz_data["step"].append(step)
            npz_data["wall_time"].append(wall_time)
            npz_data["min"].append(hist_data["min"])
            npz_data["max"].append(hist_data["max"])
            npz_data["num"].append(hist_data["num"])
            npz_data["sum"].append(hist_data["sum"])
            npz_data["sum_squares"].append(hist_data["sum_squares"])
            npz_data["bucket_limits"].append(bucket_limits)
            npz_data["bucket_counts"].append(bucket_counts)

        # Save CSV
        csv_file = histograms_dir / f"{safe_tag}.csv"
        with csv_file.open("w") as f:
            f.write(
                "step,wall_time,min,max,num,sum,sum_squares,num_buckets,bucket_limits,bucket_counts\n"
            )
            for row in csv_rows:
                f.write(
                    f"{row['step']},{row['wall_time']:.6f},{row['min']:.10g},{row['max']:.10g},"
                    f"{row['num']},{row['sum']:.10g},{row['sum_squares']:.10g},{row['num_buckets']},"
                    f'"{row["bucket_limits_str"]}","{row["bucket_counts_str"]}"\n'
                )

        # Save NPZ
        npz_file = histograms_dir / f"{safe_tag}.npz"
        # Convert lists to arrays for npz
        npz_arrays: dict[str, Any] = {
            key: np.array(values)
            if key not in ["bucket_limits", "bucket_counts"]
            else values
            for key, values in npz_data.items()
        }
        npz_arrays["tag"] = tag
        npz_arrays["format"] = "legacy_histo"
        np.savez_compressed(npz_file, **npz_arrays)

        logger.debug(
            f"Saved histogram '{tag}' to {csv_file} and {npz_file} ({len(data_points)} points)"
        )

    def _save_tensor_histogram_unified(
        self, tag: str, safe_tag: str, data_points: list, histograms_dir: Path
    ) -> None:
        """Save tensor histogram data in unified formats."""
        # For tensor histograms, save raw tensor data and metadata
        csv_rows = []
        npz_data: dict[str, list[Any]] = {
            "step": [],
            "wall_time": [],
            "tensor_shapes": [],
            "tensor_dtypes": [],
            "tensor_data": [],
        }

        for step, hist_data, wall_time in data_points:
            tensor_data = hist_data["tensor_data"]
            tensor_shape = hist_data["tensor_shape"]
            tensor_dtype = hist_data["tensor_dtype"]

            csv_rows.append(
                {
                    "step": step,
                    "wall_time": wall_time,
                    "tensor_shape": str(tensor_shape),
                    "tensor_dtype": tensor_dtype,
                    "tensor_size": len(tensor_data),
                    "tensor_data_str": "|".join(map(str, tensor_data)),
                }
            )

            npz_data["step"].append(step)
            npz_data["wall_time"].append(wall_time)
            npz_data["tensor_shapes"].append(tensor_shape)
            npz_data["tensor_dtypes"].append(tensor_dtype)
            npz_data["tensor_data"].append(tensor_data)

        # Save CSV
        csv_file = histograms_dir / f"{safe_tag}.csv"
        with csv_file.open("w") as f:
            f.write(
                "step,wall_time,tensor_shape,tensor_dtype,tensor_size,tensor_data\n"
            )
            for row in csv_rows:
                f.write(
                    f'{row["step"]},{row["wall_time"]:.6f},"{row["tensor_shape"]}",'
                    f'"{row["tensor_dtype"]}",{row["tensor_size"]},"{row["tensor_data_str"]}"\n'
                )

        # Save NPZ - handle variable-length tensor data as object arrays
        npz_file = histograms_dir / f"{safe_tag}.npz"
        np.savez_compressed(
            npz_file,
            step=np.array(npz_data["step"]),
            wall_time=np.array(npz_data["wall_time"]),
            tensor_shapes=np.array(npz_data["tensor_shapes"], dtype=object),
            tensor_dtypes=np.array(npz_data["tensor_dtypes"]),
            tensor_data=np.array(npz_data["tensor_data"], dtype=object),
            tag=tag,
            format="tensor",
        )

        logger.debug(
            f"Saved tensor histogram '{tag}' to {csv_file} and {npz_file} ({len(data_points)} points)"
        )

    def _save_image(
        self,
        event: event_pb2.Event,
        value: Any,
        output_path: Path,
        digits: int,
        image_format: str,
        image_quality: int,
    ) -> None:
        """Save image data to file, handling batches of pre-encoded images."""
        tag = value.tag
        safe_tag = tag.replace("/", "_")
        tag_dir = output_path / "images" / safe_tag
        tag_dir.mkdir(parents=True, exist_ok=True)

        image_byte_list = []
        if value.HasField("image"):
            image_byte_list.append(value.image.encoded_image_string)
        elif value.HasField("tensor"):
            if TensorDataDetector.is_image_tensor(value.tensor, tag):
                decoded_image = self._decode_image_from_tensor(value.tensor)
                if decoded_image:
                    image_byte_list.append(decoded_image)
            elif value.tensor.dtype == 7:
                try:
                    arr = tensor_util.make_ndarray(value.tensor)
                    for item in arr:
                        if isinstance(item, bytes):
                            # Filter out non-image data by checking extension
                            ext = self.get_image_extension(item, tag)
                            if ext != "bin":
                                image_byte_list.append(item)
                except Exception as e:
                    logger.warning(
                        f"Could not decode string tensor for tag '{tag}': {e}"
                    )

        if image_byte_list:
            padded_step = str(event.step).zfill(digits)
            for i, image_bytes in enumerate(image_byte_list):
                # Determine the output file extension based on the chosen format
                ext = image_format
                # Append index for batches to avoid overwriting
                filename = (
                    f"{padded_step}_{i}.{ext}"
                    if len(image_byte_list) > 1
                    else f"{padded_step}.{ext}"
                )
                image_file = tag_dir / filename

                try:
                    from PIL import Image
                    import io

                    image = Image.open(io.BytesIO(image_bytes))
                    if image.mode == "RGBA":
                        image = image.convert("RGB")
                    if image_format == "jpg":
                        image.save(image_file, format="JPEG", quality=image_quality)
                    else:  # png
                        image.save(image_file, format="PNG")
                except ImportError:
                    logger.error(
                        "Pillow (PIL) is not installed. Cannot convert image format."
                    )
                    logger.info("Please install it with: pip install Pillow")
                    sys.exit(1)
                except Exception as e:
                    logger.warning(f"Failed to save image {image_file}: {e}")
        else:
            logger.warning(
                f"Could not extract image for tag '{tag}' at step {event.step}"
            )

    def _save_video(
        self,
        event: event_pb2.Event,
        value: Any,
        output_path: Path,
        digits: int,
    ) -> None:
        """Save video data to file."""
        tag = value.tag
        safe_tag = tag.replace("/", "_")
        tag_dir = output_path / "videos" / safe_tag
        tag_dir.mkdir(parents=True, exist_ok=True)

        video_data = None
        if value.HasField("image"):
            # Video data stored as image (GIF) in TensorBoard
            video_data = value.image.encoded_image_string

        if video_data:
            padded_step = str(event.step).zfill(digits)

            # Determine file extension based on content type
            if video_data.startswith(b"GIF"):
                ext = "gif"
            else:
                ext = "bin"  # Unknown format

            filename = f"{padded_step}.{ext}"
            video_file = tag_dir / filename

            try:
                with video_file.open("wb") as f:
                    f.write(video_data)
                logger.debug(f"Saved video to {video_file} ({len(video_data)} bytes)")
            except Exception as e:
                logger.warning(f"Failed to save video {video_file}: {e}")
        else:
            logger.warning(
                f"Could not extract video data for tag '{tag}' at step {event.step}"
            )

    def _save_histogram(
        self,
        event: event_pb2.Event,
        value: Any,
        output_path: Path,
        histogram_images: bool = False,
        histogram_data_buffers: dict[str, list[tuple[int, dict, float]]] | None = None,
    ) -> None:
        """Save histogram data using unified CSV + NPZ export only."""
        tag = value.tag

        if histogram_images:
            # Save as visualization images (old behavior)
            self._save_histogram_as_image(event, value, output_path)
        else:
            # Only buffer data for unified CSV + NPZ export (no legacy text files)
            if value.HasField("histo"):
                # Legacy format with histo field
                if histogram_data_buffers is not None:
                    hist = value.histo
                    hist_data = {
                        "min": float(hist.min),
                        "max": float(hist.max),
                        "num": int(hist.num),
                        "sum": float(hist.sum),
                        "sum_squares": float(hist.sum_squares),
                        "bucket_limit": list(hist.bucket_limit),
                        "bucket": list(hist.bucket),
                        "format": "legacy_histo",
                    }
                    if tag not in histogram_data_buffers:
                        histogram_data_buffers[tag] = []
                    histogram_data_buffers[tag].append(
                        (event.step, hist_data, event.wall_time)
                    )

            elif value.HasField("tensor"):
                # TensorBoard v2 format with tensor field
                if histogram_data_buffers is not None:
                    try:
                        arr = tensor_util.make_ndarray(value.tensor)
                        hist_data = {
                            "tensor_data": arr.flatten().tolist(),
                            "tensor_shape": arr.shape,
                            "tensor_dtype": str(arr.dtype),
                            "format": "tensor",
                        }
                        if tag not in histogram_data_buffers:
                            histogram_data_buffers[tag] = []
                        histogram_data_buffers[tag].append(
                            (event.step, hist_data, event.wall_time)
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to buffer tensor histogram data for {tag}: {e}"
                        )
            else:
                logger.warning(f"Unknown histogram format for tag '{tag}'")

    def _save_histogram_as_image(
        self, event: event_pb2.Event, value: Any, output_path: Path
    ) -> None:
        """Save histogram as visualization image (legacy behavior)."""
        tag = value.tag
        safe_tag = tag.replace("/", "_")
        histogram_file = output_path / "histograms" / f"{safe_tag}.txt"
        histogram_file.parent.mkdir(parents=True, exist_ok=True)

        hist = value.histo
        with histogram_file.open("a") as f:
            f.write(f"Step: {event.step}\n")
            f.write(f"Min: {hist.min}, Max: {hist.max}\n")
            f.write(f"Count: {hist.num}, Sum: {hist.sum}\n")
            f.write("Buckets:\n")
            for limit, count in zip(hist.bucket_limit, hist.bucket):
                f.write(f"  [{limit:.6f}]: {count}\n")
            f.write("\n")

    def _get_enabled_types(
        self, all_types: set[str], type_filters: dict[str, set[str]] | None
    ) -> set[str]:
        """Determine which data types should be processed based on filters."""
        if not type_filters:
            return all_types

        ignore_types = type_filters.get("ignore", set())
        select_types = type_filters.get("select", set())

        if select_types:
            # Only process selected types
            return all_types & select_types
        elif ignore_types:
            # Process all types except ignored ones
            return all_types - ignore_types
        else:
            # No filtering
            return all_types

    def extract_all_to_directory(
        self,
        output_dir: str,
        digits: int = 6,
        image_format: str = "jpg",
        image_quality: int = 90,
        audio_format: str = "mp3",
        histogram_images: bool = False,
        ply_format: str = "binary",
        type_filters: dict[str, set[str]] | None = None,
    ) -> None:
        """Extract all data to a directory structure using modular export pipeline."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Output directory created: {output_path}")

        # Determine which data types to process based on filters
        all_types = {
            "scalar",
            "image",
            "video",
            "histogram",
            "audio",
            "text",
            "mesh",
            "hyperparameter",
            "pr_curve",
        }
        enabled_types = self._get_enabled_types(all_types, type_filters)

        if not enabled_types:
            logger.warning("No data types enabled for processing")
            return

        logger.debug(f"Processing data types: {', '.join(sorted(enabled_types))}")

        # Create export configuration
        config = ExportConfig(
            output_path=output_path,
            enabled_types=enabled_types,
            digits=digits,
            histogram_images=histogram_images,
            audio_format=audio_format,
            ply_format=ply_format,
        )

        # Create export pipeline
        pipeline = ExportPipeline(config)

        # Get events iterator with optional progress tracking
        events_iterator: Iterator[event_pb2.Event] = self._iterate_events()

        if self.show_progress:
            if self._event_count:
                events_iterator = tqdm(
                    events_iterator,
                    total=self._event_count,
                    desc="Extracting data",
                    unit=" events",
                )  # type: ignore[assignment]
            else:
                events_iterator = tqdm(
                    events_iterator, desc="Extracting data", unit=" events"
                )  # type: ignore[assignment]

        try:
            # Process all events through pipeline in single pass
            # Process all events through the pipeline
            event_count = pipeline.process_events(events_iterator)

        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            raise

        logger.debug(f"Extraction completed. Processed {event_count} events.")

    def get_virtual_paths(self, digits: int = 6) -> list[str]:
        """Get all virtual paths that would exist in the filesystem."""
        paths = []

        # Check if we have proper initialization
        if hasattr(self, "_tags_cache") and hasattr(self, "event_file_path"):
            try:
                self._scan_tags()  # Ensure cache is populated
                all_tags = self._detailed_tags
                if all_tags is None:
                    return []
            except Exception:
                # If scanning fails, fall back to mock-based method
                pass

        # Check if we have detailed tags from scanning
        if hasattr(self, "_detailed_tags") and self._detailed_tags is not None:
            all_tags = self._detailed_tags
        else:
            # Fallback for tests using mocks - call the individual methods
            all_tags = None

        # Add directories
        paths.extend(
            [
                "scalars/",
                "images/",
                "histograms/",
                "audio/",
                "text/",
                "meshes/",
                "hp_params/",
            ]
        )

        if all_tags is not None:
            # Normal mode - use scanned tags
            # Scalar paths
            for tag in all_tags["scalars"]:
                safe_tag = tag.replace("/", "_")
                paths.append(f"scalars/{safe_tag}.txt")
        else:
            # Fallback mode for tests using mocks - call the individual methods
            # Scalar paths
            try:
                for tag in self.list_scalars():
                    safe_tag = tag.replace("/", "_")
                    paths.append(f"scalars/{safe_tag}.txt")
            except Exception:
                pass

        if all_tags is not None:
            # Normal mode - use scanned tags
            # For other types, we need to iterate to get step numbers
            # This is less efficient but necessary for virtual paths

            # Image paths
            for tag in all_tags["images"]:
                safe_tag = tag.replace("/", "_")
                paths.append(f"images/{safe_tag}/")
                for data in self.iterate_image_data(tag):
                    ext = self.get_image_extension(data.encoded_image_string, tag)
                    padded_step = str(data.step).zfill(digits)
                    paths.append(f"images/{safe_tag}/{padded_step}.{ext}")

            # Histogram paths
            for tag in all_tags["histograms"]:
                safe_tag = tag.replace("/", "_")
                paths.append(f"histograms/{safe_tag}.txt")

            # Audio paths
            for tag in all_tags["audio"]:
                safe_tag = tag.replace("/", "_")
                paths.append(f"audio/{safe_tag}/")
                for data_audio in self.iterate_audio_data(tag):
                    ext = self.get_audio_extension(data_audio.content_type)
                    padded_step = str(data_audio.step).zfill(digits)
                    paths.append(f"audio/{safe_tag}/{padded_step}.{ext}")

            # Text paths
            for tag in all_tags["text"]:
                safe_tag = tag.replace("/", "_")
                paths.append(f"text/{safe_tag}/")
                for data_text in self.iterate_text_data(tag):
                    padded_step = str(data_text.step).zfill(digits)
                    paths.append(f"text/{safe_tag}/{padded_step}.txt")

            # Mesh paths - group by base tag (without _VERTEX, _FACE, _COLOR suffixes)
            mesh_base_tags = set()
            for tag in all_tags["meshes"]:
                base_tag = tag.rstrip("_VERTEX").rstrip("_FACE").rstrip("_COLOR")
                mesh_base_tags.add(base_tag)

            for base_tag in mesh_base_tags:
                safe_tag = base_tag.replace("/", "_")
                paths.append(f"meshes/{safe_tag}/")
                try:
                    for data_mesh in self.iterate_mesh_data(base_tag):
                        padded_step = str(data_mesh.step).zfill(digits)
                        paths.append(f"meshes/{safe_tag}/{padded_step}.ply")
                except Exception:
                    # Skip if mesh data iteration fails
                    pass
        else:
            # Fallback mode for tests using mocks - call the individual methods
            # Image paths
            try:
                for tag in self.list_images():
                    safe_tag = tag.replace("/", "_")
                    paths.append(f"images/{safe_tag}/")
                    try:
                        image_data = list(self.iterate_image_data(tag))
                        for image_item in image_data:
                            ext = self.get_image_extension(
                                image_item.encoded_image_string
                            )
                            padded_step = str(image_item.step).zfill(digits)
                            paths.append(f"images/{safe_tag}/{padded_step}.{ext}")
                    except Exception:
                        pass
            except Exception:
                pass

            # Histogram paths
            try:
                for tag in self.list_histograms():
                    safe_tag = tag.replace("/", "_")
                    paths.append(f"histograms/{safe_tag}.txt")
            except Exception:
                pass

            # Audio paths
            try:
                for tag in self.list_audio():
                    safe_tag = tag.replace("/", "_")
                    paths.append(f"audio/{safe_tag}/")
                    try:
                        audio_data = list(self.iterate_audio_data(tag))
                        for audio_item in audio_data:
                            ext = self.get_audio_extension(audio_item.content_type)
                            padded_step = str(audio_item.step).zfill(digits)
                            paths.append(f"audio/{safe_tag}/{padded_step}.{ext}")
                    except Exception:
                        pass
            except Exception:
                pass

            # Text paths
            try:
                for tag in self.list_text():
                    safe_tag = tag.replace("/", "_")
                    paths.append(f"text/{safe_tag}/")
                    try:
                        text_data = list(self.iterate_text_data(tag))
                        for text_item in text_data:
                            padded_step = str(text_item.step).zfill(digits)
                            paths.append(f"text/{safe_tag}/{padded_step}.txt")
                    except Exception:
                        pass
            except Exception:
                pass

            # Mesh paths - group by base tag (without _VERTEX, _FACE, _COLOR suffixes)
            try:
                mesh_base_tags = set()
                for tag in self.list_meshes():
                    base_tag = tag.rstrip("_VERTEX").rstrip("_FACE").rstrip("_COLOR")
                    mesh_base_tags.add(base_tag)

                for base_tag in mesh_base_tags:
                    safe_tag = base_tag.replace("/", "_")
                    paths.append(f"meshes/{safe_tag}/")
                    try:
                        mesh_data = list(self.iterate_mesh_data(base_tag))
                        for mesh_item in mesh_data:
                            padded_step = str(mesh_item.step).zfill(digits)
                            paths.append(f"meshes/{safe_tag}/{padded_step}.ply")
                    except Exception:
                        pass
            except Exception:
                pass

        # Hyperparameter paths - single file per entire log
        if all_tags is not None:
            if all_tags["hyperparameters"]:
                paths.append("hp_params/hp_params.yaml")
        else:
            # Fallback mode for tests
            try:
                hyperparams = self.list_hyperparameters()
                if hyperparams:
                    paths.append("hp_params/hp_params.yaml")
            except Exception:
                pass

        return sorted(set(paths))
