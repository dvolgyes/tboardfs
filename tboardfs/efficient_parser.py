"""Efficient TensorBoard event file parser using iterators.

This parser uses EventFileLoader directly to iterate over events without
loading everything into memory, making it much more efficient for large files.
"""

from pathlib import Path
from collections.abc import Iterator
from typing import Any

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.util import tensor_util
from tensorboard.compat.proto import event_pb2
from tqdm import tqdm
import sys
from loguru import logger
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
from tboardfs.core.histogram_utils import HistogramExportUtils
from tboardfs.core.constants import (
    TensorFlowDTypes,
    DEFAULT_DIGITS,
    FileSystemConstants,
)
from tboardfs.core.virtual_filesystem import VirtualFilesystemGenerator
from tboardfs.core.image_processor import ImageProcessor
from tboardfs.core.data_iterators import DataIteratorFactory

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

        # Initialize virtual filesystem generator
        self._virtual_fs = VirtualFilesystemGenerator(self)

        # Initialize data iterator factory
        self._data_iterators = DataIteratorFactory(self)

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

    def _get_cached_tags(self) -> dict[str, list[str]] | None:
        """Return cached tags if available."""
        return self._tags_cache

    def _initialize_tag_collections(self) -> dict[str, set[str]]:
        """Initialize empty tag collections for all supported types."""
        return {
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

    def _setup_event_iterator(self) -> Iterator[event_pb2.Event]:
        """Create event iterator with optional progress tracking."""
        iterator = self._iterate_events()
        if self.show_progress:
            iterator = tqdm(iterator, desc="Scanning for tags", unit=" events")  # type: ignore[assignment]
        return iterator

    def _classify_tag_by_metadata(
        self, value: Any, tags: dict[str, set[str]], tag: str
    ) -> bool:
        """Classify tag based on metadata plugin name. Returns True if classified."""
        if not value.HasField("metadata"):
            return False

        plugin_name = value.metadata.plugin_data.plugin_name

        # Plugin name to category mapping
        plugin_mappings = {
            "scalars": "scalars",
            "images": "images",
            "histograms": "histograms",
            "audio": "audio",
            "text": "text",
            "mesh": "meshes",
            "hparams": "hyperparameters",
            "pr_curves": "pr_curves",
        }

        category = plugin_mappings.get(plugin_name, "tensors")
        tags[category].add(tag)
        return True

    def _classify_tag_legacy_format(
        self, value: Any, tags: dict[str, set[str]], tag: str
    ) -> None:
        """Classify tag based on legacy format field presence."""
        if value.HasField("simple_value"):
            tags["scalars"].add(tag)
        elif value.HasField("image"):
            if TensorDataDetector.is_video_data(value.image.encoded_image_string, tag):
                tags["videos"].add(tag)
            else:
                tags["images"].add(tag)
        elif value.HasField("histo"):
            tags["histograms"].add(tag)
        elif value.HasField("audio"):
            tags["audio"].add(tag)
        elif value.HasField("tensor"):
            self._classify_tensor_tag(value.tensor, tags, tag)

    def _classify_tensor_tag(
        self, tensor: Any, tags: dict[str, set[str]], tag: str
    ) -> None:
        """Classify tag based on tensor analysis."""
        try:
            arr = tensor_util.make_ndarray(tensor)
            if arr.size == 1:
                tags["scalars"].add(tag)
            elif tensor.dtype == TensorFlowDTypes.DT_STRING:
                tags["text"].add(tag)
            elif TensorDataDetector.is_pr_curve_tensor(tensor, tag):
                tags["pr_curves"].add(tag)
            elif TensorDataDetector.is_image_tensor(tensor, tag):
                tags["images"].add(tag)
            else:
                tags["tensors"].add(tag)
        except Exception as e:
            logger.debug(
                f"Could not make ndarray for tensor tag {tag}: {e}. Treating as generic tensor."
            )
            tags["tensors"].add(tag)

    def _organize_scan_results(self, tags: dict[str, set[str]]) -> dict[str, list[str]]:
        """Convert tag sets to sorted lists and prepare final cache structure."""
        # Store detailed tags for internal use
        self._detailed_tags = {k: sorted(v) for k, v in tags.items()}

        # Create backward-compatible cache structure
        return {
            "scalars": self._detailed_tags["scalars"],
            "images": self._detailed_tags["images"],
            "videos": self._detailed_tags["videos"],
            "histograms": self._detailed_tags["histograms"],
            "audio": self._detailed_tags["audio"],
            "text": self._detailed_tags["text"],
            "meshes": self._detailed_tags["meshes"],
            "hyperparameters": self._detailed_tags["hyperparameters"],
            "pr_curves": self._detailed_tags["pr_curves"],
            "tensors": self._detailed_tags["tensors"],
        }

    def _log_scan_summary(
        self, event_count: int, tags_cache: dict[str, list[str]]
    ) -> None:
        """Log summary of tag scan results."""
        logger.debug(f"Tag scan complete. Found {event_count} events")
        logger.debug(
            f"Tags found - scalars: {len(tags_cache['scalars'])}, "
            f"images: {len(tags_cache['images'])}, "
            f"videos: {len(tags_cache['videos'])}, "
            f"histograms: {len(tags_cache['histograms'])}, "
            f"audio: {len(tags_cache['audio'])}, "
            f"text: {len(tags_cache['text'])}, "
            f"meshes: {len(tags_cache['meshes'])}, "
            f"hyperparameters: {len(tags_cache['hyperparameters'])}, "
            f"pr_curves: {len(tags_cache['pr_curves'])}, "
            f"tensors: {len(tags_cache['tensors'])}"
        )

    def _scan_tags(self) -> dict[str, list[str]]:
        """Scan the file once to build a directory of all tags by type."""
        # Check cache first
        cached_tags = self._get_cached_tags()
        if cached_tags is not None:
            return cached_tags

        logger.debug("Scanning file for tags...")

        # Initialize data structures
        tags = self._initialize_tag_collections()
        event_count = 0

        # Process all events
        iterator = self._setup_event_iterator()

        for event in iterator:
            event_count += 1

            if event.HasField("summary"):
                for value in event.summary.value:
                    tag = value.tag

                    # Try metadata-based classification first
                    if not self._classify_tag_by_metadata(value, tags, tag):
                        # Fall back to legacy format detection
                        self._classify_tag_legacy_format(value, tags, tag)

        # Organize results and update cache
        self._tags_cache = self._organize_scan_results(tags)
        self._event_count = event_count

        # Log summary
        self._log_scan_summary(event_count, self._tags_cache)

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
        return self._data_iterators.iterate_scalar_data(tag)

    def iterate_image_data(self, tag: str) -> Iterator[ImageData]:
        """Iterate over image data for a given tag."""
        return self._data_iterators.iterate_image_data(tag)

    def iterate_video_data(self, tag: str) -> Iterator[VideoData]:
        """Iterate over video data for a given tag."""
        return self._data_iterators.iterate_video_data(tag)

    def iterate_histogram_data(self, tag: str) -> Iterator[HistogramData]:
        """Iterate over histogram data for a given tag."""
        return self._data_iterators.iterate_histogram_data(tag)

    def iterate_audio_data(self, tag: str) -> Iterator[AudioData]:
        """Iterate over audio data for a given tag."""
        return self._data_iterators.iterate_audio_data(tag)

    def iterate_text_data(self, tag: str) -> Iterator[TextData]:
        """Iterate over text data for a given tag."""
        return self._data_iterators.iterate_text_data(tag)

    def iterate_mesh_data(self, tag: str) -> Iterator[MeshData]:
        """Iterate over mesh data for a given tag.

        TensorBoard mesh plugin stores 3D data as tensors with plugin_name="mesh".
        Mesh data consists of VERTEX, FACE, and COLOR components stored as separate tags.
        """
        return self._data_iterators.iterate_mesh_data(tag)

    def iterate_hyperparameter_data(self, tag: str) -> Iterator[HyperparameterData]:
        """Iterate over hyperparameter data for a given tag.

        TensorBoard hyperparameters are stored with plugin_name="hparams".
        The actual data is serialized in the metadata content field.
        """
        return self._data_iterators.iterate_hyperparameter_data(tag)

    def iterate_pr_curve_data(self, tag: str) -> Iterator[PRCurveData]:
        """Iterate over PR curve data for a given tag."""
        return self._data_iterators.iterate_pr_curve_data(tag)

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
        """Save histogram data in unified CSV + NPZ formats using shared utility."""

        def sanitize_tag(tag: str) -> str:
            return tag.replace("/", FileSystemConstants.PATH_REPLACEMENT_CHAR)

        HistogramExportUtils.save_unified_histograms(
            histogram_data_buffers, output_path, sanitize_tag_func=sanitize_tag
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
                decoded_image = ImageProcessor.decode_image_from_tensor(value.tensor)
                if decoded_image:
                    image_byte_list.append(decoded_image)
            elif value.tensor.dtype == 7:
                try:
                    arr = tensor_util.make_ndarray(value.tensor)
                    for item in arr:
                        if isinstance(item, bytes):
                            # Filter out non-image data by checking extension
                            ext = ImageProcessor.get_image_extension(item, tag)
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

    def get_virtual_paths(self, digits: int = DEFAULT_DIGITS) -> list[str]:
        """Get all virtual paths that would exist in the filesystem."""
        # Create virtual filesystem generator if not initialized (for tests)
        if not hasattr(self, "_virtual_fs"):
            self._virtual_fs = VirtualFilesystemGenerator(self)
        return self._virtual_fs.get_virtual_paths(digits)
