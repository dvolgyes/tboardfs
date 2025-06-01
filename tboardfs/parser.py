"""TensorBoard event file parser.

This parser supports TensorFlow v2 event file format where all data types
(scalars, images, histograms, audio) are stored as tensors. Text data is
identified by checking tensor dtype == 7 (DT_STRING).
"""

from dataclasses import dataclass

from tensorboard.backend.event_processing import event_accumulator
from tensorboard.util import tensor_util
from tqdm import tqdm
from loguru import logger


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


class TensorBoardParser:
    """Parser for TensorBoard event files."""

    def __init__(self, event_file_path: str, show_progress: bool = False):
        """Initialize parser with event file path."""
        from pathlib import Path
        import os
        
        logger.debug(f"Initializing TensorBoardParser for file: {event_file_path}")
        self.event_file_path = event_file_path
        self.show_progress = show_progress
        
        # Check file size and warn if large
        file_size = Path(event_file_path).stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        logger.debug(f"TensorBoard file size: {file_size_mb:.1f} MB")
        
        if file_size_mb > 100:
            logger.info(f"Large TensorBoard file detected ({file_size_mb:.1f} MB). Loading may take several minutes...")
        
        logger.debug("Creating EventAccumulator with memory-efficient settings")
        # Use smaller size guidance for memory efficiency with large files
        size_guidance = {
            event_accumulator.SCALARS: 10000,  # Limit scalars to reduce memory
            event_accumulator.IMAGES: 100,     # Limit images significantly
            event_accumulator.HISTOGRAMS: 1000,
            event_accumulator.TENSORS: 1000,
            event_accumulator.AUDIO: 100,
        } if file_size_mb > 50 else {
            event_accumulator.SCALARS: 0,
            event_accumulator.IMAGES: 0,
            event_accumulator.HISTOGRAMS: 0,
            event_accumulator.TENSORS: 0,
            event_accumulator.AUDIO: 0,
        }
        
        self.ea = event_accumulator.EventAccumulator(
            event_file_path,
            size_guidance=size_guidance,
        )
        
        logger.debug("Loading events from file (this may take time for large files)")
        if file_size_mb > 100:
            logger.info("Loading large file... Please be patient, this may take several minutes")
        
        # Load with progress indication for large files
        if file_size_mb > 50 and self.show_progress:
            self._reload_with_progress()
        else:
            self.ea.Reload()
        
        logger.debug("EventAccumulator initialized and events loaded")

    def _reload_with_progress(self):
        """Reload EventAccumulator with progress indication."""
        import threading
        import time
        
        logger.info("Loading file with progress indication...")
        
        # Flag to track if loading is complete
        loading_complete = threading.Event()
        
        def load_data():
            try:
                self.ea.Reload()
                loading_complete.set()
            except Exception as e:
                logger.error(f"Error loading TensorBoard file: {e}")
                loading_complete.set()
                raise
        
        # Start loading in a separate thread
        load_thread = threading.Thread(target=load_data)
        load_thread.daemon = True
        load_thread.start()
        
        # Show progress dots while loading
        start_time = time.time()
        while not loading_complete.is_set():
            elapsed = time.time() - start_time
            logger.info(f"Still loading... ({elapsed:.0f}s elapsed)")
            loading_complete.wait(timeout=30)  # Update every 30 seconds
        
        # Wait for thread to complete
        load_thread.join()
        logger.info("File loading completed")

    def list_scalars(self) -> list[str]:
        """List all scalar tags in the event file.

        Note: In TensorFlow v2 format, scalars may be stored as tensors
        and this list might be empty.
        """
        return list(self.ea.Tags()["scalars"])

    def list_images(self) -> list[str]:
        """List all image tags in the event file."""
        return list(self.ea.Tags()["images"])

    def list_histograms(self) -> list[str]:
        """List all histogram tags in the event file."""
        return list(self.ea.Tags()["histograms"])

    def list_tensors(self) -> list[str]:
        """List all tensor tags in the event file."""
        return list(self.ea.Tags()["tensors"])

    def list_audio(self) -> list[str]:
        """List all audio tags in the event file."""
        return list(self.ea.Tags().get("audio", []))

    def list_text(self) -> list[str]:
        """List all text tags in the event file.

        In TensorFlow v2 format, text is stored as tensors with dtype=7 (DT_STRING).
        This method identifies text tensors by checking their dtype.
        """
        # Text is stored as tensors with plugin metadata
        text_tags = []
        for tag in self.list_tensors():
            try:
                events = self.ea.Tensors(tag)
                if (
                    events and events[0].tensor_proto.dtype == 7  # DT_STRING enum value
                ):
                    text_tags.append(tag)
            except Exception:
                pass
        return text_tags

    def get_scalar_data(self, tag: str) -> list[ScalarData]:
        """Get all scalar data for a given tag."""
        scalar_events = self.ea.Scalars(tag)
        return [
            ScalarData(step=event.step, value=event.value, wall_time=event.wall_time)
            for event in scalar_events
        ]

    def get_image_data(self, tag: str) -> list[ImageData]:
        """Get all image data for a given tag."""
        image_events = self.ea.Images(tag)
        return [
            ImageData(
                step=event.step,
                encoded_image_string=event.encoded_image_string,
                width=event.width,
                height=event.height,
                wall_time=event.wall_time,
            )
            for event in image_events
        ]

    def get_histogram_data(self, tag: str) -> list[HistogramData]:
        """Get all histogram data for a given tag."""
        histogram_events = self.ea.Histograms(tag)
        return [
            HistogramData(
                step=event.step,
                min=event.histogram_value.min,
                max=event.histogram_value.max,
                num=event.histogram_value.num,
                sum=event.histogram_value.sum,
                sum_squares=event.histogram_value.sum_squares,
                bucket_limit=list(event.histogram_value.bucket_limit),
                bucket=list(event.histogram_value.bucket),
                wall_time=event.wall_time,
            )
            for event in histogram_events
        ]

    def get_audio_data(self, tag: str) -> list[AudioData]:
        """Get all audio data for a given tag."""
        audio_events = self.ea.Audio(tag)
        audio_data_list = []
        for event in audio_events:
            audio_data_list.append(
                AudioData(
                    step=event.step,
                    encoded_audio_string=event.encoded_audio_string,
                    content_type=event.content_type,
                    sample_rate=event.sample_rate,
                    length_frames=event.length_frames,
                    wall_time=event.wall_time,
                )
            )
        return audio_data_list

    def get_text_data(self, tag: str) -> list[TextData]:
        """Get all text data for a given tag."""
        text_data_list = []
        try:
            tensor_events = self.ea.Tensors(tag)
            for event in tensor_events:
                if event.tensor_proto.dtype == 7:  # DT_STRING enum value
                    # Decode the text from tensor
                    text_value = tensor_util.make_ndarray(event.tensor_proto)
                    if text_value.size > 0:
                        # Handle both scalar and array text
                        text = (
                            text_value.item()
                            if text_value.ndim == 0
                            else str(text_value[0])
                        )
                        if isinstance(text, bytes):
                            text = text.decode("utf-8", errors="replace")
                        text_data_list.append(
                            TextData(
                                step=event.step, text=text, wall_time=event.wall_time
                            )
                        )
        except Exception:
            pass
        return text_data_list

    def export_scalar_to_text(self, tag: str) -> str:
        """Export scalar data to text format (iteration, value)."""
        scalar_data = self.get_scalar_data(tag)
        lines = []
        for data in scalar_data:
            lines.append(f"{data.step}\t{data.value}")
        return "\n".join(lines)

    def export_image(self, tag: str, step: int) -> bytes | None:
        """Export a specific image by tag and step."""
        image_data = self.get_image_data(tag)
        for data in image_data:
            if data.step == step:
                return data.encoded_image_string
        return None

    def export_histogram_to_text(self, tag: str) -> str:
        """Export histogram data to text format."""
        histogram_data = self.get_histogram_data(tag)
        lines = []
        for data in histogram_data:
            # Create a simple text representation
            lines.append(f"Step: {data.step}")
            lines.append(f"Min: {data.min}, Max: {data.max}")
            lines.append(f"Count: {data.num}, Sum: {data.sum}")
            lines.append("Buckets:")
            for i, (limit, count) in enumerate(zip(data.bucket_limit, data.bucket)):
                lines.append(f"  [{limit:.6f}]: {count}")
            lines.append("")  # Empty line between steps
        return "\n".join(lines)

    def export_audio(self, tag: str, step: int) -> tuple[bytes, str] | None:
        """Export a specific audio by tag and step. Returns (audio_bytes, content_type)."""
        audio_data = self.get_audio_data(tag)
        for data in audio_data:
            if data.step == step:
                return data.encoded_audio_string, data.content_type
        return None

    def export_text(self, tag: str, step: int) -> str | None:
        """Export a specific text by tag and step."""
        text_data = self.get_text_data(tag)
        for data in text_data:
            if data.step == step:
                return data.text
        return None

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

    def get_image_extension(self, image_bytes: bytes) -> str:
        """Determine image extension from bytes."""
        if image_bytes.startswith(b"\x89PNG"):
            return "png"
        elif image_bytes.startswith(b"\xff\xd8\xff"):
            return "jpg"
        else:
            return "bin"

    def list_all_content(self) -> dict[str, list[str]]:
        """List all content organized by type."""
        return {
            "scalars": self.list_scalars(),
            "images": self.list_images(),
            "histograms": self.list_histograms(),
            "audio": self.list_audio(),
            "text": self.list_text(),
            "tensors": self.list_tensors(),
        }

    def get_virtual_paths(self, digits: int = 6) -> list[str]:
        """Get all virtual paths that would exist in the filesystem."""
        paths = []

        # Scalar paths
        for tag in self.list_scalars():
            safe_tag = tag.replace("/", "_")
            paths.append(f"scalars/{safe_tag}.txt")

        # Image paths
        image_tags = self.list_images()
        image_iterator = (
            tqdm(image_tags, desc="Processing images", leave=False)
            if self.show_progress
            else image_tags
        )
        for tag in image_iterator:
            safe_tag = tag.replace("/", "_")
            image_data = self.get_image_data(tag)
            for data in image_data:
                ext = self.get_image_extension(data.encoded_image_string)
                padded_step = str(data.step).zfill(digits)
                paths.append(f"images/{safe_tag}/{padded_step}.{ext}")

        # Histogram paths (as text files)
        for tag in self.list_histograms():
            safe_tag = tag.replace("/", "_")
            paths.append(f"histograms/{safe_tag}.txt")

        # Audio paths
        audio_tags = self.list_audio()
        audio_iterator = (
            tqdm(audio_tags, desc="Processing audio", leave=False)
            if self.show_progress
            else audio_tags
        )
        for tag in audio_iterator:
            safe_tag = tag.replace("/", "_")
            audio_data = self.get_audio_data(tag)
            for audio_item in audio_data:
                ext = self.get_audio_extension(audio_item.content_type)
                padded_step = str(audio_item.step).zfill(digits)
                paths.append(f"audio/{safe_tag}/{padded_step}.{ext}")

        # Text paths
        text_tags = self.list_text()
        text_iterator = (
            tqdm(text_tags, desc="Processing text", leave=False)
            if self.show_progress
            else text_tags
        )
        for tag in text_iterator:
            safe_tag = tag.replace("/", "_")
            text_data = self.get_text_data(tag)
            for text_item in text_data:
                padded_step = str(text_item.step).zfill(digits)
                paths.append(f"text/{safe_tag}/{padded_step}.txt")

        # Add directories
        paths.append("scalars/")
        paths.append("images/")
        paths.append("histograms/")
        paths.append("audio/")
        paths.append("text/")

        image_tags = self.list_images()
        for tag in image_tags:
            safe_tag = tag.replace("/", "_")
            paths.append(f"images/{safe_tag}/")

        audio_tags = self.list_audio()
        for tag in audio_tags:
            safe_tag = tag.replace("/", "_")
            paths.append(f"audio/{safe_tag}/")

        text_tags = self.list_text()
        for tag in text_tags:
            safe_tag = tag.replace("/", "_")
            paths.append(f"text/{safe_tag}/")

        return sorted(set(paths))

    def extract_all_to_directory(
        self, output_dir: str, sort_scalars: bool = True, digits: int = 6
    ):
        """Extract all data to a directory structure."""
        from pathlib import Path

        logger.debug(f"Starting extraction to directory: {output_dir}")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Output directory created: {output_path}")

        # Create subdirectories
        logger.debug("Creating subdirectories")
        scalars_dir = output_path / "scalars"
        images_dir = output_path / "images"
        histograms_dir = output_path / "histograms"
        audio_dir = output_path / "audio"
        text_dir = output_path / "text"

        scalars_dir.mkdir(exist_ok=True)
        images_dir.mkdir(exist_ok=True)
        histograms_dir.mkdir(exist_ok=True)
        audio_dir.mkdir(exist_ok=True)
        text_dir.mkdir(exist_ok=True)
        logger.debug("All subdirectories created")

        scalar_files = {}

        # Get content counts for debug info
        content = self.list_all_content()
        logger.debug(f"Content found - scalars: {len(content['scalars'])}, images: {len(content['images'])}, histograms: {len(content['histograms'])}, audio: {len(content['audio'])}, text: {len(content['text'])}, tensors: {len(content['tensors'])}")

        # Extract scalars - append data as we go
        logger.debug("Starting scalar extraction")
        scalar_tags = self.list_scalars()
        logger.debug(f"Found {len(scalar_tags)} scalar tags: {scalar_tags}")
        if self.show_progress:
            logger.info(f"Extracting {len(scalar_tags)} scalar tags...")
        scalar_iterator = (
            tqdm(scalar_tags, desc="Extracting scalars", leave=False)
            if self.show_progress
            else scalar_tags
        )
        for i, tag in enumerate(scalar_iterator):
            logger.debug(f"Processing scalar tag {i+1}/{len(scalar_tags)}: {tag}")
            safe_tag = tag.replace("/", "_")
            scalar_file = scalars_dir / f"{safe_tag}.txt"
            scalar_data = self.get_scalar_data(tag)
            logger.debug(f"Retrieved {len(scalar_data)} data points for tag {tag}")

            with scalar_file.open("w") as f:
                for data in scalar_data:
                    f.write(f"{data.step}\t{data.value}\n")

            if sort_scalars:
                scalar_files[scalar_file] = True
            logger.debug(f"Completed scalar tag: {tag}")
        logger.debug("Scalar extraction completed")

        # Extract images
        logger.debug("Starting image extraction")
        image_tags = self.list_images()
        logger.debug(f"Found {len(image_tags)} image tags: {image_tags}")
        if self.show_progress and image_tags:
            logger.info(f"Extracting {len(image_tags)} image tags...")
        image_iterator = (
            tqdm(image_tags, desc="Extracting images", leave=False)
            if self.show_progress
            else image_tags
        )
        for i, tag in enumerate(image_iterator):
            logger.debug(f"Processing image tag {i+1}/{len(image_tags)}: {tag}")
            safe_tag = tag.replace("/", "_")
            tag_dir = images_dir / safe_tag
            tag_dir.mkdir(exist_ok=True)

            image_data = self.get_image_data(tag)
            logger.debug(f"Retrieved {len(image_data)} images for tag {tag}")
            for j, image_item in enumerate(image_data):
                logger.debug(f"Processing image {j+1}/{len(image_data)} for tag {tag}, step {image_item.step}")
                ext = self.get_image_extension(image_item.encoded_image_string)
                padded_step = str(image_item.step).zfill(digits)
                image_file = tag_dir / f"{padded_step}.{ext}"
                with image_file.open("wb") as f:
                    f.write(image_item.encoded_image_string)
                logger.debug(f"Wrote image file: {image_file}")
            logger.debug(f"Completed image tag: {tag}")
        logger.debug("Image extraction completed")

        # Extract histograms
        logger.debug("Starting histogram extraction")
        histogram_tags = self.list_histograms()
        logger.debug(f"Found {len(histogram_tags)} histogram tags: {histogram_tags}")
        if self.show_progress and histogram_tags:
            logger.info(f"Extracting {len(histogram_tags)} histogram tags...")
        histogram_iterator = (
            tqdm(histogram_tags, desc="Extracting histograms", leave=False)
            if self.show_progress
            else histogram_tags
        )
        for i, tag in enumerate(histogram_iterator):
            logger.debug(f"Processing histogram tag {i+1}/{len(histogram_tags)}: {tag}")
            safe_tag = tag.replace("/", "_")
            histogram_file = histograms_dir / f"{safe_tag}.txt"
            histogram_text = self.export_histogram_to_text(tag)
            with histogram_file.open("w") as f:
                f.write(histogram_text)
            logger.debug(f"Completed histogram tag: {tag}")
        logger.debug("Histogram extraction completed")

        # Extract audio
        logger.debug("Starting audio extraction")
        audio_tags = self.list_audio()
        logger.debug(f"Found {len(audio_tags)} audio tags: {audio_tags}")
        if self.show_progress and audio_tags:
            logger.info(f"Extracting {len(audio_tags)} audio tags...")
        audio_iterator = (
            tqdm(audio_tags, desc="Extracting audio", leave=False)
            if self.show_progress
            else audio_tags
        )
        for i, tag in enumerate(audio_iterator):
            logger.debug(f"Processing audio tag {i+1}/{len(audio_tags)}: {tag}")
            safe_tag = tag.replace("/", "_")
            tag_dir = audio_dir / safe_tag
            tag_dir.mkdir(exist_ok=True)

            audio_data = self.get_audio_data(tag)
            logger.debug(f"Retrieved {len(audio_data)} audio files for tag {tag}")
            for j, audio_item in enumerate(audio_data):
                logger.debug(f"Processing audio {j+1}/{len(audio_data)} for tag {tag}, step {audio_item.step}")
                ext = self.get_audio_extension(audio_item.content_type)
                padded_step = str(audio_item.step).zfill(digits)
                audio_file = tag_dir / f"{padded_step}.{ext}"
                with audio_file.open("wb") as f:
                    f.write(audio_item.encoded_audio_string)
                logger.debug(f"Wrote audio file: {audio_file}")
            logger.debug(f"Completed audio tag: {tag}")
        logger.debug("Audio extraction completed")

        # Extract text
        logger.debug("Starting text extraction")
        text_tags = self.list_text()
        logger.debug(f"Found {len(text_tags)} text tags: {text_tags}")
        if self.show_progress and text_tags:
            logger.info(f"Extracting {len(text_tags)} text tags...")
        text_iterator = (
            tqdm(text_tags, desc="Extracting text", leave=False)
            if self.show_progress
            else text_tags
        )
        for i, tag in enumerate(text_iterator):
            logger.debug(f"Processing text tag {i+1}/{len(text_tags)}: {tag}")
            safe_tag = tag.replace("/", "_")
            tag_dir = text_dir / safe_tag
            tag_dir.mkdir(exist_ok=True)

            text_data = self.get_text_data(tag)
            logger.debug(f"Retrieved {len(text_data)} text items for tag {tag}")
            for j, text_item in enumerate(text_data):
                logger.debug(f"Processing text {j+1}/{len(text_data)} for tag {tag}, step {text_item.step}")
                padded_step = str(text_item.step).zfill(digits)
                text_file = tag_dir / f"{padded_step}.txt"
                with text_file.open("w", encoding="utf-8") as f:
                    f.write(text_item.text)
                logger.debug(f"Wrote text file: {text_file}")
            logger.debug(f"Completed text tag: {tag}")
        logger.debug("Text extraction completed")

        # Sort scalar files if requested
        if sort_scalars and scalar_files:
            logger.debug(f"Starting scalar file sorting for {len(scalar_files)} files")
            if self.show_progress:
                logger.info("Sorting scalar files by iteration number...")
            self._sort_scalar_files(scalar_files.keys())
            logger.debug("Scalar file sorting completed")
        
        logger.debug("All extraction completed successfully")

    def _sort_scalar_files(self, scalar_files):
        """Sort scalar files by iteration number (first column)."""
        scalar_file_list = list(scalar_files)
        file_iterator = (
            tqdm(scalar_file_list, desc="Sorting scalar files", leave=False)
            if self.show_progress
            else scalar_file_list
        )
        for file_path in file_iterator:
            # Read the file
            with file_path.open() as f:
                lines = f.readlines()

            # Parse and sort by step (first column)
            data_points = []
            for line in lines:
                if line.strip():
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        try:
                            step = int(parts[0])
                            value = float(parts[1])
                            data_points.append((step, value))
                        except ValueError:
                            continue

            # Sort by step
            data_points.sort(key=lambda x: x[0])

            # Write back sorted data
            with file_path.open("w") as f:
                for step, value in data_points:
                    f.write(f"{step}\t{value}\n")
