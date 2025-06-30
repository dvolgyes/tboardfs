"""TensorBoard event file parser.

This parser supports TensorFlow v2 event file format where all data types
(scalars, images, histograms, audio) are stored as tensors. Text data is
identified by checking tensor dtype == 7 (DT_STRING).

This implementation uses EventFileLoader for efficient streaming instead of
EventAccumulator which loads everything into memory.
"""

from tboardfs.efficient_parser import EfficientTensorBoardParser
from tboardfs.core.data_types import (
    ScalarData,
    ImageData,
    HistogramData,
    AudioData,
    TextData,
)
from loguru import logger
import magic


class TensorBoardParser:
    """Parser for TensorBoard event files.

    This is a compatibility wrapper around EfficientTensorBoardParser
    to maintain the existing API while using the efficient implementation.
    """

    def __init__(self, event_file_path: str, show_progress: bool = False):
        """Initialize parser with event file path."""
        logger.debug(
            f"Initializing TensorBoardParser (efficient) for file: {event_file_path}"
        )
        self.event_file_path = event_file_path
        self.show_progress = show_progress
        self._efficient_parser = EfficientTensorBoardParser(
            event_file_path, show_progress
        )
        # For backward compatibility
        self.ea = self._efficient_parser

    def list_scalars(self) -> list[str]:
        """List all scalar tags in the event file.

        Note: In TensorFlow v2 format, scalars may be stored as tensors
        and this list might be empty.
        """
        if hasattr(self, "_efficient_parser"):
            return self._efficient_parser.list_scalars()
        return []

    def list_images(self) -> list[str]:
        """List all image tags in the event file."""
        if hasattr(self, "_efficient_parser"):
            return self._efficient_parser.list_images()
        return []

    def list_histograms(self) -> list[str]:
        """List all histogram tags in the event file."""
        if hasattr(self, "_efficient_parser"):
            return self._efficient_parser.list_histograms()
        return []

    def list_tensors(self) -> list[str]:
        """List all tensor tags in the event file."""
        if hasattr(self, "_efficient_parser"):
            return self._efficient_parser.list_tensors()
        return []

    def list_audio(self) -> list[str]:
        """List all audio tags in the event file."""
        if hasattr(self, "_efficient_parser"):
            return self._efficient_parser.list_audio()
        return []

    def list_text(self) -> list[str]:
        """List all text tags in the event file.

        In TensorFlow v2 format, text is stored as tensors with dtype=7 (DT_STRING).
        This method identifies text tensors by checking their dtype.
        """
        if hasattr(self, "_efficient_parser"):
            return self._efficient_parser.list_text()
        return []

    def get_scalar_data(self, tag: str) -> list[ScalarData]:
        """Get all scalar data for a given tag."""
        if hasattr(self, "_efficient_parser"):
            return list(self._efficient_parser.iterate_scalar_data(tag))
        return []

    def get_image_data(self, tag: str) -> list[ImageData]:
        """Get all image data for a given tag."""
        if hasattr(self, "_efficient_parser"):
            return list(self._efficient_parser.iterate_image_data(tag))
        return []

    def get_histogram_data(self, tag: str) -> list[HistogramData]:
        """Get all histogram data for a given tag."""
        if hasattr(self, "_efficient_parser"):
            return list(self._efficient_parser.iterate_histogram_data(tag))
        return []

    def get_audio_data(self, tag: str) -> list[AudioData]:
        """Get all audio data for a given tag."""
        if hasattr(self, "_efficient_parser"):
            return list(self._efficient_parser.iterate_audio_data(tag))
        return []

    def get_text_data(self, tag: str) -> list[TextData]:
        """Get all text data for a given tag."""
        if hasattr(self, "_efficient_parser"):
            return list(self._efficient_parser.iterate_text_data(tag))
        return []

    def export_scalar_to_text(self, tag: str) -> str:
        """Export scalar data to text format (iteration, value)."""
        return self._efficient_parser.export_scalar_to_text(tag)

    def export_image(self, tag: str, step: int) -> bytes | None:
        """Export a specific image by tag and step."""
        for data in self._efficient_parser.iterate_image_data(tag):
            if data.step == step:
                return data.encoded_image_string
        return None

    def export_histogram_to_text(self, tag: str) -> str:
        """Export histogram data to text format."""
        lines = []
        for data in self._efficient_parser.iterate_histogram_data(tag):
            lines.append(f"Step: {data.step}")
            lines.append(f"Min: {data.min}, Max: {data.max}")
            lines.append(f"Count: {data.num}, Sum: {data.sum}")
            lines.append("Buckets:")
            for limit, count in zip(data.bucket_limit, data.bucket):
                lines.append(f"  [{limit:.6f}]: {count}")
            lines.append("")  # Empty line between steps
        return "\n".join(lines)

    def export_audio(self, tag: str, step: int) -> tuple[bytes, str] | None:
        """Export a specific audio by tag and step. Returns (audio_bytes, content_type)."""
        for data in self._efficient_parser.iterate_audio_data(tag):
            if data.step == step:
                return data.encoded_audio_string, data.content_type
        return None

    def export_text(self, tag: str, step: int) -> str | None:
        """Export a specific text by tag and step."""
        for data in self._efficient_parser.iterate_text_data(tag):
            if data.step == step:
                return data.text
        return None

    def get_audio_extension(self, content_type: str) -> str:
        """Determine audio extension from content type."""
        if hasattr(self, "_efficient_parser"):
            return self._efficient_parser.get_audio_extension(content_type)
        # Fallback for tests that use __new__
        if "wav" in content_type:
            return "wav"
        elif "mp3" in content_type:
            return "mp3"
        elif "ogg" in content_type:
            return "ogg"
        else:
            return "audio"

    def get_image_extension(self, image_bytes: bytes) -> str:
        """Determine image extension from bytes using python-magic."""
        if hasattr(self, "_efficient_parser"):
            return self._efficient_parser.get_image_extension(image_bytes)

        # Fallback for tests that use __new__
        mime_type = magic.from_buffer(image_bytes, mime=True)

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

        return mime_to_ext.get(mime_type, "bin")

    def list_all_content(self) -> dict[str, list[str]]:
        """List all content organized by type."""
        return self._efficient_parser.list_all_content()

    def get_virtual_paths(self, digits: int = 6) -> list[str]:
        """Get all virtual paths that would exist in the filesystem."""
        if hasattr(self, "_efficient_parser"):
            return self._efficient_parser.get_virtual_paths(digits)

        # Fallback for tests using mocks - call the individual methods
        paths = []

        # Add directories
        paths.extend(["scalars/", "images/", "histograms/", "audio/", "text/"])

        # Scalar paths
        try:
            for tag in self.list_scalars():
                safe_tag = tag.replace("/", "_")
                paths.append(f"scalars/{safe_tag}.txt")
        except Exception:
            pass

        # Image paths
        try:
            for tag in self.list_images():
                safe_tag = tag.replace("/", "_")
                paths.append(f"images/{safe_tag}/")
                try:
                    image_data = self.get_image_data(tag)
                    for image_item in image_data:
                        ext = self.get_image_extension(image_item.encoded_image_string)
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
                    audio_data = self.get_audio_data(tag)
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
                    text_data = self.get_text_data(tag)
                    for text_item in text_data:
                        padded_step = str(text_item.step).zfill(digits)
                        paths.append(f"text/{safe_tag}/{padded_step}.txt")
                except Exception:
                    pass
        except Exception:
            pass

        return sorted(set(paths))

    def extract_all_to_directory(
        self,
        output_dir: str,
        sort_scalars: bool = True,
        digits: int = 6,
        image_format: str = "jpg",
        image_quality: int = 90,
    ):
        """Extract all data to a directory structure."""
        return self._efficient_parser.extract_all_to_directory(
            output_dir, sort_scalars, digits, image_format, image_quality
        )

    def _sort_scalar_files(self, scalar_files):
        """Sort scalar files by iteration number (first column). For backward compatibility."""
        from pathlib import Path
        from tqdm import tqdm

        scalar_file_list = list(scalar_files)
        file_iterator = (
            tqdm(scalar_file_list, desc="Sorting scalar files", leave=False)
            if self.show_progress
            else scalar_file_list
        )
        for file_path in file_iterator:
            # Read the file
            with Path(file_path).open() as f:
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
            with Path(file_path).open("w") as f:
                for step, value in data_points:
                    f.write(f"{step}\t{value}\n")
