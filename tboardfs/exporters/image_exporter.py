"""Image and video data exporter for TensorBoard events."""

from pathlib import Path
from typing import Any
import io
import sys
from tensorboard.compat.proto import event_pb2
from tensorboard.util import tensor_util
from loguru import logger

from .base_exporter import BaseExporter
from ..core.data_detector import TensorDataDetector
from ..core.constants import (
    TensorFlowDTypes,
    ImageFormats,
    DEFAULT_DIGITS,
)


class ImageExporter(BaseExporter):
    """Export image and video data from TensorBoard events."""

    def __init__(self, output_path: Path, digits: int = DEFAULT_DIGITS):
        """Initialize image exporter."""
        super().__init__(output_path, digits)

    def save_image(
        self,
        event: event_pb2.Event,
        value: Any,
        image_format: str = ImageFormats.DEFAULT_FORMAT,
        image_quality: int = ImageFormats.DEFAULT_QUALITY,
        **kwargs: Any,
    ) -> None:
        """Save image data from a TensorBoard event.

        Args:
            event: TensorBoard event containing the image data
            value: Summary value containing image information
            image_format: Output image format (jpg/png)
            image_quality: JPEG quality setting (1-100)
        """
        tag = value.tag
        safe_tag = self._sanitize_tag(tag)
        tag_dir = self.output_path / "images" / safe_tag
        self._ensure_directory_exists(tag_dir)

        image_byte_list = self._extract_image_bytes(value, tag)

        if image_byte_list:
            self._save_image_files(
                event, tag_dir, image_byte_list, image_format, image_quality
            )
        else:
            logger.warning(
                f"Could not extract image for tag '{tag}' at step {event.step}"
            )

    def save_video(self, event: event_pb2.Event, value: Any, **kwargs: Any) -> None:
        """Save video data from a TensorBoard event.

        Args:
            event: TensorBoard event containing the video data
            value: Summary value containing video information
        """
        tag = value.tag
        safe_tag = self._sanitize_tag(tag)
        tag_dir = self.output_path / "videos" / safe_tag
        self._ensure_directory_exists(tag_dir)

        video_data = self._extract_video_data(value)

        if video_data:
            self._save_video_file(event, tag_dir, video_data)
        else:
            logger.warning(
                f"Could not extract video data for tag '{tag}' at step {event.step}"
            )

    def _extract_image_bytes(self, value: Any, tag: str) -> list[bytes]:
        """Extract image bytes from summary value."""
        image_byte_list = []

        if value.HasField("image"):
            image_byte_list.append(value.image.encoded_image_string)
        elif value.HasField("tensor"):
            if TensorDataDetector.is_encoded_image_tensor(value.tensor, tag):
                decoded_image = TensorDataDetector.decode_image_from_tensor(
                    value.tensor
                )
                if decoded_image:
                    image_byte_list.append(decoded_image)
            elif value.tensor.dtype == TensorFlowDTypes.DT_STRING:
                try:
                    arr = tensor_util.make_ndarray(value.tensor)
                    for item in arr:
                        if isinstance(item, bytes):
                            # Filter out non-image data by checking if it's actually image bytes
                            if TensorDataDetector.is_valid_image_bytes(item):
                                image_byte_list.append(item)
                except Exception as e:
                    logger.warning(
                        f"Could not decode string tensor for tag '{tag}': {e}"
                    )

        return image_byte_list

    def _extract_video_data(self, value: Any) -> bytes | None:
        """Extract video data from summary value."""
        if value.HasField("image"):
            # Video data stored as image (GIF) in TensorBoard
            return value.image.encoded_image_string
        return None

    def _save_image_files(
        self,
        event: event_pb2.Event,
        tag_dir: Path,
        image_byte_list: list[bytes],
        image_format: str,
        image_quality: int,
    ) -> None:
        """Save image files to disk."""
        padded_step = self._format_step(event.step)

        for i, image_bytes in enumerate(image_byte_list):
            # Append index for batches to avoid overwriting
            filename = (
                f"{padded_step}_{i}.{image_format}"
                if len(image_byte_list) > 1
                else f"{padded_step}.{image_format}"
            )
            image_file = tag_dir / filename

            try:
                self._convert_and_save_image(
                    image_bytes, image_file, image_format, image_quality
                )
            except Exception as e:
                logger.warning(f"Failed to save image {image_file}: {e}")

    def _save_video_file(
        self, event: event_pb2.Event, tag_dir: Path, video_data: bytes
    ) -> None:
        """Save video file to disk."""
        padded_step = self._format_step(event.step)

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

    def _convert_and_save_image(
        self,
        image_bytes: bytes,
        output_file: Path,
        image_format: str,
        image_quality: int,
    ) -> None:
        """Convert and save image using PIL."""
        try:
            from PIL import Image

            image = Image.open(io.BytesIO(image_bytes))
            if image.mode == "RGBA":
                image = image.convert("RGB")

            if image_format == "jpg":
                image.save(output_file, format="JPEG", quality=image_quality)
            else:  # png
                image.save(output_file, format="PNG")

        except ImportError:
            logger.error("Pillow (PIL) is not installed. Cannot convert image format.")
            logger.info("Please install it with: pip install Pillow")
            sys.exit(1)

    def _is_video_data(self, image_data: bytes, tag: str) -> bool:
        """Check if image data is actually video data."""
        return TensorDataDetector.is_video_data(image_data, tag)

    # Expose save_data for BaseExporter compatibility
    def save_data(self, event: event_pb2.Event, value: Any, **kwargs: Any) -> None:
        """Save data from a TensorBoard event (dispatches to image or video)."""
        if self._is_video_data(
            value.image.encoded_image_string if value.HasField("image") else b"",
            value.tag,
        ):
            self.save_video(event, value, **kwargs)
        else:
            self.save_image(event, value, **kwargs)
