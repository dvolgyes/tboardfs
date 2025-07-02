"""Text data exporter for TensorBoard events."""

from pathlib import Path
from typing import Any
from tensorboard.compat.proto import event_pb2
from tensorboard.util import tensor_util
from loguru import logger

from .base_exporter import BaseExporter


class TextExporter(BaseExporter):
    """Export text data from TensorBoard events."""

    def __init__(self, output_path: Path, digits: int = 6):
        """Initialize text exporter."""
        super().__init__(output_path, digits)

    def save_data(self, event: event_pb2.Event, value: Any, **kwargs: Any) -> None:
        """Save text data from a TensorBoard event.

        Args:
            event: TensorBoard event containing the text data
            value: Summary value containing text information
        """
        tag = value.tag
        safe_tag = self._sanitize_tag(tag)
        tag_dir = self.output_path / "text" / safe_tag
        self._ensure_directory_exists(tag_dir)

        try:
            # Decode text from tensor
            text_value = tensor_util.make_ndarray(value.tensor)
            if text_value.size > 0:
                text = text_value.item() if text_value.ndim == 0 else text_value[0]
                if isinstance(text, bytes):
                    text = text.decode("utf-8", errors="replace")
                else:
                    text = str(text)

                padded_step = self._format_step(event.step)
                text_file = tag_dir / f"{padded_step}.txt"

                with text_file.open("w", encoding="utf-8") as f:
                    f.write(text)

                logger.debug(f"Saved text data to {text_file}")
            else:
                logger.warning(f"Empty text tensor for tag {tag}")

        except Exception as e:
            logger.error(f"Failed to save text for tag {tag}: {e}")
