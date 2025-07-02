"""Base exporter class for TensorBoard data types."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from tensorboard.compat.proto import event_pb2
from loguru import logger


class BaseExporter(ABC):
    """Abstract base class for TensorBoard data exporters."""

    def __init__(self, output_path: Path, digits: int = 6):
        """Initialize base exporter.

        Args:
            output_path: Directory to save exported data
            digits: Number of digits for zero-padding step numbers
        """
        self.output_path = output_path
        self.digits = digits

    @abstractmethod
    def save_data(self, event: event_pb2.Event, value: Any, **kwargs: Any) -> None:
        """Save data from a TensorBoard event."""
        pass

    def _sanitize_tag(self, tag: str) -> str:
        """Sanitize tag name for filesystem safety."""
        return tag.replace("/", "_")

    def _ensure_directory_exists(self, directory: Path) -> None:
        """Create directory if it doesn't exist."""
        directory.mkdir(parents=True, exist_ok=True)

    def _format_step(self, step: int) -> str:
        """Format step number with zero padding."""
        return str(step).zfill(self.digits)

    def _log_save_error(self, tag: str, error: Exception) -> None:
        """Log save error with context."""
        logger.warning(f"Failed to save data for tag '{tag}': {error}")
