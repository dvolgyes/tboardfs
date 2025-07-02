"""Base exporter class for TensorBoard data types."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol
from tensorboard.compat.proto import event_pb2
from loguru import logger

from ..core.unified_virtual_paths import VirtualPathBuilder
from ..core.constants import DEFAULT_DIGITS


class SummaryValue(Protocol):
    """Protocol for TensorBoard summary values."""

    tag: str

    def HasField(self, field_name: str) -> bool:
        """Check if a field is present in the summary value."""
        ...


class BaseExporter(ABC):
    """Abstract base class for TensorBoard data exporters."""

    def __init__(self, output_path: Path, digits: int = DEFAULT_DIGITS):
        """Initialize base exporter.

        Args:
            output_path: Directory to save exported data
            digits: Number of digits for zero-padding step numbers
        """
        self.output_path = output_path
        self.digits = digits
        self.path_builder = VirtualPathBuilder(digits=digits)

    @abstractmethod
    def save_data(
        self, event: event_pb2.Event, value: SummaryValue, **kwargs: Any
    ) -> None:
        """Save data from a TensorBoard event.

        Args:
            event: TensorBoard event containing metadata
            value: Summary value containing the actual data
            **kwargs: Additional exporter-specific parameters
        """
        pass

    def _sanitize_tag(self, tag: str) -> str:
        """Sanitize tag name for filesystem safety."""
        return self.path_builder.sanitize_tag(tag)

    def _ensure_directory_exists(self, directory: Path) -> None:
        """Create directory if it doesn't exist."""
        directory.mkdir(parents=True, exist_ok=True)

    def _format_step(self, step: int) -> str:
        """Format step number with zero padding."""
        return self.path_builder.format_step(step)

    def _log_save_error(self, tag: str, error: Exception) -> None:
        """Log save error with context."""
        logger.warning(f"Failed to save data for tag '{tag}': {error}")
