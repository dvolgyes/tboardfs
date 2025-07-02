"""DataSource abstraction layer for flexible data access.

This module provides an abstraction layer over different data sources to prepare
for virtual filesystem support while maintaining backward compatibility with
existing file-based access patterns.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from collections.abc import Iterator
from collections.abc import Sequence
from tensorboard.compat.proto import event_pb2
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from loguru import logger


class DataSource(ABC):
    """Abstract base class for TensorBoard data sources."""

    @abstractmethod
    def get_event_iterator(self) -> Iterator[event_pb2.Event]:
        """Get iterator over events in this data source.

        Returns:
            Iterator yielding TensorBoard Event protobuf objects
        """
        pass

    @abstractmethod
    def get_source_info(self) -> dict[str, Any]:
        """Get metadata about this data source.

        Returns:
            Dictionary containing source metadata (size, path, etc.)
        """
        pass

    @abstractmethod
    def get_identifier(self) -> str:
        """Get unique identifier for this data source.

        Returns:
            String identifier for this data source
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the data source is available and accessible.

        Returns:
            True if the data source can be accessed, False otherwise
        """
        pass


class FileDataSource(DataSource):
    """File-based data source using EventFileLoader."""

    def __init__(self, file_path: str | Path):
        """Initialize file data source.

        Args:
            file_path: Path to the TensorBoard event file
        """
        self.file_path = Path(file_path)
        self._validate_file()

    def _validate_file(self) -> None:
        """Validate that the file exists and is accessible."""
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"TensorBoard event file not found: {self.file_path}"
            )

        if not self.file_path.is_file():
            raise ValueError(f"Path is not a file: {self.file_path}")

    def get_event_iterator(self) -> Iterator[event_pb2.Event]:
        """Get iterator over events using EventFileLoader."""
        loader = EventFileLoader(str(self.file_path))
        try:
            yield from loader.Load()
        except Exception as e:
            logger.error(f"Failed to load events from {self.file_path}: {e}")
            raise

    def get_source_info(self) -> dict[str, Any]:
        """Get file metadata information."""
        try:
            stat = self.file_path.stat()
            return {
                "type": "file",
                "path": str(self.file_path),
                "size": stat.st_size,
                "size_mb": stat.st_size / (1024 * 1024),
                "modified_time": stat.st_mtime,
            }
        except OSError as e:
            logger.warning(f"Could not get file stats for {self.file_path}: {e}")
            return {
                "type": "file",
                "path": str(self.file_path),
                "size": 0,
                "size_mb": 0.0,
            }

    def get_identifier(self) -> str:
        """Get file path as identifier."""
        return str(self.file_path)

    def is_available(self) -> bool:
        """Check if file exists and is readable."""
        return self.file_path.exists() and self.file_path.is_file()


class CompositeDataSource(DataSource):
    """Composite data source combining multiple data sources."""

    def __init__(self, sources: Sequence[DataSource]):
        """Initialize composite data source.

        Args:
            sources: Sequence of data sources to combine
        """
        self.sources = list(sources)
        if not self.sources:
            raise ValueError("CompositeDataSource requires at least one source")

    def get_event_iterator(self) -> Iterator[event_pb2.Event]:
        """Get iterator over events from all sources sequentially."""
        for source in self.sources:
            if source.is_available():
                try:
                    yield from source.get_event_iterator()
                except Exception as e:
                    logger.error(
                        f"Failed to read from source {source.get_identifier()}: {e}"
                    )
                    # Continue with other sources
            else:
                logger.warning(
                    f"Skipping unavailable source: {source.get_identifier()}"
                )

    def get_source_info(self) -> dict[str, Any]:
        """Get aggregated information from all sources."""
        total_size = 0
        source_count = 0
        source_details = []

        for source in self.sources:
            if source.is_available():
                info = source.get_source_info()
                source_details.append(info)
                total_size += info.get("size", 0)
                source_count += 1

        return {
            "type": "composite",
            "source_count": source_count,
            "total_sources": len(self.sources),
            "total_size": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "sources": source_details,
        }

    def get_identifier(self) -> str:
        """Get composite identifier listing all sources."""
        identifiers = [source.get_identifier() for source in self.sources]
        return (
            f"composite({len(identifiers)} sources): {', '.join(identifiers[:3])}"
            + (f" ... and {len(identifiers) - 3} more" if len(identifiers) > 3 else "")
        )

    def is_available(self) -> bool:
        """Check if at least one source is available."""
        return any(source.is_available() for source in self.sources)


class DataSourceFactory:
    """Factory for creating DataSource instances from various inputs."""

    @staticmethod
    def from_path(path: str | Path) -> DataSource:
        """Create DataSource from file or directory path.

        Args:
            path: File path or directory path

        Returns:
            FileDataSource for files, CompositeDataSource for directories
        """
        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if path_obj.is_file():
            return FileDataSource(path_obj)
        elif path_obj.is_dir():
            return DataSourceFactory.from_directory(path_obj)
        else:
            raise ValueError(f"Path is neither file nor directory: {path}")

    @staticmethod
    def from_directory(directory: Path) -> CompositeDataSource:
        """Create CompositeDataSource from directory containing event files.

        Args:
            directory: Directory containing TensorBoard event files

        Returns:
            CompositeDataSource with all event files in directory
        """
        from .file_utils import get_event_files_sorted

        event_files = get_event_files_sorted(directory)
        if not event_files:
            raise ValueError(f"No TensorBoard event files found in {directory}")

        sources = [FileDataSource(file_path) for file_path in event_files]
        return CompositeDataSource(sources)

    @staticmethod
    def from_files(file_paths: Sequence[str | Path]) -> CompositeDataSource:
        """Create CompositeDataSource from multiple file paths.

        Args:
            file_paths: Sequence of file paths

        Returns:
            CompositeDataSource with all specified files
        """
        if not file_paths:
            raise ValueError("At least one file path required")

        sources = [FileDataSource(file_path) for file_path in file_paths]
        return CompositeDataSource(sources)
