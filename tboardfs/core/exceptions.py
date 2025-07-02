"""Custom exception hierarchy for tboardfs.

This module defines a comprehensive exception hierarchy to standardize
error handling patterns throughout the codebase.
"""

from pathlib import Path
from typing import Any


class TBoardFSError(Exception):
    """Base exception for all tboardfs errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        """Initialize exception with message and optional details.

        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return formatted error message."""
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


class DataSourceError(TBoardFSError):
    """Base exception for data source related errors."""

    pass


class FileNotFoundError(DataSourceError):
    """Raised when a required file cannot be found."""

    def __init__(self, file_path: str | Path, message: str | None = None):
        """Initialize with file path and optional message."""
        self.file_path = Path(file_path)
        msg = message or f"File not found: {self.file_path}"
        super().__init__(msg, {"file_path": str(self.file_path)})


class InvalidFileFormatError(DataSourceError):
    """Raised when file format is invalid or unsupported."""

    def __init__(self, file_path: str | Path, expected_format: str | None = None):
        """Initialize with file path and expected format."""
        self.file_path = Path(file_path)
        self.expected_format = expected_format
        msg = f"Invalid file format: {self.file_path}"
        if expected_format:
            msg += f" (expected: {expected_format})"
        super().__init__(
            msg, {"file_path": str(self.file_path), "expected_format": expected_format}
        )


class CorruptedDataError(DataSourceError):
    """Raised when data is corrupted or unreadable."""

    def __init__(self, source: str, message: str | None = None):
        """Initialize with data source and optional message."""
        self.source = source
        msg = message or f"Corrupted data in: {source}"
        super().__init__(msg, {"source": source})


class ParsingError(TBoardFSError):
    """Base exception for parsing related errors."""

    pass


class EventParsingError(ParsingError):
    """Raised when TensorBoard event parsing fails."""

    def __init__(
        self,
        event_file: str | Path,
        step: int | None = None,
        tag: str | None = None,
        cause: Exception | None = None,
    ):
        """Initialize with event file details."""
        self.event_file = Path(event_file)
        self.step = step
        self.tag = tag
        self.cause = cause

        msg = f"Failed to parse event from {self.event_file}"
        if step is not None:
            msg += f" at step {step}"
        if tag:
            msg += f" for tag '{tag}'"
        if cause:
            msg += f": {cause}"

        details: dict[str, Any] = {"event_file": str(self.event_file)}
        if step is not None:
            details["step"] = step
        if tag:
            details["tag"] = tag
        if cause:
            details["cause"] = str(cause)

        super().__init__(msg, details)


class TensorDecodingError(ParsingError):
    """Raised when tensor data cannot be decoded."""

    def __init__(
        self, tag: str, tensor_type: str | None = None, cause: Exception | None = None
    ):
        """Initialize with tensor details."""
        self.tag = tag
        self.tensor_type = tensor_type
        self.cause = cause

        msg = f"Failed to decode tensor for tag '{tag}'"
        if tensor_type:
            msg += f" (type: {tensor_type})"
        if cause:
            msg += f": {cause}"

        details: dict[str, Any] = {"tag": tag}
        if tensor_type:
            details["tensor_type"] = tensor_type
        if cause:
            details["cause"] = str(cause)

        super().__init__(msg, details)


class ExportError(TBoardFSError):
    """Base exception for export related errors."""

    pass


class DirectoryCreationError(ExportError):
    """Raised when directory creation fails."""

    def __init__(self, directory: str | Path, cause: Exception | None = None):
        """Initialize with directory path and cause."""
        self.directory = Path(directory)
        self.cause = cause

        msg = f"Failed to create directory: {self.directory}"
        if cause:
            msg += f": {cause}"

        details = {"directory": str(self.directory)}
        if cause:
            details["cause"] = str(cause)

        super().__init__(msg, details)


class FileWriteError(ExportError):
    """Raised when file writing fails."""

    def __init__(self, file_path: str | Path, cause: Exception | None = None):
        """Initialize with file path and cause."""
        self.file_path = Path(file_path)
        self.cause = cause

        msg = f"Failed to write file: {self.file_path}"
        if cause:
            msg += f": {cause}"

        details = {"file_path": str(self.file_path)}
        if cause:
            details["cause"] = str(cause)

        super().__init__(msg, details)


class UnsupportedFormatError(ExportError):
    """Raised when export format is not supported."""

    def __init__(
        self,
        format_type: str,
        data_type: str | None = None,
        supported_formats: list[str] | None = None,
    ):
        """Initialize with format details."""
        self.format_type = format_type
        self.data_type = data_type
        self.supported_formats = supported_formats or []

        msg = f"Unsupported format: {format_type}"
        if data_type:
            msg += f" for {data_type} data"
        if supported_formats:
            msg += f" (supported: {', '.join(supported_formats)})"

        details: dict[str, Any] = {"format_type": format_type}
        if data_type:
            details["data_type"] = data_type
        if supported_formats:
            details["supported_formats"] = supported_formats

        super().__init__(msg, details)


class ValidationError(TBoardFSError):
    """Base exception for validation errors."""

    pass


class InvalidPathError(ValidationError):
    """Raised when path is invalid or unsafe."""

    def __init__(self, path: str | Path, reason: str | None = None):
        """Initialize with path and reason."""
        self.path = Path(path)
        self.reason = reason

        msg = f"Invalid path: {self.path}"
        if reason:
            msg += f" ({reason})"

        details = {"path": str(self.path)}
        if reason:
            details["reason"] = reason

        super().__init__(msg, details)


class InvalidConfigurationError(ValidationError):
    """Raised when configuration is invalid."""

    def __init__(self, config_key: str, config_value: Any, reason: str | None = None):
        """Initialize with configuration details."""
        self.config_key = config_key
        self.config_value = config_value
        self.reason = reason

        msg = f"Invalid configuration: {config_key}={config_value}"
        if reason:
            msg += f" ({reason})"

        details = {"config_key": config_key, "config_value": config_value}
        if reason:
            details["reason"] = reason

        super().__init__(msg, details)


class DataValidationError(ValidationError):
    """Raised when data validation fails."""

    def __init__(
        self,
        data_type: str,
        field: str | None = None,
        expected: Any | None = None,
        actual: Any | None = None,
    ):
        """Initialize with validation details."""
        self.data_type = data_type
        self.field = field
        self.expected = expected
        self.actual = actual

        msg = f"Data validation failed for {data_type}"
        if field:
            msg += f".{field}"
        if expected is not None and actual is not None:
            msg += f": expected {expected}, got {actual}"

        details = {"data_type": data_type}
        if field:
            details["field"] = field
        if expected is not None:
            details["expected"] = expected
        if actual is not None:
            details["actual"] = actual

        super().__init__(msg, details)


class ProcessingError(TBoardFSError):
    """Base exception for processing pipeline errors."""

    pass


class PipelineStageError(ProcessingError):
    """Raised when a pipeline stage fails."""

    def __init__(
        self,
        stage_name: str,
        input_data: Any | None = None,
        cause: Exception | None = None,
    ):
        """Initialize with stage details."""
        self.stage_name = stage_name
        self.input_data = input_data
        self.cause = cause

        msg = f"Pipeline stage '{stage_name}' failed"
        if cause:
            msg += f": {cause}"

        details = {"stage_name": stage_name}
        if input_data is not None:
            details["input_data_type"] = type(input_data).__name__
        if cause:
            details["cause"] = str(cause)

        super().__init__(msg, details)


class ResourceError(TBoardFSError):
    """Base exception for resource-related errors."""

    pass


class OutOfMemoryError(ResourceError):
    """Raised when operation runs out of memory."""

    def __init__(self, operation: str, required_memory: int | None = None):
        """Initialize with operation details."""
        self.operation = operation
        self.required_memory = required_memory

        msg = f"Out of memory during {operation}"
        if required_memory:
            msg += f" (required: {required_memory} bytes)"

        details: dict[str, Any] = {"operation": operation}
        if required_memory:
            details["required_memory"] = required_memory

        super().__init__(msg, details)


class TimeoutError(ResourceError):
    """Raised when operation times out."""

    def __init__(self, operation: str, timeout_seconds: float | None = None):
        """Initialize with timeout details."""
        self.operation = operation
        self.timeout_seconds = timeout_seconds

        msg = f"Operation '{operation}' timed out"
        if timeout_seconds:
            msg += f" after {timeout_seconds}s"

        details: dict[str, Any] = {"operation": operation}
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds

        super().__init__(msg, details)
