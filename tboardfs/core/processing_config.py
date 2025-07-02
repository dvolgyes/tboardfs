"""Processing configuration management for tboardfs.

This module contains configuration classes for data processing operations,
including memory management, validation limits, and progress tracking options.
"""

from dataclasses import dataclass, field
from typing import Any

from .constants import (
    ExportSettings,
    ValidationLimits,
)
from .exceptions import InvalidConfigurationError


@dataclass
class MemoryConfig:
    """Configuration for memory management during processing."""

    max_events_in_memory: int = ExportSettings.MAX_EVENTS_IN_MEMORY
    chunk_size_bytes: int = ExportSettings.CHUNK_SIZE_BYTES
    max_buffer_size_mb: int = 100  # Maximum buffer size in MB
    enable_memory_monitoring: bool = False
    memory_warning_threshold: float = 0.8  # Warn at 80% memory usage

    def validate(self) -> None:
        """Validate memory configuration."""
        if self.max_events_in_memory < 1:
            raise InvalidConfigurationError(
                "max_events_in_memory", self.max_events_in_memory, "must be at least 1"
            )

        if self.chunk_size_bytes < 1024:  # 1KB minimum
            raise InvalidConfigurationError(
                "chunk_size_bytes", self.chunk_size_bytes, "must be at least 1024 bytes"
            )

        if self.max_buffer_size_mb < 1:
            raise InvalidConfigurationError(
                "max_buffer_size_mb", self.max_buffer_size_mb, "must be at least 1 MB"
            )

        if not (0.1 <= self.memory_warning_threshold <= 1.0):
            raise InvalidConfigurationError(
                "memory_warning_threshold",
                self.memory_warning_threshold,
                "must be between 0.1 and 1.0",
            )


@dataclass
class ValidationConfig:
    """Configuration for data validation limits."""

    max_tag_length: int = ValidationLimits.MAX_TAG_LENGTH
    max_filename_length: int = ValidationLimits.MAX_FILENAME_LENGTH
    max_tensor_size_bytes: int = ValidationLimits.MAX_TENSOR_SIZE_BYTES
    max_string_length: int = ValidationLimits.MAX_STRING_LENGTH
    max_tags_per_type: int = ValidationLimits.MAX_TAGS_PER_TYPE
    max_steps_per_tag: int = ValidationLimits.MAX_STEPS_PER_TAG

    # Additional validation options
    strict_mode: bool = False  # Fail on any validation error
    skip_invalid_data: bool = True  # Skip invalid data instead of failing
    validate_tensor_shapes: bool = True  # Validate tensor dimensions
    validate_data_types: bool = True  # Validate data type consistency

    def validate(self) -> None:
        """Validate validation configuration."""
        if self.max_tag_length < 1:
            raise InvalidConfigurationError(
                "max_tag_length", self.max_tag_length, "must be at least 1"
            )

        if self.max_filename_length < 1:
            raise InvalidConfigurationError(
                "max_filename_length", self.max_filename_length, "must be at least 1"
            )

        if self.max_tensor_size_bytes < 1:
            raise InvalidConfigurationError(
                "max_tensor_size_bytes",
                self.max_tensor_size_bytes,
                "must be at least 1",
            )

        if self.max_string_length < 1:
            raise InvalidConfigurationError(
                "max_string_length", self.max_string_length, "must be at least 1"
            )

        if self.max_tags_per_type < 1:
            raise InvalidConfigurationError(
                "max_tags_per_type", self.max_tags_per_type, "must be at least 1"
            )

        if self.max_steps_per_tag < 1:
            raise InvalidConfigurationError(
                "max_steps_per_tag", self.max_steps_per_tag, "must be at least 1"
            )


@dataclass
class ProgressConfig:
    """Configuration for progress tracking and logging."""

    show_progress: bool = ExportSettings.SHOW_PROGRESS_DEFAULT
    progress_update_interval: float = 0.1  # Update every 100ms
    log_processing_stats: bool = False
    log_memory_usage: bool = False
    verbose_logging: bool = False

    # Progress bar customization
    progress_bar_width: int = 50
    progress_bar_format: str = (
        "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )

    def validate(self) -> None:
        """Validate progress configuration."""
        if not (0.01 <= self.progress_update_interval <= 10.0):
            raise InvalidConfigurationError(
                "progress_update_interval",
                self.progress_update_interval,
                "must be between 0.01 and 10.0 seconds",
            )

        if not (10 <= self.progress_bar_width <= 200):
            raise InvalidConfigurationError(
                "progress_bar_width",
                self.progress_bar_width,
                "must be between 10 and 200 characters",
            )


@dataclass
class ConcurrencyConfig:
    """Configuration for concurrent processing options."""

    max_workers: int = 1  # Number of worker threads/processes
    use_multiprocessing: bool = False  # Use processes instead of threads
    worker_timeout: float = 300.0  # Worker timeout in seconds
    enable_parallel_export: bool = False  # Parallel export of different data types

    # Thread pool configuration
    thread_pool_size: int = 4
    io_thread_pool_size: int = 2

    def validate(self) -> None:
        """Validate concurrency configuration."""
        if not (1 <= self.max_workers <= 32):
            raise InvalidConfigurationError(
                "max_workers", self.max_workers, "must be between 1 and 32"
            )

        if not (1.0 <= self.worker_timeout <= 3600.0):
            raise InvalidConfigurationError(
                "worker_timeout",
                self.worker_timeout,
                "must be between 1.0 and 3600.0 seconds",
            )

        if not (1 <= self.thread_pool_size <= 16):
            raise InvalidConfigurationError(
                "thread_pool_size", self.thread_pool_size, "must be between 1 and 16"
            )

        if not (1 <= self.io_thread_pool_size <= 8):
            raise InvalidConfigurationError(
                "io_thread_pool_size",
                self.io_thread_pool_size,
                "must be between 1 and 8",
            )


@dataclass
class ProcessingConfig:
    """Comprehensive configuration for data processing operations."""

    memory: MemoryConfig = field(default_factory=MemoryConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    progress: ProgressConfig = field(default_factory=ProgressConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)

    def validate(self) -> None:
        """Validate all processing configurations."""
        self.memory.validate()
        self.validation.validate()
        self.progress.validate()
        self.concurrency.validate()

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "ProcessingConfig":
        """Create processing configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            ProcessingConfig instance

        Raises:
            InvalidConfigurationError: If configuration is invalid
        """
        try:
            # Extract sub-configurations
            memory_dict = config_dict.get("memory", {})
            validation_dict = config_dict.get("validation", {})
            progress_dict = config_dict.get("progress", {})
            concurrency_dict = config_dict.get("concurrency", {})

            # Create sub-configuration objects
            memory = MemoryConfig(**memory_dict)
            validation = ValidationConfig(**validation_dict)
            progress = ProgressConfig(**progress_dict)
            concurrency = ConcurrencyConfig(**concurrency_dict)

            # Create main configuration
            config = cls(
                memory=memory,
                validation=validation,
                progress=progress,
                concurrency=concurrency,
            )

            # Validate configuration
            config.validate()

            return config

        except Exception as e:
            if isinstance(e, InvalidConfigurationError):
                raise
            raise InvalidConfigurationError(
                "config_dict",
                str(config_dict),
                f"failed to parse processing configuration: {e}",
            ) from e

    def to_dict(self) -> dict[str, Any]:
        """Convert processing configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            "memory": {
                "max_events_in_memory": self.memory.max_events_in_memory,
                "chunk_size_bytes": self.memory.chunk_size_bytes,
                "max_buffer_size_mb": self.memory.max_buffer_size_mb,
                "enable_memory_monitoring": self.memory.enable_memory_monitoring,
                "memory_warning_threshold": self.memory.memory_warning_threshold,
            },
            "validation": {
                "max_tag_length": self.validation.max_tag_length,
                "max_filename_length": self.validation.max_filename_length,
                "max_tensor_size_bytes": self.validation.max_tensor_size_bytes,
                "max_string_length": self.validation.max_string_length,
                "max_tags_per_type": self.validation.max_tags_per_type,
                "max_steps_per_tag": self.validation.max_steps_per_tag,
                "strict_mode": self.validation.strict_mode,
                "skip_invalid_data": self.validation.skip_invalid_data,
                "validate_tensor_shapes": self.validation.validate_tensor_shapes,
                "validate_data_types": self.validation.validate_data_types,
            },
            "progress": {
                "show_progress": self.progress.show_progress,
                "progress_update_interval": self.progress.progress_update_interval,
                "log_processing_stats": self.progress.log_processing_stats,
                "log_memory_usage": self.progress.log_memory_usage,
                "verbose_logging": self.progress.verbose_logging,
                "progress_bar_width": self.progress.progress_bar_width,
                "progress_bar_format": self.progress.progress_bar_format,
            },
            "concurrency": {
                "max_workers": self.concurrency.max_workers,
                "use_multiprocessing": self.concurrency.use_multiprocessing,
                "worker_timeout": self.concurrency.worker_timeout,
                "enable_parallel_export": self.concurrency.enable_parallel_export,
                "thread_pool_size": self.concurrency.thread_pool_size,
                "io_thread_pool_size": self.concurrency.io_thread_pool_size,
            },
        }


# Convenience functions for creating common processing configurations


def create_minimal_processing_config() -> ProcessingConfig:
    """Create minimal processing configuration with default settings.

    Returns:
        ProcessingConfig with minimal resource usage
    """
    return ProcessingConfig()


def create_high_performance_config() -> ProcessingConfig:
    """Create high-performance processing configuration.

    Returns:
        ProcessingConfig optimized for performance
    """
    memory = MemoryConfig(
        max_events_in_memory=50000,  # Increased for speed
        chunk_size_bytes=5 * 1024 * 1024,  # 5MB chunks
        max_buffer_size_mb=500,  # 500MB buffer
        enable_memory_monitoring=True,
    )

    progress = ProgressConfig(
        show_progress=True,
        log_processing_stats=True,
        progress_update_interval=0.5,  # Less frequent updates for better performance
    )

    concurrency = ConcurrencyConfig(
        max_workers=4,
        enable_parallel_export=True,
        thread_pool_size=8,
        io_thread_pool_size=4,
    )

    return ProcessingConfig(
        memory=memory,
        progress=progress,
        concurrency=concurrency,
    )


def create_memory_efficient_config() -> ProcessingConfig:
    """Create memory-efficient processing configuration.

    Returns:
        ProcessingConfig optimized for low memory usage
    """
    memory = MemoryConfig(
        max_events_in_memory=1000,  # Reduced for memory efficiency
        chunk_size_bytes=512 * 1024,  # 512KB chunks
        max_buffer_size_mb=50,  # 50MB buffer
        enable_memory_monitoring=True,
        memory_warning_threshold=0.7,  # Lower threshold
    )

    validation = ValidationConfig(
        strict_mode=True,  # Fail early to prevent memory issues
        max_tensor_size_bytes=50 * 1024 * 1024,  # 50MB tensor limit
    )

    concurrency = ConcurrencyConfig(
        max_workers=1,  # Single worker to minimize memory usage
        enable_parallel_export=False,
    )

    return ProcessingConfig(
        memory=memory,
        validation=validation,
        concurrency=concurrency,
    )
