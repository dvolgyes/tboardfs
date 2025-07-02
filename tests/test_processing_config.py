"""Tests for processing configuration management."""

import pytest

from tboardfs.core.processing_config import (
    MemoryConfig,
    ValidationConfig,
    ProgressConfig,
    ConcurrencyConfig,
    ProcessingConfig,
    create_minimal_processing_config,
    create_high_performance_config,
    create_memory_efficient_config,
)
from tboardfs.core.constants import ExportSettings, ValidationLimits
from tboardfs.core.exceptions import InvalidConfigurationError


class TestMemoryConfig:
    """Test MemoryConfig class."""

    def test_default_initialization(self):
        """Test default configuration values."""
        config = MemoryConfig()

        assert config.max_events_in_memory == ExportSettings.MAX_EVENTS_IN_MEMORY
        assert config.chunk_size_bytes == ExportSettings.CHUNK_SIZE_BYTES
        assert config.max_buffer_size_mb == 100
        assert config.enable_memory_monitoring is False
        assert config.memory_warning_threshold == 0.8

    def test_custom_initialization(self):
        """Test configuration with custom values."""
        config = MemoryConfig(
            max_events_in_memory=50000,
            chunk_size_bytes=2048,
            max_buffer_size_mb=200,
            enable_memory_monitoring=True,
            memory_warning_threshold=0.9,
        )

        assert config.max_events_in_memory == 50000
        assert config.chunk_size_bytes == 2048
        assert config.max_buffer_size_mb == 200
        assert config.enable_memory_monitoring is True
        assert config.memory_warning_threshold == 0.9

    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = MemoryConfig()
        config.validate()  # Should not raise

    def test_validate_invalid_max_events_in_memory(self):
        """Test validation with invalid max events in memory."""
        config = MemoryConfig(max_events_in_memory=0)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "max_events_in_memory" in str(exc_info.value)

    def test_validate_invalid_chunk_size_bytes(self):
        """Test validation with invalid chunk size."""
        config = MemoryConfig(chunk_size_bytes=512)  # Below 1024 minimum

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "chunk_size_bytes" in str(exc_info.value)

    def test_validate_invalid_max_buffer_size_mb(self):
        """Test validation with invalid buffer size."""
        config = MemoryConfig(max_buffer_size_mb=0)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "max_buffer_size_mb" in str(exc_info.value)

    def test_validate_invalid_memory_warning_threshold_below_range(self):
        """Test validation with memory warning threshold below valid range."""
        config = MemoryConfig(memory_warning_threshold=0.05)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "memory_warning_threshold" in str(exc_info.value)

    def test_validate_invalid_memory_warning_threshold_above_range(self):
        """Test validation with memory warning threshold above valid range."""
        config = MemoryConfig(memory_warning_threshold=1.5)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "memory_warning_threshold" in str(exc_info.value)


class TestValidationConfig:
    """Test ValidationConfig class."""

    def test_default_initialization(self):
        """Test default configuration values."""
        config = ValidationConfig()

        assert config.max_tag_length == ValidationLimits.MAX_TAG_LENGTH
        assert config.max_filename_length == ValidationLimits.MAX_FILENAME_LENGTH
        assert config.max_tensor_size_bytes == ValidationLimits.MAX_TENSOR_SIZE_BYTES
        assert config.max_string_length == ValidationLimits.MAX_STRING_LENGTH
        assert config.max_tags_per_type == ValidationLimits.MAX_TAGS_PER_TYPE
        assert config.max_steps_per_tag == ValidationLimits.MAX_STEPS_PER_TAG
        assert config.strict_mode is False
        assert config.skip_invalid_data is True
        assert config.validate_tensor_shapes is True
        assert config.validate_data_types is True

    def test_custom_initialization(self):
        """Test configuration with custom values."""
        config = ValidationConfig(
            max_tag_length=100,
            max_filename_length=150,
            strict_mode=True,
            skip_invalid_data=False,
        )

        assert config.max_tag_length == 100
        assert config.max_filename_length == 150
        assert config.strict_mode is True
        assert config.skip_invalid_data is False

    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = ValidationConfig()
        config.validate()  # Should not raise

    def test_validate_invalid_max_tag_length(self):
        """Test validation with invalid max tag length."""
        config = ValidationConfig(max_tag_length=0)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "max_tag_length" in str(exc_info.value)

    def test_validate_invalid_max_filename_length(self):
        """Test validation with invalid max filename length."""
        config = ValidationConfig(max_filename_length=0)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "max_filename_length" in str(exc_info.value)

    def test_validate_invalid_max_tensor_size_bytes(self):
        """Test validation with invalid max tensor size."""
        config = ValidationConfig(max_tensor_size_bytes=0)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "max_tensor_size_bytes" in str(exc_info.value)

    def test_validate_invalid_max_string_length(self):
        """Test validation with invalid max string length."""
        config = ValidationConfig(max_string_length=0)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "max_string_length" in str(exc_info.value)

    def test_validate_invalid_max_tags_per_type(self):
        """Test validation with invalid max tags per type."""
        config = ValidationConfig(max_tags_per_type=0)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "max_tags_per_type" in str(exc_info.value)

    def test_validate_invalid_max_steps_per_tag(self):
        """Test validation with invalid max steps per tag."""
        config = ValidationConfig(max_steps_per_tag=0)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "max_steps_per_tag" in str(exc_info.value)


class TestProgressConfig:
    """Test ProgressConfig class."""

    def test_default_initialization(self):
        """Test default configuration values."""
        config = ProgressConfig()

        assert config.show_progress == ExportSettings.SHOW_PROGRESS_DEFAULT
        assert config.progress_update_interval == 0.1
        assert config.log_processing_stats is False
        assert config.log_memory_usage is False
        assert config.verbose_logging is False
        assert config.progress_bar_width == 50
        assert "{l_bar}{bar}" in config.progress_bar_format

    def test_custom_initialization(self):
        """Test configuration with custom values."""
        config = ProgressConfig(
            show_progress=True,
            progress_update_interval=0.5,
            log_processing_stats=True,
            progress_bar_width=80,
        )

        assert config.show_progress is True
        assert config.progress_update_interval == 0.5
        assert config.log_processing_stats is True
        assert config.progress_bar_width == 80

    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = ProgressConfig()
        config.validate()  # Should not raise

    def test_validate_invalid_progress_update_interval_below_range(self):
        """Test validation with progress update interval below valid range."""
        config = ProgressConfig(progress_update_interval=0.005)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "progress_update_interval" in str(exc_info.value)

    def test_validate_invalid_progress_update_interval_above_range(self):
        """Test validation with progress update interval above valid range."""
        config = ProgressConfig(progress_update_interval=15.0)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "progress_update_interval" in str(exc_info.value)

    def test_validate_invalid_progress_bar_width_below_range(self):
        """Test validation with progress bar width below valid range."""
        config = ProgressConfig(progress_bar_width=5)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "progress_bar_width" in str(exc_info.value)

    def test_validate_invalid_progress_bar_width_above_range(self):
        """Test validation with progress bar width above valid range."""
        config = ProgressConfig(progress_bar_width=250)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "progress_bar_width" in str(exc_info.value)


class TestConcurrencyConfig:
    """Test ConcurrencyConfig class."""

    def test_default_initialization(self):
        """Test default configuration values."""
        config = ConcurrencyConfig()

        assert config.max_workers == 1
        assert config.use_multiprocessing is False
        assert config.worker_timeout == 300.0
        assert config.enable_parallel_export is False
        assert config.thread_pool_size == 4
        assert config.io_thread_pool_size == 2

    def test_custom_initialization(self):
        """Test configuration with custom values."""
        config = ConcurrencyConfig(
            max_workers=4,
            use_multiprocessing=True,
            worker_timeout=600.0,
            enable_parallel_export=True,
            thread_pool_size=8,
        )

        assert config.max_workers == 4
        assert config.use_multiprocessing is True
        assert config.worker_timeout == 600.0
        assert config.enable_parallel_export is True
        assert config.thread_pool_size == 8

    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = ConcurrencyConfig()
        config.validate()  # Should not raise

    def test_validate_invalid_max_workers_below_range(self):
        """Test validation with max workers below valid range."""
        config = ConcurrencyConfig(max_workers=0)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "max_workers" in str(exc_info.value)

    def test_validate_invalid_max_workers_above_range(self):
        """Test validation with max workers above valid range."""
        config = ConcurrencyConfig(max_workers=50)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "max_workers" in str(exc_info.value)

    def test_validate_invalid_worker_timeout_below_range(self):
        """Test validation with worker timeout below valid range."""
        config = ConcurrencyConfig(worker_timeout=0.5)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "worker_timeout" in str(exc_info.value)

    def test_validate_invalid_worker_timeout_above_range(self):
        """Test validation with worker timeout above valid range."""
        config = ConcurrencyConfig(worker_timeout=4000.0)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "worker_timeout" in str(exc_info.value)

    def test_validate_invalid_thread_pool_size_below_range(self):
        """Test validation with thread pool size below valid range."""
        config = ConcurrencyConfig(thread_pool_size=0)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "thread_pool_size" in str(exc_info.value)

    def test_validate_invalid_thread_pool_size_above_range(self):
        """Test validation with thread pool size above valid range."""
        config = ConcurrencyConfig(thread_pool_size=20)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "thread_pool_size" in str(exc_info.value)

    def test_validate_invalid_io_thread_pool_size_below_range(self):
        """Test validation with IO thread pool size below valid range."""
        config = ConcurrencyConfig(io_thread_pool_size=0)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "io_thread_pool_size" in str(exc_info.value)

    def test_validate_invalid_io_thread_pool_size_above_range(self):
        """Test validation with IO thread pool size above valid range."""
        config = ConcurrencyConfig(io_thread_pool_size=10)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "io_thread_pool_size" in str(exc_info.value)


class TestProcessingConfig:
    """Test ProcessingConfig class."""

    def test_default_initialization(self):
        """Test default configuration values."""
        config = ProcessingConfig()

        assert isinstance(config.memory, MemoryConfig)
        assert isinstance(config.validation, ValidationConfig)
        assert isinstance(config.progress, ProgressConfig)
        assert isinstance(config.concurrency, ConcurrencyConfig)

    def test_custom_initialization(self):
        """Test configuration with custom sub-configs."""
        memory_config = MemoryConfig(max_events_in_memory=5000)
        validation_config = ValidationConfig(strict_mode=True)

        config = ProcessingConfig(memory=memory_config, validation=validation_config)

        assert config.memory.max_events_in_memory == 5000
        assert config.validation.strict_mode is True

    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = ProcessingConfig()
        config.validate()  # Should not raise

    def test_validate_invalid_subconfig(self):
        """Test validation with invalid sub-configuration."""
        config = ProcessingConfig()
        config.memory.max_events_in_memory = 0

        with pytest.raises(InvalidConfigurationError):
            config.validate()

    def test_from_dict_minimal(self):
        """Test creating configuration from minimal dictionary."""
        config_dict = {}

        config = ProcessingConfig.from_dict(config_dict)

        assert isinstance(config.memory, MemoryConfig)
        assert isinstance(config.validation, ValidationConfig)
        assert isinstance(config.progress, ProgressConfig)
        assert isinstance(config.concurrency, ConcurrencyConfig)

    def test_from_dict_partial(self):
        """Test creating configuration from partial dictionary."""
        config_dict = {
            "memory": {"max_events_in_memory": 5000, "chunk_size_bytes": 2048},
            "progress": {"show_progress": True},
        }

        config = ProcessingConfig.from_dict(config_dict)

        assert config.memory.max_events_in_memory == 5000
        assert config.memory.chunk_size_bytes == 2048
        assert config.progress.show_progress is True
        assert config.progress.progress_update_interval == 0.1  # Default

    def test_from_dict_full(self):
        """Test creating configuration from full dictionary."""
        config_dict = {
            "memory": {
                "max_events_in_memory": 5000,
                "chunk_size_bytes": 2048,
                "max_buffer_size_mb": 200,
                "enable_memory_monitoring": True,
                "memory_warning_threshold": 0.9,
            },
            "validation": {
                "max_tag_length": 100,
                "strict_mode": True,
                "skip_invalid_data": False,
            },
            "progress": {
                "show_progress": True,
                "progress_update_interval": 0.5,
                "log_processing_stats": True,
                "progress_bar_width": 80,
            },
            "concurrency": {
                "max_workers": 4,
                "use_multiprocessing": True,
                "enable_parallel_export": True,
            },
        }

        config = ProcessingConfig.from_dict(config_dict)

        assert config.memory.max_events_in_memory == 5000
        assert config.memory.chunk_size_bytes == 2048
        assert config.memory.max_buffer_size_mb == 200
        assert config.memory.enable_memory_monitoring is True
        assert config.memory.memory_warning_threshold == 0.9
        assert config.validation.max_tag_length == 100
        assert config.validation.strict_mode is True
        assert config.validation.skip_invalid_data is False
        assert config.progress.show_progress is True
        assert config.progress.progress_update_interval == 0.5
        assert config.progress.log_processing_stats is True
        assert config.progress.progress_bar_width == 80
        assert config.concurrency.max_workers == 4
        assert config.concurrency.use_multiprocessing is True
        assert config.concurrency.enable_parallel_export is True

    def test_from_dict_invalid_config(self):
        """Test creating configuration from invalid dictionary."""
        config_dict = {"memory": {"max_events_in_memory": 0}}

        with pytest.raises(InvalidConfigurationError):
            ProcessingConfig.from_dict(config_dict)

    def test_from_dict_malformed(self):
        """Test creating configuration from malformed dictionary."""
        config_dict = {"memory": "not_a_dict"}

        with pytest.raises(InvalidConfigurationError):
            ProcessingConfig.from_dict(config_dict)

    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = ProcessingConfig()
        config.memory.max_events_in_memory = 5000
        config.progress.show_progress = True
        config.concurrency.max_workers = 4

        result = config.to_dict()

        assert result["memory"]["max_events_in_memory"] == 5000
        assert result["progress"]["show_progress"] is True
        assert result["concurrency"]["max_workers"] == 4

    def test_roundtrip_dict_conversion(self):
        """Test that to_dict -> from_dict preserves configuration."""
        original_config = ProcessingConfig()
        original_config.memory.max_events_in_memory = 5000
        original_config.memory.chunk_size_bytes = 2048
        original_config.progress.show_progress = True
        original_config.concurrency.max_workers = 4

        config_dict = original_config.to_dict()
        restored_config = ProcessingConfig.from_dict(config_dict)

        assert (
            restored_config.memory.max_events_in_memory
            == original_config.memory.max_events_in_memory
        )
        assert (
            restored_config.memory.chunk_size_bytes
            == original_config.memory.chunk_size_bytes
        )
        assert (
            restored_config.progress.show_progress
            == original_config.progress.show_progress
        )
        assert (
            restored_config.concurrency.max_workers
            == original_config.concurrency.max_workers
        )


class TestConvenienceFunctions:
    """Test convenience functions for creating processing configurations."""

    def test_create_minimal_processing_config(self):
        """Test creating minimal processing configuration."""
        config = create_minimal_processing_config()

        assert isinstance(config, ProcessingConfig)
        assert isinstance(config.memory, MemoryConfig)
        assert isinstance(config.validation, ValidationConfig)
        assert isinstance(config.progress, ProgressConfig)
        assert isinstance(config.concurrency, ConcurrencyConfig)

        # Should validate without errors
        config.validate()

    def test_create_high_performance_config(self):
        """Test creating high-performance configuration."""
        config = create_high_performance_config()

        assert isinstance(config, ProcessingConfig)
        assert config.memory.max_events_in_memory == 50000
        assert config.memory.chunk_size_bytes == 5 * 1024 * 1024
        assert config.memory.max_buffer_size_mb == 500
        assert config.memory.enable_memory_monitoring is True
        assert config.progress.show_progress is True
        assert config.progress.log_processing_stats is True
        assert config.progress.progress_update_interval == 0.5
        assert config.concurrency.max_workers == 4
        assert config.concurrency.enable_parallel_export is True
        assert config.concurrency.thread_pool_size == 8
        assert config.concurrency.io_thread_pool_size == 4

        # Should validate without errors
        config.validate()

    def test_create_memory_efficient_config(self):
        """Test creating memory-efficient configuration."""
        config = create_memory_efficient_config()

        assert isinstance(config, ProcessingConfig)
        assert config.memory.max_events_in_memory == 1000
        assert config.memory.chunk_size_bytes == 512 * 1024
        assert config.memory.max_buffer_size_mb == 50
        assert config.memory.enable_memory_monitoring is True
        assert config.memory.memory_warning_threshold == 0.7
        assert config.validation.strict_mode is True
        assert config.validation.max_tensor_size_bytes == 50 * 1024 * 1024
        assert config.concurrency.max_workers == 1
        assert config.concurrency.enable_parallel_export is False

        # Should validate without errors
        config.validate()
