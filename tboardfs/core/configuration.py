"""Centralized configuration management for tboardfs.

This module provides a comprehensive configuration system that consolidates
export settings, format options, and processing parameters into a single
configurable class hierarchy. It now uses modular configuration classes
for better organization and maintainability.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Import the new modular configuration classes
from .export_config import (
    ExportFormatConfig,
    OutputOrganizationConfig,
)
from .format_config import FormatConfig
from .processing_config import (
    ProcessingConfig,
    create_minimal_processing_config,
    create_high_performance_config,
)
from .constants import (
    ImageFormats,
    AudioFormats,
    PLYFormats,
)
from .exceptions import InvalidConfigurationError


@dataclass
class TBoardFSConfig:
    """Main configuration class for tboardfs operations.

    This class maintains backward compatibility while using the new modular
    configuration system under the hood.
    """

    # Output path
    output_path: Path = field(default_factory=lambda: Path.cwd() / "exported_data")

    # Sub-configurations (using new modular classes)
    export_formats: ExportFormatConfig = field(default_factory=ExportFormatConfig)
    output_organization: OutputOrganizationConfig = field(
        default_factory=OutputOrganizationConfig
    )
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)

    # New modular format configuration
    format_config: FormatConfig = field(default_factory=FormatConfig)

    def validate(self) -> None:
        """Validate the entire configuration."""
        # Validate sub-configurations
        self.export_formats.validate()
        self.output_organization.validate()
        self.processing.validate()
        self.format_config.validate()

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TBoardFSConfig":
        """Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            TBoardFSConfig instance

        Raises:
            InvalidConfigurationError: If configuration is invalid
        """
        try:
            # Extract output path
            output_path = config_dict.get("output_path", Path.cwd() / "exported_data")
            if isinstance(output_path, str):
                output_path = Path(output_path)

            # Extract sub-configurations
            export_formats_dict = config_dict.get("export_formats", {})
            output_org_dict = config_dict.get("output_organization", {})
            processing_dict = config_dict.get("processing", {})
            format_config_dict = config_dict.get("format_config", {})

            # Create sub-configuration objects
            export_formats = ExportFormatConfig(**export_formats_dict)
            output_organization = OutputOrganizationConfig(**output_org_dict)
            processing = ProcessingConfig.from_dict(processing_dict)
            format_config = FormatConfig.from_dict(format_config_dict)

            # Create main configuration
            config = cls(
                output_path=output_path,
                export_formats=export_formats,
                output_organization=output_organization,
                processing=processing,
                format_config=format_config,
            )

            # Validate configuration
            config.validate()

            return config

        except Exception as e:
            if isinstance(e, InvalidConfigurationError):
                raise
            raise InvalidConfigurationError(
                "config_dict", str(config_dict), f"failed to parse configuration: {e}"
            ) from e

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            "output_path": str(self.output_path),
            "export_formats": {
                "image_format": self.export_formats.image_format,
                "image_quality": self.export_formats.image_quality,
                "audio_format": self.export_formats.audio_format,
                "audio_sample_rate": self.export_formats.audio_sample_rate,
                "ply_format": self.export_formats.ply_format,
                "step_digits": self.export_formats.step_digits,
            },
            "output_organization": {
                "create_step_directories": self.output_organization.create_step_directories,
                "aggregate_scalars": self.output_organization.aggregate_scalars,
                "aggregate_histograms": self.output_organization.aggregate_histograms,
                "use_zero_padding": self.output_organization.use_zero_padding,
                "sanitize_tags": self.output_organization.sanitize_tags,
                "scalar_formats": self.output_organization.scalar_formats,
                "histogram_formats": self.output_organization.histogram_formats,
                "image_formats": self.output_organization.image_formats,
                "video_formats": self.output_organization.video_formats,
                "audio_formats": self.output_organization.audio_formats,
                "text_formats": self.output_organization.text_formats,
                "mesh_formats": self.output_organization.mesh_formats,
                "hyperparameter_formats": self.output_organization.hyperparameter_formats,
                "pr_curve_formats": self.output_organization.pr_curve_formats,
            },
            "processing": self.processing.to_dict(),
            "format_config": self.format_config.to_dict(),
        }


# Convenience functions for creating common configurations


def create_minimal_config(output_path: str | Path) -> TBoardFSConfig:
    """Create minimal configuration with just output path.

    Args:
        output_path: Directory to save exported data

    Returns:
        TBoardFSConfig with minimal settings
    """
    return TBoardFSConfig(
        output_path=Path(output_path),
        processing=create_minimal_processing_config(),
    )


def create_high_quality_config(output_path: str | Path) -> TBoardFSConfig:
    """Create high-quality export configuration.

    Args:
        output_path: Directory to save exported data

    Returns:
        TBoardFSConfig optimized for quality
    """
    export_formats = ExportFormatConfig(
        image_format=ImageFormats.PNG,
        image_quality=95,
        audio_format=AudioFormats.WAV,
        ply_format=PLYFormats.BINARY,
    )

    output_org = OutputOrganizationConfig(
        create_step_directories=True,
        aggregate_scalars=False,  # Keep individual files for analysis
        aggregate_histograms=False,
    )

    return TBoardFSConfig(
        output_path=Path(output_path),
        export_formats=export_formats,
        output_organization=output_org,
        processing=create_minimal_processing_config(),
    )


def create_fast_config(output_path: str | Path) -> TBoardFSConfig:
    """Create fast export configuration optimized for speed.

    Args:
        output_path: Directory to save exported data

    Returns:
        TBoardFSConfig optimized for performance
    """
    export_formats = ExportFormatConfig(
        image_format=ImageFormats.JPG,
        image_quality=75,
        audio_format=AudioFormats.MP3,
        ply_format=PLYFormats.BINARY,
    )

    output_org = OutputOrganizationConfig(
        create_step_directories=False,
        aggregate_scalars=True,
        aggregate_histograms=True,
        scalar_formats=["txt"],  # Fastest format
        histogram_formats=["npz"],  # Binary format for speed
    )

    return TBoardFSConfig(
        output_path=Path(output_path),
        export_formats=export_formats,
        output_organization=output_org,
        processing=create_high_performance_config(),
    )
