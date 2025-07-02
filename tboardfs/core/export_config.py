"""Export-specific configuration management for tboardfs.

This module contains configuration classes specifically for export operations,
including format settings and output organization options.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from .constants import (
    ImageFormats,
    AudioFormats,
    PLYFormats,
    ExportSettings,
    TensorBoardConstants,
)
from .exceptions import InvalidConfigurationError


@dataclass
class ExportFormatConfig:
    """Configuration for export format options."""

    # Image export settings
    image_format: str = ImageFormats.DEFAULT_FORMAT
    image_quality: int = ImageFormats.DEFAULT_QUALITY

    # Audio export settings
    audio_format: str = AudioFormats.DEFAULT_FORMAT
    audio_sample_rate: float = AudioFormats.DEFAULT_SAMPLE_RATE

    # PLY mesh export settings
    ply_format: str = PLYFormats.DEFAULT_FORMAT

    # Step formatting
    step_digits: int = TensorBoardConstants.DEFAULT_STEP_DIGITS

    def validate(self) -> None:
        """Validate export format configuration."""
        if self.image_format not in {
            ImageFormats.PNG,
            ImageFormats.JPG,
            ImageFormats.JPEG,
            ImageFormats.GIF,
        }:
            raise InvalidConfigurationError(
                "image_format",
                self.image_format,
                f"must be one of: {ImageFormats.PNG}, {ImageFormats.JPG}, {ImageFormats.JPEG}, {ImageFormats.GIF}",
            )

        if not (0 <= self.image_quality <= 100):
            raise InvalidConfigurationError(
                "image_quality", self.image_quality, "must be between 0 and 100"
            )

        if self.audio_format not in {AudioFormats.WAV, AudioFormats.MP3}:
            raise InvalidConfigurationError(
                "audio_format",
                self.audio_format,
                f"must be one of: {AudioFormats.WAV}, {AudioFormats.MP3}",
            )

        if not (
            AudioFormats.MIN_SAMPLE_RATE
            <= self.audio_sample_rate
            <= AudioFormats.MAX_SAMPLE_RATE
        ):
            raise InvalidConfigurationError(
                "audio_sample_rate",
                self.audio_sample_rate,
                f"must be between {AudioFormats.MIN_SAMPLE_RATE} and {AudioFormats.MAX_SAMPLE_RATE}",
            )

        if self.ply_format not in {
            PLYFormats.BINARY,
            PLYFormats.TEXT,
            PLYFormats.ASCII,
        }:
            raise InvalidConfigurationError(
                "ply_format",
                self.ply_format,
                f"must be one of: {PLYFormats.BINARY}, {PLYFormats.TEXT}, {PLYFormats.ASCII}",
            )

        if not (1 <= self.step_digits <= 20):
            raise InvalidConfigurationError(
                "step_digits", self.step_digits, "must be between 1 and 20"
            )


@dataclass
class OutputOrganizationConfig:
    """Configuration for output file organization."""

    # Directory structure options
    create_step_directories: bool = ExportSettings.CREATE_STEP_DIRECTORIES
    aggregate_scalars: bool = ExportSettings.AGGREGATE_SCALARS
    aggregate_histograms: bool = ExportSettings.AGGREGATE_HISTOGRAMS

    # File naming options
    use_zero_padding: bool = ExportSettings.USE_ZERO_PADDING
    sanitize_tags: bool = ExportSettings.SANITIZE_TAGS

    # Output formats by data type
    scalar_formats: list[str] = field(default_factory=lambda: ["txt", "csv"])
    histogram_formats: list[str] = field(default_factory=lambda: ["csv", "npz"])
    image_formats: list[str] = field(default_factory=lambda: ["png"])
    video_formats: list[str] = field(default_factory=lambda: ["gif"])
    audio_formats: list[str] = field(default_factory=lambda: ["wav"])
    text_formats: list[str] = field(default_factory=lambda: ["txt"])
    mesh_formats: list[str] = field(default_factory=lambda: ["ply"])
    hyperparameter_formats: list[str] = field(default_factory=lambda: ["yaml"])
    pr_curve_formats: list[str] = field(default_factory=lambda: ["csv", "npz"])

    # Supported formats by type (for validation)
    _SUPPORTED_FORMATS: ClassVar[dict[str, list[str]]] = {
        "scalar": ["txt", "csv", "json"],
        "histogram": ["txt", "csv", "npz"],
        "image": ["png", "jpg", "jpeg", "gif", "bmp"],
        "video": ["gif", "mp4", "avi"],
        "audio": ["wav", "mp3", "flac"],
        "text": ["txt", "md", "json"],
        "mesh": ["ply", "obj", "stl"],
        "hyperparameter": ["yaml", "json", "txt"],
        "pr_curve": ["csv", "npz", "json"],
    }

    def validate(self) -> None:
        """Validate output organization configuration."""
        format_configs = {
            "scalar": self.scalar_formats,
            "histogram": self.histogram_formats,
            "image": self.image_formats,
            "video": self.video_formats,
            "audio": self.audio_formats,
            "text": self.text_formats,
            "mesh": self.mesh_formats,
            "hyperparameter": self.hyperparameter_formats,
            "pr_curve": self.pr_curve_formats,
        }

        for data_type, formats in format_configs.items():
            supported = self._SUPPORTED_FORMATS[data_type]
            for fmt in formats:
                if fmt not in supported:
                    raise InvalidConfigurationError(
                        f"{data_type}_formats",
                        fmt,
                        f"unsupported format (supported: {', '.join(supported)})",
                    )


@dataclass
class TBoardFSExportConfig:
    """Main export configuration class for tboardfs operations."""

    # Output path
    output_path: Path = field(default_factory=lambda: Path.cwd() / "exported_data")

    # Sub-configurations
    export_formats: ExportFormatConfig = field(default_factory=ExportFormatConfig)
    output_organization: OutputOrganizationConfig = field(
        default_factory=OutputOrganizationConfig
    )

    def validate(self) -> None:
        """Validate the export configuration."""
        # Validate sub-configurations
        self.export_formats.validate()
        self.output_organization.validate()

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TBoardFSExportConfig":
        """Create export configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            TBoardFSExportConfig instance

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

            # Create sub-configuration objects
            export_formats = ExportFormatConfig(**export_formats_dict)
            output_organization = OutputOrganizationConfig(**output_org_dict)

            # Create main configuration
            config = cls(
                output_path=output_path,
                export_formats=export_formats,
                output_organization=output_organization,
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
                f"failed to parse export configuration: {e}",
            ) from e

    def to_dict(self) -> dict[str, Any]:
        """Convert export configuration to dictionary.

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
        }


# Convenience functions for creating common export configurations


def create_minimal_export_config(output_path: str | Path) -> TBoardFSExportConfig:
    """Create minimal export configuration with just output path.

    Args:
        output_path: Directory to save exported data

    Returns:
        TBoardFSExportConfig with minimal settings
    """
    return TBoardFSExportConfig(output_path=Path(output_path))


def create_high_quality_export_config(output_path: str | Path) -> TBoardFSExportConfig:
    """Create high-quality export configuration.

    Args:
        output_path: Directory to save exported data

    Returns:
        TBoardFSExportConfig optimized for quality
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

    return TBoardFSExportConfig(
        output_path=Path(output_path),
        export_formats=export_formats,
        output_organization=output_org,
    )


def create_fast_export_config(output_path: str | Path) -> TBoardFSExportConfig:
    """Create fast export configuration optimized for speed.

    Args:
        output_path: Directory to save exported data

    Returns:
        TBoardFSExportConfig optimized for performance
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

    return TBoardFSExportConfig(
        output_path=Path(output_path),
        export_formats=export_formats,
        output_organization=output_org,
    )
