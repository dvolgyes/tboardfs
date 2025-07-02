"""Format configuration and validation utilities for tboardfs.

This module contains format-specific configuration classes and validation
functions for different data types supported by TensorBoard.
"""

from dataclasses import dataclass
from typing import Any, ClassVar

from .constants import (
    ImageFormats,
    AudioFormats,
    PLYFormats,
)
from .exceptions import InvalidConfigurationError


@dataclass
class ImageFormatConfig:
    """Configuration for image format options."""

    format: str = ImageFormats.DEFAULT_FORMAT
    quality: int = ImageFormats.DEFAULT_QUALITY

    # Supported formats
    _SUPPORTED_FORMATS: ClassVar[set[str]] = {
        ImageFormats.PNG,
        ImageFormats.JPG,
        ImageFormats.JPEG,
        ImageFormats.GIF,
        "bmp",
        "tiff",
        "webp",
    }

    def validate(self) -> None:
        """Validate image format configuration."""
        if self.format not in self._SUPPORTED_FORMATS:
            raise InvalidConfigurationError(
                "image_format",
                self.format,
                f"must be one of: {', '.join(sorted(self._SUPPORTED_FORMATS))}",
            )

        if not (0 <= self.quality <= 100):
            raise InvalidConfigurationError(
                "image_quality", self.quality, "must be between 0 and 100"
            )


@dataclass
class AudioFormatConfig:
    """Configuration for audio format options."""

    format: str = AudioFormats.DEFAULT_FORMAT
    sample_rate: float = AudioFormats.DEFAULT_SAMPLE_RATE
    bitrate: int = 128  # Default bitrate in kbps

    # Supported formats
    _SUPPORTED_FORMATS: ClassVar[set[str]] = {
        AudioFormats.WAV,
        AudioFormats.MP3,
        "flac",
        "ogg",
    }

    def validate(self) -> None:
        """Validate audio format configuration."""
        if self.format not in self._SUPPORTED_FORMATS:
            raise InvalidConfigurationError(
                "audio_format",
                self.format,
                f"must be one of: {', '.join(sorted(self._SUPPORTED_FORMATS))}",
            )

        if not (
            AudioFormats.MIN_SAMPLE_RATE
            <= self.sample_rate
            <= AudioFormats.MAX_SAMPLE_RATE
        ):
            raise InvalidConfigurationError(
                "audio_sample_rate",
                self.sample_rate,
                f"must be between {AudioFormats.MIN_SAMPLE_RATE} and {AudioFormats.MAX_SAMPLE_RATE}",
            )

        if not (32 <= self.bitrate <= 320):  # Reasonable bitrate range for audio
            raise InvalidConfigurationError(
                "audio_bitrate",
                self.bitrate,
                "must be between 32 and 320 kbps",
            )


@dataclass
class MeshFormatConfig:
    """Configuration for 3D mesh format options."""

    format: str = PLYFormats.DEFAULT_FORMAT
    precision: int = 6  # Default floating point precision

    # Supported formats
    _SUPPORTED_FORMATS: ClassVar[set[str]] = {
        PLYFormats.BINARY,
        PLYFormats.TEXT,
        PLYFormats.ASCII,
    }

    def validate(self) -> None:
        """Validate mesh format configuration."""
        if self.format not in self._SUPPORTED_FORMATS:
            raise InvalidConfigurationError(
                "ply_format",
                self.format,
                f"must be one of: {', '.join(sorted(self._SUPPORTED_FORMATS))}",
            )

        if not (1 <= self.precision <= 16):
            raise InvalidConfigurationError(
                "ply_precision", self.precision, "must be between 1 and 16"
            )


@dataclass
class TextFormatConfig:
    """Configuration for text format options."""

    encoding: str = "utf-8"
    line_ending: str = "\n"  # Unix-style by default
    max_line_length: int = 120

    # Supported encodings
    _SUPPORTED_ENCODINGS: ClassVar[set[str]] = {
        "utf-8",
        "ascii",
        "latin-1",
        "utf-16",
        "utf-32",
    }

    # Supported line endings
    _SUPPORTED_LINE_ENDINGS: ClassVar[set[str]] = {
        "\n",  # Unix
        "\r\n",  # Windows
        "\r",  # Mac Classic
    }

    def validate(self) -> None:
        """Validate text format configuration."""
        if self.encoding not in self._SUPPORTED_ENCODINGS:
            raise InvalidConfigurationError(
                "text_encoding",
                self.encoding,
                f"must be one of: {', '.join(sorted(self._SUPPORTED_ENCODINGS))}",
            )

        if self.line_ending not in self._SUPPORTED_LINE_ENDINGS:
            raise InvalidConfigurationError(
                "text_line_ending",
                repr(self.line_ending),
                "must be one of: \\n (Unix), \\r\\n (Windows), \\r (Mac)",
            )

        if not (10 <= self.max_line_length <= 1000):
            raise InvalidConfigurationError(
                "text_max_line_length",
                self.max_line_length,
                "must be between 10 and 1000",
            )


@dataclass
class DataFormatConfig:
    """Configuration for structured data formats (CSV, JSON, etc.)."""

    csv_delimiter: str = ","
    csv_quote_char: str = '"'
    csv_escape_char: str = "\\"
    json_indent: int = 2
    json_sort_keys: bool = True

    # Supported CSV delimiters
    _SUPPORTED_CSV_DELIMITERS: ClassVar[set[str]] = {
        ",",  # Comma
        ";",  # Semicolon
        "\t",  # Tab
        "|",  # Pipe
    }

    def validate(self) -> None:
        """Validate data format configuration."""
        if self.csv_delimiter not in self._SUPPORTED_CSV_DELIMITERS:
            raise InvalidConfigurationError(
                "csv_delimiter",
                self.csv_delimiter,
                "must be one of: , (comma), ; (semicolon), \\t (tab), | (pipe)",
            )

        if len(self.csv_quote_char) != 1:
            raise InvalidConfigurationError(
                "csv_quote_char", self.csv_quote_char, "must be a single character"
            )

        if len(self.csv_escape_char) != 1:
            raise InvalidConfigurationError(
                "csv_escape_char", self.csv_escape_char, "must be a single character"
            )

        if not (0 <= self.json_indent <= 8):
            raise InvalidConfigurationError(
                "json_indent", self.json_indent, "must be between 0 and 8"
            )


@dataclass
class FormatConfig:
    """Comprehensive format configuration for all data types."""

    image: ImageFormatConfig = ImageFormatConfig()
    audio: AudioFormatConfig = AudioFormatConfig()
    mesh: MeshFormatConfig = MeshFormatConfig()
    text: TextFormatConfig = TextFormatConfig()
    data: DataFormatConfig = DataFormatConfig()

    def validate(self) -> None:
        """Validate all format configurations."""
        self.image.validate()
        self.audio.validate()
        self.mesh.validate()
        self.text.validate()
        self.data.validate()

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "FormatConfig":
        """Create format configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            FormatConfig instance

        Raises:
            InvalidConfigurationError: If configuration is invalid
        """
        try:
            # Extract sub-configurations
            image_dict = config_dict.get("image", {})
            audio_dict = config_dict.get("audio", {})
            mesh_dict = config_dict.get("mesh", {})
            text_dict = config_dict.get("text", {})
            data_dict = config_dict.get("data", {})

            # Create sub-configuration objects
            image = ImageFormatConfig(**image_dict)
            audio = AudioFormatConfig(**audio_dict)
            mesh = MeshFormatConfig(**mesh_dict)
            text = TextFormatConfig(**text_dict)
            data = DataFormatConfig(**data_dict)

            # Create main configuration
            config = cls(
                image=image,
                audio=audio,
                mesh=mesh,
                text=text,
                data=data,
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
                f"failed to parse format configuration: {e}",
            ) from e

    def to_dict(self) -> dict[str, Any]:
        """Convert format configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            "image": {
                "format": self.image.format,
                "quality": self.image.quality,
            },
            "audio": {
                "format": self.audio.format,
                "sample_rate": self.audio.sample_rate,
                "bitrate": self.audio.bitrate,
            },
            "mesh": {
                "format": self.mesh.format,
                "precision": self.mesh.precision,
            },
            "text": {
                "encoding": self.text.encoding,
                "line_ending": self.text.line_ending,
                "max_line_length": self.text.max_line_length,
            },
            "data": {
                "csv_delimiter": self.data.csv_delimiter,
                "csv_quote_char": self.data.csv_quote_char,
                "csv_escape_char": self.data.csv_escape_char,
                "json_indent": self.data.json_indent,
                "json_sort_keys": self.data.json_sort_keys,
            },
        }


# Format validation functions


def validate_image_format_options(format_name: str, quality: int | None = None) -> None:
    """Validate image format and quality options.

    Args:
        format_name: Image format name
        quality: Image quality (0-100), optional

    Raises:
        InvalidConfigurationError: If validation fails
    """
    config = ImageFormatConfig(format=format_name)
    if quality is not None:
        config.quality = quality
    config.validate()


def validate_audio_format_options(
    format_name: str, sample_rate: float | None = None, bitrate: int | None = None
) -> None:
    """Validate audio format options.

    Args:
        format_name: Audio format name
        sample_rate: Sample rate in Hz, optional
        bitrate: Bitrate in kbps, optional

    Raises:
        InvalidConfigurationError: If validation fails
    """
    config = AudioFormatConfig(format=format_name)
    if sample_rate is not None:
        config.sample_rate = sample_rate
    if bitrate is not None:
        config.bitrate = bitrate
    config.validate()


def validate_ply_format_options(format_name: str, precision: int | None = None) -> None:
    """Validate PLY format options.

    Args:
        format_name: PLY format name
        precision: Floating point precision, optional

    Raises:
        InvalidConfigurationError: If validation fails
    """
    config = MeshFormatConfig(format=format_name)
    if precision is not None:
        config.precision = precision
    config.validate()


def validate_text_format_options(
    encoding: str | None = None,
    line_ending: str | None = None,
    max_line_length: int | None = None,
) -> None:
    """Validate text format options.

    Args:
        encoding: Text encoding, optional
        line_ending: Line ending style, optional
        max_line_length: Maximum line length, optional

    Raises:
        InvalidConfigurationError: If validation fails
    """
    config = TextFormatConfig()
    if encoding is not None:
        config.encoding = encoding
    if line_ending is not None:
        config.line_ending = line_ending
    if max_line_length is not None:
        config.max_line_length = max_line_length
    config.validate()


def validate_data_format_options(
    csv_delimiter: str | None = None,
    csv_quote_char: str | None = None,
    csv_escape_char: str | None = None,
    json_indent: int | None = None,
    json_sort_keys: bool | None = None,
) -> None:
    """Validate structured data format options.

    Args:
        csv_delimiter: CSV delimiter character, optional
        csv_quote_char: CSV quote character, optional
        csv_escape_char: CSV escape character, optional
        json_indent: JSON indentation level, optional
        json_sort_keys: Whether to sort JSON keys, optional

    Raises:
        InvalidConfigurationError: If validation fails
    """
    config = DataFormatConfig()
    if csv_delimiter is not None:
        config.csv_delimiter = csv_delimiter
    if csv_quote_char is not None:
        config.csv_quote_char = csv_quote_char
    if csv_escape_char is not None:
        config.csv_escape_char = csv_escape_char
    if json_indent is not None:
        config.json_indent = json_indent
    if json_sort_keys is not None:
        config.json_sort_keys = json_sort_keys
    config.validate()
