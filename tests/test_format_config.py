"""Tests for format configuration management."""

import pytest

from tboardfs.core.format_config import (
    ImageFormatConfig,
    AudioFormatConfig,
    MeshFormatConfig,
    TextFormatConfig,
    DataFormatConfig,
    FormatConfig,
    validate_image_format_options,
    validate_audio_format_options,
    validate_ply_format_options,
    validate_text_format_options,
    validate_data_format_options,
)
from tboardfs.core.constants import ImageFormats, AudioFormats, PLYFormats
from tboardfs.core.exceptions import InvalidConfigurationError


class TestImageFormatConfig:
    """Test ImageFormatConfig class."""

    def test_default_initialization(self):
        """Test default configuration values."""
        config = ImageFormatConfig()

        assert config.format == ImageFormats.DEFAULT_FORMAT
        assert config.quality == ImageFormats.DEFAULT_QUALITY

    def test_custom_initialization(self):
        """Test configuration with custom values."""
        config = ImageFormatConfig(format="png", quality=95)

        assert config.format == "png"
        assert config.quality == 95

    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = ImageFormatConfig(format="png", quality=90)
        config.validate()  # Should not raise

    def test_validate_invalid_format(self):
        """Test validation with invalid format."""
        config = ImageFormatConfig(format="invalid")

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "image_format" in str(exc_info.value)
        assert "invalid" in str(exc_info.value)

    def test_validate_invalid_quality_below_range(self):
        """Test validation with quality below valid range."""
        config = ImageFormatConfig(quality=-1)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "image_quality" in str(exc_info.value)

    def test_validate_invalid_quality_above_range(self):
        """Test validation with quality above valid range."""
        config = ImageFormatConfig(quality=101)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "image_quality" in str(exc_info.value)

    def test_supported_formats(self):
        """Test all supported formats validate correctly."""
        supported_formats = ["png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp"]

        for format_name in supported_formats:
            config = ImageFormatConfig(format=format_name)
            config.validate()  # Should not raise


class TestAudioFormatConfig:
    """Test AudioFormatConfig class."""

    def test_default_initialization(self):
        """Test default configuration values."""
        config = AudioFormatConfig()

        assert config.format == AudioFormats.DEFAULT_FORMAT
        assert config.sample_rate == AudioFormats.DEFAULT_SAMPLE_RATE
        assert config.bitrate == 128

    def test_custom_initialization(self):
        """Test configuration with custom values."""
        config = AudioFormatConfig(format="mp3", sample_rate=48000.0, bitrate=256)

        assert config.format == "mp3"
        assert config.sample_rate == 48000.0
        assert config.bitrate == 256

    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = AudioFormatConfig(format="wav", sample_rate=44100.0, bitrate=128)
        config.validate()  # Should not raise

    def test_validate_invalid_format(self):
        """Test validation with invalid format."""
        config = AudioFormatConfig(format="invalid")

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "audio_format" in str(exc_info.value)
        assert "invalid" in str(exc_info.value)

    def test_validate_invalid_sample_rate_below_range(self):
        """Test validation with sample rate below valid range."""
        config = AudioFormatConfig(sample_rate=1000.0)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "audio_sample_rate" in str(exc_info.value)

    def test_validate_invalid_sample_rate_above_range(self):
        """Test validation with sample rate above valid range."""
        config = AudioFormatConfig(sample_rate=200000.0)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "audio_sample_rate" in str(exc_info.value)

    def test_validate_invalid_bitrate_below_range(self):
        """Test validation with bitrate below valid range."""
        config = AudioFormatConfig(bitrate=16)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "audio_bitrate" in str(exc_info.value)

    def test_validate_invalid_bitrate_above_range(self):
        """Test validation with bitrate above valid range."""
        config = AudioFormatConfig(bitrate=512)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "audio_bitrate" in str(exc_info.value)

    def test_supported_formats(self):
        """Test all supported formats validate correctly."""
        supported_formats = ["wav", "mp3", "flac", "ogg"]

        for format_name in supported_formats:
            config = AudioFormatConfig(format=format_name)
            config.validate()  # Should not raise


class TestMeshFormatConfig:
    """Test MeshFormatConfig class."""

    def test_default_initialization(self):
        """Test default configuration values."""
        config = MeshFormatConfig()

        assert config.format == PLYFormats.DEFAULT_FORMAT
        assert config.precision == 6

    def test_custom_initialization(self):
        """Test configuration with custom values."""
        config = MeshFormatConfig(format="text", precision=8)

        assert config.format == "text"
        assert config.precision == 8

    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = MeshFormatConfig(format="binary", precision=6)
        config.validate()  # Should not raise

    def test_validate_invalid_format(self):
        """Test validation with invalid format."""
        config = MeshFormatConfig(format="invalid")

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "ply_format" in str(exc_info.value)
        assert "invalid" in str(exc_info.value)

    def test_validate_invalid_precision_below_range(self):
        """Test validation with precision below valid range."""
        config = MeshFormatConfig(precision=0)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "ply_precision" in str(exc_info.value)

    def test_validate_invalid_precision_above_range(self):
        """Test validation with precision above valid range."""
        config = MeshFormatConfig(precision=20)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "ply_precision" in str(exc_info.value)

    def test_supported_formats(self):
        """Test all supported formats validate correctly."""
        supported_formats = ["binary", "text", "ascii"]

        for format_name in supported_formats:
            config = MeshFormatConfig(format=format_name)
            config.validate()  # Should not raise


class TestTextFormatConfig:
    """Test TextFormatConfig class."""

    def test_default_initialization(self):
        """Test default configuration values."""
        config = TextFormatConfig()

        assert config.encoding == "utf-8"
        assert config.line_ending == "\n"
        assert config.max_line_length == 120

    def test_custom_initialization(self):
        """Test configuration with custom values."""
        config = TextFormatConfig(
            encoding="ascii", line_ending="\r\n", max_line_length=80
        )

        assert config.encoding == "ascii"
        assert config.line_ending == "\r\n"
        assert config.max_line_length == 80

    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = TextFormatConfig(
            encoding="utf-8", line_ending="\n", max_line_length=120
        )
        config.validate()  # Should not raise

    def test_validate_invalid_encoding(self):
        """Test validation with invalid encoding."""
        config = TextFormatConfig(encoding="invalid")

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "text_encoding" in str(exc_info.value)
        assert "invalid" in str(exc_info.value)

    def test_validate_invalid_line_ending(self):
        """Test validation with invalid line ending."""
        config = TextFormatConfig(line_ending="invalid")

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "text_line_ending" in str(exc_info.value)

    def test_validate_invalid_max_line_length_below_range(self):
        """Test validation with max line length below valid range."""
        config = TextFormatConfig(max_line_length=5)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "text_max_line_length" in str(exc_info.value)

    def test_validate_invalid_max_line_length_above_range(self):
        """Test validation with max line length above valid range."""
        config = TextFormatConfig(max_line_length=2000)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "text_max_line_length" in str(exc_info.value)

    def test_supported_encodings(self):
        """Test all supported encodings validate correctly."""
        supported_encodings = ["utf-8", "ascii", "latin-1", "utf-16", "utf-32"]

        for encoding in supported_encodings:
            config = TextFormatConfig(encoding=encoding)
            config.validate()  # Should not raise

    def test_supported_line_endings(self):
        """Test all supported line endings validate correctly."""
        supported_line_endings = ["\n", "\r\n", "\r"]

        for line_ending in supported_line_endings:
            config = TextFormatConfig(line_ending=line_ending)
            config.validate()  # Should not raise


class TestDataFormatConfig:
    """Test DataFormatConfig class."""

    def test_default_initialization(self):
        """Test default configuration values."""
        config = DataFormatConfig()

        assert config.csv_delimiter == ","
        assert config.csv_quote_char == '"'
        assert config.csv_escape_char == "\\"
        assert config.json_indent == 2
        assert config.json_sort_keys is True

    def test_custom_initialization(self):
        """Test configuration with custom values."""
        config = DataFormatConfig(
            csv_delimiter=";",
            csv_quote_char="'",
            csv_escape_char="/",
            json_indent=4,
            json_sort_keys=False,
        )

        assert config.csv_delimiter == ";"
        assert config.csv_quote_char == "'"
        assert config.csv_escape_char == "/"
        assert config.json_indent == 4
        assert config.json_sort_keys is False

    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = DataFormatConfig()
        config.validate()  # Should not raise

    def test_validate_invalid_csv_delimiter(self):
        """Test validation with invalid CSV delimiter."""
        config = DataFormatConfig(csv_delimiter="x")

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "csv_delimiter" in str(exc_info.value)

    def test_validate_invalid_csv_quote_char_empty(self):
        """Test validation with empty CSV quote char."""
        config = DataFormatConfig(csv_quote_char="")

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "csv_quote_char" in str(exc_info.value)

    def test_validate_invalid_csv_quote_char_multiple(self):
        """Test validation with multiple character CSV quote char."""
        config = DataFormatConfig(csv_quote_char="ab")

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "csv_quote_char" in str(exc_info.value)

    def test_validate_invalid_csv_escape_char_empty(self):
        """Test validation with empty CSV escape char."""
        config = DataFormatConfig(csv_escape_char="")

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "csv_escape_char" in str(exc_info.value)

    def test_validate_invalid_csv_escape_char_multiple(self):
        """Test validation with multiple character CSV escape char."""
        config = DataFormatConfig(csv_escape_char="ab")

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "csv_escape_char" in str(exc_info.value)

    def test_validate_invalid_json_indent_below_range(self):
        """Test validation with JSON indent below valid range."""
        config = DataFormatConfig(json_indent=-1)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "json_indent" in str(exc_info.value)

    def test_validate_invalid_json_indent_above_range(self):
        """Test validation with JSON indent above valid range."""
        config = DataFormatConfig(json_indent=10)

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "json_indent" in str(exc_info.value)

    def test_supported_csv_delimiters(self):
        """Test all supported CSV delimiters validate correctly."""
        supported_delimiters = [",", ";", "\t", "|"]

        for delimiter in supported_delimiters:
            config = DataFormatConfig(csv_delimiter=delimiter)
            config.validate()  # Should not raise


class TestFormatConfig:
    """Test FormatConfig class."""

    def test_default_initialization(self):
        """Test default configuration values."""
        config = FormatConfig()

        assert isinstance(config.image, ImageFormatConfig)
        assert isinstance(config.audio, AudioFormatConfig)
        assert isinstance(config.mesh, MeshFormatConfig)
        assert isinstance(config.text, TextFormatConfig)
        assert isinstance(config.data, DataFormatConfig)

    def test_custom_initialization(self):
        """Test configuration with custom sub-configs."""
        image_config = ImageFormatConfig(format="png")
        audio_config = AudioFormatConfig(format="mp3")

        config = FormatConfig(image=image_config, audio=audio_config)

        assert config.image.format == "png"
        assert config.audio.format == "mp3"

    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = FormatConfig()
        config.validate()  # Should not raise

    def test_validate_invalid_subconfig(self):
        """Test validation with invalid sub-configuration."""
        config = FormatConfig()
        config.image.format = "invalid"

        with pytest.raises(InvalidConfigurationError):
            config.validate()

    def test_from_dict_minimal(self):
        """Test creating configuration from minimal dictionary."""
        config_dict = {}

        config = FormatConfig.from_dict(config_dict)

        assert isinstance(config.image, ImageFormatConfig)
        assert isinstance(config.audio, AudioFormatConfig)
        assert isinstance(config.mesh, MeshFormatConfig)
        assert isinstance(config.text, TextFormatConfig)
        assert isinstance(config.data, DataFormatConfig)

    def test_from_dict_partial(self):
        """Test creating configuration from partial dictionary."""
        config_dict = {
            "image": {"format": "png", "quality": 95},
            "audio": {"format": "wav"},
        }

        config = FormatConfig.from_dict(config_dict)

        assert config.image.format == "png"
        assert config.image.quality == 95
        assert config.audio.format == "wav"
        assert config.audio.sample_rate == AudioFormats.DEFAULT_SAMPLE_RATE  # Default

    def test_from_dict_full(self):
        """Test creating configuration from full dictionary."""
        config_dict = {
            "image": {"format": "png", "quality": 95},
            "audio": {"format": "wav", "sample_rate": 48000.0, "bitrate": 256},
            "mesh": {"format": "text", "precision": 8},
            "text": {"encoding": "ascii", "line_ending": "\r\n", "max_line_length": 80},
            "data": {"csv_delimiter": ";", "json_indent": 4, "json_sort_keys": False},
        }

        config = FormatConfig.from_dict(config_dict)

        assert config.image.format == "png"
        assert config.image.quality == 95
        assert config.audio.format == "wav"
        assert config.audio.sample_rate == 48000.0
        assert config.audio.bitrate == 256
        assert config.mesh.format == "text"
        assert config.mesh.precision == 8
        assert config.text.encoding == "ascii"
        assert config.text.line_ending == "\r\n"
        assert config.text.max_line_length == 80
        assert config.data.csv_delimiter == ";"
        assert config.data.json_indent == 4
        assert config.data.json_sort_keys is False

    def test_from_dict_invalid_config(self):
        """Test creating configuration from invalid dictionary."""
        config_dict = {"image": {"format": "invalid"}}

        with pytest.raises(InvalidConfigurationError):
            FormatConfig.from_dict(config_dict)

    def test_from_dict_malformed(self):
        """Test creating configuration from malformed dictionary."""
        config_dict = {"image": "not_a_dict"}

        with pytest.raises(InvalidConfigurationError):
            FormatConfig.from_dict(config_dict)

    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = FormatConfig()
        config.image.format = "png"
        config.image.quality = 95
        config.audio.format = "wav"

        result = config.to_dict()

        assert result["image"]["format"] == "png"
        assert result["image"]["quality"] == 95
        assert result["audio"]["format"] == "wav"

    def test_roundtrip_dict_conversion(self):
        """Test that to_dict -> from_dict preserves configuration."""
        original_config = FormatConfig()
        original_config.image.format = "png"
        original_config.image.quality = 95
        original_config.audio.format = "wav"
        original_config.mesh.precision = 8

        config_dict = original_config.to_dict()
        restored_config = FormatConfig.from_dict(config_dict)

        assert restored_config.image.format == original_config.image.format
        assert restored_config.image.quality == original_config.image.quality
        assert restored_config.audio.format == original_config.audio.format
        assert restored_config.mesh.precision == original_config.mesh.precision


class TestValidationFunctions:
    """Test standalone validation functions."""

    def test_validate_image_format_options_valid(self):
        """Test valid image format validation."""
        validate_image_format_options("png", 90)  # Should not raise

    def test_validate_image_format_options_invalid_format(self):
        """Test invalid image format validation."""
        with pytest.raises(InvalidConfigurationError):
            validate_image_format_options("invalid")

    def test_validate_image_format_options_invalid_quality(self):
        """Test invalid image quality validation."""
        with pytest.raises(InvalidConfigurationError):
            validate_image_format_options("png", 150)

    def test_validate_audio_format_options_valid(self):
        """Test valid audio format validation."""
        validate_audio_format_options("wav", 44100.0, 128)  # Should not raise

    def test_validate_audio_format_options_invalid_format(self):
        """Test invalid audio format validation."""
        with pytest.raises(InvalidConfigurationError):
            validate_audio_format_options("invalid")

    def test_validate_audio_format_options_invalid_sample_rate(self):
        """Test invalid audio sample rate validation."""
        with pytest.raises(InvalidConfigurationError):
            validate_audio_format_options("wav", 1000.0)

    def test_validate_audio_format_options_invalid_bitrate(self):
        """Test invalid audio bitrate validation."""
        with pytest.raises(InvalidConfigurationError):
            validate_audio_format_options("mp3", None, 512)

    def test_validate_ply_format_options_valid(self):
        """Test valid PLY format validation."""
        validate_ply_format_options("binary", 6)  # Should not raise

    def test_validate_ply_format_options_invalid_format(self):
        """Test invalid PLY format validation."""
        with pytest.raises(InvalidConfigurationError):
            validate_ply_format_options("invalid")

    def test_validate_ply_format_options_invalid_precision(self):
        """Test invalid PLY precision validation."""
        with pytest.raises(InvalidConfigurationError):
            validate_ply_format_options("binary", 20)

    def test_validate_text_format_options_valid(self):
        """Test valid text format validation."""
        validate_text_format_options("utf-8", "\n", 120)  # Should not raise

    def test_validate_text_format_options_invalid_encoding(self):
        """Test invalid text encoding validation."""
        with pytest.raises(InvalidConfigurationError):
            validate_text_format_options(encoding="invalid")

    def test_validate_text_format_options_invalid_line_ending(self):
        """Test invalid text line ending validation."""
        with pytest.raises(InvalidConfigurationError):
            validate_text_format_options(line_ending="invalid")

    def test_validate_text_format_options_invalid_max_line_length(self):
        """Test invalid text max line length validation."""
        with pytest.raises(InvalidConfigurationError):
            validate_text_format_options(max_line_length=5)

    def test_validate_data_format_options_valid(self):
        """Test valid data format validation."""
        validate_data_format_options(",", '"', "\\", 2, True)  # Should not raise

    def test_validate_data_format_options_invalid_delimiter(self):
        """Test invalid data delimiter validation."""
        with pytest.raises(InvalidConfigurationError):
            validate_data_format_options(csv_delimiter="x")

    def test_validate_data_format_options_invalid_quote_char(self):
        """Test invalid data quote char validation."""
        with pytest.raises(InvalidConfigurationError):
            validate_data_format_options(csv_quote_char="")

    def test_validate_data_format_options_invalid_escape_char(self):
        """Test invalid data escape char validation."""
        with pytest.raises(InvalidConfigurationError):
            validate_data_format_options(csv_escape_char="ab")

    def test_validate_data_format_options_invalid_json_indent(self):
        """Test invalid JSON indent validation."""
        with pytest.raises(InvalidConfigurationError):
            validate_data_format_options(json_indent=10)
