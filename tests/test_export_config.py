"""Tests for export configuration management."""

import pytest
from pathlib import Path

from tboardfs.core.export_config import (
    ExportFormatConfig,
    OutputOrganizationConfig,
    TBoardFSExportConfig,
    create_minimal_export_config,
    create_high_quality_export_config,
    create_fast_export_config,
)
from tboardfs.core.constants import ImageFormats, AudioFormats, PLYFormats
from tboardfs.core.exceptions import InvalidConfigurationError


class TestExportFormatConfig:
    """Test ExportFormatConfig class."""

    def test_default_initialization(self):
        """Test default configuration values."""
        config = ExportFormatConfig()

        assert config.image_format == ImageFormats.DEFAULT_FORMAT
        assert config.image_quality == ImageFormats.DEFAULT_QUALITY
        assert config.audio_format == AudioFormats.DEFAULT_FORMAT
        assert config.audio_sample_rate == AudioFormats.DEFAULT_SAMPLE_RATE
        assert config.ply_format == PLYFormats.DEFAULT_FORMAT
        assert config.step_digits == 6  # TensorBoardConstants.DEFAULT_STEP_DIGITS

    def test_custom_initialization(self):
        """Test configuration with custom values."""
        config = ExportFormatConfig(
            image_format="png",
            image_quality=95,
            audio_format="wav",
            audio_sample_rate=48000.0,
            ply_format="binary",
            step_digits=8,
        )

        assert config.image_format == "png"
        assert config.image_quality == 95
        assert config.audio_format == "wav"
        assert config.audio_sample_rate == 48000.0
        assert config.ply_format == "binary"
        assert config.step_digits == 8

    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = ExportFormatConfig(
            image_format="png",
            image_quality=90,
            audio_format="wav",
            audio_sample_rate=44100.0,
            ply_format="binary",
            step_digits=6,
        )

        # Should not raise exception
        config.validate()

    def test_validate_invalid_image_format(self):
        """Test validation with invalid image format."""
        config = ExportFormatConfig(image_format="invalid")

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "image_format" in str(exc_info.value)
        assert "invalid" in str(exc_info.value)

    def test_validate_invalid_image_quality(self):
        """Test validation with invalid image quality."""
        # Test below range
        config = ExportFormatConfig(image_quality=-1)
        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()
        assert "image_quality" in str(exc_info.value)

        # Test above range
        config = ExportFormatConfig(image_quality=101)
        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()
        assert "image_quality" in str(exc_info.value)

    def test_validate_invalid_audio_format(self):
        """Test validation with invalid audio format."""
        config = ExportFormatConfig(audio_format="invalid")

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "audio_format" in str(exc_info.value)
        assert "invalid" in str(exc_info.value)

    def test_validate_invalid_audio_sample_rate(self):
        """Test validation with invalid audio sample rate."""
        # Test below range
        config = ExportFormatConfig(audio_sample_rate=100.0)
        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()
        assert "audio_sample_rate" in str(exc_info.value)

        # Test above range
        config = ExportFormatConfig(audio_sample_rate=200000.0)
        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()
        assert "audio_sample_rate" in str(exc_info.value)

    def test_validate_invalid_ply_format(self):
        """Test validation with invalid PLY format."""
        config = ExportFormatConfig(ply_format="invalid")

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "ply_format" in str(exc_info.value)
        assert "invalid" in str(exc_info.value)

    def test_validate_invalid_step_digits(self):
        """Test validation with invalid step digits."""
        # Test below range
        config = ExportFormatConfig(step_digits=0)
        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()
        assert "step_digits" in str(exc_info.value)

        # Test above range
        config = ExportFormatConfig(step_digits=21)
        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()
        assert "step_digits" in str(exc_info.value)


class TestOutputOrganizationConfig:
    """Test OutputOrganizationConfig class."""

    def test_default_initialization(self):
        """Test default configuration values."""
        config = OutputOrganizationConfig()

        assert config.create_step_directories is True
        assert config.aggregate_scalars is True
        assert config.aggregate_histograms is True
        assert config.use_zero_padding is True
        assert config.sanitize_tags is True
        assert config.scalar_formats == ["txt", "csv"]
        assert config.histogram_formats == ["csv", "npz"]
        assert config.image_formats == ["png"]

    def test_custom_initialization(self):
        """Test configuration with custom values."""
        config = OutputOrganizationConfig(
            create_step_directories=True,
            aggregate_scalars=False,
            scalar_formats=["json"],
            image_formats=["jpg", "png"],
        )

        assert config.create_step_directories is True
        assert config.aggregate_scalars is False
        assert config.scalar_formats == ["json"]
        assert config.image_formats == ["jpg", "png"]

    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = OutputOrganizationConfig(
            scalar_formats=["txt", "csv", "json"],
            image_formats=["png", "jpg"],
            audio_formats=["wav", "mp3"],
        )

        # Should not raise exception
        config.validate()

    def test_validate_invalid_scalar_format(self):
        """Test validation with invalid scalar format."""
        config = OutputOrganizationConfig(scalar_formats=["invalid"])

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "scalar_formats" in str(exc_info.value)
        assert "invalid" in str(exc_info.value)

    def test_validate_invalid_image_format(self):
        """Test validation with invalid image format."""
        config = OutputOrganizationConfig(image_formats=["invalid"])

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "image_formats" in str(exc_info.value)
        assert "invalid" in str(exc_info.value)

    def test_validate_mixed_valid_invalid_formats(self):
        """Test validation with mix of valid and invalid formats."""
        config = OutputOrganizationConfig(scalar_formats=["txt", "invalid"])

        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()

        assert "scalar_formats" in str(exc_info.value)
        assert "invalid" in str(exc_info.value)


class TestTBoardFSExportConfig:
    """Test TBoardFSExportConfig class."""

    def test_default_initialization(self):
        """Test default configuration values."""
        config = TBoardFSExportConfig()

        assert config.output_path == Path.cwd() / "exported_data"
        assert isinstance(config.export_formats, ExportFormatConfig)
        assert isinstance(config.output_organization, OutputOrganizationConfig)

    def test_custom_initialization(self):
        """Test configuration with custom values."""
        output_path = Path("/tmp/test")
        export_formats = ExportFormatConfig(image_format="png")
        output_org = OutputOrganizationConfig(create_step_directories=True)

        config = TBoardFSExportConfig(
            output_path=output_path,
            export_formats=export_formats,
            output_organization=output_org,
        )

        assert config.output_path == output_path
        assert config.export_formats.image_format == "png"
        assert config.output_organization.create_step_directories is True

    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = TBoardFSExportConfig()

        # Should not raise exception
        config.validate()

    def test_validate_invalid_export_formats(self):
        """Test validation with invalid export formats."""
        config = TBoardFSExportConfig()
        config.export_formats.image_format = "invalid"

        with pytest.raises(InvalidConfigurationError):
            config.validate()

    def test_validate_invalid_output_organization(self):
        """Test validation with invalid output organization."""
        config = TBoardFSExportConfig()
        config.output_organization.scalar_formats = ["invalid"]

        with pytest.raises(InvalidConfigurationError):
            config.validate()

    def test_from_dict_minimal(self):
        """Test creating configuration from minimal dictionary."""
        config_dict = {"output_path": "/tmp/test"}

        config = TBoardFSExportConfig.from_dict(config_dict)

        assert config.output_path == Path("/tmp/test")
        assert isinstance(config.export_formats, ExportFormatConfig)
        assert isinstance(config.output_organization, OutputOrganizationConfig)

    def test_from_dict_full(self):
        """Test creating configuration from full dictionary."""
        config_dict = {
            "output_path": "/tmp/test",
            "export_formats": {
                "image_format": "png",
                "image_quality": 95,
                "audio_format": "wav",
                "step_digits": 8,
            },
            "output_organization": {
                "create_step_directories": True,
                "aggregate_scalars": False,
                "scalar_formats": ["json"],
            },
        }

        config = TBoardFSExportConfig.from_dict(config_dict)

        assert config.output_path == Path("/tmp/test")
        assert config.export_formats.image_format == "png"
        assert config.export_formats.image_quality == 95
        assert config.export_formats.audio_format == "wav"
        assert config.export_formats.step_digits == 8
        assert config.output_organization.create_step_directories is True
        assert config.output_organization.aggregate_scalars is False
        assert config.output_organization.scalar_formats == ["json"]

    def test_from_dict_invalid_config(self):
        """Test creating configuration from invalid dictionary."""
        config_dict = {"export_formats": {"image_format": "invalid"}}

        with pytest.raises(InvalidConfigurationError):
            TBoardFSExportConfig.from_dict(config_dict)

    def test_from_dict_malformed(self):
        """Test creating configuration from malformed dictionary."""
        config_dict = {"export_formats": "not_a_dict"}

        with pytest.raises(InvalidConfigurationError):
            TBoardFSExportConfig.from_dict(config_dict)

    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = TBoardFSExportConfig(output_path=Path("/tmp/test"))
        config.export_formats.image_format = "png"
        config.export_formats.image_quality = 95
        config.output_organization.create_step_directories = True

        result = config.to_dict()

        assert result["output_path"] == "/tmp/test"
        assert result["export_formats"]["image_format"] == "png"
        assert result["export_formats"]["image_quality"] == 95
        assert result["output_organization"]["create_step_directories"] is True

    def test_roundtrip_dict_conversion(self):
        """Test that to_dict -> from_dict preserves configuration."""
        original_config = TBoardFSExportConfig(output_path=Path("/tmp/test"))
        original_config.export_formats.image_format = "png"
        original_config.export_formats.image_quality = 95
        original_config.output_organization.create_step_directories = True

        config_dict = original_config.to_dict()
        restored_config = TBoardFSExportConfig.from_dict(config_dict)

        assert restored_config.output_path == original_config.output_path
        assert (
            restored_config.export_formats.image_format
            == original_config.export_formats.image_format
        )
        assert (
            restored_config.export_formats.image_quality
            == original_config.export_formats.image_quality
        )
        assert (
            restored_config.output_organization.create_step_directories
            == original_config.output_organization.create_step_directories
        )


class TestConvenienceFunctions:
    """Test convenience functions for creating export configurations."""

    def test_create_minimal_export_config(self):
        """Test creating minimal export configuration."""
        output_path = "/tmp/test"

        config = create_minimal_export_config(output_path)

        assert config.output_path == Path(output_path)
        assert isinstance(config.export_formats, ExportFormatConfig)
        assert isinstance(config.output_organization, OutputOrganizationConfig)

        # Should validate without errors
        config.validate()

    def test_create_high_quality_export_config(self):
        """Test creating high-quality export configuration."""
        output_path = "/tmp/test"

        config = create_high_quality_export_config(output_path)

        assert config.output_path == Path(output_path)
        assert config.export_formats.image_format == ImageFormats.PNG
        assert config.export_formats.image_quality == 95
        assert config.export_formats.audio_format == AudioFormats.WAV
        assert config.export_formats.ply_format == PLYFormats.BINARY
        assert config.output_organization.create_step_directories is True
        assert config.output_organization.aggregate_scalars is False
        assert config.output_organization.aggregate_histograms is False

        # Should validate without errors
        config.validate()

    def test_create_fast_export_config(self):
        """Test creating fast export configuration."""
        output_path = "/tmp/test"

        config = create_fast_export_config(output_path)

        assert config.output_path == Path(output_path)
        assert config.export_formats.image_format == ImageFormats.JPG
        assert config.export_formats.image_quality == 75
        assert config.export_formats.audio_format == AudioFormats.MP3
        assert config.export_formats.ply_format == PLYFormats.BINARY
        assert config.output_organization.create_step_directories is False
        assert config.output_organization.aggregate_scalars is True
        assert config.output_organization.aggregate_histograms is True
        assert config.output_organization.scalar_formats == ["txt"]
        assert config.output_organization.histogram_formats == ["npz"]

        # Should validate without errors
        config.validate()

    def test_convenience_functions_with_path_objects(self):
        """Test convenience functions with Path objects."""
        output_path = Path("/tmp/test")

        configs = [
            create_minimal_export_config(output_path),
            create_high_quality_export_config(output_path),
            create_fast_export_config(output_path),
        ]

        for config in configs:
            assert config.output_path == output_path
            config.validate()  # Should all validate without errors
