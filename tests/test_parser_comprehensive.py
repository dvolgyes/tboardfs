"""Comprehensive tests for TensorBoard parser using real data and error handling."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from tboardfs.efficient_parser import EfficientTensorBoardParser
from tboardfs.core.data_types import ImageData


class TestTensorBoardParserWithRealData:
    """Test parser with real TensorBoard log files from example-logs/."""

    @pytest.fixture
    def real_log_file(self):
        """Use a real TensorBoard log file from test-logs."""
        log_path = Path(
            "test-logs/images/events.out.tfevents.1751399792.FG-OSL-WS122.70334.26"
        )
        if log_path.exists():
            return str(log_path)
        pytest.skip("Real log file not found")

    def test_parser_with_real_data(self, real_log_file):
        """Test parser functionality with real TensorBoard data."""
        parser = EfficientTensorBoardParser(real_log_file)

        # Test basic listing
        content = parser.list_all_content()
        assert isinstance(content, dict)
        assert "scalars" in content
        assert "images" in content
        assert "histograms" in content
        assert "audio" in content
        assert "text" in content
        assert "tensors" in content

        # Test image data (we know this file has images)
        images = parser.list_images()
        assert isinstance(images, list)
        assert len(images) > 0

        # Test image data retrieval
        if images:
            image_data = list(parser.iterate_image_data(images[0]))
            assert isinstance(image_data, list)
            assert len(image_data) > 0
            assert all(isinstance(item, ImageData) for item in image_data)

            # Test image export
            first_image = image_data[0]
            # Find the image data for the specific step
            exported_image = None
            for data in parser.iterate_image_data(images[0]):
                if data.step == first_image.step:
                    exported_image = data.encoded_image_string
                    break
            assert exported_image is not None
            assert isinstance(exported_image, bytes)
            assert len(exported_image) > 0

            # Test image extension detection
            from tboardfs.core.image_processor import ImageProcessor

            ext = ImageProcessor.get_image_extension(exported_image)
            assert ext in ["png", "jpg", "bin"]

    def test_virtual_paths_with_real_data(self, real_log_file):
        """Test virtual path generation with real data."""
        parser = EfficientTensorBoardParser(real_log_file)

        paths = parser.get_virtual_paths()
        assert isinstance(paths, list)
        assert len(paths) > 0

        # Check for expected directory structure
        expected_dirs = ["scalars/", "images/", "histograms/", "audio/", "text/"]
        for dir_path in expected_dirs:
            assert dir_path in paths

        # Test custom digits
        paths_custom = parser.get_virtual_paths(digits=8)
        assert isinstance(paths_custom, list)
        assert len(paths_custom) > 0

    def test_extraction_with_real_data(self, real_log_file):
        """Test data extraction with real data."""
        parser = EfficientTensorBoardParser(real_log_file)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test extraction
            parser.extract_all_to_directory(tmp_dir)

            output_path = Path(tmp_dir)
            assert output_path.exists()

            # Check that only directories with actual content are created
            images = parser.list_images()
            scalars = parser.list_scalars()
            histograms = parser.list_histograms()
            audio = parser.list_audio()
            text = parser.list_text()

            # Only check for directories that should contain data
            if scalars:
                assert (output_path / "scalars").exists()
            else:
                assert not (output_path / "scalars").exists()

            if images:
                assert (output_path / "images").exists()
            else:
                assert not (output_path / "images").exists()

            if histograms:
                assert (output_path / "histograms").exists()
            else:
                assert not (output_path / "histograms").exists()

            if audio:
                assert (output_path / "audio").exists()
            else:
                assert not (output_path / "audio").exists()

            if text:
                assert (output_path / "text").exists()
            else:
                assert not (output_path / "text").exists()

            # Check that image files were created
            if images:
                # Find image directory
                for tag in images:
                    safe_tag = tag.replace("/", "_")
                    tag_dir = output_path / "images" / safe_tag
                    if tag_dir.exists():
                        image_files = list(tag_dir.glob("*.png")) + list(
                            tag_dir.glob("*.jpg")
                        )
                        assert len(image_files) > 0

    def test_progress_bar_functionality(self, real_log_file):
        """Test progress bar integration."""
        # Test with progress disabled
        parser_no_progress = EfficientTensorBoardParser(
            real_log_file, show_progress=False
        )
        paths_no_progress = parser_no_progress.get_virtual_paths()

        # Test with progress enabled
        parser_with_progress = EfficientTensorBoardParser(
            real_log_file, show_progress=True
        )
        paths_with_progress = parser_with_progress.get_virtual_paths()

        # Results should be the same regardless of progress setting
        assert len(paths_no_progress) == len(paths_with_progress)


class TestTensorBoardParserErrorHandling:
    """Test error handling and edge cases in TensorBoard parser."""

    def test_invalid_file_path(self):
        """Test parser with invalid file path."""
        with pytest.raises(Exception):
            EfficientTensorBoardParser("nonexistent_file.tfevents")

    def test_empty_file(self):
        """Test parser with empty file."""
        with tempfile.NamedTemporaryFile(suffix=".tfevents") as tmp_file:
            # Create empty file
            tmp_file.write(b"")
            tmp_file.flush()

            # This should not crash but return empty results
            parser = EfficientTensorBoardParser(tmp_file.name)
            assert parser.list_scalars() == []
            assert parser.list_images() == []
            assert parser.list_histograms() == []
            assert parser.list_audio() == []

    def test_nonexistent_tag_handling(self):
        """Test handling of nonexistent tags."""
        # Use a real file but test with nonexistent tags
        log_path = Path(
            "test-logs/images/events.out.tfevents.1751399792.FG-OSL-WS122.70334.26"
        )
        if not log_path.exists():
            pytest.skip("Real log file not found")

        parser = EfficientTensorBoardParser(str(log_path))

        # Test with nonexistent scalar tag - should return empty list
        scalar_data = list(parser.iterate_scalar_data("nonexistent_scalar"))
        assert scalar_data == []

        # Test with nonexistent image tag - should return empty list
        image_data = list(parser.iterate_image_data("nonexistent_image"))
        assert image_data == []

        # Test export with nonexistent tag - should return None
        # Try to find nonexistent image
        export_result = None
        for data in parser.iterate_image_data("nonexistent_image"):
            if data.step == 0:
                export_result = data.encoded_image_string
                break
        assert export_result is None

        # Test export with nonexistent step
        images = parser.list_images()
        if images:
            # Try to find image with very high step
            result = None
            for data in parser.iterate_image_data(images[0]):
                if data.step == 999999:  # Very high step
                    result = data.encoded_image_string
                    break
            assert result is None

    def test_text_data_exception_handling(self):
        """Test exception handling in text data processing."""
        # Mock the event accumulator to simulate errors
        with patch(
            "tensorboard.backend.event_processing.event_accumulator.EventAccumulator"
        ) as mock_ea_class:
            mock_ea = MagicMock()
            mock_ea_class.return_value = mock_ea

            # Mock Tags() to return some tensors
            mock_ea.Tags.return_value = {"tensors": ["test/tag"]}

            # Mock Tensors() to raise an exception
            mock_ea.Tensors.side_effect = Exception("Test exception")

            # Create a temp file for testing
            with tempfile.NamedTemporaryFile(
                suffix=".tfevents", delete=False
            ) as tmp_file:
                tmp_file.write(b"\x08\x01\x12\x00\x1a\x00")
                tmp_file.flush()
                parser = EfficientTensorBoardParser(tmp_file.name)

                # This should handle the exception gracefully and return empty list
                text_tags = parser.list_text()
                assert text_tags == []

                # Test get_text_data exception handling
                text_data = list(parser.iterate_text_data("test/tag"))
                assert text_data == []

            # Clean up
            Path(tmp_file.name).unlink()

    def test_image_extension_detection(self):
        """Test image extension detection for various formats."""
        # Test PNG detection with proper header
        from tboardfs.core.image_processor import ImageProcessor

        png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        assert ImageProcessor.get_image_extension(png_bytes) == "png"

        # Test JPEG detection with proper header
        jpg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C"
        assert ImageProcessor.get_image_extension(jpg_bytes) == "jpg"

        # Test unknown format
        unknown_bytes = b"unknown format"
        assert ImageProcessor.get_image_extension(unknown_bytes) == "bin"

    def test_audio_extension_detection(self):
        """Test audio extension detection for various formats."""
        parser = EfficientTensorBoardParser.__new__(
            EfficientTensorBoardParser
        )  # Create without __init__

        # Test WAV detection
        assert parser.get_audio_extension("audio/wav") == "wav"
        assert parser.get_audio_extension("audio/x-wav") == "wav"

        # Test MP3 detection
        assert parser.get_audio_extension("audio/mp3") == "mp3"
        assert parser.get_audio_extension("audio/mp3-stream") == "mp3"

        # Test OGG detection
        assert parser.get_audio_extension("audio/ogg") == "ogg"

        # Test unknown format
        assert parser.get_audio_extension("audio/unknown") == "audio"

    def test_tag_name_sanitization(self):
        """Test tag name sanitization for filesystem safety."""
        parser = EfficientTensorBoardParser.__new__(EfficientTensorBoardParser)
        # Initialize attributes that __init__ would normally set
        parser._tags_cache = None
        parser.show_progress = False
        parser.event_file_path = "dummy_file"

        # Mock some content with slash-containing tags
        with patch.object(
            parser, "list_scalars", return_value=["metrics/precision", "loss"]
        ):
            with patch.object(parser, "list_images", return_value=["images/sample"]):
                with patch.object(parser, "list_histograms", return_value=[]):
                    with patch.object(parser, "list_audio", return_value=[]):
                        with patch.object(parser, "list_text", return_value=[]):
                            with patch.object(
                                parser, "iterate_image_data", return_value=iter([])
                            ):
                                paths = parser.get_virtual_paths()

                                # Check that slashes in tags are replaced with underscores
                                assert "scalars/metrics_precision.txt" in paths
                                assert "scalars/loss.txt" in paths
                                assert "images/images_sample/" in paths
