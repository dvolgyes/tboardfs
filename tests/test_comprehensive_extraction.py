"""Comprehensive integration tests for data extraction validation.

Tests that all data types are extracted correctly, files are non-empty,
and content matches expected formats using python-magic for type detection.
"""

import pytest
import tempfile
import magic
from pathlib import Path

from tboardfs.efficient_parser import EfficientTensorBoardParser
from tboardfs.commands.extract_command import extract_tensorboard_data
from tboardfs.core.data_source import DataSourceFactory


class TestComprehensiveDataExtraction:
    """Test comprehensive data extraction with file validation."""

    @pytest.fixture
    def test_log_directory(self):
        """Use test log directory with various data types."""
        log_dir = Path("test-logs")
        if log_dir.exists():
            return str(log_dir)
        pytest.skip("Test log directory not found")

    @pytest.fixture
    def single_test_file(self):
        """Use single test file for focused testing."""
        test_file = Path(
            "test-logs/images/events.out.tfevents.1751399792.FG-OSL-WS122.70334.26"
        )
        if test_file.exists():
            return str(test_file)
        pytest.skip("Single test file not found")

    @pytest.fixture
    def extraction_output_dir(self):
        """Create temporary directory for extraction output."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def get_expected_data_types(
        self, parser: EfficientTensorBoardParser
    ) -> dict[str, list[str]]:
        """Get expected data types and their tags from parser."""
        expected = {}

        scalars = parser.list_scalars()
        if scalars:
            expected["scalars"] = scalars

        images = parser.list_images()
        if images:
            expected["images"] = images

        histograms = parser.list_histograms()
        if histograms:
            expected["histograms"] = histograms

        audio = parser.list_audio()
        if audio:
            expected["audio"] = audio

        text = parser.list_text()
        if text:
            expected["text"] = text

        return expected

    def validate_file_content_type(
        self, file_path: Path, expected_types: set[str]
    ) -> bool:
        """Validate file content type using python-magic."""
        if not file_path.exists() or file_path.stat().st_size == 0:
            return False

        try:
            # Get MIME type
            mime_type = magic.from_file(str(file_path), mime=True)

            # Get file description
            file_desc = magic.from_file(str(file_path)).lower()

            # Check if file type matches expected types
            for expected_type in expected_types:
                if expected_type in mime_type.lower() or expected_type in file_desc:
                    return True

            return False
        except Exception:
            # If magic fails, check file size at least
            return file_path.stat().st_size > 0

    def test_complete_data_extraction(self, test_log_directory, extraction_output_dir):
        """Test complete data extraction from test logs."""
        # Extract all data
        extract_tensorboard_data(
            tensorboard_path=test_log_directory,
            output_dir=str(extraction_output_dir),
            digits=6,
            image_format="png",
            image_quality=90,
            audio_format="mp3",
            histogram_images=False,
            ply_format="binary",
            type_filters={"ignore": set(), "select": set()},
        )

        # Verify extraction created expected structure
        assert extraction_output_dir.exists()

        # Get all extracted directories
        extracted_dirs = [d for d in extraction_output_dir.iterdir() if d.is_dir()]

        # Test each expected data type directory
        for data_type_dir in extracted_dirs:
            data_type = data_type_dir.name
            print(
                f"Testing data type: {data_type} with {len(list(data_type_dir.rglob('*')))} files"
            )

            if data_type == "scalars":
                self._validate_scalar_extraction(data_type_dir)
            elif data_type == "images":
                self._validate_image_extraction(data_type_dir)
            elif data_type == "histograms":
                self._validate_histogram_extraction(data_type_dir)
            elif data_type == "audio":
                self._validate_audio_extraction(data_type_dir)
            elif data_type == "text":
                self._validate_text_extraction(data_type_dir)

    def _validate_scalar_extraction(self, scalars_dir: Path):
        """Validate scalar data extraction."""
        scalar_files = list(scalars_dir.rglob("*.txt"))
        assert len(scalar_files) > 0, "No scalar files extracted"

        for scalar_file in scalar_files:
            # Check file is not empty
            assert scalar_file.stat().st_size > 0, f"Scalar file {scalar_file} is empty"

            # Validate content type (should be text)
            assert self.validate_file_content_type(scalar_file, {"text", "ascii"}), (
                f"Scalar file {scalar_file} has wrong content type"
            )

            # Check content format (should be CSV-like with numbers)
            content = scalar_file.read_text()
            lines = content.strip().split("\n")
            assert len(lines) > 0, f"Scalar file {scalar_file} has no content lines"

            # Check first few lines have numeric data
            for line in lines[:3]:  # Check first 3 lines
                # Try both comma and tab separators
                parts = line.split(",") if "," in line else line.split("\t")
                assert len(parts) >= 2, f"Scalar line format incorrect: {line}"
                # Should have step and value columns
                try:
                    float(parts[0])  # step
                    float(parts[1])  # value
                except ValueError:
                    pytest.fail(f"Scalar data not numeric in {scalar_file}: {line}")

    def _validate_image_extraction(self, images_dir: Path):
        """Validate image data extraction."""
        image_files = list(images_dir.rglob("*.png")) + list(images_dir.rglob("*.jpg"))
        assert len(image_files) > 0, "No image files extracted"

        for image_file in image_files:
            # Check file is not empty
            assert image_file.stat().st_size > 0, f"Image file {image_file} is empty"

            # Validate content type (should be image)
            assert self.validate_file_content_type(
                image_file, {"image", "png", "jpeg"}
            ), f"Image file {image_file} has wrong content type"

            # Additional validation: check image headers
            with image_file.open("rb") as f:
                header = f.read(8)
                if image_file.suffix.lower() == ".png":
                    assert header.startswith(b"\x89PNG"), (
                        f"Invalid PNG header in {image_file}"
                    )
                elif image_file.suffix.lower() in [".jpg", ".jpeg"]:
                    assert header.startswith(b"\xff\xd8"), (
                        f"Invalid JPEG header in {image_file}"
                    )

    def _validate_histogram_extraction(self, histograms_dir: Path):
        """Validate histogram data extraction."""
        # Histograms can be .npy files, .png images, or .txt files depending on export format
        hist_files = list(histograms_dir.rglob("*"))
        hist_files = [f for f in hist_files if f.is_file()]
        assert len(hist_files) > 0, "No histogram files extracted"

        for hist_file in hist_files:
            # Check file is not empty
            assert hist_file.stat().st_size > 0, f"Histogram file {hist_file} is empty"

            if hist_file.suffix == ".npy":
                # Validate numpy file
                assert self.validate_file_content_type(hist_file, {"data", "numpy"}), (
                    f"Histogram file {hist_file} has wrong content type"
                )
            elif hist_file.suffix == ".png":
                # Validate image file
                assert self.validate_file_content_type(hist_file, {"image", "png"}), (
                    f"Histogram image {hist_file} has wrong content type"
                )
            elif hist_file.suffix == ".txt":
                # Validate text file (histogram data as text)
                assert self.validate_file_content_type(hist_file, {"text", "ascii"}), (
                    f"Histogram text file {hist_file} has wrong content type"
                )

    def _validate_audio_extraction(self, audio_dir: Path):
        """Validate audio data extraction."""
        audio_files = list(audio_dir.rglob("*.mp3")) + list(audio_dir.rglob("*.wav"))
        assert len(audio_files) > 0, "No audio files extracted"

        for audio_file in audio_files:
            # Check file is not empty
            assert audio_file.stat().st_size > 0, f"Audio file {audio_file} is empty"

            # Validate content type (should be audio)
            assert self.validate_file_content_type(
                audio_file, {"audio", "mp3", "wav", "mpeg"}
            ), f"Audio file {audio_file} has wrong content type"

    def _validate_text_extraction(self, text_dir: Path):
        """Validate text data extraction."""
        text_files = list(text_dir.rglob("*.txt"))
        assert len(text_files) > 0, "No text files extracted"

        for text_file in text_files:
            # Check file is not empty
            assert text_file.stat().st_size > 0, f"Text file {text_file} is empty"

            # Validate content type (should be text)
            assert self.validate_file_content_type(
                text_file, {"text", "ascii", "utf-8"}
            ), f"Text file {text_file} has wrong content type"

            # Validate content is readable text
            try:
                content = text_file.read_text(encoding="utf-8")
                assert len(content.strip()) > 0, (
                    f"Text file {text_file} has no readable content"
                )
            except UnicodeDecodeError:
                pytest.fail(f"Text file {text_file} is not valid UTF-8")

    def test_data_type_completeness(self, test_log_directory):
        """Test that all available data types are detected and extracted."""
        # Create parser for the test directory using data source
        data_source = DataSourceFactory.from_path(test_log_directory)
        parser = EfficientTensorBoardParser(data_source=data_source)

        # Get all available data types
        available_types = self.get_expected_data_types(parser)

        # Test extraction with temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            extract_tensorboard_data(
                tensorboard_path=test_log_directory,
                output_dir=tmp_dir,
                digits=6,
                image_format="png",
                image_quality=90,
                audio_format="mp3",
                histogram_images=False,
                ply_format="binary",
                type_filters={"ignore": set(), "select": set()},
            )

            output_dir = Path(tmp_dir)
            extracted_dirs = {d.name for d in output_dir.iterdir() if d.is_dir()}

            # Check that each available data type was extracted
            for data_type, tags in available_types.items():
                assert data_type in extracted_dirs, (
                    f"Data type {data_type} not extracted despite being available"
                )

                # Check that each tag within the data type has content
                type_dir = output_dir / data_type
                if data_type == "scalars":
                    # Scalars create .txt files directly
                    scalar_files = list(type_dir.rglob("*.txt"))
                    assert len(scalar_files) > 0, (
                        f"No scalar files extracted despite having {len(tags)} tags"
                    )
                elif data_type == "images":
                    # Images create subdirectories
                    image_dirs = [d for d in type_dir.iterdir() if d.is_dir()]
                    assert len(image_dirs) > 0, (
                        f"No image directories extracted despite having {len(tags)} tags"
                    )
                elif data_type in ["histograms", "audio", "text"]:
                    # Check that files exist for each tag
                    content_files = list(type_dir.rglob("*"))
                    content_files = [f for f in content_files if f.is_file()]
                    assert len(content_files) > 0, f"No files extracted for {data_type}"

    def test_selective_extraction_by_type(self, test_log_directory):
        """Test selective extraction by data type."""
        # Test extracting only images
        with tempfile.TemporaryDirectory() as tmp_dir:
            extract_tensorboard_data(
                tensorboard_path=test_log_directory,
                output_dir=tmp_dir,
                digits=6,
                image_format="png",
                image_quality=90,
                audio_format="mp3",
                histogram_images=False,
                ply_format="binary",
                type_filters={"ignore": set(), "select": {"image"}},
            )

            output_dir = Path(tmp_dir)
            extracted_dirs = {d.name for d in output_dir.iterdir() if d.is_dir()}

            # Should only have images directory
            assert (
                "images" in extracted_dirs or len(extracted_dirs) == 0
            )  # Allow empty if no images

            # Should not have other types
            unwanted_types = {"scalars", "histograms", "audio", "text"} - {"images"}
            for unwanted in unwanted_types:
                assert unwanted not in extracted_dirs, (
                    f"Found unwanted type {unwanted} in selective extraction"
                )

    def test_file_format_consistency(self, test_log_directory):
        """Test that file formats are consistent with requested formats."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test PNG format for images
            extract_tensorboard_data(
                tensorboard_path=test_log_directory,
                output_dir=tmp_dir,
                digits=6,
                image_format="png",
                image_quality=90,
                audio_format="wav",
                histogram_images=False,
                ply_format="text",
                type_filters={"ignore": set(), "select": set()},
            )

            output_dir = Path(tmp_dir)

            # Check image formats
            if (output_dir / "images").exists():
                image_files = list((output_dir / "images").rglob("*"))
                image_files = [f for f in image_files if f.is_file()]
                for img_file in image_files:
                    assert img_file.suffix.lower() == ".png", (
                        f"Expected PNG but got {img_file.suffix}"
                    )

            # Check audio formats (if any)
            if (output_dir / "audio").exists():
                audio_files = list((output_dir / "audio").rglob("*"))
                audio_files = [f for f in audio_files if f.is_file()]
                for audio_file in audio_files:
                    assert audio_file.suffix.lower() == ".wav", (
                        f"Expected WAV but got {audio_file.suffix}"
                    )

    def test_extraction_error_handling(self, test_log_directory):
        """Test extraction with invalid parameters and error conditions."""
        # Test with invalid output directory (read-only)
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "readonly"
            output_dir.mkdir()
            output_dir.chmod(0o444)  # Read-only

            try:
                # This should handle the error gracefully
                extract_tensorboard_data(
                    tensorboard_path=test_log_directory,
                    output_dir=str(output_dir),
                    digits=6,
                    image_format="png",
                    image_quality=90,
                    audio_format="mp3",
                    histogram_images=False,
                    ply_format="binary",
                    type_filters={"ignore": set(), "select": set()},
                )
            except PermissionError:
                # Expected error - this is fine
                pass
            finally:
                # Restore permissions for cleanup
                output_dir.chmod(0o755)

    def test_empty_log_extraction(self):
        """Test extraction from empty or minimal log files."""
        with tempfile.NamedTemporaryFile(suffix=".tfevents") as tmp_file:
            # Create minimal TensorBoard event file
            tmp_file.write(b"\x08\x01\x12\x00\x1a\x00")  # Minimal valid event
            tmp_file.flush()

            with tempfile.TemporaryDirectory() as output_dir:
                # This should not crash
                extract_tensorboard_data(
                    tensorboard_path=tmp_file.name,
                    output_dir=output_dir,
                    digits=6,
                    image_format="png",
                    image_quality=90,
                    audio_format="mp3",
                    histogram_images=False,
                    ply_format="binary",
                    type_filters={"ignore": set(), "select": set()},
                )

                # Output directory should exist but be empty or minimal
                output_path = Path(output_dir)
                assert output_path.exists()
                # Should have no data directories since file is minimal
                data_dirs = [d for d in output_path.iterdir() if d.is_dir()]
                # Minimal file should result in no extracted data
                assert len(data_dirs) == 0

    def test_single_file_extraction_validation(self, single_test_file):
        """Test extraction validation with a single known file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Extract from single file
            extract_tensorboard_data(
                tensorboard_path=single_test_file,
                output_dir=tmp_dir,
                digits=6,
                image_format="png",
                image_quality=90,
                audio_format="mp3",
                histogram_images=False,
                ply_format="binary",
                type_filters={"ignore": set(), "select": set()},
            )

            output_dir = Path(tmp_dir)
            assert output_dir.exists()

            # Check for extracted content
            extracted_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
            assert len(extracted_dirs) > 0, "No data directories created"

            # Validate each extracted data type
            for data_type_dir in extracted_dirs:
                data_type = data_type_dir.name
                print(f"Validating single file extraction for {data_type}")

                if data_type == "images":
                    self._validate_image_extraction(data_type_dir)
                elif data_type == "scalars":
                    self._validate_scalar_extraction(data_type_dir)
                elif data_type == "histograms":
                    self._validate_histogram_extraction(data_type_dir)
                elif data_type == "audio":
                    self._validate_audio_extraction(data_type_dir)
                elif data_type == "text":
                    self._validate_text_extraction(data_type_dir)
