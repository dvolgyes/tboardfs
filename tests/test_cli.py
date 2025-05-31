"""Test CLI functionality."""

import pytest
from click.testing import CliRunner
from pathlib import Path
from tboardfs.cli import main
from loguru import logger
import sys


class TestCLI:
    """Test CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        # Configure logger to capture output in tests
        logger.remove()
        logger.add(sys.stderr, level="INFO", format="{message}")
        return CliRunner()

    def test_main_help(self, runner):
        """Test main help command."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "TensorBoard filesystem interface CLI" in result.output
        assert "list" in result.output
        assert "export" in result.output
        assert "extract" in result.output

    def test_list_help(self, runner):
        """Test list help command."""
        result = runner.invoke(main, ["list", "--help"])
        assert result.exit_code == 0
        assert "List contents of TensorBoard log file(s)" in result.output
        assert "--recursive" in result.output
        assert "--digits" in result.output

    def test_list_single_file(self, runner, test_event_file):
        """Test listing contents of a single event file."""
        result = runner.invoke(main, ["list", test_event_file])
        assert result.exit_code == 0

        # Check output contains expected sections
        assert "Scalars:" in result.output
        assert "loss" in result.output
        assert "accuracy" in result.output

        assert "Images:" in result.output
        assert "sample_images/rgb" in result.output

        assert "Histograms:" in result.output
        assert "distributions/normal" in result.output

        assert "Audio:" in result.output
        assert "sounds/sine_wave" in result.output

        assert "Text:" in result.output
        assert "logs/info" in result.output

        assert "Virtual filesystem paths:" in result.output
        assert "scalars/loss.txt" in result.output

    def test_list_directory(self, runner, temp_dir, minimal_event_file):
        """Test listing event files in a directory."""
        log_dir = Path(minimal_event_file).parent
        result = runner.invoke(main, ["list", str(log_dir)])
        assert result.exit_code == 0
        assert "Found 1 TensorBoard event file(s)" in result.output
        assert "events.out.tfevents" in result.output

    def test_list_directory_recursive(
        self, runner, temp_dir, test_event_file, minimal_event_file
    ):
        """Test recursive listing."""
        result = runner.invoke(main, ["list", "-r", str(temp_dir)])
        assert result.exit_code == 0
        # Should show contents of both event files
        assert result.output.count("Contents of") >= 2

    def test_list_invalid_path(self, runner):
        """Test listing with invalid path."""
        result = runner.invoke(main, ["list", "/nonexistent/path"])
        assert result.exit_code == 1

    def test_export_scalar(self, runner, test_event_file, temp_dir):
        """Test exporting scalar data."""
        output_file = temp_dir / "loss.txt"
        result = runner.invoke(
            main,
            ["export", test_event_file, "scalars/loss.txt", "-o", str(output_file)],
        )
        assert result.exit_code == 0
        assert "Exported scalar data to" in result.output

        # Check output file
        assert output_file.exists()
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 11

    def test_export_scalar_stdout(self, runner, test_event_file):
        """Test exporting scalar data to stdout."""
        result = runner.invoke(main, ["export", test_event_file, "scalars/loss.txt"])
        assert result.exit_code == 0

        # Output should contain tab-delimited data
        lines = result.output.strip().split("\n")
        assert len(lines) == 11
        assert "\t" in lines[0]

    def test_export_image(self, runner, test_event_file, temp_dir):
        """Test exporting image data."""
        output_file = temp_dir / "image.png"
        result = runner.invoke(
            main,
            [
                "export",
                test_event_file,
                "images/sample_images_rgb/000000.png",
                "-o",
                str(output_file),
            ],
        )
        assert result.exit_code == 0
        assert "Exported image to" in result.output

        # Check output file
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_export_histogram(self, runner, test_event_file, temp_dir):
        """Test exporting histogram data."""
        output_file = temp_dir / "histogram.txt"
        result = runner.invoke(
            main,
            [
                "export",
                test_event_file,
                "histograms/distributions_normal.txt",
                "-o",
                str(output_file),
            ],
        )
        assert result.exit_code == 0
        assert "Exported histogram data to" in result.output

        # Check output file
        assert output_file.exists()
        content = output_file.read_text()
        assert "Step:" in content
        assert "Buckets:" in content

    def test_export_audio(self, runner, test_event_file, temp_dir):
        """Test exporting audio data."""
        output_file = temp_dir / "audio.wav"
        result = runner.invoke(
            main,
            [
                "export",
                test_event_file,
                "audio/sounds_sine_wave/000000.wav",
                "-o",
                str(output_file),
            ],
        )
        assert result.exit_code == 0
        assert "Exported audio to" in result.output

        # Check output file
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_export_text(self, runner, test_event_file, temp_dir):
        """Test exporting text data."""
        output_file = temp_dir / "text.txt"
        result = runner.invoke(
            main,
            [
                "export",
                test_event_file,
                "text/logs_info/000000.txt",
                "-o",
                str(output_file),
            ],
        )
        assert result.exit_code == 0
        assert "Exported text to" in result.output

        # Check output file
        assert output_file.exists()
        content = output_file.read_text()
        assert "Iteration 0" in content

    def test_export_nonexistent_tag(self, runner, test_event_file):
        """Test exporting with non-existent tag."""
        result = runner.invoke(
            main, ["export", test_event_file, "scalars/nonexistent.txt"]
        )
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_export_invalid_path(self, runner, test_event_file):
        """Test exporting with invalid virtual path."""
        result = runner.invoke(main, ["export", test_event_file, "invalid"])
        assert result.exit_code == 1
        assert "Invalid virtual path format" in result.output

    def test_extract_help(self, runner):
        """Test extract help command."""
        result = runner.invoke(main, ["extract", "--help"])
        assert result.exit_code == 0
        assert "Extract all data from TensorBoard log" in result.output
        assert "--no-sort" in result.output
        assert "--digits" in result.output

    def test_extract_all(self, runner, test_event_file, temp_dir):
        """Test extracting all data."""
        output_dir = temp_dir / "extracted"
        result = runner.invoke(
            main, ["extract", test_event_file, "-o", str(output_dir)]
        )
        assert result.exit_code == 0

        # Check output message
        assert "Extracted TensorBoard data to:" in result.output
        assert "5 scalar tag(s)" in result.output
        assert "2 image tag(s)" in result.output
        assert "3 histogram tag(s)" in result.output
        assert "1 audio tag(s)" in result.output
        assert "2 text tag(s)" in result.output
        assert "Scalar files sorted by iteration number" in result.output

        # Check directory structure
        assert output_dir.exists()
        assert (output_dir / "scalars").exists()
        assert (output_dir / "images").exists()
        assert (output_dir / "histograms").exists()
        assert (output_dir / "audio").exists()
        assert (output_dir / "text").exists()

    def test_extract_no_sort(self, runner, test_event_file, temp_dir):
        """Test extracting with no-sort option."""
        output_dir = temp_dir / "extracted_nosort"
        result = runner.invoke(
            main, ["extract", test_event_file, "-o", str(output_dir), "--no-sort"]
        )
        assert result.exit_code == 0
        assert "Scalar files not sorted (--no-sort specified)" in result.output

    def test_extract_custom_digits(self, runner, test_event_file, temp_dir):
        """Test extracting with custom digit padding."""
        output_dir = temp_dir / "extracted_custom"
        result = runner.invoke(
            main, ["extract", test_event_file, "-o", str(output_dir), "--digits", "3"]
        )
        assert result.exit_code == 0

        # Check padding
        rgb_dir = output_dir / "images" / "sample_images_rgb"
        assert (rgb_dir / "000.png").exists()
        assert (rgb_dir / "010.png").exists()

    def test_extract_invalid_file(self, runner, temp_dir):
        """Test extracting from invalid file."""
        invalid_file = temp_dir / "not_tfevents.txt"
        invalid_file.write_text("invalid")

        result = runner.invoke(
            main, ["extract", str(invalid_file), "-o", str(temp_dir / "output")]
        )
        assert result.exit_code == 1
        assert "not a valid TensorBoard event file" in result.output
