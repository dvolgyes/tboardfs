"""Test CLI functionality with v2 format."""

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
        assert "--no-recursive" in result.output
        assert "--digits" in result.output

    def test_list_single_file(self, runner, test_event_file):
        """Test listing contents of a single event file."""
        result = runner.invoke(main, ["list", test_event_file])
        assert result.exit_code == 0

        # Check output contains expected sections
        # TensorFlow v2 stores everything as tensors
        assert "Tensors:" in result.output or "Scalars:" in result.output
        assert "loss" in result.output
        assert "accuracy" in result.output

        # Check for virtual paths
        assert "Virtual filesystem paths:" in result.output

        # Check some content exists in tensors or other categories
        assert (
            "distributions/normal" in result.output
            or "distributions_normal" in result.output
        )
        assert (
            "sounds/sine_wave" in result.output or "sounds_sine_wave" in result.output
        )
        assert "logs/info" in result.output or "logs_info" in result.output

    def test_list_directory(self, runner, temp_dir, minimal_event_file):
        """Test listing event files in a directory."""
        log_dir = Path(minimal_event_file).parent
        result = runner.invoke(main, ["list", str(log_dir)])
        assert result.exit_code == 0
        # With the new default recursive behavior, it processes the single file and shows content
        assert (
            "Contents of" in result.output or "Aggregated contents of" in result.output
        )

    def test_list_directory_recursive(
        self, runner, temp_dir, test_event_file, minimal_event_file
    ):
        """Test recursive listing."""
        # Use the tests/example-data directory which has multiple event files
        # Recursive is now default for directories
        result = runner.invoke(main, ["list", "tests/example-data"])
        assert result.exit_code == 0
        # With aggregated view, should show "Aggregated contents of" instead of multiple "Contents of"
        assert (
            "Aggregated contents of" in result.output
            or result.output.count("Contents of") >= 2
        )

    def test_list_directory_no_recursive(self, runner):
        """Test non-recursive directory listing."""
        result = runner.invoke(
            main, ["list", "--no-recursive", "tests/example-data/full_log"]
        )
        assert result.exit_code == 0
        # Should show list of files, not aggregated content
        assert "Found" in result.output and "TensorBoard event file(s)" in result.output

    def test_list_invalid_path(self, runner):
        """Test listing with invalid path."""
        result = runner.invoke(main, ["list", "/nonexistent/path"])
        assert result.exit_code == 2  # Click returns 2 for invalid arguments

    def test_export_text(self, runner, test_event_file, temp_dir):
        """Test exporting text data (v2 format stores text as tensors)."""
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

    def test_export_text_stdout(self, runner, test_event_file):
        """Test exporting text data to stdout."""
        result = runner.invoke(
            main, ["export", test_event_file, "text/logs_info/000000.txt"]
        )
        assert result.exit_code == 0
        # Output should contain the text content
        assert "Iteration 0" in result.output

    def test_export_nonexistent_path(self, runner, test_event_file):
        """Test exporting with non-existent virtual path."""
        result = runner.invoke(
            main, ["export", test_event_file, "text/nonexistent/000000.txt"]
        )
        assert result.exit_code == 1
        assert "not found" in result.output or "error" in result.output.lower()

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

    def test_extract_text_only(self, runner, test_event_file, temp_dir):
        """Test extracting data (v2 format mostly has text)."""
        output_dir = temp_dir / "extracted"
        result = runner.invoke(
            main, ["extract", test_event_file, "-o", str(output_dir)]
        )
        assert result.exit_code == 0

        # Check output message
        assert "Extracted TensorBoard data to:" in result.output

        # Check directory structure
        assert output_dir.exists()
        assert (output_dir / "text").exists()

        # Check that text files were created
        text_files = list((output_dir / "text").rglob("*.txt"))
        assert len(text_files) > 0

    def test_extract_no_sort(self, runner, test_event_file, temp_dir):
        """Test extracting with no sorting option."""
        output_dir = temp_dir / "extracted_nosort"
        result = runner.invoke(
            main, ["extract", test_event_file, "-o", str(output_dir), "--no-sort"]
        )
        assert result.exit_code == 0
        assert output_dir.exists()

    def test_extract_custom_digits(self, runner, test_event_file, temp_dir):
        """Test extracting with custom step digits."""
        output_dir = temp_dir / "extracted_digits"
        result = runner.invoke(
            main, ["extract", test_event_file, "-o", str(output_dir), "--digits", "4"]
        )
        assert result.exit_code == 0

        # Check that files use 4-digit padding
        text_files = list((output_dir / "text").rglob("*.txt"))
        if text_files:
            # File names should have 4-digit padding
            assert any("0000" in f.name for f in text_files)

    def test_extract_invalid_file(self, runner):
        """Test extracting with invalid file."""
        result = runner.invoke(main, ["extract", "/nonexistent/file"])
        assert result.exit_code != 0
