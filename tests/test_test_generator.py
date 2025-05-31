"""Test parsing of pre-generated test data."""

from pathlib import Path
from tboardfs.parser import TensorBoardParser


class TestPreGeneratedData:
    """Test parsing of pre-generated test event files."""

    def test_parse_full_log(self):
        """Test parsing a full test log."""
        event_file = "tests/example-data/full_log/events.out.tfevents.1748727850.FG-OSL-WS122.7152.0.v2"

        assert Path(event_file).exists()

        # Parse and verify content
        parser = TensorBoardParser(event_file)

        # Check that we can parse without tensorflow
        content = parser.list_all_content()
        assert "tensors" in content
        assert len(content["tensors"]) > 0

        # Check text detection works with hardcoded dtype
        text_tags = parser.list_text()
        assert len(text_tags) > 0

    def test_parse_minimal_log(self):
        """Test parsing a minimal test log."""
        event_file = "tests/example-data/minimal_log/events.out.tfevents.1748727851.FG-OSL-WS122.7152.1.v2"

        assert Path(event_file).exists()

        # Parse and verify minimal content
        parser = TensorBoardParser(event_file)

        content = parser.list_all_content()
        assert "tensors" in content

    def test_parse_different_iterations(self):
        """Test parsing logs with different iteration counts."""
        for n in [5, 7, 20]:
            event_file = f"tests/example-data/log_{n}_iterations/events.out.tfevents.1748727851.FG-OSL-WS122.7152.{n // 5 + 1}.v2"

            if not Path(event_file).exists():
                # Try alternate naming
                event_dir = Path(f"tests/example-data/log_{n}_iterations")
                event_files = list(event_dir.glob("events.out.tfevents.*"))
                assert len(event_files) == 1
                event_file = str(event_files[0])

            parser = TensorBoardParser(event_file)
            content = parser.list_all_content()
            assert len(content["tensors"]) > 0
