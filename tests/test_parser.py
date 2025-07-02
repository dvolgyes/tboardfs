"""Test TensorBoard parser functionality with v2 format."""

from tboardfs.efficient_parser import EfficientTensorBoardParser


class TestEfficientTensorBoardParser:
    """Test TensorBoard parser functionality."""

    def test_parser_initialization(self, test_event_file):
        """Test parser can be initialized with event file."""
        parser = EfficientTensorBoardParser(test_event_file)
        assert parser.event_file_path == test_event_file
        assert parser.event_file_path == test_event_file

    def test_list_tensors(self, test_event_file):
        """Test listing tensor tags (v2 format has proper categorization)."""
        parser = EfficientTensorBoardParser(test_event_file)
        tensors = parser.list_tensors()

        assert isinstance(tensors, list)
        # v2 format properly categorizes data, tensors list may be empty
        # since scalars are in scalars, not tensors

    def test_list_text(self, test_event_file):
        """Test listing text tags."""
        parser = EfficientTensorBoardParser(test_event_file)
        text_tags = parser.list_text()

        assert isinstance(text_tags, list)
        assert len(text_tags) > 0
        # Check our hardcoded dtype value works
        assert any("logs" in tag for tag in text_tags)

    def test_get_text_data(self, test_event_file):
        """Test getting text data."""
        parser = EfficientTensorBoardParser(test_event_file)
        text_tags = parser.list_text()

        if text_tags:
            tag = text_tags[0]
            data = list(parser.iterate_text_data(tag))
            assert isinstance(data, list)
            assert len(data) > 0
            assert hasattr(data[0], "text")
            assert hasattr(data[0], "step")

    def test_list_all_content(self, test_event_file):
        """Test listing all content types."""
        parser = EfficientTensorBoardParser(test_event_file)
        content = parser.list_all_content()

        assert isinstance(content, dict)
        assert "tensors" in content
        assert "text" in content
        assert "scalars" in content
        # v2 format: properly categorizes data
        assert len(content["scalars"]) > 0  # v2 properly categorizes scalars
        assert len(content["text"]) > 0

    def test_get_virtual_paths(self, test_event_file):
        """Test getting virtual filesystem paths."""
        parser = EfficientTensorBoardParser(test_event_file)
        paths = parser.get_virtual_paths()

        assert isinstance(paths, list)
        assert len(paths) > 0
        # Check directory structure
        assert "text/" in paths
        assert "scalars/" in paths  # Empty but present
        assert "images/" in paths
        assert "histograms/" in paths
        assert "audio/" in paths

    def test_get_image_extension(self):
        """Test image extension detection."""
        parser = EfficientTensorBoardParser.__new__(EfficientTensorBoardParser)

        # PNG magic bytes with proper header
        png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        assert parser.get_image_extension(png_bytes) == "png"

        # JPEG magic bytes with proper header
        jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C"
        assert parser.get_image_extension(jpeg_bytes) == "jpg"

        # Unknown
        assert parser.get_image_extension(b"unknown") == "bin"

    def test_get_audio_extension(self):
        """Test audio extension detection."""
        parser = EfficientTensorBoardParser.__new__(EfficientTensorBoardParser)

        assert parser.get_audio_extension("audio/wav") == "wav"
        assert parser.get_audio_extension("audio/mp3") == "mp3"
        assert parser.get_audio_extension("audio/ogg") == "ogg"
        assert parser.get_audio_extension("unknown") == "audio"

    def test_minimal_event_file(self, minimal_event_file):
        """Test parsing minimal event file."""
        parser = EfficientTensorBoardParser(minimal_event_file)
        content = parser.list_all_content()

        assert isinstance(content, dict)
        assert "tensors" in content
        # Minimal file should have some content
        assert sum(len(v) for v in content.values()) > 0

    def test_export_text(self, test_event_file):
        """Test exporting text data."""
        parser = EfficientTensorBoardParser(test_event_file)
        text_tags = parser.list_text()

        if text_tags:
            tag = text_tags[0]
            data = list(parser.iterate_text_data(tag))
            if data:
                # Find the text data for the specific step
                text_data = None
                for text_item in parser.iterate_text_data(tag):
                    if text_item.step == data[0].step:
                        text_data = text_item.text
                        break
                exported = text_data
                assert exported is not None
                assert isinstance(exported, str)

    def test_progress_bar_disabled_by_default(self, test_event_file):
        """Test that progress bar is disabled by default."""
        parser = EfficientTensorBoardParser(test_event_file)
        assert not parser.show_progress

    def test_progress_bar_enabled(self, test_event_file):
        """Test parser with progress bar enabled."""
        parser = EfficientTensorBoardParser(test_event_file, show_progress=True)
        assert parser.show_progress
        # Should still work
        paths = parser.get_virtual_paths()
        assert len(paths) > 0
