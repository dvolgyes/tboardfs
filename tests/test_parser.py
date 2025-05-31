"""Test TensorBoard parser functionality."""

from tboardfs.parser import TensorBoardParser


class TestTensorBoardParser:
    """Test TensorBoard parser functionality."""

    def test_parser_initialization(self, test_event_file):
        """Test parser can be initialized with event file."""
        parser = TensorBoardParser(test_event_file)
        assert parser.event_file_path == test_event_file
        assert parser.ea is not None

    def test_list_scalars(self, test_event_file):
        """Test listing scalar tags."""
        parser = TensorBoardParser(test_event_file)
        scalars = parser.list_scalars()

        assert isinstance(scalars, list)
        assert len(scalars) > 0
        assert "loss" in scalars
        assert "accuracy" in scalars
        assert "learning_rate" in scalars
        assert "metrics/precision" in scalars
        assert "metrics/recall" in scalars

    def test_list_images(self, test_event_file):
        """Test listing image tags."""
        parser = TensorBoardParser(test_event_file)
        images = parser.list_images()

        assert isinstance(images, list)
        assert len(images) > 0
        assert "sample_images/rgb" in images
        assert "sample_images/grayscale" in images

    def test_list_histograms(self, test_event_file):
        """Test listing histogram tags."""
        parser = TensorBoardParser(test_event_file)
        histograms = parser.list_histograms()

        assert isinstance(histograms, list)
        assert len(histograms) > 0
        assert "distributions/normal" in histograms
        assert "distributions/uniform" in histograms
        assert "model/weights" in histograms

    def test_list_audio(self, test_event_file):
        """Test listing audio tags."""
        parser = TensorBoardParser(test_event_file)
        audio = parser.list_audio()

        assert isinstance(audio, list)
        assert len(audio) > 0
        assert "sounds/sine_wave" in audio

    def test_list_text(self, test_event_file):
        """Test listing text tags."""
        parser = TensorBoardParser(test_event_file)
        text = parser.list_text()

        assert isinstance(text, list)
        assert len(text) > 0
        assert "logs/info" in text
        assert "model/architecture" in text

    def test_get_scalar_data(self, test_event_file):
        """Test getting scalar data."""
        parser = TensorBoardParser(test_event_file)
        scalar_data = parser.get_scalar_data("loss")

        assert len(scalar_data) == 11  # 11 iterations
        assert all(hasattr(d, "step") for d in scalar_data)
        assert all(hasattr(d, "value") for d in scalar_data)
        assert all(hasattr(d, "wall_time") for d in scalar_data)

        # Check steps are in order
        steps = [d.step for d in scalar_data]
        assert steps == list(range(11))

    def test_get_image_data(self, test_event_file):
        """Test getting image data."""
        parser = TensorBoardParser(test_event_file)
        image_data = parser.get_image_data("sample_images/rgb")

        assert len(image_data) == 11  # 11 iterations
        assert all(hasattr(d, "step") for d in image_data)
        assert all(hasattr(d, "encoded_image_string") for d in image_data)
        assert all(hasattr(d, "width") for d in image_data)
        assert all(hasattr(d, "height") for d in image_data)

        # Check image properties
        first_image = image_data[0]
        assert first_image.width == 100
        assert first_image.height == 100
        assert len(first_image.encoded_image_string) > 0

    def test_get_histogram_data(self, test_event_file):
        """Test getting histogram data."""
        parser = TensorBoardParser(test_event_file)
        histogram_data = parser.get_histogram_data("distributions/normal")

        assert len(histogram_data) == 11  # 11 iterations
        assert all(hasattr(d, "step") for d in histogram_data)
        assert all(hasattr(d, "min") for d in histogram_data)
        assert all(hasattr(d, "max") for d in histogram_data)
        assert all(hasattr(d, "num") for d in histogram_data)
        assert all(hasattr(d, "bucket_limit") for d in histogram_data)
        assert all(hasattr(d, "bucket") for d in histogram_data)

    def test_get_audio_data(self, test_event_file):
        """Test getting audio data."""
        parser = TensorBoardParser(test_event_file)
        audio_data = parser.get_audio_data("sounds/sine_wave")

        assert len(audio_data) == 11  # 11 iterations
        assert all(hasattr(d, "step") for d in audio_data)
        assert all(hasattr(d, "encoded_audio_string") for d in audio_data)
        assert all(hasattr(d, "content_type") for d in audio_data)
        assert all(hasattr(d, "sample_rate") for d in audio_data)
        assert all(hasattr(d, "length_frames") for d in audio_data)

    def test_get_text_data(self, test_event_file):
        """Test getting text data."""
        parser = TensorBoardParser(test_event_file)
        text_data = parser.get_text_data("logs/info")

        assert len(text_data) == 11  # 11 iterations
        assert all(hasattr(d, "step") for d in text_data)
        assert all(hasattr(d, "text") for d in text_data)
        assert all(hasattr(d, "wall_time") for d in text_data)

        # Check text content
        first_text = text_data[0]
        assert "Iteration 0" in first_text.text

    def test_export_scalar_to_text(self, test_event_file):
        """Test exporting scalar data to text format."""
        parser = TensorBoardParser(test_event_file)
        text_output = parser.export_scalar_to_text("loss")

        lines = text_output.strip().split("\n")
        assert len(lines) == 11  # 11 iterations

        # Check format
        for i, line in enumerate(lines):
            parts = line.split("\t")
            assert len(parts) == 2
            assert int(parts[0]) == i
            assert isinstance(float(parts[1]), float)

    def test_export_image(self, test_event_file):
        """Test exporting specific image."""
        parser = TensorBoardParser(test_event_file)

        # Export image at step 0
        image_bytes = parser.export_image("sample_images/rgb", 0)
        assert image_bytes is not None
        assert len(image_bytes) > 0

        # Check it's a valid PNG
        assert parser.get_image_extension(image_bytes) == "png"

        # Non-existent step should return None
        assert parser.export_image("sample_images/rgb", 999) is None

    def test_export_histogram_to_text(self, test_event_file):
        """Test exporting histogram data to text format."""
        parser = TensorBoardParser(test_event_file)
        text_output = parser.export_histogram_to_text("distributions/normal")

        assert "Step:" in text_output
        assert "Min:" in text_output
        assert "Max:" in text_output
        assert "Count:" in text_output
        assert "Buckets:" in text_output

    def test_export_audio(self, test_event_file):
        """Test exporting specific audio."""
        parser = TensorBoardParser(test_event_file)

        # Export audio at step 0
        audio_result = parser.export_audio("sounds/sine_wave", 0)
        assert audio_result is not None

        audio_bytes, content_type = audio_result
        assert len(audio_bytes) > 0
        assert "audio" in content_type

        # Non-existent step should return None
        assert parser.export_audio("sounds/sine_wave", 999) is None

    def test_export_text(self, test_event_file):
        """Test exporting specific text."""
        parser = TensorBoardParser(test_event_file)

        # Export text at step 0
        text = parser.export_text("logs/info", 0)
        assert text is not None
        assert "Iteration 0" in text

        # Non-existent step should return None
        assert parser.export_text("logs/info", 999) is None

    def test_get_image_extension(self, test_event_file):
        """Test image extension detection."""
        parser = TensorBoardParser(test_event_file)

        # PNG header
        assert parser.get_image_extension(b"\x89PNG\r\n\x1a\n") == "png"

        # JPEG header
        assert parser.get_image_extension(b"\xff\xd8\xff\xe0") == "jpg"

        # Unknown
        assert parser.get_image_extension(b"unknown") == "bin"

    def test_get_audio_extension(self, test_event_file):
        """Test audio extension detection."""
        parser = TensorBoardParser(test_event_file)

        assert parser.get_audio_extension("audio/wav") == "wav"
        assert parser.get_audio_extension("audio/mp3") == "mp3"
        assert parser.get_audio_extension("audio/ogg") == "ogg"
        assert parser.get_audio_extension("audio/unknown") == "audio"

    def test_list_all_content(self, test_event_file):
        """Test listing all content types."""
        parser = TensorBoardParser(test_event_file)
        content = parser.list_all_content()

        assert isinstance(content, dict)
        assert "scalars" in content
        assert "images" in content
        assert "histograms" in content
        assert "audio" in content
        assert "text" in content
        assert "tensors" in content

        # Check all lists are populated
        assert len(content["scalars"]) > 0
        assert len(content["images"]) > 0
        assert len(content["histograms"]) > 0
        assert len(content["audio"]) > 0
        assert len(content["text"]) > 0

    def test_get_virtual_paths(self, test_event_file):
        """Test virtual filesystem path generation."""
        parser = TensorBoardParser(test_event_file)
        paths = parser.get_virtual_paths()

        assert isinstance(paths, list)
        assert len(paths) > 0

        # Check directory entries
        assert "scalars/" in paths
        assert "images/" in paths
        assert "histograms/" in paths
        assert "audio/" in paths
        assert "text/" in paths

        # Check scalar files
        assert "scalars/loss.txt" in paths
        assert "scalars/accuracy.txt" in paths
        assert "scalars/metrics_precision.txt" in paths  # '/' replaced with '_'

        # Check image files with padding
        assert any("images/sample_images_rgb/000000.png" in p for p in paths)

        # Check histogram files
        assert "histograms/distributions_normal.txt" in paths

        # Check audio files
        assert any("audio/sounds_sine_wave/" in p for p in paths)

        # Check text files
        assert any("text/logs_info/" in p for p in paths)

    def test_get_virtual_paths_custom_digits(self, test_event_file):
        """Test virtual filesystem path generation with custom padding."""
        parser = TensorBoardParser(test_event_file)
        paths = parser.get_virtual_paths(digits=3)

        # Check image files with 3-digit padding
        assert any("images/sample_images_rgb/000.png" in p for p in paths)
        assert any("images/sample_images_rgb/010.png" in p for p in paths)

    def test_extract_all_to_directory(self, test_event_file, temp_dir):
        """Test extracting all data to directory structure."""
        parser = TensorBoardParser(test_event_file)
        output_dir = temp_dir / "extracted"

        parser.extract_all_to_directory(str(output_dir))

        # Check directory structure
        assert output_dir.exists()
        assert (output_dir / "scalars").exists()
        assert (output_dir / "images").exists()
        assert (output_dir / "histograms").exists()
        assert (output_dir / "audio").exists()
        assert (output_dir / "text").exists()

        # Check scalar files
        loss_file = output_dir / "scalars" / "loss.txt"
        assert loss_file.exists()
        lines = loss_file.read_text().strip().split("\n")
        assert len(lines) == 11

        # Check scalar file is sorted
        steps = [int(line.split("\t")[0]) for line in lines]
        assert steps == list(range(11))

        # Check images
        rgb_dir = output_dir / "images" / "sample_images_rgb"
        assert rgb_dir.exists()
        assert (rgb_dir / "000000.png").exists()
        assert (rgb_dir / "000010.png").exists()

        # Check histograms
        hist_file = output_dir / "histograms" / "distributions_normal.txt"
        assert hist_file.exists()
        assert hist_file.stat().st_size > 0

        # Check audio
        audio_dir = output_dir / "audio" / "sounds_sine_wave"
        assert audio_dir.exists()
        assert len(list(audio_dir.glob("*.wav"))) == 11

        # Check text
        text_dir = output_dir / "text" / "logs_info"
        assert text_dir.exists()
        assert (text_dir / "000000.txt").exists()
        text_content = (text_dir / "000000.txt").read_text()
        assert "Iteration 0" in text_content

    def test_extract_all_no_sort(self, test_event_file, temp_dir):
        """Test extracting with no sorting option."""
        parser = TensorBoardParser(test_event_file)
        output_dir = temp_dir / "extracted_nosort"

        parser.extract_all_to_directory(str(output_dir), sort_scalars=False)

        # Scalar files should still exist but may not be sorted
        loss_file = output_dir / "scalars" / "loss.txt"
        assert loss_file.exists()
        lines = loss_file.read_text().strip().split("\n")
        assert len(lines) == 11

    def test_tag_name_sanitization(self, test_event_file):
        """Test that tag names with '/' are properly sanitized."""
        parser = TensorBoardParser(test_event_file)
        paths = parser.get_virtual_paths()

        # Tags with '/' should be replaced with '_'
        assert "scalars/metrics_precision.txt" in paths
        assert "scalars/metrics/precision.txt" not in paths

        assert "images/sample_images_rgb/" in paths
        assert "images/sample_images/rgb/" not in paths
