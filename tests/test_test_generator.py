"""Test the test generator functionality."""

from pathlib import Path
from tboardfs.test_generator import (
    generate_test_tensorboard_log,
    generate_minimal_test_log,
)
from tboardfs.parser import TensorBoardParser


class TestTestGenerator:
    """Test the test data generator."""

    def test_generate_test_tensorboard_log(self, temp_dir):
        """Test generating a full test log."""
        log_dir = temp_dir / "test_log"
        event_file = generate_test_tensorboard_log(str(log_dir), num_iterations=5)

        assert event_file is not None
        assert Path(event_file).exists()
        assert "tfevents" in event_file

        # Parse and verify content
        parser = TensorBoardParser(event_file)

        # Check scalars
        scalars = parser.list_scalars()
        assert "loss" in scalars
        assert "accuracy" in scalars
        assert "learning_rate" in scalars
        assert "metrics/precision" in scalars
        assert "metrics/recall" in scalars

        # Check each scalar has correct number of data points
        loss_data = parser.get_scalar_data("loss")
        assert len(loss_data) == 5

        # Check images
        images = parser.list_images()
        assert "sample_images/rgb" in images
        assert "sample_images/grayscale" in images

        rgb_data = parser.get_image_data("sample_images/rgb")
        assert len(rgb_data) == 5
        assert rgb_data[0].width == 100
        assert rgb_data[0].height == 100

        # Check histograms
        histograms = parser.list_histograms()
        assert "distributions/normal" in histograms
        assert "distributions/uniform" in histograms
        assert "model/weights" in histograms

        # Check audio
        audio = parser.list_audio()
        assert "sounds/sine_wave" in audio

        audio_data = parser.get_audio_data("sounds/sine_wave")
        assert len(audio_data) == 5
        assert audio_data[0].sample_rate == 8000

        # Check text
        text = parser.list_text()
        assert "logs/info" in text
        assert "model/architecture" in text

        text_data = parser.get_text_data("logs/info")
        assert len(text_data) == 5
        assert "Iteration 0" in text_data[0].text

    def test_generate_test_tensorboard_log_custom_iterations(self, temp_dir):
        """Test generating log with custom number of iterations."""
        log_dir = temp_dir / "test_log_20"
        event_file = generate_test_tensorboard_log(str(log_dir), num_iterations=20)

        parser = TensorBoardParser(event_file)
        loss_data = parser.get_scalar_data("loss")
        assert len(loss_data) == 20

    def test_generate_minimal_test_log(self, temp_dir):
        """Test generating a minimal test log."""
        log_dir = temp_dir / "minimal_log"
        event_file = generate_minimal_test_log(str(log_dir))

        assert event_file is not None
        assert Path(event_file).exists()

        # Parse and verify minimal content
        parser = TensorBoardParser(event_file)

        scalars = parser.list_scalars()
        assert "test/metric" in scalars

        scalar_data = parser.get_scalar_data("test/metric")
        assert len(scalar_data) == 3  # steps 0, 5, 10
        assert [d.step for d in scalar_data] == [0, 5, 10]
        assert [d.value for d in scalar_data] == [0.0, 5.0, 10.0]

        images = parser.list_images()
        assert "test/image" in images

        image_data = parser.get_image_data("test/image")
        assert len(image_data) == 3
        assert image_data[0].width == 2
        assert image_data[0].height == 2

    def test_generated_data_consistency(self, temp_dir):
        """Test that generated data is consistent across all types."""
        log_dir = temp_dir / "consistency_test"
        event_file = generate_test_tensorboard_log(str(log_dir), num_iterations=7)

        parser = TensorBoardParser(event_file)

        # All data types should have same number of iterations
        assert len(parser.get_scalar_data("loss")) == 7
        assert len(parser.get_image_data("sample_images/rgb")) == 7
        assert len(parser.get_histogram_data("distributions/normal")) == 7
        assert len(parser.get_audio_data("sounds/sine_wave")) == 7
        assert len(parser.get_text_data("logs/info")) == 7

        # All should have matching step numbers
        scalar_steps = [d.step for d in parser.get_scalar_data("loss")]
        image_steps = [d.step for d in parser.get_image_data("sample_images/rgb")]
        histogram_steps = [
            d.step for d in parser.get_histogram_data("distributions/normal")
        ]
        audio_steps = [d.step for d in parser.get_audio_data("sounds/sine_wave")]
        text_steps = [d.step for d in parser.get_text_data("logs/info")]

        expected_steps = list(range(7))
        assert scalar_steps == expected_steps
        assert image_steps == expected_steps
        assert histogram_steps == expected_steps
        assert audio_steps == expected_steps
        assert text_steps == expected_steps
