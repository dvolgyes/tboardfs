"""Generate test TensorBoard event files with various data types."""

import numpy as np
import tensorflow as tf
from pathlib import Path


def generate_test_tensorboard_log(output_dir: str, num_iterations: int = 11):
    """Generate a TensorBoard log with dummy data for all supported types.

    Args:
        output_dir: Directory to write the event file
        num_iterations: Number of iterations to generate (default: 11)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create TensorBoard writer
    writer = tf.summary.create_file_writer(str(output_path))

    with writer.as_default():
        for step in range(num_iterations):
            # Scalars
            tf.summary.scalar("loss", 0.5 + 0.1 * np.sin(step), step=step)
            tf.summary.scalar("accuracy", 0.8 + 0.1 * np.cos(step), step=step)
            tf.summary.scalar("learning_rate", 0.001 * (0.9**step), step=step)
            tf.summary.scalar(
                "metrics/precision", 0.85 + 0.05 * np.sin(step * 2), step=step
            )
            tf.summary.scalar(
                "metrics/recall", 0.90 + 0.03 * np.cos(step * 2), step=step
            )

            # Images (create simple synthetic images)
            # RGB image
            rgb_image = np.zeros((100, 100, 3), dtype=np.uint8)
            rgb_image[:, :, 0] = int(255 * (step / num_iterations))  # Red gradient
            rgb_image[:, :, 1] = int(128 + 127 * np.sin(step))  # Green wave
            rgb_image[:, :, 2] = 128  # Constant blue
            tf.summary.image("sample_images/rgb", rgb_image[np.newaxis, ...], step=step)

            # Grayscale image
            gray_image = np.zeros((64, 64, 1), dtype=np.uint8)
            gray_image[:, :, 0] = np.uint8(255 * np.random.rand(64, 64))
            tf.summary.image(
                "sample_images/grayscale", gray_image[np.newaxis, ...], step=step
            )

            # Histograms
            normal_dist = np.random.normal(0, 1, 1000) + step * 0.1
            tf.summary.histogram("distributions/normal", normal_dist, step=step)

            uniform_dist = np.random.uniform(-1, 1, 1000)
            tf.summary.histogram("distributions/uniform", uniform_dist, step=step)

            # Simulated weights histogram
            weights = np.random.randn(100, 50) * 0.1
            tf.summary.histogram("model/weights", weights.flatten(), step=step)

            # Audio (create simple sine wave)
            sample_rate = 8000
            duration = 1.0  # 1 second
            frequency = 440 + step * 20  # A4 + step offset
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
            audio_data = audio_data.astype(np.float32)
            tf.summary.audio(
                "sounds/sine_wave",
                audio_data[np.newaxis, np.newaxis, :],
                sample_rate=sample_rate,
                step=step,
            )

            # Text summaries
            tf.summary.text(
                "logs/info", f"Iteration {step}: Training in progress", step=step
            )
            tf.summary.text(
                "model/architecture",
                "Layer 1: Dense(128)\nLayer 2: ReLU\nLayer 3: Dense(10)",
                step=step,
            )

            # Flush to ensure data is written
            writer.flush()

    writer.close()

    print(f"Generated test TensorBoard log in: {output_path}")

    # Return the path to the generated event file
    event_files = list(output_path.glob("events.out.tfevents.*"))
    if event_files:
        return str(event_files[0])
    return None


def generate_minimal_test_log(output_dir: str):
    """Generate a minimal TensorBoard log for quick testing.

    Args:
        output_dir: Directory to write the event file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    writer = tf.summary.create_file_writer(str(output_path))

    with writer.as_default():
        # Just a few data points
        for step in [0, 5, 10]:
            tf.summary.scalar("test/metric", float(step), step=step)

            # Simple 2x2 image
            img = np.array(
                [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]],
                dtype=np.uint8,
            )
            tf.summary.image("test/image", img[np.newaxis, ...], step=step)

            writer.flush()

    writer.close()

    event_files = list(output_path.glob("events.out.tfevents.*"))
    if event_files:
        return str(event_files[0])
    return None


if __name__ == "__main__":
    # Generate test data when run as script
    import sys

    output_dir = sys.argv[1] if len(sys.argv) > 1 else "./test_tensorboard_logs"
    generate_test_tensorboard_log(output_dir)
