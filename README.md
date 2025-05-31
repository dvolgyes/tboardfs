# tboardfs

A FUSE filesystem interface for TensorBoard logs that allows mounting TensorBoard event files as a virtual filesystem.

## Features

- Parse TensorBoard event files
- Support for all major TensorBoard data types:
  - Scalars: Exported as tab-delimited text files with all values collected
  - Images: Exported in original format (PNG/JPEG)
  - Histograms: Exported as human-readable text files
  - Audio: Exported in original format (WAV/MP3/OGG)
  - Text: Exported as UTF-8 text files
- Extract all data to organized directory structure
- Zero-padded iteration numbers for proper filesystem sorting
- Command-line interface with list, export, and extract commands

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd tboardfs

# Install with uv (recommended)
uv pip install -e .

# Or install with pip
pip install -e .

# Install test dependencies
uv pip install -e ".[test]"
```

## Usage

### List contents of TensorBoard logs

```bash
# List single file
tboardfs list path/to/events.out.tfevents.xxx

# List directory
tboardfs list path/to/log/directory/

# List recursively
tboardfs list -r path/to/log/directory/

# Custom padding for iteration numbers
tboardfs list --digits 8 path/to/events.out.tfevents.xxx
```

### Export specific data

```bash
# Export scalar data to stdout
tboardfs export path/to/events.out.tfevents.xxx scalars/loss.txt

# Export scalar data to file
tboardfs export path/to/events.out.tfevents.xxx scalars/loss.txt -o loss_values.txt

# Export image
tboardfs export path/to/events.out.tfevents.xxx images/sample_image/000042.png -o image.png

# Export histogram
tboardfs export path/to/events.out.tfevents.xxx histograms/weights.txt -o weights_dist.txt

# Export audio
tboardfs export path/to/events.out.tfevents.xxx audio/sound/000001.wav -o sound.wav

# Export text
tboardfs export path/to/events.out.tfevents.xxx text/logs/000010.txt -o log.txt
```

### Extract all data

```bash
# Extract all data to directory
tboardfs extract path/to/events.out.tfevents.xxx -o output_directory

# Extract without sorting scalar files
tboardfs extract path/to/events.out.tfevents.xxx -o output_directory --no-sort

# Extract with custom digit padding
tboardfs extract path/to/events.out.tfevents.xxx -o output_directory --digits 4
```

## Virtual Filesystem Structure

The virtual filesystem organizes TensorBoard data as follows:

```
scalars/
  loss.txt                    # All loss values (iteration<tab>value)
  accuracy.txt               # All accuracy values
  metrics_precision.txt      # Note: '/' in tags replaced with '_'
images/
  sample_images_rgb/
    000000.png               # Image at iteration 0
    000001.png               # Image at iteration 1
    ...
histograms/
  distributions_normal.txt   # Histogram data in readable format
  model_weights.txt         # Model weights distribution
audio/
  sounds_sine_wave/
    000000.wav              # Audio at iteration 0
    000001.wav              # Audio at iteration 1
    ...
text/
  logs_info/
    000000.txt              # Text log at iteration 0
    000001.txt              # Text log at iteration 1
    ...
```

The `--digits` option controls the zero-padding of iteration numbers in filenames (default: 6 digits).

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=tboardfs

# Run specific test file
uv run pytest tests/test_parser.py
```

### Test Data Generation

The package includes a test data generator that creates TensorBoard event files with all supported data types:

```python
from tboardfs.test_generator import generate_test_tensorboard_log

# Generate test event file with 11 iterations
event_file = generate_test_tensorboard_log("./test_logs", num_iterations=11)
```

This creates dummy data for:

- Scalars: loss, accuracy, learning_rate, metrics
- Images: RGB and grayscale samples
- Histograms: normal/uniform distributions, model weights
- Audio: sine wave samples
- Text: iteration logs, model architecture

## Future Work

- FUSE filesystem interface for mounting TensorBoard logs
- Support for additional TensorBoard data types
- Real-time monitoring of growing event files

## License

[License information to be added]
