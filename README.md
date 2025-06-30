# tboardfs

A powerful command-line tool for extracting and analyzing TensorBoard log data. Parse TensorBoard event files, export specific metrics, and extract complete datasets with ease.

## 🚀 Features

- **Universal TensorBoard Support**: Works with TensorFlow v2 event files where all data is stored as tensors
- **Complete Data Coverage**: Handles all major TensorBoard data types:
  - **Scalars**: Metrics, losses, accuracy curves
  - **Images**: Training visualizations, model outputs
  - **Histograms**: Weight distributions, activation patterns
  - **Audio**: Audio samples and generated sounds
  - **Text**: Logs, predictions, debug information
- **Smart Directory Processing**: Automatically aggregates multi-file experiments
- **Flexible Export Options**: Export specific items or entire datasets
- **Organized Output**: Clean directory structure with proper file naming
- **High Performance**: Memory-efficient processing for large log files

## 📦 Installation

### Using uvx (Recommended)

```bash
# Run directly without installation
uvx tboardfs list path/to/your/logs/

# Install for repeated use
uvx install tboardfs
```

### Using pip

```bash
pip install tboardfs
```

### Development Installation

```bash
git clone https://github.com/your-username/tboardfs.git
cd tboardfs
uv sync
```

## 🛠️ Usage

### List TensorBoard Contents

Explore what's in your TensorBoard logs:

```bash
# List single event file
uvx tboardfs list path/to/events.out.tfevents.1234567890.hostname.pid.0

# List entire experiment directory (aggregated view)
uvx tboardfs list path/to/experiment/logs/

# List without aggregation (show individual files)
uvx tboardfs list path/to/experiment/logs/ --no-recursive

# Custom precision for iteration numbers
uvx tboardfs list path/to/logs/ --digits 8
```

**Example Output:**

```
Aggregated contents of /experiments/my_model/logs:
============================================================

Scalars:
  - train_loss
  - train_accuracy
  - val_loss
  - val_accuracy
  - train_F1_cls_00
  - train_F1_cls_01
  - val_F1_cls_00

Images:
  - sample_predictions
  - model_architecture

Histograms:
  - layer_weights
  - gradients
```

### Extract Complete Datasets

Export all TensorBoard data to organized directories:

```bash
# Extract everything to a directory
uvx tboardfs extract path/to/logs/ -o extracted_data/

# Extract with custom settings
uvx tboardfs extract path/to/logs/ -o data/ --digits 4 --png --quality 95

# Extract without sorting scalars by iteration
uvx tboardfs extract path/to/logs/ -o data/ --no-sort
```

**Directory Structure Created:**

```
extracted_data/
├── scalars/
│   ├── train_loss.txt         # iteration    value
│   ├── train_accuracy.txt     # 0           0.123
│   └── val_loss.txt           # 1           0.098
├── images/                    # ...
│   └── sample_predictions/
│       ├── 000000.jpg
│       ├── 000001.jpg
│       └── 000002.jpg
├── histograms/
│   └── layer_weights.txt
└── audio/
    └── generated_sounds/
        ├── 000000.wav
        └── 000001.wav
```

### Export Specific Items

Target exactly what you need:

```bash
# Export scalar data to stdout
uvx tboardfs export path/to/logs/ scalars/train_loss.txt

# Export scalar data to file
uvx tboardfs export path/to/logs/ scalars/val_accuracy.txt -o accuracy.txt

# Export specific image
uvx tboardfs export path/to/logs/ images/predictions/000042.jpg -o prediction.jpg

# Export histogram data
uvx tboardfs export path/to/logs/ histograms/weights.txt -o weights_distribution.txt

# Export audio file
uvx tboardfs export path/to/logs/ audio/samples/000010.wav -o sample.wav

# Export text logs
uvx tboardfs export path/to/logs/ text/debug/000005.txt -o debug_log.txt
```

### Advanced Multi-Experiment Processing

Handle complex experimental setups with class-specific metrics:

```bash
# Process experiments with multiple classes/categories
uvx tboardfs list /experiments/multiclass_model/
# Shows: train_F1_cls_00, train_F1_cls_01, ..., val_F1_cls_16

# Extract preserving class structure
uvx tboardfs extract /experiments/multiclass_model/ -o results/
# Creates: train_F1_cls_00/, train_F1_cls_01/, etc.
```

## 🎯 Image Format Options

Control image output format and quality:

```bash
# Export images as PNG (lossless)
uvx tboardfs extract path/to/logs/ -o data/ --png

# Export images as JPEG with custom quality
uvx tboardfs extract path/to/logs/ -o data/ --jpg --quality 85

# Export specific image format
uvx tboardfs export path/to/logs/ images/plot/000001.png -o plot.png --png
```

## 📊 Understanding TensorBoard v2 Format

Modern TensorBoard files use TensorFlow v2 format where:

- **All data types** (scalars, images, etc.) are stored as tensors
- **Hierarchical organization** uses subdirectories for related metrics
- **Class-specific metrics** get separate subdirectories (e.g., `train_F1_cls_00/`, `val_F1_cls_01/`)
- **Aggregated views** combine related metrics for easier analysis

tboardfs automatically handles this complexity and presents data in an intuitive virtual filesystem structure.

## 🔧 Command Reference

### Global Options

```bash
--logfile FILE      Log to file
--debug             Enable debug logging
```

### List Command

```bash
uvx tboardfs list [OPTIONS] TENSORBOARD_PATH

Options:
  --no-recursive    Disable recursive listing for directories
  --digits INTEGER  Number of digits for iteration padding (default: 6)
```

### Extract Command

```bash
uvx tboardfs extract [OPTIONS] TENSORBOARD_PATH

Options:
  -o, --output PATH        Output directory [required]
  --no-sort               Disable sorting of scalar files
  --digits INTEGER        Iteration number padding (default: 6)
  --png                   Export images as PNG
  --jpg                   Export images as JPEG (default)
  --quality INTEGER       JPEG quality 0-100 (default: 90)
```

### Export Command

```bash
uvx tboardfs export [OPTIONS] TENSORBOARD_PATH VIRTUAL_PATH

Options:
  -o, --output PATH       Output file (default: stdout)
  --png                   Export images as PNG
  --jpg                   Export images as JPEG (default)
  --quality INTEGER       JPEG quality 0-100 (default: 90)
```

## 📁 Virtual Filesystem Paths

tboardfs organizes TensorBoard data into intuitive paths:

| Data Type  | Path Pattern                 | Example                        |
| ---------- | ---------------------------- | ------------------------------ |
| Scalars    | `scalars/tag_name.txt`       | `scalars/train_loss.txt`       |
| Images     | `images/tag_name/NNNNNN.ext` | `images/samples/000042.png`    |
| Histograms | `histograms/tag_name.txt`    | `histograms/layer_weights.txt` |
| Audio      | `audio/tag_name/NNNNNN.ext`  | `audio/sounds/000001.wav`      |
| Text       | `text/tag_name/NNNNNN.txt`   | `text/logs/000010.txt`         |

**Note**: Forward slashes in tag names are converted to underscores for filesystem compatibility.

## 🧪 Development & Testing

### Running Tests

```bash
# Install development dependencies
uv sync

# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov

# Run specific test
uv run pytest tests/test_cli.py -v
```

### Code Quality

```bash
# Run linting and formatting
uv run pre-commit run --all-files

# Type checking
uv run mypy tboardfs/
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
1. Create a feature branch
1. Add tests for new functionality
1. Ensure all tests pass
1. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/tboardfs/issues)
- **Documentation**: This README and inline help (`uvx tboardfs --help`)
- **Examples**: See `tests/` directory for usage examples

______________________________________________________________________

**⚡ Quick Start**: `uvx tboardfs list path/to/your/tensorboard/logs/`
