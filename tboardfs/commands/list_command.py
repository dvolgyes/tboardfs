"""List command implementation for tboardfs."""

from pathlib import Path
from typing import Any

from loguru import logger

from ..core.file_utils import create_parser_with_progress
from ..core.reporting import ContentReporter, list_directory_files


def list_single_file(
    file_path: Path, digits: int = 6, context: dict[str, Any] | None = None
) -> None:
    """List contents of a single TensorBoard event file."""
    try:
        show_progress = context.get("cli_mode", False) if context else False
        parser = create_parser_with_progress(str(file_path), show_progress)
        reporter = ContentReporter(parser)

        reporter.display_file_header(file_path)

        content = parser.list_all_content()
        reporter.display_content_by_type(content)
        reporter.display_virtual_paths(digits)

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")


def list_directory(directory: Path) -> None:
    """List TensorBoard files in a directory."""
    list_directory_files(directory)


def list_directory_recursive(
    directory: Path, digits: int = 6, context: dict[str, Any] | None = None
) -> None:
    """List TensorBoard files recursively in a directory."""
    event_files = list(directory.rglob("*.tfevents.*"))
    if not event_files:
        logger.warning(f"No TensorBoard event files found in {directory}")
        return

    # Check if we should provide aggregated view (TensorBoard-like)
    show_progress = context.get("cli_mode", False) if context else False
    if len(event_files) > 1 and context and context.get("aggregated", True):
        list_directory_aggregated(directory, event_files, digits, show_progress)
    else:
        for file in sorted(event_files):
            list_single_file(file, digits, context)


def list_directory_aggregated(
    directory: Path,
    event_files: list[Path],
    digits: int = 6,
    show_progress: bool = False,
) -> None:
    """List TensorBoard files with aggregated view like TensorBoard UI."""
    logger.info(f"Aggregated contents of {directory}:")
    logger.info("=" * 60)

    # Aggregate all content across files with directory context
    aggregated_content: dict[str, list[str]] = {
        "scalars": [],
        "images": [],
        "histograms": [],
        "tensors": [],
        "audio": [],
        "text": [],
    }

    for file_path in sorted(event_files):
        try:
            parser = create_parser_with_progress(str(file_path), show_progress)
            content = parser.list_all_content()

            # Get directory context for naming
            relative_path = file_path.relative_to(directory)
            parent_dir = relative_path.parent

            # Create contextual tag names
            for data_type, tags in content.items():
                for tag in tags:
                    if parent_dir.name and parent_dir.name != ".":
                        # For class-specific metrics, use directory name instead of nested tag
                        # e.g., "train_F1_cls_00" instead of "train_F1_cls_00/train/F1"
                        if tag.split("/")[-1] in parent_dir.name:
                            # Directory already contains the metric name, use directory name
                            contextual_tag = parent_dir.name
                        else:
                            # Use directory name as prefix
                            contextual_tag = f"{parent_dir.name}/{tag}"
                    else:
                        # Use original tag for root directory files
                        contextual_tag = tag

                    if contextual_tag not in aggregated_content[data_type]:
                        aggregated_content[data_type].append(contextual_tag)

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    # Sort all aggregated tags
    for data_type in aggregated_content:
        aggregated_content[data_type].sort()

    # Display aggregated content using existing reporter logic
    _display_aggregated_content(aggregated_content)


def _display_aggregated_content(content: dict[str, list[str]]) -> None:
    """Display aggregated content organized by type."""
    if content["scalars"]:
        logger.info("\nScalars:")
        for tag in content["scalars"]:
            logger.info(f"  - {tag}")

    if content["images"]:
        logger.info("\nImages:")
        for tag in content["images"]:
            logger.info(f"  - {tag}")

    if content["histograms"]:
        logger.info("\nHistograms:")
        for tag in content["histograms"]:
            logger.info(f"  - {tag}")

    if content["tensors"]:
        logger.info("\nTensors:")
        for tag in content["tensors"]:
            logger.info(f"  - {tag}")

    if content["audio"]:
        logger.info("\nAudio:")
        for tag in content["audio"]:
            logger.info(f"  - {tag}")

    if content["text"]:
        logger.info("\nText:")
        for tag in content["text"]:
            logger.info(f"  - {tag}")
