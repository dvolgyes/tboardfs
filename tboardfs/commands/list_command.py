"""List command implementation for tboardfs."""

from pathlib import Path
from typing import Any

from loguru import logger

from ..core.file_utils import create_parser_with_progress, get_event_files_sorted
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
    event_files = get_event_files_sorted(directory)
    if not event_files:
        return

    # Check if we should provide aggregated view (TensorBoard-like)
    show_progress = context.get("cli_mode", False) if context else False
    if len(event_files) > 1 and context and context.get("aggregated", True):
        list_directory_aggregated(directory, event_files, digits, show_progress)
    else:
        for file in event_files:
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

    # Aggregate content from all event files
    aggregated_content = _aggregate_content_from_files(
        directory, event_files, show_progress
    )

    # Sort all aggregated tags
    for data_type in aggregated_content:
        aggregated_content[data_type].sort()

    # Display aggregated content using consolidated reporter logic
    ContentReporter.display_content_by_type_static(aggregated_content)


def _aggregate_content_from_files(
    directory: Path, event_files: list[Path], show_progress: bool = False
) -> dict[str, list[str]]:
    """Aggregate content from multiple TensorBoard event files."""
    aggregated_content: dict[str, list[str]] = {
        "scalars": [],
        "images": [],
        "histograms": [],
        "tensors": [],
        "audio": [],
        "text": [],
    }

    for file_path in event_files:
        try:
            parser = create_parser_with_progress(str(file_path), show_progress)
            content = parser.list_all_content()

            # Process content with directory context
            _merge_content_with_context(
                aggregated_content, content, file_path, directory
            )

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    return aggregated_content


def _merge_content_with_context(
    aggregated_content: dict[str, list[str]],
    content: dict[str, list[str]],
    file_path: Path,
    directory: Path,
) -> None:
    """Merge content from a file into aggregated content with directory context."""
    # Get directory context for naming
    relative_path = file_path.relative_to(directory)
    parent_dir = relative_path.parent

    # Create contextual tag names
    for data_type, tags in content.items():
        for tag in tags:
            contextual_tag = _create_contextual_tag_name(tag, parent_dir)

            if contextual_tag not in aggregated_content[data_type]:
                aggregated_content[data_type].append(contextual_tag)


def _create_contextual_tag_name(tag: str, parent_dir: Path) -> str:
    """Create a contextual tag name based on directory structure."""
    if parent_dir.name and parent_dir.name != ".":
        # For class-specific metrics, use directory name instead of nested tag
        # e.g., "train_F1_cls_00" instead of "train_F1_cls_00/train/F1"
        if tag.split("/")[-1] in parent_dir.name:
            # Directory already contains the metric name, use directory name
            return parent_dir.name
        else:
            # Use directory name as prefix
            return f"{parent_dir.name}/{tag}"
    else:
        # Use original tag for root directory files
        return tag
