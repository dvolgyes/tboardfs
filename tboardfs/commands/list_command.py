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

    for file in sorted(event_files):
        list_single_file(file, digits, context)
