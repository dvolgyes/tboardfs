"""File and path utilities for tboardfs."""

import sys
from pathlib import Path
from typing import Any

import click
from loguru import logger

from ..parser import TensorBoardParser


def validate_tensorboard_file(path: Path) -> bool:
    """Validate that the path is a valid TensorBoard event file."""
    return path.is_file() and "tfevents" in path.name


def setup_cli_context(ctx: click.Context) -> dict[str, Any]:
    """Set up CLI context with common configuration."""
    ctx.ensure_object(dict)
    ctx.obj["cli_mode"] = True
    ctx.obj["aggregated"] = True  # Enable TensorBoard-like aggregated view by default
    return ctx.obj


def create_parser_with_progress(
    file_path: str, show_progress: bool = False
) -> TensorBoardParser:
    """Create a TensorBoardParser with optional progress display."""
    logger.debug(f"Creating parser for file: {file_path} (progress: {show_progress})")
    return TensorBoardParser(file_path, show_progress=show_progress)


def validate_and_exit_on_error(tensorboard_path: str) -> Path:
    """Validate TensorBoard file path and exit with error if invalid."""
    logger.debug(f"Validating TensorBoard file: {tensorboard_path}")
    path = Path(tensorboard_path)
    if not validate_tensorboard_file(path):
        logger.error(f"{tensorboard_path} is not a valid TensorBoard event file")
        sys.exit(1)
    logger.debug(f"File validation passed: {path}")
    return path


def sanitize_tag_for_path(tag: str) -> str:
    """Convert tag name to filesystem-safe format."""
    return tag.replace("/", "_")


def restore_tag_from_path(safe_tag: str) -> str:
    """Restore original tag name from filesystem-safe format."""
    return safe_tag.replace("_", "/")


def extract_step_from_filename(filename: str) -> int:
    """Extract step number from padded filename."""
    return int(filename.split(".")[0])


def handle_standard_error(error: Exception, context: str) -> None:
    """Handle standard error logging and re-raise."""
    logger.exception(f"Error {context}: {error}")
    raise error


def validate_image_format_options(png: bool, jpg: bool) -> str:
    """Validate image format options and return the selected format.

    Args:
        png: Whether PNG format was requested
        jpg: Whether JPG format was requested

    Returns:
        The selected image format as a string ("png" or "jpg")

    Raises:
        SystemExit: If both PNG and JPG are specified
    """
    if png and jpg:
        logger.error("Cannot specify both --png and --jpg. Please choose one.")
        sys.exit(1)

    return "png" if png else "jpg"


def get_event_files_sorted(directory: Path) -> list[Path]:
    """Get all TensorBoard event files in a directory, sorted by path.

    Args:
        directory: The directory to search for event files

    Returns:
        A sorted list of paths to TensorBoard event files

    Raises:
        None, but logs a warning if no files are found
    """
    event_files = list(directory.rglob("*.tfevents.*"))
    if not event_files:
        logger.warning(f"No TensorBoard event files found in {directory}")
        return []

    return sorted(event_files)
