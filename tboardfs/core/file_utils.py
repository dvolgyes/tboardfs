"""File and path utilities for tboardfs."""

import sys
from pathlib import Path
from typing import Any

import click
from loguru import logger

from ..parser import TensorBoardParser


# Supported data types for filtering
SUPPORTED_DATA_TYPES = {
    "image",
    "audio",
    "video",
    "text",
    "scalar",
    "histogram",
    "mesh",
    "hyperparameter",
}


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


def validate_audio_format_options(wav: bool, mp3: bool) -> str:
    """Validate audio format options and return the selected format.

    Args:
        wav: Whether WAV format was requested
        mp3: Whether MP3 format was requested

    Returns:
        The selected audio format as a string ("wav" or "mp3")

    Raises:
        SystemExit: If both WAV and MP3 are specified
    """
    if wav and mp3:
        logger.error("Cannot specify both --wav and --mp3. Please choose one.")
        sys.exit(1)

    return "wav" if wav else "mp3"


def validate_ply_format_options(ply_bin: bool, ply_txt: bool) -> str:
    """Validate PLY format options and return the selected format.

    Args:
        ply_bin: Whether binary PLY format was requested
        ply_txt: Whether text PLY format was requested

    Returns:
        The selected PLY format as a string ("binary" or "text")

    Raises:
        SystemExit: If both binary and text PLY are specified
    """
    if ply_bin and ply_txt:
        logger.error("Cannot specify both --ply-bin and --ply-txt. Please choose one.")
        sys.exit(1)

    return "text" if ply_txt else "binary"


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


def parse_data_type_list(data_types: tuple[str, ...]) -> set[str]:
    """Parse comma-separated data types from CLI arguments.

    Args:
        data_types: Tuple of data type strings, each may contain comma-separated values

    Returns:
        Set of individual data type strings

    Raises:
        SystemExit: If any unsupported data types are specified
    """
    result = set()

    for type_group in data_types:
        # Split by comma and strip whitespace
        types = [t.strip().lower() for t in type_group.split(",")]
        result.update(types)

    # Validate all types are supported
    invalid_types = result - SUPPORTED_DATA_TYPES
    if invalid_types:
        logger.error(f"Unsupported data types: {', '.join(sorted(invalid_types))}")
        logger.error(f"Supported types: {', '.join(sorted(SUPPORTED_DATA_TYPES))}")
        sys.exit(1)

    return result


def validate_type_filtering_options(
    ignore_types: tuple[str, ...], select_types: tuple[str, ...]
) -> dict[str, set[str]]:
    """Validate and process data type filtering options.

    Args:
        ignore_types: Types to ignore during processing
        select_types: Types to select (process only these)

    Returns:
        Dictionary with 'ignore' and 'select' keys containing sets of data types

    Raises:
        SystemExit: If both ignore and select are specified or if invalid types are given
    """
    ignore_set = parse_data_type_list(ignore_types) if ignore_types else set()
    select_set = parse_data_type_list(select_types) if select_types else set()

    # Check for mutual exclusivity
    if ignore_set and select_set:
        logger.error(
            "Cannot specify both --ignore and --select options. Please choose one."
        )
        sys.exit(1)

    return {"ignore": ignore_set, "select": select_set}
