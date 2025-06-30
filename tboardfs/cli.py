"""CLI interface for tboardfs."""

import sys
from typing import Any
import click
from pathlib import Path
from loguru import logger

from .core.file_utils import (
    setup_cli_context,
    handle_standard_error,
    validate_image_format_options,
)
from .commands.list_command import (
    list_single_file,
    list_directory,
    list_directory_recursive,
)
from .commands.extract_command import extract_tensorboard_data
from .commands.export_command import export_virtual_path


def setup_logging(logfile: str | None = None, debug: bool = False) -> None:
    """Configure loguru logger."""
    # Remove default handler
    logger.remove()

    # Set log level based on debug flag
    log_level = "DEBUG" if debug else "INFO"

    # Add stderr handler with appropriate level
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
        colorize=True,
    )

    # Add file handler if logfile is specified
    if logfile:
        logger.add(
            logfile,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level="DEBUG",
            rotation="10 MB",
        )
        logger.info(f"Logging to file: {logfile}")


@click.group()
@click.option("--logfile", type=click.Path(), help="Log file path")
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.pass_context
def main(ctx: Any, logfile: str | None, debug: bool) -> None:
    """TensorBoard filesystem interface CLI.

    This CLI tool works with TensorFlow v2 event file format where data is stored
    as tensors. The tool extracts and organizes the data into a logical filesystem
    structure for easy access.
    """
    setup_logging(logfile, debug)
    # Store in context for access by subcommands
    ctx.ensure_object(dict)
    ctx.obj["logfile"] = logfile
    ctx.obj["debug"] = debug


@main.command()
@click.argument("tensorboard_path", type=click.Path(exists=True))
@click.option(
    "--no-recursive",
    is_flag=True,
    help="Disable recursive listing for directories (list event files only)",
)
@click.option(
    "--digits",
    type=int,
    default=6,
    help="Number of digits for padding iteration numbers (default: 6)",
)
@click.pass_context
def list(ctx: Any, tensorboard_path: str, no_recursive: bool, digits: int) -> None:
    """List contents of TensorBoard log file(s).

    For directories, recursively lists and aggregates all event files by default.
    Use --no-recursive to only list the event files without processing content.
    """
    logger.info(f"Listing contents of {tensorboard_path}")
    path = Path(tensorboard_path)
    context = setup_cli_context(ctx)

    try:
        if path.is_file() and "tfevents" in path.name:
            list_single_file(path, digits, context)
        elif path.is_dir():
            if no_recursive:
                list_directory(path)
            else:
                list_directory_recursive(path, digits, context)
        else:
            logger.error(
                f"{tensorboard_path} is not a valid TensorBoard log file or directory"
            )
            sys.exit(1)
    except Exception as e:
        handle_standard_error(e, "listing contents")


@main.command()
@click.argument("tensorboard_path", type=click.Path(exists=True))
@click.option(
    "-o", "--output", type=click.Path(), required=True, help="Output directory path"
)
@click.option(
    "--no-sort",
    is_flag=True,
    help="Disable sorting of scalar files by iteration number",
)
@click.option(
    "--digits",
    type=int,
    default=6,
    help="Number of digits for padding iteration numbers (default: 6)",
)
@click.option(
    "--png",
    is_flag=True,
    help="Export images in PNG format (default: JPG)",
)
@click.option(
    "--jpg",
    is_flag=True,
    help="Export images in JPG format (default: JPG)",
)
@click.option(
    "--quality",
    type=click.IntRange(0, 100),
    default=90,
    help="Quality for JPG images (0-100, default: 90)",
)
@click.pass_context
def extract(
    ctx: Any,
    tensorboard_path: str,
    output: str,
    no_sort: bool,
    digits: int,
    png: bool,
    jpg: bool,
    quality: int,
) -> None:
    """Extract all data from TensorBoard log(s) to directory structure.

    For directories, automatically processes all event files recursively and
    organizes output by context (subdirectories for class-specific metrics).
    """
    setup_cli_context(ctx)

    image_format = validate_image_format_options(png, jpg)

    try:
        sort_scalars = not no_sort
        extract_tensorboard_data(
            tensorboard_path, output, sort_scalars, digits, image_format, quality
        )
    except Exception as e:
        handle_standard_error(e, "extracting data")


@main.command()
@click.argument("tensorboard_path", type=click.Path(exists=True))
@click.argument("virtual_path", type=str)
@click.option("-o", "--output", type=click.Path(), help="Output file path")
@click.option(
    "--png",
    is_flag=True,
    help="Export images in PNG format (default: JPG)",
)
@click.option(
    "--jpg",
    is_flag=True,
    help="Export images in JPG format (default: JPG)",
)
@click.option(
    "--quality",
    type=click.IntRange(0, 100),
    default=90,
    help="Quality for JPG images (0-100, default: 90)",
)
@click.pass_context
def export(
    ctx: Any,
    tensorboard_path: str,
    virtual_path: str,
    output: str | None,
    png: bool,
    jpg: bool,
    quality: int,
) -> None:
    """Export a specific item from TensorBoard log(s).

    For directories, automatically searches through all event files to find
    the requested virtual path and exports from the correct source.
    """
    logger.info(f"Exporting {virtual_path} from {tensorboard_path}")
    context = setup_cli_context(ctx)

    image_format = validate_image_format_options(png, jpg)

    try:
        show_progress = context.get("cli_mode", False)
        export_virtual_path(
            tensorboard_path, virtual_path, output, show_progress, image_format, quality
        )
    except Exception as e:
        handle_standard_error(e, "exporting data")


if __name__ == "__main__":
    main()
