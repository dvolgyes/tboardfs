"""CLI interface for tboardfs."""

import sys
import click
from pathlib import Path
from loguru import logger

from .core.file_utils import setup_cli_context, handle_standard_error
from .commands.list_command import (
    list_single_file,
    list_directory,
    list_directory_recursive,
)
from .commands.extract_command import extract_tensorboard_data
from .commands.export_command import export_virtual_path


def setup_logging(logfile: str | None = None):
    """Configure loguru logger."""
    # Remove default handler
    logger.remove()

    # Add stderr handler with INFO level
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
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
@click.pass_context
def main(ctx, logfile: str | None):
    """TensorBoard filesystem interface CLI.

    This CLI tool works with TensorFlow v2 event file format where data is stored
    as tensors. The tool extracts and organizes the data into a logical filesystem
    structure for easy access.
    """
    setup_logging(logfile)
    # Store in context for access by subcommands
    ctx.ensure_object(dict)
    ctx.obj["logfile"] = logfile


@main.command()
@click.argument("tensorboard_path", type=click.Path(exists=True))
@click.option("--recursive", "-r", is_flag=True, help="List recursively")
@click.option(
    "--digits",
    type=int,
    default=6,
    help="Number of digits for padding iteration numbers (default: 6)",
)
@click.pass_context
def list(ctx, tensorboard_path: str, recursive: bool, digits: int):
    """List contents of TensorBoard log file(s)."""
    path = Path(tensorboard_path)
    context = setup_cli_context(ctx)

    try:
        if path.is_file() and "tfevents" in path.name:
            list_single_file(path, digits, context)
        elif path.is_dir():
            if recursive:
                list_directory_recursive(path, digits, context)
            else:
                list_directory(path)
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
@click.pass_context
def extract(ctx, tensorboard_path: str, output: str, no_sort: bool, digits: int):
    """Extract all data from TensorBoard log to directory structure."""
    setup_cli_context(ctx)

    try:
        sort_scalars = not no_sort
        extract_tensorboard_data(tensorboard_path, output, sort_scalars, digits)
    except Exception as e:
        handle_standard_error(e, "extracting data")


@main.command()
@click.argument("tensorboard_path", type=click.Path(exists=True))
@click.argument("virtual_path", type=str)
@click.option("-o", "--output", type=click.Path(), help="Output file path")
@click.pass_context
def export(ctx, tensorboard_path: str, virtual_path: str, output: str | None):
    """Export a specific item from TensorBoard log."""
    context = setup_cli_context(ctx)

    try:
        show_progress = context.get("cli_mode", False)
        export_virtual_path(tensorboard_path, virtual_path, output, show_progress)
    except Exception as e:
        handle_standard_error(e, "exporting data")


if __name__ == "__main__":
    main()
