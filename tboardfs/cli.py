from pathlib import Path
import sys
from typing import cast

import click
from loguru import logger
from mfusepy import FUSE, Operations

from tboardfs.filesystem import TensorBoardFS


@click.command()
@click.argument("source", type=click.Path(exists=True, file_okay=False))
@click.argument("mountpoint", type=click.Path(file_okay=False))
@click.option("--step-digits", type=int, default=6, show_default=True)
@click.option("--refresh-age-seconds", type=float, default=60.0, show_default=True)
@click.option("--scalar-format", default="json,tsv,npz", show_default=True)
@click.option("--foreground/--background", default=False, show_default=True)
@click.option("--logfile", type=click.Path(dir_okay=False))
@click.option("--loglevel", default="INFO", show_default=True)
def main(
    **params: object,
) -> None:
    """Mount TensorBoard logs as a read-only virtual filesystem."""
    source_path = Path(cast(str, params["source"]))
    mountpoint_path = Path(cast(str, params["mountpoint"]))
    logfile = cast(str | None, params["logfile"])
    logfile_path = Path(logfile) if logfile is not None else None
    loglevel = cast(str, params["loglevel"])
    foreground = cast(bool, params["foreground"])
    _configure_logging(logfile_path, loglevel, foreground)
    if foreground:
        logger.info(
            "background mode is usually preferred; foreground mode is recommended "
            "only for testing"
        )
    logger.info("mounting {} at {}", source_path, mountpoint_path)

    class MountedTensorBoardFS(TensorBoardFS, Operations):  # type: ignore[misc]
        """Concrete mfusepy operations class."""

    filesystem = MountedTensorBoardFS(
        source_path,
        step_digits=cast(int, params["step_digits"]),
        refresh_age_seconds=cast(float, params["refresh_age_seconds"]),
        scalar_formats=cast(str, params["scalar_format"]),
        log_to_screen=foreground,
    )
    FUSE(filesystem, str(mountpoint_path), foreground=foreground, ro=True)


def _configure_logging(logfile: Path | None, loglevel: str, foreground: bool) -> None:
    """Configure loguru output for the CLI."""
    logger.remove()
    if foreground:
        logger.add(
            sys.stderr, level=loglevel.upper(), format="<level>{message}</level>"
        )
    if logfile is not None:
        logger.add(logfile, level=loglevel.upper())
