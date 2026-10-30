from pathlib import Path
import sys
from typing import cast

import click
from loguru import logger
from mfusepy import FUSE

from tboardfs.cli_options import LOGLEVEL, SCALAR_FORMAT, STEP_DIGITS
from tboardfs.constants import DEFAULT_CACHE_SIZE_MB, LOG_FORMAT
from tboardfs.mounted import MountedTensorBoardFS


@click.command()
@click.argument("source", type=click.Path(exists=True, file_okay=False))
@click.argument("mountpoint", type=click.Path(file_okay=False))
@STEP_DIGITS
@click.option("--refresh-age-seconds", type=float, default=60.0, show_default=True)
@SCALAR_FORMAT
@click.option(
    "--cache-size-mb",
    type=click.IntRange(min=0),
    default=DEFAULT_CACHE_SIZE_MB,
    show_default=True,
    help="Memory for rendered files; 0 turns the cache off.",
)
@click.option("--foreground/--background", default=False, show_default=True)
@click.option("--logfile", type=click.Path(dir_okay=False))
@LOGLEVEL
def main(
    **params: object,
) -> None:
    """Mount TensorBoard logs as a read-only virtual filesystem."""
    source_path = Path(cast(str, params["source"])).resolve()
    mountpoint_path = Path(cast(str, params["mountpoint"])).resolve()
    if mountpoint_path.is_relative_to(source_path):
        # The run scan would walk into the mount and call back into itself.
        raise click.BadParameter(
            f"{mountpoint_path} is inside the source {source_path}",
            param_hint="MOUNTPOINT",
        )
    logfile = cast(str | None, params["logfile"])
    logfile_path = Path(logfile).resolve() if logfile is not None else None
    loglevel = cast(str, params["loglevel"])
    foreground = cast(bool, params["foreground"])
    _configure_logging(logfile_path, loglevel, foreground)
    if foreground:
        logger.info(
            "background mode is usually preferred; foreground mode is recommended "
            "only for testing"
        )
    logger.info("mounting {} at {}", source_path, mountpoint_path)

    filesystem = MountedTensorBoardFS(
        source_path,
        step_digits=cast(int, params["step_digits"]),
        refresh_age_seconds=cast(float, params["refresh_age_seconds"]),
        scalar_formats=cast(str, params["scalar_format"]),
        log_to_screen=foreground,
        cache_size_mb=cast(int, params["cache_size_mb"]),
    )
    # attr_timeout=0 makes fstat and SEEK_END ask for the size of the open
    # handle; a cached size from an older version would cut its data.
    FUSE(
        filesystem,
        str(mountpoint_path),
        foreground=foreground,
        ro=True,
        raw_fi=True,
        attr_timeout=0,
    )


def _configure_logging(logfile: Path | None, loglevel: str, foreground: bool) -> None:
    """Configure loguru output for the CLI."""
    logger.remove()
    if foreground:
        logger.add(sys.stderr, level=loglevel.upper(), format=LOG_FORMAT)
    if logfile is not None:
        logger.add(logfile, level=loglevel.upper(), format=LOG_FORMAT)
