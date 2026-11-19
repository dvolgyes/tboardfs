"""Configure command diagnostics separately from exported stdout data."""

from pathlib import Path
import sys

import click
from loguru import logger

from tboardfs.cli_options import LOGLEVEL
from tboardfs.constants import LOG_FORMAT
from tboardfs.file_cli_commands import register_file_cli_commands


@click.group()
@click.option("--logfile", type=click.Path(dir_okay=False))
@LOGLEVEL
def main(logfile: str | None, loglevel: str) -> None:
    """Inspect a TensorBoard event file or log directory without FUSE."""
    try:
        _configure_logging(Path(logfile) if logfile is not None else None, loglevel)
    except OSError as error:
        raise click.ClickException(
            f"could not open logfile {logfile}: {error}"
        ) from error


def _configure_logging(logfile: Path | None, loglevel: str) -> None:
    """Configure loguru for non-data CLI output."""
    logger.remove()
    minimum = logger.level(loglevel.upper()).no

    for sink in (sys.stderr, logfile):
        if sink is not None:
            logger.add(
                sink,
                level=0,
                filter=lambda record: (
                    bool(record["extra"].get("command_status"))
                    or record["level"].no >= minimum
                ),
                format=LOG_FORMAT,
            )


register_file_cli_commands(main)
