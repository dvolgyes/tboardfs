from pathlib import Path
import sys

import click
from loguru import logger

from tboardfs.constants import LOG_FORMAT
from tboardfs.file_cli_commands import register_file_cli_commands


@click.group()
@click.option("--logfile", type=click.Path(dir_okay=False))
@click.option("--loglevel", default="INFO", show_default=True)
def main(logfile: str | None, loglevel: str) -> None:
    """Inspect one TensorBoard event file without mounting FUSE."""
    _configure_logging(Path(logfile) if logfile is not None else None, loglevel)


def _configure_logging(logfile: Path | None, loglevel: str) -> None:
    """Configure loguru for non-data CLI output."""
    logger.remove()
    logger.add(sys.stderr, level=loglevel.upper(), format=LOG_FORMAT)
    if logfile is not None:
        logger.add(logfile, level=loglevel.upper(), format=LOG_FORMAT)


register_file_cli_commands(main)
