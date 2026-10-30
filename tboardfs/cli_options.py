"""Click options that the mount and single-file commands share."""

import click

from tboardfs.constants import LOG_LEVELS, MAX_STEP_DIGITS, SCALAR_FORMATS
from tboardfs.paths import _Paths


def _scalar_formats(_ctx: click.Context, _param: click.Parameter, value: str) -> str:
    """Refuse unsupported format names before a tree is built."""
    try:
        _Paths.normalize_formats(value)
    except ValueError as error:
        raise click.BadParameter(str(error)) from error
    return value


STEP_DIGITS = click.option(
    "--step-digits",
    type=click.IntRange(1, MAX_STEP_DIGITS),
    default=6,
    show_default=True,
    help="Zero-padding width of step file names.",
)
SCALAR_FORMAT = click.option(
    "--scalar-format",
    default="json,tsv,npz",
    show_default=True,
    callback=_scalar_formats,
    help=f"Comma-separated table formats: {', '.join(SCALAR_FORMATS)}.",
)
LOGLEVEL = click.option(
    "--loglevel",
    type=click.Choice(LOG_LEVELS, case_sensitive=False),
    default="INFO",
    show_default=True,
)
