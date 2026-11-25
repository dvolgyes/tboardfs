"""Click options that the mount and single-file commands share."""

import click

from tboardfs.constants import (
    DEFAULT_HISTOGRAM_FORMATS,
    DEFAULT_SCALAR_FORMATS,
    LOG_LEVELS,
    MAX_STEP_DIGITS,
    TABLE_FORMATS,
)
from tboardfs.export_formats import normalize_formats


def _table_formats(_ctx: click.Context, param: click.Parameter, value: str) -> str:
    """Refuse unsupported format names before a tree is built."""
    try:
        normalize_formats(value, (param.name or "").removesuffix("_format"))
    except ValueError as error:
        raise click.BadParameter(str(error), param=param) from error
    return value


STEP_DIGITS = click.option(
    "--step-digits",
    type=click.IntRange(1, MAX_STEP_DIGITS),
    default=6,
    show_default=True,
    help="Zero-padding width of step file names.",
)
SCALAR_FORMAT = click.option(
    "-S",
    "--scalar-format",
    metavar="LIST",
    default=",".join(DEFAULT_SCALAR_FORMATS),
    show_default=True,
    callback=_table_formats,
    help=f"Replace scalar formats with a comma-separated list: {', '.join(TABLE_FORMATS)}.",
)
HISTOGRAM_FORMAT = click.option(
    "-H",
    "--histogram-format",
    metavar="LIST",
    default=",".join(DEFAULT_HISTOGRAM_FORMATS),
    show_default=True,
    callback=_table_formats,
    help=(
        "Replace the formats for both histograms and distributions with a "
        f"comma-separated list: {', '.join(TABLE_FORMATS)}."
    ),
)
PR_CURVE_NPY = click.option(
    "-P",
    "--pr-curve-npy",
    is_flag=True,
    help="Add NPY files for PR curves. JSON and TSV are always available.",
)
EPOCH_BREAKDOWN = click.option(
    "-E",
    "--epoch-breakdown",
    is_flag=True,
    help="Group eligible step files and scalar rows by recorded epoch markers.",
)
LOGLEVEL = click.option(
    "--loglevel",
    type=click.Choice(LOG_LEVELS, case_sensitive=False),
    default="INFO",
    show_default=True,
)
