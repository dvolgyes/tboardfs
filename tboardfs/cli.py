import math
import os
import sys
from pathlib import Path
from typing import cast

import click
from loguru import logger

from tboardfs._mount_session import mount
from tboardfs.cli_options import (
    EPOCH_BREAKDOWN,
    HISTOGRAM_FORMAT,
    LOGLEVEL,
    PR_CURVE_NPY,
    SCALAR_FORMAT,
    STEP_DIGITS,
)
from tboardfs.constants import (
    DEFAULT_CACHE_SIZE_MB,
    DEFAULT_DISCOVERY_SECONDS,
    DEFAULT_INDEX_WORKERS,
    DEFAULT_SOURCE_CHECK_SECONDS,
    LOG_FORMAT,
)
from tboardfs.mount_index import _DEFAULT_DISK_INDEX_SIZE_MB
from tboardfs.mounted import MountedTensorBoardFS
from tboardfs.stdlib_logging import StdlibToLoguru


@click.command()
@click.argument("source", type=click.Path(exists=True, file_okay=False))
@click.argument("mountpoint", type=click.Path(file_okay=False))
@STEP_DIGITS
@click.option(
    "--refresh-age-seconds",
    type=click.FloatRange(min=0),
    default=DEFAULT_DISCOVERY_SECONDS,
    show_default=True,
    callback=lambda _ctx, _param, value: _number(value),
    help="Interval for local discovery; inf freezes each directory after its first visit.",
)
@click.option(
    "--source-check-seconds",
    type=click.FloatRange(min=0),
    default=DEFAULT_SOURCE_CHECK_SECONDS,
    show_default=True,
    callback=lambda _ctx, _param, value: _number(value, finite=True),
    help=(
        "Finite interval for checking known event files in an accessed run; "
        "0 checks on every live access."
    ),
)
@SCALAR_FORMAT
@HISTOGRAM_FORMAT
@PR_CURVE_NPY
@click.option(
    "--cache-size-mb",
    type=click.IntRange(min=0),
    default=DEFAULT_CACHE_SIZE_MB,
    show_default=True,
    help="Memory for rendered files; 0 turns the cache off.",
)
@click.option(
    "-j",
    "--index-workers",
    type=click.IntRange(min=1),
    metavar="N",
    default=DEFAULT_INDEX_WORKERS,
    show_default=True,
    help="Maximum workers for cold indexing of event files at least 8 MiB.",
)
@click.option(
    "--disk-index",
    is_flag=True,
    help="Reuse verified whole-event-file indexes across mounts.",
)
@click.option(
    "--disk-index-size-mb",
    type=click.IntRange(min=1),
    default=_DEFAULT_DISK_INDEX_SIZE_MB,
    show_default=True,
    help="Disk budget for persistent indexes (requires --disk-index).",
)
@click.option("-r", "--raw", is_flag=True, help="Expose exact event frames under raw/.")
@EPOCH_BREAKDOWN
@click.option("--foreground/--background", default=False, show_default=True)
@click.option(
    "--root-owner/--user-owner",
    default=True,
    show_default=True,
    help="Owner of every file and directory: root, or the user who mounts.",
)
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
    if source_path.is_relative_to(mountpoint_path):
        # The mount hides the source, so each read of it would call the mount.
        raise click.BadParameter(
            f"{source_path} is inside the mountpoint {mountpoint_path}",
            param_hint="SOURCE",
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
        mountpoint=mountpoint_path,
        step_digits=cast(int, params["step_digits"]),
        refresh_age_seconds=cast(float, params["refresh_age_seconds"]),
        source_check_seconds=cast(float, params["source_check_seconds"]),
        scalar_formats=cast(str, params["scalar_format"]),
        histogram_formats=cast(str, params["histogram_format"]),
        pr_curve_npy=cast(bool, params["pr_curve_npy"]),
        epoch_breakdown=cast(bool, params["epoch_breakdown"]),
        log_to_screen=foreground,
        cache_size_mb=cast(int, params["cache_size_mb"]),
        owner=(0, 0) if params["root_owner"] else (os.getuid(), os.getgid()),
        index_workers=cast(int, params["index_workers"]),
        disk_index=cast(bool, params["disk_index"]),
        disk_index_size_mb=cast(int, params["disk_index_size_mb"]),
        raw=cast(bool, params["raw"]),
    )
    try:
        mount(
            filesystem,
            str(mountpoint_path),
            foreground=foreground,
            ro=True,
            raw_fi=True,
            attr_timeout=0,
        )
    except RuntimeError as error:
        # libfuse writes the reason to stderr; mfusepy raises its exit status.
        raise click.ClickException(
            f"could not mount {mountpoint_path} (FUSE status {error})"
        ) from error


def _number(value: float, *, finite: bool = False) -> float:
    """Refuse NaN and optionally infinity after Click's range check."""
    if math.isnan(value):
        raise click.BadParameter(f"{value} is not a number")
    if finite and math.isinf(value):
        raise click.BadParameter(f"{value} is not finite")
    return value


def _configure_logging(logfile: Path | None, loglevel: str, foreground: bool) -> None:
    """Configure loguru output for the CLI."""
    logger.remove()
    if foreground:
        logger.add(sys.stderr, level=loglevel.upper(), format=LOG_FORMAT)
    if logfile is not None:
        logger.add(logfile, level=loglevel.upper(), format=LOG_FORMAT)
    StdlibToLoguru.install()
