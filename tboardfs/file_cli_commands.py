"""Register virtual-tree commands for event files and log directories."""

import sys
from pathlib import PurePosixPath
from typing import cast

import click
from loguru import logger

from tboardfs.cli_options import (
    HISTOGRAM_FORMAT,
    PR_CURVE_NPY,
    SCALAR_FORMAT,
    STEP_DIGITS,
)
from tboardfs.file_tree import SingleEventTree, _CopyError, _NoEventError
from tboardfs.host_publish import OutputPublisher, _OutputConflictError
from tboardfs.model import TFRecordError
from tboardfs.paths import _Paths

_VIEW = click.option("-v", "--view", type=click.Choice(["typed", "raw", "both"]))
_status = logger.bind(command_status=True)


def register_file_cli_commands(group: click.Group) -> None:
    """Register single-event file commands on a Click group."""

    @group.command("list")
    @click.argument("source", type=click.Path(exists=True))
    @click.argument("prefix", required=False, default="/")
    @_VIEW
    @STEP_DIGITS
    @SCALAR_FORMAT
    @HISTOGRAM_FORMAT
    @PR_CURVE_NPY
    def list_files(**params: object) -> None:
        """Print virtual file paths below PREFIX."""
        tree, view = _FileCli.open_tree(cast(str, params["source"]), params)
        prefix = cast(str, params["prefix"])
        try:
            for path in tree.iter_file_paths(prefix=prefix, view=view):
                sys.stdout.write(path + "\n")
        except FileNotFoundError as error:
            raise click.ClickException(f"virtual path not found: {prefix}") from error

    @group.command("get")
    @click.argument("source", type=click.Path(exists=True))
    @click.argument("vpath")
    @click.option("-o", "--output", required=True, metavar="PATH")
    @click.option("--force", is_flag=True)
    @_VIEW
    @STEP_DIGITS
    @SCALAR_FORMAT
    @HISTOGRAM_FORMAT
    @PR_CURVE_NPY
    def get_file(**params: object) -> None:
        """Write one virtual file to OUTPUT."""
        source = cast(str, params["source"])
        vpath = cast(str, params["vpath"])
        output = cast(str, params["output"])
        force = cast(bool, params["force"])
        tree, view = _FileCli.open_tree(source, params)
        try:
            data = tree.read_file(vpath, view=view or "both")
        except FileNotFoundError as error:
            raise click.ClickException(f"virtual path not found: {vpath}") from error
        except IsADirectoryError as error:
            raise click.ClickException(
                f"virtual path is a directory: {vpath}"
            ) from error
        except (OSError, TFRecordError) as error:
            raise click.ClickException(
                f"could not read {source}:{vpath}: {error}"
            ) from error
        if output == "-":
            sys.stdout.buffer.write(data)
            return
        try:
            output_path = OutputPublisher.publish_get(
                output,
                PurePosixPath(_Paths.norm_virtual_path(vpath)).name,
                data,
                "overwrite" if force else "fail",
            )
        except _OutputConflictError as error:
            raise click.ClickException(
                f"output already exists: {output} ({error})\nUse --force for forced overwriting."
            ) from error
        except (OSError, ValueError) as error:
            raise click.ClickException(f"could not write {output}: {error}") from error
        _status.info("wrote {} to {}", _Paths.norm_virtual_path(vpath), output_path)

    @group.command("copy-all")
    @click.argument("source", type=click.Path(exists=True))
    @click.argument("outdir")
    @click.option("-f", "--force", is_flag=True)
    @click.option("-s", "--skip-existing", is_flag=True)
    @_VIEW
    @STEP_DIGITS
    @SCALAR_FORMAT
    @HISTOGRAM_FORMAT
    @PR_CURVE_NPY
    def copy_all(**params: object) -> None:
        """Copy every virtual file into OUTDIR."""
        force = cast(bool, params["force"])
        skip_existing = cast(bool, params["skip_existing"])
        if force and skip_existing:
            raise click.UsageError("Use either --force or --skip-existing, not both.")
        existing = "overwrite" if force else "skip" if skip_existing else "fail"
        tree, view = _FileCli.open_tree(cast(str, params["source"]), params)
        try:
            copied, skipped = tree.copy_all(
                cast(str, params["outdir"]),
                existing=existing,
                view=view or "typed",
            )
        except _CopyError as error:
            _status.info("Copied successfully: {} files", len(error.copied_paths))
            for path in error.copied_paths:
                _status.info("{}", path)
            for path in error.skipped_paths:
                _status.warning("Skipped existing: {}", path)
            raise click.ClickException(str(error)) from error
        for path in skipped:
            _status.warning("Skipped existing: {}", path)
        _status.info("copied {} files", copied)


class _FileCli:
    """Open sources and report results for the single-file commands."""

    @staticmethod
    def open_tree(
        source: str, params: dict[str, object]
    ) -> tuple[SingleEventTree, str | None]:
        """Return the tree of a source; a source without events is a CLI error."""
        framing_only = click.get_current_context().meta.get("framing_only", False)
        view = cast(str | None, params["view"])
        if framing_only and view not in (None, "raw"):
            raise click.UsageError("--framing-only requires --view raw or no --view.")
        try:
            if framing_only:
                return SingleEventTree.from_framing(source), "raw"
            return SingleEventTree(
                source,
                step_digits=cast(int, params["step_digits"]),
                scalar_format=cast(str, params["scalar_format"]),
                histogram_format=cast(str, params["histogram_format"]),
                pr_curve_npy=cast(bool, params["pr_curve_npy"]),
            ), view
        except _NoEventError as error:
            raise click.ClickException(str(error)) from error
        except (OSError, TFRecordError) as error:
            raise click.ClickException(f"could not read {source}: {error}") from error
