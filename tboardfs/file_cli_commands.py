from pathlib import Path
from pathlib import PurePosixPath
import sys
from typing import cast

import click
from loguru import logger

from tboardfs.cli_options import SCALAR_FORMAT, STEP_DIGITS
from tboardfs.file_tree import SingleEventTree, _CopyConflictError, _NoEventError
from tboardfs.paths import _Paths


def register_file_cli_commands(group: click.Group) -> None:
    """Register single-event file commands on a Click group."""

    @group.command("list")
    @click.argument("source", type=click.Path(exists=True))
    @click.argument("prefix", required=False, default="/")
    @STEP_DIGITS
    @SCALAR_FORMAT
    def list_files(
        source: str, prefix: str, step_digits: int, scalar_format: str
    ) -> None:
        """Print virtual file paths below PREFIX."""
        tree = _FileCli.open_tree(source, step_digits, scalar_format)
        try:
            paths = tree.list_file_paths(prefix=prefix)
        except FileNotFoundError as error:
            raise click.ClickException(f"virtual path not found: {prefix}") from error
        for path in paths:
            sys.stdout.write(path + "\n")

    @group.command("get")
    @click.argument("source", type=click.Path(exists=True))
    @click.argument("vpath")
    @click.option(
        "-o",
        "--output",
        required=True,
        type=click.Path(writable=True),
    )
    @click.option("--force", is_flag=True)
    @STEP_DIGITS
    @SCALAR_FORMAT
    def get_file(
        source: str,
        vpath: str,
        output: str,
        force: bool,
        step_digits: int,
        scalar_format: str,
    ) -> None:
        """Write one virtual file to OUTPUT."""
        tree = _FileCli.open_tree(source, step_digits, scalar_format)
        try:
            data = tree.read_file(vpath)
        except FileNotFoundError as error:
            raise click.ClickException(f"virtual path not found: {vpath}") from error
        except IsADirectoryError as error:
            raise click.ClickException(
                f"virtual path is a directory: {vpath}"
            ) from error
        if output == "-":
            sys.stdout.buffer.write(data)
            return
        output_path = _FileCli.output_path(vpath, output)
        if output_path.exists() and not force:
            raise click.ClickException(
                f"output already exists: {output_path}\n"
                "Use --force for forced overwriting."
            )
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(data)
        except OSError as error:
            raise click.ClickException(
                f"could not write {output_path}: {error.strerror or error}"
            ) from error
        logger.info("wrote {} to {}", _Paths.norm_virtual_path(vpath), output_path)

    @group.command("copy-all")
    @click.argument("source", type=click.Path(exists=True))
    @click.argument("outdir", type=click.Path(file_okay=False))
    @click.option("-f", "--force", is_flag=True)
    @click.option("-s", "--skip-existing", is_flag=True)
    @STEP_DIGITS
    @SCALAR_FORMAT
    def copy_all(**params: object) -> None:
        """Copy every virtual file into OUTDIR."""
        force = cast(bool, params["force"])
        skip_existing = cast(bool, params["skip_existing"])
        if force and skip_existing:
            raise click.UsageError("Use either --force or --skip-existing, not both.")
        existing = "overwrite" if force else "skip" if skip_existing else "fail"
        tree = _FileCli.open_tree(
            cast(str, params["source"]),
            cast(int, params["step_digits"]),
            cast(str, params["scalar_format"]),
        )
        try:
            copied, skipped = tree.copy_all(
                cast(str, params["outdir"]), existing=existing
            )
        except _CopyConflictError as error:
            _FileCli.report_copy_conflict(error)
            raise click.exceptions.Exit(1) from error
        except OSError as error:
            raise click.ClickException(str(error)) from error
        for path in skipped:
            logger.warning("Skipped existing: {}", path)
        logger.info("copied {} files", copied)


class _FileCli:
    """Open sources and report results for the single-file commands."""

    @staticmethod
    def open_tree(source: str, step_digits: int, scalar_format: str) -> SingleEventTree:
        """Return the tree of a source; a source without events is a CLI error."""
        try:
            return SingleEventTree(
                source, step_digits=step_digits, scalar_format=scalar_format
            )
        except _NoEventError as error:
            raise click.ClickException(str(error)) from error

    @staticmethod
    def output_path(vpath: str, output: str) -> Path:
        """Return the file path that ``get`` writes a virtual file to."""
        output_path = Path(output)
        if output_path.exists() and output_path.is_dir():
            return output_path / PurePosixPath(_Paths.norm_virtual_path(vpath)).name
        if output.endswith("/"):
            return output_path / PurePosixPath(_Paths.norm_virtual_path(vpath)).name
        return output_path

    @staticmethod
    def report_copy_conflict(error: _CopyConflictError) -> None:
        """Log the copied files and the target that stopped ``copy-all``."""
        copied_count = len(error.copied_paths)
        logger.info("Copied successfully: {} files", copied_count)
        for path in error.copied_paths:
            logger.info(path)
        logger.error("Error: output already exists: {}", error.target)
        logger.error("Use --force for forced overwriting.")
