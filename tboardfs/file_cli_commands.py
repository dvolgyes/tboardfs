from pathlib import Path
import sys

import click

from tboardfs.file_tree import SingleEventTree


def register_file_cli_commands(group: click.Group) -> None:
    """Register single-event file commands on a Click group."""

    @group.command("list")
    @click.argument("source", type=click.Path(exists=True, dir_okay=False))
    @click.argument("prefix", required=False, default="/")
    @click.option("--step-digits", type=int, default=6, show_default=True)
    @click.option("--scalar-format", default="json,tsv,npz", show_default=True)
    def list_files(
        source: str, prefix: str, step_digits: int, scalar_format: str
    ) -> None:
        """Print virtual file paths below PREFIX."""
        tree = SingleEventTree(
            source, step_digits=step_digits, scalar_format=scalar_format
        )
        for path in tree.list_file_paths(prefix=prefix):
            sys.stdout.write(path + "\n")

    @group.command("get")
    @click.argument("source", type=click.Path(exists=True, dir_okay=False))
    @click.argument("vpath")
    @click.option(
        "-o",
        "--output",
        required=True,
        type=click.Path(dir_okay=False, writable=True),
    )
    @click.option("--force", is_flag=True)
    @click.option("--step-digits", type=int, default=6, show_default=True)
    @click.option("--scalar-format", default="json,tsv,npz", show_default=True)
    def get_file(
        source: str,
        vpath: str,
        output: str,
        force: bool,
        step_digits: int,
        scalar_format: str,
    ) -> None:
        """Write one virtual file to OUTPUT."""
        tree = SingleEventTree(
            source, step_digits=step_digits, scalar_format=scalar_format
        )
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
        output_path = Path(output)
        if output_path.exists() and not force:
            raise click.ClickException(f"output already exists: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)

    @group.command("copy-all")
    @click.argument("source", type=click.Path(exists=True, dir_okay=False))
    @click.argument("outdir", type=click.Path(file_okay=False))
    @click.option("--force", is_flag=True)
    @click.option("--step-digits", type=int, default=6, show_default=True)
    @click.option("--scalar-format", default="json,tsv,npz", show_default=True)
    def copy_all(
        source: str,
        outdir: str,
        force: bool,
        step_digits: int,
        scalar_format: str,
    ) -> None:
        """Copy every virtual file into OUTDIR."""
        tree = SingleEventTree(
            source, step_digits=step_digits, scalar_format=scalar_format
        )
        try:
            tree.copy_all(outdir, force=force)
        except FileExistsError as error:
            raise click.ClickException(f"output already exists: {error}") from error
