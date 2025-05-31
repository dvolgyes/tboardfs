"""CLI interface for tboardfs."""

import sys
import click
from pathlib import Path
from loguru import logger

from .parser import TensorBoardParser


def setup_logging(logfile: str | None = None):
    """Configure loguru logger."""
    # Remove default handler
    logger.remove()

    # Add stderr handler with INFO level
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        colorize=True,
    )

    # Add file handler if logfile is specified
    if logfile:
        logger.add(
            logfile,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level="DEBUG",
            rotation="10 MB",
        )
        logger.info(f"Logging to file: {logfile}")


@click.group()
@click.option("--logfile", type=click.Path(), help="Log file path")
@click.pass_context
def main(ctx, logfile: str | None):
    """TensorBoard filesystem interface CLI."""
    setup_logging(logfile)
    # Store in context for access by subcommands
    ctx.ensure_object(dict)
    ctx.obj["logfile"] = logfile


@main.command()
@click.argument("tensorboard_path", type=click.Path(exists=True))
@click.option("--recursive", "-r", is_flag=True, help="List recursively")
@click.option(
    "--digits",
    type=int,
    default=6,
    help="Number of digits for padding iteration numbers (default: 6)",
)
@click.pass_context
def list(ctx, tensorboard_path: str, recursive: bool, digits: int):
    """List contents of TensorBoard log file(s)."""
    path = Path(tensorboard_path)

    # Pass context to parser for progress bar control
    ctx.ensure_object(dict)
    ctx.obj["cli_mode"] = True

    if path.is_file() and "tfevents" in path.name:
        _list_single_file(path, digits, ctx.obj)
    elif path.is_dir():
        if recursive:
            _list_directory_recursive(path, digits, ctx.obj)
        else:
            _list_directory(path)
    else:
        logger.error(
            f"{tensorboard_path} is not a valid TensorBoard log file or directory"
        )
        sys.exit(1)


def _list_single_file(file_path: Path, digits: int = 6, context: dict | None = None):
    """List contents of a single TensorBoard event file."""
    try:
        parser = TensorBoardParser(
            str(file_path),
            show_progress=context.get("cli_mode", False) if context else False,
        )
        content = parser.list_all_content()

        logger.info(f"Contents of {file_path}:")
        logger.info("=" * 60)

        if content["scalars"]:
            logger.info("\nScalars:")
            for tag in content["scalars"]:
                logger.info(f"  - {tag}")

        if content["images"]:
            logger.info("\nImages:")
            for tag in content["images"]:
                image_data = parser.get_image_data(tag)
                logger.info(f"  - {tag} ({len(image_data)} steps)")

        if content["histograms"]:
            logger.info("\nHistograms:")
            for tag in content["histograms"]:
                logger.info(f"  - {tag}")

        if content["tensors"]:
            logger.info("\nTensors:")
            for tag in content["tensors"]:
                logger.info(f"  - {tag}")

        if content["audio"]:
            logger.info("\nAudio:")
            for tag in content["audio"]:
                audio_data = parser.get_audio_data(tag)
                logger.info(f"  - {tag} ({len(audio_data)} steps)")

        if content["text"]:
            logger.info("\nText:")
            for tag in content["text"]:
                text_data = parser.get_text_data(tag)
                logger.info(f"  - {tag} ({len(text_data)} steps)")

        logger.info("\nVirtual filesystem paths:")
        paths = parser.get_virtual_paths(digits=digits)
        for path in paths:
            logger.info(f"  {path}")

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")


def _list_directory(directory: Path):
    """List TensorBoard files in a directory."""
    event_files = list(directory.glob("*.tfevents.*"))
    if not event_files:
        logger.warning(f"No TensorBoard event files found in {directory}")
        return

    logger.info(f"Found {len(event_files)} TensorBoard event file(s) in {directory}:")
    for file in event_files:
        logger.info(f"  - {file.name}")


def _list_directory_recursive(directory: Path, digits: int = 6, context: dict | None = None):
    """List TensorBoard files recursively in a directory."""
    event_files = list(directory.rglob("*.tfevents.*"))
    if not event_files:
        logger.warning(f"No TensorBoard event files found in {directory}")
        return

    for file in sorted(event_files):
        _list_single_file(file, digits, context)


@main.command()
@click.argument("tensorboard_path", type=click.Path(exists=True))
@click.option(
    "-o", "--output", type=click.Path(), required=True, help="Output directory path"
)
@click.option(
    "--no-sort",
    is_flag=True,
    help="Disable sorting of scalar files by iteration number",
)
@click.option(
    "--digits",
    type=int,
    default=6,
    help="Number of digits for padding iteration numbers (default: 6)",
)
@click.pass_context
def extract(ctx, tensorboard_path: str, output: str, no_sort: bool, digits: int):
    """Extract all data from TensorBoard log to directory structure."""
    path = Path(tensorboard_path)

    # Pass context to parser for progress bar control
    ctx.ensure_object(dict)
    ctx.obj["cli_mode"] = True

    if not path.is_file() or "tfevents" not in path.name:
        logger.error(f"{tensorboard_path} is not a valid TensorBoard event file")
        sys.exit(1)

    try:
        parser = TensorBoardParser(str(path), show_progress=True)

        # Extract all data
        sort_scalars = not no_sort
        parser.extract_all_to_directory(
            output, sort_scalars=sort_scalars, digits=digits
        )

        # Report what was extracted
        content = parser.list_all_content()
        logger.success(f"Extracted TensorBoard data to: {output}")

        if content["scalars"]:
            logger.info(f"  - {len(content['scalars'])} scalar tag(s)")

        if content["images"]:
            total_images = sum(
                len(parser.get_image_data(tag)) for tag in content["images"]
            )
            logger.info(
                f"  - {len(content['images'])} image tag(s) ({total_images} total images)"
            )

        if content["histograms"]:
            logger.info(f"  - {len(content['histograms'])} histogram tag(s)")

        if content["tensors"]:
            logger.info(f"  - {len(content['tensors'])} tensor tag(s)")

        if content["audio"]:
            total_audio = sum(
                len(parser.get_audio_data(tag)) for tag in content["audio"]
            )
            logger.info(
                f"  - {len(content['audio'])} audio tag(s) ({total_audio} total audio files)"
            )

        if content["text"]:
            total_text = sum(len(parser.get_text_data(tag)) for tag in content["text"])
            logger.info(
                f"  - {len(content['text'])} text tag(s) ({total_text} total text entries)"
            )

        if content["scalars"] and sort_scalars:
            logger.info("  - Scalar files sorted by iteration number")
        elif content["scalars"] and no_sort:
            logger.info("  - Scalar files not sorted (--no-sort specified)")

    except Exception as e:
        logger.exception(f"Error extracting data: {e}")
        sys.exit(1)


@main.command()
@click.argument("tensorboard_path", type=click.Path(exists=True))
@click.argument("virtual_path", type=str)
@click.option("-o", "--output", type=click.Path(), help="Output file path")
@click.pass_context
def export(ctx, tensorboard_path: str, virtual_path: str, output: str | None):
    """Export a specific item from TensorBoard log."""
    path = Path(tensorboard_path)

    # Pass context to parser for progress bar control
    ctx.ensure_object(dict)
    ctx.obj["cli_mode"] = True

    if not path.is_file() or "tfevents" not in path.name:
        logger.error(f"{tensorboard_path} is not a valid TensorBoard event file")
        sys.exit(1)

    try:
        parser = TensorBoardParser(
            str(path), show_progress=ctx.obj.get("cli_mode", False)
        )

        # Parse the virtual path
        parts = virtual_path.strip("/").split("/")

        if len(parts) < 2:
            logger.error(f"Invalid virtual path format: {virtual_path}")
            sys.exit(1)

        data_type = parts[0]

        if data_type == "scalars":
            # Format: scalars/tag_name.txt
            if not parts[1].endswith(".txt"):
                logger.error(f"Scalar path must end with .txt: {virtual_path}")
                sys.exit(1)

            tag = parts[1][:-4].replace("_", "/")  # Remove .txt and restore slashes
            scalars = parser.list_scalars()

            if tag not in scalars:
                logger.error(f"Scalar tag '{tag}' not found")
                logger.error(f"Available scalar tags: {', '.join(scalars)}")
                sys.exit(1)

            data = parser.export_scalar_to_text(tag)

            if output:
                Path(output).write_text(data)
                logger.success(f"Exported scalar data to {output}")
            else:
                # For stdout output, use print to avoid logger formatting
                print(data)

        elif data_type == "images":
            # Format: images/tag_name/step.ext
            if len(parts) != 3:
                logger.error(
                    f"Image path must be in format 'images/tag/step.ext': {virtual_path}"
                )
                sys.exit(1)

            tag = parts[1].replace("_", "/")
            step_file = parts[2]
            # Remove padding zeros and extract step
            step = int(step_file.split(".")[0])

            image_bytes = parser.export_image(tag, step)

            if image_bytes is None:
                logger.error(f"Image not found for tag '{tag}' at step {step}")
                sys.exit(1)

            if output:
                Path(output).write_bytes(image_bytes)
                logger.success(f"Exported image to {output}")
            else:
                logger.info(
                    f"Image data available ({len(image_bytes)} bytes). Use -o to save to file."
                )

        elif data_type == "histograms":
            # Format: histograms/tag_name.txt
            if not parts[1].endswith(".txt"):
                logger.error(f"Histogram path must end with .txt: {virtual_path}")
                sys.exit(1)

            tag = parts[1][:-4].replace("_", "/")  # Remove .txt and restore slashes
            histograms = parser.list_histograms()

            if tag not in histograms:
                logger.error(f"Histogram tag '{tag}' not found")
                logger.error(f"Available histogram tags: {', '.join(histograms)}")
                sys.exit(1)

            data = parser.export_histogram_to_text(tag)

            if output:
                Path(output).write_text(data)
                logger.success(f"Exported histogram data to {output}")
            else:
                # For stdout output, use print to avoid logger formatting
                print(data)

        elif data_type == "audio":
            # Format: audio/tag_name/step.ext
            if len(parts) != 3:
                logger.error(
                    f"Audio path must be in format 'audio/tag/step.ext': {virtual_path}"
                )
                sys.exit(1)

            tag = parts[1].replace("_", "/")
            step_file = parts[2]
            # Remove padding zeros and extract step
            step = int(step_file.split(".")[0])

            audio_result = parser.export_audio(tag, step)

            if audio_result is None:
                logger.error(f"Audio not found for tag '{tag}' at step {step}")
                sys.exit(1)

            audio_bytes, content_type = audio_result

            if output:
                Path(output).write_bytes(audio_bytes)
                logger.success(f"Exported audio to {output} (type: {content_type})")
            else:
                logger.info(
                    f"Audio data available ({len(audio_bytes)} bytes, type: {content_type}). Use -o to save to file."
                )

        elif data_type == "text":
            # Format: text/tag_name/step.txt
            if len(parts) != 3:
                logger.error(
                    f"Text path must be in format 'text/tag/step.txt': {virtual_path}"
                )
                sys.exit(1)

            tag = parts[1].replace("_", "/")
            step_file = parts[2]
            if not step_file.endswith(".txt"):
                logger.error(f"Text file must end with .txt: {virtual_path}")
                sys.exit(1)

            # Remove padding zeros and extract step
            step = int(step_file[:-4])  # Remove .txt

            text_data = parser.export_text(tag, step)

            if text_data is None:
                logger.error(f"Text not found for tag '{tag}' at step {step}")
                sys.exit(1)

            if output:
                Path(output).write_text(text_data, encoding="utf-8")
                logger.success(f"Exported text to {output}")
            else:
                # For stdout output, use print to avoid logger formatting
                print(text_data)

        else:
            logger.error(f"Unsupported data type: {data_type}")
            logger.error("Supported types: scalars, images, histograms, audio, text")
            sys.exit(1)

    except Exception as e:
        logger.exception(f"Error exporting data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
