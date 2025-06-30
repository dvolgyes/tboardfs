"""Export command implementation for tboardfs."""

from pathlib import Path
from loguru import logger
from ..core.file_utils import validate_and_exit_on_error, create_parser_with_progress
from ..core.virtual_paths import VirtualPathHandler


def export_virtual_path(
    tensorboard_path: str,
    virtual_path: str,
    output_file: str | None = None,
    show_progress: bool = False,
    image_format: str = "jpg",
    image_quality: int = 90,
) -> None:
    """Export a specific item from TensorBoard log(s) using virtual path."""
    path = Path(tensorboard_path)

    if path.is_file() and "tfevents" in path.name:
        # Single file export
        export_from_single_file(
            tensorboard_path,
            virtual_path,
            output_file,
            show_progress,
            image_format,
            image_quality,
        )
    elif path.is_dir():
        # Directory export (search for matching virtual path)
        export_from_directory(
            path, virtual_path, output_file, show_progress, image_format, image_quality
        )
    else:
        logger.error(
            f"{tensorboard_path} is not a valid TensorBoard log file or directory"
        )
        raise ValueError(f"Invalid TensorBoard path: {tensorboard_path}")


def export_from_single_file(
    file_path: str,
    virtual_path: str,
    output_file: str | None = None,
    show_progress: bool = False,
    image_format: str = "jpg",
    image_quality: int = 90,
) -> None:
    """Export from a single TensorBoard event file."""
    # Validate input file
    validated_path = validate_and_exit_on_error(file_path)

    # Create parser
    parser = create_parser_with_progress(
        str(validated_path), show_progress=show_progress
    )

    # Create virtual path handler and process export
    handler = VirtualPathHandler(parser)
    path_info = handler.parse_virtual_path(virtual_path)
    handler.export_data(path_info, output_file, image_format, image_quality)


def export_from_directory(
    directory: Path,
    virtual_path: str,
    output_file: str | None = None,
    show_progress: bool = False,
    image_format: str = "jpg",
    image_quality: int = 90,
) -> None:
    """Export from TensorBoard directory by finding the correct event file."""
    logger.info(f"Searching for '{virtual_path}' in directory {directory}")

    # Find all event files
    event_files = list(directory.rglob("*.tfevents.*"))
    if not event_files:
        logger.error(f"No TensorBoard event files found in {directory}")
        raise FileNotFoundError(f"No event files in {directory}")

    # Check if virtual_path looks like a contextual path (e.g., train_F1_cls_00)
    # If so, try to map it to the correct subdirectory
    target_file = None

    # Look for class-specific metrics in subdirectories
    for subdir_name in [d.name for d in directory.iterdir() if d.is_dir()]:
        if virtual_path.startswith(f"scalars/{subdir_name}"):
            # Found matching subdirectory for scalar
            subdir_event_files = list((directory / subdir_name).glob("*.tfevents.*"))
            if subdir_event_files:
                target_file = subdir_event_files[0]
                # Adjust virtual path for the subdirectory context
                virtual_path = virtual_path.replace(
                    f"scalars/{subdir_name}", "scalars/train_F1"
                )  # Map to generic tag
                break

    # If no specific subdirectory match, search through all files
    if target_file is None:
        logger.info(
            f"Searching through {len(event_files)} event files for '{virtual_path}'"
        )

        for file_path in sorted(event_files):
            try:
                parser = create_parser_with_progress(
                    str(file_path), show_progress=False
                )
                handler = VirtualPathHandler(parser)

                # Try to parse the virtual path for this file
                try:
                    handler.parse_virtual_path(virtual_path)
                    # If parsing succeeds, this file contains the requested data
                    target_file = file_path
                    logger.info(
                        f"Found '{virtual_path}' in {file_path.relative_to(directory)}"
                    )
                    break
                except Exception:
                    # Path not found in this file, continue searching
                    continue

            except Exception as e:
                logger.debug(f"Error checking {file_path}: {e}")
                continue

    if target_file is None:
        logger.error(f"Virtual path '{virtual_path}' not found in any event files")
        raise FileNotFoundError(f"Virtual path '{virtual_path}' not found")

    # Export from the target file
    logger.info(f"Exporting from {target_file.relative_to(directory)}")
    export_from_single_file(
        str(target_file),
        virtual_path,
        output_file,
        show_progress,
        image_format,
        image_quality,
    )
