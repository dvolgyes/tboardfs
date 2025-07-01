"""Extract command implementation for tboardfs."""

from pathlib import Path
from loguru import logger
from ..core.file_utils import (
    validate_and_exit_on_error,
    create_parser_with_progress,
    get_event_files_sorted,
)
from ..core.reporting import ContentReporter


def extract_tensorboard_data(
    tensorboard_path: str,
    output_dir: str,
    digits: int = 6,
    image_format: str = "jpg",
    image_quality: int = 90,
) -> None:
    """Extract all data from TensorBoard log(s) to directory structure."""
    logger.info(f"Starting extraction from {tensorboard_path} to {output_dir}")
    logger.debug(f"Extract parameters: digits={digits}")

    path = Path(tensorboard_path)

    if path.is_file() and "tfevents" in path.name:
        # Single file extraction
        extract_single_file(
            tensorboard_path,
            output_dir,
            digits,
            image_format,
            image_quality,
        )
    elif path.is_dir():
        # Directory extraction (aggregated)
        extract_directory_aggregated(
            path, output_dir, digits, image_format, image_quality
        )
    else:
        logger.error(
            f"{tensorboard_path} is not a valid TensorBoard log file or directory"
        )
        raise ValueError(f"Invalid TensorBoard path: {tensorboard_path}")


def extract_single_file(
    file_path: str,
    output_dir: str,
    digits: int = 6,
    image_format: str = "jpg",
    image_quality: int = 90,
) -> None:
    """Extract data from a single TensorBoard event file."""
    # Validate input file
    logger.debug("Validating input TensorBoard file")
    validated_path = validate_and_exit_on_error(file_path)
    logger.debug(f"Input file validated: {validated_path}")

    # Create parser with progress display
    logger.debug("Creating TensorBoard parser")
    parser = create_parser_with_progress(str(validated_path), show_progress=True)
    logger.debug("Parser created successfully")

    # Extract all data
    logger.info("Beginning data extraction")
    parser.extract_all_to_directory(
        output_dir,
        digits=digits,
        image_format=image_format,
        image_quality=image_quality,
    )
    logger.info("Data extraction completed")

    # Display summary of what was extracted
    logger.debug("Generating extraction summary")
    reporter = ContentReporter(parser)
    reporter.display_extraction_summary(output_dir)
    reporter.display_sorting_info(True, False)  # ScalarFile always sorts
    logger.debug("Extraction summary displayed")


def extract_directory_aggregated(
    directory: Path,
    output_dir: str,
    digits: int = 6,
    image_format: str = "jpg",
    image_quality: int = 90,
) -> None:
    """Extract data from all TensorBoard event files in a directory with aggregated naming."""
    logger.info(f"Starting aggregated extraction from directory {directory}")

    # Find all event files
    event_files = get_event_files_sorted(directory)
    if not event_files:
        return

    logger.info(f"Found {len(event_files)} event files to process")

    # Setup extraction environment
    output_path = _setup_extraction_directory(output_dir)

    # Process all files and track results
    total_extracted = _process_all_event_files(
        directory,
        event_files,
        output_path,
        digits,
        image_format,
        image_quality,
    )

    # Display results
    _display_extraction_summary(output_dir, total_extracted)


def _setup_extraction_directory(output_dir: str) -> Path:
    """Create and return the output directory path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _process_all_event_files(
    directory: Path,
    event_files: list[Path],
    output_path: Path,
    digits: int,
    image_format: str,
    image_quality: int,
) -> dict[str, int]:
    """Process all event files and return extraction statistics."""
    total_extracted = {
        "scalars": 0,
        "images": 0,
        "histograms": 0,
        "audio": 0,
        "text": 0,
    }

    for file_path in event_files:
        try:
            logger.info(f"Processing {file_path.relative_to(directory)}")

            file_stats = _extract_single_file_with_context(
                file_path,
                directory,
                output_path,
                digits,
                image_format,
                image_quality,
            )

            # Aggregate statistics
            for data_type, count in file_stats.items():
                if data_type in total_extracted:
                    total_extracted[data_type] += count

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    return total_extracted


def _extract_single_file_with_context(
    file_path: Path,
    directory: Path,
    output_path: Path,
    digits: int,
    image_format: str,
    image_quality: int,
) -> dict[str, int]:
    """Extract data from a single file with directory context."""
    # Create parser
    parser = create_parser_with_progress(str(file_path), show_progress=True)

    # Get directory context for naming
    relative_path = file_path.relative_to(directory)
    parent_dir = relative_path.parent

    # Create context-specific output directory
    extract_output = _get_context_output_directory(parent_dir, output_path)

    # Extract data
    parser.extract_all_to_directory(
        extract_output,
        digits=digits,
        image_format=image_format,
        image_quality=image_quality,
    )

    # Count extracted items
    content = parser.list_all_content()
    return {data_type: len(tags) for data_type, tags in content.items()}


def _get_context_output_directory(parent_dir: Path, output_path: Path) -> str:
    """Get the appropriate output directory based on context."""
    if parent_dir.name and parent_dir.name != ".":
        # Extract to subdirectory with contextual naming
        context_output_dir = output_path / parent_dir.name
        context_output_dir.mkdir(parents=True, exist_ok=True)
        return str(context_output_dir)
    else:
        # Extract to main output directory
        return str(output_path)


def _display_extraction_summary(
    output_dir: str, total_extracted: dict[str, int]
) -> None:
    """Display the final extraction summary."""
    logger.success(f"Aggregated extraction completed to: {output_dir}")
    logger.info("Aggregated extraction summary:")
    for data_type, count in total_extracted.items():
        if count > 0:
            logger.info(f"  - {count} {data_type} tag(s)")
