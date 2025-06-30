"""Extract command implementation for tboardfs."""

from pathlib import Path
from loguru import logger
from ..core.file_utils import validate_and_exit_on_error, create_parser_with_progress
from ..core.reporting import ContentReporter


def extract_tensorboard_data(
    tensorboard_path: str,
    output_dir: str,
    sort_scalars: bool = True,
    digits: int = 6,
    image_format: str = "jpg",
    image_quality: int = 90,
) -> None:
    """Extract all data from TensorBoard log(s) to directory structure."""
    logger.info(f"Starting extraction from {tensorboard_path} to {output_dir}")
    logger.debug(f"Extract parameters: sort_scalars={sort_scalars}, digits={digits}")

    path = Path(tensorboard_path)

    if path.is_file() and "tfevents" in path.name:
        # Single file extraction
        extract_single_file(
            tensorboard_path,
            output_dir,
            sort_scalars,
            digits,
            image_format,
            image_quality,
        )
    elif path.is_dir():
        # Directory extraction (aggregated)
        extract_directory_aggregated(
            path, output_dir, sort_scalars, digits, image_format, image_quality
        )
    else:
        logger.error(
            f"{tensorboard_path} is not a valid TensorBoard log file or directory"
        )
        raise ValueError(f"Invalid TensorBoard path: {tensorboard_path}")


def extract_single_file(
    file_path: str,
    output_dir: str,
    sort_scalars: bool = True,
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
        sort_scalars=sort_scalars,
        digits=digits,
        image_format=image_format,
        image_quality=image_quality,
    )
    logger.info("Data extraction completed")

    # Display summary of what was extracted
    logger.debug("Generating extraction summary")
    reporter = ContentReporter(parser)
    reporter.display_extraction_summary(output_dir)
    reporter.display_sorting_info(sort_scalars, not sort_scalars)
    logger.debug("Extraction summary displayed")


def extract_directory_aggregated(
    directory: Path,
    output_dir: str,
    sort_scalars: bool = True,
    digits: int = 6,
    image_format: str = "jpg",
    image_quality: int = 90,
) -> None:
    """Extract data from all TensorBoard event files in a directory with aggregated naming."""
    logger.info(f"Starting aggregated extraction from directory {directory}")

    # Find all event files
    event_files = list(directory.rglob("*.tfevents.*"))
    if not event_files:
        logger.warning(f"No TensorBoard event files found in {directory}")
        return

    logger.info(f"Found {len(event_files)} event files to process")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_extracted = {
        "scalars": 0,
        "images": 0,
        "histograms": 0,
        "audio": 0,
        "text": 0,
    }

    for file_path in sorted(event_files):
        try:
            logger.info(f"Processing {file_path.relative_to(directory)}")

            # Create parser
            parser = create_parser_with_progress(str(file_path), show_progress=True)

            # Get directory context for naming
            relative_path = file_path.relative_to(directory)
            parent_dir = relative_path.parent

            # Create context-specific output directory
            if parent_dir.name and parent_dir.name != ".":
                # Extract to subdirectory with contextual naming
                context_output_dir = output_path / parent_dir.name
                context_output_dir.mkdir(parents=True, exist_ok=True)
                extract_output = str(context_output_dir)
            else:
                # Extract to main output directory
                extract_output = output_dir

            # Extract data
            parser.extract_all_to_directory(
                extract_output,
                sort_scalars=sort_scalars,
                digits=digits,
                image_format=image_format,
                image_quality=image_quality,
            )

            # Track what was extracted
            content = parser.list_all_content()
            for data_type, tags in content.items():
                if data_type in total_extracted:
                    total_extracted[data_type] += len(tags)

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    # Display aggregated summary
    logger.success(f"Aggregated extraction completed to: {output_dir}")
    logger.info("Aggregated extraction summary:")
    for data_type, count in total_extracted.items():
        if count > 0:
            logger.info(f"  - {count} {data_type} tag(s)")

    if sort_scalars:
        logger.info("  - Scalar files sorted by iteration number")
    logger.info("Data organized by context (subdirectories represent metric groups)")
