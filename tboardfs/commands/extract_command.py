"""Extract command implementation for tboardfs."""

from loguru import logger
from ..core.file_utils import validate_and_exit_on_error, create_parser_with_progress
from ..core.reporting import ContentReporter


def extract_tensorboard_data(
    tensorboard_path: str, output_dir: str, sort_scalars: bool = True, digits: int = 6
) -> None:
    """Extract all data from TensorBoard log to directory structure."""
    logger.info(f"Starting extraction from {tensorboard_path} to {output_dir}")
    logger.debug(f"Extract parameters: sort_scalars={sort_scalars}, digits={digits}")

    # Validate input file
    logger.debug("Validating input TensorBoard file")
    file_path = validate_and_exit_on_error(tensorboard_path)
    logger.debug(f"Input file validated: {file_path}")

    # Create parser with progress display
    logger.debug("Creating TensorBoard parser")
    parser = create_parser_with_progress(str(file_path), show_progress=True)
    logger.debug("Parser created successfully")

    # Extract all data
    logger.info("Beginning data extraction")
    parser.extract_all_to_directory(
        output_dir, sort_scalars=sort_scalars, digits=digits
    )
    logger.info("Data extraction completed")

    # Display summary of what was extracted
    logger.debug("Generating extraction summary")
    reporter = ContentReporter(parser)
    reporter.display_extraction_summary(output_dir)
    reporter.display_sorting_info(sort_scalars, not sort_scalars)
    logger.debug("Extraction summary displayed")
