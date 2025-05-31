"""Extract command implementation for tboardfs."""

from ..core.file_utils import validate_and_exit_on_error, create_parser_with_progress
from ..core.reporting import ContentReporter


def extract_tensorboard_data(
    tensorboard_path: str, output_dir: str, sort_scalars: bool = True, digits: int = 6
) -> None:
    """Extract all data from TensorBoard log to directory structure."""
    # Validate input file
    file_path = validate_and_exit_on_error(tensorboard_path)

    # Create parser with progress display
    parser = create_parser_with_progress(str(file_path), show_progress=True)

    # Extract all data
    parser.extract_all_to_directory(
        output_dir, sort_scalars=sort_scalars, digits=digits
    )

    # Display summary of what was extracted
    reporter = ContentReporter(parser)
    reporter.display_extraction_summary(output_dir)
    reporter.display_sorting_info(sort_scalars, not sort_scalars)
