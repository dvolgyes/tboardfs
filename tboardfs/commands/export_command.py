"""Export command implementation for tboardfs."""

from ..core.file_utils import validate_and_exit_on_error, create_parser_with_progress
from ..core.virtual_paths import VirtualPathHandler


def export_virtual_path(
    tensorboard_path: str,
    virtual_path: str,
    output_file: str | None = None,
    show_progress: bool = False,
) -> None:
    """Export a specific item from TensorBoard log using virtual path."""
    # Validate input file
    file_path = validate_and_exit_on_error(tensorboard_path)

    # Create parser
    parser = create_parser_with_progress(str(file_path), show_progress=show_progress)

    # Create virtual path handler and process export
    handler = VirtualPathHandler(parser)
    path_info = handler.parse_virtual_path(virtual_path)
    handler.export_data(path_info, output_file)
