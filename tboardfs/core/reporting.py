"""Content reporting and display utilities for tboardfs."""

from pathlib import Path

from loguru import logger

from ..parser import TensorBoardParser


class ContentReporter:
    """Handles content summary and display functionality."""

    def __init__(self, parser: TensorBoardParser):
        self.parser = parser

    def display_file_header(self, file_path: Path) -> None:
        """Display header information for a file."""
        logger.info(f"Contents of {file_path}:")
        logger.info("=" * 60)

    def display_content_by_type(self, content: dict[str, list[str]]) -> None:
        """Display content organized by type with counts."""
        if content["scalars"]:
            logger.info("\nScalars:")
            for tag in content["scalars"]:
                logger.info(f"  - {tag}")

        if content["images"]:
            logger.info("\nImages:")
            for tag in content["images"]:
                image_data = self.parser.get_image_data(tag)
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
                audio_data = self.parser.get_audio_data(tag)
                logger.info(f"  - {tag} ({len(audio_data)} steps)")

        if content["text"]:
            logger.info("\nText:")
            for tag in content["text"]:
                text_data = self.parser.get_text_data(tag)
                logger.info(f"  - {tag} ({len(text_data)} steps)")

    def display_virtual_paths(self, digits: int = 6) -> None:
        """Display virtual filesystem paths."""
        logger.info("\nVirtual filesystem paths:")
        paths = self.parser.get_virtual_paths(digits=digits)
        for path in paths:
            logger.info(f"  {path}")

    def display_extraction_summary(self, output_dir: str) -> None:
        """Display summary of extracted content."""
        content = self.parser.list_all_content()
        logger.success(f"Extracted TensorBoard data to: {output_dir}")

        if content["scalars"]:
            logger.info(f"  - {len(content['scalars'])} scalar tag(s)")

        if content["images"]:
            total_images = sum(
                len(self.parser.get_image_data(tag)) for tag in content["images"]
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
                len(self.parser.get_audio_data(tag)) for tag in content["audio"]
            )
            logger.info(
                f"  - {len(content['audio'])} audio tag(s) ({total_audio} total audio files)"
            )

        if content["text"]:
            total_text = sum(
                len(self.parser.get_text_data(tag)) for tag in content["text"]
            )
            logger.info(
                f"  - {len(content['text'])} text tag(s) ({total_text} total text entries)"
            )

    def display_sorting_info(self, sort_scalars: bool, no_sort: bool) -> None:
        """Display scalar sorting information."""
        content = self.parser.list_all_content()
        if content["scalars"] and sort_scalars:
            logger.info("  - Scalar files sorted by iteration number")
        elif content["scalars"] and no_sort:
            logger.info("  - Scalar files not sorted (--no-sort specified)")


def list_directory_files(directory: Path) -> None:
    """List TensorBoard files in a directory."""
    event_files = list(directory.glob("*.tfevents.*"))
    if not event_files:
        logger.warning(f"No TensorBoard event files found in {directory}")
        return

    logger.info(f"Found {len(event_files)} TensorBoard event file(s) in {directory}:")
    for file in event_files:
        logger.info(f"  - {file.name}")
