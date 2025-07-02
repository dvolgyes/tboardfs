"""Content reporting and display utilities for tboardfs."""

from pathlib import Path

from loguru import logger

from ..efficient_parser import EfficientTensorBoardParser


class ContentReporter:
    """Handles content summary and display functionality."""

    def __init__(self, parser: EfficientTensorBoardParser | None):
        self.parser = parser

    def display_file_header(self, file_path: Path) -> None:
        """Display header information for a file."""
        logger.info(f"Contents of {file_path}:")
        logger.info("=" * 60)

    def display_content_by_type(
        self, content: dict[str, list[str]], show_step_counts: bool = True
    ) -> None:
        """Display content organized by type with optional step counts."""
        if content["scalars"]:
            logger.info("\nScalars:")
            for tag in content["scalars"]:
                logger.info(f"  - {tag}")

        if content["images"]:
            logger.info("\nImages:")
            for tag in content["images"]:
                if show_step_counts and self.parser:
                    image_data = list(self.parser.iterate_image_data(tag))
                    logger.info(f"  - {tag} ({len(image_data)} steps)")
                else:
                    logger.info(f"  - {tag}")

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
                if show_step_counts and self.parser:
                    audio_data = list(self.parser.iterate_audio_data(tag))
                    logger.info(f"  - {tag} ({len(audio_data)} steps)")
                else:
                    logger.info(f"  - {tag}")

        if content["text"]:
            logger.info("\nText:")
            for tag in content["text"]:
                if show_step_counts and self.parser:
                    text_data = list(self.parser.iterate_text_data(tag))
                    logger.info(f"  - {tag} ({len(text_data)} steps)")
                else:
                    logger.info(f"  - {tag}")

    def display_virtual_paths(self, digits: int = 6) -> None:
        """Display virtual filesystem paths."""
        if self.parser is None:
            logger.error("No parser available for virtual paths display")
            return
        logger.info("\nVirtual filesystem paths:")
        paths = self.parser.get_virtual_paths(digits=digits)
        for path in paths:
            logger.info(f"  {path}")

    def display_extraction_summary(self, output_dir: str) -> None:
        """Display summary of extracted content."""
        if self.parser is None:
            logger.error("No parser available for extraction summary")
            return
        content = self.parser.list_all_content()
        logger.success(f"Extracted TensorBoard data to: {output_dir}")

        # Help mypy understand parser is not None
        assert self.parser is not None

        if content["scalars"]:
            logger.info(f"  - {len(content['scalars'])} scalar tag(s)")

        if content["images"]:
            total_images = sum(
                len(list(self.parser.iterate_image_data(tag)))
                for tag in content["images"]
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
                len(list(self.parser.iterate_audio_data(tag)))
                for tag in content["audio"]
            )
            logger.info(
                f"  - {len(content['audio'])} audio tag(s) ({total_audio} total audio files)"
            )

        if content["text"]:
            total_text = sum(
                len(list(self.parser.iterate_text_data(tag))) for tag in content["text"]
            )
            logger.info(
                f"  - {len(content['text'])} text tag(s) ({total_text} total text entries)"
            )

    def display_sorting_info(self, sort_scalars: bool, no_sort: bool) -> None:
        """Display scalar sorting information."""
        if self.parser is None:
            return
        content = self.parser.list_all_content()
        if content["scalars"] and sort_scalars:
            logger.info("  - Scalar files sorted by iteration number")
        elif content["scalars"] and no_sort:
            logger.info("  - Scalar files not sorted (--no-sort specified)")

    @staticmethod
    def display_content_by_type_static(content: dict[str, list[str]]) -> None:
        """Static method to display content organized by type without step counts."""
        # Create a dummy ContentReporter for the static method
        dummy_reporter = ContentReporter(None)
        dummy_reporter.display_content_by_type(content, show_step_counts=False)


def list_directory_files(directory: Path) -> None:
    """List TensorBoard files in a directory."""
    event_files = list(directory.glob("*.tfevents.*"))
    if not event_files:
        logger.warning(f"No TensorBoard event files found in {directory}")
        return

    logger.info(f"Found {len(event_files)} TensorBoard event file(s) in {directory}:")
    for file in event_files:
        logger.info(f"  - {file.name}")
