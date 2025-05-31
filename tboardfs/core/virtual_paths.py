"""Virtual path parsing and export handling for tboardfs."""

import sys
from pathlib import Path
from dataclasses import dataclass

from loguru import logger

from ..parser import TensorBoardParser
from .file_utils import restore_tag_from_path, extract_step_from_filename


@dataclass
class VirtualPathInfo:
    """Information parsed from a virtual path."""

    data_type: str
    tag: str
    step: int | None = None
    extension: str | None = None


class VirtualPathHandler:
    """Handles virtual path parsing and data export operations."""

    def __init__(self, parser: TensorBoardParser):
        self.parser = parser

    def parse_virtual_path(self, virtual_path: str) -> VirtualPathInfo:
        """Parse virtual path into components."""
        parts = virtual_path.strip("/").split("/")

        if len(parts) < 2:
            logger.error(f"Invalid virtual path format: {virtual_path}")
            sys.exit(1)

        data_type = parts[0]

        if data_type == "scalars":
            return self._parse_scalar_path(parts, virtual_path)
        elif data_type == "images":
            return self._parse_image_path(parts, virtual_path)
        elif data_type == "histograms":
            return self._parse_histogram_path(parts, virtual_path)
        elif data_type == "audio":
            return self._parse_audio_path(parts, virtual_path)
        elif data_type == "text":
            return self._parse_text_path(parts, virtual_path)
        else:
            logger.error(f"Unsupported data type: {data_type}")
            logger.error("Supported types: scalars, images, histograms, audio, text")
            sys.exit(1)

    def _parse_scalar_path(self, parts: list, virtual_path: str) -> VirtualPathInfo:
        """Parse scalar path: scalars/tag_name.txt"""
        if not parts[1].endswith(".txt"):
            logger.error(f"Scalar path must end with .txt: {virtual_path}")
            sys.exit(1)

        tag = restore_tag_from_path(parts[1][:-4])  # Remove .txt
        return VirtualPathInfo(data_type="scalars", tag=tag, extension="txt")

    def _parse_image_path(self, parts: list, virtual_path: str) -> VirtualPathInfo:
        """Parse image path: images/tag_name/step.ext"""
        if len(parts) != 3:
            logger.error(
                f"Image path must be in format 'images/tag/step.ext': {virtual_path}"
            )
            sys.exit(1)

        tag = restore_tag_from_path(parts[1])
        step_file = parts[2]
        step = extract_step_from_filename(step_file)
        extension = step_file.split(".")[-1]

        return VirtualPathInfo(
            data_type="images", tag=tag, step=step, extension=extension
        )

    def _parse_histogram_path(self, parts: list, virtual_path: str) -> VirtualPathInfo:
        """Parse histogram path: histograms/tag_name.txt"""
        if not parts[1].endswith(".txt"):
            logger.error(f"Histogram path must end with .txt: {virtual_path}")
            sys.exit(1)

        tag = restore_tag_from_path(parts[1][:-4])  # Remove .txt
        return VirtualPathInfo(data_type="histograms", tag=tag, extension="txt")

    def _parse_audio_path(self, parts: list, virtual_path: str) -> VirtualPathInfo:
        """Parse audio path: audio/tag_name/step.ext"""
        if len(parts) != 3:
            logger.error(
                f"Audio path must be in format 'audio/tag/step.ext': {virtual_path}"
            )
            sys.exit(1)

        tag = restore_tag_from_path(parts[1])
        step_file = parts[2]
        step = extract_step_from_filename(step_file)
        extension = step_file.split(".")[-1]

        return VirtualPathInfo(
            data_type="audio", tag=tag, step=step, extension=extension
        )

    def _parse_text_path(self, parts: list, virtual_path: str) -> VirtualPathInfo:
        """Parse text path: text/tag_name/step.txt"""
        if len(parts) != 3:
            logger.error(
                f"Text path must be in format 'text/tag/step.txt': {virtual_path}"
            )
            sys.exit(1)

        tag = restore_tag_from_path(parts[1])
        step_file = parts[2]

        if not step_file.endswith(".txt"):
            logger.error(f"Text file must end with .txt: {virtual_path}")
            sys.exit(1)

        step = int(step_file[:-4])  # Remove .txt and get step
        return VirtualPathInfo(data_type="text", tag=tag, step=step, extension="txt")

    def export_data(
        self, path_info: VirtualPathInfo, output_file: str | None = None
    ) -> None:
        """Export data based on parsed virtual path information."""
        if path_info.data_type == "scalars":
            self._export_scalar_data(path_info, output_file)
        elif path_info.data_type == "images":
            self._export_image_data(path_info, output_file)
        elif path_info.data_type == "histograms":
            self._export_histogram_data(path_info, output_file)
        elif path_info.data_type == "audio":
            self._export_audio_data(path_info, output_file)
        elif path_info.data_type == "text":
            self._export_text_data(path_info, output_file)

    def _export_scalar_data(
        self, path_info: VirtualPathInfo, output_file: str | None
    ) -> None:
        """Export scalar data to text format."""
        scalars = self.parser.list_scalars()

        if path_info.tag not in scalars:
            logger.error(f"Scalar tag '{path_info.tag}' not found")
            logger.error(f"Available scalar tags: {', '.join(scalars)}")
            sys.exit(1)

        data = self.parser.export_scalar_to_text(path_info.tag)
        self._handle_output(data, output_file, "scalar data")

    def _export_image_data(
        self, path_info: VirtualPathInfo, output_file: str | None
    ) -> None:
        """Export image data."""
        if path_info.step is None:
            logger.error("Step is required for image export")
            sys.exit(1)
        image_bytes = self.parser.export_image(path_info.tag, path_info.step)

        if image_bytes is None:
            logger.error(
                f"Image not found for tag '{path_info.tag}' at step {path_info.step}"
            )
            sys.exit(1)

        if output_file:
            Path(output_file).write_bytes(image_bytes)
            logger.success(f"Exported image to {output_file}")
        else:
            logger.info(
                f"Image data available ({len(image_bytes)} bytes). Use -o to save to file."
            )

    def _export_histogram_data(
        self, path_info: VirtualPathInfo, output_file: str | None
    ) -> None:
        """Export histogram data to text format."""
        histograms = self.parser.list_histograms()

        if path_info.tag not in histograms:
            logger.error(f"Histogram tag '{path_info.tag}' not found")
            logger.error(f"Available histogram tags: {', '.join(histograms)}")
            sys.exit(1)

        data = self.parser.export_histogram_to_text(path_info.tag)
        self._handle_output(data, output_file, "histogram data")

    def _export_audio_data(
        self, path_info: VirtualPathInfo, output_file: str | None
    ) -> None:
        """Export audio data."""
        if path_info.step is None:
            logger.error("Step is required for audio export")
            sys.exit(1)
        audio_result = self.parser.export_audio(path_info.tag, path_info.step)

        if audio_result is None:
            logger.error(
                f"Audio not found for tag '{path_info.tag}' at step {path_info.step}"
            )
            sys.exit(1)

        audio_bytes, content_type = audio_result

        if output_file:
            Path(output_file).write_bytes(audio_bytes)
            logger.success(f"Exported audio to {output_file} (type: {content_type})")
        else:
            logger.info(
                f"Audio data available ({len(audio_bytes)} bytes, type: {content_type}). Use -o to save to file."
            )

    def _export_text_data(
        self, path_info: VirtualPathInfo, output_file: str | None
    ) -> None:
        """Export text data."""
        if path_info.step is None:
            logger.error("Step is required for text export")
            sys.exit(1)
        text_data = self.parser.export_text(path_info.tag, path_info.step)

        if text_data is None:
            logger.error(
                f"Text not found for tag '{path_info.tag}' at step {path_info.step}"
            )
            sys.exit(1)

        if output_file:
            Path(output_file).write_text(text_data, encoding="utf-8")
            logger.success(f"Exported text to {output_file}")
        else:
            # For stdout output, use print to avoid logger formatting
            print(text_data)

    def _handle_output(
        self, data: str, output_file: str | None, data_type: str
    ) -> None:
        """Handle text data output to file or stdout."""
        if output_file:
            Path(output_file).write_text(data)
            logger.success(f"Exported {data_type} to {output_file}")
        else:
            # For stdout output, use print to avoid logger formatting
            print(data)
