"""Virtual path parsing and export handling for tboardfs.

This module provides backward compatibility for the original virtual path system
while migrating to the unified virtual path system.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from loguru import logger

from ..efficient_parser import EfficientTensorBoardParser
from .unified_virtual_paths import (
    VirtualPathParser,
    VirtualPathInfo as UnifiedVirtualPathInfo,
)


@dataclass
class VirtualPathInfo:
    """Information parsed from a virtual path (legacy compatibility)."""

    data_type: str
    tag: str
    step: int | None = None
    extension: str | None = None

    @classmethod
    def from_unified(cls, unified_info: UnifiedVirtualPathInfo) -> "VirtualPathInfo":
        """Create legacy VirtualPathInfo from unified version."""
        return cls(
            data_type=unified_info.data_type.value,
            tag=unified_info.tag,
            step=unified_info.step,
            extension=unified_info.extension,
        )


class VirtualPathHandler:
    """Handles virtual path parsing and data export operations."""

    def __init__(self, parser: EfficientTensorBoardParser):
        self.parser = parser
        self.unified_parser = VirtualPathParser()

    def parse_virtual_path(self, virtual_path: str) -> VirtualPathInfo:
        """Parse virtual path into components using unified system."""
        try:
            unified_info = self.unified_parser.parse(virtual_path)
            return VirtualPathInfo.from_unified(unified_info)
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)

    # Legacy parsing methods removed - now handled by unified system

    def _is_data_type_enabled(
        self, data_type: str, type_filters: dict[str, set[str]] | None
    ) -> bool:
        """Check if a data type should be processed based on filters."""
        if not type_filters:
            return True

        # Use the unified data type mapping from VirtualPathConfig
        from .unified_virtual_paths import VirtualPathConfig

        # Map directory names to DataType values
        data_type_obj = VirtualPathConfig.from_directory_name(data_type)
        filter_type = data_type_obj.value if data_type_obj else data_type

        ignore_types = type_filters.get("ignore", set())
        select_types = type_filters.get("select", set())

        if select_types:
            return filter_type in select_types
        elif ignore_types:
            return filter_type not in ignore_types
        else:
            return True

    def export_data(
        self,
        path_info: VirtualPathInfo,
        output_file: str | None = None,
        image_format: str = "jpg",
        image_quality: int = 90,
        audio_format: str = "mp3",
        histogram_images: bool = False,
        ply_format: str = "binary",
        type_filters: dict[str, set[str]] | None = None,
    ) -> None:
        """Export data based on parsed virtual path information."""
        # Check if this data type is enabled for processing
        if not self._is_data_type_enabled(path_info.data_type, type_filters):
            logger.warning(
                f"Data type '{path_info.data_type}' is filtered out, skipping export"
            )
            return

        if path_info.data_type == "scalars":
            self._export_scalar_data(path_info, output_file)
        elif path_info.data_type == "images":
            self._export_image_data(path_info, output_file, image_format, image_quality)
        elif path_info.data_type == "histograms":
            self._export_histogram_data(path_info, output_file, histogram_images)
        elif path_info.data_type == "audio":
            self._export_audio_data(path_info, output_file, audio_format)
        elif path_info.data_type == "text":
            self._export_text_data(path_info, output_file)
        elif path_info.data_type == "meshes":
            self._export_mesh_data(path_info, output_file, ply_format)
        elif path_info.data_type == "hp_params":
            self._export_hyperparameter_data(path_info, output_file)

    def _export_scalar_data(
        self, path_info: VirtualPathInfo, output_file: str | None
    ) -> None:
        """Export scalar data to text format."""
        scalars = self.parser.list_scalars()

        if path_info.tag not in scalars:
            logger.error(f"Scalar tag '{path_info.tag}' not found")
            logger.error(f"Available scalar tags: {', '.join(scalars)}")
            sys.exit(1)

        lines = []
        for scalar_data in self.parser.iterate_scalar_data(path_info.tag):
            lines.append(f"{scalar_data.step}\t{scalar_data.value}")
        data = "\n".join(lines)
        self._handle_output(data, output_file, "scalar data")

    def _export_image_data(
        self,
        path_info: VirtualPathInfo,
        output_file: str | None,
        image_format: str,
        image_quality: int,
    ) -> None:
        """Export image data."""
        if path_info.step is None:
            logger.error("Step is required for image export")
            sys.exit(1)
        # Find the image data for the specific step
        image_bytes = None
        for data in self.parser.iterate_image_data(path_info.tag):
            if data.step == path_info.step:
                image_bytes = data.encoded_image_string
                break

        if image_bytes is None:
            logger.error(
                f"Image not found for tag '{path_info.tag}' at step {path_info.step}"
            )
            sys.exit(1)

        if output_file:
            # Determine the output file extension based on the chosen format
            output_path = Path(output_file)
            if output_path.suffix:
                # If a suffix exists, replace it with the chosen format
                final_output_file = output_path.with_suffix(f".{image_format}")
            else:
                # If no suffix, append the chosen format
                final_output_file = output_path.with_suffix(f".{image_format}")

            # Convert bytes to image and save with specified format and quality
            try:
                from PIL import Image
                import io

                image = Image.open(io.BytesIO(image_bytes))
                if image_format == "jpg":
                    image.save(final_output_file, format="JPEG", quality=image_quality)
                else:
                    image.save(final_output_file, format="PNG")
                logger.success(f"Exported image to {final_output_file}")
            except ImportError:
                logger.error(
                    "Pillow (PIL) is not installed. Cannot convert image format."
                )
                logger.info("Please install it with: pip install Pillow")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Failed to save image: {e}")
                sys.exit(1)
        else:
            logger.info(
                f"Image data available ({len(image_bytes)} bytes). Use -o to save to file."
            )

    def _export_histogram_data(
        self,
        path_info: VirtualPathInfo,
        output_file: str | None,
        histogram_images: bool = False,
    ) -> None:
        """Export histogram data to text format."""
        histograms = self.parser.list_histograms()

        if path_info.tag not in histograms:
            logger.error(f"Histogram tag '{path_info.tag}' not found")
            logger.error(f"Available histogram tags: {', '.join(histograms)}")
            sys.exit(1)

        # Export histogram data to text format
        lines = []
        for hist_data in self.parser.iterate_histogram_data(path_info.tag):
            lines.append(f"Step: {hist_data.step}")
            lines.append(f"Min: {hist_data.min}, Max: {hist_data.max}")
            lines.append(f"Count: {hist_data.num}, Sum: {hist_data.sum}")
            lines.append("Buckets:")
            for limit, count in zip(hist_data.bucket_limit, hist_data.bucket):
                lines.append(f"  [{limit:.6f}]: {count}")
            lines.append("")  # Empty line between steps
        data = "\n".join(lines)
        self._handle_output(data, output_file, "histogram data")

    def _export_audio_data(
        self,
        path_info: VirtualPathInfo,
        output_file: str | None,
        audio_format: str = "mp3",
    ) -> None:
        """Export audio data."""
        if path_info.step is None:
            logger.error("Step is required for audio export")
            sys.exit(1)
        # Find the audio data for the specific step
        audio_result = None
        for data in self.parser.iterate_audio_data(path_info.tag):
            if data.step == path_info.step:
                audio_result = (data.encoded_audio_string, data.content_type)
                break

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
        # Find the text data for the specific step
        text_data = None
        for data in self.parser.iterate_text_data(path_info.tag):
            if data.step == path_info.step:
                text_data = data.text
                break

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

    def _export_mesh_data(
        self,
        path_info: VirtualPathInfo,
        output_file: str | None,
        ply_format: str = "binary",
    ) -> None:
        """Export mesh data to PLY format."""
        if path_info.step is None:
            logger.error("Step is required for mesh export")
            sys.exit(1)

        # Get mesh data from parser
        mesh_data_list = list(self.parser.iterate_mesh_data(path_info.tag))

        if not mesh_data_list:
            logger.error(f"Mesh not found for tag '{path_info.tag}'")
            sys.exit(1)

        # Find the specific step
        mesh_data = None
        for data in mesh_data_list:
            if data.step == path_info.step:
                mesh_data = data
                break

        if mesh_data is None:
            available_steps = [str(data.step) for data in mesh_data_list]
            logger.error(
                f"Mesh not found for tag '{path_info.tag}' at step {path_info.step}"
            )
            logger.error(f"Available steps: {', '.join(available_steps)}")
            sys.exit(1)

        if output_file:
            from tboardfs.core.ply_writer import write_mesh_as_ply

            output_path = Path(output_file)

            # Ensure output file has .ply extension
            if not output_path.suffix:
                output_path = output_path.with_suffix(".ply")
            elif output_path.suffix.lower() != ".ply":
                output_path = output_path.with_suffix(".ply")

            write_mesh_as_ply(mesh_data, output_path, ply_format)

            mesh_type = "point cloud" if mesh_data.is_point_cloud else "mesh"
            logger.success(
                f"Exported {mesh_type} to {output_path} "
                f"({mesh_data.num_vertices} vertices"
                f"{f', {mesh_data.num_faces} faces' if not mesh_data.is_point_cloud else ''}"
                f"{', with colors' if mesh_data.has_colors else ''}, "
                f"{ply_format} PLY format)"
            )
        else:
            mesh_type = "point cloud" if mesh_data.is_point_cloud else "mesh"
            logger.info(
                f"{mesh_type.title()} data available for tag '{path_info.tag}' at step {path_info.step}: "
                f"{mesh_data.num_vertices} vertices"
                f"{f', {mesh_data.num_faces} faces' if not mesh_data.is_point_cloud else ''}"
                f"{', with colors' if mesh_data.has_colors else ''}. "
                f"Use -o to save as PLY file."
            )

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

    def _export_hyperparameter_data(
        self, path_info: VirtualPathInfo, output_file: str | None
    ) -> None:
        """Export hyperparameter data as YAML."""
        try:
            import yaml
        except ImportError:
            logger.error("PyYAML not available. Cannot export hyperparameters.")
            logger.info("Please install it with: pip install PyYAML")
            sys.exit(1)

        # Collect all hyperparameters from all tags
        all_hyperparams: dict[str, Any] = {}
        hyperparameters_tags = self.parser.list_hyperparameters()

        if not hyperparameters_tags:
            logger.error("No hyperparameters found in the TensorBoard log")
            sys.exit(1)

        for tag in hyperparameters_tags:
            hyperparam_data_list = list(self.parser.iterate_hyperparameter_data(tag))

            for hyperparam_data in hyperparam_data_list:
                session_key = tag or f"session_{len(all_hyperparams)}"
                all_hyperparams[session_key] = hyperparam_data.hparams

        # Organize data for export
        export_data = {}
        if len(all_hyperparams) == 1:
            # Single session - export hyperparameters directly
            export_data = list(all_hyperparams.values())[0]
        else:
            # Multiple sessions - export as nested structure
            export_data = all_hyperparams

        if output_file:
            try:
                with Path(output_file).open("w") as f:
                    yaml.dump(export_data, f, default_flow_style=False, sort_keys=True)
                logger.success(f"Exported hyperparameters to {output_file}")
            except Exception as e:
                logger.error(f"Failed to write hyperparameters file: {e}")
                sys.exit(1)
        else:
            # Output to stdout
            yaml_output = yaml.dump(
                export_data, default_flow_style=False, sort_keys=True
            )
            print(yaml_output)
