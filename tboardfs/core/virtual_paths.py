"""Virtual path parsing and export handling for tboardfs."""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from loguru import logger

from ..efficient_parser import EfficientTensorBoardParser
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

    def __init__(self, parser: EfficientTensorBoardParser):
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
        elif data_type == "meshes":
            return self._parse_mesh_path(parts, virtual_path)
        elif data_type == "hp_params":
            return self._parse_hyperparameter_path(parts, virtual_path)
        else:
            logger.error(f"Unsupported data type: {data_type}")
            logger.error(
                "Supported types: scalars, images, histograms, audio, text, meshes, hp_params"
            )
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

    def _parse_mesh_path(self, parts: list, virtual_path: str) -> VirtualPathInfo:
        """Parse mesh path: meshes/tag_name/step.ply"""
        if len(parts) != 3:
            logger.error(
                f"Mesh path must be in format 'meshes/tag/step.ply': {virtual_path}"
            )
            sys.exit(1)

        # For meshes, we need special handling since tags can contain both / and _
        # The virtual path format is: meshes/Tag_With_Underscores_Replacing_Slashes/step.ply
        # We need to convert back: underscores to slashes for the base tag
        safe_tag = parts[1]
        step_file = parts[2]

        if not step_file.endswith(".ply"):
            logger.error(f"Mesh file must end with .ply: {virtual_path}")
            sys.exit(1)

        step = int(step_file[:-4])  # Remove .ply and get step

        # Convert safe_tag back to original tag
        # We need to find the correct base tag from available mesh tags
        mesh_tags = self.parser.list_meshes()
        base_tags = set()
        for mesh_tag in mesh_tags:
            base_tag = mesh_tag.rstrip("_VERTEX").rstrip("_FACE").rstrip("_COLOR")
            base_tags.add(base_tag)

        # Find the base tag that matches our safe_tag when encoded
        tag = None
        for base_tag in base_tags:
            encoded_base_tag = base_tag.replace("/", "_")
            if encoded_base_tag == safe_tag:
                tag = base_tag
                break

        if tag is None:
            logger.error(f"Cannot find mesh base tag for virtual path: {virtual_path}")
            logger.error(f"Available base tags: {sorted(base_tags)}")
            sys.exit(1)

        return VirtualPathInfo(data_type="meshes", tag=tag, step=step, extension="ply")

    def _parse_hyperparameter_path(
        self, parts: list, virtual_path: str
    ) -> VirtualPathInfo:
        """Parse hyperparameter path: hp_params/hp_params.yaml"""
        if len(parts) != 2 or parts[1] != "hp_params.yaml":
            logger.error(
                f"Hyperparameter path must be 'hp_params/hp_params.yaml': {virtual_path}"
            )
            sys.exit(1)

        return VirtualPathInfo(
            data_type="hp_params", tag="hyperparameters", extension="yaml"
        )

    def _is_data_type_enabled(
        self, data_type: str, type_filters: dict[str, set[str]] | None
    ) -> bool:
        """Check if a data type should be processed based on filters."""
        if not type_filters:
            return True

        # Map plural data types to singular for filter matching
        type_mapping = {
            "scalars": "scalar",
            "images": "image",
            "histograms": "histogram",
            "audio": "audio",
            "text": "text",
            "meshes": "mesh",
            "hp_params": "hyperparameter",
        }

        filter_type = type_mapping.get(data_type, data_type)
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
