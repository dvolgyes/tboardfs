"""Virtual filesystem path generation for TensorBoard data.

This module contains the logic for generating virtual filesystem paths
from TensorBoard event data, providing a unified interface for creating
directory structures and file paths for different data types.
"""

from collections.abc import Iterator
from typing import Any, TYPE_CHECKING


from tboardfs.core.constants import DEFAULT_DIGITS, FileSystemConstants
from tboardfs.core.image_processor import ImageProcessor

if TYPE_CHECKING:
    from tboardfs.efficient_parser import EfficientTensorBoardParser


class VirtualFilesystemGenerator:
    """Handles generation of virtual filesystem paths for TensorBoard data."""

    def __init__(self, parser: "EfficientTensorBoardParser") -> None:
        """Initialize the generator with a parser instance.

        Args:
            parser: The TensorBoard parser instance that provides data access methods.
        """
        self.parser = parser

    def get_virtual_paths(self, digits: int = DEFAULT_DIGITS) -> list[str]:
        """Get all virtual paths that would exist in the filesystem.

        Args:
            digits: Number of digits for zero-padding step numbers.

        Returns:
            Sorted list of all virtual file and directory paths.
        """
        # Determine processing mode
        all_tags = self._determine_processing_mode()

        # Start with base directories
        paths = self._get_base_directories()

        # Generate paths for each data type
        # Scalars (simple files)
        scalar_tags = self._get_tags_with_fallback("scalars", all_tags)
        paths.extend(self._generate_simple_paths("scalars", scalar_tags, "txt"))

        # Histograms (simple files)
        histogram_tags = self._get_tags_with_fallback("histograms", all_tags)
        paths.extend(self._generate_simple_paths("histograms", histogram_tags, "txt"))

        # Step-based data types
        for data_type in ["images", "audio", "text"]:
            tags = self._get_tags_with_fallback(data_type, all_tags)
            paths.extend(
                self._generate_step_based_paths(data_type, tags, digits, all_tags)
            )

        # Meshes (special handling for base tags)
        mesh_tags = self._get_tags_with_fallback("meshes", all_tags)
        paths.extend(self._generate_mesh_paths(mesh_tags, digits, all_tags))

        # Hyperparameters (special case)
        paths.extend(self._generate_hyperparameter_paths(all_tags))

        return sorted(set(paths))

    def _determine_processing_mode(self) -> dict[str, list[str]] | None:
        """Determine processing mode and return tags if available."""
        # Check if we have proper initialization
        if hasattr(self.parser, "_tags_cache") and hasattr(
            self.parser, "event_file_path"
        ):
            try:
                self.parser._scan_tags()  # Ensure cache is populated
                all_tags = self.parser._detailed_tags
                if all_tags is None:
                    return None
                return all_tags
            except Exception:
                # If scanning fails, fall back to mock-based method
                pass

        # Check if we have detailed tags from scanning
        if (
            hasattr(self.parser, "_detailed_tags")
            and self.parser._detailed_tags is not None
        ):
            return self.parser._detailed_tags
        else:
            # Fallback for tests using mocks - call the individual methods
            return None

    def _get_base_directories(self) -> list[str]:
        """Return base directory paths for virtual filesystem."""
        return [
            "scalars/",
            "images/",
            "histograms/",
            "audio/",
            "text/",
            "meshes/",
            "hp_params/",
        ]

    def _sanitize_tag_for_path(self, tag: str) -> str:
        """Convert tag to filesystem-safe name."""
        return tag.replace("/", FileSystemConstants.PATH_REPLACEMENT_CHAR)

    def _get_tags_with_fallback(
        self, data_type: str, all_tags: dict[str, list[str]] | None
    ) -> list[str]:
        """Get tags using normal mode or fallback to individual methods."""
        if all_tags is not None:
            return all_tags.get(data_type, [])

        # Fallback mode - call individual list methods
        method_map = {
            "scalars": self.parser.list_scalars,
            "images": self.parser.list_images,
            "histograms": self.parser.list_histograms,
            "audio": self.parser.list_audio,
            "text": self.parser.list_text,
            "meshes": self.parser.list_meshes,
            "hyperparameters": self.parser.list_hyperparameters,
        }

        try:
            method = method_map.get(data_type)
            if method:
                return method()
            return []
        except Exception:
            return []

    def _generate_simple_paths(
        self, data_type: str, tags: list[str], extension: str
    ) -> list[str]:
        """Generate simple file paths for data types without steps."""
        paths = []
        for tag in tags:
            safe_tag = self._sanitize_tag_for_path(tag)
            paths.append(f"{data_type}/{safe_tag}.{extension}")
        return paths

    def _iterate_data_with_fallback(
        self, data_type: str, tag: str, all_tags: dict[str, list[str]] | None
    ) -> Iterator[Any]:
        """Iterate data using normal mode or fallback with error handling."""
        if all_tags is not None:
            # Normal mode - direct iteration
            method_map = {
                "images": self.parser.iterate_image_data,
                "audio": self.parser.iterate_audio_data,
                "text": self.parser.iterate_text_data,
                "meshes": self.parser.iterate_mesh_data,
            }
            method = method_map.get(data_type)
            if method:
                return method(tag)
            return iter([])
        else:
            # Fallback mode with error handling
            try:
                method_map = {
                    "images": self.parser.iterate_image_data,
                    "audio": self.parser.iterate_audio_data,
                    "text": self.parser.iterate_text_data,
                    "meshes": self.parser.iterate_mesh_data,
                }
                method = method_map.get(data_type)
                if method:
                    return method(tag)
                return iter([])
            except Exception:
                return iter([])

    def _get_file_extension(self, data_type: str, data_item: Any = None) -> str:
        """Get file extension for data type."""
        if data_type == "images":
            if data_item and hasattr(data_item, "encoded_image_string"):
                return ImageProcessor.get_image_extension(
                    data_item.encoded_image_string
                )
            return "png"
        elif data_type == "audio":
            if data_item and hasattr(data_item, "content_type"):
                return self.parser.get_audio_extension(data_item.content_type)
            return "wav"
        elif data_type == "text":
            return "txt"
        elif data_type == "meshes":
            return "ply"
        else:
            return "txt"

    def _generate_step_based_paths(
        self,
        data_type: str,
        tags: list[str],
        digits: int,
        all_tags: dict[str, list[str]] | None,
    ) -> list[str]:
        """Generate step-based file paths with error handling."""
        paths = []

        for tag in tags:
            safe_tag = self._sanitize_tag_for_path(tag)
            # Add tag directory
            paths.append(f"{data_type}/{safe_tag}/")

            # Add individual step files
            for data_item in self._iterate_data_with_fallback(data_type, tag, all_tags):
                step = data_item.step
                padded_step = str(step).zfill(digits)
                extension = self._get_file_extension(data_type, data_item)
                paths.append(f"{data_type}/{safe_tag}/{padded_step}.{extension}")

        return paths

    def _generate_mesh_paths(
        self, tags: list[str], digits: int, all_tags: dict[str, list[str]] | None
    ) -> list[str]:
        """Generate mesh paths with base tag grouping."""
        paths = []

        # Group mesh tags by base tag
        base_tags = set()
        for tag in tags:
            if (
                tag.endswith("_VERTEX")
                or tag.endswith("_FACE")
                or tag.endswith("_COLOR")
            ):
                base_tag = tag.rsplit("_", 1)[0]
                base_tags.add(base_tag)

        for base_tag in base_tags:
            safe_tag = self._sanitize_tag_for_path(base_tag)
            # Add tag directory
            paths.append(f"meshes/{safe_tag}/")

            # Get steps from vertex tag (most reliable)
            vertex_tag = f"{base_tag}_VERTEX"
            if vertex_tag in tags:
                for data_item in self._iterate_data_with_fallback(
                    "meshes", vertex_tag, all_tags
                ):
                    step = data_item.step
                    padded_step = str(step).zfill(digits)
                    paths.append(f"meshes/{safe_tag}/{padded_step}.ply")

        return paths

    def _generate_hyperparameter_paths(
        self, all_tags: dict[str, list[str]] | None
    ) -> list[str]:
        """Generate hyperparameter paths."""
        tags = self._get_tags_with_fallback("hyperparameters", all_tags)
        if tags:
            return ["hp_params/hp_params.yaml"]
        return []
