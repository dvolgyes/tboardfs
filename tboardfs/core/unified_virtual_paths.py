"""Unified virtual path system for tboardfs.

This module consolidates scattered virtual path handling logic into a unified system
for consistent path construction, sanitization, and parsing across the codebase.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import ClassVar
from enum import Enum


class DataType(Enum):
    """Supported data types for virtual filesystem."""

    SCALAR = "scalar"
    IMAGE = "image"
    VIDEO = "video"
    HISTOGRAM = "histogram"
    AUDIO = "audio"
    TEXT = "text"
    MESH = "mesh"
    HYPERPARAMETER = "hyperparameter"
    PR_CURVE = "pr_curve"


@dataclass(frozen=True)
class VirtualPathConfig:
    """Configuration for virtual path construction."""

    # Directory names for each data type
    DIRECTORIES: ClassVar[dict[DataType, str]] = {
        DataType.SCALAR: "scalars",
        DataType.IMAGE: "images",
        DataType.VIDEO: "videos",
        DataType.HISTOGRAM: "histograms",
        DataType.AUDIO: "audio",
        DataType.TEXT: "text",
        DataType.MESH: "meshes",
        DataType.HYPERPARAMETER: "hp_params",
        DataType.PR_CURVE: "pr_curves",
    }

    # Default file extensions for each data type
    DEFAULT_EXTENSIONS: ClassVar[dict[DataType, str]] = {
        DataType.SCALAR: "txt",
        DataType.IMAGE: "png",
        DataType.VIDEO: "gif",
        DataType.HISTOGRAM: "txt",
        DataType.AUDIO: "wav",
        DataType.TEXT: "txt",
        DataType.MESH: "ply",
        DataType.HYPERPARAMETER: "yaml",
        DataType.PR_CURVE: "csv",
    }

    # Data types that require step-based directories
    STEP_BASED_TYPES: ClassVar[set[DataType]] = {
        DataType.IMAGE,
        DataType.VIDEO,
        DataType.AUDIO,
        DataType.TEXT,
        DataType.MESH,
        DataType.PR_CURVE,
    }

    # Data types that have aggregated files (no steps)
    AGGREGATED_TYPES: ClassVar[set[DataType]] = {
        DataType.SCALAR,
        DataType.HISTOGRAM,
        DataType.HYPERPARAMETER,
    }

    @classmethod
    def get_directory_name(cls, data_type: DataType) -> str:
        """Get directory name for data type."""
        return cls.DIRECTORIES[data_type]

    @classmethod
    def get_default_extension(cls, data_type: DataType) -> str:
        """Get default file extension for data type."""
        return cls.DEFAULT_EXTENSIONS[data_type]

    @classmethod
    def is_step_based(cls, data_type: DataType) -> bool:
        """Check if data type uses step-based directory structure."""
        return data_type in cls.STEP_BASED_TYPES

    @classmethod
    def from_directory_name(cls, directory: str) -> DataType | None:
        """Get DataType from directory name."""
        for data_type, dir_name in cls.DIRECTORIES.items():
            if dir_name == directory:
                return data_type
        return None


@dataclass
class VirtualPathInfo:
    """Information about a parsed virtual path."""

    data_type: DataType
    tag: str
    step: int | None = None
    extension: str | None = None

    @property
    def safe_tag(self) -> str:
        """Get filesystem-safe version of tag."""
        return VirtualPathBuilder.sanitize_tag(self.tag)


class VirtualPathBuilder:
    """Builder for constructing virtual paths consistently."""

    def __init__(self, digits: int = 6):
        """Initialize path builder.

        Args:
            digits: Number of digits for step padding
        """
        self.digits = digits

    @staticmethod
    def sanitize_tag(tag: str) -> str:
        """Convert tag name to filesystem-safe format.

        Replaces forward slashes with underscores to make tags safe for use
        as directory and file names.

        Args:
            tag: Original tag name

        Returns:
            Filesystem-safe tag name
        """
        return tag.replace("/", "_")

    @staticmethod
    def restore_tag(safe_tag: str) -> str:
        """Restore original tag name from filesystem-safe format.

        Args:
            safe_tag: Filesystem-safe tag name

        Returns:
            Original tag name with slashes restored
        """
        return safe_tag.replace("_", "/")

    def format_step(self, step: int) -> str:
        """Format step number with zero padding.

        Args:
            step: Step number to format

        Returns:
            Zero-padded step string
        """
        return str(step).zfill(self.digits)

    def build_path(
        self,
        data_type: DataType,
        tag: str,
        step: int | None = None,
        extension: str | None = None,
        base_path: Path | None = None,
    ) -> Path:
        """Build virtual path for given data.

        Args:
            data_type: Type of data
            tag: Tag name
            step: Step number (for step-based types)
            extension: File extension (uses default if not provided)
            base_path: Base directory path

        Returns:
            Complete virtual path
        """
        if base_path is None:
            base_path = Path()

        if extension is None:
            extension = VirtualPathConfig.get_default_extension(data_type)

        directory = VirtualPathConfig.get_directory_name(data_type)
        safe_tag = self.sanitize_tag(tag)

        if VirtualPathConfig.is_step_based(data_type):
            if step is None:
                raise ValueError(f"Step required for {data_type.value} data type")

            # Structure: base_path/directory/tag/step.ext
            padded_step = self.format_step(step)
            return base_path / directory / safe_tag / f"{padded_step}.{extension}"
        else:
            # Structure: base_path/directory/tag.ext
            if data_type == DataType.HYPERPARAMETER:
                # Special case: hp_params/hp_params.yaml
                return base_path / directory / f"{directory}.{extension}"
            else:
                return base_path / directory / f"{safe_tag}.{extension}"

    def build_tag_directory(
        self,
        data_type: DataType,
        tag: str,
        base_path: Path | None = None,
    ) -> Path:
        """Build directory path for a tag.

        Args:
            data_type: Type of data
            tag: Tag name
            base_path: Base directory path

        Returns:
            Directory path for the tag
        """
        if base_path is None:
            base_path = Path()

        directory = VirtualPathConfig.get_directory_name(data_type)
        safe_tag = self.sanitize_tag(tag)

        if VirtualPathConfig.is_step_based(data_type):
            return base_path / directory / safe_tag
        else:
            return base_path / directory

    def get_all_directories(self, base_path: Path | None = None) -> list[Path]:
        """Get all virtual filesystem directories.

        Args:
            base_path: Base directory path

        Returns:
            List of all virtual directories
        """
        if base_path is None:
            base_path = Path()

        return [
            base_path / directory
            for directory in VirtualPathConfig.DIRECTORIES.values()
        ]


class VirtualPathParser:
    """Parser for virtual paths."""

    def __init__(self) -> None:
        """Initialize virtual path parser."""
        pass

    def parse(self, virtual_path: str) -> VirtualPathInfo:
        """Parse virtual path into components.

        Args:
            virtual_path: Virtual path string to parse

        Returns:
            Parsed path information

        Raises:
            ValueError: If path format is invalid
        """
        path_str = virtual_path.strip("/")
        if not path_str:
            raise ValueError("Empty virtual path")

        parts = path_str.split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid virtual path format: {virtual_path}")

        directory = parts[0]
        data_type = VirtualPathConfig.from_directory_name(directory)

        if data_type is None:
            valid_dirs = list(VirtualPathConfig.DIRECTORIES.values())
            raise ValueError(
                f"Unknown directory '{directory}'. Valid directories: {valid_dirs}"
            )

        if VirtualPathConfig.is_step_based(data_type):
            return self._parse_step_based_path(data_type, parts, virtual_path)
        else:
            return self._parse_aggregated_path(data_type, parts, virtual_path)

    def _parse_step_based_path(
        self, data_type: DataType, parts: list[str], virtual_path: str
    ) -> VirtualPathInfo:
        """Parse step-based path format: directory/tag/step.ext"""
        if len(parts) != 3:
            raise ValueError(
                f"Step-based path must have format 'directory/tag/step.ext': {virtual_path}"
            )

        tag = VirtualPathBuilder.restore_tag(parts[1])
        step_file = parts[2]

        # Extract step and extension
        if "." not in step_file:
            raise ValueError(f"Step file must have extension: {step_file}")

        step_str, extension = step_file.rsplit(".", 1)

        try:
            step = int(step_str)
        except ValueError:
            raise ValueError(f"Invalid step number: {step_str}")

        return VirtualPathInfo(
            data_type=data_type,
            tag=tag,
            step=step,
            extension=extension,
        )

    def _parse_aggregated_path(
        self, data_type: DataType, parts: list[str], virtual_path: str
    ) -> VirtualPathInfo:
        """Parse aggregated path format: directory/file.ext"""
        if data_type == DataType.HYPERPARAMETER:
            # Special case: hp_params/hp_params.yaml
            if len(parts) != 2 or not parts[1].startswith("hp_params."):
                raise ValueError(
                    f"Hyperparameter path must be 'hp_params/hp_params.yaml': {virtual_path}"
                )
            extension = parts[1].split(".", 1)[1]
            return VirtualPathInfo(
                data_type=data_type,
                tag="hp_params",  # Fixed tag for hyperparameters
                extension=extension,
            )
        else:
            # Standard aggregated format: directory/tag.ext
            if len(parts) != 2:
                raise ValueError(
                    f"Aggregated path must have format 'directory/tag.ext': {virtual_path}"
                )

            file_part = parts[1]
            if "." not in file_part:
                raise ValueError(f"File must have extension: {file_part}")

            tag_part, extension = file_part.rsplit(".", 1)
            tag = VirtualPathBuilder.restore_tag(tag_part)

            return VirtualPathInfo(
                data_type=data_type,
                tag=tag,
                extension=extension,
            )


# Convenience instances
default_path_builder = VirtualPathBuilder()
default_path_parser = VirtualPathParser()


# Convenience functions for backward compatibility
def sanitize_tag_for_path(tag: str) -> str:
    """Convert tag name to filesystem-safe format."""
    return VirtualPathBuilder.sanitize_tag(tag)


def restore_tag_from_path(safe_tag: str) -> str:
    """Restore original tag name from filesystem-safe format."""
    return VirtualPathBuilder.restore_tag(safe_tag)


def format_step_with_padding(step: int, digits: int = 6) -> str:
    """Format step number with zero padding."""
    return str(step).zfill(digits)
