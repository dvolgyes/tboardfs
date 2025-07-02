"""CLI validation utilities for tboardfs.

This module contains validation functions for CLI options and arguments,
providing centralized validation logic for format options and data type filtering.
"""

import sys

from loguru import logger


# Supported data types for filtering
SUPPORTED_DATA_TYPES = {
    "image",
    "audio",
    "video",
    "text",
    "scalar",
    "histogram",
    "mesh",
    "hyperparameter",
}


class CLIValidator:
    """Centralized CLI validation functionality."""

    @staticmethod
    def validate_image_format_options(png: bool, jpg: bool) -> str:
        """Validate image format options and return the selected format.

        Args:
            png: Whether PNG format was requested
            jpg: Whether JPG format was requested

        Returns:
            The selected image format as a string ("png" or "jpg")

        Raises:
            SystemExit: If both PNG and JPG are specified
        """
        if png and jpg:
            logger.error("Cannot specify both --png and --jpg. Please choose one.")
            sys.exit(1)

        return "png" if png else "jpg"

    @staticmethod
    def validate_audio_format_options(wav: bool, mp3: bool) -> str:
        """Validate audio format options and return the selected format.

        Args:
            wav: Whether WAV format was requested
            mp3: Whether MP3 format was requested

        Returns:
            The selected audio format as a string ("wav" or "mp3")

        Raises:
            SystemExit: If both WAV and MP3 are specified
        """
        if wav and mp3:
            logger.error("Cannot specify both --wav and --mp3. Please choose one.")
            sys.exit(1)

        return "wav" if wav else "mp3"

    @staticmethod
    def validate_ply_format_options(ply_bin: bool, ply_txt: bool) -> str:
        """Validate PLY format options and return the selected format.

        Args:
            ply_bin: Whether binary PLY format was requested
            ply_txt: Whether text PLY format was requested

        Returns:
            The selected PLY format as a string ("binary" or "text")

        Raises:
            SystemExit: If both binary and text PLY are specified
        """
        if ply_bin and ply_txt:
            logger.error(
                "Cannot specify both --ply-bin and --ply-txt. Please choose one."
            )
            sys.exit(1)

        return "text" if ply_txt else "binary"

    @staticmethod
    def parse_data_type_list(data_types: tuple[str, ...]) -> set[str]:
        """Parse comma-separated data types from CLI arguments.

        Args:
            data_types: Tuple of data type strings, each may contain comma-separated values

        Returns:
            Set of individual data type strings

        Raises:
            SystemExit: If any unsupported data types are specified
        """
        result = set()

        for type_group in data_types:
            # Split by comma and strip whitespace
            types = [t.strip().lower() for t in type_group.split(",")]
            result.update(types)

        # Validate all types are supported
        invalid_types = result - SUPPORTED_DATA_TYPES
        if invalid_types:
            logger.error(f"Unsupported data types: {', '.join(sorted(invalid_types))}")
            logger.error(f"Supported types: {', '.join(sorted(SUPPORTED_DATA_TYPES))}")
            sys.exit(1)

        return result

    @staticmethod
    def validate_type_filtering_options(
        ignore_types: tuple[str, ...], select_types: tuple[str, ...]
    ) -> dict[str, set[str]]:
        """Validate and process data type filtering options.

        Args:
            ignore_types: Types to ignore during processing
            select_types: Types to select (process only these)

        Returns:
            Dictionary with 'ignore' and 'select' keys containing sets of data types

        Raises:
            SystemExit: If both ignore and select are specified or if invalid types are given
        """
        ignore_set = (
            CLIValidator.parse_data_type_list(ignore_types) if ignore_types else set()
        )
        select_set = (
            CLIValidator.parse_data_type_list(select_types) if select_types else set()
        )

        # Check for mutual exclusivity
        if ignore_set and select_set:
            logger.error(
                "Cannot specify both --ignore and --select options. Please choose one."
            )
            sys.exit(1)

        return {"ignore": ignore_set, "select": select_set}


# Convenience functions for backward compatibility
def validate_image_format_options(png: bool, jpg: bool) -> str:
    """Validate image format options and return the selected format."""
    return CLIValidator.validate_image_format_options(png, jpg)


def validate_audio_format_options(wav: bool, mp3: bool) -> str:
    """Validate audio format options and return the selected format."""
    return CLIValidator.validate_audio_format_options(wav, mp3)


def validate_ply_format_options(ply_bin: bool, ply_txt: bool) -> str:
    """Validate PLY format options and return the selected format."""
    return CLIValidator.validate_ply_format_options(ply_bin, ply_txt)


def validate_type_filtering_options(
    ignore_types: tuple[str, ...], select_types: tuple[str, ...]
) -> dict[str, set[str]]:
    """Validate and process data type filtering options."""
    return CLIValidator.validate_type_filtering_options(ignore_types, select_types)


def parse_data_type_list(data_types: tuple[str, ...]) -> set[str]:
    """Parse comma-separated data types from CLI arguments."""
    return CLIValidator.parse_data_type_list(data_types)
