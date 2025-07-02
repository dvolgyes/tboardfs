"""Shared histogram export utilities for TensorBoard data.

This module provides unified histogram export functionality to eliminate
code duplication between the parser and histogram exporter.
"""

import csv
from pathlib import Path
from typing import Any
from collections.abc import Callable

import numpy as np
from loguru import logger

from .constants import FileSystemConstants
from .exceptions import FileWriteError, DirectoryCreationError


class HistogramExportUtils:
    """Utilities for exporting histogram data in unified formats."""

    @staticmethod
    def save_unified_histograms(
        histogram_data_buffers: dict[str, list[tuple[int, dict, float]]],
        output_path: Path,
        sanitize_tag_func: Callable[[str], str] | None = None,
        ensure_directory_func: Callable[[Path], None] | None = None,
    ) -> None:
        """Save histogram data in unified CSV + NPZ formats.

        Args:
            histogram_data_buffers: Dictionary mapping tags to list of data points
            output_path: Base output directory path
            sanitize_tag_func: Function to sanitize tag names (optional)
            ensure_directory_func: Function to ensure directory exists (optional)
        """
        histograms_dir = output_path / "histograms"

        for tag, data_points in histogram_data_buffers.items():
            if not data_points:
                continue

            # Create histograms directory only when we have data to save
            if ensure_directory_func:
                try:
                    ensure_directory_func(histograms_dir)
                except Exception as e:
                    raise DirectoryCreationError(histograms_dir, e) from e
            else:
                try:
                    histograms_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    raise DirectoryCreationError(histograms_dir, e) from e

            # Sanitize tag name
            if sanitize_tag_func:
                safe_tag = sanitize_tag_func(tag)
            else:
                safe_tag = tag.replace("/", FileSystemConstants.PATH_REPLACEMENT_CHAR)

            # Sort by step
            data_points.sort(key=lambda x: x[0])

            # Check histogram format (legacy vs tensor)
            first_format = data_points[0][1].get("format", "unknown")

            if first_format == "legacy_histo":
                # Save legacy histogram format
                HistogramExportUtils._save_legacy_histogram_unified(
                    tag, safe_tag, data_points, histograms_dir
                )
            elif first_format == "tensor":
                # Save tensor histogram format
                HistogramExportUtils._save_tensor_histogram_unified(
                    tag, safe_tag, data_points, histograms_dir
                )
            else:
                logger.warning(
                    f"Unknown histogram format for tag '{tag}': {first_format}"
                )

    @staticmethod
    def _save_legacy_histogram_unified(
        tag: str,
        safe_tag: str,
        data_points: list[tuple[int, dict, float]],
        histograms_dir: Path,
    ) -> None:
        """Save legacy histogram data in unified formats."""
        # Prepare data for CSV
        csv_rows = []
        npz_data: dict[str, list[Any]] = {"steps": [], "buckets": [], "counts": []}

        for step, histogram_data, wall_time in data_points:
            hist = histogram_data["histogram"]

            # CSV format: one row per step
            csv_row = {
                "step": step,
                "wall_time": wall_time,
                "min": hist.min,
                "max": hist.max,
                "num": hist.num,
                "sum": hist.sum,
                "sum_squares": hist.sum_squares,
                "bucket_limit": list(hist.bucket_limit),
                "bucket": list(hist.bucket),
            }
            csv_rows.append(csv_row)

            # NPZ format: separate arrays
            npz_data["steps"].append(step)
            npz_data["buckets"].append(list(hist.bucket_limit))
            npz_data["counts"].append(list(hist.bucket))

        # Save CSV file
        csv_file = histograms_dir / f"{safe_tag}.csv"
        HistogramExportUtils._write_csv_file(csv_file, csv_rows)

        # Save NPZ file
        npz_file = histograms_dir / f"{safe_tag}.npz"
        HistogramExportUtils._write_npz_file(npz_file, npz_data)

        logger.debug(f"Saved legacy histogram '{tag}' to {csv_file} and {npz_file}")

    @staticmethod
    def _save_tensor_histogram_unified(
        tag: str,
        safe_tag: str,
        data_points: list[tuple[int, dict, float]],
        histograms_dir: Path,
    ) -> None:
        """Save tensor histogram data in unified formats."""
        # Prepare data for CSV
        csv_rows = []
        npz_data: dict[str, list[Any]] = {"steps": [], "values": []}

        for step, histogram_data, wall_time in data_points:
            values = histogram_data["values"]

            # CSV format: one row per step
            csv_row = {
                "step": step,
                "wall_time": wall_time,
                "values": values.tolist()
                if hasattr(values, "tolist")
                else list(values),
            }
            csv_rows.append(csv_row)

            # NPZ format: separate arrays
            npz_data["steps"].append(step)
            npz_data["values"].append(values)

        # Save CSV file
        csv_file = histograms_dir / f"{safe_tag}.csv"
        HistogramExportUtils._write_tensor_csv_file(csv_file, csv_rows)

        # Save NPZ file
        npz_file = histograms_dir / f"{safe_tag}.npz"
        HistogramExportUtils._write_npz_file(npz_file, npz_data)

        logger.debug(f"Saved tensor histogram '{tag}' to {csv_file} and {npz_file}")

    @staticmethod
    def _write_csv_file(csv_file: Path, csv_rows: list[dict[str, Any]]) -> None:
        """Write CSV file for legacy histogram format."""
        try:
            with csv_file.open("w", newline="", encoding="utf-8") as f:
                if csv_rows:
                    # Use the first row to determine fieldnames
                    fieldnames = list(csv_rows[0].keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

                    for row in csv_rows:
                        # Convert list fields to string representation for CSV
                        csv_row = {}
                        for key, value in row.items():
                            if isinstance(value, list):
                                csv_row[key] = ";".join(map(str, value))
                            else:
                                csv_row[key] = value
                        writer.writerow(csv_row)
        except Exception as e:
            logger.error(f"Failed to write CSV file {csv_file}: {e}")
            raise FileWriteError(csv_file, e) from e

    @staticmethod
    def _write_tensor_csv_file(csv_file: Path, csv_rows: list[dict[str, Any]]) -> None:
        """Write CSV file for tensor histogram format."""
        try:
            with csv_file.open("w", newline="", encoding="utf-8") as f:
                if csv_rows:
                    # Use the first row to determine fieldnames
                    fieldnames = list(csv_rows[0].keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

                    for row in csv_rows:
                        # Convert list/array fields to string representation for CSV
                        csv_row = {}
                        for key, value in row.items():
                            if isinstance(value, (list, np.ndarray)):
                                csv_row[key] = ";".join(map(str, value))
                            else:
                                csv_row[key] = value
                        writer.writerow(csv_row)
        except Exception as e:
            logger.error(f"Failed to write tensor CSV file {csv_file}: {e}")
            raise FileWriteError(csv_file, e) from e

    @staticmethod
    def _write_npz_file(npz_file: Path, npz_data: dict[str, Any]) -> None:
        """Write NPZ file for histogram data."""
        try:
            # Convert all data to numpy arrays
            np_data = {}
            for key, value in npz_data.items():
                if isinstance(value, list):
                    np_data[key] = np.array(value, dtype=object)
                else:
                    np_data[key] = np.array(value)

            np.savez_compressed(npz_file, **np_data)
        except Exception as e:
            logger.error(f"Failed to write NPZ file {npz_file}: {e}")
            raise FileWriteError(npz_file, e) from e
