"""Scalar data exporter for TensorBoard events."""

from pathlib import Path
from typing import Any
import numpy as np
from tensorboard.compat.proto import event_pb2
from tensorboard.util import tensor_util
from loguru import logger

from .base_exporter import BaseExporter
from ..scalar_file import ScalarFile


class ScalarExporter(BaseExporter):
    """Export scalar data from TensorBoard events."""

    def __init__(self, output_path: Path, digits: int = 6):
        """Initialize scalar exporter."""
        super().__init__(output_path, digits)
        self.scalar_files: dict[Path, ScalarFile] = {}
        self.scalar_data_buffers: dict[str, list[tuple[int, float, float]]] = {}

    def save_data(
        self,
        event: event_pb2.Event,
        value: Any,
        enable_legacy_format: bool = True,
        **kwargs: Any,
    ) -> None:
        """Save scalar data from a TensorBoard event.

        Args:
            event: TensorBoard event containing the scalar data
            value: Summary value containing scalar information
            enable_legacy_format: Whether to save legacy .txt format
        """
        tag = value.tag
        safe_tag = self._sanitize_tag(tag)

        # Extract scalar value
        scalar_val = self._extract_scalar_value(value, tag)
        if scalar_val is None:
            logger.warning(
                f"Could not extract scalar value for tag '{tag}' at step {event.step}"
            )
            return

        # Save to legacy text format if enabled
        if enable_legacy_format:
            self._save_legacy_format(event, tag, safe_tag, scalar_val)

        # Buffer data for unified CSV + NPZ export
        self._buffer_for_unified_export(event, tag, scalar_val)

    def _extract_scalar_value(self, value: Any, tag: str) -> float | None:
        """Extract scalar value from summary value."""
        if value.HasField("simple_value"):
            return value.simple_value
        elif value.HasField("tensor"):
            try:
                arr = tensor_util.make_ndarray(value.tensor)
                if arr.size == 1:
                    return float(arr.item())
                else:
                    logger.warning(
                        f"Tensor for scalar tag '{tag}' has more than one element (size={arr.size}). Skipping."
                    )
            except Exception as e:
                logger.warning(
                    f"Could not extract scalar from tensor for tag '{tag}': {e}"
                )
        return None

    def _save_legacy_format(
        self, event: event_pb2.Event, tag: str, safe_tag: str, scalar_val: float
    ) -> None:
        """Save scalar data in legacy text format."""
        scalar_file_path = self.output_path / "scalars" / f"{safe_tag}.txt"

        if scalar_file_path not in self.scalar_files:
            self.scalar_files[scalar_file_path] = ScalarFile(scalar_file_path)
        self.scalar_files[scalar_file_path].append(event.step, scalar_val)

    def _buffer_for_unified_export(
        self, event: event_pb2.Event, tag: str, scalar_val: float
    ) -> None:
        """Buffer scalar data for unified CSV + NPZ export."""
        if tag not in self.scalar_data_buffers:
            self.scalar_data_buffers[tag] = []
        self.scalar_data_buffers[tag].append((event.step, scalar_val, event.wall_time))

    def finalize_export(self) -> None:
        """Finalize export by closing files and saving unified formats."""
        # Close all scalar files to ensure data is written and sorted
        for scalar_file in self.scalar_files.values():
            scalar_file.close()

        # Export unified formats
        if self.scalar_data_buffers:
            logger.debug("Exporting unified scalar formats (CSV + NPZ)")
            self._save_unified_scalars()

    def _save_unified_scalars(self) -> None:
        """Save scalar data in unified CSV + NPZ formats."""
        scalars_dir = self.output_path / "scalars"

        for tag, data_points in self.scalar_data_buffers.items():
            if not data_points:
                continue

            # Create scalars directory only when we have data to save
            self._ensure_directory_exists(scalars_dir)

            safe_tag = self._sanitize_tag(tag)

            # Sort by step
            data_points.sort(key=lambda x: x[0])

            steps = np.array([dp[0] for dp in data_points])
            values = np.array([dp[1] for dp in data_points])
            wall_times = np.array([dp[2] for dp in data_points])

            # Save CSV format
            csv_file = scalars_dir / f"{safe_tag}.csv"
            with csv_file.open("w") as f:
                f.write("step,value,wall_time\n")
                for step, value, wall_time in data_points:
                    f.write(f"{step},{value:.10g},{wall_time:.6f}\n")

            # Save NPZ format
            npz_file = scalars_dir / f"{safe_tag}.npz"
            np.savez_compressed(
                npz_file, step=steps, value=values, wall_time=wall_times, tag=tag
            )

            logger.debug(
                f"Saved scalar '{tag}' to {csv_file} and {npz_file} ({len(data_points)} points)"
            )
