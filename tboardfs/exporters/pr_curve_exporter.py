"""Precision-Recall curve data exporter for TensorBoard events."""

from pathlib import Path
from typing import Any
import numpy as np
from tensorboard.compat.proto import event_pb2
from tensorboard.util import tensor_util
from loguru import logger

from .base_exporter import BaseExporter
from ..core.data_detector import TensorDataDetector


class PRCurveExporter(BaseExporter):
    """Export precision-recall curve data from TensorBoard events."""

    def __init__(self, output_path: Path, digits: int = 6):
        """Initialize PR curve exporter."""
        super().__init__(output_path, digits)

    def save_data(self, event: event_pb2.Event, value: Any, **kwargs: Any) -> None:
        """Save PR curve data from a TensorBoard event.

        Args:
            event: TensorBoard event containing the PR curve data
            value: Summary value containing PR curve tensor information
        """
        tag = value.tag
        safe_tag = self._sanitize_tag(tag)
        tag_dir = self.output_path / "pr_curves" / safe_tag
        self._ensure_directory_exists(tag_dir)

        if value.HasField("tensor") and TensorDataDetector.is_pr_curve_tensor(
            value.tensor, tag
        ):
            try:
                # Extract tensor data
                arr = tensor_util.make_ndarray(value.tensor)
                logger.debug(
                    f"Saving PR curve data for tag '{tag}', shape: {arr.shape}"
                )

                # Extract precision and recall data
                num_thresholds = arr.shape[1]
                thresholds = np.linspace(0.0, 1.0, num_thresholds)
                precision = arr[4, :]  # Row 4 contains precision
                recall = arr[5, :]  # Row 5 contains recall

                # Create padded step filename
                padded_step = self._format_step(event.step)
                csv_file = tag_dir / f"{padded_step}.csv"
                npz_file = tag_dir / f"{padded_step}.npz"

                # Write CSV file with headers
                with csv_file.open("w") as f:
                    f.write("threshold,precision,recall\n")
                    for i in range(num_thresholds):
                        f.write(
                            f"{thresholds[i]:.6f},{precision[i]:.6f},{recall[i]:.6f}\n"
                        )

                # Write NPZ file
                np.savez_compressed(
                    npz_file,
                    step=event.step,
                    wall_time=event.wall_time,
                    threshold=thresholds,
                    precision=precision,
                    recall=recall,
                    tag=tag,
                    num_thresholds=num_thresholds,
                )

                logger.debug(
                    f"Saved PR curve to {csv_file} and {npz_file} ({num_thresholds} points)"
                )

            except Exception as e:
                logger.warning(f"Failed to save PR curve {tag}: {e}")
        else:
            logger.warning(f"No valid PR curve tensor found for tag {tag}")
