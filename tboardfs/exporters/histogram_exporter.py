"""Histogram data exporter for TensorBoard events."""

from pathlib import Path
from typing import Any
from tensorboard.compat.proto import event_pb2
from tensorboard.util import tensor_util
from loguru import logger

from .base_exporter import BaseExporter
from ..core.histogram_utils import HistogramExportUtils
from ..core.constants import DEFAULT_DIGITS


class HistogramExporter(BaseExporter):
    """Export histogram data from TensorBoard events."""

    def __init__(self, output_path: Path, digits: int = DEFAULT_DIGITS):
        """Initialize histogram exporter."""
        super().__init__(output_path, digits)
        self.histogram_data_buffers: dict[str, list[tuple[int, dict, float]]] = {}

    def save_data(
        self,
        event: event_pb2.Event,
        value: Any,
        histogram_images: bool = False,
        **kwargs: Any,
    ) -> None:
        """Save histogram data from a TensorBoard event.

        Args:
            event: TensorBoard event containing the histogram data
            value: Summary value containing histogram information
            histogram_images: Whether to save as visualization images (legacy)
        """
        tag = value.tag

        if histogram_images:
            # Save as visualization images (old behavior)
            self._save_histogram_as_image(event, value)
        else:
            # Buffer data for unified CSV + NPZ export
            self._buffer_histogram_data(event, value, tag)

    def _buffer_histogram_data(
        self, event: event_pb2.Event, value: Any, tag: str
    ) -> None:
        """Buffer histogram data for unified export."""
        if value.HasField("histo"):
            # Legacy format with histo field
            hist = value.histo
            hist_data = {
                "histogram": hist,
                "format": "legacy_histo",
            }
            if tag not in self.histogram_data_buffers:
                self.histogram_data_buffers[tag] = []
            self.histogram_data_buffers[tag].append(
                (event.step, hist_data, event.wall_time)
            )

        elif value.HasField("tensor"):
            # TensorBoard v2 format with tensor field
            try:
                arr = tensor_util.make_ndarray(value.tensor)
                hist_data = {
                    "values": arr,
                    "format": "tensor",
                }
                if tag not in self.histogram_data_buffers:
                    self.histogram_data_buffers[tag] = []
                self.histogram_data_buffers[tag].append(
                    (event.step, hist_data, event.wall_time)
                )
            except Exception as e:
                logger.warning(f"Failed to buffer tensor histogram data for {tag}: {e}")
        else:
            logger.warning(f"Unknown histogram format for tag '{tag}'")

    def _save_histogram_as_image(self, event: event_pb2.Event, value: Any) -> None:
        """Save histogram as visualization image (legacy behavior)."""
        tag = value.tag
        safe_tag = self._sanitize_tag(tag)
        histogram_file = self.output_path / "histograms" / f"{safe_tag}.txt"
        self._ensure_directory_exists(histogram_file.parent)

        hist = value.histo
        with histogram_file.open("a") as f:
            f.write(f"Step: {event.step}\n")
            f.write(f"Min: {hist.min}, Max: {hist.max}\n")
            f.write(f"Count: {hist.num}, Sum: {hist.sum}\n")
            f.write("Buckets:\n")
            for limit, count in zip(hist.bucket_limit, hist.bucket):
                f.write(f"  [{limit:.6f}]: {count}\n")
            f.write("\n")

    def finalize_export(self) -> None:
        """Finalize export by saving unified formats."""
        if self.histogram_data_buffers:
            logger.debug("Exporting unified histogram formats (CSV + NPZ)")
            HistogramExportUtils.save_unified_histograms(
                self.histogram_data_buffers,
                self.output_path,
                sanitize_tag_func=self._sanitize_tag,
                ensure_directory_func=self._ensure_directory_exists,
            )
