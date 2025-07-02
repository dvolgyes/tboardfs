"""Histogram data exporter for TensorBoard events."""

from pathlib import Path
from typing import Any
import numpy as np
from tensorboard.compat.proto import event_pb2
from tensorboard.util import tensor_util
from loguru import logger

from .base_exporter import BaseExporter


class HistogramExporter(BaseExporter):
    """Export histogram data from TensorBoard events."""

    def __init__(self, output_path: Path, digits: int = 6):
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
                "min": float(hist.min),
                "max": float(hist.max),
                "num": int(hist.num),
                "sum": float(hist.sum),
                "sum_squares": float(hist.sum_squares),
                "bucket_limit": list(hist.bucket_limit),
                "bucket": list(hist.bucket),
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
                    "tensor_data": arr.flatten().tolist(),
                    "tensor_shape": arr.shape,
                    "tensor_dtype": str(arr.dtype),
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
            self._save_unified_histograms()

    def _save_unified_histograms(self) -> None:
        """Save histogram data in unified CSV + NPZ formats."""
        histograms_dir = self.output_path / "histograms"

        for tag, data_points in self.histogram_data_buffers.items():
            if not data_points:
                continue

            # Create histograms directory only when we have data to save
            self._ensure_directory_exists(histograms_dir)

            safe_tag = self._sanitize_tag(tag)

            # Sort by step
            data_points.sort(key=lambda x: x[0])

            # Check histogram format (legacy vs tensor)
            first_format = data_points[0][1].get("format", "unknown")

            if first_format == "legacy_histo":
                # Save legacy histogram format
                self._save_legacy_histogram_unified(
                    tag, safe_tag, data_points, histograms_dir
                )
            elif first_format == "tensor":
                # Save tensor histogram format
                self._save_tensor_histogram_unified(
                    tag, safe_tag, data_points, histograms_dir
                )
            else:
                logger.warning(
                    f"Unknown histogram format for tag '{tag}': {first_format}"
                )

    def _save_legacy_histogram_unified(
        self, tag: str, safe_tag: str, data_points: list, histograms_dir: Path
    ) -> None:
        """Save legacy histogram data in unified formats."""
        # Prepare data for CSV
        csv_rows = []
        npz_data: dict[str, list[Any]] = {
            "step": [],
            "wall_time": [],
            "min": [],
            "max": [],
            "num": [],
            "sum": [],
            "sum_squares": [],
            "bucket_limits": [],
            "bucket_counts": [],
        }

        for step, hist_data, wall_time in data_points:
            csv_rows.append(
                {
                    "step": step,
                    "wall_time": wall_time,
                    "min": hist_data["min"],
                    "max": hist_data["max"],
                    "num": hist_data["num"],
                    "sum": hist_data["sum"],
                    "sum_squares": hist_data["sum_squares"],
                    "bucket_limits": "|".join(map(str, hist_data["bucket_limit"])),
                    "bucket_counts": "|".join(map(str, hist_data["bucket"])),
                }
            )

            # Add to NPZ data
            npz_data["step"].append(step)
            npz_data["wall_time"].append(wall_time)
            npz_data["min"].append(hist_data["min"])
            npz_data["max"].append(hist_data["max"])
            npz_data["num"].append(hist_data["num"])
            npz_data["sum"].append(hist_data["sum"])
            npz_data["sum_squares"].append(hist_data["sum_squares"])
            npz_data["bucket_limits"].append(hist_data["bucket_limit"])
            npz_data["bucket_counts"].append(hist_data["bucket"])

        # Save CSV
        csv_file = histograms_dir / f"{safe_tag}.csv"
        with csv_file.open("w") as f:
            f.write(
                "step,wall_time,min,max,num,sum,sum_squares,bucket_limits,bucket_counts\n"
            )
            for row in csv_rows:
                f.write(
                    f"{row['step']},{row['wall_time']:.6f},{row['min']:.10g},"
                    f"{row['max']:.10g},{row['num']},{row['sum']:.10g},"
                    f"{row['sum_squares']:.10g},{row['bucket_limits']},{row['bucket_counts']}\n"
                )

        # Save NPZ
        npz_file = histograms_dir / f"{safe_tag}.npz"
        np.savez_compressed(
            npz_file,
            step=np.array(npz_data["step"]),
            wall_time=np.array(npz_data["wall_time"]),
            min=np.array(npz_data["min"]),
            max=np.array(npz_data["max"]),
            num=np.array(npz_data["num"]),
            sum=np.array(npz_data["sum"]),
            sum_squares=np.array(npz_data["sum_squares"]),
            bucket_limits=np.array(npz_data["bucket_limits"], dtype=object),
            bucket_counts=np.array(npz_data["bucket_counts"], dtype=object),
            tag=tag,
        )

        logger.debug(
            f"Saved histogram '{tag}' to {csv_file} and {npz_file} ({len(data_points)} points)"
        )

    def _save_tensor_histogram_unified(
        self, tag: str, safe_tag: str, data_points: list, histograms_dir: Path
    ) -> None:
        """Save tensor histogram data in unified formats."""
        # For tensor histograms, save raw tensor data and metadata
        csv_rows = []
        npz_data: dict[str, list[Any]] = {
            "step": [],
            "wall_time": [],
            "tensor_data": [],
            "tensor_shape": [],
            "tensor_dtype": [],
        }

        for step, hist_data, wall_time in data_points:
            csv_rows.append(
                {
                    "step": step,
                    "wall_time": wall_time,
                    "tensor_shape": str(hist_data["tensor_shape"]),
                    "tensor_dtype": hist_data["tensor_dtype"],
                    "tensor_data_summary": f"Array with {len(hist_data['tensor_data'])} elements",
                }
            )

            # Add to NPZ data
            npz_data["step"].append(step)
            npz_data["wall_time"].append(wall_time)
            npz_data["tensor_data"].append(hist_data["tensor_data"])
            npz_data["tensor_shape"].append(hist_data["tensor_shape"])
            npz_data["tensor_dtype"].append(hist_data["tensor_dtype"])

        # Save CSV (metadata only for tensor format)
        csv_file = histograms_dir / f"{safe_tag}.csv"
        with csv_file.open("w") as f:
            f.write("step,wall_time,tensor_shape,tensor_dtype,tensor_data_summary\n")
            for row in csv_rows:
                f.write(
                    f"{row['step']},{row['wall_time']:.6f},{row['tensor_shape']},"
                    f"{row['tensor_dtype']},{row['tensor_data_summary']}\n"
                )

        # Save NPZ (full tensor data)
        npz_file = histograms_dir / f"{safe_tag}.npz"
        np.savez_compressed(
            npz_file,
            step=np.array(npz_data["step"]),
            wall_time=np.array(npz_data["wall_time"]),
            tensor_data=np.array(npz_data["tensor_data"], dtype=object),
            tensor_shape=np.array(npz_data["tensor_shape"], dtype=object),
            tensor_dtype=np.array(npz_data["tensor_dtype"], dtype=object),
            tag=tag,
        )

        logger.debug(
            f"Saved tensor histogram '{tag}' to {csv_file} and {npz_file} ({len(data_points)} points)"
        )
