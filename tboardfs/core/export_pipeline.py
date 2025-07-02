"""Export pipeline for modular data processing and export."""

from pathlib import Path
from typing import Any
from collections.abc import Iterator
from dataclasses import dataclass
from tensorboard.compat.proto import event_pb2
from loguru import logger

from ..core.data_detector import TensorDataDetector
from ..exporters.scalar_exporter import ScalarExporter
from ..exporters.image_exporter import ImageExporter
from ..exporters.histogram_exporter import HistogramExporter
from ..exporters.audio_exporter import AudioExporter
from ..exporters.text_exporter import TextExporter
from ..exporters.mesh_exporter import MeshExporter
from ..exporters.hyperparameter_exporter import HyperparameterExporter
from ..exporters.pr_curve_exporter import PRCurveExporter


@dataclass
class ExportConfig:
    """Configuration for export pipeline."""

    output_path: Path
    enabled_types: set[str]
    digits: int = 6
    histogram_images: bool = False
    audio_format: str = "wav"
    ply_format: str = "ascii"


class ExportProcessor:
    """Base class for data processors in export pipeline."""

    def __init__(self, config: ExportConfig):
        self.config = config

    def can_handle(self, event: event_pb2.Event, value: Any) -> bool:
        """Check if this processor can handle the given data."""
        raise NotImplementedError

    def process(self, event: event_pb2.Event, value: Any) -> None:
        """Process and export the data."""
        raise NotImplementedError

    def finalize(self) -> None:
        """Finalize processing (called after all events processed)."""
        pass


class ScalarProcessor(ExportProcessor):
    """Process scalar data using ScalarExporter."""

    def __init__(self, config: ExportConfig):
        super().__init__(config)
        self.exporter = ScalarExporter(config.output_path, config.digits)

    def can_handle(self, event: event_pb2.Event, value: Any) -> bool:
        if "scalar" not in self.config.enabled_types:
            return False

        plugin_name = None
        if value.HasField("metadata"):
            plugin_name = value.metadata.plugin_data.plugin_name

        return plugin_name == "scalars" or value.HasField("simple_value")

    def process(self, event: event_pb2.Event, value: Any) -> None:
        self.exporter.save_data(event, value)


class ImageVideoProcessor(ExportProcessor):
    """Process image and video data using ImageExporter."""

    def __init__(self, config: ExportConfig):
        super().__init__(config)
        self.has_image_types = bool(
            {"image", "video"}.intersection(config.enabled_types)
        )
        if self.has_image_types:
            self.exporter = ImageExporter(config.output_path, config.digits)

    def can_handle(self, event: event_pb2.Event, value: Any) -> bool:
        if not self.has_image_types:
            return False

        plugin_name = None
        if value.HasField("metadata"):
            plugin_name = value.metadata.plugin_data.plugin_name

        if plugin_name == "images" or value.HasField("image"):
            return True

        if value.HasField("tensor") and TensorDataDetector.is_image_tensor(
            value.tensor, value.tag
        ):
            return True

        return False

    def process(self, event: event_pb2.Event, value: Any) -> None:
        if self.has_image_types:
            self.exporter.save_data(event, value)


class HistogramProcessor(ExportProcessor):
    """Process histogram data using HistogramExporter."""

    def __init__(self, config: ExportConfig):
        super().__init__(config)
        self.exporter: HistogramExporter | None = None
        if "histogram" in config.enabled_types:
            self.exporter = HistogramExporter(config.output_path, config.digits)

    def can_handle(self, event: event_pb2.Event, value: Any) -> bool:
        if "histogram" not in self.config.enabled_types:
            return False

        plugin_name = None
        if value.HasField("metadata"):
            plugin_name = value.metadata.plugin_data.plugin_name

        return plugin_name == "histograms" or value.HasField("histo")

    def process(self, event: event_pb2.Event, value: Any) -> None:
        if self.exporter:
            self.exporter.save_data(
                event, value, histogram_images=self.config.histogram_images
            )


class AudioProcessor(ExportProcessor):
    """Process audio data using AudioExporter."""

    def __init__(self, config: ExportConfig):
        super().__init__(config)
        self.exporter: AudioExporter | None = None
        if "audio" in config.enabled_types:
            self.exporter = AudioExporter(
                config.output_path, config.digits, config.audio_format
            )

    def can_handle(self, event: event_pb2.Event, value: Any) -> bool:
        if "audio" not in self.config.enabled_types:
            return False

        plugin_name = None
        if value.HasField("metadata"):
            plugin_name = value.metadata.plugin_data.plugin_name

        return plugin_name == "audio" or value.HasField("audio")

    def process(self, event: event_pb2.Event, value: Any) -> None:
        if self.exporter:
            self.exporter.save_data(event, value, audio_format=self.config.audio_format)


class TextProcessor(ExportProcessor):
    """Process text data using TextExporter."""

    def __init__(self, config: ExportConfig):
        super().__init__(config)
        self.exporter: TextExporter | None = None
        if "text" in config.enabled_types:
            self.exporter = TextExporter(config.output_path, config.digits)

    def can_handle(self, event: event_pb2.Event, value: Any) -> bool:
        if "text" not in self.config.enabled_types:
            return False

        return value.HasField("tensor") and TensorDataDetector.is_text_tensor(
            value.tensor
        )

    def process(self, event: event_pb2.Event, value: Any) -> None:
        if self.exporter:
            self.exporter.save_data(event, value)


class MeshProcessor(ExportProcessor):
    """Process mesh data using MeshExporter."""

    def __init__(self, config: ExportConfig):
        super().__init__(config)
        self.exporter: MeshExporter | None = None
        if "mesh" in config.enabled_types:
            self.exporter = MeshExporter(
                config.output_path, config.digits, config.ply_format
            )

    def can_handle(self, event: event_pb2.Event, value: Any) -> bool:
        if "mesh" not in self.config.enabled_types:
            return False

        plugin_name = None
        if value.HasField("metadata"):
            plugin_name = value.metadata.plugin_data.plugin_name

        return plugin_name == "mesh" or TensorDataDetector.is_mesh_tensor(value.tag)

    def process(self, event: event_pb2.Event, value: Any) -> None:
        if self.exporter:
            self.exporter.save_data(event, value, ply_format=self.config.ply_format)

    def finalize(self) -> None:
        if self.exporter:
            self.exporter.finalize()


class HyperparameterProcessor(ExportProcessor):
    """Process hyperparameter data using HyperparameterExporter."""

    def __init__(self, config: ExportConfig):
        super().__init__(config)
        self.exporter: HyperparameterExporter | None = None
        if "hyperparameter" in config.enabled_types:
            self.exporter = HyperparameterExporter(config.output_path, config.digits)

    def can_handle(self, event: event_pb2.Event, value: Any) -> bool:
        if "hyperparameter" not in self.config.enabled_types:
            return False

        plugin_name = None
        if value.HasField("metadata"):
            plugin_name = value.metadata.plugin_data.plugin_name

        return plugin_name == "hparams"

    def process(self, event: event_pb2.Event, value: Any) -> None:
        if self.exporter:
            self.exporter.save_data(event, value)

    def finalize(self) -> None:
        if self.exporter:
            self.exporter.finalize()


class PRCurveProcessor(ExportProcessor):
    """Process precision-recall curve data using PRCurveExporter."""

    def __init__(self, config: ExportConfig):
        super().__init__(config)
        self.exporter: PRCurveExporter | None = None
        if "pr_curve" in config.enabled_types:
            self.exporter = PRCurveExporter(config.output_path, config.digits)

    def can_handle(self, event: event_pb2.Event, value: Any) -> bool:
        if "pr_curve" not in self.config.enabled_types:
            return False

        return value.HasField("tensor") and TensorDataDetector.is_pr_curve_tensor(
            value.tensor, value.tag
        )

    def process(self, event: event_pb2.Event, value: Any) -> None:
        if self.exporter:
            self.exporter.save_data(event, value)


class ExportPipeline:
    """Modular export pipeline for TensorBoard data."""

    def __init__(self, config: ExportConfig):
        self.config = config
        self.processors: list[ExportProcessor] = []
        self._setup_processors()

    def _setup_processors(self) -> None:
        """Initialize processors based on enabled types."""
        # Add all available processors
        self.processors.extend(
            [
                ScalarProcessor(self.config),
                ImageVideoProcessor(self.config),
                HistogramProcessor(self.config),
                AudioProcessor(self.config),
                TextProcessor(self.config),
                MeshProcessor(self.config),
                HyperparameterProcessor(self.config),
                PRCurveProcessor(self.config),
            ]
        )

    def process_events(self, events: Iterator[event_pb2.Event]) -> int:
        """Process all events through the pipeline.

        Args:
            events: Iterator of TensorBoard events

        Returns:
            Number of events processed
        """
        event_count = 0

        for event in events:
            event_count += 1
            self._process_single_event(event)

        # Finalize all processors
        for processor in self.processors:
            processor.finalize()

        return event_count

    def _process_single_event(self, event: event_pb2.Event) -> None:
        """Process a single event through appropriate processors."""
        if event.HasField("summary"):
            for value in event.summary.value:
                self._process_value(event, value)

    def _process_value(self, event: event_pb2.Event, value: Any) -> None:
        """Process a single summary value through appropriate processor."""
        for processor in self.processors:
            if processor.can_handle(event, value):
                try:
                    processor.process(event, value)
                    return  # Only one processor should handle each value
                except Exception as e:
                    logger.error(
                        f"Error processing {value.tag} with {processor.__class__.__name__}: {e}"
                    )

        # If no processor handled it, log a warning
        plugin_name = None
        if value.HasField("metadata"):
            plugin_name = value.metadata.plugin_data.plugin_name

        logger.debug(
            f"No processor found for tag '{value.tag}' (plugin: {plugin_name})"
        )
