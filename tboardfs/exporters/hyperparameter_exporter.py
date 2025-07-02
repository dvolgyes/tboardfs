"""Hyperparameter data exporter for TensorBoard events."""

from pathlib import Path
from typing import Any
from tensorboard.compat.proto import event_pb2
from loguru import logger

from .base_exporter import BaseExporter

# Import hyperparameter protobuf definitions
try:
    from tensorboard.plugins.hparams import plugin_data_pb2 as hparams_pb2
    from google.protobuf.struct_pb2 import Value as protobuf_Value

    HPARAMS_AVAILABLE = True
except ImportError:
    HPARAMS_AVAILABLE = False
    hparams_pb2 = None
    protobuf_Value = Any  # type: ignore


class HyperparameterExporter(BaseExporter):
    """Export hyperparameter data from TensorBoard events."""

    def __init__(self, output_path: Path, digits: int = 6):
        """Initialize hyperparameter exporter."""
        super().__init__(output_path, digits)
        self.hyperparameters_data: dict[str, Any] = {}

    def save_data(self, event: event_pb2.Event, value: Any, **kwargs: Any) -> None:
        """Collect hyperparameter data from a TensorBoard event.

        Note: Hyperparameters are collected and exported in finalize().
        """
        if not HPARAMS_AVAILABLE:
            logger.warning("TensorBoard hparams plugin not available")
            return

        try:
            # Parse the hyperparameter data from metadata
            plugin_data = hparams_pb2.HParamsPluginData.FromString(
                value.metadata.plugin_data.content
            )

            if plugin_data.HasField("session_start_info"):
                session_info = plugin_data.session_start_info

                # Extract hyperparameters from protobuf Value objects
                session_hparams = {}
                for param_name, param_value in session_info.hparams.items():
                    session_hparams[param_name] = self._extract_protobuf_value(
                        param_value
                    )

                # Use tag as session identifier, or fall back to a counter
                session_key = value.tag or f"session_{len(self.hyperparameters_data)}"

                self.hyperparameters_data[session_key] = {
                    "hyperparameters": session_hparams,
                    "step": event.step,
                    "wall_time": event.wall_time,
                }

                # Add optional fields if present
                if session_info.model_uri:
                    self.hyperparameters_data[session_key]["model_uri"] = (
                        session_info.model_uri
                    )
                if session_info.monitor_url:
                    self.hyperparameters_data[session_key]["monitor_url"] = (
                        session_info.monitor_url
                    )
                if session_info.group_name:
                    self.hyperparameters_data[session_key]["group_name"] = (
                        session_info.group_name
                    )

        except Exception as e:
            logger.warning(
                f"Failed to collect hyperparameter data for tag '{value.tag}': {e}"
            )

    def finalize(self) -> None:
        """Export collected hyperparameters to hp_params/hp_params.yaml."""
        if not self.hyperparameters_data:
            return

        try:
            import yaml
        except ImportError:
            logger.error(
                "PyYAML not available. Cannot export hyperparameters to YAML format."
            )
            logger.info("Please install it with: pip install PyYAML")
            return

        hp_params_dir = self.output_path / "hp_params"
        hp_params_dir.mkdir(exist_ok=True)

        yaml_file = hp_params_dir / "hp_params.yaml"

        # Organize data for YAML export
        export_data = {}

        if len(self.hyperparameters_data) == 1:
            # Single session - export hyperparameters directly
            session_data = list(self.hyperparameters_data.values())[0]
            export_data = session_data["hyperparameters"]
        else:
            # Multiple sessions - export as nested structure
            export_data = {
                session_key: session_data["hyperparameters"]
                for session_key, session_data in self.hyperparameters_data.items()
            }

        try:
            with yaml_file.open("w") as f:
                yaml.dump(export_data, f, default_flow_style=False, sort_keys=True)

            logger.info(f"Exported hyperparameters to {yaml_file}")
            logger.debug(f"Hyperparameter sessions: {len(self.hyperparameters_data)}")

        except Exception as e:
            logger.error(f"Failed to write hyperparameters YAML file: {e}")

    def _extract_protobuf_value(self, proto_value: Any) -> Any:
        """Extract Python value from protobuf Value object."""
        if not HPARAMS_AVAILABLE:
            return None

        if proto_value.HasField("number_value"):
            return proto_value.number_value
        elif proto_value.HasField("string_value"):
            return proto_value.string_value
        elif proto_value.HasField("bool_value"):
            return proto_value.bool_value
        elif proto_value.HasField("list_value"):
            return [
                self._extract_protobuf_value(item)
                for item in proto_value.list_value.values
            ]
        elif proto_value.HasField("struct_value"):
            return {
                key: self._extract_protobuf_value(val)
                for key, val in proto_value.struct_value.fields.items()
            }
        else:
            # Return raw value if we can't decode it
            return str(proto_value)
