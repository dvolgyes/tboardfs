#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy==2.4.6",
#   "open3d-cpu==0.20.0",
#   "protobuf==7.36.2",
#   "tensorboard==2.21.0",
#   "torch @ https://download.pytorch.org/whl/cpu/torch-2.15.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl",
# ]
# ///
"""Generate and reopen native Open3D TensorBoard geometry on Linux x86-64."""

import argparse
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from numpy.typing import NDArray
from open3d.t.geometry import PointCloud, TriangleMesh
from open3d.visualization.tensorboard_plugin import plugin_data_pb2, summary
from open3d.visualization.tensorboard_plugin.util import Open3DPluginDataReader
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
from tensorboard.summary.writer.record_writer import RecordWriter
from torch.utils.tensorboard import SummaryWriter

TAGS = ("colored_cloud", "triangle_mesh")
EVENT_NAME = "events.out.tfevents.open3d"
PLUGIN_DIR = Path("plugins", "Open3D")
BATCHES = 2

type _GeometryInput = NDArray[np.float32] | NDArray[np.uint8] | NDArray[np.int32] | int


def _normalize(source: Path, output: Path) -> None:
    """Keep the native geometry and indexes; normalize names and wall times."""
    (events_path,) = source.glob("events.out.tfevents.*")
    plugin_dir = output / PLUGIN_DIR
    plugin_dir.mkdir(parents=True)
    names = {}
    for tag in TAGS:
        (sidecar,) = (source / PLUGIN_DIR).glob(f"{tag}.*.msgpack")
        names[sidecar.name] = f"{tag}.msgpack"
        sidecar.rename(plugin_dir / names[sidecar.name])
    with (output / EVENT_NAME).open("wb") as handle:
        writer = RecordWriter(handle)
        for event in LegacyEventFileLoader(str(events_path)).Load():
            event.wall_time = 1_700_000_000 + event.step
            for value in event.summary.value:
                data = plugin_data_pb2.Open3DPluginData.FromString(
                    value.tensor.string_val[0]
                )
                data.batch_index.filename = names[data.batch_index.filename]
                value.tensor.string_val[0] = data.SerializeToString(deterministic=True)
            writer.write(event.SerializeToString(deterministic=True))


@dataclass(frozen=True)
class _Geometry:
    """Expected geometry for both batches and both steps."""

    positions: NDArray[np.float32]
    colors: tuple[NDArray[np.uint8], NDArray[np.uint8]]
    faces: tuple[NDArray[np.int32], NDArray[np.int32]]

    def _check_geometry(
        self, geometry: object, tag: str, step: int, index: int
    ) -> None:
        """Compare the reconstruction to the data passed to the writer."""
        if tag == "triangle_mesh" and isinstance(geometry, TriangleMesh):
            vertices = geometry.vertex
            np.testing.assert_array_equal(
                geometry.triangle.indices.numpy(), self.faces[step - 1][index]
            )
        elif tag == "colored_cloud" and isinstance(geometry, PointCloud):
            vertices = geometry.point
        else:
            raise TypeError(
                f"Unexpected Open3D geometry for {tag}: {type(geometry).__name__}"
            )
        np.testing.assert_array_equal(vertices.positions.numpy(), self.positions[index])
        np.testing.assert_array_equal(
            vertices.colors.numpy(), self.colors[step - 1][index]
        )

    @staticmethod
    def _check_index(
        output: Path, tag: str, data: tuple[plugin_data_pb2.Open3DPluginData, ...]
    ) -> None:
        """Ensure references and batch ranges cover the native sidecar."""
        name = f"{tag}.msgpack"
        entries: list[plugin_data_pb2.Open3DPluginData.StartSize] = []
        for step, message in enumerate(data, start=1):
            expected = [] if step == 1 else [(0, 1)]
            refs = [
                (ref.geometry_property, ref.step_ref)
                for ref in message.property_references
            ]
            if message.version != 14 or refs != expected:
                raise ValueError(
                    f"Unexpected Open3D schema or property references for {tag}"
                )
            batch = message.batch_index
            if batch.filename != name or len(batch.start_size) != BATCHES:
                raise ValueError(f"Unexpected Open3D batch index for {tag}")
            entries.extend(batch.start_size)
        offsets = list(accumulate((entry.size for entry in entries), initial=0))
        if [entry.start for entry in entries] != offsets[:-1]:
            raise ValueError(f"Unexpected Open3D batch offsets for {tag}")
        if offsets[-1] != (output / PLUGIN_DIR / name).stat().st_size:
            raise ValueError(f"Unindexed Open3D sidecar bytes for {tag}")

    def verify(self, output: Path) -> None:
        """Reopen native data and validate geometry with CRC and reference resolution."""
        reader = Open3DPluginDataReader(str(output))
        if reader.run_to_tags != {".": list(TAGS)}:
            raise ValueError("Open3D did not discover both fixture tags")
        for tag, events in reader.tensor_events(".").items():
            steps = {event.step: index for index, event in enumerate(events)}
            if len(events) != 2 or steps != {1: 0, 2: 1}:
                raise ValueError(f"Unexpected Open3D steps for {tag}: {steps}")
            data = tuple(
                plugin_data_pb2.Open3DPluginData.FromString(
                    event.tensor_proto.string_val[0]
                )
                for event in events
            )
            self._check_index(output, tag, data)
            for step in (2, 1):
                for index in range(BATCHES):
                    geometry, _ = reader.read_geometry(".", tag, step, index, steps)
                    self._check_geometry(geometry, tag, step, index)
        reader.reload_events()


def main() -> None:
    """Write four native Open3D summaries with two geometry items each."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    output: Path = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")

    square = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    positions = np.stack((square, square + np.array([2, 0, 1], dtype=np.float32)))
    palette = np.array(
        [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]], dtype=np.uint8
    )
    colors = (np.stack((palette, palette[::-1])), np.stack((palette[::-1], palette)))
    first = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    second = np.array([[0, 1, 3], [1, 2, 3]], dtype=np.int32)
    faces = (np.stack((first, second)), np.stack((second, first)))
    with TemporaryDirectory(dir=output) as temporary:
        with SummaryWriter(temporary) as writer:
            for tag in TAGS:
                for step in (1, 2):
                    data: dict[str, _GeometryInput] = {
                        "vertex_positions": positions if step == 1 else 1,
                        "vertex_colors": colors[step - 1],
                    }
                    if tag == "triangle_mesh":
                        data["triangle_indices"] = faces[step - 1]
                    writer.add_3d(tag, data, step=step, max_outputs=BATCHES)
        native_writer = summary._async_data_writer
        native_writer._writer_thread.join()
        for handle in native_writer._file_handles.values():
            handle.close()
        _normalize(Path(temporary), output)
    _Geometry(positions, colors, faces).verify(output)


if __name__ == "__main__":
    main()
