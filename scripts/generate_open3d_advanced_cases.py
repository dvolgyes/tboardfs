#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy==2.4.6",
#   "open3d-cpu[ml]==0.20.0",
#   "protobuf==7.36.2",
#   "tensorboard==2.21.0",
#   "tensorflow-cpu==2.21.0",
# ]
# ///
"""Generate native Open3D detection, segmentation, lines, and material logs."""

import argparse
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from open3d.ml.vis import BoundingBox3D
from open3d.t.geometry import LineSet, PointCloud, TriangleMesh
from open3d.visualization.tensorboard_plugin import plugin_data_pb2, summary
from open3d.visualization.tensorboard_plugin.util import Open3DPluginDataReader
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import masked_crc32c
from tensorboard.summary.writer.record_writer import RecordWriter
import tensorflow as tf

PLUGIN_DIR = Path("plugins", "Open3D")
EVENT_NAME = "events.out.tfevents.open3d-advanced"
TAGS = ("detections", "line_set", "segmentation", "textured_mesh")
STEP = 5


class _Cases:
    """Geometry inputs and independent checks of native reader output."""

    boxes: list[BoundingBox3D]
    labels: dict[int, str]
    lines: dict[str, object]
    cloud: dict[str, object]
    mesh: dict[str, object]

    def __init__(self) -> None:
        """Build four small examples from fixed integer or dyadic inputs."""
        self.labels = {2: "marker", 7: "crate"}
        self.boxes = [
            BoundingBox3D(
                center, (0, 0, 1), (0, 1, 0), (1, 0, 0), size, label, confidence
            )
            for center, size, label, confidence in (
                ((0, 0, 0), (2, 2, 2), 2, 0.875),
                ((4, 0, 1), (4, 2, 2), 7, 0.625),
            )
        ]
        positions = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0]], dtype=np.float32)
        self.lines = {
            "vertex_positions": positions,
            "line_indices": np.array([[0, 1], [1, 2]], dtype=np.int32),
            "line_colors": np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8),
        }
        self.cloud = {
            "vertex_positions": positions,
            "vertex_labels": np.array([[2], [7], [2]], dtype=np.int32),
            "vertex_scores": np.array(
                [[0.75, 0.25], [0.125, 0.875], [1, 0]], dtype=np.float32
            ),
        }
        self.mesh = {
            "vertex_positions": positions,
            "triangle_indices": np.array([[0, 1, 2]], dtype=np.int32),
            "triangle_texture_uvs": np.array(
                [[[0, 0], [1, 0], [0, 1]]], dtype=np.float32
            ),
            "material_name": "defaultLit",
            "material_scalar_metallic": 0.25,
            "material_vector_base_color": np.array([1, 0.5, 0.25, 1], dtype=np.float32),
            "material_texture_map_albedo": np.array(
                [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]],
                dtype=np.uint8,
            ),
        }

    def write(self, source: Path) -> None:
        """Use the official TensorFlow summary API and close native sidecars."""
        with (
            closing(tf.summary.create_file_writer(str(source))) as writer,
            writer.as_default(),
        ):
            for tag, data in zip(
                TAGS,
                ({"bboxes": self.boxes}, self.lines, self.cloud, self.mesh),
                strict=True,
            ):
                names = self.labels if tag in {"detections", "segmentation"} else None
                if not summary.add_3d(
                    tag, data, step=STEP, logdir=str(source), label_to_names=names
                ):
                    raise RuntimeError(f"TensorFlow skipped writing Open3D tag {tag}")
        native_writer = summary._async_data_writer
        native_writer._writer_thread.join()
        for handle in native_writer._file_handles.values():
            handle.close()

    def _check_detection(
        self, geometry: object, aux: plugin_data_pb2.InferenceData
    ) -> None:
        """Verify both native box geometries and both inference records."""
        if not isinstance(geometry, LineSet):
            raise TypeError("Bounding boxes did not reopen as a LineSet")
        expected = BoundingBox3D.create_lines(self.boxes, out_format="dict")
        np.testing.assert_array_equal(
            geometry.point.positions.numpy(), expected["vertex_positions"]
        )
        np.testing.assert_array_equal(
            geometry.line.indices.numpy(), expected["line_indices"]
        )
        actual = [(item.label, item.confidence) for item in aux.inference_result]
        if actual != [(2, 0.875), (7, 0.625)]:
            raise ValueError(f"Unexpected Open3D box inference data: {actual}")

    def _check_mesh(self, geometry: object) -> None:
        """Check exact triangle indices, UVs and PBR material after reopening."""
        if not isinstance(geometry, TriangleMesh):
            raise TypeError("Expected a native textured TriangleMesh")
        np.testing.assert_array_equal(
            geometry.vertex.positions.numpy(), self.mesh["vertex_positions"]
        )
        np.testing.assert_array_equal(
            geometry.triangle.indices.numpy(), self.mesh["triangle_indices"]
        )
        np.testing.assert_array_equal(
            geometry.triangle.texture_uvs.numpy(), self.mesh["triangle_texture_uvs"]
        )
        material = geometry.material
        if (
            material.material_name != "defaultLit"
            or material.scalar_properties["metallic"] != 0.25
        ):
            raise ValueError(f"Unexpected Open3D material: {material}")
        np.testing.assert_array_equal(
            material.vector_properties["base_color"],
            self.mesh["material_vector_base_color"],
        )
        np.testing.assert_array_equal(
            material.texture_maps["albedo"].as_tensor().numpy(),
            self.mesh["material_texture_map_albedo"],
        )

    def _check_line_and_cloud(self, line: object, cloud: object) -> None:
        """Check standalone line colors and per-point custom segmentation attributes."""
        if not isinstance(line, LineSet) or not isinstance(cloud, PointCloud):
            raise TypeError("Expected native LineSet and PointCloud geometry")
        np.testing.assert_array_equal(
            line.point.positions.numpy(), self.lines["vertex_positions"]
        )
        np.testing.assert_array_equal(
            line.line.indices.numpy(), self.lines["line_indices"]
        )
        np.testing.assert_array_equal(
            line.line.colors.numpy(), self.lines["line_colors"]
        )
        np.testing.assert_array_equal(
            cloud.point.positions.numpy(), self.cloud["vertex_positions"]
        )
        np.testing.assert_array_equal(
            cloud.point.labels.numpy(), self.cloud["vertex_labels"]
        )
        np.testing.assert_array_equal(
            cloud.point.scores.numpy(), self.cloud["vertex_scores"]
        )

    def verify(self, output: Path) -> None:
        """Independently reopen and verify the event indexes and all properties."""
        reader = Open3DPluginDataReader(str(output))
        if reader.run_to_tags != {".": list(TAGS)}:
            raise ValueError("Open3D did not discover every advanced geometry tag")
        geometry = {}
        for tag, events in reader.tensor_events(".").items():
            (event,) = events
            if event.step != STEP:
                raise ValueError(f"Unexpected step for {tag}: {event.step}")
            data = plugin_data_pb2.Open3DPluginData.FromString(
                event.tensor_proto.string_val[0]
            )
            _check_ranges(output, tag, data)
            value, aux = reader.read_geometry(".", tag, STEP, 0, {STEP: 0})
            if tag == "detections":
                self._check_detection(value, aux)
            if (
                tag in {"detections", "segmentation"}
                and reader.get_label_to_names(".", tag) != self.labels
            ):
                raise ValueError(f"Missing label definitions for {tag}")
            geometry[tag] = value
        self._check_line_and_cloud(geometry["line_set"], geometry["segmentation"])
        self._check_mesh(geometry["textured_mesh"])
        reader.reload_events()


def _check_ranges(
    output: Path, tag: str, data: plugin_data_pb2.Open3DPluginData
) -> None:
    """Verify that native geometry and aux checksums cover the entire sidecar."""
    name = f"{tag}.msgpack"
    if (
        data.version != 14
        or data.batch_index.filename != name
        or data.property_references
    ):
        raise ValueError(f"Unexpected Open3D event schema for {tag}")
    (entry,) = data.batch_index.start_size
    contents = (output / PLUGIN_DIR / name).read_bytes()
    if entry.start or not entry.size or entry.size + entry.aux_size != len(contents):
        raise ValueError(f"Unexpected sidecar offsets or sizes for {tag}")
    if masked_crc32c(contents[: entry.size]) != entry.masked_crc32c:
        raise ValueError(f"Invalid geometry checksum for {tag}")
    if tag == "detections":
        if not entry.aux_size or entry.aux_start != entry.size:
            raise ValueError("Open3D detection aux range is missing or not contiguous")
        if masked_crc32c(contents[entry.aux_start :]) != entry.aux_masked_crc32c:
            raise ValueError("Invalid Open3D detection aux checksum")
    elif entry.aux_start or entry.aux_size or entry.aux_masked_crc32c:
        raise ValueError(f"Unexpected Open3D aux data for {tag}")


def _normalize(source: Path, output: Path) -> None:
    """Remove volatile filename components and wall times, retaining native payloads."""
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


def main() -> None:
    """Generate and reopen a single Open3D TensorFlow run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", required=True, type=Path)
    output: Path = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")
    cases = _Cases()
    with TemporaryDirectory(dir=output) as temporary:
        source = Path(temporary)
        cases.write(source)
        _normalize(source, output)
    cases.verify(output)


if __name__ == "__main__":
    main()
