#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["tensorboardX==2.6.5", "numpy==2.2.6", "pillow==11.3.0", "onnx==1.19.1", "openvino==2026.4.1"]
# ///
"""Generate TensorBoardX convenience-writer inputs, not new event wire types.

https://github.com/lanpa/tensorboardX/tree/v2.6.5/tensorboardX
The ONNX and OpenVINO importers convert model files to GraphDef; OpenVINO
retains edge endpoints but reports a generic op. add_figure renders a PNG
via add_image and is already exercised by generate_test_logs.py.
"""

import argparse
import sys
from contextlib import redirect_stdout
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import openvino as ov
from onnx import TensorProto, helper
from PIL import Image
from tensorboardX import SummaryWriter
from tensorboardX.proto.event_pb2 import Event


def _onnx_model(path: Path) -> None:
    """Write a small real ONNX graph with a named edge and output."""
    nodes = [
        helper.make_node("Relu", ["input"], ["activated"], name="relu_node"),
        helper.make_node("Identity", ["activated"], ["output"], name="output_node"),
    ]
    graph = helper.make_graph(
        nodes,
        "writer_input_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
    )
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), path
    )


def _verify_images(event_path: Path) -> None:
    """Decode the emitted PNGs and compare every pixel with the expected tiles and box."""
    grid = np.empty((4, 10, 3), dtype=np.uint8)
    grid[:, :5] = (255, 0, 0)
    grid[:, 5:] = (0, 0, 128)
    boxed = np.full((8, 10, 3), (0, 96, 0), dtype=np.uint8)
    boxed[1:8, 1:7] = (255, 0, 0)
    boxed[3:7, 3:5] = (0, 96, 0)
    expected = {"grid": grid, "boxed": boxed}
    with event_path.open("rb") as stream:
        while header := stream.read(12):
            record = stream.read(int.from_bytes(header[:8], "little"))
            stream.read(4)
            for value in Event.FromString(record).summary.value:
                if value.HasField("image"):
                    with Image.open(BytesIO(value.image.encoded_image_string)) as image:
                        np.testing.assert_array_equal(
                            np.asarray(image), expected.pop(value.tag)
                        )
    if expected:
        raise ValueError(f"Missing generated images: {sorted(expected)}")


def main() -> None:
    """Write real convenience-writer records into an empty output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")

    images = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    images[0, 0] = 255
    images[1, 2] = 128
    boxes_image = np.zeros((3, 8, 10), dtype=np.uint8)
    boxes_image[1] = 96
    boxes = np.array([[1, 2, 6, 7]], dtype=np.float32)

    with TemporaryDirectory() as temp, SummaryWriter(str(output)) as writer:
        writer.add_scalars("shared", {"left": 1.25, "right": 2.5}, 3, walltime=10)
        writer.add_images("grid", images, 4, walltime=11, dataformats="NCHW")
        writer.add_image_with_boxes(
            "boxed",
            boxes_image,
            boxes,
            5,
            walltime=12,
            dataformats="CHW",
        )
        model_path = Path(temp) / "input.onnx"
        _onnx_model(model_path)
        with redirect_stdout(sys.stderr):
            writer.add_onnx_graph(str(model_path))
        x = ov.opset13.parameter([1, 2], np.float32, name="features")
        relu = ov.opset13.relu(x, name="positive")
        result = ov.opset13.result(relu, name="prediction")
        xml_path = Path(temp) / "input.xml"
        ov.serialize(ov.Model([result], [x]), xml_path)
        with redirect_stdout(sys.stderr):
            writer.add_openvino_graph(str(xml_path))

    for event in output.rglob("events.out.tfevents.*"):
        event.rename(event.with_name("events.out.tfevents.writer-inputs"))
    _verify_images(output / "events.out.tfevents.writer-inputs")


if __name__ == "__main__":
    main()
