"""TensorBoardX input conveniences become ordinary image, scalar and graph events."""

import json
import struct
from hashlib import sha256
from pathlib import Path

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "tests/data/producer-cases/writer-inputs"


def test_add_scalars_keeps_identical_tags_in_separate_runs() -> None:
    fs = TensorBoardFS(RUN)
    assert fs.readdir("/scalars")[2:] == []
    assert fs.readdir("/shared")[2:] == ["left", "right"]
    for subrun, expected in (("left", 1.25), ("right", 2.5)):
        path = f"/shared/{subrun}/scalars/shared.json"
        rows = json.loads(fs.read(path, 4096, 0))
        assert [(row["step"], row["value"]) for row in rows] == [(3, expected)]


def test_image_grid_and_box_are_baked_into_single_rasters() -> None:
    """Compare both PNGs to the generator's pixel-verified references."""
    fs = TensorBoardFS(RUN)
    for path, size, digest in (
        (
            "grid/000004",
            (10, 4),
            "19c712f3de57a9e9c500fc3a13c46517c9ade946d4626bdefd8edecaaa281f53",
        ),
        (
            "boxed/000005",
            (10, 8),
            "a5a9533c7dd317d0189ae5380c850c7228693ec34b99e5cc1bb8645b29aec501",
        ),
    ):
        data = fs.read(f"/images/{path}.png", 4096, 0)
        assert struct.unpack_from(">II", data, 16) == size
        assert sha256(data).hexdigest() == digest


def test_existing_matplotlib_figure_is_a_raster_not_a_new_type() -> None:
    fs = TensorBoardFS(ROOT / "test-logs/tensorboardx")
    data = fs.read("/images/figure/curve/000001.png", 65536, 0)
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert struct.unpack_from(">IIBB", data, 16) == (200, 200, 8, 2)


def test_onnx_and_openvino_inputs_become_graphdefs() -> None:
    fs = TensorBoardFS(RUN)
    expected = [
        [
            ("input", "Variable", []),
            ("output", "Variable", []),
            ("activated", "Relu", ["input"]),
            ("output", "Identity", ["activated"]),
        ],
        [("positive", "op", ["features"]), ("prediction", "op", ["positive"])],
    ]
    for index, nodes in enumerate(expected):
        data = fs.read(f"/graphs/graph_{index}.pb", 4096, 0)
        graph = compat_pb2.GraphDef.FromString(data)
        assert [(node.name, node.op, list(node.input)) for node in graph.node] == nodes
    assert fs.read("/graphs/graph.pb", 4096, 0) == fs.read(
        "/graphs/graph_1.pb", 4096, 0
    )
