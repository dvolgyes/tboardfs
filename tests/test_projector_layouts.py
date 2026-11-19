"""Resolve both projector layouts from the exported configs without producer imports."""

import re
import struct
from pathlib import Path

import pytest

from tboardfs import TensorBoardFS

ROOT = Path(__file__).parent / "data/producer-cases/projector-layouts"


@pytest.mark.parametrize("variant", ["historical", "binary"])
def test_projector_sidecars(variant: str) -> None:
    run = ROOT / variant
    assets = run
    if variant == "historical":
        assert not (run / "projector_config.pbtxt").exists()
        assets /= "plugins/org_tensorflow_tensorboard_projector"
    fs = TensorBoardFS(run)

    def read(name: str) -> bytes:
        path = assets / name
        virtual = f"/projector/{path.relative_to(run)}"
        data = fs.read(virtual, 4096, 0)
        assert data == path.read_bytes()
        return data

    config = read("projector_config.pbtxt").decode()
    (tensor,) = re.findall(r'tensor_path: "([^"]+)"', config)
    (labels,) = re.findall(r'metadata_path: "([^"]+)"', config)
    assert re.findall(r"tensor_shape: (\d+)", config) == ["3", "2"]
    header, *names = read(labels).decode().splitlines()
    assert header == "species\thabitat"
    data = read(tensor)
    if variant == "historical":
        coordinates = [
            tuple(map(float, line.split(b"\t"))) for line in data.splitlines()
        ]
    else:
        coordinates = list(struct.iter_unpack("<ff", data))
    assert list(zip(names, coordinates, strict=True)) == [
        ("alder\twetland", (-0.5, 0.875)),
        ("birch\tupland", (0.25, -0.5)),
        ("willow\twetland", (-0.625, 1.0)),
    ]
