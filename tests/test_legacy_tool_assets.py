"""Distinguish native Beholder arrays from an illustrative legacy TPU profile."""

import json
from pathlib import Path
from pickletools import genops

import numpy as np

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2

RUN = Path(__file__).parent / "data/producer-cases/legacy-tool-assets"
BEHOLDER = RUN / "plugins/beholder"


def _literal_maps(name: str) -> list[dict[str | int, str | int]]:
    """Read flat key/value operands, including memoized strings, without unpickling."""
    mappings = []
    operands: list[str | int] = []
    strings: dict[int, str] = {}
    for op, arg, _ in genops((BEHOLDER / name).read_bytes()):
        if op.name in {"BINUNICODE", "BININT1"}:
            operands.append(arg)
        elif op.name == "NEWFALSE":
            operands.append(False)
        elif op.name == "BINPUT" and operands and isinstance(operands[-1], str):
            strings[arg] = operands[-1]
        elif op.name == "BINGET":
            operands.append(strings[arg])
        elif op.name == "SETITEMS":
            mappings.append(dict(zip(operands[::2], operands[1::2], strict=True)))
            operands.clear()
        else:
            assert op.name in {
                "PROTO",
                "EMPTY_DICT",
                "EMPTY_LIST",
                "BINPUT",
                "MARK",
                "APPENDS",
                "STOP",
            }
    return mappings


def test_beholder_arrays_config_and_section_statistics() -> None:
    (config,) = _literal_maps("config.pkl")
    assert config == {
        "values": "arrays",
        "mode": "current",
        "scaling": "layer",
        "colormap": "grayscale",
        "FPS": 10,
        "window_size": 15,
        "is_recording": False,
        "show_all": False,
    }
    assert _literal_maps("section-info.pkl") == [
        {
            "name": 0,
            "shape": "(2, 3)",
            "min": "-3.000e+00",
            "mean": "3.333e-01",
            "max": "3.000e+00",
            "range": "6.000e+00",
            "height": 128,
        },
        {
            "name": 1,
            "shape": "(8,)",
            "min": "0.000e+00",
            "mean": "3.500e+00",
            "max": "7.000e+00",
            "range": "7.000e+00",
            "height": 128,
        },
    ]


def test_native_frame_contains_both_sections_and_padding() -> None:
    (value,) = compat_pb2.Summary.FromString(
        (BEHOLDER / "frame.summary").read_bytes()
    ).value
    assert value.tag == "beholder-frame"
    shape = tuple(dim.size for dim in value.tensor.tensor_shape.dim)
    assert shape == (261, 768, 1)
    pixels = np.frombuffer(value.tensor.tensor_content, np.uint8).reshape(shape)
    np.testing.assert_array_equal(pixels[0, ::256, 0], [0, 127, 255])
    np.testing.assert_array_equal(pixels[127, ::256, 0], [170, 85, 212])
    assert not pixels[128:133].any()
    np.testing.assert_array_equal(
        pixels[133, ::96, 0], [0, 36, 72, 109, 145, 182, 218, 255]
    )


def test_beholder_ancillaries_are_not_events_or_typed_assets() -> None:
    fs = TensorBoardFS(RUN, raw=True)
    assert {"plugins", "tensors", "images"}.isdisjoint(fs.readdir("/"))
    assert fs.readdir("/raw")[2:] == ["events.out.tfevents.legacy-tools"]


def test_schema_only_op_profile_passes_through_with_instruction_hierarchy() -> None:
    fs = TensorBoardFS(RUN)
    virtual = "/profile/schema-only/host.op_profile.json"
    content = fs.read(virtual, 8192, 0)
    assert content == (RUN / "plugins" / virtual.lstrip("/")).read_bytes()
    root = json.loads(content)["byProgram"]
    assert (root["name"], root["numChildren"]) == ("schema-only-program", 2)
    assert [
        (n["xla"]["op"], n["metrics"]["time"], n["metrics"]["rawTime"])
        for n in root["children"]
    ] == [
        ("dot", 0.6, 1200),
        ("add", 0.4, 800),
    ]
