"""Check native file eligibility independently of other tensor members."""

from io import BytesIO
from pathlib import Path

import magic
import numpy as np
import pytest

from tboardfs import TensorBoardFS, parse_file
from tboardfs.proto import compat_pb2
from test_incremental_parse import _files, _frame


def _npy(count: int) -> bytes:
    data = BytesIO()
    np.save(data, np.arange(count, dtype="<i4"), allow_pickle=False)
    return data.getvalue()


@pytest.mark.parametrize("layout", ["single", "small-pair", "mixed"])
def test_embedded_strings_keep_original_bytes_and_indices(
    tmp_path: Path, layout: str
) -> None:
    small = _npy(4)
    large = _npy(512)
    items, expected = {
        "single": ([small], {"000007.npy": small}),
        "small-pair": ([small, small], {"000007_0.npy": small, "000007_1.npy": small}),
        "mixed": (
            [b"site A", small, b"\x00\x01", large],
            {"000007_1.npy": small, "000007_3.npy": large},
        ),
    }[layout]
    event = compat_pb2.Event(step=7)
    tensor = event.summary.value.add(tag="samples").tensor
    tensor.dtype = compat_pb2.DT_STRING
    tensor.tensor_shape.dim.add(size=len(items))
    tensor.string_val.extend(items)
    frame = _frame(event)
    (tmp_path / "events.out.tfevents.files").write_bytes(frame)
    fs = TensorBoardFS(tmp_path)
    files = list(_files(fs))
    exported = {
        p.split("/")[-1]: fs.read(p, 8192, 0)
        for p in files
        if p.startswith("/tensors/samples/")
    }
    assert exported == expected
    assert any(p.startswith("/text/samples/") for p in files)
    assert (
        fs.read(next(p for p in files if p.startswith("/raw/")), len(frame), 0) == frame
    )


def test_small_uint8_file_keeps_distinct_outer_array(tmp_path: Path) -> None:
    inner = _npy(5)
    event = compat_pb2.Event(step=7)
    tensor = event.summary.value.add(tag="wrapped").tensor
    tensor.dtype = compat_pb2.DT_UINT8
    tensor.tensor_shape.dim.add(size=len(inner))
    tensor.tensor_content = inner
    (tmp_path / "events.out.tfevents.files").write_bytes(_frame(event))
    fs = TensorBoardFS(tmp_path)
    outer = np.load(
        BytesIO(fs.read("/tensors/wrapped/000007.npy", 4096, 0)), allow_pickle=False
    )
    assert outer.dtype == np.uint8 and outer.tobytes() == inner
    assert fs.read("/tensors/wrapped/000007.embedded.npy", 4096, 0) == inner


@pytest.mark.parametrize(
    ("dtype", "width", "first", "revised"),
    [
        (compat_pb2.DT_FLOAT, 4, "bin", "bin"),
        (compat_pb2.DT_DOUBLE, 8, "png", "gif"),
        (compat_pb2.DT_UINT8, 1, "png", "gif"),
    ],
)
def test_raw_tensor_file_type_under_custom_magic_rules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dtype: int,
    width: int,
    first: str,
    revised: str,
) -> None:
    """Only packed float32 ignores custom rules that match beyond the prefix."""
    offset = 65540
    marker = b"UNIQUE-END"
    rules = tmp_path / "magic"
    rules.write_text(
        f"{offset} string UNIQUE-END png image custom late-offset control\n"
        "!:mime application/octet-stream\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MAGIC", str(rules))
    monkeypatch.setattr(magic, "_instances", {})
    opaque = bytes(1 << 20)
    recognized = opaque[:offset] + marker + opaque[offset + len(marker) :]
    path = tmp_path / "events.out.tfevents.types"
    with path.open("wb") as stream:
        for step, content in enumerate((opaque, opaque, recognized, recognized)):
            event = compat_pb2.Event(step=step)
            tensor = event.summary.value.add(tag="typed").tensor
            tensor.dtype = dtype
            tensor.tensor_shape.dim.add(size=len(content) // width)
            tensor.tensor_content = content
            frame = _frame(event)
            stream.write(frame)

    assert [entry.extension for entry in parse_file(path)["binary_entries"]] == [
        "bin",
        "bin",
        first,
        first,
    ]

    rules.write_text(
        f"{offset} string UNIQUE-END gif image alternate\n", encoding="utf-8"
    )
    monkeypatch.setattr(magic, "_instances", {})
    path.write_bytes(frame)
    assert [entry.extension for entry in parse_file(path)["binary_entries"]] == [
        revised,
    ]
