"""Check native file eligibility independently of other tensor members."""

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from tboardfs import TensorBoardFS
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
