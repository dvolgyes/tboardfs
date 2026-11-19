"""A one-value tensor of any real NumPy dtype is a scalar, not a tensor."""

import io
import json
from pathlib import Path

import numpy as np
import pytest

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2

from test_incremental_parse import _frame


def _mount(tmp_path: Path, tensor: compat_pb2.TensorProto) -> TensorBoardFS:
    event = compat_pb2.Event(step=1, wall_time=3.0)
    event.summary.value.add(tag="x").tensor.CopyFrom(tensor)
    (tmp_path / "events.out.tfevents.1.host").write_bytes(_frame(event))
    return TensorBoardFS(tmp_path, scalar_formats="json,npz")


@pytest.mark.parametrize(
    ("dtype", "field", "item", "value", "np_dtype"),
    [
        (19, "half_val", 0x3C00, 1.0, "float16"),
        (14, "half_val", 0x3FC0, 1.5, "float32"),
        (6, "int_val", -3, -3, "int8"),
        (4, "int_val", 200, 200, "uint8"),
        (5, "int_val", -300, -300, "int16"),
        (17, "int_val", 60000, 60000, "uint16"),
        (2, "double_val", 1 + 2**-50, 1 + 2**-50, "float64"),
        (9, "int64_val", -(2**60 + 1), -(2**60 + 1), "int64"),
        (23, "uint64_val", 2**64 - 1, 2**64 - 1, "uint64"),
        (10, "bool_val", True, True, "bool"),
        (13, "int_val", -(2**31), -(2**31), "int32"),
    ],
)
def test_one_value_numeric_tensor_keeps_its_scalar_precision(
    tmp_path: Path,
    dtype: int,
    field: str,
    item: float | int,
    value: float | int,
    np_dtype: str,
) -> None:
    tensor = compat_pb2.TensorProto(dtype=dtype)
    getattr(tensor, field).append(item)
    fs = _mount(tmp_path, tensor)

    rows = json.loads(fs.read("/scalars/x.json", 1 << 20, 0))
    assert [row["value"] for row in rows] == [value]
    assert type(rows[0]["value"]) is type(value)
    with np.load(io.BytesIO(fs.read("/scalars/x.npz", 1 << 20, 0))) as archive:
        assert archive["value"].dtype == np.dtype(np_dtype)
        assert archive["value"].tolist() == [value]
    assert fs.readdir("/tensors")[2:] == []


def test_a_one_value_complex_tensor_stays_a_tensor(tmp_path: Path) -> None:
    """A complex number has no place in the scalar table."""
    tensor = compat_pb2.TensorProto(dtype=8)
    tensor.scomplex_val.extend([1.0, 2.0])
    fs = _mount(tmp_path, tensor)

    assert fs.readdir("/scalars")[2:] == []
    assert fs.readdir("/tensors")[2:] == ["x"]
