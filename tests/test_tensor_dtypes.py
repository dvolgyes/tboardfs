"""Tensor exports decode every numeric TensorProto dtype that NumPy can hold."""

import io
import json
from pathlib import Path

import numpy as np
import pytest

from tboardfs import TensorBoardFS
from tboardfs.proto import event_pb2

from test_incremental_parse import _files, _frame

# TensorProto field numbers from tensorflow/core/framework/tensor.proto.
SCOMPLEX_VAL = 9
DCOMPLEX_VAL = 12
HALF_VAL = 13
INT_VAL = 7


def _varint(number: int) -> bytes:
    out = bytearray()
    while True:
        byte, number = number & 0x7F, number >> 7
        out.append(byte | (0x80 if number else 0))
        if not number:
            return bytes(out)


def _packed(field: int, payload: bytes) -> bytes:
    """Encode a packed repeated field without the vendored descriptor."""
    return _varint(field << 3 | 2) + _varint(len(payload)) + payload


def _varints(values: list[int]) -> bytes:
    return b"".join(_varint(value & 0xFFFFFFFFFFFFFFFF) for value in values)


def _log(tmp_path: Path, dtype: int, shape: list[int], wire: bytes) -> TensorBoardFS:
    event = event_pb2.Event(wall_time=1.0, step=3)
    value = event.summary.value.add(tag="t")
    value.metadata.plugin_data.plugin_name = "tensors"
    value.tensor.dtype = dtype
    value.tensor.tensor_shape.dim.extend(
        event_pb2.TensorShapeDim(size=s) for s in shape
    )
    value.tensor.MergeFromString(wire)
    (tmp_path / "events.out.tfevents.host").write_bytes(_frame(event))
    return TensorBoardFS(tmp_path)


def _npy(fs: TensorBoardFS) -> np.ndarray:
    data: np.ndarray = np.load(
        io.BytesIO(fs.read("/tensors/t/000003.npy", 1 << 20, 0)), allow_pickle=False
    )
    return data


def _content(array: np.ndarray) -> bytes:
    tensor_content = 4
    return _packed(tensor_content, array.tobytes())


HALF = np.array([1.5, -2.0, 65504.0], dtype="<f2")
BF16_BITS = np.array([0x3FC0, 0xC000], dtype="<u2")  # 1.5, -2.0
COMPLEX = np.array([1 + 2j, -3.5 - 0.5j])

FIELD_CASES = [
    ("half", 19, _packed(HALF_VAL, _varints(HALF.view("<u2").tolist())), HALF),
    (
        "bfloat16",
        14,
        _packed(HALF_VAL, _varints(BF16_BITS.tolist())),
        np.array([1.5, -2.0], dtype="<f4"),
    ),
    ("int8", 6, _packed(INT_VAL, _varints([-128, 127])), np.array([-128, 127], "i1")),
    ("int16", 5, _packed(INT_VAL, _varints([-32768, 7])), np.array([-32768, 7], "<i2")),
    ("uint16", 17, _packed(INT_VAL, _varints([65535, 1])), np.array([65535, 1], "<u2")),
    (
        "complex64",
        8,
        _packed(SCOMPLEX_VAL, np.array([1, 2, -3.5, -0.5], "<f4").tobytes()),
        COMPLEX.astype("<c8"),
    ),
    (
        "complex128",
        18,
        _packed(DCOMPLEX_VAL, np.array([1, 2, -3.5, -0.5], "<f8").tobytes()),
        COMPLEX.astype("<c16"),
    ),
]


@pytest.mark.parametrize(
    ("dtype", "wire", "expected"),
    [case[1:] for case in FIELD_CASES],
    ids=[case[0] for case in FIELD_CASES],
)
def test_value_fields_decode_to_the_native_dtype(
    tmp_path: Path, dtype: int, wire: bytes, expected: np.ndarray
) -> None:
    """Values in the repeated fields keep their dtype; bfloat16 widens exactly."""
    data = _npy(_log(tmp_path, dtype, [expected.size], wire))

    assert data.dtype == expected.dtype
    np.testing.assert_array_equal(data, expected)


CONTENT_CASES = [
    ("half", 19, HALF, HALF),
    ("bfloat16", 14, BF16_BITS, np.array([1.5, -2.0], dtype="<f4")),
    ("int8", 6, np.array([-1, 2], "i1"), np.array([-1, 2], "i1")),
    ("complex64", 8, COMPLEX.astype("<c8"), COMPLEX.astype("<c8")),
    ("complex128", 18, COMPLEX.astype("<c16"), COMPLEX.astype("<c16")),
]


@pytest.mark.parametrize(
    ("dtype", "stored", "expected"),
    [case[1:] for case in CONTENT_CASES],
    ids=[case[0] for case in CONTENT_CASES],
)
def test_tensor_content_decodes_to_the_native_dtype(
    tmp_path: Path, dtype: int, stored: np.ndarray, expected: np.ndarray
) -> None:
    """Packed tensor_content bytes use the same dtype mapping."""
    data = _npy(_log(tmp_path, dtype, [stored.size], _content(stored)))

    assert data.dtype == expected.dtype
    np.testing.assert_array_equal(data, expected)


def test_complex_json_values_are_real_imaginary_pairs(tmp_path: Path) -> None:
    """Strict JSON has no complex type, so each value is [real, imag]."""
    fs = _log(tmp_path, 18, [2], _content(COMPLEX.astype("<c16")))
    data = json.loads(fs.read("/tensors/t/000003.json", 1 << 20, 0))

    assert data["dtype"] == "complex128"
    assert data["values"] == [[1.0, 2.0], [-3.5, -0.5]]


MALFORMED = [
    (
        "odd complex parts",
        8,
        _packed(SCOMPLEX_VAL, np.array([1, 2, 3], "<f4").tobytes()),
    ),
    ("int8 out of range", 6, _packed(INT_VAL, _varints([300]))),
    ("negative uint16", 17, _packed(INT_VAL, _varints([-1]))),
    ("half pattern too wide", 19, _packed(HALF_VAL, _varints([70000]))),
    ("partial item", 19, _content(np.array([1, 2, 3], "u1"))),
]


@pytest.mark.parametrize(
    ("dtype", "wire"), [case[1:] for case in MALFORMED], ids=[c[0] for c in MALFORMED]
)
def test_malformed_tensors_are_left_out(
    tmp_path: Path, dtype: int, wire: bytes
) -> None:
    """A corrupt record does not break the listing of the run."""
    fs = _log(tmp_path, dtype, [], wire)

    assert not [path for path in _files(fs) if path.endswith(".npy")]
