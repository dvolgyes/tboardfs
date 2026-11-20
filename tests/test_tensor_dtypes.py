"""Tensor exports decode every numeric TensorProto dtype that NumPy can hold."""

import io
import json
from pathlib import Path

import numpy as np
import pytest
from test_incremental_parse import _files, _frame

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2
from tboardfs.tensor import tensor_to_array

# TensorProto field numbers from tensorflow/core/framework/tensor.proto.
SCOMPLEX_VAL = 9
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
    event = compat_pb2.Event(wall_time=1.0, step=3)
    value = event.summary.value.add(tag="t")
    value.metadata.plugin_data.plugin_name = "tensors"
    value.tensor.dtype = dtype
    value.tensor.tensor_shape.dim.extend(
        compat_pb2.TensorShapeProto.Dim(size=s) for s in shape
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


CORPUS = Path(__file__).parent / "data" / "producer-cases" / "dtypes"
COMPLEX = np.array([1 + 2j, -3.5 - 0.5j])


@pytest.mark.parametrize(
    ("dtype", "storage", "values"),
    [
        (1, "<f4", [1.25, -2.5]),
        (2, "<f8", [1 + 2**-50, -2.5]),
        (3, "<i4", [-(2**31), 17]),
        (4, "u1", [0, 255]),
        (5, "<i2", [-32768, 7]),
        (6, "i1", [-128, 127]),
        (8, "<c8", COMPLEX.tolist()),
        (9, "<i8", [-(2**63), 17]),
        (10, "?", [False, True]),
        (11, "i1", [-128, 127]),
        (12, "u1", [0, 255]),
        (13, "<i4", [-(2**31), 2**31 - 1]),
        (14, "<f4", [1.5, -2.0]),
        (15, "<i2", [-32768, 32767]),
        (16, "<u2", [0, 65535]),
        (17, "<u2", [65535, 1]),
        (18, "<c16", COMPLEX.tolist()),
        (19, "<f2", [1.5, -2, 65504]),
        (22, "<u4", [0, 2**32 - 1]),
        (23, "<u8", [0, 2**64 - 1]),
    ],
)
def test_both_tensor_encodings_keep_dtype_and_values(
    dtype: int, storage: str, values: list[object]
) -> None:
    fs = TensorBoardFS(CORPUS)
    for mode in ("fields", "content"):
        path = f"/tensors/{mode}-{dtype}/000003.npy"
        array = np.load(io.BytesIO(fs.read(path, 4096, 0)), allow_pickle=False)
        assert array.dtype == np.dtype(storage)
        assert array.tolist() == values


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


@pytest.mark.parametrize(
    ("shape", "values"),
    [
        ([3], [1.0, 2.0]),
        ([2], [1.0, 2.0, 3.0]),
        ([], [1.0, 2.0]),
        ([0], [1.0, 2.0]),
        ([-1, -2], [1.0, 2.0]),
    ],
)
def test_values_that_do_not_fill_the_shape_are_left_out(
    tmp_path: Path, shape: list[int], values: list[float]
) -> None:
    """TensorBoard cannot shape these values, so they have no array file."""
    float_val = 5
    wire = _packed(float_val, np.array(values, "<f4").tobytes())
    fs = _log(tmp_path, 1, shape, wire)

    assert not [path for path in _files(fs) if path.endswith(".npy")]


STRING_VAL = 8
FILL_CASES = [
    ("float", 1, [2, 2], _packed(5, np.array([1.5], "<f4").tobytes()), [1.5] * 4),
    (
        "complex",
        8,
        [3],
        _packed(SCOMPLEX_VAL, np.array([1, 2], "<f4").tobytes()),
        [1 + 2j] * 3,
    ),
    ("bfloat16", 14, [2], _packed(HALF_VAL, _varints([0x3FC0])), [1.5, 1.5]),
    ("string", 7, [3], _varint(STRING_VAL << 3 | 2) + b"\x01a", ["a"] * 3),
    ("no elements", 1, [0], _packed(5, np.array([1.0], "<f4").tobytes()), []),
    (
        "at the limit",
        1,
        [256],
        _packed(5, np.array([2.0], "<f4").tobytes()),
        [2.0] * 256,
    ),
]


@pytest.mark.parametrize(
    ("dtype", "shape", "wire", "expected"),
    [case[1:] for case in FILL_CASES],
    ids=[case[0] for case in FILL_CASES],
)
def test_a_single_value_fills_the_shape(
    tmp_path: Path, dtype: int, shape: list[int], wire: bytes, expected: list[object]
) -> None:
    """As in TensorBoard's make_ndarray, one value is repeated to the shape size."""
    data = _npy(_log(tmp_path, dtype, shape, wire))

    assert list(data.shape) == shape
    assert data.ravel().tolist() == expected


@pytest.mark.parametrize(
    ("shape", "wire"),
    [
        ([3], _content(np.array([1.0], "<f4"))),
        ([257], _packed(5, np.array([1.0], "<f4").tobytes())),
        ([1 << 40], _packed(5, np.array([1.0], "<f4").tobytes())),
    ],
    ids=["tensor_content", "over the limit", "huge shape"],
)
def test_a_single_value_that_cannot_fill_the_shape_is_left_out(
    tmp_path: Path, shape: list[int], wire: bytes
) -> None:
    """tensor_content is never repeated, and the result has a memory limit.

    TensorBoard has no limit; a record of a few bytes would otherwise claim
    memory for any shape that it declares.
    """
    fs = _log(tmp_path, 1, shape, wire)

    assert not [path for path in _files(fs) if path.endswith(".npy")]


def test_string_tensors_load_without_pickle(tmp_path: Path) -> None:
    """String tensors are fixed-width text arrays, not pickled objects."""
    string_val = 8
    wire = b"".join(
        _varint(string_val << 3 | 2) + _varint(len(item)) + item
        for item in ["a", "bé", "c\n", "d"]
        for item in [item.encode()]
    )
    data = _npy(_log(tmp_path, 7, [2, 2], wire))

    assert data.dtype.kind == "U"
    assert data.tolist() == [["a", "bé"], ["c\n", "d"]]


@pytest.mark.parametrize("dtype", [0, 20, 21, *range(24, 34), 101])
def test_a_dtype_that_numpy_cannot_hold_does_not_decode(dtype: int) -> None:
    """Float8, 4-bit, 2-bit, resource, variant and reference bytes are not guessed."""
    tensor = {"dtype": dtype, "shape": [4], "tensor_content": b"\x01\x02\x03\x04"}

    assert tensor_to_array(tensor) is None


def test_opaque_tensors_and_binary_strings_survive_raw_frame_export() -> None:
    source = CORPUS / "events.out.tfevents.dtypes"
    fs = TensorBoardFS(CORPUS)
    frame = fs.read(f"/raw/{source.name}/{0:020d}.tfrecord", source.stat().st_size, 0)
    assert frame == source.read_bytes()
    event = compat_pb2.Event.FromString(frame[12:-4])
    tensors = {value.tag: value.tensor for value in event.summary.value}
    assert list(tensors["binary-string"].string_val) == [b"a\0b", b"\xff\xfe"]
    assert tensors["opaque-20"].resource_handle_val[0].name == "weights"
    variant = tensors["opaque-21"].variant_val[0]
    assert (variant.metadata, variant.tensors[0].string_val[0]) == (
        b"\0\xff",
        b"nested\0bytes",
    )
    for dtype in [0, 20, 21, *range(24, 34), *range(101, 134), 999]:
        assert tensors[f"opaque-{dtype}"].dtype == dtype
    files = set(_files(fs))
    assert "/tensors/fields-1/000003.npy" in files
    assert not any("/opaque-" in path and path.endswith(".npy") for path in files)
