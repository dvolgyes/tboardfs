"""Verify high-level PyTorch writer records without importing PyTorch."""

import json
from io import BytesIO

import numpy as np
import pytest
from test_incremental_parse import _files
from test_tf_producer_cases import CORPUS, _events

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2


@pytest.fixture(scope="module")
def fs() -> TensorBoardFS:
    return TensorBoardFS(CORPUS / "torch")


@pytest.fixture(scope="module")
def values() -> dict[str, compat_pb2.Summary.Value]:
    return {v.tag: v for event in _events("torch") for v in event.summary.value}


def test_scalar_styles_preserve_only_the_producers_precision(
    fs: TensorBoardFS, values: dict
) -> None:
    legacy = values["scalar/legacy"]
    assert legacy.WhichOneof("value") == "simple_value"
    assert legacy.simple_value == 16_777_220
    for tag, dtype, field, expected in [
        ("new-style", 1, "float_val", 16_777_220),
        ("double-precision", 2, "double_val", 16_777_219),
    ]:
        value = values[f"scalar/{tag}"]
        assert value.metadata.plugin_data.plugin_name == "scalars"
        assert value.tensor.dtype == dtype
        assert list(getattr(value.tensor, field)) == [expected]
    for tag, expected in [
        ("legacy", 16_777_220),
        ("new-style", 16_777_220),
        ("double-precision", 16_777_219),
    ]:
        (row,) = json.loads(fs.read(f"/scalars/scalar/{tag}.json", 4096, 0))
        assert (row["step"], row["value"]) == (1, expected)


@pytest.mark.parametrize(
    ("tag", "dtype", "field", "expected"),
    [
        ("float32", 1, "float_val", [1.25, -2.5]),
        ("float64", 2, "double_val", [1.25, -2.5]),
        ("int8", 6, "int_val", [-128, 127]),
        ("int16", 5, "int_val", [-32768, 32767]),
        ("int32", 3, "int_val", [-2147483648, 2147483647]),
        ("int64", 9, "int64_val", [-(2**63), 2**63 - 1]),
        ("bool", 10, "bool_val", [False, True]),
        ("complex64", 8, "scomplex_val", [1 + 2j, -3.5 - 0.5j]),
        ("complex128", 18, "dcomplex_val", [1 + 2j, -3.5 - 0.5j]),
    ],
)
def test_add_tensor_typed_export(
    fs: TensorBoardFS, values: dict, tag: str, dtype: int, field: str, expected: list
) -> None:
    value = values[f"tensor/{tag}"]
    assert value.metadata.plugin_data.plugin_name == "tensor"
    assert value.tensor.dtype == dtype
    assert list(getattr(value.tensor, field)) == (
        [1, 2, -3.5, -0.5] if tag.startswith("complex") else expected
    )
    array = np.load(
        BytesIO(fs.read(f"/tensors/tensor/{tag}/000001.npy", 4096, 0)),
        allow_pickle=False,
    )
    assert array.dtype == np.dtype(tag)
    assert array.tolist() == expected


def test_noncanonical_tensors_remain_exact_in_raw_view(
    fs: TensorBoardFS, values: dict
) -> None:
    files = set(_files(fs))
    for tag, dtype, field, expected in [
        ("float16", 19, "half_val", [1069547520, -1073741824]),
        ("bfloat16", 14, "half_val", [1069547520, -1073741824]),
        ("uint8", 4, "uint32_val", [0, 255]),
    ]:
        tensor = values[f"tensor/{tag}"].tensor
        assert tensor.dtype == dtype
        assert list(getattr(tensor, field)) == expected
        assert not any(
            path.startswith(f"/tensors/tensor/{tag}/") and path.endswith(".npy")
            for path in files
        )
    source = CORPUS / "torch" / "events.out.tfevents.torch"
    raw = sorted(path for path in files if path.startswith("/raw/"))
    assert b"".join(fs.read(path, 4096, 0) for path in raw) == source.read_bytes()
