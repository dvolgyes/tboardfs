from typing import Any

import numpy as np

from tboardfs.decode import _Decode


def tensor_to_array(tensor: dict[str, Any]) -> np.ndarray:
    """Convert a parsed TensorProto dictionary to a NumPy array."""
    dtype = tensor.get("dtype")
    if dtype == 7:
        string_values: list[object] = [
            _Decode.text(value) for value in tensor.get("string_val", [])
        ]
        return _shape_array(np.asarray(string_values, dtype=object), tensor)

    np_dtype = _TensorDType.numpy_dtype(dtype)
    if np_dtype is None:
        return np.asarray([])

    content = tensor.get("tensor_content")
    if content:
        content_values = np.frombuffer(bytes(content), dtype=np_dtype).copy()
        return _shape_array(content_values, tensor)

    field = _TensorDType.value_field(dtype)
    if field is None:
        return np.asarray([], dtype=np_dtype)
    all_values = tensor.get("values", {})
    raw_values = all_values.get(field, all_values.get(int(field), []))
    if not isinstance(raw_values, list):
        return np.asarray([], dtype=np_dtype)
    return _shape_array(np.asarray(raw_values, dtype=np_dtype), tensor)


def _shape_array(values: np.ndarray, tensor: dict[str, Any]) -> np.ndarray:
    shape = tensor.get("shape") or []
    if not shape:
        return values
    size = int(np.prod(shape))
    if size != values.size:
        return values
    return values.reshape(tuple(int(dim) for dim in shape))


class _TensorDType:
    """Map TensorProto dtype ids to NumPy storage."""

    @staticmethod
    def numpy_dtype(dtype: object) -> np.dtype | None:
        """Return NumPy dtype for supported TensorProto dtype ids."""
        if not isinstance(dtype, int):
            return None
        mapping = {
            1: np.dtype("<f4"),
            2: np.dtype("<f8"),
            3: np.dtype("<i4"),
            4: np.dtype("u1"),
            9: np.dtype("<i8"),
            10: np.dtype("?"),
            22: np.dtype("<u4"),
            23: np.dtype("<u8"),
        }
        return mapping.get(dtype)

    @staticmethod
    def value_field(dtype: object) -> str | None:
        """Return parsed values field key for supported TensorProto dtype ids."""
        if not isinstance(dtype, int):
            return None
        mapping = {
            1: "5",
            2: "6",
            3: "7",
            4: "7",
            9: "10",
            10: "11",
            22: "16",
            23: "17",
        }
        return mapping.get(dtype)
