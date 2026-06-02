from io import BytesIO
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
    values: list[Any]
    if dtype == 3:
        int_values: list[int] = []
        for value in raw_values:
            if not isinstance(value, int):
                return np.asarray([], dtype=np_dtype)
            int_values.append(_Decode.signed_int32(value))
        values = int_values
    else:
        values = raw_values
    return _shape_array(np.asarray(values, dtype=np_dtype), tensor)


def array_to_npy(data: np.ndarray) -> bytes:
    """Serialize one array as a .npy payload."""
    handle = BytesIO()
    np.save(handle, data)
    return handle.getvalue()


def array_to_json(data: np.ndarray) -> dict[str, Any]:
    """Return a JSON-safe array description."""
    return {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "values": data.tolist(),
    }


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
            9: "10",
            10: "11",
            22: "16",
            23: "17",
        }
        return mapping.get(dtype)
