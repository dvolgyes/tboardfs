from typing import Any, ClassVar

import numpy as np

from tboardfs.decode import _Decode

BFLOAT16 = 14


def tensor_to_array(tensor: dict[str, Any]) -> np.ndarray:
    """Convert a parsed TensorProto dictionary to a NumPy array."""
    dtype = tensor.get("dtype")
    if dtype == 7:
        string_values = [_Decode.text(value) for value in tensor.get("string_val", [])]
        return _shape_array(np.asarray(string_values, dtype=np.str_), tensor)

    spec = _TensorDType.spec(dtype)
    if spec is None:
        return np.asarray([])
    storage, field, field_dtype = spec

    content = tensor.get("tensor_content")
    if content:
        if len(content) % np.dtype(storage).itemsize:
            return np.asarray([], dtype=storage)
        values = np.frombuffer(bytes(content), dtype=storage).copy()
    else:
        all_values = tensor.get("values", {})
        raw_values = all_values.get(field, all_values.get(str(field), []))
        if not _TensorDType.fits(raw_values, field_dtype, storage):
            return np.asarray([], dtype=storage)
        values = np.asarray(raw_values, dtype=field_dtype).view(storage)
    if dtype == BFLOAT16:
        values = _TensorDType.widen_bfloat16(values)
    return _shape_array(values, tensor)


def _shape_array(values: np.ndarray, tensor: dict[str, Any]) -> np.ndarray:
    shape = tensor.get("shape") or []
    if not shape:
        return values
    size = int(np.prod(shape))
    if size != values.size:
        return values
    return values.reshape(tuple(int(dim) for dim in shape))


class _TensorDType:
    """Map TensorProto dtype ids to NumPy storage.

    Each id maps to the NumPy dtype of its tensor_content bytes, the
    TensorProto field number of its repeated values, and the dtype of one
    item in that field. half_val holds 16-bit patterns, and the complex
    fields hold real and imaginary parts in turn, so their items are viewed
    as the storage dtype. NumPy has no bfloat16; it widens exactly to
    float32. Float8, 4-bit and resource/variant tensors are not decoded.
    """

    SPECS: ClassVar[dict[int, tuple[str, int, str]]] = {
        1: ("<f4", 5, "<f4"),
        2: ("<f8", 6, "<f8"),
        3: ("<i4", 7, "<i4"),
        4: ("u1", 7, "u1"),
        5: ("<i2", 7, "<i2"),
        6: ("i1", 7, "i1"),
        8: ("<c8", 9, "<f4"),
        9: ("<i8", 10, "<i8"),
        10: ("?", 11, "?"),
        11: ("i1", 7, "i1"),
        12: ("u1", 7, "u1"),
        13: ("<i4", 7, "<i4"),
        BFLOAT16: ("<u2", 13, "<u2"),
        15: ("<i2", 7, "<i2"),
        16: ("<u2", 7, "<u2"),
        17: ("<u2", 7, "<u2"),
        18: ("<c16", 12, "<f8"),
        19: ("<f2", 13, "<u2"),
        22: ("<u4", 16, "<u4"),
        23: ("<u8", 17, "<u8"),
    }

    @staticmethod
    def spec(dtype: object) -> tuple[str, int, str] | None:
        """Return storage dtype, value field and field item dtype for an id."""
        if not isinstance(dtype, int):
            return None
        return _TensorDType.SPECS.get(dtype)

    @staticmethod
    def fits(raw_values: object, field_dtype: str, storage: str) -> bool:
        """Return true when field items convert to whole storage items."""
        if not isinstance(raw_values, list):
            return False
        item = np.dtype(field_dtype)
        if len(raw_values) * item.itemsize % np.dtype(storage).itemsize:
            return False
        if item.kind not in "iu" or not raw_values:
            return True
        info = np.iinfo(item)
        return bool(info.min <= min(raw_values) and max(raw_values) <= info.max)

    @staticmethod
    def widen_bfloat16(bits: np.ndarray) -> np.ndarray:
        """Return bfloat16 bit patterns as the float32 values they encode."""
        return (bits.astype("<u4") << 16).view("<f4")
