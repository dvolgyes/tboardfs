import math
from collections.abc import Sequence
from errno import EIO
from typing import Any, ClassVar

import numpy as np

from tboardfs.constants import DT_STRING, MAX_REPEATED_TENSOR_BYTES
from tboardfs.decode import _Decode
from tboardfs.errors import fuse_error
from tboardfs.model.json_entry import JsonEntry

BFLOAT16 = 14
_MAX_NUMPY_DIMS = 64


def tensor_to_array(
    tensor: dict[str, Any], *, copy_content: bool = True
) -> np.ndarray | None:
    """Convert a parsed TensorProto dictionary to a NumPy array.

    Returns None when the tensor does not decode: an unsupported dtype,
    malformed values, or values that do not fill the shape.
    Set copy_content to False when a read-only view of stored bytes suffices.
    """
    dtype = tensor.get("dtype")
    if dtype == DT_STRING:
        values = np.asarray(_TensorDType.strings(tensor), dtype=np.str_)
        return _TensorDType.shaped(values, tensor, fill=True)

    spec = _TensorDType.spec(dtype)
    if spec is None:
        return None
    storage, field, field_dtype = spec

    content = tensor.get("tensor_content")
    if content:
        if len(content) % np.dtype(storage).itemsize:
            return None
        values = np.frombuffer(bytes(content), dtype=storage)
        if copy_content and dtype != BFLOAT16:
            values = values.copy()
    else:
        all_values = tensor.get("values", {})
        raw_values = all_values.get(field, all_values.get(str(field), []))
        if not _TensorDType.fits(raw_values, field_dtype, storage):
            return None
        values = np.asarray(raw_values, dtype=field_dtype).view(storage)
    if dtype == BFLOAT16:
        values = _TensorDType.widen_bfloat16(values)
    return _TensorDType.shaped(values, tensor, fill=not content)


def _read_tensor_array(entry: JsonEntry) -> np.ndarray:
    """Decode a source-backed tensor without modifying the indexed payload."""
    source = entry.tensor_source
    if source is None:
        raise fuse_error(EIO)
    content = source.read()
    if not content:
        raise fuse_error(EIO)
    tensor = {**entry.payload["tensor"], "tensor_content": content}
    data = tensor_to_array(tensor, copy_content=False)
    if data is None:
        raise fuse_error(EIO)
    return data


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
    def strings(tensor: dict[str, Any]) -> list[str]:
        """Decode string_val items, replacing invalid UTF-8."""
        return [_Decode.text(value) for value in tensor.get("string_val", [])]

    @staticmethod
    def string_preview(
        tensor: dict[str, Any],
    ) -> tuple[np.ndarray, np.dtype] | None:
        """Keep exact decoded strings and their fixed-width Unicode dtype."""
        strings = _TensorDType.strings(tensor)
        width = max(1, max(map(len, strings), default=0))
        dtype = np.dtype(f"U{width}")
        values = np.asarray(strings, dtype=object)
        cells = _TensorDType.shaped(values, tensor, fill=True, itemsize=dtype.itemsize)
        return None if cells is None else (cells, dtype)

    @staticmethod
    def shaped(
        values: np.ndarray,
        tensor: dict[str, Any],
        *,
        fill: bool,
        itemsize: int | None = None,
    ) -> np.ndarray | None:
        """Return the values in the tensor shape, or None when they do not fill it.

        A tensor without dimensions holds one value, a dimension of size 0
        holds none, and a negative size is invalid, as in TensorBoard's
        make_ndarray. With fill, a single value is repeated to the shape size,
        as make_ndarray does for the repeated value fields; the result may
        hold at most MAX_REPEATED_TENSOR_BYTES, so that a small record cannot
        claim memory for any shape that it declares. For string previews,
        itemsize is the fixed-width Unicode size, not the object pointer size.
        """
        shape = tuple(int(dim) for dim in tensor.get("shape") or [])
        width = values.itemsize if itemsize is None else itemsize
        size = _TensorDType._shape_size(shape, max(values.itemsize, width))
        if size is None:
            return None
        if fill and values.size == 1 and size != 1:
            if size * width > MAX_REPEATED_TENSOR_BYTES:
                return None
            values = np.repeat(values, size)
        if values.size != size:
            return None
        return values.reshape(shape)

    @staticmethod
    def accepts_extent(tensor: dict[str, Any], size: int) -> bool:
        """Return whether this parsed numeric tensor can decode the given byte extent."""
        dtype = tensor.get("dtype")
        spec = _TensorDType.spec(dtype)
        if spec is None or size <= 0:
            return False
        storage_width = np.dtype(spec[0]).itemsize
        array_width = np.dtype("<f4").itemsize if dtype == BFLOAT16 else storage_width
        count = _TensorDType._shape_size(tensor.get("shape") or (), array_width)
        return count is not None and size == count * storage_width

    @staticmethod
    def _shape_size(shape: Sequence[int], itemsize: int) -> int | None:
        if min(shape, default=0) < 0 or not _TensorDType.fits_numpy(shape, itemsize):
            return None
        return math.prod(shape)

    @staticmethod
    def fits_numpy(shape: Sequence[int], itemsize: int) -> bool:
        """Return whether NumPy can make an array of this shape and item size.

        NumPy limits an array to 64 dimensions, and the bytes of its nonzero
        dimensions to the largest index, also when another dimension is 0.
        """
        nonzero = math.prod(dim for dim in shape if dim)
        return (
            len(shape) <= _MAX_NUMPY_DIMS
            and nonzero * max(itemsize, 1) <= np.iinfo(np.intp).max
        )

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
