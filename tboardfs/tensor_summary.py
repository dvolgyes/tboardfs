"""Read media items and histogram buckets from TF1 fields or TF2 tensors."""

from typing import Any

from tboardfs.classify import _Classifier
from tboardfs.tensor import tensor_to_array

DT_STRING = 7
GIF_HEADERS = (b"GIF87a", b"GIF89a")
# Columns of a TF2 histogram tensor: bucket left, bucket right, count.
HISTOGRAM_COLUMNS = 3


class _TensorSummary:
    """Unpack the summaries that TF2 writers store as tensors.

    An image tensor is [width, height, image_1, ..., image_k]. An audio tensor
    has k rows of [encoded audio, label]. A histogram tensor has k rows of
    [left, right, count]. TF1 writers use the image, audio and histo fields.
    """

    @staticmethod
    def media_parts(value: dict[str, Any]) -> list[tuple[str, int, str]] | None:
        """Return (kind, string_val index, file-name suffix) per media item.

        One item has no suffix. Items of a step are _<i>, with i padded to one
        width. Returns None when the value is not a TF2 image or audio tensor.
        """
        tensor = value.get("tensor") or {}
        if tensor.get("dtype") != DT_STRING:
            return None
        plugin = (value.get("plugin_name") or "").lower()
        strings = tensor.get("string_val", [])
        shape = tensor.get("shape") or []
        if plugin == "images" and len(strings) >= 2:
            indices, kind = range(2, len(strings)), "images"
        elif plugin == "audio" and shape[1:] == [2] and len(strings) == 2 * shape[0]:
            indices, kind = range(0, len(strings), 2), "audio"
        else:
            return None
        width = len(str(len(indices) - 1))
        return [
            (
                "videos" if bytes(strings[index]).startswith(GIF_HEADERS) else kind,
                index,
                "" if len(indices) == 1 else f"_{sample:0{width}d}",
            )
            for sample, index in enumerate(indices)
        ]

    @staticmethod
    def histogram_buckets(
        value: dict[str, Any],
    ) -> list[tuple[float, float, float]] | None:
        """Return (left, right, count) per bucket, or None for no histogram.

        A TF1 bucket starts at the limit of the bucket before it, and the
        first one at the histogram minimum.
        """
        histogram = value.get("histo")
        if histogram is not None:
            limits = [float(limit) for limit in histogram.get("bucket_limit", [])]
            lefts = [float(histogram.get("min", float("-inf"))), *limits[:-1]]
            counts = [float(count) for count in histogram.get("bucket", [])]
            return list(zip(lefts, limits, counts, strict=False))
        if (value.get("plugin_name") or "").lower() != "histograms":
            return None
        data = tensor_to_array(value.get("tensor") or {})
        if data.dtype.kind not in "fiu" or data.shape[1:] != (HISTOGRAM_COLUMNS,):
            return None
        return [
            (float(left), float(right), float(count)) for left, right, count in data
        ]

    @staticmethod
    def unpacked(value: dict[str, Any]) -> bool:
        """Return true when a plugin record becomes files or table rows.

        Such a record has no JSON file of its own, as its TF1 form has none.
        """
        if (value.get("plugin_name") or "").lower() == "scalars":
            return _Classifier.scalar_value(value) is not None
        return (
            _TensorSummary.media_parts(value) is not None
            or _TensorSummary.histogram_buckets(value) is not None
        )
