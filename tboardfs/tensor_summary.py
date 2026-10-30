"""Read media items and histogram buckets from TF1 fields or TF2 tensors."""

from types import MappingProxyType
from typing import Any

from tboardfs.classify import _Classifier, detect_extension
from tboardfs.tensor import tensor_to_array

DT_STRING = 7
GIF_HEADERS = (b"GIF87a", b"GIF89a")
# Columns of a TF2 histogram tensor: bucket left, bucket right, count.
HISTOGRAM_COLUMNS = 3
# TensorBoard's graph plugins: file-name suffix and extension of the stored
# Keras config or RunMetadata, and whether the TensorBoard Graphs tab draws
# its graph and shows its profile.
GRAPH_PLUGINS = MappingProxyType(
    {
        "graph_keras_model": ("", "json", True, False),
        "graph_run_metadata": (".run_metadata", "pb", True, True),
        "graph_run_metadata_graph": (".run_metadata", "pb", True, False),
        "graph_tagged_run_metadata": (".run_metadata", "pb", False, True),
    }
)
# The one graph plugin that TensorBoard reads without plugin content b"1".
TAGGED_RUN_METADATA = "graph_tagged_run_metadata"


class _TensorSummary:
    """Unpack the summaries that TF2 writers store as tensors.

    An image tensor is [width, height, image_1, ..., image_k]. An audio tensor
    has k rows of [encoded audio, label]. A histogram tensor has k rows of
    [left, right, count]. TF1 writers use the image, audio and histo fields.
    A graph tensor holds a Keras config or a RunMetadata as its first string.
    """

    @staticmethod
    def media_parts(value: dict[str, Any]) -> list[tuple[str, int, str]] | None:
        """Return (kind, string_val index, file-name suffix) per media item.

        One item has no suffix. Items of a step are _<i>, with i padded to one
        width. Returns None when the value is not a TF2 image, audio or graph
        tensor. As in TensorBoard, a graph tensor other than a tagged run
        metadata needs plugin content b"1".
        """
        tensor = value.get("tensor") or {}
        if tensor.get("dtype") != DT_STRING:
            return None
        plugin = (value.get("plugin_name") or "").lower()
        strings = tensor.get("string_val", [])
        shape = tensor.get("shape") or []
        if plugin in GRAPH_PLUGINS:
            content = value.get("plugin_content")
            if not strings or (plugin != TAGGED_RUN_METADATA and content != b"1"):
                return None
            return [("graphs", 0, GRAPH_PLUGINS[plugin][0])]
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
    def part_extension(value: dict[str, Any], kind: str, blob: bytes) -> str:
        """Return the file extension of one item from media_parts."""
        if kind == "graphs":
            return GRAPH_PLUGINS[(value.get("plugin_name") or "").lower()][1]
        return detect_extension(blob, kind)

    @staticmethod
    def graph_views(value: dict[str, Any]) -> tuple[bool, bool]:
        """Return whether TensorBoard draws the graph and shows the profile of a graph tensor."""
        _, _, graph, profile = GRAPH_PLUGINS[(value.get("plugin_name") or "").lower()]
        return graph, profile

    @staticmethod
    def histogram_buckets(
        value: dict[str, Any],
    ) -> list[tuple[float, float, float]] | None:
        """Return (left, right, count) per bucket, or None for no histogram."""
        histogram = value.get("histo")
        if histogram is not None:
            return _TensorSummary.tf1_histogram_buckets(histogram)
        if (value.get("plugin_name") or "").lower() != "histograms":
            return None
        data = tensor_to_array(value.get("tensor") or {})
        if (
            data is None
            or data.dtype.kind not in "fiu"
            or data.shape[1:] != (HISTOGRAM_COLUMNS,)
        ):
            return None
        return [
            (float(left), float(right), float(count)) for left, right, count in data
        ]

    @staticmethod
    def tf1_histogram_buckets(
        histogram: dict[str, Any],
    ) -> list[tuple[float, float, float]]:
        """Return (left, right, count) per bucket of a TF1 HistogramProto.

        The buckets are read as TensorBoard's data_compat migrates them: the
        empty buckets at the two ends are dropped, and the buckets run from
        the minimum to the maximum.
        """
        counts = [float(count) for count in histogram.get("bucket", [])]
        filled = [index for index, count in enumerate(counts) if count > 0]
        if not filled:
            return []
        start, end = filled[0], filled[-1]
        limits = histogram.get("bucket_limit", [])
        inner = [float(limit) for limit in limits[start:end]]
        lefts = [float(histogram["min"]), *inner]
        rights = [*inner, float(histogram["max"])]
        # A malformed histogram with too few limits keeps the buckets it has.
        return list(zip(lefts, rights, counts[start : end + 1], strict=False))

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
