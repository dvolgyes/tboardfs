import struct
from typing import Any

import magic
import numpy as np

from tboardfs.constants import DT_STRING, LARGE_TENSOR_THRESHOLD, PLUGIN_TAB
from tboardfs.model import Scalar
from tboardfs.tensor import _MAX_NUMPY_DIMS, tensor_to_array


def detect_extension(blob: bytes | None) -> str:
    """Infer a stable file extension from encoded summary bytes."""
    if not blob:
        return "bin"
    extension = _Classifier.extension_from_magic(blob)
    if extension is not None:
        return extension
    extension = _Classifier.extension_from_signature(blob)
    if extension is not None:
        return extension
    return "bin"


class _Classifier:
    """Stateless helpers for classifying parsed TensorBoard summary values."""

    @staticmethod
    def scalar_value(value: dict[str, Any]) -> Scalar | None:
        """Return a scalar value from a scalar-shaped summary record."""
        if "simple_value" in value:
            return Scalar(value["simple_value"], np.dtype("float32"))

        tensor = value.get("tensor")
        if not tensor:
            return None
        shape = tensor.get("shape", ())
        if len(shape) > _MAX_NUMPY_DIMS or any(dim != 1 for dim in shape):
            return None

        scalar = _Classifier.common_scalar(tensor)
        if scalar is not None:
            return scalar
        # The other real dtypes are rare, so they take the slower general path.
        array = tensor_to_array(tensor)
        if array is None or array.size != 1 or array.dtype.kind not in "biuf":
            return None
        return Scalar(array.item(), array.dtype)

    @staticmethod
    def common_scalar(tensor: dict[str, Any]) -> Scalar | None:
        """Return a scalar of a common dtype from its fields, without NumPy."""
        dtype = tensor.get("dtype")
        values = tensor.get("values", {})
        content = tensor.get("tensor_content")
        if content:
            return _Classifier.tensor_content_scalar(dtype, content)

        # dtype id: (TensorProto repeated field number, NumPy dtype)
        fields = {
            1: (5, "float32"),
            2: (6, "float64"),
            3: (7, "int32"),
            9: (10, "int64"),
            10: (11, "bool"),
            22: (16, "uint32"),
            23: (17, "uint64"),
        }
        if dtype not in fields:
            return None
        field, np_dtype = fields[dtype]
        if len(values.get(field, ())) != 1:
            return None
        return Scalar(values[field][0], np.dtype(np_dtype))

    @staticmethod
    def tensor_content_scalar(dtype: int | None, content: bytes) -> Scalar | None:
        """Return a scalar from fixed-width tensor content when supported."""
        specs = {
            1: ("<f", np.dtype("float32")),
            2: ("<d", np.dtype("float64")),
            3: ("<i", np.dtype("int32")),
            9: ("<q", np.dtype("int64")),
            10: ("<?", np.dtype("bool")),
            22: ("<I", np.dtype("uint32")),
            23: ("<Q", np.dtype("uint64")),
        }
        if dtype is None:
            return None
        spec = specs.get(dtype)
        if spec is None:
            return None

        fmt, np_dtype = spec
        if len(content) != struct.calcsize(fmt):
            return None
        return Scalar(struct.unpack(fmt, content)[0], np_dtype)

    @staticmethod
    def plugin_name(value: dict[str, Any]) -> str:
        """Return the plugin name of a summary value in lower case, or ""."""
        return str(value.get("plugin_name") or "").lower()

    @staticmethod
    def binary_kind(value: dict[str, Any]) -> str | None:
        """Classify summary values whose raw bytes should be exposed as files."""
        image_blob = value.get("image", {}).get("encoded_image_string")
        if image_blob and bytes(image_blob).startswith((b"GIF87a", b"GIF89a")):
            return "videos"
        if image_blob:
            return "images"
        if value.get("audio", {}).get("encoded_audio_string"):
            return "audio"

        plugin_name = _Classifier.plugin_name(value)
        if plugin_name in {"video", "videos"}:
            return "videos"

        tensor = value.get("tensor")
        if (
            tensor
            and _Classifier.raw_tensor_summary(value)
            and _Classifier.large_tensor_like(tensor)
        ):
            return "tensors"
        return None

    @staticmethod
    def raw_tensor_summary(value: dict[str, Any]) -> bool:
        """Return whether a tensor has no tab of its own to show it.

        Histograms, PR curves, meshes and text have their own tabs; a raw
        copy under /tensors would be a duplicate in the wrong tab. A tensor
        without a plugin name keeps it: its strings can be a native file,
        such as a PNG, which the raw file shows with its extension.
        """
        plugin = _Classifier.plugin_name(value)
        return PLUGIN_TAB.get(plugin, "plugins") in {"tensors", "plugins"}

    @staticmethod
    def large_tensor_like(tensor: dict[str, Any]) -> bool:
        """Return true for tensor values large enough to expose as blobs."""
        tensor_content = tensor.get("tensor_content")
        if tensor_content and len(tensor_content) >= LARGE_TENSOR_THRESHOLD:
            return True
        return any(
            len(item) >= LARGE_TENSOR_THRESHOLD for item in tensor.get("string_val", [])
        )

    @staticmethod
    def binary_blob(value: dict[str, Any]) -> bytes | None:
        """Return raw payload bytes from a binary summary value."""
        if value.get("image", {}).get("encoded_image_string"):
            return bytes(value["image"]["encoded_image_string"])
        if value.get("audio", {}).get("encoded_audio_string"):
            return bytes(value["audio"]["encoded_audio_string"])
        tensor = value.get("tensor") or {}
        if tensor.get("tensor_content"):
            return bytes(tensor["tensor_content"])
        return None

    @staticmethod
    def plugin_json_payload(value: dict[str, Any]) -> dict[str, Any] | None:
        """Return JSON-safe plugin metadata for known TensorBoard plugin records."""
        plugin_name = value.get("plugin_name") or _Classifier.unnamed_plugin(value)
        if not plugin_name:
            return None
        if str(plugin_name).lower() in {"video", "videos"} and (
            _Classifier.binary_blob(value) or value.get("tensor", {}).get("string_val")
        ):
            return None
        payload = {"plugin_name": plugin_name, "tag": value.get("tag")}
        if value.get("plugin_content"):
            payload["plugin_content"] = value["plugin_content"]
        tensor = value.get("tensor")
        if tensor:
            payload["tensor"] = tensor
        return payload

    @staticmethod
    def unnamed_plugin(value: dict[str, Any]) -> str | None:
        """Return the tab plugin for a tensor written without a plugin name.

        String tensors are text. Numeric tensors are tensors, unless they are
        scalars, which the scalar table holds.
        """
        tensor = value.get("tensor") or {}
        if not tensor:
            return None
        if tensor.get("dtype") == DT_STRING:
            return "text" if tensor.get("string_val") else None
        if _Classifier.scalar_value(value) is not None:
            return None
        return "tensors"

    @staticmethod
    def extension_from_magic(blob: bytes) -> str | None:
        """Return a native extension when libmagic recognizes the payload."""
        mime = magic.from_buffer(blob, mime=True).lower()
        description = magic.from_buffer(blob).lower()
        if description.startswith("mpeg adts, aac"):
            return "aac"
        if blob[4:12] == b"ftypM4A ":
            return "m4a"
        mime_extensions = {
            "application/json": "json",
            "audio/aac": "aac",
            "audio/aiff": "aiff",
            "audio/flac": "flac",
            "audio/mp4": "m4a",
            "audio/mpeg": "mp3",
            "audio/ogg": "ogg",
            "audio/wav": "wav",
            "audio/webm": "webm",
            "audio/x-aiff": "aiff",
            "audio/x-hx-aac-adts": "aac",
            "audio/x-m4a": "m4a",
            "audio/x-flac": "flac",
            "audio/x-wav": "wav",
            "image/bmp": "bmp",
            "image/x-ms-bmp": "bmp",
            "image/gif": "gif",
            "image/jpeg": "jpg",
            "image/png": "png",
            "image/webp": "webp",
            "text/plain": "txt",
            "video/mp4": "mp4",
            "video/ogg": "ogg",
            "video/quicktime": "mov",
            "video/webm": "webm",
            "video/x-matroska": "mkv",
            "video/x-msvideo": "avi",
        }
        if mime in mime_extensions:
            return mime_extensions[mime]
        return _Classifier.extension_from_magic_description(description)

    @staticmethod
    def extension_from_magic_description(description: str) -> str | None:
        """Map reliable libmagic descriptions when MIME is too generic."""
        description_extensions = (
            ("json text", "json"),
            ("png image", "png"),
            ("jpeg image", "jpg"),
            ("gif image", "gif"),
            ("wave audio", "wav"),
            ("ogg data", "ogg"),
            ("web/p image", "webp"),
            ("webm", "webm"),
            ("mp4 base media", "mp4"),
            ("quicktime", "mov"),
            ("mpeg adts, layer iii", "mp3"),
            ("flac audio", "flac"),
        )
        for needle, extension in description_extensions:
            if needle in description:
                return extension
        return None

    @staticmethod
    def extension_from_signature(blob: bytes) -> str | None:
        """Recognize common encoded payloads when fixture bytes are very small."""
        if blob.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png"
        if blob.startswith(b"\xff\xd8\xff"):
            return "jpg"
        if blob.startswith((b"GIF87a", b"GIF89a")):
            return "gif"
        if blob.startswith(b"RIFF") and blob[8:12] == b"WAVE":
            return "wav"
        if blob.startswith(b"OggS"):
            return "ogg"
        if blob[4:8] == b"ftyp" and blob[8:12] in (b"mp41", b"mp42"):
            return "mp4"
        return None
