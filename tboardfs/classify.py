import struct
from typing import Any

import magic
import numpy as np

from tboardfs.constants import LARGE_TENSOR_THRESHOLD
from tboardfs.decode import _Decode
from tboardfs.model import Scalar


def detect_extension(blob: bytes | None, kind: str) -> str:
    """Infer a stable file extension from encoded summary bytes."""
    if not blob:
        return "bin"
    extension = _Classifier.extension_from_magic(blob)
    if extension is not None:
        return extension
    extension = _Classifier.extension_from_signature(blob)
    if extension is not None:
        return extension
    del kind
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

        dtype = tensor.get("dtype")
        values = tensor.get("values", {})
        content = tensor.get("tensor_content")
        if content:
            scalar = _Classifier.tensor_content_scalar(dtype, content)
            if scalar is not None:
                return scalar

        if dtype == 1 and values.get(5):
            return Scalar(values[5][0], np.dtype("float32"))
        if dtype == 2 and values.get(6):
            return Scalar(values[6][0], np.dtype("float64"))
        if dtype == 3 and values.get(7):
            return Scalar(values[7][0], np.dtype("int32"))
        if dtype == 9 and values.get(10):
            return Scalar(values[10][0], np.dtype("int64"))
        if dtype == 10 and values.get(11):
            return Scalar(values[11][0], np.dtype("bool"))
        if dtype == 22 and values.get(16):
            return Scalar(values[16][0], np.dtype("uint32"))
        if dtype == 23 and values.get(17):
            return Scalar(values[17][0], np.dtype("uint64"))
        return None

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
    def binary_kind(value: dict[str, Any]) -> str | None:
        """Classify summary values whose raw bytes should be exposed as files."""
        image_blob = value.get("image", {}).get("encoded_image_string")
        if image_blob and bytes(image_blob).startswith((b"GIF87a", b"GIF89a")):
            return "videos"
        if image_blob:
            return "images"
        if value.get("audio", {}).get("encoded_audio_string"):
            return "audio"

        plugin_name = (value.get("plugin_name") or "").lower()
        if plugin_name in {"video", "videos"}:
            return "videos"

        tensor = value.get("tensor")
        if tensor and _Classifier.large_tensor_like(tensor):
            return "tensors"
        return None

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
        if tensor.get("string_val"):
            return bytes(tensor["string_val"][0])
        return None

    @staticmethod
    def plugin_json_payload(value: dict[str, Any]) -> dict[str, Any] | None:
        """Return JSON-safe plugin metadata for known TensorBoard plugin records."""
        plugin_name = value.get("plugin_name")
        if not plugin_name:
            tensor = value.get("tensor") or {}
            if tensor.get("dtype") == 7 and tensor.get("string_val"):
                plugin_name = "text"
            else:
                return None
        if str(plugin_name).lower() in {"video", "videos"} and _Classifier.binary_blob(
            value
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
    def json_tensor(tensor: dict[str, Any]) -> dict[str, Any]:
        """Return a JSON-friendly view of TensorProto fields."""
        out = dict(tensor)
        if "values" in out:
            out["values"] = {str(key): value for key, value in out["values"].items()}
        if "tensor_content" in out:
            out["tensor_content"] = {"bytes": len(out["tensor_content"])}
        if "string_val" in out:
            out["string_val"] = [_Decode.text(value) for value in out["string_val"]]
        return out

    @staticmethod
    def extension_from_magic(blob: bytes) -> str | None:
        """Return a native extension when libmagic recognizes the payload."""
        mime = magic.from_buffer(blob, mime=True).lower()
        description = magic.from_buffer(blob).lower()
        mime_extensions = {
            "application/json": "json",
            "audio/flac": "flac",
            "audio/mpeg": "mp3",
            "audio/ogg": "ogg",
            "audio/wav": "wav",
            "audio/webm": "webm",
            "audio/x-flac": "flac",
            "audio/x-wav": "wav",
            "image/gif": "gif",
            "image/jpeg": "jpg",
            "image/png": "png",
            "image/webp": "webp",
            "text/plain": "txt",
            "video/mp4": "mp4",
            "video/ogg": "ogg",
            "video/quicktime": "mov",
            "video/webm": "webm",
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
            ("riff", "wav"),
            ("wave audio", "wav"),
            ("ogg data", "ogg"),
            ("web/p image", "webp"),
            ("webm", "webm"),
            ("matroska", "webm"),
            ("iso media", "mp4"),
            ("quicktime", "mov"),
            ("mpeg adts", "mp3"),
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
        if blob.startswith(b"GIF87a") or blob.startswith(b"GIF89a"):
            return "gif"
        if blob.startswith(b"RIFF") and blob[8:12] == b"WAVE":
            return "wav"
        if blob.startswith(b"OggS"):
            return "ogg"
        if len(blob) >= 8 and blob[4:8] == b"ftyp":
            return "mp4"
        return None
