import re
import struct
from importlib import import_module
from types import MappingProxyType
from typing import Any, Literal
from zlib import crc32

import numpy as np

from tboardfs.constants import DT_STRING, DT_UINT8, LARGE_TENSOR_THRESHOLD, PLUGIN_TAB
from tboardfs.data_formats import _Mesh, data_extension
from tboardfs.media_containers import media_extension
from tboardfs.model import Scalar
from tboardfs.native_files import _Framing, native_extension
from tboardfs.tensor import _MAX_NUMPY_DIMS, tensor_to_array
from tboardfs.zip_types import zip_extension

_UNTRUSTED_SIGNATURE_EXTENSIONS = frozenset(
    {
        "aac",
        "arrow",
        "asf",
        "avi",
        "glb",
        "h5",
        "jp2",
        "j2k",
        "mpg",
        "npy",
        "parquet",
        "pdf",
        "ply",
        "sndr",
        "tiff",
        "wma",
        "wmv",
        "xxx",
    }
)
_LABEL = re.compile(rb"[\w .-]*")
_SCALAR_FIELDS = MappingProxyType(
    {
        dtype: (field, np.dtype(name), struct.Struct(fmt))
        for dtype, field, name, fmt in (
            (1, 5, "float32", "<f"),
            (2, 6, "float64", "<d"),
            (3, 7, "int32", "<i"),
            (9, 10, "int64", "<q"),
            (10, 11, "bool", "<?"),
            (22, 16, "uint32", "<I"),
            (23, 17, "uint64", "<Q"),
        )
    }
)


def _canonical_png(blob: bytes) -> bool:
    """Recognize PNG chunk framing without reading compressed image data."""
    if not blob.startswith(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"):
        return False
    if not blob.endswith(b"\x00\x00\x00\x00IEND\xaeB`\x82"):
        return False
    if crc32(blob[12:29]) != int.from_bytes(blob[29:33], "big"):
        return False
    end = len(blob) - 12
    offset = 33
    has_data = False
    for _ in range(256):
        if offset + 12 > end:
            break
        size, kind = struct.unpack_from(">I4s", blob, offset)
        if kind in (b"IHDR", b"IEND"):
            return False
        if kind == b"IDAT" and size > 0:
            has_data = True
        offset += size + 12
        if offset == end:
            return has_data
    return False


def detect_extension(blob: bytes | None) -> str:
    """Infer a file extension; valid HDF5 and STL framing can also fit inside a PNG."""
    if not blob:
        return "bin"
    if _canonical_png(blob):
        if _Framing.hdf(blob):
            return "h5"
        return "stl" if _Mesh.stl(blob) else "png"
    extension = _Classifier.container_extension(blob)
    if extension is not None:
        return extension
    extension = _Classifier.extension_from_puremagic(blob)
    if extension is not None:
        return extension
    if _Classifier.text_extension(blob):
        return "txt"
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
        if dtype is None:
            return None
        spec = _SCALAR_FIELDS.get(dtype)
        if spec is None:
            return None
        field, np_dtype, layout = spec
        content = tensor.get("tensor_content")
        if content:
            if len(content) != layout.size:
                return None
            return Scalar(layout.unpack(content)[0], np_dtype)
        values = tensor.get("values", {}).get(field, ())
        if len(values) != 1:
            return None
        return Scalar(values[0], np_dtype)

    @staticmethod
    def plugin_name(value: dict[str, Any]) -> str:
        """Return the plugin name of a summary value in lower case, or ""."""
        return str(value.get("plugin_name") or "").lower()

    @staticmethod
    def binary_kind(value: dict[str, Any]) -> str | None:
        """Classify summary values whose raw bytes should be exposed as files."""
        image_kind = _Classifier.legacy_image_kind(
            value.get("image", {}).get("encoded_image_string")
        )
        if image_kind is not None:
            return image_kind
        if value.get("audio", {}).get("encoded_audio_string"):
            return "audio"

        plugin_name = _Classifier.plugin_name(value)
        if plugin_name in {"video", "videos"}:
            return "videos"

        tensor = value.get("tensor")
        if tensor and _Classifier.raw_tensor_summary(value):
            content = tensor.get("tensor_content") or b""
            if tensor.get("string_val") or (
                content
                and (
                    tensor.get("dtype") == DT_UINT8
                    or len(content) >= LARGE_TENSOR_THRESHOLD
                )
            ):
                return "tensors"
        return None

    @staticmethod
    def legacy_image_kind(blob: bytes | None) -> str | None:
        """Classify GIF images as videos and omit empty legacy image fields."""
        if not blob:
            return None
        return "videos" if bytes(blob).startswith((b"GIF87a", b"GIF89a")) else "images"

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
    def embedded_extension(blob: bytes) -> str | None:
        """Keep recognized small files and preserve large fields even when opaque."""
        if len(blob) >= LARGE_TENSOR_THRESHOLD:
            return detect_extension(blob)
        if _LABEL.fullmatch(blob):
            return None
        extension = detect_extension(blob)
        return None if extension in {"bin", "txt"} else extension

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
    def container_extension(blob: bytes) -> str | None:
        """Resolve containers with overlapping format signatures."""
        if blob.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x06\x06")):
            return zip_extension(blob)
        if blob.startswith((b"RF64", b"BW64")):
            return _Classifier.wave64_extension(blob)
        if blob.startswith(b"RIFF") and blob[8:12] == b"RMID":
            return (
                "rmi"
                if 12 <= int.from_bytes(blob[4:8], "little") + 8 <= len(blob)
                else None
            )
        if blob.startswith(b"MThd") and len(blob) >= 14:
            size = int.from_bytes(blob[4:8], "big")
            return "mid" if 6 <= size <= len(blob) - 8 else None
        if blob.startswith(b"\x93NUMPY"):
            return _Classifier.numpy_extension(blob)
        if blob[4:8] == b"ftyp":
            extension = _Classifier.image_brand(blob)
            if extension is None and blob[8:12] == b"M4A ":
                return "m4a"
            return extension
        if blob[:4] in (b"II*\0", b"MM\0*"):
            return _Classifier.tiff_extension(blob)
        if blob[:4] in (b"II+\0", b"MM\0+"):
            return _Classifier.tiff_extension(blob)
        if blob.startswith(b"\x00\x00\x00\x0cjP  \r\n\x87\n"):
            return "jp2"
        if blob.startswith(b"\xff\x4f\xff\x51"):
            return _Classifier.jpeg2000_codestream(blob)
        return native_extension(blob) or media_extension(blob) or data_extension(blob)

    @staticmethod
    def tiff_extension(blob: bytes) -> str | None:
        """Validate the first TIFF image file directory using bounded framing."""
        endian: Literal["little", "big"]
        if blob[:2] == b"II":
            endian = "little"
        elif blob[:2] == b"MM":
            endian = "big"
        else:
            return None
        version = int.from_bytes(blob[2:4], endian)
        if version == 42 and len(blob) >= 8:
            offset = int.from_bytes(blob[4:8], endian)
            if offset < 8 or offset + 2 > len(blob):
                return None
            count = int.from_bytes(blob[offset : offset + 2], endian)
            end = offset + 2 + count * 12 + 4
            return "tiff" if count and end <= len(blob) else None
        if (
            version != 43
            or len(blob) < 16
            or blob[4:6] != (8).to_bytes(2, endian)
            or blob[6:8] != b"\0\0"
        ):
            return None
        offset = int.from_bytes(blob[8:16], endian)
        if offset < 16 or offset + 8 > len(blob):
            return None
        count = int.from_bytes(blob[offset : offset + 8], endian)
        end = offset + 8 + count * 20 + 8
        return "tiff" if count and end <= len(blob) else None

    @staticmethod
    def jpeg2000_codestream(blob: bytes) -> str | None:
        """Check JPEG 2000 image dimensions and the complete SIZ marker."""
        if len(blob) < 42:
            return None
        size = int.from_bytes(blob[4:6], "big")
        components = int.from_bytes(blob[40:42], "big")
        if size != 38 + 3 * components or 4 + size > len(blob):
            return None
        width = int.from_bytes(blob[8:12], "big")
        height = int.from_bytes(blob[12:16], "big")
        x_origin = int.from_bytes(blob[16:20], "big")
        y_origin = int.from_bytes(blob[20:24], "big")
        x_tile = int.from_bytes(blob[24:28], "big")
        y_tile = int.from_bytes(blob[28:32], "big")
        x_tile_origin = int.from_bytes(blob[32:36], "big")
        y_tile_origin = int.from_bytes(blob[36:40], "big")
        return (
            "j2k"
            if components
            and width > x_origin
            and height > y_origin
            and x_tile > 0
            and y_tile > 0
            and x_tile_origin <= x_origin
            and y_tile_origin <= y_origin
            else None
        )

    @staticmethod
    def text_extension(blob: bytes) -> bool:
        """Recognize complete UTF-8 text without control bytes or NULs."""
        if any(
            byte == 127 or (byte < 32 and byte not in (9, 10, 12, 13)) for byte in blob
        ):
            return False
        try:
            blob.decode("utf-8")
        except UnicodeDecodeError:
            return False
        return True

    @staticmethod
    def wave64_extension(blob: bytes) -> str:
        """Require the 64-bit WAVE size chunk and its complete entry table."""
        if len(blob) < 48 or blob[4:16] != b"\xff\xff\xff\xffWAVEds64":
            return "bin"
        size = int.from_bytes(blob[16:20], "little")
        entries = int.from_bytes(blob[44:48], "little")
        return "wav" if 28 + 12 * entries <= size <= len(blob) - 20 else "bin"

    @staticmethod
    def numpy_extension(blob: bytes) -> str | None:
        """Recognize the versioned NPY header without loading the array."""
        if blob[6:8] not in (b"\x01\x00", b"\x02\x00", b"\x03\x00"):
            return None
        start = 10 if blob[6] == 1 else 12
        end = start + int.from_bytes(blob[8:start], "little")
        return "npy" if start < end <= len(blob) and blob[end - 1] == 10 else None

    @staticmethod
    def image_brand(blob: bytes) -> str | None:
        """Read aligned codec brands from a complete, bounded ISO file-type box."""
        size = int.from_bytes(blob[:4], "big")
        start = 8
        if size == 1:
            size = int.from_bytes(blob[8:16], "big")
            start = 16
        elif size == 0:
            size = len(blob)
        if not start + 8 <= size <= min(4096, len(blob)) or size % 4:
            return "bin"
        brands = {blob[i : i + 4] for i in range(start + 8, size, 4)}
        brands.add(blob[start : start + 4])
        avif = not brands.isdisjoint((b"avif", b"avis"))
        heic = not brands.isdisjoint(
            (b"heic", b"heix", b"heim", b"heis", b"hevc", b"hevx", b"hevm", b"hevs")
        )
        if heic and avif:
            return "bin"
        if heic:
            return "heic"
        if avif:
            return "avif"
        return "heif" if not brands.isdisjoint((b"mif1", b"msf1")) else None

    @staticmethod
    def mpeg_layer(blob: bytes) -> str | None:
        """Read the MPEG audio layer after an optional length-delimited ID3v2 tag."""
        offset = 0
        if blob.startswith(b"ID3"):
            if (
                len(blob) < 10
                or blob[3] not in (2, 3, 4)
                or any(n & 0x80 for n in blob[6:10])
            ):
                return None
            for n in blob[6:10]:
                offset = (offset << 7) | n
            offset += 20 if blob[3] == 4 and blob[5] & 0x10 else 10
        header = blob[offset : offset + 4]
        if len(header) < 4 or header[0] != 255 or header[1] & 0xE0 != 0xE0:
            return None
        if header[1] & 0x18 == 8 or header[2] & 0x0C == 0x0C or header[2] >> 4 == 15:
            return None
        return {2: "mp2", 1: "mp3"}.get((header[1] >> 1) & 3)

    @staticmethod
    def extension_from_puremagic(blob: bytes) -> str | None:
        """Map puremagic's signature matches to stable filename extensions."""
        matcher = import_module("tboardfs._magic_signatures")
        matches = matcher.magic_matches(blob)
        if matches is None:
            return None
        mime_extensions = {
            "application/json": "json",
            "audio/aac": "aac",
            "audio/aiff": "aiff",
            "audio/flac": "flac",
            "audio/mp4": "m4a",
            "audio/ogg": "ogg",
            "audio/wav": "wav",
            "audio/webm": "webm",
            "audio/x-aiff": "aiff",
            "audio/x-hx-aac-adts": "aac",
            "audio/x-m4a": "m4a",
            "audio/x-flac": "flac",
            "audio/x-wav": "wav",
            "image/avif": "avif",
            "image/bmp": "bmp",
            "image/x-ms-bmp": "bmp",
            "image/gif": "gif",
            "image/jp2": "jp2",
            "image/x-jp2-codestream": "j2k",
            "image/jxl": "jxl",
            "image/x-exr": "exr",
            "image/svg+xml": "svg",
            "image/vnd.microsoft.icon": "ico",
            "image/x-icon": "ico",
            "image/png": "png",
            "image/tiff": "tiff",
            "image/webp": "webp",
            "text/plain": "txt",
            "video/mp4": "mp4",
            "video/ogg": "ogg",
            "video/quicktime": "mov",
            "video/webm": "webm",
            "video/x-ms-asf": "asf",
            "audio/x-ms-asf": "asf",
            "application/vnd.ms-asf": "asf",
            "video/x-matroska": "mkv",
            "video/x-msvideo": "avi",
        }
        aliases = {"jpeg": "jpg", "jpe": "jpg", "tif": "tiff"}
        for match in matches:
            if (
                blob.startswith(b"RIFF")
                and match.offset == 0
                and match.byte_match == b"RIFF"
            ):
                continue
            mime = (match.mime_type or "").lower()
            if mime == "audio/mpeg":
                if layer := _Classifier.mpeg_layer(blob):
                    return layer
                continue
            if extension := mime_extensions.get(mime):
                if extension in _UNTRUSTED_SIGNATURE_EXTENSIONS:
                    continue
                return extension
            extensions = match.extension
            if not extensions:
                continue
            if isinstance(extensions, str):
                extensions = [extensions]
            for value in extensions:
                extension = value.removeprefix(".").lower()
                if extension in aliases:
                    extension = aliases[extension]
                if extension in _UNTRUSTED_SIGNATURE_EXTENSIONS:
                    continue
                if extension.isascii() and extension.isalnum() and len(extension) <= 16:
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
