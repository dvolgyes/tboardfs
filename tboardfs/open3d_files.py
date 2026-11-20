"""Turn Open3D summary tensors into JSON indexes of native sidecar ranges."""

from collections.abc import Iterable
from typing import Any, cast

from google.protobuf.message import DecodeError, Message

from tboardfs.constants import DT_STRING
from tboardfs.leaf_spec import LeafSpec
from tboardfs.model import JsonEntry, SidecarEntry
from tboardfs.open3d_asset import OPEN3D_ASSETS, OPEN3D_DIRECTORY, OPEN3D_PLUGIN
from tboardfs.open3d_schema import GEOMETRY_PROPERTIES, GeometryIndex
from tboardfs.paths import _Paths
from tboardfs.run_files import RunFiles

_MAX_INDEX_BYTES = 1 << 16
_RANGE_FIELDS = (
    "start",
    "size",
    "masked_crc32c",
    "aux_start",
    "aux_size",
    "aux_masked_crc32c",
)


class Open3DFiles:
    """Build per-step manifests and link only to discovered, in-bounds assets."""

    @staticmethod
    def add(
        files: RunFiles,
        entries: list[JsonEntry],
        sidecars: tuple[SidecarEntry, ...],
        step_digits: int,
    ) -> None:
        """Add a manifest beside each recognized Open3D plugin JSON record."""
        assets = {
            entry.source_path.name: entry.size
            for entry in sidecars
            if entry.tab == OPEN3D_PLUGIN
        }
        for entry in entries:
            manifest = Open3DFiles._manifest(entry, assets, files.name_map)
            if manifest is not None:
                files.add(
                    OPEN3D_DIRECTORY,
                    entry.tag,
                    f"{_Paths.step_name(entry.step, step_digits)}.manifest.json",
                    LeafSpec("json", entry.wall_time, "data", manifest),
                )

    @staticmethod
    def _manifest(
        entry: JsonEntry, assets: dict[str, int], name_map: dict[str, str]
    ) -> dict[str, Any] | None:
        blob = Open3DFiles._tensor_bytes(entry)
        if blob is None:
            return None
        try:
            index = cast(Any, GeometryIndex.FromString(blob))
        except DecodeError:
            return None
        filename = index.batch_index.filename
        if index.version != 14 or not Open3DFiles._is_basename(filename):
            return None
        ranges = [
            {name: getattr(row, name) for name in _RANGE_FIELDS}
            for row in index.batch_index.start_size
        ]
        if not ranges or any(row["size"] == 0 for row in ranges):
            return None
        refs = Open3DFiles._references(index.property_references, entry.step)
        if refs is None:
            return None
        end = max(
            max(row["start"] + row["size"], row["aux_start"] + row["aux_size"])
            if row["aux_size"]
            else row["start"] + row["size"]
            for row in ranges
        )
        asset = None
        if end <= assets.get(filename, -1):
            name = _Paths.safe_component(filename, name_map=name_map)
            asset = "/".join((*OPEN3D_ASSETS, name))
        return {
            "asset": asset,
            "filename": filename,
            "property_references": refs,
            "start_size": ranges,
            "version": index.version,
        }

    @staticmethod
    def _tensor_bytes(entry: JsonEntry) -> bytes | None:
        if entry.payload.get("plugin_name") != OPEN3D_PLUGIN:
            return None
        tensor = entry.payload.get("tensor") or {}
        if (
            tensor.get("dtype") != DT_STRING
            or tensor.get("shape") != []
            or tensor.get("unknown_rank")
            or tensor.get("tensor_content")
        ):
            return None
        values = tensor.get("string_val") or []
        if len(values) == 1 and isinstance(values[0], bytes):
            blob: bytes = values[0]
            if 0 < len(blob) <= _MAX_INDEX_BYTES:
                return blob
        return None

    @staticmethod
    def _is_basename(filename: str) -> bool:
        return (
            filename.endswith(".msgpack")
            and "/" not in filename
            and "\0" not in filename
        )

    @staticmethod
    def _references(
        refs: Iterable[Message], step: int | float
    ) -> dict[str, int] | None:
        result: dict[str, int] = {}
        for reference in refs:
            ref = cast(Any, reference)
            prop = GEOMETRY_PROPERTIES.get(ref.geometry_property)
            if prop is None or prop.name in result or ref.step_ref >= step:
                return None
            result[prop.name] = ref.step_ref
        return result
