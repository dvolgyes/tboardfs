"""Identify parser, classifier, and replay behavior for the durable index."""

import json
import os
import stat
import sys
import zlib
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import google.protobuf
import numpy as np
import xxhash
from google.protobuf.descriptor import FileDescriptor
from google.protobuf.internal import api_implementation

from tboardfs.open3d_schema import _FILE as _OPEN3D_SCHEMA
from tboardfs.proto import (
    compat_pb2,
    custom_scalar_layout_pb2,
    hparams_api_pb2,
    hparams_plugin_data_pb2,
    mesh_plugin_data_pb2,
)

_SCHEMA = 5
_MAX_DATA_BYTES = 128 << 20


def _signature_records(
    matches: Iterable[Any],
) -> tuple[tuple[object, ...], ...]:
    """Return stable tuples for the active byte signature records."""
    return tuple(
        (
            record.byte_match.hex(),
            record.offset,
            record.extension,
            record.mime_type,
            record.name,
        )
        for record in matches
    )


def _magic_data_path() -> Path:
    return Path(str(distribution("puremagic").locate_file("puremagic/magic_data.json")))


@dataclass(frozen=True, slots=True)
class IndexPolicy:
    """Identify parser, classifier, and replay behavior; bump the schema when it changes."""

    id: str

    @classmethod
    def capture(cls, *, mountpoint: Path | None = None) -> "IndexPolicy | None":
        """Fingerprint loaded classifier data and decoder behavior before mounting."""
        try:
            path = _magic_data_path()
            before = path.stat()
            signature = cls._signature(before)
            if (
                not path.is_absolute()
                or not stat.S_ISREG(before.st_mode)
                or before.st_size > _MAX_DATA_BYTES
            ):
                return None
            if mountpoint is not None and cls._crosses_mount(path, mountpoint):
                return None
            module = import_module("puremagic.main")
            classifier = cls._classifier_revision(module)
            if signature != cls._signature(path.stat()):
                return None
            payload = (
                cls._decoder_revision(),
                cls._descriptor_revision(),
                classifier,
            )
            encoded = json.dumps(payload, separators=(",", ":")).encode("ascii")
            return cls(xxhash.xxh3_128_hexdigest(encoded))
        except (OSError, ValueError, PackageNotFoundError):
            return None

    def unchanged(self) -> bool:
        """Report that puremagic has no runtime classifier settings."""
        return True

    @staticmethod
    def _crosses_mount(path: Path, mountpoint: Path) -> bool:
        """Check every traversal step, including symlink targets and parent steps."""
        current = Path("/")
        pending = deque(path.parts[1:])
        links = 0
        for _ in range(1024):
            if current.is_relative_to(mountpoint):
                return True
            if not pending:
                return False
            component = pending.popleft()
            if component == "..":
                current = current.parent
                continue
            candidate = current / component
            if candidate.is_relative_to(mountpoint):
                return True
            try:
                mode = candidate.lstat().st_mode
            except FileNotFoundError:
                return False
            if not stat.S_ISLNK(mode):
                current = candidate
                continue
            links += 1
            if links > 40:
                break
            target = candidate.readlink()
            parts = target.parts
            if target.is_absolute():
                current = Path("/")
                parts = parts[1:]
            pending.extendleft(reversed(parts))
        raise ValueError("classifier data path traversal exceeds the probe bound")

    @staticmethod
    def _signature(status: os.stat_result) -> tuple[int, ...]:
        return (
            status.st_dev,
            status.st_ino,
            status.st_mode,
            status.st_size,
            status.st_mtime_ns,
            status.st_ctime_ns,
        )

    @staticmethod
    def _classifier_revision(module: ModuleType) -> str:
        """Fingerprint the signature records used by byte-only classification."""
        signatures = (
            _signature_records(module.magic_header_array),
            _signature_records(module.magic_footer_array),
            tuple(
                (key.hex(), _signature_records(matches))
                for key, matches in sorted(module.multi_part_dict.items())
            ),
        )
        encoded = json.dumps(signatures, separators=(",", ":")).encode("ascii")
        return cast(str, xxhash.xxh3_128_hexdigest(encoded))

    @staticmethod
    def _decoder_revision() -> tuple[object, ...]:
        """Identify package revisions and runtime behavior relevant to decoding."""
        return (
            _SCHEMA,
            version("tboardfs"),
            sys.implementation.name,
            tuple(sys.version_info),
            sys.byteorder,
            np.__version__,
            google.protobuf.__version__,
            api_implementation.Type(),
            version("puremagic"),
            zlib.ZLIB_RUNTIME_VERSION,
        )

    @staticmethod
    def _descriptor_revision() -> tuple[str, ...]:
        """Hash the vendored binary descriptors actually used by the parser."""
        modules = (
            custom_scalar_layout_pb2,
            hparams_api_pb2,
            hparams_plugin_data_pb2,
            mesh_plugin_data_pb2,
        )
        descriptors = [
            cast(FileDescriptor, vars(module)["DESCRIPTOR"]) for module in modules
        ]
        descriptors.extend(
            cast(FileDescriptor, descriptor)
            for name, descriptor in sorted(vars(compat_pb2).items())
            if name.startswith("_FILE_")
        )
        descriptors.append(_OPEN3D_SCHEMA)
        return tuple(
            xxhash.xxh3_128_hexdigest(desc.serialized_pb) for desc in descriptors
        )
