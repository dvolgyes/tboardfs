"""Identify parser and host classifier behavior for the durable index."""

from collections import deque
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import stat
import sys
from typing import cast
import zlib

import google.protobuf
from google.protobuf.descriptor import FileDescriptor
from google.protobuf.internal import api_implementation
import magic
import numpy as np
import xxhash

from tboardfs.index_magic import OBSERVATION_ERRORS, MagicHost
from tboardfs.open3d_schema import _FILE as _OPEN3D_SCHEMA
from tboardfs.proto import (
    compat_pb2,
    custom_scalar_layout_pb2,
    hparams_api_pb2,
    hparams_plugin_data_pb2,
    mesh_plugin_data_pb2,
)

_SCHEMA = 4
_MAX_RULE_BYTES = 128 << 20


@dataclass(frozen=True, slots=True)
class IndexPolicy:
    """Identify parser, classifier, and replay behavior; bump the schema when it changes."""

    id: str
    _host: MagicHost

    @classmethod
    def capture(cls, *, mountpoint: Path | None = None) -> "IndexPolicy | None":
        """Fingerprint before mounting; restart the process after host rule upgrades."""
        try:
            host = MagicHost.observe()
            if not host.files_unchanged():
                return None
            if mountpoint is not None and any(
                cls._crosses_mount(path, mountpoint) for path, _ in host.files
            ):
                return None
            digest = cls._digest(host)
            if host.files_unchanged():
                return cls(digest, host)
        except (*OBSERVATION_ERRORS, PackageNotFoundError):
            return None
        return None

    def unchanged(self) -> bool:
        """Check process compatibility without host I/O on a live mount."""
        return self._host.unchanged()

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
        raise ValueError("libmagic rule path traversal exceeds the probe bound")

    @classmethod
    def _digest(cls, host: MagicHost) -> str:
        """Fingerprint rule contents and decoder behavior with a bounded read budget."""
        remaining = _MAX_RULE_BYTES
        rules = []
        for path, signature in host.files:
            marker = "missing"
            if signature:
                marker = "directory"
                if stat.S_ISREG(signature[2]):
                    marker, remaining = cls._file_digest(path, signature, remaining)
            rules.append((str(path), marker))
        payload = (
            cls._decoder_revision(),
            cls._descriptor_revision(),
            host.environment,
            rules,
        )
        encoded = json.dumps(payload, separators=(",", ":")).encode("ascii")
        fingerprint: str = xxhash.xxh3_128_hexdigest(encoded)
        return fingerprint

    @staticmethod
    def _file_digest(
        path: Path, signature: tuple[int, ...], remaining: int
    ) -> tuple[str, int]:
        """Hash a regular file, rejecting changes and late substitutions."""
        size = signature[3]
        if size > remaining:
            raise ValueError("libmagic rule inputs exceed the hashing budget")
        flags = os.O_RDONLY | os.O_NONBLOCK
        with os.fdopen(os.open(path, flags), "rb") as stream:
            if MagicHost.signature(os.fstat(stream.fileno())) != signature:
                raise ValueError("libmagic input changed before hashing")
            digest = xxhash.xxh3_128()
            length = 0
            while chunk := stream.read(min(1 << 20, size + 1 - length)):
                digest.update(chunk)
                length += len(chunk)
            if (
                length != size
                or MagicHost.signature(os.fstat(stream.fileno())) != signature
            ):
                raise ValueError("libmagic input changed while hashing")
        return digest.hexdigest(), remaining - size

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
            version("python-magic"),
            magic.version(),
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
