"""Expose each validated source frame without interpreting its payload."""

import os
from bisect import bisect_left
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar

from tboardfs.event_source import EventSource
from tboardfs.file_status import FileIdentity
from tboardfs.frame_index import FrameIndex
from tboardfs.frame_ref import FrameRef
from tboardfs.indexed_frame import IndexedFrame
from tboardfs.model import EventFileCache, RunCache, TFRecordError
from tboardfs.paths import _DirectoryNode, _Paths
from tboardfs.tfrecord import _iter_tfrecords

_OFFSET_DIGITS = 20
_EXTENSION = ".tfrecord"


class _RawFrameChildren(Mapping[str, dict[str, Any]]):
    """Expose frame names and nodes on demand from a published frame index."""

    __slots__: ClassVar[tuple[str, ...]] = ("frames", "identity", "source")

    source: Path
    identity: FileIdentity
    frames: FrameIndex

    def __init__(
        self, source: Path, identity: FileIdentity, frames: FrameIndex
    ) -> None:
        self.source = source
        self.identity = identity
        self.frames = frames

    def __len__(self) -> int:
        return len(self.frames.ends)

    def __iter__(self) -> Iterator[str]:
        offset = 0
        for end in self.frames.ends:
            yield self._name(offset)
            offset = end

    def iter_nodes(self) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yield raw files in ascending offset order without lookup or sorting."""
        for offset, size, checksum in self.frames:
            yield self._name(offset), self._node(offset, size, checksum)

    def __getitem__(self, name: str) -> dict[str, Any]:
        match = self._index(name)
        if match is None:
            raise KeyError(name)
        index, offset = match
        return self._node(
            offset, self.frames.ends[index] - offset, self.frames.checksums[index]
        )

    def __contains__(self, name: object) -> bool:
        return self._index(name) is not None

    @staticmethod
    def _name(offset: int) -> str:
        return f"{offset:0{_OFFSET_DIGITS}d}{_EXTENSION}"

    def _index(self, name: object) -> tuple[int, int] | None:
        if (
            not isinstance(name, str)
            or len(name) != _OFFSET_DIGITS + len(_EXTENSION)
            or not name.endswith(_EXTENSION)
        ):
            return None
        digits = name[:_OFFSET_DIGITS]
        if not digits.isascii() or not digits.isdecimal():
            return None
        offset = int(digits)
        ends = self.frames.ends
        if not ends:
            return None
        if offset == 0:
            return 0, offset
        previous = bisect_left(ends, offset, hi=len(ends) - 1)
        if previous == len(ends) - 1 or ends[previous] != offset:
            return None
        return previous + 1, offset

    def _node(self, offset: int, size: int, checksum: int) -> dict[str, Any]:
        return {
            "type": "raw",
            "source": self.source,
            "frame": FrameRef(self.identity, offset, size, checksum),
        }


class RawFrames:
    """Build and render the exact raw view independently of typed restart purges."""

    @staticmethod
    def index_source(path: Path) -> _DirectoryNode:
        """Index complete frames on a checked descriptor, without decoding events."""
        frames = FrameIndex()
        with EventSource.open(path) as handle:
            identity = FileIdentity.of(os.fstat(handle.fileno()))
            try:
                for record in _iter_tfrecords(handle):
                    frames.append(
                        record.record_offset + record.total_size, record.payload_crc
                    )
            except TFRecordError as error:
                raise TFRecordError(
                    f"{path}: {error}", corrupt=error.corrupt
                ) from error

        directory = _DirectoryNode()
        directory["view"] = "raw"
        RawFrames._add_source(directory, path, identity, frames)
        return directory

    @staticmethod
    def add(directory: dict[str, Any], run: RunCache, name_map: dict[str, str]) -> None:
        """Attach one lazy raw directory per source with validated frames."""
        directory["view"] = "raw"
        for cache in run.files.values():
            progress = cache.progress
            RawFrames._add_source(
                directory,
                cache.source_path,
                progress.identity,
                progress.frames,
                name_map=name_map,
            )

    @staticmethod
    def extend(
        directory: dict[str, Any], cache: EventFileCache, name_map: dict[str, str]
    ) -> _DirectoryNode:
        """Publish the appended frame index without replacing other raw listings."""
        raw = _DirectoryNode()
        raw.update(view="raw", children=directory["children"].copy())
        progress = cache.progress
        RawFrames._add_source(
            raw,
            cache.source_path,
            progress.identity,
            progress.frames,
            name_map=name_map,
        )
        return raw

    @staticmethod
    def _add_source(
        directory: dict[str, Any],
        path: Path,
        identity: FileIdentity,
        frames: FrameIndex,
        *,
        name_map: dict[str, str] | None = None,
    ) -> None:
        basename = _Paths.safe_component(path.name, name_map=name_map)
        if not frames.ends:
            return
        source = _DirectoryNode()
        source.update(
            children=_RawFrameChildren(path, identity, frames),
            view="raw",
            mtime=0.0,
        )
        directory["children"][basename] = source

    @staticmethod
    def read(node: dict[str, Any]) -> bytes:
        """Return the original header, payload, and footer after validation."""
        return IndexedFrame.read(node["source"], node["frame"], raw=True)
