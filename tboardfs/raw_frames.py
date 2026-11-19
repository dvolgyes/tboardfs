"""Expose each validated source frame without interpreting its payload."""

from bisect import bisect_left
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar

from tboardfs.file_status import FileIdentity
from tboardfs.frame_index import FrameIndex
from tboardfs.frame_ref import FrameRef
from tboardfs.indexed_frame import IndexedFrame
from tboardfs.model import RunCache
from tboardfs.paths import _DirectoryNode, _Paths

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
    def add(directory: dict[str, Any], run: RunCache, name_map: dict[str, str]) -> None:
        """Attach one lazy raw directory per source with validated frames."""
        directory["view"] = "raw"
        for path, cache in run.files.items():
            basename = _Paths.safe_component(path.name, name_map=name_map)
            progress = cache.progress
            if progress.frames.ends:
                source = _DirectoryNode()
                source.update(
                    children=_RawFrameChildren(
                        path, progress.identity, progress.frames
                    ),
                    view="raw",
                    mtime=0.0,
                )
                directory["children"][basename] = source

    @staticmethod
    def read(node: dict[str, Any]) -> bytes:
        """Return the original header, payload, and footer after validation."""
        record = IndexedFrame.read(node["source"], node["frame"])
        return b"".join((record.header, record.payload, record.footer))
