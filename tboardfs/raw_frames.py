"""Expose each validated source frame without interpreting its payload."""

from typing import Any

from tboardfs.frame_ref import FrameRef
from tboardfs.indexed_frame import IndexedFrame
from tboardfs.model import RunCache
from tboardfs.paths import _Paths
from tboardfs.run_files import RunFiles


class RawFrames:
    """Build and render the exact raw view independently of typed restart purges."""

    @staticmethod
    def add(files: RunFiles, run: RunCache) -> None:
        """Add one export per validated frame."""
        for path, cache in run.files.items():
            basename = _Paths.safe_component(path.name, name_map=files.name_map)
            for offset, size, checksum in cache.progress.frames:
                files.add_fixed(
                    ("raw", basename, f"{offset:020d}.tfrecord"),
                    {
                        "type": "raw",
                        "source": path,
                        "frame": FrameRef(
                            cache.progress.identity, offset, size, checksum
                        ),
                    },
                )

    @staticmethod
    def read(node: dict[str, Any]) -> bytes:
        """Return the original header, payload, and footer after validation."""
        record = IndexedFrame.read(node["source"], node["frame"])
        return b"".join((record.header, record.payload, record.footer))
