from dataclasses import dataclass
from pathlib import Path


@dataclass
class SidecarEntry:
    """Index entry for a projector, profile or hparams sidecar file.

    :ivar source_path: source sidecar file
    :ivar tab: virtual tab name
    :ivar rel_parts: virtual relative path parts
    :ivar size: source file size when the run was refreshed
    :ivar mtime_ns: source file mtime when the run was refreshed
    """

    source_path: Path
    tab: str
    rel_parts: tuple[str, ...]
    size: int
    mtime_ns: int
