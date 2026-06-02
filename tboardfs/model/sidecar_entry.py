from dataclasses import dataclass
from pathlib import Path


@dataclass
class SidecarEntry:
    """Index entry for a projector/profile sidecar file.

    :ivar source_path: source sidecar file
    :ivar tab: virtual tab name
    :ivar rel_parts: virtual relative path parts
    """

    source_path: Path
    tab: str
    rel_parts: tuple[str, ...]
