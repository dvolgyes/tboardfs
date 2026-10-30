from dataclasses import dataclass, field
from pathlib import Path

from tboardfs.model.event_file_cache import EventFileCache
from tboardfs.model.sidecar_entry import SidecarEntry


@dataclass
class RunCache:
    """In-memory cache for one TensorBoard run directory.

    :ivar run_parts: virtual run path parts
    :ivar files: event-file caches by source path
    :ivar refreshed_at: last refresh timestamp
    :ivar version: data version, unique among all runs; changes with the data
    :ivar sidecars: projector/profile files found at the last refresh
    """

    run_parts: tuple[str, ...]
    files: dict[Path, EventFileCache] = field(default_factory=dict)
    refreshed_at: float = 0.0
    version: int = 0
    sidecars: tuple[SidecarEntry, ...] = ()
