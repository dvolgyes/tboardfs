from dataclasses import dataclass, field
from pathlib import Path

from tboardfs.model.event_file_cache import EventFileCache


@dataclass
class RunCache:
    """In-memory cache for one TensorBoard run directory.

    :ivar run_parts: virtual run path parts
    :ivar files: event-file caches by source path
    :ivar scalar_links: in-memory scalar format aliases
    :ivar refreshed_at: last refresh timestamp
    """

    run_parts: tuple[str, ...]
    files: dict[Path, EventFileCache] = field(default_factory=dict)
    scalar_links: set[str] = field(default_factory=set)
    refreshed_at: float = 0.0
