from dataclasses import dataclass
from pathlib import Path


@dataclass
class GraphEntry:
    """Index entry for an event graph payload.

    :ivar source_path: source event file
    :ivar name: virtual graph file name
    :ivar wall_time: event wall time
    :ivar payload_offset: event payload byte offset
    :ivar payload_size: event payload byte length
    """

    source_path: Path
    name: str
    wall_time: float
    payload_offset: int
    payload_size: int
