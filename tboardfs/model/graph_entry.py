from dataclasses import dataclass
from pathlib import Path

from tboardfs.frame_ref import FrameRef


@dataclass
class GraphEntry:
    """Index entry for an event graph payload.

    :ivar source_path: source event file
    :ivar wall_time: event wall time
    :ivar frame: verified source frame and generation
    """

    source_path: Path
    wall_time: float
    frame: FrameRef
