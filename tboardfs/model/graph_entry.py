from dataclasses import dataclass
from pathlib import Path

from tboardfs.model.indexed_blob import _IndexedBlobEntry


@dataclass
class GraphEntry(_IndexedBlobEntry):
    """Index entry for an event graph payload.

    :ivar source_path: source event file
    :ivar wall_time: event wall time
    """

    source_path: Path
    wall_time: float
