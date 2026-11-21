"""The parsed index and resume state of one event file."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tboardfs.model.graph_entry import GraphEntry
from tboardfs.model.json_entry import JsonEntry
from tboardfs.model.packed_binaries import BinaryEntries
from tboardfs.model.parse_progress import ParseProgress
from tboardfs.model.restart_mark import RestartMark


@dataclass
class EventFileCache:
    """In-memory parse cache for one event file.

    :ivar source_path: source event file
    :ivar progress: where the parse stopped, for appended records
    :ivar scalars: scalar series by tag
    :ivar scalar_mtimes: scalar modification times by tag
    :ivar histograms: histogram series by tag
    :ivar histogram_mtimes: histogram modification times by tag
    :ivar binaries: indexed binary payloads
    :ivar json_entries: indexed JSON plugin entries
    :ivar graphs: indexed graph payloads
    :ivar restarts: file version and SessionLog.START events in file order
    """

    source_path: Path
    progress: ParseProgress = field(default_factory=ParseProgress)
    scalars: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    scalar_mtimes: dict[str, float] = field(default_factory=dict)
    histograms: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    histogram_mtimes: dict[str, float] = field(default_factory=dict)
    binaries: BinaryEntries = field(default_factory=list)
    json_entries: list[JsonEntry] = field(default_factory=list)
    graphs: list[GraphEntry] = field(default_factory=list)
    restarts: list[RestartMark] = field(default_factory=list)
