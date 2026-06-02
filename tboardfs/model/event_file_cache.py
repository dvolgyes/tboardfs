from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tboardfs.model.binary_entry import BinaryEntry
from tboardfs.model.graph_entry import GraphEntry
from tboardfs.model.json_entry import JsonEntry


@dataclass
class EventFileCache:
    """In-memory parse cache for one event file.

    :ivar source_path: source event file
    :ivar processed_pos: byte offset parsed so far
    :ivar last_checked_size: source file size at last parse
    :ivar scalars: scalar series by tag
    :ivar scalar_mtimes: scalar modification times by tag
    :ivar histograms: histogram series by tag
    :ivar histogram_mtimes: histogram modification times by tag
    :ivar binaries: indexed binary payloads
    :ivar json_entries: indexed JSON plugin entries
    :ivar graphs: indexed graph payloads
    """

    source_path: Path
    processed_pos: int = 0
    last_checked_size: int = 0
    scalars: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    scalar_mtimes: dict[str, float] = field(default_factory=dict)
    histograms: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    histogram_mtimes: dict[str, float] = field(default_factory=dict)
    binaries: list[BinaryEntry] = field(default_factory=list)
    json_entries: list[JsonEntry] = field(default_factory=list)
    graphs: list[GraphEntry] = field(default_factory=list)
