from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tboardfs.file_status import FileIdentity
from tboardfs.frame_index import FrameIndex

_UNINDEXED = FileIdentity(0, 0, None)


@dataclass
class _ParseProvenance:
    """Record whether the index reused a prefix or would hide diagnostics."""

    resumed: bool = False
    cacheable: bool = field(default=True, compare=False, repr=False)


@dataclass
class ParseProgress:
    """Where the parse of one event file stopped, and what it carries on.

    :ivar processed_pos: byte offset parsed so far
    :ivar last_checked_size: source file size at last parse
    :ivar identity: device, inode and birth time if the filesystem supplies it
    :ivar first_time: wall time that relative times count from
    :ivar epoch: last epoch scalar, carried into appended rows
    :ivar checked_status: device, inode, size, mtime and ctime of the file
        when it was last checked; while they stay the same, the file is not
        opened again
    :ivar provenance: origin and persistent eligibility of this index
    """

    processed_pos: int = 0
    last_checked_size: int = 0
    identity: FileIdentity = _UNINDEXED
    first_time: float | None = None
    epoch: object = np.nan
    checked_status: tuple[int | None, ...] = ()
    initial_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    effective_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    frames: FrameIndex = field(default_factory=FrameIndex)
    provenance: _ParseProvenance = field(default_factory=_ParseProvenance)

    @property
    def resumed(self) -> bool:
        """Return whether this parse extended the supplied index."""
        return self.provenance.resumed
