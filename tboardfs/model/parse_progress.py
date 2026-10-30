from dataclasses import dataclass

import numpy as np


@dataclass
class ParseProgress:
    """Where the parse of one event file stopped, and what it carries on.

    :ivar processed_pos: byte offset parsed so far
    :ivar last_checked_size: source file size at last parse
    :ivar identity: device, inode and first bytes of the parsed file
    :ivar first_time: wall time that relative times count from
    :ivar epoch: last epoch scalar, carried into appended rows
    :ivar checked_status: device, inode, size, mtime and ctime of the file
        when it was last checked; while they stay the same, the file is not
        opened again
    """

    processed_pos: int = 0
    last_checked_size: int = 0
    identity: tuple[int, int, bytes] = (0, 0, b"")
    first_time: float | None = None
    epoch: object = np.nan
    checked_status: tuple[int, ...] = ()

    def continues_in(self, identity: tuple[int, int, bytes]) -> bool:
        """Return true when ``identity`` names this file, possibly grown.

        A replaced file can reuse the inode, so its first bytes must match too.
        """
        device, inode, head = self.identity
        return identity[:2] == (device, inode) and identity[2].startswith(head)
