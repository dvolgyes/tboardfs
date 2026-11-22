"""Represent the complete stat attributes of a virtual file or directory."""

import stat
from typing import NamedTuple


class _FileAttributes(NamedTuple):
    """Values that determine all exported stat fields."""

    directory: bool
    size: int
    mtime: float
    owner: tuple[int, int]

    def as_dict(self) -> dict[str, int | float]:
        """Expand the resolved values into the attributes expected by mfusepy."""
        uid, gid = self.owner
        return {
            "st_mode": stat.S_IFDIR | 0o555 if self.directory else stat.S_IFREG | 0o444,
            "st_nlink": 2 if self.directory else 1,
            "st_size": self.size,
            "st_blocks": (self.size + 511) // 512,
            "st_ctime": self.mtime,
            "st_mtime": self.mtime,
            "st_atime": self.mtime,
            "st_uid": uid,
            "st_gid": gid,
        }
