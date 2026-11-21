"""A file leaf that resolves a packed binary row on demand."""

from dataclasses import dataclass

from tboardfs.model.binary_entry import BinaryEntry
from tboardfs.model.packed_source import PackedSource


@dataclass(slots=True, eq=False)
class PackedLeaf:
    """Resolve a typed file's row once, without changing its source snapshot."""

    source: PackedSource | BinaryEntry
    index: int
    wall_time: float

    def entry(self) -> BinaryEntry:
        """Cache the full entry on the first lookup of this leaf."""
        source = self.source
        if isinstance(source, BinaryEntry):
            return source
        self.source = entry = source.entry_at(self.index)
        return entry
