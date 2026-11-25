"""Keep resolved run names and compact values until a file node is needed."""

from collections.abc import Iterator, MutableMapping
from typing import Any, ClassVar

from tboardfs.leaf_spec import LeafSpec, LeafValue

type _LogicalFile = tuple[
    tuple[str, ...], str | None, str, bool, str, int | float | None, str
]


class RunListing(MutableMapping[str, dict[str, Any]]):
    """Directory children with separate provenance for their direct files."""

    __slots__: ClassVar[tuple[str, ...]] = ("entries", "logical")

    entries: dict[str, LeafValue]
    logical: dict[str, _LogicalFile]

    def __init__(
        self,
        entries: dict[str, LeafValue],
        *,
        logical: dict[str, _LogicalFile] | None = None,
    ) -> None:
        self.entries = entries
        self.logical = {} if logical is None else logical

    def __getitem__(self, name: str) -> dict[str, Any]:
        return LeafSpec.node(self.entries[name])

    def __setitem__(self, name: str, node: dict[str, Any]) -> None:
        self.entries[name] = node

    def __delitem__(self, name: str) -> None:
        del self.entries[name]
        self.logical.pop(name, None)

    def __iter__(self) -> Iterator[str]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, name: object) -> bool:
        return name in self.entries

    def set_leaf(self, name: str, value: LeafValue) -> None:
        """Write an entry without constructing its file node."""
        self.entries[name] = value

    def stored(self, name: str) -> LeafValue | None:
        """Look up a child without constructing its file node."""
        return self.entries.get(name)

    def copy(self) -> "RunListing":
        """Copy child entries and provenance without materializing leaves."""
        return RunListing(self.entries.copy(), logical=self.logical.copy())
