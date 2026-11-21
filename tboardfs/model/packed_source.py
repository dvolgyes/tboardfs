"""Projection and random-access interface for packed binary rows."""

from collections.abc import Iterator
from typing import Protocol

from tboardfs.model.binary_entry import BinaryEntry
from tboardfs.model.graph_summary_entry import GraphSummaryEntry

type BinaryKey = tuple[str, str, str, str]
type BinaryProjection = tuple[BinaryKey, int | float, float, int]


class PackedSource(Protocol):
    """A table of binary rows with lightweight projections."""

    def __len__(self) -> int: ...

    def entry_at(self, index: int) -> BinaryEntry:
        """Resolve a full entry by its row ordinal."""
        ...

    def project(self) -> Iterator[BinaryProjection]:
        """Yield filename metadata and row ordinals in producer order."""
        ...

    def summaries(self) -> Iterator[GraphSummaryEntry]:
        """Yield graph summaries in producer order."""
        ...
