"""Compact boundaries and checksums of complete source frames."""

from array import array
from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass(slots=True)
class FrameIndex:
    """Frame end offsets and masked payload CRCs, in source order."""

    ends: "array[int]" = field(default_factory=lambda: array("Q"))
    checksums: "array[int]" = field(default_factory=lambda: array("I"))

    def copy(self) -> "FrameIndex":
        """Copy the compact prefix for an independent refresh result."""
        return FrameIndex(self.ends[:], self.checksums[:])

    def append(self, end: int, checksum: int) -> None:
        """Remember the ending offset and validated checksum."""
        self.ends.append(end)
        self.checksums.append(checksum)

    def __iter__(self) -> Iterator[tuple[int, int, int]]:
        start = 0
        for end, checksum in zip(self.ends, self.checksums, strict=True):
            yield start, end - start, checksum
            start = end

    def last(self) -> tuple[int, int, int] | None:
        """Return the last indexed frame, if any."""
        if not self.ends:
            return None
        start = self.ends[-2] if len(self.ends) > 1 else 0
        return start, self.ends[-1] - start, self.checksums[-1]
