from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class _TensorContentSource(Protocol):
    """Read-only evidence for a deferred tensor buffer."""

    @property
    def source_path(self) -> Path:
        """Return the file containing the indexed buffer."""
        ...

    @property
    def extent(self) -> tuple[int, int, int]:
        """Return its absolute offset, size, and checksum."""
        ...

    def read(self) -> bytes:
        """Return verified bytes from the indexed buffer."""
        ...


@dataclass
class JsonEntry:
    """Index entry for plugin metadata rendered as JSON.

    :ivar tag: TensorBoard summary tag
    :ivar tab: virtual tab name
    :ivar step: TensorBoard step
    :ivar wall_time: event wall time
    :ivar payload: JSON-safe payload
    :ivar tensor_extent: verified absolute offset, size, and masked CRC32C
    :ivar tensor_source: deferred content from the verified source
    """

    tag: str
    tab: str
    step: int | float
    wall_time: float
    payload: dict[str, Any]
    tensor_extent: tuple[int, int, int] | None = field(
        default=None, compare=False, repr=False
    )
    tensor_source: _TensorContentSource | None = field(
        default=None, compare=False, repr=False
    )
