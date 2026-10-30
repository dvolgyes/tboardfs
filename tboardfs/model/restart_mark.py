from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RestartMark:
    """One event that TensorBoard's restart purge looks at.

    :ivar kind: ``version`` for a file version, ``start`` for SessionLog.START
    :ivar value: parsed file version, or the START step (a missing step is 0)
    :ivar counts: rows of the file before a START, by (table, tag); the
        ``binaries`` and ``json_entries`` lists use the tag ""
    """

    kind: str
    value: float
    counts: Mapping[tuple[str, str], int] = field(default_factory=dict)
