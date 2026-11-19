from dataclasses import dataclass
from pathlib import Path

from tboardfs.model.indexed_blob import _IndexedBlobEntry


@dataclass
class BinaryEntry(_IndexedBlobEntry):
    """Index entry for a lazily extracted binary summary payload.

    :ivar source_path: source event file
    :ivar tag: TensorBoard summary tag
    :ivar kind: virtual tab for the binary payload
    :ivar step: TensorBoard step
    :ivar wall_time: event wall time
    :ivar extension: detected file extension
    :ivar part: string_val index of a TF2 media item, or None for the value blob
    :ivar suffix: file-name suffix after the step, for one of several items
    """

    source_path: Path
    tag: str
    kind: str
    step: int | float
    wall_time: float
    extension: str
    value_ordinal: int
    part: int | None = None
    suffix: str = ""
