from dataclasses import dataclass
from pathlib import Path


@dataclass
class BinaryEntry:
    """Index entry for a lazily extracted binary summary payload.

    :ivar source_path: source event file
    :ivar tag: TensorBoard summary tag
    :ivar kind: virtual tab for the binary payload
    :ivar step: TensorBoard step
    :ivar wall_time: event wall time
    :ivar payload_offset: event payload byte offset
    :ivar payload_size: event payload byte length
    :ivar record_size: full TFRecord byte length
    :ivar extension: detected file extension
    :ivar blob_size: extracted blob byte length when known
    """

    source_path: Path
    tag: str
    kind: str
    step: int | float
    wall_time: float
    payload_offset: int
    payload_size: int
    record_size: int
    extension: str
    blob_size: int | None = None
