from dataclasses import dataclass

from tboardfs.file_status import FileIdentity
from tboardfs.frame_ref import FrameRef


@dataclass(frozen=True)
class Record:
    """One complete TFRecord payload and its source offsets.

    :ivar record_offset: TFRecord start byte offset
    :ivar payload_offset: protobuf payload byte offset
    :ivar payload_size: protobuf payload byte length
    :ivar total_size: full TFRecord byte length
    :ivar payload: protobuf payload bytes
    :ivar payload_crc: validated masked payload checksum
    :ivar header: exact length and length checksum bytes
    :ivar footer: exact payload checksum bytes
    """

    record_offset: int
    payload_offset: int
    payload_size: int
    total_size: int
    payload: bytes
    payload_crc: int
    header: bytes
    footer: bytes

    def reference(
        self,
        identity: FileIdentity,
        *,
        fingerprint: tuple[int | None, ...] | None = None,
    ) -> FrameRef:
        """Keep the generation and frame evidence needed for a lazy read."""
        return FrameRef(
            identity, self.record_offset, self.total_size, self.payload_crc, fingerprint
        )
