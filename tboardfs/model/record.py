from dataclasses import dataclass


@dataclass(frozen=True)
class Record:
    """One complete TFRecord payload and its source offsets.

    :ivar record_offset: TFRecord start byte offset
    :ivar payload_offset: protobuf payload byte offset
    :ivar payload_size: protobuf payload byte length
    :ivar total_size: full TFRecord byte length
    :ivar payload: protobuf payload bytes
    """

    record_offset: int
    payload_offset: int
    payload_size: int
    total_size: int
    payload: bytes
