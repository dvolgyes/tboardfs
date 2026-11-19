"""The source generation and verified boundaries of one indexed TFRecord."""

from dataclasses import dataclass

from tboardfs.file_status import FileIdentity


@dataclass(frozen=True, slots=True)
class FrameRef:
    """Enough evidence to detect a changed indexed frame on a later read."""

    identity: FileIdentity
    offset: int
    size: int
    checksum: int

    @property
    def payload_offset(self) -> int:
        """Return the starting byte after the TFRecord header."""
        return self.offset + 12

    @property
    def payload_size(self) -> int:
        """Return the payload size without either framing checksum or length."""
        return self.size - 16
