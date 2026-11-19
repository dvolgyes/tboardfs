"""Framing failure raised by the TFRecord reader."""


class TFRecordError(ValueError):
    """A TFRecord cannot be framed.

    Attributes:
        corrupt: True for a checksum mismatch or excessive frame size;
            false when the last record is only incomplete.
    """

    corrupt: bool

    def __init__(self, message: str, *, corrupt: bool) -> None:
        super().__init__(message)
        self.corrupt = corrupt
