from dataclasses import dataclass


@dataclass(frozen=True)
class Field:
    """Parsed protobuf field metadata retained for compatibility.

    :ivar number: protobuf field number
    :ivar wire_type: protobuf wire type
    :ivar value: decoded protobuf field value
    """

    number: int
    wire_type: int
    value: int | float | bytes
