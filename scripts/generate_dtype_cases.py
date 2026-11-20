#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["tensorboard==2.21.0"]
# ///
"""Write controlled TensorProto encodings with TensorBoard's RecordWriter."""

import argparse
import struct
from pathlib import Path

from tensorboard.compat.proto import event_pb2
from tensorboard.summary.writer.record_writer import RecordWriter


def main() -> None:
    """Write the dtype corpus into a new or empty directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")
    event = event_pb2.Event(wall_time=1.0, step=3)
    cases: list[tuple[int, str, str, list[int | float]]] = [
        (1, "float_val", "f", [1.25, -2.5]),
        (2, "double_val", "d", [1 + 2**-50, -2.5]),
        (3, "int_val", "i", [-(2**31), 17]),
        (4, "int_val", "B", [0, 255]),
        (5, "int_val", "h", [-32768, 7]),
        (6, "int_val", "b", [-128, 127]),
        (8, "scomplex_val", "f", [1, 2, -3.5, -0.5]),
        (9, "int64_val", "q", [-(2**63), 17]),
        (10, "bool_val", "?", [False, True]),
        (11, "int_val", "b", [-128, 127]),
        (12, "int_val", "B", [0, 255]),
        (13, "int_val", "i", [-(2**31), 2**31 - 1]),
        (14, "half_val", "H", [0x3FC0, 0xC000]),
        (15, "int_val", "h", [-32768, 32767]),
        (16, "int_val", "H", [0, 65535]),
        (17, "int_val", "H", [65535, 1]),
        (18, "dcomplex_val", "d", [1, 2, -3.5, -0.5]),
        (19, "half_val", "H", [0x3E00, 0xC000, 0x7BFF]),
        (22, "uint32_val", "I", [0, 2**32 - 1]),
        (23, "uint64_val", "Q", [0, 2**64 - 1]),
    ]
    for dtype, field, fmt, values in cases:
        for mode in ["fields", "content"]:
            value = event.summary.value.add(tag=f"{mode}-{dtype}")
            value.metadata.plugin_data.plugin_name = "tensors"
            tensor = value.tensor
            tensor.dtype = dtype
            tensor.tensor_shape.dim.add(size=2 if dtype in (8, 18) else len(values))
            if mode == "fields":
                getattr(tensor, field).extend(values)
            else:
                tensor.tensor_content = struct.pack("<" + fmt * len(values), *values)
    strings = event.summary.value.add(tag="binary-string").tensor
    strings.dtype = 7
    strings.tensor_shape.dim.add(size=2)
    strings.string_val.extend([b"a\0b", b"\xff\xfe"])
    for dtype in [0, 20, 21, *range(24, 34), *range(101, 134), 999]:
        tensor = event.summary.value.add(tag=f"opaque-{dtype}").tensor
        tensor.dtype = dtype
        if dtype == 20:
            tensor.resource_handle_val.add(container="resources", name="weights")
        elif dtype == 21:
            variant = tensor.variant_val.add(type_name="nested", metadata=b"\0\xff")
            variant.tensors.add(dtype=7).string_val.append(b"nested\0bytes")
        elif 24 <= dtype <= 28:
            tensor.float8_val = b"\x81"
        else:
            tensor.tensor_content = b"\x01"
    with (output / "events.out.tfevents.dtypes").open("xb") as handle:
        RecordWriter(handle).write(event.SerializeToString())


if __name__ == "__main__":
    main()
