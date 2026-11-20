"""Exercise TensorProto field preservation in the Event parser."""

from tboardfs.proto import compat_pb2
from tboardfs.summary import parse_event


def test_tensor_keeps_all_supported_values_regardless_of_dtype() -> None:
    event = compat_pb2.Event()
    tensor = event.summary.value.add(tag="mixed").tensor
    tensor.dtype = compat_pb2.DT_STRING
    tensor.tensor_shape.dim.add(size=-1)
    tensor.tensor_shape.unknown_rank = True
    tensor.tensor_content = b"content"
    tensor.float_val.append(0.0)
    tensor.double_val.append(1.5)
    tensor.int_val.append(-1)
    tensor.string_val.extend([b"", b"\xff"])
    tensor.scomplex_val.extend([0.0, -2.0])
    tensor.int64_val.append(-(1 << 63))
    tensor.bool_val.append(False)
    tensor.dcomplex_val.extend([1.0, -3.0])
    tensor.half_val.append(0)
    tensor.uint32_val.append((1 << 32) - 1)
    tensor.uint64_val.append((1 << 64) - 1)
    tensor.float8_val = b"ignored"
    unknown_field_100 = b"\xa0\x06\x01"
    tensor.MergeFromString(unknown_field_100)

    assert parse_event(event.SerializeToString())["values"] == [
        {
            "tag": "mixed",
            "tensor": {
                "dtype": compat_pb2.DT_STRING,
                "shape": [-1],
                "unknown_rank": True,
                "tensor_content": b"content",
                "values": {
                    5: [0.0],
                    6: [1.5],
                    7: [-1],
                    9: [0.0, -2.0],
                    10: [-(1 << 63)],
                    11: [False],
                    12: [1.0, -3.0],
                    13: [0],
                    16: [(1 << 32) - 1],
                    17: [(1 << 64) - 1],
                },
                "string_val": [b"", b"\xff"],
            },
        }
    ]


def test_tensor_preserves_explicit_empty_shape() -> None:
    event = compat_pb2.Event()
    event.summary.value.add(tag="absent").tensor.SetInParent()
    event.summary.value.add(tag="empty").tensor.tensor_shape.SetInParent()

    assert parse_event(event.SerializeToString())["values"] == [
        {"tag": "absent", "tensor": {"dtype": compat_pb2.DT_INVALID}},
        {"tag": "empty", "tensor": {"dtype": compat_pb2.DT_INVALID, "shape": []}},
    ]
