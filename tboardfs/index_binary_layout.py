"""Wire layout of binary rows in the persistent index."""

import struct

type Label = tuple[str, str, str, str]

_HEADER = struct.Struct("<III")
_LABEL = struct.Struct("<IIII")
_FIELDS = "dIIIIIIII"
_SIGNED = struct.Struct("<Bq" + _FIELDS)
_UNSIGNED = struct.Struct("<BQ" + _FIELDS)
_FLOAT = struct.Struct("<Bd" + _FIELDS)
_SPAN, _PART, _GRAPH, _CONVERTS, _PROFILES, _UINT, _REAL = (1, 2, 4, 8, 16, 32, 64)
_TFRECORD_OVERHEAD = 16
