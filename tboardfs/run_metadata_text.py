"""Render the profile text that TensorBoard shows for a RunMetadata."""

from typing import Any, cast

from google.protobuf.message import DecodeError
from loguru import logger

from tboardfs.model import BinaryEntry
from tboardfs.proto import compat_pb2


def run_metadata_text(entry: BinaryEntry, blob: bytes) -> bytes | None:
    """Return a serialized RunMetadata in protobuf text format.

    The TensorBoard 2.21 graph plugin shows this text, str() of the message,
    as the profile of a graph_run_metadata or graph_tagged_run_metadata
    summary. A RunMetadata that TensorBoard cannot parse gives None and a
    warning.
    """
    run_metadata = cast(Any, getattr(compat_pb2, "RunMetadata")())
    try:
        run_metadata.ParseFromString(blob)
    except DecodeError as error:
        logger.warning(
            "run metadata {} at step {} in {} has no profile file: {}",
            entry.tag,
            entry.step,
            entry.source_path,
            error,
        )
        return None
    return str(run_metadata).encode()
