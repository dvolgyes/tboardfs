"""Each failure gives the errno that a file system user expects."""

from collections.abc import Callable
import errno
import os
from pathlib import Path

import pytest

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2

from test_incremental_parse import _frame

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 32


def _image_log(tmp_path: Path) -> tuple[TensorBoardFS, Path]:
    event = compat_pb2.Event(step=1, wall_time=3.0)
    image = event.summary.value.add(tag="picture").image
    image.encoded_image_string = PNG
    path = tmp_path / "events.out.tfevents.host"
    path.write_bytes(_frame(event))
    return TensorBoardFS(tmp_path), path


@pytest.mark.parametrize(
    "call",
    [
        lambda fs, path: fs.getattr(path),
        lambda fs, path: fs.readdir(path),
        lambda fs, path: fs.open(path, os.O_RDONLY),
        lambda fs, path: fs.read(path, 10, 0),
    ],
    ids=["getattr", "readdir", "open", "read"],
)
@pytest.mark.parametrize("path", ["/missing", "/images/picture/000001.png/below"])
def test_a_missing_path_is_enoent(
    tmp_path: Path, call: Callable[[TensorBoardFS, str], object], path: str
) -> None:
    fs, _ = _image_log(tmp_path)

    with pytest.raises(OSError) as raised:
        call(fs, path)
    assert raised.value.errno == errno.ENOENT


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data.replace(b"picture", b"another"),
        lambda data: data.replace(bytes([len(PNG)]) + PNG, bytes([99]) + PNG),
        lambda data: data[:-20],
    ],
    ids=["other tag", "not a message", "shorter"],
)
def test_an_image_of_a_changed_event_file_is_eio(
    tmp_path: Path, change: Callable[[bytes], bytes]
) -> None:
    """The file is listed, but its record is gone from the event file."""
    fs, path = _image_log(tmp_path)
    assert fs.readdir("/images/picture")[2:] == ["000001.png"]
    status = path.stat()
    path.write_bytes(change(path.read_bytes()))
    # The same mtime, before the refresh age: the mount does not index it again.
    os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns))

    with pytest.raises(OSError) as raised:
        fs.read("/images/picture/000001.png", 1 << 20, 0)
    assert raised.value.errno == errno.EIO
