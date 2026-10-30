"""The mount adapter maps raw FUSE file info onto TensorBoardFS handles."""

import json
from pathlib import Path
from types import SimpleNamespace

from tboardfs.mounted import MountedTensorBoardFS

from test_open_handles import LOSS, _append


def _mounted(tmp_path: Path) -> tuple[MountedTensorBoardFS, Path]:
    event = tmp_path / "events.out.tfevents"
    _append(event, 0, 1)
    return MountedTensorBoardFS(tmp_path, refresh_age_seconds=0), event


def test_open_uses_direct_io_and_keeps_a_version(tmp_path: Path) -> None:
    """The kernel must not cache pages or sizes across file versions."""
    fs, event = _mounted(tmp_path)
    info = SimpleNamespace(flags=0, fh=0, direct_io=0)

    assert fs.open(LOSS, info) == 0
    _append(event, 2)

    assert info.direct_io == 1
    data = fs.read(LOSS, 1 << 20, 0, info)
    assert len(json.loads(data)) == 2
    assert fs.getattr(LOSS, info)["st_size"] == len(data)
    assert fs.getattr(LOSS, None)["st_size"] > len(data)


def test_release_frees_the_version_of_the_file_info(tmp_path: Path) -> None:
    """After release the same file info number reads the current version."""
    fs, event = _mounted(tmp_path)
    info = SimpleNamespace(flags=0, fh=0, direct_io=0)
    fs.open(LOSS, info)
    _append(event, 2)

    fs.release(LOSS, info)

    assert len(json.loads(fs.read(LOSS, 1 << 20, 0, info))) == 3
