"""The mount adapter maps raw FUSE file info onto TensorBoardFS handles."""

import json
from ctypes import memset, pointer, sizeof
from errno import ENOENT, ENOTDIR
from pathlib import Path
from stat import S_IFDIR, S_IFMT, S_IFREG
from types import SimpleNamespace

import mfusepy
import pytest
from test_open_handles import LOSS, _append

from tboardfs.constants import DEFAULT_SOURCE_CHECK_SECONDS
from tboardfs.file_attributes import _FileAttributes
from tboardfs.mount_binding import MountBinding
from tboardfs.mounted import MountedTensorBoardFS


def _mounted(
    tmp_path: Path, *, source_check_seconds: float = DEFAULT_SOURCE_CHECK_SECONDS
) -> tuple[MountedTensorBoardFS, Path]:
    event = tmp_path / "events.out.tfevents"
    _append(event, 0, 1)
    fs = MountedTensorBoardFS(
        tmp_path, refresh_age_seconds=0, source_check_seconds=source_check_seconds
    )
    return fs, event


def _binding(fs: MountedTensorBoardFS, monkeypatch: pytest.MonkeyPatch) -> MountBinding:
    """Initialize the real adapter without entering the kernel mount loop."""

    def init(
        binding: mfusepy.FUSE,
        operations: MountedTensorBoardFS,
        _mountpoint: str,
        **options: bool,
    ) -> None:
        binding.operations = operations
        binding.encoding = "utf-8"
        binding.errors = "strict"
        binding.raw_fi = options["raw_fi"]
        binding.use_ns = False

    monkeypatch.setattr(mfusepy.FUSE, "__init__", init)
    return MountBinding(fs, "/unused", raw_fi=True)


def _native_stat(
    binding: MountBinding, path: str, info: mfusepy.fuse_file_info | None = None
) -> mfusepy.c_stat:
    actual, expected = mfusepy.c_stat(), mfusepy.c_stat()
    actual_ptr, expected_ptr = pointer(actual), pointer(expected)
    info_ptr = pointer(info) if info is not None else None
    encoded = path.encode()
    memset(actual_ptr, 0xFF, sizeof(actual))
    assert binding.fgetattr(encoded, actual_ptr, info_ptr) == 0
    assert mfusepy.FUSE.fgetattr(binding, encoded, expected_ptr, info_ptr) == 0
    assert bytes(actual) == bytes(expected)
    return actual


def test_native_getattr_repopulates_stat_from_resolved_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Equal sizes and whole seconds cannot hide mode, owner or mtime changes."""
    fs, _ = _mounted(tmp_path)
    binding = _binding(fs, monkeypatch)
    directory = _FileAttributes(True, 0, 100.125, (1001, 1002))
    empty = directory._replace(directory=False)
    file = empty._replace(size=513)
    other_owner = file._replace(owner=(1003, 1004))
    later = file._replace(mtime=100.625)
    attrs = {LOSS: directory}

    def resolve(path: str, _file_handles: tuple[object, ...]) -> _FileAttributes:
        if path not in attrs:
            raise FileNotFoundError(ENOENT, "No such virtual file", path)
        return attrs[path]

    monkeypatch.setattr(fs, "_attributes", resolve)
    for current in (directory, empty, file, other_owner, later, directory, file):
        attrs[LOSS] = current
        result = _native_stat(binding, LOSS)
        assert S_IFMT(result.st_mode) == (S_IFDIR if current.directory else S_IFREG)
        assert result.st_size == current.size
        assert (result.st_uid, result.st_gid) == current.owner
        if hasattr(result, "st_blocks"):
            assert result.st_blocks == (current.size + 511) // 512
        if hasattr(result, "st_mtimespec"):
            assert result.st_mtimespec.tv_nsec == int((current.mtime % 1) * 10**9)

    del attrs[LOSS]
    with pytest.raises(OSError) as raised:
        binding.fgetattr(LOSS.encode(), pointer(mfusepy.c_stat()), None)
    assert raised.value.errno == ENOENT


def test_native_getattr_keeps_old_handle_after_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fs, event = _mounted(tmp_path, source_check_seconds=0)
    binding = _binding(fs, monkeypatch)
    info = mfusepy.fuse_file_info()
    assert fs.open(LOSS, info) == 0
    original = _native_stat(binding, LOSS, info)
    assert _native_stat(binding, LOSS).st_size == original.st_size

    _append(event, 2)

    current = _native_stat(binding, LOSS)
    assert current.st_size > original.st_size
    pinned = _native_stat(binding, LOSS, info)
    assert pinned.st_size == len(fs.read(LOSS, 1 << 20, 0, info))
    assert bytes(pinned) == bytes(original)


def test_open_uses_direct_io_and_keeps_a_version(tmp_path: Path) -> None:
    """The kernel must not cache pages or sizes across file versions."""
    fs, event = _mounted(tmp_path, source_check_seconds=0)
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
    fs, event = _mounted(tmp_path, source_check_seconds=0)
    info = SimpleNamespace(flags=0, fh=0, direct_io=0)
    fs.open(LOSS, info)
    _append(event, 2)

    fs.release(LOSS, info)

    assert len(json.loads(fs.read(LOSS, 1 << 20, 0, info))) == 3


def test_directory_hints_report_virtual_types_and_keep_listing_errors(
    tmp_path: Path,
) -> None:
    fs, _ = _mounted(tmp_path)
    assert ("scalars", S_IFDIR, 0) in fs.readdir("/", 0)
    assert list(fs.readdir("/scalars", 0)) == [
        (".", S_IFDIR, 0),
        ("..", S_IFDIR, 0),
        ("loss.json", S_IFREG, 0),
        ("loss.npz", S_IFREG, 0),
        ("loss.tsv", S_IFREG, 0),
    ]
    for path, code in ((LOSS, ENOTDIR), ("/missing", ENOENT)):
        with pytest.raises(OSError) as raised:
            list(fs.readdir(path, 0))
        assert raised.value.errno == code
