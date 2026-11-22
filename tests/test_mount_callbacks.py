"""Exercise the native FUSE2 stat slots without mounting a kernel filesystem."""

import logging
import signal
from collections.abc import Callable
from ctypes import memset, pointer, sizeof
from errno import EACCES, EFAULT, EINVAL, ENOENT
from pathlib import Path
from stat import S_IFMT, S_IFREG
from typing import Any

import mfusepy
import pytest
from test_open_handles import LOSS, _append

from tboardfs import _mount_callbacks, mount_binding
from tboardfs._mount_session import mount
from tboardfs.file_attributes import _FileAttributes
from tboardfs.mounted import MountedTensorBoardFS


@pytest.fixture
def native_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """Require the actual optional bridge, and reject any unexpected fallback."""
    if not _mount_callbacks._supported() or _mount_callbacks._load_extension() is None:
        pytest.skip("optional native FUSE2 stat callbacks are unavailable")
    monkeypatch.setattr(
        mount_binding.MountBinding,
        "__init__",
        lambda *_args, **_kw: pytest.fail("unexpected fallback"),
    )


def _mount_with_loop(
    monkeypatch: pytest.MonkeyPatch,
    fs: MountedTensorBoardFS,
    loop: Callable[[Any], int],
    *,
    foreground: bool = True,
) -> None:
    """Substitute the FUSE loop/context while calling the registered ABI slots."""
    context = mfusepy.fuse_context()
    context_pointer = pointer(context)

    def main(argc: int, argv: Any, ops_pointer: Any, _size: int, data: Any) -> int:
        assert (b"-f" in argv[:argc]) is foreground
        assert data.value
        context.private_data = data.value
        ops = ops_pointer.contents
        context.private_data = ops.init(pointer(mfusepy.fuse_conn_info()))
        assert context.private_data == data.value
        return loop(ops)

    monkeypatch.setattr(mfusepy._libfuse, "fuse_get_context", lambda: context_pointer)
    monkeypatch.setattr(mfusepy, "fuse_main_real", main)
    mount(fs, "/unused", foreground=foreground, ro=True, raw_fi=True, attr_timeout=0)


def _stat_buffer() -> mfusepy.c_stat:
    result = mfusepy.c_stat()
    memset(pointer(result), 0xFF, sizeof(result))
    return result


@pytest.mark.parametrize(
    "attrs",
    [
        pytest.param(_FileAttributes(False, -513, -0.25, (-1, -2)), id="negative"),
        pytest.param(
            _FileAttributes(True, (1 << 70) + 513, (1 << 40) + 0.25, (1 << 40, 9)),
            id="wide",
        ),
        pytest.param(
            _FileAttributes(False, 7, float(1 << 63), (2, 3)), id="wrapped-seconds"
        ),
        pytest.param(
            _FileAttributes(False, 7, -float((1 << 63) + 4096), (2, 3)),
            id="wrapped-negative-seconds",
        ),
    ],
)
def test_native_stat_matches_mfusepy_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    native_transport: None,
    attrs: _FileAttributes,
) -> None:
    fs = MountedTensorBoardFS(tmp_path)
    monkeypatch.setattr(fs, "_attributes", lambda _path, _handles: attrs)
    expected = mfusepy.c_stat()
    mfusepy.set_st_attrs(expected, attrs.as_dict())

    def loop(ops: Any) -> int:
        result = _stat_buffer()
        assert ops.getattr(b"/", pointer(result)) == 0
        assert bytes(result) == bytes(expected)
        result = _stat_buffer()
        assert ops.fgetattr(b"/", pointer(result), None) == 0
        assert bytes(result) == bytes(expected)
        return 0

    _mount_with_loop(monkeypatch, fs, loop)


@pytest.mark.parametrize(
    "mtime", [float("nan"), float("inf"), -float("inf")], ids=["nan", "inf", "-inf"]
)
def test_native_stat_zeros_nonfinite_timestamps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    native_transport: None,
    mtime: float,
) -> None:
    fs = MountedTensorBoardFS(tmp_path)
    attrs = _FileAttributes(False, 513, mtime, (2, 3))
    monkeypatch.setattr(fs, "_attributes", lambda _path, _handles: attrs)

    def loop(ops: Any) -> int:
        result = _stat_buffer()
        assert ops.getattr(b"/", pointer(result)) == -EINVAL
        assert bytes(result) == bytes(sizeof(result))
        result = _stat_buffer()
        assert ops.fgetattr(b"/", pointer(result), None) == -EINVAL
        assert bytes(result) == bytes(sizeof(result))
        return 0

    _mount_with_loop(monkeypatch, fs, loop)


def test_native_stat_keeps_the_open_version_after_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_transport: None
) -> None:
    event = tmp_path / "events.out.tfevents"
    _append(event, 0, 1)
    fs = MountedTensorBoardFS(tmp_path, refresh_age_seconds=0, source_check_seconds=0)
    path = LOSS.encode()

    def loop(ops: Any) -> int:
        original, current, pinned = _stat_buffer(), _stat_buffer(), _stat_buffer()
        info = mfusepy.fuse_file_info()
        info_pointer = pointer(info)
        assert ops.open(path, info_pointer) == 0
        try:
            assert ops.getattr(path, pointer(original)) == 0
            assert S_IFMT(original.st_mode) == S_IFREG
            assert original.st_size == len(fs.read(LOSS, 1 << 20, 0, info))
            _append(event, 2)
            assert ops.fgetattr(path, pointer(current), None) == 0
            assert current.st_size > original.st_size
            assert ops.fgetattr(None, pointer(pinned), info_pointer) == 0
            assert bytes(pinned) == bytes(original)
            assert pinned.st_size == len(fs.read(LOSS, 1 << 20, 0, info))
        finally:
            assert ops.release(path, info_pointer) == 0
        released = _stat_buffer()
        assert ops.fgetattr(path, pointer(released), info_pointer) == 0
        assert bytes(released) == bytes(current)
        return 0

    _mount_with_loop(monkeypatch, fs, loop)


def test_native_stat_decodes_path_bytes_and_zeros_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_transport: None
) -> None:
    event = tmp_path / "events.out.tfevents"
    _append(event, 0, 1)
    fs = MountedTensorBoardFS(tmp_path)
    real_attributes = fs._attributes
    decoded_paths: list[str | None] = []

    def attributes(path: str | None, handles: tuple[object, ...]) -> _FileAttributes:
        decoded_paths.append(path)
        if path == "/raw-\udcff":
            return real_attributes(LOSS, handles)
        if path == "/denied":
            raise PermissionError(EACCES, "denied")
        if path == "/invalid-errno":
            raise OSError(-1, "invalid errno")
        if path == "/failure":
            raise ValueError("resolution failure")
        return real_attributes(path, handles)

    monkeypatch.setattr(fs, "_attributes", attributes)

    def loop(ops: Any) -> int:
        result = _stat_buffer()
        assert ops.getattr(b"/raw-\xff", pointer(result)) == 0
        assert S_IFMT(result.st_mode) == S_IFREG
        assert result.st_size == fs.getattr(LOSS)["st_size"]
        assert decoded_paths[0] == "/raw-\udcff"
        for path, errno in [
            (b"/missing", ENOENT),
            (b"/denied", EACCES),
            (b"/invalid-errno", EINVAL),
            (b"/failure", EINVAL),
        ]:
            result = _stat_buffer()
            assert ops.getattr(path, pointer(result)) == -errno
            assert bytes(result) == bytes(sizeof(result))
        return 0

    _mount_with_loop(monkeypatch, fs, loop, foreground=False)


def test_native_stat_aborts_and_reraises_the_fatal_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    native_transport: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fs = MountedTensorBoardFS(tmp_path)
    fatal = KeyboardInterrupt("interrupt in stat")
    exits: list[bool] = []

    def interrupted(*_args: object) -> _FileAttributes:
        raise fatal

    monkeypatch.setattr(fs, "_attributes", interrupted)
    monkeypatch.setattr(mfusepy, "fuse_exit", lambda: exits.append(True))

    def loop(ops: Any) -> int:
        result = _stat_buffer()
        assert ops.fgetattr(b"/", pointer(result), None) == -EFAULT
        assert bytes(result) == bytes(sizeof(result))
        assert exits == [True]
        return 7

    with pytest.raises(KeyboardInterrupt) as raised:
        _mount_with_loop(monkeypatch, fs, loop)
    assert raised.value is fatal
    assert any(
        record.levelno == logging.CRITICAL
        and record.exc_info is not None
        and record.exc_info[1] is fatal
        for record in caplog.records
    )


def test_native_mount_status_does_not_retry_with_ctypes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_transport: None
) -> None:
    with pytest.raises(RuntimeError) as raised:
        _mount_with_loop(monkeypatch, MountedTensorBoardFS(tmp_path), lambda _: 7)
    assert raised.value.args == (7,)


def test_native_mount_restores_sigint_when_the_loop_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_transport: None
) -> None:
    failure = OSError("native entry failure")

    def interrupt(_signum: int, _frame: object) -> None:
        pytest.fail("the test did not send SIGINT")

    def failed_loop(_ops: Any) -> int:
        assert signal.getsignal(signal.SIGINT) == signal.SIG_DFL
        raise failure

    original = signal.signal(signal.SIGINT, interrupt)
    try:
        with pytest.raises(OSError) as raised:
            _mount_with_loop(monkeypatch, MountedTensorBoardFS(tmp_path), failed_loop)
        assert raised.value is failure
        assert signal.getsignal(signal.SIGINT) is interrupt
    finally:
        signal.signal(signal.SIGINT, original)


@pytest.mark.parametrize(
    "supported", [False, True], ids=["unsupported", "no-extension"]
)
def test_unavailable_bridge_selects_ctypes_before_native_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, supported: bool
) -> None:
    monkeypatch.setattr(_mount_callbacks, "_supported", lambda: supported)
    monkeypatch.setattr(_mount_callbacks, "find_spec", lambda _: None)
    monkeypatch.setattr(
        mfusepy, "fuse_main_real", lambda *_: pytest.fail("unexpected native entry")
    )
    calls: list[tuple[MountedTensorBoardFS, str, dict[str, object]]] = []

    def fallback(
        _binding: mount_binding.MountBinding,
        fs: MountedTensorBoardFS,
        path: str,
        **options: object,
    ) -> None:
        calls.append((fs, path, options))

    monkeypatch.setattr(mount_binding.MountBinding, "__init__", fallback)
    fs = MountedTensorBoardFS(tmp_path)
    options = {"foreground": True, "ro": True, "raw_fi": True, "attr_timeout": 0}
    mount(fs, "/unused", foreground=True, ro=True, raw_fi=True, attr_timeout=0)
    assert calls == [(fs, "/unused", options)]


def test_broken_installed_bridge_is_not_treated_as_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = ImportError("broken native dependency")

    def import_broken(_name: str) -> None:
        raise failure

    monkeypatch.setattr(_mount_callbacks, "_supported", lambda: True)
    monkeypatch.setattr(_mount_callbacks, "find_spec", lambda _name: object())
    monkeypatch.setattr(_mount_callbacks, "import_module", import_broken)
    monkeypatch.setattr(
        mount_binding.MountBinding,
        "__init__",
        lambda *_args, **_kw: pytest.fail("broken installed bridge must not fall back"),
    )

    with pytest.raises(ImportError) as raised:
        mount(
            MountedTensorBoardFS(tmp_path),
            "/unused",
            foreground=True,
            ro=True,
            raw_fi=True,
            attr_timeout=0,
        )
    assert raised.value is failure
