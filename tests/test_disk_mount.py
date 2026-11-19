"""Persistent mount indexes preserve the result of a fresh serial parse."""

import os
from pathlib import Path

from click.testing import CliRunner
import pytest

from tboardfs import TensorBoardFS, cli, filesystem, parallel_index
from tboardfs.indexer import _EventIndexer

from test_incremental_parse import _contents, _frame, _loss
from test_payload_spans import _PNG as PNG
from test_tf2_summaries import _images


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    source = tmp_path / "runs"
    source.mkdir()
    storage = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(storage))
    return source, storage


@pytest.mark.parametrize("workers", [1, 2])
def test_disk_mount_reopens_without_parsing(
    roots: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, workers: int
) -> None:
    source, _ = roots
    (source / "events.out.tfevents.1").write_bytes(
        b"".join(_loss(step, step / 2) for step in range(1, 9))
    )
    expected = _contents(TensorBoardFS(source))
    monkeypatch.setattr(filesystem, "_PARALLEL_MIN_BYTES", 1)
    monkeypatch.setattr(parallel_index, "_available_cpus", lambda: 2)
    monkeypatch.chdir(source.parent)
    relative_source = Path(source.name)

    def refuse_parse(*_args: object, **_kwargs: object) -> None:
        pytest.fail("unexpected parse instead of parallel indexing or disk reuse")

    if workers > 1:
        monkeypatch.setattr(_EventIndexer, "parse_event_file", refuse_parse)
    assert (
        _contents(
            TensorBoardFS(relative_source, disk_index=True, index_workers=workers)
        )
        == expected
    )

    monkeypatch.setattr(_EventIndexer, "parse_event_file", refuse_parse)
    monkeypatch.setattr(parallel_index, "parallel_index", refuse_parse)
    assert (
        _contents(
            TensorBoardFS(relative_source, disk_index=True, index_workers=workers)
        )
        == expected
    )


def test_disk_mount_rejects_same_size_rewrite(roots: tuple[Path, Path]) -> None:
    source, _ = roots
    event = source / "events.out.tfevents.1"
    event.write_bytes(_loss(1, 1.0))
    before = _contents(TensorBoardFS(source, disk_index=True))
    status = event.stat()
    assert event.write_bytes(_loss(1, 9.0)) == status.st_size
    os.utime(event, ns=(status.st_atime_ns, status.st_mtime_ns))
    after = _contents(TensorBoardFS(source, disk_index=True))
    assert after != before
    assert after == _contents(TensorBoardFS(source))


def test_disk_mount_uses_current_sorted_inherited_metadata(
    roots: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = roots
    definition = source / "events.out.tfevents.1"
    definition.write_bytes(_frame(_images(1, PNG)))
    dependent = _images(2, PNG)
    dependent.summary.value[0].ClearField("metadata")
    (source / "events.out.tfevents.2").write_bytes(_frame(dependent))
    before = _contents(TensorBoardFS(source, disk_index=True))
    image = "/images/img/000002.png"
    fallback = "/tensors/img/000002_2.png"
    assert before[image] == PNG
    assert fallback not in before

    def refuse_parse(*_args: object, **_kwargs: object) -> None:
        pytest.fail("the dependent index must have been persisted before reordering")

    with monkeypatch.context() as patch:
        patch.setattr(_EventIndexer, "parse_event_file", refuse_parse)
        assert _contents(TensorBoardFS(source, disk_index=True)) == before

    definition.rename(source / "events.out.tfevents.3")
    after = _contents(TensorBoardFS(source, disk_index=True))
    assert image not in after
    assert after[fallback] == PNG
    assert after == _contents(TensorBoardFS(source))


def test_disk_mount_keeps_live_appends_and_refreshes_on_reopen(
    roots: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from tboardfs.disk_index import DiskIndex

    source, _ = roots
    event = source / "events.out.tfevents.1"
    event.write_bytes(_loss(1, 1.0))
    _contents(TensorBoardFS(source, disk_index=True))
    opened = TensorBoardFS(source, disk_index=True)
    before = _contents(opened)
    with event.open("ab") as handle:
        handle.write(_loss(2, 2.0))

    def refuse_disk(*args: object, **kwargs: object) -> None:
        pytest.fail("a known live append must not consult or update disk indexes")

    with monkeypatch.context() as patch:
        patch.setattr(DiskIndex, "candidates", refuse_disk)
        patch.setattr(DiskIndex, "save", refuse_disk)
        live = _contents(opened)
    assert live != before
    assert live == _contents(TensorBoardFS(source, disk_index=True))
    assert live == _contents(TensorBoardFS(source))


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_disk_index_is_opt_in(
    roots: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    source, storage = roots
    (source / "events.out.tfevents.1").write_bytes(_loss(1, 1.0))
    monkeypatch.setattr(cli, "FUSE", lambda fs, *_args, **_kwargs: _contents(fs))
    args = [str(source), str(source.parent / "mnt"), "--disk-index-size-mb", "1"]
    result = CliRunner().invoke(cli.main, args + (["--disk-index"] if enabled else []))
    assert result.exit_code == 0, result.output
    assert storage.exists() is enabled


@pytest.mark.parametrize("location", ["runs", "mnt"])
def test_disk_mount_declines_cache_symlink_into_source_or_mount(
    roots: tuple[Path, Path], location: str
) -> None:
    source, storage = roots
    mountpoint = source.parent / "mnt"
    mountpoint.mkdir()
    target = source.parent / location
    storage.symlink_to(target, target_is_directory=True)
    (source / "events.out.tfevents.1").write_bytes(_loss(1, 1.0))
    assert _contents(TensorBoardFS(source, mountpoint=mountpoint, disk_index=True))
    assert not (target / "tboardfs").exists()


def test_disk_mount_declines_symlink_to_an_existing_private_cache(
    roots: tuple[Path, Path],
) -> None:
    source, storage = roots
    parent = storage / "tboardfs"
    parent.mkdir(parents=True)
    foreign = source.parent / "private"
    foreign.mkdir(mode=0o700)
    (parent / "indexes-v1").symlink_to(foreign, target_is_directory=True)
    (source / "events.out.tfevents.1").write_bytes(_loss(1, 1.0))
    expected = _contents(TensorBoardFS(source))
    assert _contents(TensorBoardFS(source, disk_index=True)) == expected
    assert not list(foreign.iterdir())


def test_disk_mount_declines_storage_redirected_after_construction(
    roots: tuple[Path, Path],
) -> None:
    source, storage = roots
    mountpoint = source.parent / "mnt"
    mountpoint.mkdir()
    (source / "events.out.tfevents.1").write_bytes(_loss(1, 1.0))
    expected = _contents(TensorBoardFS(source))
    fs = TensorBoardFS(source, mountpoint=mountpoint, disk_index=True)
    storage.symlink_to(mountpoint, target_is_directory=True)
    assert _contents(fs) == expected
    assert not (mountpoint / "tboardfs").exists()


def test_disk_mount_does_not_select_an_event_link_into_storage(
    roots: tuple[Path, Path],
) -> None:
    source, storage = roots
    directory = storage / "tboardfs" / "indexes-v1"
    directory.mkdir(parents=True)
    stored = directory / "other"
    stored.write_bytes(_loss(2, 2.0))
    (source / "events.out.tfevents.1").write_bytes(_loss(1, 1.0))
    expected = _contents(TensorBoardFS(source))
    (source / "events.out.tfevents.2").symlink_to(stored)
    assert _contents(TensorBoardFS(source, disk_index=True)) == expected


def test_disk_mount_does_not_persist_an_incomplete_prefix(
    roots: tuple[Path, Path],
) -> None:
    source, storage = roots
    (source / "events.out.tfevents.1").write_bytes(_loss(1, 1.0) + _loss(2, 2.0)[:-1])
    expected = _contents(TensorBoardFS(source))
    assert _contents(TensorBoardFS(source, disk_index=True)) == expected
    assert expected
    assert not storage.exists()


@pytest.mark.parametrize("size", ["0", "-1", "1.5"])
def test_cli_refuses_invalid_disk_index_size(
    roots: tuple[Path, Path], size: str
) -> None:
    source, _ = roots
    result = CliRunner().invoke(
        cli.main,
        [str(source), str(source.parent / "mnt"), "--disk-index-size-mb", size],
    )
    assert result.exit_code == 2
    assert "--disk-index-size-mb" in result.stderr
