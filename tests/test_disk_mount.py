"""Persistent mount indexes preserve the result of a fresh serial parse."""

import json
import os
from errno import ENOENT
from pathlib import Path

import pytest
import xxhash
from click.testing import CliRunner
from test_incremental_parse import _contents, _frame, _loss
from test_payload_spans import _PNG as PNG
from test_tf2_summaries import _images

from tboardfs import TensorBoardFS, cli, filesystem, index_policy, parallel_index
from tboardfs.index_catalog import CacheCatalog
from tboardfs.indexer import _EventIndexer


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
    source, storage = roots
    event = source / "events.out.tfevents.1"
    catalog = CacheCatalog(storage / "tboardfs" / "indexes-v1", 1 << 20)
    policy = index_policy.IndexPolicy.capture()
    assert policy is not None
    raw = _loss(1, 1.0)
    event.write_bytes(raw)
    before = _contents(TensorBoardFS(source, disk_index=True))
    status = event.stat()
    initial = catalog.find(event, status.st_size, policy.id)
    assert initial is not None
    assert initial.digest == xxhash.xxh3_128_digest(raw)
    changed = _loss(1, 9.0)
    assert event.write_bytes(changed) == status.st_size
    os.utime(event, ns=(status.st_atime_ns, status.st_mtime_ns))
    after = _contents(TensorBoardFS(source, disk_index=True))
    assert after != before
    assert after == _contents(TensorBoardFS(source))
    updated = catalog.find(event, status.st_size, policy.id)
    assert updated is not None
    assert updated.digest == xxhash.xxh3_128_digest(changed)
    assert updated.digest != initial.digest


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


def test_periodic_discovery_binds_new_upstream_metadata_sibling(
    roots: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discovery reparses a known downstream event and watches the new sibling."""
    source, _ = roots
    mountpoint = source.parent / "mount"
    mountpoint.mkdir()
    now = 1000.0
    monkeypatch.setattr(filesystem.time, "time", lambda: now)
    dependent = _images(2, PNG)
    dependent.summary.value[0].ClearField("metadata")
    (source / "events.out.tfevents.2").write_bytes(_frame(dependent))
    (source / "events.out.tfevents.3").write_bytes(_loss(3, 3.0))
    fs = TensorBoardFS(source, mountpoint=mountpoint, refresh_age_seconds=60)
    fallback = "/tensors/img/000002_2.png"
    image = "/images/img/000002.png"
    assert fs.read(fallback, 4096, 0) == PNG
    assert fs.read(fallback, 4096, 0) == PNG

    definition = source / "events.out.tfevents.1"
    definition.write_bytes(_frame(_images(1, PNG)))
    now += 59
    assert fs.read(fallback, 4096, 0) == PNG
    now += 1
    assert "img" in fs.readdir("/images")
    assert fs.read(image, 4096, 0) == PNG
    with pytest.raises(FileNotFoundError):
        fs.getattr(fallback)

    definition.unlink()
    assert fs.read(fallback, 4096, 0) == PNG
    with pytest.raises(FileNotFoundError):
        fs.getattr(image)


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


@pytest.mark.parametrize("disk_index", [False, True], ids=["memory", "disk"])
@pytest.mark.parametrize("sibling", [False, True], ids=["single", "siblings"])
@pytest.mark.parametrize("append", [False, True], ids=["replace", "append"])
def test_open_refreshes_scalar_after_getattr(
    roots: tuple[Path, Path], disk_index: bool, sibling: bool, append: bool
) -> None:
    source, _ = roots
    event = source / "events.out.tfevents.2"
    event.write_bytes(_loss(2, 2.0))
    if sibling:
        (source / "events.out.tfevents.1").write_bytes(_loss(1, 1.0))
    virtual = "/scalars/loss.json"
    if disk_index:
        assert TensorBoardFS(source, disk_index=True).getattr(virtual)["st_size"] > 0
    fs = TensorBoardFS(source, disk_index=disk_index, refresh_age_seconds=3600)
    assert fs.getattr(virtual)["st_size"] > 0

    expected = [(1, 1.0)] if sibling else []
    if append:
        with event.open("ab") as stream:
            stream.write(_loss(3, 3.0))
        expected.extend([(2, 2.0), (3, 3.0)])
    else:
        status = event.stat()
        replacement = source / "replacement"
        assert replacement.write_bytes(_loss(2, 9.0)) == status.st_size
        os.utime(replacement, ns=(status.st_atime_ns, status.st_mtime_ns))
        replacement.replace(event)
        expected.append((2, 9.0))

    handle = fs.open(virtual, os.O_RDONLY)
    try:
        rows = json.loads(fs.read(virtual, 4096, 0, handle))
        assert [(row["step"], row["value"]) for row in rows] == expected
    finally:
        fs.release(virtual, handle)


def test_first_binary_read_after_restored_index_grows(
    roots: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = roots
    event = source / "events.out.tfevents.1"
    event.write_bytes(_frame(_images(1, *([PNG] * 80))))
    directory = "/images/img"
    original = TensorBoardFS(source, disk_index=True).readdir(directory)

    def refuse_parse(*_args: object, **_kwargs: object) -> None:
        pytest.fail("the complete source must reopen from its saved index")

    with monkeypatch.context() as hit:
        hit.setattr(_EventIndexer, "parse_event_file", refuse_parse)
        fs = TensorBoardFS(source, disk_index=True)
        assert fs.readdir(directory) == original
    with event.open("ab") as handle:
        handle.write(_frame(_images(2, PNG)))
    assert fs.read(f"{directory}/000002.png", len(PNG), 0) == PNG
    old_image = f"{directory}/000001_79.png"
    assert fs.getattr(old_image)["st_size"] == len(PNG)
    assert fs.read(old_image, 19, 7) == PNG[7:26]


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_disk_index_is_opt_in(
    roots: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    source, storage = roots
    (source / "events.out.tfevents.1").write_bytes(_loss(1, 1.0))

    def read_loss(fs: TensorBoardFS, *_args: object, **_kwargs: object) -> None:
        assert fs.read("/scalars/loss.json", 4096, 0) == expected

    expected = TensorBoardFS(source).read("/scalars/loss.json", 4096, 0)
    monkeypatch.setattr(cli, "MountBinding", read_loss)
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


@pytest.mark.parametrize("mounted", [False, True], ids=["storage-only", "mounted"])
@pytest.mark.parametrize("inherited", [False, True], ids=["single", "reparse"])
def test_open_rejects_known_event_retargeted_into_disk_storage(
    roots: tuple[Path, Path], inherited: bool, mounted: bool
) -> None:
    source, storage = roots
    event = source / "events.out.tfevents.2"
    image = _images(2, PNG)
    definition = source / "events.out.tfevents.1"
    if inherited:
        definition.write_bytes(_frame(_images(1, PNG)))
        image.summary.value[0].ClearField("metadata")
    event.write_bytes(_frame(image))
    fs = TensorBoardFS(
        source,
        mountpoint=source.parent / "mnt" if mounted else None,
        disk_index=True,
        refresh_age_seconds=3600,
    )
    virtual = "/images/img/000002.png"
    assert fs.getattr(virtual)["st_size"] == len(PNG)

    stored = storage / "tboardfs" / "indexes-v1" / event.name
    stored.write_bytes(_frame(_images(2, PNG)) + _loss(99, 99.0))
    if inherited:
        replacement = source / "replacement"
        replacement.write_bytes(_loss(1, 1.0))
        replacement.replace(definition)
    event.unlink()
    event.symlink_to(stored)

    with pytest.raises(OSError) as raised:
        fs.open(virtual, os.O_RDONLY)
    assert raised.value.errno == ENOENT
    if inherited:
        rows = json.loads(fs.read("/scalars/loss.json", 4096, 0))
        assert [(row["step"], row["value"]) for row in rows] == [(1, 1.0)]
    else:
        with pytest.raises(FileNotFoundError):
            fs.getattr("/scalars/loss.json")


@pytest.mark.parametrize("target", ["directory", "eligible-link", "mount", "storage"])
def test_known_siblings_follow_lexical_parent_replacement(
    roots: tuple[Path, Path], target: str
) -> None:
    """Known names are checked under the current parent, not the old directory."""
    source, storage = roots
    mountpoint = source.parent / "mount"
    mountpoint.mkdir()
    names = ("events.out.tfevents.1", "events.out.tfevents.2")
    for step, name in enumerate(names, start=1):
        (source / name).write_bytes(_loss(step, float(step)))
    fs = TensorBoardFS(
        source, mountpoint=mountpoint, disk_index=True, refresh_age_seconds=3600
    )
    virtual = "/scalars/loss.json"

    def steps() -> list[int]:
        return [row["step"] for row in json.loads(fs.read(virtual, 4096, 0))]

    assert steps() == [1, 2]
    assert steps() == [1, 2]
    source.rename(source.with_name("old-run"))
    if target == "mount":
        replacement = mountpoint
    elif target == "storage":
        replacement = storage / "tboardfs" / "indexes-v1" / "run"
        replacement.mkdir(parents=True)
    else:
        replacement = source if target == "directory" else source.with_name("new-run")
        replacement.mkdir()
    for step, name in enumerate(names, start=3):
        (replacement / name).write_bytes(_loss(step, float(step)))
    if replacement != source:
        source.symlink_to(replacement, target_is_directory=True)

    if target in {"mount", "storage"}:
        with pytest.raises(FileNotFoundError):
            fs.read(virtual, 4096, 0)
    else:
        assert steps() == [3, 4]


def test_disk_index_without_mount_drops_invalid_event_link(
    roots: tuple[Path, Path],
) -> None:
    source, _ = roots
    data = source / "data"
    data.write_bytes(_loss(1, 1.0))
    link = source / "events.out.tfevents.1"
    link.symlink_to(data.name)
    fs = TensorBoardFS(source, disk_index=True, refresh_age_seconds=3600)
    virtual = "/scalars/loss.json"
    assert fs.getattr(virtual)["st_size"] > 0
    link.unlink()
    os.symlink(f"{data.name}/", link)
    with pytest.raises(FileNotFoundError):
        fs.open(virtual, os.O_RDONLY)


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
