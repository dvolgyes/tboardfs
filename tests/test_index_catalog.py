"""Exercise persistence and filesystem trust boundaries of the index catalog."""

import fcntl
import os
import sqlite3
import stat
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest

from tboardfs import index_catalog
from tboardfs.index_cache_files import IndexCacheFiles
from tboardfs.index_catalog import CacheCatalog

_DIGEST = b"x" * 16


def _payload(directory: Path) -> Path:
    (payload,) = directory.glob("*.idx")
    return payload


def test_lookup_is_keyed_and_does_not_read_payload(tmp_path: Path) -> None:
    directory = tmp_path / "cache with ? and #"
    source = tmp_path / "source'; --"
    CacheCatalog(directory, 100).put(source, 123, "policy", _DIGEST, b"index")
    catalog = CacheCatalog(directory, 100)
    record = catalog.find(source, 123, "policy")
    assert record is not None
    assert record.digest == _DIGEST
    assert catalog.read(record) == b"index"
    assert catalog.find(source, 124, "policy") is None
    assert catalog.find(source, 123, "other") is None
    assert catalog.find(tmp_path / "other", 123, "policy") is None

    _payload(directory).unlink()
    assert catalog.find(source, 123, "policy") == record
    assert catalog.read(record) is None


def test_checksum_and_include_only_regular_private_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "cache"
    source = tmp_path / "source"
    catalog = CacheCatalog(directory, 100)
    catalog.put(source, 123, "policy", _DIGEST, b"index")
    record = catalog.find(source, 123, "policy")
    assert record is not None
    payload = _payload(directory)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(payload.stat().st_mode) == 0o600

    payload.write_bytes(b"wrong")
    assert catalog.read(record) is None

    def reject_read(_fd: int, _length: int) -> bytes:
        pytest.fail("an unsafe payload must be rejected before reading")

    monkeypatch.setattr(IndexCacheFiles, "read", reject_read)
    payload.write_bytes(b"index" * 21)
    assert catalog.read(record) is None
    payload.unlink()
    external = tmp_path / "external"
    external.write_bytes(b"index")
    external.chmod(0o600)
    assert catalog.read(replace(record, name="../external")) is None
    payload.symlink_to(external)
    assert catalog.read(record) is None
    payload.unlink()
    payload.hardlink_to(external)
    assert catalog.read(record) is None
    payload.unlink()
    os.mkfifo(payload, 0o600)
    assert catalog.read(record) is None
    assert external.read_bytes() == b"index"


def test_expiry_is_from_write_and_quota_evicts_oldest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "cache"
    catalog = CacheCatalog(directory, 8)
    first, second = tmp_path / "first", tmp_path / "second"
    now = 100_000
    monkeypatch.setattr(index_catalog.time, "time", lambda: now)
    catalog.put(first, 123, "policy", _DIGEST, b"first")
    old = catalog.find(first, 123, "policy")
    assert old is not None
    now += 1
    catalog.put(second, 123, "policy", _DIGEST, b"second")
    assert catalog.find(first, 123, "policy") is None
    assert catalog.read(old) is None
    current = catalog.find(second, 123, "policy")
    assert current is not None
    assert catalog.read(current) == b"second"
    assert _payload(directory).stat().st_size == 6
    now += 30 * 24 * 60 * 60 - 1
    assert catalog.find(second, 123, "policy") == current
    now += 1
    assert catalog.find(second, 123, "policy") is None
    assert catalog.read(current) is None
    catalog.put(first, 123, "policy", _DIGEST, b"new")
    assert _payload(directory).read_bytes() == b"new"
    catalog.put(second, 123, "policy", _DIGEST, b"oversized")
    assert catalog.find(second, 123, "policy") is None
    assert _payload(directory).read_bytes() == b"new"


def test_per_payload_ceiling_is_separate_from_quota(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "cache"
    large, first, second = (tmp_path / name for name in ("large", "first", "second"))
    original = CacheCatalog(directory, 100)
    original.put(large, 123, "policy", _DIGEST, b"oversized")
    record = original.find(large, 123, "policy")
    assert record is not None
    monkeypatch.setattr(index_catalog, "MAX_INDEX_BYTES", 8)
    catalog = CacheCatalog(directory, 100)
    assert catalog.find(large, 123, "policy") is None
    assert catalog.read(record) is None
    catalog.put(large, 123, "policy", _DIGEST, b"too large")
    assert original.find(large, 123, "policy") == record
    catalog.put(first, 123, "policy", _DIGEST, b"first---")
    catalog.put(second, 123, "policy", _DIGEST, b"second--")
    assert catalog.find(first, 123, "policy") is not None
    assert catalog.find(second, 123, "policy") is not None


def test_publish_cleans_crashed_payloads_only(tmp_path: Path) -> None:
    directory = tmp_path / "cache"
    catalog = CacheCatalog(directory, 100)
    source = tmp_path / "source"
    catalog.put(source, 123, "policy", _DIGEST, b"first")
    payload = _payload(directory)
    directory.joinpath("catalog.sqlite3").unlink()  # crash before metadata publication
    unrelated = directory / "unrelated.idx"
    unrelated.write_bytes(b"keep")
    unrelated.chmod(0o600)
    dangling = directory / f"index-{'0' * 32}.idx"
    dangling.symlink_to(unrelated)
    pending = directory / f"pending-{'1' * 32}.tmp"
    pending.write_bytes(b"incomplete")
    pending.chmod(0o600)
    catalog.put(source, 123, "policy", _DIGEST, b"second")
    assert not payload.exists()
    assert not pending.exists()
    assert dangling.is_symlink()
    assert unrelated.read_bytes() == b"keep"


def test_damaged_catalog_is_reinitialized_on_save(tmp_path: Path) -> None:
    directory = tmp_path / "cache"
    source = tmp_path / "source"
    catalog = CacheCatalog(directory, 100)
    catalog.put(source, 123, "policy", _DIGEST, b"first")
    old = _payload(directory)
    directory.joinpath("catalog.sqlite3").write_bytes(b"damaged catalog")
    assert catalog.find(source, 123, "policy") is None
    catalog.put(source, 123, "policy", _DIGEST, b"second")
    assert not old.exists()
    record = catalog.find(source, 123, "policy")
    assert record is not None
    assert catalog.read(record) == b"second"


def test_damaged_entries_page_is_reinitialized_on_save(tmp_path: Path) -> None:
    directory = tmp_path / "cache"
    source = tmp_path / "source"
    catalog = CacheCatalog(directory, 100)
    catalog.put(source, 123, "policy", _DIGEST, b"first")
    metadata = directory / "catalog.sqlite3"
    old = _payload(directory)
    with closing(sqlite3.connect(metadata)) as connection:
        (page,) = connection.execute(
            "SELECT rootpage FROM sqlite_schema WHERE name = 'entries'"
        ).fetchone()
        (page_size,) = connection.execute("PRAGMA page_size").fetchone()
    with metadata.open("r+b") as handle:
        handle.seek((page - 1) * page_size)
        handle.write(b"\0")
    assert catalog.find(source, 123, "policy") is None
    catalog.put(source, 123, "policy", _DIGEST, b"second")
    assert not old.exists()
    record = catalog.find(source, 123, "policy")
    assert record is not None
    assert catalog.read(record) == b"second"


def test_foreign_sqlite_schema_is_not_replaced(tmp_path: Path) -> None:
    directory = tmp_path / "cache"
    directory.mkdir(mode=0o700)
    metadata = directory / "catalog.sqlite3"
    with closing(sqlite3.connect(metadata)) as connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")
    metadata.chmod(0o600)
    original = metadata.read_bytes()
    source = tmp_path / "source"
    catalog = CacheCatalog(directory, 100)
    catalog.put(source, 123, "policy", _DIGEST, b"index")
    assert catalog.find(source, 123, "policy") is None
    assert metadata.read_bytes() == original
    assert not list(directory.glob("*.idx"))


def test_unsafe_directory_and_metadata_are_optional_misses(tmp_path: Path) -> None:
    source = tmp_path / "source"
    actual = tmp_path / "actual"
    actual.mkdir(mode=0o700)
    parent = tmp_path / "link"
    parent.symlink_to(actual, target_is_directory=True)
    directory = parent / "cache"
    catalog = CacheCatalog(directory, 100)
    catalog.put(source, 123, "policy", _DIGEST, b"index")
    assert catalog.find(source, 123, "policy") is None
    assert not list(actual.iterdir())

    catalog = CacheCatalog(actual, 100)
    actual.chmod(0o755)
    catalog.put(source, 123, "policy", _DIGEST, b"index")
    assert catalog.find(source, 123, "policy") is None
    assert not list(actual.iterdir())
    actual.chmod(0o700)
    external = tmp_path / "external"
    external.write_bytes(b"untouched")
    external.chmod(0o600)
    actual.joinpath("catalog.sqlite3").symlink_to(external)
    catalog.put(source, 123, "policy", _DIGEST, b"index")
    assert catalog.find(source, 123, "policy") is None
    assert external.read_bytes() == b"untouched"
    assert not list(actual.glob("*.idx"))


def test_sqlite_sidecar_is_not_used_or_removed(tmp_path: Path) -> None:
    directory = tmp_path / "cache"
    catalog = CacheCatalog(directory, 100)
    source = tmp_path / "source"
    catalog.put(source, 123, "policy", _DIGEST, b"first")
    external = tmp_path / "external"
    external.write_bytes(b"untouched")
    sidecar = directory / "catalog.sqlite3-journal"
    sidecar.symlink_to(external)
    catalog.put(source, 123, "policy", _DIGEST, b"second")
    assert sidecar.is_symlink()
    assert external.read_bytes() == b"untouched"
    record = catalog.find(source, 123, "policy")
    assert record is not None
    assert catalog.read(record) == b"second"


def test_independent_writers_keep_all_records(tmp_path: Path) -> None:
    directory = tmp_path / "cache"
    sources = [tmp_path / f"source{i}" for i in range(8)]

    def save(source: Path) -> None:
        catalog = CacheCatalog(directory, 100)
        catalog.put(source, 123, "policy", _DIGEST, source.name.encode())

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(save, sources))

    catalog = CacheCatalog(directory, 100)
    for source in sources:
        record = catalog.find(source, 123, "policy")
        assert record is not None
        assert catalog.read(record) == source.name.encode()


def test_contended_writer_does_not_wait_for_lock_release(tmp_path: Path) -> None:
    directory = tmp_path / "cache"
    directory.mkdir(mode=0o700)
    catalog = CacheCatalog(directory, 100)
    source = tmp_path / "source"
    with ThreadPoolExecutor(max_workers=1) as executor:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            pending = executor.submit(
                catalog.put, source, 123, "policy", _DIGEST, b"index"
            )
            pending.result(timeout=2)
            assert not list(directory.iterdir())
        finally:
            os.close(fd)
    catalog.put(source, 123, "policy", _DIGEST, b"index")
    assert catalog.find(source, 123, "policy") is not None
