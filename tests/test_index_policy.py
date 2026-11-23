"""The durable index accepts only an identified classifier generation."""

from importlib import import_module
from pathlib import Path

import pytest

from tboardfs import index_policy
from tboardfs.classify import detect_extension
from tboardfs.index_policy import IndexPolicy


def test_default_policy_is_stable_after_first_classifier_use() -> None:
    policy = IndexPolicy.capture()
    assert policy is not None
    assert len(bytes.fromhex(policy.id)) == 16
    assert detect_extension(b'{"value": 3}') == "json"
    assert policy.unchanged()
    repeated = IndexPolicy.capture()
    assert repeated is not None and repeated.id == policy.id


def test_live_policy_check_does_not_probe_classifier_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = IndexPolicy.capture()
    assert policy is not None

    def forbid_probe(*_args: object) -> None:
        pytest.fail("live classifier compatibility must not probe host paths")

    monkeypatch.setattr(IndexPolicy, "_crosses_mount", forbid_probe)
    assert policy.unchanged()


def test_decoder_revision_namespaces_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    original = IndexPolicy.capture()
    assert original is not None
    monkeypatch.setattr(index_policy, "_SCHEMA", index_policy._SCHEMA + 1)
    revised = IndexPolicy.capture()
    assert revised is not None and revised.id != original.id


def test_loaded_signature_change_requires_a_new_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("puremagic.main")
    policy = IndexPolicy.capture()
    assert policy is not None
    signature = module.PureMagic(
        b"TBOARD-TEST", 0, ".tboard", "application/x-tboard", "tboardfs test"
    )
    monkeypatch.setattr(
        module, "magic_header_array", [*module.magic_header_array, signature]
    )
    revised = IndexPolicy.capture()
    assert revised is not None and revised.id != policy.id
    assert detect_extension(b"TBOARD-TEST") == "tboard"


def test_policy_declines_classifier_data_through_mountpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mountpoint = tmp_path / "mount"
    mountpoint.mkdir()
    data = mountpoint / "magic_data.json"
    data.write_bytes(b"[]")
    monkeypatch.setattr(index_policy, "_magic_data_path", lambda: data)
    assert IndexPolicy.capture() is not None
    assert IndexPolicy.capture(mountpoint=mountpoint) is None


def test_policy_declines_classifier_data_symlink_chain_through_mountpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mountpoint = tmp_path / "mount"
    nested = mountpoint / "nested"
    nested.mkdir(parents=True)
    data = tmp_path / "magic_data.json"
    data.write_bytes(b"[]")
    (nested / "out").symlink_to("../../magic_data.json")
    entry = tmp_path / "outside"
    entry.symlink_to("mount/nested/out")
    assert entry.resolve() == data
    monkeypatch.setattr(index_policy, "_magic_data_path", lambda: entry)
    assert IndexPolicy.capture() is not None
    assert IndexPolicy.capture(mountpoint=mountpoint) is None


def test_relative_classifier_data_path_is_unobservable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = Path("relative/magic_data.json")
    monkeypatch.setattr(index_policy, "_magic_data_path", lambda: path)
    assert IndexPolicy.capture() is None
