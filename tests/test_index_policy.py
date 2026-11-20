"""The durable index accepts only an identified classifier generation."""

import os
from pathlib import Path

import magic
import pytest

from tboardfs import index_magic, index_policy
from tboardfs.index_magic import MagicHost
from tboardfs.index_policy import IndexPolicy


def test_default_policy_is_stable_after_first_classifier_use() -> None:
    policy = IndexPolicy.capture()
    assert policy is not None
    assert len(bytes.fromhex(policy.id)) == 16
    for mime in (True, False):
        assert MagicHost.classify(b"plain text\n", mime=mime)
    assert policy.unchanged()
    repeated = IndexPolicy.capture()
    assert repeated is not None and repeated.id == policy.id


def test_live_policy_check_does_not_probe_rule_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = IndexPolicy.capture()
    assert policy is not None

    def forbid_probe(*_args: object) -> None:
        pytest.fail("live classifier compatibility must not probe host paths")

    monkeypatch.setattr(index_magic._RuleFiles, "status", forbid_probe)
    assert policy.unchanged()


def test_decoder_revision_namespaces_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    original = IndexPolicy.capture()
    assert original is not None
    monkeypatch.setattr(index_policy, "_SCHEMA", index_policy._SCHEMA + 1)
    revised = IndexPolicy.capture()
    assert revised is not None and revised.id != original.id


def test_rule_change_requires_fresh_process_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rules = tmp_path / "magic"
    rules.write_bytes(b"0 string TBOARD-TEST one\n")
    monkeypatch.setenv("MAGIC", str(rules))
    monkeypatch.setattr(magic, "_instances", {})
    policy = IndexPolicy.capture()
    assert policy is not None
    assert MagicHost.classify(b"TBOARD-TEST") == "one"
    assert policy.unchanged()
    status = rules.stat()
    rules.write_bytes(b"0 string TBOARD-TEST two\n")
    os.utime(rules, ns=(status.st_atime_ns, status.st_mtime_ns))
    assert IndexPolicy.capture() is None
    assert MagicHost.classify(b"TBOARD-TEST") == "one"

    monkeypatch.setattr(magic, "_instances", {})
    revised = IndexPolicy.capture()
    assert revised is not None and revised.id != policy.id
    assert MagicHost.classify(b"TBOARD-TEST") == "two"


def test_transient_magic_override_cannot_bless_a_stale_cookie(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rules = tmp_path / "magic"
    rules.write_bytes(b"0 string TBOARD-TEST alternate\n")
    monkeypatch.setattr(magic, "_instances", {})
    policy = IndexPolicy.capture()
    assert policy is not None
    with monkeypatch.context() as changed:
        changed.setenv("MAGIC", str(rules))
        assert not policy.unchanged()
        assert MagicHost.classify(b"TBOARD-TEST") == "alternate"
    assert not policy.unchanged()
    assert IndexPolicy.capture() is None


def test_policy_declines_rule_path_through_mountpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mountpoint = tmp_path / "mount"
    mountpoint.mkdir()
    rules = mountpoint / "magic"
    rules.write_bytes(b"0 string TBOARD-TEST custom\n")
    link = tmp_path / "link"
    link.symlink_to(rules)
    monkeypatch.setenv("MAGIC", str(link))
    monkeypatch.setattr(magic, "_instances", {})
    assert IndexPolicy.capture() is not None
    assert IndexPolicy.capture(mountpoint=mountpoint) is None


def test_policy_declines_rule_symlink_chain_that_leaves_mountpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mountpoint = tmp_path / "mount"
    nested = mountpoint / "nested"
    nested.mkdir(parents=True)
    rules = tmp_path / "actual.magic"
    rules.write_bytes(b"0 string TBOARD-TEST custom\n")
    (nested / "out").symlink_to("../../actual.magic")
    entry = tmp_path / "outside"
    entry.symlink_to("mount/nested/out")
    assert entry.resolve() == rules
    monkeypatch.setenv("MAGIC", str(entry))
    monkeypatch.setattr(magic, "_instances", {})
    assert IndexPolicy.capture() is not None
    assert IndexPolicy.capture(mountpoint=mountpoint) is None


def test_relative_rule_path_is_unobservable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAGIC", "relative/custom.magic")
    monkeypatch.setattr(magic, "_instances", {})
    assert IndexPolicy.capture() is None
