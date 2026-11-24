"""Epoch marker lookup and canonical alias projection."""

import math
from pathlib import Path

import numpy as np
import pytest

from tboardfs.epoch_breakdown import _EpochBreakdown
from tboardfs.epochs import _Epochs
from tboardfs.leaf_spec import LeafSpec
from tboardfs.model import EventFileCache
from tboardfs.paths import _DirectoryNode
from tboardfs.run_listing import RunListing


def _markers(markers: dict[str, dict[int, float | None]]) -> _Epochs:
    cache = EventFileCache(Path("event"))
    for tag, rows in markers.items():
        cache.scalars[tag] = {
            "step": np.asarray(list(rows), dtype=np.int64),
            "value": np.asarray(list(rows.values()), dtype=np.float64),
        }
    return _Epochs.from_caches((cache,))


def test_marker_selection_prefers_exact_scoped_then_flat_fallback() -> None:
    epochs = _markers({"epoch": {0: 1, 20: 3}, "train/epoch": {10: 2}})

    assert epochs.at("train/loss", 0) == 1
    assert epochs.at("train/loss", 10) == 2
    assert epochs.at("train/loss", 19) == 2
    assert epochs.at("train/loss", 20) == 2
    assert epochs.at("validation/loss", 20) == 3
    assert epochs.at("validation/loss", 21) == 3


def test_invalid_scoped_marker_stops_flat_inheritance() -> None:
    epochs = _markers({"epoch": {0: 1, 5: 2}, "train/epoch": {3: None}})

    assert epochs.at("train/loss", 2) == 1
    assert epochs.at("train/loss", 3) is None
    assert epochs.at("train/loss", 8) is None
    assert epochs.at("validation/loss", 8) == 2


def test_projection_preserves_canonical_nested_path_and_data() -> None:
    leaf = LeafSpec("bytes", 1.0, "data", b"payload")
    run = _DirectoryNode()
    run["children"]["images"] = _DirectoryNode()
    run["children"]["images"]["children"]["train"] = _DirectoryNode()
    train = run["children"]["images"]["children"]["train"]
    train["children"] = RunListing({"000010.png": leaf})

    breakdown = _EpochBreakdown(_markers({"epoch": {0: 1}}))
    breakdown.track(leaf, "train/image", 10)
    breakdown.project(run)

    epoch = run["children"]["epoch-breakdown"]["children"]["epoch-1"]
    alias = epoch["children"]["images"]["children"]["train"]["children"]["000010.png"]
    assert alias["data"] == b"payload"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (3.0, 3),
        (-1, None),
        (1.5, None),
        (math.nan, None),
        (math.inf, None),
        (float(2**63), None),
    ],
)
def test_epoch_marker_must_be_a_finite_nonnegative_integer(
    value: float, expected: int | None
) -> None:
    assert _markers({"epoch": {0: value}}).at("train/loss", 0) == expected


def test_projection_ignores_non_integer_step_values() -> None:
    leaf = LeafSpec("bytes", 1.0, "data", b"payload")
    run = _DirectoryNode()
    run["children"]["images"] = _DirectoryNode()
    run["children"]["images"]["children"]["000010.png"] = leaf

    breakdown = _EpochBreakdown(_markers({"epoch": {0: 1}}))
    breakdown.track(leaf, "image", 10.0)
    breakdown.project(run)

    assert "epoch-breakdown" not in run["children"]


def test_cached_markers_use_int64_steps_and_last_producer_duplicate() -> None:
    first = EventFileCache(Path("first"))
    first.scalars["epoch"] = {
        "step": np.asarray([3, 3], dtype=np.int64),
        "value": np.asarray([1.0, 2.0]),
    }
    second = EventFileCache(Path("second"))
    second.scalars["epoch"] = {
        "step": np.asarray([3, 4], dtype=np.int64),
        "value": np.asarray([5.0, math.nan]),
    }

    epochs = _Epochs.from_caches([first, second])

    assert epochs.at("train/loss", 3) == 5
    assert epochs.at("train/loss", 4) is None


def test_cached_float_steps_do_not_become_integer_markers() -> None:
    cache = EventFileCache(Path("event"))
    cache.scalars["epoch"] = {
        "step": np.asarray([3.0], dtype=np.float64),
        "value": np.asarray([1.0]),
    }

    epochs = _Epochs.from_caches([cache])

    assert epochs.at("train/loss", 3) is None


def test_raw_traversal_tag_selects_marker_without_creating_path_components() -> None:
    leaf = LeafSpec("json", 1.0, "data", {"ok": True})
    run = _DirectoryNode()
    run["children"]["plugins"] = _DirectoryNode()
    run["children"]["plugins"]["children"]["%2E%2E"] = _DirectoryNode()
    parent = run["children"]["plugins"]["children"]["%2E%2E"]
    parent["children"] = RunListing({"000010.json": leaf})

    breakdown = _EpochBreakdown(_markers({"../epoch": {0: 2}}))
    breakdown.track(leaf, "../escape", 10)
    breakdown.project(run)

    epoch = run["children"]["epoch-breakdown"]["children"]["epoch-2"]
    virtual = epoch["children"]["plugins"]
    assert set(virtual["children"]) == {"%2E%2E"}
    assert virtual["children"]["%2E%2E"]["children"]["000010.json"]["data"] == {
        "ok": True
    }
