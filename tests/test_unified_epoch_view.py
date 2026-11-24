"""Exercise canonical scalar epochs and epoch-first files through the mounted view."""

import csv
import json
from pathlib import Path

import pytest
from test_incremental_parse import PNG, _frame

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2

LOSS = "/scalars/train/loss"
EPOCHS = "/epoch-breakdown"
IMAGE = "/images/train/prediction/000003.png"
COLUMNS = ("step", "epoch", "value", "wall_time", "relative_time")


def _scalar(step: int, tag: str, value: float, wall_time: float) -> bytes:
    event = compat_pb2.Event(step=step, wall_time=wall_time)
    event.summary.value.add(tag=tag, simple_value=value)
    return _frame(event)


def _image(step: int, tag: str = "train/prediction") -> bytes:
    event = compat_pb2.Event(step=step, wall_time=110)
    image = event.summary.value.add(tag=tag).image
    image.height = image.width = 1
    image.encoded_image_string = PNG
    return _frame(event)


@pytest.fixture
def scoped_log(tmp_path: Path) -> Path:
    (tmp_path / "events.out.tfevents.0").write_bytes(
        _scalar(0, "train/loss", 10, 100)
        + _scalar(1, "epoch", 2, 101)
        + _scalar(2, "train/loss", 20, 102)
        + _scalar(4, "train/loss", 40, 103)
        + _scalar(3, "train/loss", 30, 104)
        + _scalar(3, "train/epoch", 4, 105)
        + _scalar(3, "val/loss", 50, 106)
        + _scalar(3, "val/epoch", 8, 107)
        + _scalar(3, "train/loss", 31, 108)
        + _scalar(3, "train/epoch", 5, 109)
        + _image(3)
        + _scalar(4, "epoch", 2, 111)
        + _scalar(5, "train/epoch", float("nan"), 112)
        + _scalar(6, "train/loss", 60, 113)
        + _image(6)
    )
    return tmp_path


def _json_rows(fs: TensorBoardFS, path: str) -> list[tuple[float | None, ...]]:
    rows = json.loads(fs.read(path + ".json", 65536, 0))
    return [tuple(row[column] for column in COLUMNS) for row in rows]


def _text_rows(
    fs: TensorBoardFS, path: str, extension: str = "tsv"
) -> list[tuple[float | None, ...]]:
    lines = fs.read(f"{path}.{extension}", 65536, 0).decode().splitlines()
    delimiter = "\t" if extension == "tsv" else ","
    return [
        tuple(
            None if row[column] in ("", "nan") else float(row[column])
            for column in COLUMNS
        )
        for row in csv.DictReader(lines, delimiter=delimiter)
    ]


def test_default_mount_exports_scoped_epochs_in_canonical_tables(
    scoped_log: Path,
) -> None:
    fs = TensorBoardFS(scoped_log)
    expected = [
        (0, None, 10, 100, 0),
        (2, 2, 20, 102, 2),
        (3, 5, 30, 104, 4),
        (3, 5, 31, 108, 8),
        (4, 5, 40, 103, 3),
        (6, None, 60, 113, 13),
    ]
    assert _json_rows(fs, LOSS) == expected
    assert _text_rows(fs, LOSS) == expected
    assert _json_rows(fs, "/scalars/val/loss") == [(3, 8, 50, 106, 6)]
    assert "epoch-breakdown" not in fs.readdir("/")
    assert fs.read(IMAGE, len(PNG), 0) == PNG


def test_epoch_first_mount_filters_tables_and_media(scoped_log: Path) -> None:
    fs = TensorBoardFS(scoped_log, epoch_breakdown=True)
    assert set(fs.readdir(EPOCHS)) == {".", "..", "epoch-2", "epoch-5", "epoch-8"}
    assert _json_rows(fs, EPOCHS + "/epoch-2" + LOSS) == [(2, 2, 20, 102, 2)]
    train = EPOCHS + "/epoch-5" + LOSS
    expected = [(3, 5, 30, 104, 4), (3, 5, 31, 108, 8), (4, 5, 40, 103, 3)]
    assert _json_rows(fs, train) == expected
    assert _text_rows(fs, train) == expected
    assert _json_rows(fs, EPOCHS + "/epoch-8/scalars/val/loss") == [(3, 8, 50, 106, 6)]
    assert "train" not in fs.readdir(EPOCHS + "/epoch-8/scalars")
    assert "images" not in fs.readdir(EPOCHS + "/epoch-2")
    assert "images" not in fs.readdir(EPOCHS + "/epoch-8")
    assert fs.readdir(EPOCHS + "/epoch-5/images/train/prediction") == [
        ".",
        "..",
        "000003.png",
    ]
    assert fs.read(EPOCHS + "/epoch-5" + IMAGE, len(PNG), 0) == PNG
    assert fs.read("/images/train/prediction/000006.png", len(PNG), 0) == PNG


def test_epoch_first_tables_respect_scalar_formats(scoped_log: Path) -> None:
    fs = TensorBoardFS(scoped_log, epoch_breakdown=True, scalar_formats="csv")
    train = EPOCHS + "/epoch-5" + LOSS
    for folder in ("/scalars/train", EPOCHS + "/epoch-5/scalars/train"):
        names = fs.readdir(folder)
        assert {name for name in names if name.startswith("loss.")} == {"loss.csv"}
    assert _text_rows(fs, train, "csv") == [
        (3, 5, 30, 104, 4),
        (3, 5, 31, 108, 8),
        (4, 5, 40, 103, 3),
    ]


@pytest.mark.parametrize("epoch_breakdown", [False, True])
def test_appended_marker_moves_existing_rows_and_media(
    tmp_path: Path, epoch_breakdown: bool
) -> None:
    source = tmp_path / "events.out.tfevents.0"
    source.write_bytes(
        _scalar(3, "train/loss", 30, 100)
        + _scalar(3, "train/epoch", 5, 101)
        + _scalar(4, "train/loss", 40, 102)
        + _image(3)
    )
    fs = TensorBoardFS(
        tmp_path,
        epoch_breakdown=epoch_breakdown,
        refresh_age_seconds=0,
        source_check_seconds=0,
    )
    before = [(3, 5, 30, 100, 0), (4, 5, 40, 102, 2)]
    after = [(3, 8, 30, 100, 0), (4, 8, 40, 102, 2)]
    old = EPOCHS + "/epoch-5"
    new = EPOCHS + "/epoch-8"
    assert _json_rows(fs, LOSS) == before
    assert _text_rows(fs, LOSS) == before
    if epoch_breakdown:
        assert _json_rows(fs, old + LOSS) == before
        assert fs.read(old + IMAGE, len(PNG), 0) == PNG

    with source.open("ab") as stream:
        stream.write(_scalar(3, "train/epoch", 8, 111))

    assert _json_rows(fs, LOSS) == after
    assert _text_rows(fs, LOSS) == after
    assert fs.read(IMAGE, len(PNG), 0) == PNG
    if epoch_breakdown:
        assert fs.readdir(EPOCHS) == [".", "..", "epoch-8"]
        assert _json_rows(fs, new + LOSS) == after
        assert fs.read(new + IMAGE, len(PNG), 0) == PNG
    else:
        assert "epoch-breakdown" not in fs.readdir("/")


def test_returning_epoch_keeps_disjoint_rows_and_exact_steps(tmp_path: Path) -> None:
    step = 2**53
    source = tmp_path / "events.out.tfevents.0"
    source.write_bytes(
        _scalar(step, "train/epoch", 2, 100)
        + _scalar(step, "train/loss", 10, 101)
        + _scalar(step + 1, "train/epoch", 3, 102)
        + _scalar(step + 1, "train/loss", 20, 103)
        + _scalar(step + 2, "train/epoch", 2, 104)
        + _scalar(step + 2, "train/loss", 30, 105)
    )
    fs = TensorBoardFS(tmp_path, epoch_breakdown=True)
    assert _json_rows(fs, EPOCHS + "/epoch-2" + LOSS) == [
        (step, 2, 10, 101, 1),
        (step + 2, 2, 30, 105, 5),
    ]
    assert _json_rows(fs, EPOCHS + "/epoch-3" + LOSS) == [(step + 1, 3, 20, 103, 3)]


def test_epoch_alias_uses_the_escaped_canonical_tag(tmp_path: Path) -> None:
    (tmp_path / "events.out.tfevents.0").write_bytes(
        _scalar(0, "../epoch", 2, 100) + _image(10, "../escape")
    )
    fs = TensorBoardFS(tmp_path, epoch_breakdown=True)
    prefix = EPOCHS + "/epoch-2/images"
    assert fs.readdir(prefix) == [".", "..", "%2E%2E"]
    assert fs.read(prefix + "/%2E%2E/escape/000010.png", len(PNG), 0) == PNG
