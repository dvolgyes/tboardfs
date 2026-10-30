"""Each name fits the 255-byte limit of the kernel, and different names stay different."""

from pathlib import Path

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2

from test_incremental_parse import _frame

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 32
LONG = "a" * 300


def _log(run_dir: Path, scalar_tags: list[str], image_tag: str) -> None:
    event = compat_pb2.Event(step=1, wall_time=1.0)
    for tag in scalar_tags:
        event.summary.value.add(tag=tag, simple_value=1.0)
    event.summary.value.add(tag=image_tag).image.encoded_image_string = PNG
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "events.out.tfevents.host").write_bytes(_frame(event))


def _names(fs: TensorBoardFS, path: str) -> list[str]:
    names = []
    for name in fs.readdir(path)[2:]:
        names.append(name)
        child = f"{path.rstrip('/')}/{name}"
        if fs.getattr(child)["st_nlink"] == 2:
            names.extend(_names(fs, child))
    return names


def test_long_tags_give_short_names_that_differ(tmp_path: Path) -> None:
    _log(tmp_path, [LONG + "1", LONG + "2", "é" * 200], LONG)
    fs = TensorBoardFS(tmp_path, scalar_formats="json")

    assert all(len(name.encode()) <= 255 for name in _names(fs, "/"))
    scalars = fs.readdir("/scalars")[2:]
    assert len(scalars) == 3
    assert all(name.endswith(".json") for name in scalars)
    for name in scalars:
        assert fs.read(f"/scalars/{name}", 1 << 20, 0).startswith(b"[")
    (image_dir,) = fs.readdir("/images")[2:]
    assert image_dir.startswith("a" * 200)
    assert fs.read(f"/images/{image_dir}/000001.png", 1 << 20, 0) == PNG
