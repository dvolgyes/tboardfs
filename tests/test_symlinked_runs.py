"""Symlinked run directories are runs; link loops are walked once."""

from pathlib import Path
import threading

from tboardfs import TensorBoardFS, find_tensorboard_files

from test_incremental_parse import _files
from test_open_handles import _append

EVENTS = "events.out.tfevents.host"


def _run_dir(path: Path) -> Path:
    path.mkdir(parents=True)
    _append(path / EVENTS, 0)
    return path


def test_a_linked_run_shows_under_the_link_name(tmp_path: Path) -> None:
    """A run kept elsewhere and linked into the logdir is listed and readable."""
    run = _run_dir(tmp_path / "elsewhere" / "run")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "link").symlink_to(run)

    fs = TensorBoardFS(tmp_path / "logs")

    assert fs.readdir("/")[2:] == ["link"]
    assert "/link/scalars/loss.json" in _files(fs)


def test_sidecars_of_a_linked_run_are_listed(tmp_path: Path) -> None:
    """Files of the linked run may resolve outside the logdir."""
    run = _run_dir(tmp_path / "elsewhere" / "run")
    (run / "projector_config.pbtxt").write_text("config")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "link").symlink_to(run)

    fs = TensorBoardFS(tmp_path / "logs")

    assert fs.read("/link/projector/projector_config.pbtxt", 100, 0) == b"config"


def test_sidecar_links_out_of_a_linked_run_stay_hidden(tmp_path: Path) -> None:
    """Following a run link does not open other paths of its target."""
    run = _run_dir(tmp_path / "elsewhere" / "run")
    (tmp_path / "secret").write_text("secret")
    (run / "projector").mkdir()
    (run / "projector" / "leak.tsv").symlink_to(tmp_path / "secret")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "link").symlink_to(run)

    fs = TensorBoardFS(tmp_path / "logs")

    assert "/link/projector/leak.tsv" not in _files(fs)


def test_a_link_loop_ends(tmp_path: Path) -> None:
    """A link back to an ancestor is walked once."""
    _run_dir(tmp_path / "logs" / "run")
    (tmp_path / "logs" / "run" / "up").symlink_to(tmp_path / "logs")

    assert find_tensorboard_files(tmp_path / "logs") == [
        tmp_path / "logs" / "run" / EVENTS
    ]


def test_two_links_to_one_run_give_one_run(tmp_path: Path) -> None:
    """A directory reached by several paths is read once, by the first in order."""
    run = _run_dir(tmp_path / "elsewhere" / "run")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "b").symlink_to(run)
    (tmp_path / "logs" / "a").symlink_to(run)

    assert find_tensorboard_files(tmp_path / "logs") == [
        tmp_path / "logs" / "a" / EVENTS
    ]


def test_two_link_loops_end_quickly(tmp_path: Path) -> None:
    """A directory reached again is not walked again, so loops do not multiply."""
    _run_dir(tmp_path / "logs" / "run")
    (tmp_path / "logs" / "run" / "up").symlink_to(tmp_path / "logs")
    (tmp_path / "logs" / "run" / "again").symlink_to(tmp_path / "logs")
    found: list[list[Path]] = []
    walker = threading.Thread(
        target=lambda: found.append(find_tensorboard_files(tmp_path / "logs")),
        daemon=True,
    )

    walker.start()
    walker.join(timeout=10)

    assert found == [[tmp_path / "logs" / "run" / EVENTS]]


def test_a_dangling_event_file_link_is_skipped(tmp_path: Path) -> None:
    """Only event files that can be read are returned."""
    _run_dir(tmp_path / "logs" / "run")
    (tmp_path / "logs" / "run" / f"{EVENTS}.gone").symlink_to(tmp_path / "missing")

    assert find_tensorboard_files(tmp_path / "logs") == [
        tmp_path / "logs" / "run" / EVENTS
    ]
