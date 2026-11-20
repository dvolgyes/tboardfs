import subprocess
import sys
import wave
from errno import ENOENT
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import magic
import pytest
from click.testing import CliRunner
from test_parse_file import _record

from tboardfs import TensorBoardFS, find_tensorboard_files
from tboardfs.file_cli import main
from tboardfs.file_tree import SingleEventTree
from tboardfs.proto import compat_pb2

ROOT = Path(__file__).resolve().parent.parent


FIXTURE_EVENT = min(find_tensorboard_files(ROOT / "test-logs" / "tensorboard"))


def test_file_cli_list_prints_representative_virtual_files() -> None:
    """Single-file listing prints virtual file paths only."""
    runner = CliRunner()

    result = runner.invoke(main, ["list", str(FIXTURE_EVENT), "--step-digits", "3"])

    assert result.exit_code == 0
    paths = set(result.output.splitlines())
    assert "/scalars/loss.json" in paths
    assert "/custom_scalars/layout.json" in paths
    assert "/hparams/hparams.json" in paths
    assert "/tensors/activations/001.npy" in paths
    assert "/meshes/box/001.obj" in paths
    assert all(not path.endswith("/") for path in paths)


def test_file_cli_list_prefix_recurses_under_virtual_directory() -> None:
    """Optional PREFIX limits recursive listing."""
    runner = CliRunner()

    result = runner.invoke(
        main, ["list", str(FIXTURE_EVENT), "/meshes/box", "--step-digits", "3"]
    )

    assert result.exit_code == 0
    assert result.output.splitlines() == [
        "/meshes/box/001.json",
        "/meshes/box/001.npz",
        "/meshes/box/001.obj",
        "/meshes/box/001.ply",
        "/meshes/box/002.json",
        "/meshes/box/002.npz",
        "/meshes/box/002.obj",
        "/meshes/box/002.ply",
        "/meshes/box/003.json",
        "/meshes/box/003.npz",
        "/meshes/box/003.obj",
        "/meshes/box/003.ply",
    ]


def test_file_cli_get_writes_shared_reader_bytes(tmp_path: Path) -> None:
    """get writes the same bytes as shared virtual-tree materialization."""
    runner = CliRunner()
    output = tmp_path / "loss.json"
    vpath = "/scalars/loss.json"

    result = runner.invoke(
        main,
        ["get", str(FIXTURE_EVENT), vpath, "-o", str(output), "--step-digits", "3"],
    )

    tree = SingleEventTree(FIXTURE_EVENT, step_digits=3)
    expected = tree.read_file(vpath)
    assert result.exit_code == 0
    assert output.read_bytes() == expected
    assert result.stdout == ""


def test_file_cli_get_accepts_path_without_leading_slash(tmp_path: Path) -> None:
    """get accepts the same virtual path with or without a leading slash."""
    runner = CliRunner()
    output = tmp_path / "loss.json"

    result = runner.invoke(
        main,
        [
            "get",
            str(FIXTURE_EVENT),
            "scalars/loss.json",
            "-o",
            str(output),
            "--step-digits",
            "3",
        ],
    )

    assert result.exit_code == 0
    assert output.read_bytes() == SingleEventTree(
        FIXTURE_EVENT, step_digits=3
    ).read_file("/scalars/loss.json")


def test_file_cli_get_accepts_existing_output_directory(tmp_path: Path) -> None:
    """get writes the virtual basename into an output directory."""
    runner = CliRunner()
    outdir = tmp_path / "out"
    outdir.mkdir()

    result = runner.invoke(
        main,
        [
            "get",
            str(FIXTURE_EVENT),
            "/scalars/loss.json",
            "-o",
            str(outdir),
            "--step-digits",
            "3",
        ],
    )

    assert result.exit_code == 0
    assert outdir.joinpath("loss.json").read_bytes() == SingleEventTree(
        FIXTURE_EVENT, step_digits=3
    ).read_file("/scalars/loss.json")


def test_file_cli_get_treats_trailing_slash_output_as_directory(
    tmp_path: Path,
) -> None:
    """get creates a missing output directory when -o ends with a slash."""
    runner = CliRunner()
    outdir = tmp_path / "missing-dir"

    result = runner.invoke(
        main,
        [
            "get",
            str(FIXTURE_EVENT),
            "scalars/loss.json",
            "-o",
            str(outdir) + "/",
            "--step-digits",
            "3",
        ],
    )

    assert result.exit_code == 0
    assert outdir.joinpath("loss.json").is_file()


def test_file_cli_get_stdout_writes_raw_bytes_only() -> None:
    """get -o - writes raw file data without status text."""
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "get",
            str(FIXTURE_EVENT),
            "/tensors/activations/001.npy",
            "-o",
            "-",
            "--step-digits",
            "3",
        ],
    )

    assert result.exit_code == 0
    assert result.stdout_bytes.startswith(b"\x93NUMPY")
    assert b"INFO" not in result.stdout_bytes


def test_file_cli_get_rejects_missing_directory_and_existing_output(
    tmp_path: Path,
) -> None:
    """get fails for non-file targets and protected outputs."""
    runner = CliRunner()
    output = tmp_path / "existing.json"
    output.write_text("keep")

    missing = runner.invoke(
        main, ["get", str(FIXTURE_EVENT), "/missing.json", "-o", str(tmp_path / "x")]
    )
    directory = runner.invoke(
        main, ["get", str(FIXTURE_EVENT), "/scalars", "-o", str(tmp_path / "x")]
    )
    existing = runner.invoke(
        main,
        ["get", str(FIXTURE_EVENT), "/scalars/loss.json", "-o", str(output)],
    )

    assert missing.exit_code != 0
    assert "virtual path not found" in missing.output
    assert directory.exit_code != 0
    assert "virtual path is a directory" in directory.output
    assert existing.exit_code != 0
    assert "output already exists" in existing.output
    assert "Use --force for forced overwriting." in existing.output
    assert output.read_text() == "keep"


def test_file_cli_copy_all_creates_tree_and_reports_conflicts(
    tmp_path: Path,
) -> None:
    """copy-all preserves structure and reports partial conflict writes."""
    runner = CliRunner()
    outdir = tmp_path / "out"
    virtual_paths = SingleEventTree(FIXTURE_EVENT, step_digits=3).list_file_paths()

    first = runner.invoke(
        main, ["copy-all", str(FIXTURE_EVENT), str(outdir), "--step-digits", "3"]
    )
    expected_count = len(virtual_paths)
    original = outdir.joinpath("scalars", "loss.json").read_bytes()
    conflict_target = _target_for_virtual_path(tmp_path / "conflict", virtual_paths[1])
    conflict_target.parent.mkdir(parents=True)
    conflict_target.write_text("conflict")
    second = runner.invoke(
        main,
        [
            "copy-all",
            str(FIXTURE_EVENT),
            str(tmp_path / "conflict"),
            "--step-digits",
            "3",
        ],
    )
    forced = runner.invoke(
        main,
        [
            "copy-all",
            str(FIXTURE_EVENT),
            str(outdir),
            "-f",
            "--step-digits",
            "3",
        ],
    )

    assert first.exit_code == 0
    assert len([path for path in outdir.rglob("*") if path.is_file()]) == expected_count
    assert f"copied {expected_count} files" in first.stderr
    assert first.stdout == ""
    assert second.exit_code != 0
    assert "Copied successfully: 1 files" in second.stderr
    assert virtual_paths[0] in second.stderr
    assert "output already exists" in second.output
    assert "Use --force for forced overwriting." in second.stderr
    assert outdir.joinpath("scalars", "loss.json").read_bytes() == original
    assert forced.exit_code == 0
    assert "\nv " in outdir.joinpath("meshes", "box", "001.obj").read_text()


def test_file_cli_copy_all_skip_existing_warns_and_continues(
    tmp_path: Path,
) -> None:
    """copy-all --skip-existing warns for existing files and copies the rest."""
    runner = CliRunner()
    outdir = tmp_path / "skip"
    virtual_paths = SingleEventTree(FIXTURE_EVENT, step_digits=3).list_file_paths()
    existing_target = _target_for_virtual_path(outdir, virtual_paths[1])
    existing_target.parent.mkdir(parents=True)
    existing_target.write_text("keep")

    result = runner.invoke(
        main,
        [
            "copy-all",
            str(FIXTURE_EVENT),
            str(outdir),
            "-s",
            "--step-digits",
            "3",
        ],
    )

    assert result.exit_code == 0
    assert result.stdout == ""
    assert "Skipped existing:" in result.stderr
    assert virtual_paths[1] in result.stderr
    assert f"copied {len(virtual_paths) - 1} files" in result.stderr
    assert existing_target.read_text() == "keep"
    assert _target_for_virtual_path(outdir, virtual_paths[0]).is_file()
    assert _target_for_virtual_path(outdir, virtual_paths[2]).is_file()


def test_file_cli_copy_all_rejects_force_with_skip_existing(tmp_path: Path) -> None:
    """copy-all accepts one overwrite policy at a time."""
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "copy-all",
            str(FIXTURE_EVENT),
            str(tmp_path / "out"),
            "-f",
            "-s",
        ],
    )

    assert result.exit_code == 2
    assert "Use either --force or --skip-existing" in result.output


def test_file_cli_copy_all_reports_symlink_escape(tmp_path: Path) -> None:
    """copy-all fails cleanly when an output symlink leaves OUTDIR."""
    outdir = tmp_path / "out"
    outdir.mkdir()
    (outdir / "scalars").symlink_to(tmp_path, target_is_directory=True)

    result = CliRunner().invoke(main, ["copy-all", str(FIXTURE_EVENT), str(outdir)])

    assert result.exit_code == 1
    assert "outside the output directory" in result.output
    assert not isinstance(result.exception, PermissionError)


def test_single_file_tree_matches_filesystem_for_one_event_file(tmp_path: Path) -> None:
    """Single-event mode and FUSE mode expose identical file bytes."""
    source = tmp_path / FIXTURE_EVENT.name
    source.write_bytes(FIXTURE_EVENT.read_bytes())
    fs = TensorBoardFS(tmp_path, step_digits=3)
    tree = SingleEventTree(source, step_digits=3)

    fs_paths = _filesystem_file_paths(fs, "/")
    tree_paths = tree.list_file_paths(view="both")

    assert fs_paths == tree_paths
    for path in fs_paths:
        assert fs.read(path, 100_000_000, 0) == tree.read_file(path)


def test_installed_cli_can_export_the_original_event_file(tmp_path: Path) -> None:
    """A raw export from outside the checkout reconstructs the producer's file."""
    command = Path(sys.executable).with_name("tboardfs-file")
    subprocess.run(
        [command, "copy-all", "--view", "raw", FIXTURE_EVENT, "export"],
        cwd=tmp_path,
        check=True,
    )
    frames = sorted((tmp_path / "export/raw" / FIXTURE_EVENT.name).glob("*.tfrecord"))
    assert b"".join(path.read_bytes() for path in frames) == FIXTURE_EVENT.read_bytes()


def _filesystem_file_paths(fs: TensorBoardFS, path: str) -> list[str]:
    paths = []
    for name in fs.readdir(path):
        if name in {".", ".."}:
            continue
        child = path.rstrip("/") + "/" + name if path != "/" else "/" + name
        try:
            fs.readdir(child)
        except OSError:
            paths.append(child)
        else:
            paths.extend(_filesystem_file_paths(fs, child))
    return sorted(paths)


def _target_for_virtual_path(outdir: Path, virtual_path: str) -> Path:
    return outdir.joinpath(*virtual_path.lstrip("/").split("/"))


def test_file_cli_list_reports_a_missing_prefix() -> None:
    result = CliRunner().invoke(main, ["list", str(FIXTURE_EVENT), "/missing"])

    assert result.exit_code == 1
    assert "virtual path not found: /missing" in result.stderr
    assert isinstance(result.exception, SystemExit)


def test_file_cli_reports_an_output_that_cannot_be_written(tmp_path: Path) -> None:
    """A file where a directory must be is an error, not a traceback."""
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "scalars").write_text("")
    runner = CliRunner()

    got = runner.invoke(
        main, ["get", str(FIXTURE_EVENT), "/scalars/loss.json", "-o", f"{blocker}/x"]
    )
    copied = runner.invoke(
        main, ["copy-all", str(FIXTURE_EVENT), str(tmp_path / "out"), "--force"]
    )

    for result in (got, copied):
        assert result.exit_code == 1
        assert "Error: " in result.stderr
        assert isinstance(result.exception, SystemExit)


def test_file_cli_and_library_work_without_libfuse() -> None:
    """Importing mfusepy needs libfuse, which a machine without FUSE has not."""
    code = (
        "import sys; sys.modules['mfusepy'] = None\n"
        "from tboardfs import TensorBoardFS\n"
        "from tboardfs.file_cli import main\n"
        f"main(['list', {str(FIXTURE_EVENT)!r}])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    assert "/scalars/loss.json\n" in result.stdout


@pytest.mark.parametrize("command", [["list"], ["get", "/x", "-o", "-"]])
@pytest.mark.parametrize("content", [b"", b"hello, not an event file\n"])
def test_file_cli_rejects_a_file_without_events(
    tmp_path: Path, command: list[str], content: bytes
) -> None:
    source = tmp_path / "notes.txt"
    source.write_bytes(content)

    result = CliRunner().invoke(main, [command[0], str(source), *command[1:]])

    assert result.exit_code == 1
    assert str(source) in result.stderr
    assert ("checksum" if content else "no TensorBoard event") in result.stderr
    assert isinstance(result.exception, SystemExit)


def test_single_event_tree_names_a_missing_path() -> None:
    tree = SingleEventTree(FIXTURE_EVENT)

    with pytest.raises(FileNotFoundError) as caught:
        tree.read_file("/scalars/missing.json")

    assert caught.value.errno == ENOENT
    assert caught.value.filename == "/scalars/missing.json"


@pytest.mark.parametrize("command", ["list", "get", "copy-all"])
@pytest.mark.parametrize("damage", ["checksum", "truncated"])
def test_single_file_commands_reject_bad_framing(
    tmp_path: Path, command: str, damage: str
) -> None:
    source = tmp_path / "broken.tfevents"
    data = FIXTURE_EVENT.read_bytes()
    source.write_bytes(
        data[:-1] + bytes([data[-1] ^ 1]) if damage == "checksum" else data + b"x"
    )
    out = tmp_path / "output"
    args = {
        "list": [],
        "get": ["/scalars/loss.json", "-o", str(out)],
        "copy-all": [str(out)],
    }
    result = CliRunner().invoke(main, [command, str(source), *args[command]])
    assert result.exit_code == 1 and isinstance(result.exception, SystemExit)
    assert result.stdout == "" and not out.exists()
    assert str(source) in result.stderr and damage in result.stderr


def test_raw_selection_and_publication_policies(tmp_path: Path) -> None:
    from test_open_handles import _append

    source = tmp_path / "events.out.tfevents"
    _append(source, 1, 2)
    runner = CliRunner()
    raw = runner.invoke(main, ["list", str(source), "--view", "raw"])
    assert raw.exit_code == 0
    paths = raw.stdout.splitlines()
    assert len(paths) == 2
    assert runner.invoke(main, ["list", str(source), "/raw"]).stdout == raw.stdout
    data = [runner.invoke(main, ["get", str(source), p, "-o", "-"]) for p in paths]
    assert all(result.exit_code == 0 for result in data)
    assert b"".join(result.stdout_bytes for result in data) == source.read_bytes()
    excluded = runner.invoke(
        main, ["get", str(source), paths[0], "-o", "-", "--view", "typed"]
    )
    assert excluded.exit_code == 1 and not excluded.stdout_bytes

    out = tmp_path / "out"
    second = out / paths[1][1:]
    second.parent.mkdir(parents=True)
    second.write_bytes(b"existing")
    copy = ["--loglevel", "ERROR", "copy-all", str(source), str(out), "--view", "raw"]
    conflict = runner.invoke(main, copy)
    assert conflict.exit_code == 1 and "Copied successfully: 1 files" in conflict.stderr
    skipped = runner.invoke(main, [*copy, "--skip-existing"])
    assert skipped.exit_code == 0 and "copied 0 files" in skipped.stderr
    assert second.read_bytes() == b"existing"
    forced = runner.invoke(main, [*copy, "--force"])
    assert forced.exit_code == 0 and "copied 2 files" in forced.stderr
    assert not conflict.stdout and not skipped.stdout and not forced.stdout
    assert second.read_bytes() == data[1].stdout_bytes
    second.unlink()
    second.mkdir()
    failed = runner.invoke(main, [*copy, "--force"])
    assert failed.exit_code == 1 and isinstance(failed.exception, SystemExit)
    assert "Copied successfully: 1 files" in failed.stderr and paths[0] in failed.stderr
    assert "not a regular file" in failed.stderr
    assert not list(out.rglob(".tboardfs-*.tmp"))


def test_framing_only_commands_skip_magic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep raw frames independent of custom classifications and typed diagnostics."""
    audio = BytesIO()
    with wave.open(audio, "wb") as wav:
        wav.setparams((1, 1, 8000, 0, "NONE", "not compressed"))
        wav.writeframes(b"TEST" * 32)
    blob = audio.getvalue()
    event = compat_pb2.Event(step=1)
    clip = event.summary.value.add(tag="clip").audio
    clip.content_type = "audio/wav"
    clip.encoded_audio_string = blob
    invalid = event.summary.value.add(tag="broken-layout")
    invalid.metadata.plugin_data.plugin_name = "custom_scalars"
    invalid.tensor.dtype = 7
    invalid.tensor.string_val.append(b"\xff\xff\xff")
    frames = (_record(event.SerializeToString()), _record(b"\xff\xff\xff"))
    source = tmp_path / "events.out.tfevents.audio"
    source.write_bytes(b"".join(frames))
    raw = f"/raw/{source.name}"
    paths = [f"{raw}/{offset:020d}.tfrecord" for offset in (0, len(frames[0]))]

    rules = tmp_path / "magic"
    rules.write_text(
        "44 string TEST custom WAVE test audio\n!:mime audio/flac\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MAGIC", str(rules))
    monkeypatch.setattr(magic, "_instances", {})
    runner = CliRunner()
    with patch.object(magic, "from_buffer", wraps=magic.from_buffer) as detect:
        for args in ([], ["--view", "raw"], ["/raw"], [raw]):
            result = runner.invoke(main, ["--framing-only", "list", str(source), *args])
            assert result.exit_code == 0, result.output
            assert result.stdout.splitlines() == paths
            assert "broken-layout" not in result.stderr
            detect.assert_not_called()

        for path, frame in zip(paths, frames, strict=True):
            result = runner.invoke(
                main, ["--framing-only", "get", str(source), path, "-o", "-"]
            )
            assert result.exit_code == 0, result.output
            assert result.stdout_bytes == frame
            detect.assert_not_called()

        out = tmp_path / "raw-export"
        result = runner.invoke(
            main, ["--framing-only", "copy-all", str(source), str(out)]
        )
        assert result.exit_code == 0, result.output
        assert sorted(p for p in out.rglob("*") if p.is_file()) == [
            _target_for_virtual_path(out, path) for path in paths
        ]
        assert (
            b"".join(_target_for_virtual_path(out, path).read_bytes() for path in paths)
            == source.read_bytes()
        )
        detect.assert_not_called()

        checked = runner.invoke(main, ["list", str(source), "--view", "raw"])
        assert checked.exit_code == 0 and checked.stdout.splitlines() == paths
        assert "skipping malformed record" in checked.stderr
        assert "custom_scalars record broken-layout at step 1" in checked.stderr

        typed = "/audio/clip/000001.flac"
        result = runner.invoke(main, ["list", str(source), "--view", "both"])
        assert result.exit_code == 0, result.output
        assert result.stdout.splitlines() == [typed, *paths]
        detect.assert_called_with(blob, mime=True)
        result = runner.invoke(main, ["get", str(source), typed, "-o", "-"])
        assert result.exit_code == 0, result.output
        assert result.stdout_bytes == blob


@pytest.mark.parametrize("command", ["list", "get-stdout", "get-file", "copy-all"])
@pytest.mark.parametrize("damage", ["checksum", "truncated"])
def test_single_file_raw_commands_validate_later_frame_before_output(
    tmp_path: Path, command: str, damage: str
) -> None:
    """Even an explicit read of the first frame rejects damage farther on."""
    source = tmp_path / "events.out.tfevents.broken"
    first = _record(compat_pb2.Event(file_version="brain.Event:2").SerializeToString())
    second = _record(b"\xff\xff\xff")
    if damage == "checksum":
        second = second[:-1] + bytes([second[-1] ^ 1])
    else:
        second = second[:-1]
    source.write_bytes(first + second)
    raw = f"/raw/{source.name}/{0:020d}.tfrecord"
    out = tmp_path / "output"
    args = {
        "list": ["list", str(source), raw],
        "get-stdout": ["get", str(source), raw, "-o", "-"],
        "get-file": ["get", str(source), raw, "-o", str(out), "--view", "raw"],
        "copy-all": ["copy-all", str(source), str(out), "--view", "raw"],
    }
    result = CliRunner().invoke(main, ["--framing-only", *args[command]])
    assert result.exit_code == 1 and isinstance(result.exception, SystemExit)
    assert not result.stdout_bytes and not out.exists()
    assert str(source) in result.stderr and damage in result.stderr


@pytest.mark.parametrize("view", ["typed", "both"])
def test_framing_only_rejects_typed_views(view: str) -> None:
    result = CliRunner().invoke(
        main, ["--framing-only", "list", str(FIXTURE_EVENT), "--view", view]
    )
    assert result.exit_code != 0 and not result.stdout_bytes
    assert "--view" in result.stderr and "--framing-only" in result.stderr


def test_framing_only_rejects_directory_sources() -> None:
    result = CliRunner().invoke(
        main, ["--framing-only", "list", str(FIXTURE_EVENT.parent)]
    )
    assert result.exit_code != 0 and not result.stdout_bytes
    assert "--framing-only" in result.stderr


def test_failed_get_preserves_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "loss.json"
    target.write_bytes(b"keep")
    script = (
        "import resource, signal\n"
        "from tboardfs.file_cli import main\n"
        "signal.signal(signal.SIGXFSZ, signal.SIG_IGN)\n"
        "resource.setrlimit(resource.RLIMIT_FSIZE, (8, 8))\n"
        "main()\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "get",
            str(FIXTURE_EVENT),
            "/scalars/loss.json",
            "-o",
            str(target),
            "--force",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1 and not result.stdout
    assert "could not write" in result.stderr and "Traceback" not in result.stderr
    assert target.read_bytes() == b"keep"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("directory_link", [False, True])
def test_get_refuses_link_destinations_even_with_force(
    tmp_path: Path, directory_link: bool
) -> None:
    target = tmp_path / "regular"
    target.write_text("keep")
    link = tmp_path / "output"
    link.symlink_to(
        tmp_path if directory_link else target, target_is_directory=directory_link
    )
    result = CliRunner().invoke(
        main,
        ["get", str(FIXTURE_EVENT), "/scalars/loss.json", "-o", str(link), "--force"],
    )
    assert result.exit_code == 1 and isinstance(result.exception, SystemExit)
    assert result.stdout == "" and "not a regular file" in result.stderr
    assert target.read_text() == "keep" and not (tmp_path / "loss.json").exists()
