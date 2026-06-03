import tomllib
from pathlib import Path

from click.testing import CliRunner

from tboardfs import TensorBoardFS, find_tensorboard_files
from tboardfs.file_cli import main
from tboardfs.file_tree import SingleEventTree


FIXTURE_EVENT = sorted(find_tensorboard_files(Path("test-logs") / "tensorboard"))[0]


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
        "/meshes/box/002.json",
        "/meshes/box/002.npz",
        "/meshes/box/002.obj",
        "/meshes/box/003.json",
        "/meshes/box/003.npz",
        "/meshes/box/003.obj",
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
            "--force",
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
            "--skip-existing",
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
            "--force",
            "--skip-existing",
        ],
    )

    assert result.exit_code != 0
    assert "Use either --force or --skip-existing" in result.output


def test_single_file_tree_matches_filesystem_for_one_event_file(tmp_path: Path) -> None:
    """Single-event mode and FUSE mode expose identical file bytes."""
    source = tmp_path / FIXTURE_EVENT.name
    source.write_bytes(FIXTURE_EVENT.read_bytes())
    fs = TensorBoardFS(tmp_path, step_digits=3)
    tree = SingleEventTree(source, step_digits=3)

    fs_paths = _filesystem_file_paths(fs, "/")
    tree_paths = tree.list_file_paths()

    assert fs_paths == tree_paths
    for path in fs_paths:
        assert fs.read(path, 100_000_000, 0) == tree.read_file(path)


def test_metadata_and_readme_document_both_access_modes() -> None:
    """Packaging metadata and README mention both commands."""
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    readme = Path("README.md").read_text()

    assert project["version"] == "0.2.0"
    assert project["license"] == "MIT"
    assert project["scripts"]["tboardfs"] == "tboardfs:main"
    assert project["scripts"]["tboardfs-file"] == "tboardfs.file_cli:main"
    assert "tboardfs SOURCE MOUNTPOINT" in readme
    assert "tboardfs-file list" in readme
    assert "tboardfs-file get" in readme
    assert "tboardfs-file copy-all" in readme
    assert "--skip-existing" in readme
    assert "/pr_curves" in readme
    assert "/projector" in readme
    assert "/profile" in readme


def _filesystem_file_paths(fs: TensorBoardFS, path: str) -> list[str]:
    paths = []
    for name in fs.readdir(path):
        if name in {".", "..", ".cache", ".in_memory"}:
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
