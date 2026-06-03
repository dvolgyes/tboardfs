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
    assert output.read_text() == "keep"


def test_file_cli_copy_all_creates_tree_and_preflights_conflicts(
    tmp_path: Path,
) -> None:
    """copy-all preserves structure and avoids partial conflict writes."""
    runner = CliRunner()
    outdir = tmp_path / "out"

    first = runner.invoke(
        main, ["copy-all", str(FIXTURE_EVENT), str(outdir), "--step-digits", "3"]
    )
    expected_count = len(
        SingleEventTree(FIXTURE_EVENT, step_digits=3).list_file_paths()
    )
    original = outdir.joinpath("scalars", "loss.json").read_bytes()
    outdir.joinpath("meshes", "box", "001.obj").write_text("conflict")
    second = runner.invoke(
        main, ["copy-all", str(FIXTURE_EVENT), str(outdir), "--step-digits", "3"]
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
    assert second.exit_code != 0
    assert "output already exists" in second.output
    assert outdir.joinpath("scalars", "loss.json").read_bytes() == original
    assert forced.exit_code == 0
    assert "\nv " in outdir.joinpath("meshes", "box", "001.obj").read_text()


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
