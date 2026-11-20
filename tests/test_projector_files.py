"""Embedding files named in projector_config.pbtxt are listed under /projector."""

from pathlib import Path

import pytest
from test_incremental_parse import _files
from test_open_handles import _append

from tboardfs import TensorBoardFS
from tboardfs.projector import CONFIG_NAME, _ProjectorConfig

EVENTS = "events.out.tfevents.host"

# The layout that PyTorch and tensorboardX add_embedding write.
CONFIG = """embeddings {
tensor_name: "feat:00003"
tensor_path: "00003/feat/tensors.tsv"
metadata_path: "00003/feat/metadata.tsv"
sprite {
image_path: "00003/feat/sprite.png"
single_image_dim: 8
}
}
"""


def _run(tmp_path: Path, config: str, *files: str) -> TensorBoardFS:
    run = tmp_path / "run"
    run.mkdir()
    _append(run / EVENTS, 0)
    (run / "projector_config.pbtxt").write_text(config)
    for name in files:
        path = run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"data of {name}")
    return TensorBoardFS(tmp_path)


def _projector(fs: TensorBoardFS) -> list[str]:
    return [p for p in _files(fs) if p.startswith("/run/projector/")]


def test_referenced_files_keep_their_relative_paths(tmp_path: Path) -> None:
    """Paths in the config resolve inside /projector, as next to the config."""
    fs = _run(
        tmp_path,
        CONFIG,
        "00003/feat/tensors.tsv",
        "00003/feat/metadata.tsv",
        "00003/feat/sprite.png",
    )

    assert _projector(fs) == [
        "/run/projector/00003/feat/metadata.tsv",
        "/run/projector/00003/feat/sprite.png",
        "/run/projector/00003/feat/tensors.tsv",
        "/run/projector/projector_config.pbtxt",
    ]
    assert fs.read("/run/projector/00003/feat/tensors.tsv", 100, 0) == (
        b"data of 00003/feat/tensors.tsv"
    )


def test_files_that_the_config_does_not_name_stay_hidden(tmp_path: Path) -> None:
    """Other files in the step directories are not projector files."""
    fs = _run(tmp_path, CONFIG, "00003/feat/tensors.tsv", "00003/feat/notes.txt")

    assert _projector(fs) == [
        "/run/projector/00003/feat/tensors.tsv",
        "/run/projector/projector_config.pbtxt",
    ]


def test_paths_outside_the_run_are_not_followed(tmp_path: Path) -> None:
    """Parent, absolute, and other-run paths in the config expose nothing."""
    other = tmp_path / "other"
    other.mkdir()
    _append(other / EVENTS, 0)
    (other / "secret.tsv").write_text("secret")
    (tmp_path / "outside.tsv").write_text("outside")
    config = "".join(
        f'embeddings {{ tensor_path: "{path}" }}\n'
        for path in (
            "../outside.tsv",
            "../other/secret.tsv",
            str(tmp_path / "outside.tsv"),
            "a/../../outside.tsv",
            EVENTS,
            "a\\\\b.tsv",
            "",
        )
    )
    fs = _run(tmp_path, config, "a/b.tsv")

    assert _projector(fs) == ["/run/projector/projector_config.pbtxt"]


def test_an_absolute_path_inside_the_run_is_not_listed(tmp_path: Path) -> None:
    """Only relative paths map to a place under /projector."""
    config = f'embeddings {{ tensor_path: "{tmp_path / "run" / "a" / "b.tsv"}" }}\n'
    fs = _run(tmp_path, config, "a/b.tsv")

    assert _projector(fs) == ["/run/projector/projector_config.pbtxt"]


def test_escaped_strings_are_not_read_as_file_names(tmp_path: Path) -> None:
    """The pbtxt string "a\\\\b.tsv" means a\\b.tsv, not a file named a\\\\b.tsv."""
    fs = _run(tmp_path, 'embeddings { tensor_path: "a\\\\b.tsv" }\n', "a\\\\b.tsv")

    assert _projector(fs) == ["/run/projector/projector_config.pbtxt"]


def test_a_named_file_must_be_a_candidate(tmp_path: Path) -> None:
    """The config selects among the walked sidecar files; it opens nothing else."""
    config = tmp_path / CONFIG_NAME
    config.write_text('embeddings { tensor_path: "outside.tsv" }\n')
    (tmp_path / "outside.tsv").write_text("outside")

    assert _ProjectorConfig.references([config]) == {}


@pytest.mark.parametrize(
    ("padding", "listed"),
    [
        ((1 << 20) - 100, ["/run/projector/a/b.tsv"]),
        (1 << 20, []),
    ],
)
def test_only_the_first_mebibyte_of_the_config_is_read(
    tmp_path: Path, padding: int, listed: list[str]
) -> None:
    """A huge config does not make each refresh read all of it."""
    config = "#" * (padding - 1) + '\nembeddings { tensor_path: "a/b.tsv" }\n'
    fs = _run(tmp_path, config, "a/b.tsv")

    assert _projector(fs) == [*listed, "/run/projector/projector_config.pbtxt"]
