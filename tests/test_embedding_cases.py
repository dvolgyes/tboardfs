"""Export native TensorBoardX projector sidecars and match their rows to portraits."""

from hashlib import sha256
import json
from pathlib import Path

from click.testing import CliRunner
import pytest

from tboardfs import TensorBoardFS
from tboardfs.file_cli import main


ROOT = Path(__file__).parent / "data" / "producer-cases" / "embedding-cases"
BASE = (
    ("Étoile", "amber", -1.0, -0.75),
    ("星", "blue", 0.5, -0.5),
    ("Stern", "red", -0.25, 0.75),
    ("Зоря", "green", 0.75, -1.25),
)
LATER = (
    ("Stern", "red", 0.0, 0.75),
    ("Étoile", "amber", -0.75, -0.75),
    ("Зоря", "green", 1.0, -1.25),
    ("星", "blue", 0.75, -0.5),
)


def _portrait_hash(rows: tuple[tuple[str, str, float, float], ...]) -> str:
    """Reconstruct the expected RGB atlas using only the labeled feature rows."""
    colors = {
        "amber": (255, 192, 0),
        "red": (255, 48, 48),
        "green": (32, 255, 96),
        "blue": (0, 160, 255),
    }
    rgb = bytearray(32 * 32 * 3)
    for i, (_, hue, x, y) in enumerate(rows):
        cx = int(8 + x * 4) + i % 2 * 16
        cy = int(8 + y * 4) + i // 2 * 16
        for delta in range(-2, 3):
            for col, row in ((cx + delta, cy), (cx, cy + delta)):
                start = 3 * (row * 32 + col)
                rgb[start : start + 3] = bytes(colors[hue])
    return sha256(rgb).hexdigest()


@pytest.mark.parametrize(
    ("directory", "step", "labels", "sprite"),
    [
        ("00000/sky%252fpositions", 0, False, False),
        ("00002/catalog", 2, True, False),
        ("00004/finder", 4, False, True),
        ("00007/sky%252fpositions", 7, True, True),
    ],
)
def test_projector_layout_rows(
    directory: str, step: int, labels: bool, sprite: bool
) -> None:
    fs = TensorBoardFS(ROOT)
    folder = f"/projector/{directory}"
    names = {"tensors.tsv"}
    rows = LATER if step == 7 else BASE
    lines = fs.read(f"{folder}/tensors.tsv", 4096, 0).decode().splitlines()
    assert [tuple(map(float, line.split("\t"))) for line in lines] == [
        (x, y) for _, _, x, y in rows
    ]
    if labels:
        names.add("metadata.tsv")
        header, *ids = fs.read(f"{folder}/metadata.tsv", 4096, 0).decode().splitlines()
        assert header == "observation\tcolor"
        assert ids == [f"{name}\t{color}" for name, color, _, _ in rows]
    if sprite:
        names.add("sprite.png")
        hashes = json.loads((ROOT / "decoded.json").read_bytes())[str(step)]
        assert (
            sha256(fs.read(f"{folder}/sprite.png", 8192, 0)).hexdigest()
            == hashes["png"]
        )
        assert _portrait_hash(rows) == hashes["rgb"]
    assert set(fs.readdir(folder, None)[2:]) == names


def test_cli_export_preserves_config_and_escapes_paths(tmp_path: Path) -> None:
    destination = tmp_path / "assets"
    result = CliRunner().invoke(main, ["copy-all", str(ROOT), str(destination)])
    assert result.exit_code == 0, result.output
    projector = destination / "projector"
    config = (projector / "projector_config.pbtxt").read_bytes()
    assert config == (ROOT / "projector_config.pbtxt").read_bytes()
    blocks = config.decode().split("embeddings {")[1:]
    assert len(blocks) == 4
    for block, step, tag, files in zip(
        blocks,
        (0, 2, 4, 7),
        ("sky/positions", "catalog", "finder", "sky/positions"),
        (
            ("tensors.tsv",),
            ("tensors.tsv", "metadata.tsv"),
            ("tensors.tsv", "sprite.png"),
            ("tensors.tsv", "metadata.tsv", "sprite.png"),
        ),
        strict=True,
    ):
        assert f'tensor_name: "{tag}:{step:05d}"' in block
        native = f"{step:05d}/{tag.replace('/', '%2f')}"
        for name in files:
            reference = f"{native}/{name}"
            assert f'"{reference}"' in block
            assert (projector / reference.replace("%", "%25")).read_bytes() == (
                ROOT / reference
            ).read_bytes()
            assert (projector / reference).exists() == ("/" not in tag)
        if "sprite.png" in files:
            assert block.count("single_image_dim: 16") == 2
