#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["tensorboardX==2.6.5", "numpy==2.2.6", "pillow==11.3.0"]
# ///
"""Generate four TensorBoardX projector layouts from labeled star portraits."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tensorboardX import SummaryWriter


STARS = (
    ("Étoile", "amber", 4, 5, (255, 192, 0)),
    ("星", "blue", 10, 6, (0, 160, 255)),
    ("Stern", "red", 7, 11, (255, 48, 48)),
    ("Зоря", "green", 11, 3, (32, 255, 96)),
)


def _embedding(
    writer: SummaryWriter, output: Path, step: int, tag: str, labels: bool, sprite: bool
) -> dict[str, str] | None:
    """Write normalized position features and decode every corresponding 16x16 tile."""
    order = (2, 0, 3, 1) if step == 7 else (0, 1, 2, 3)
    images = np.zeros((4, 16, 16, 3), dtype=np.uint8)
    vectors = []
    metadata = []
    for row, index in enumerate(order):
        label, hue, x, y, color = STARS[index]
        x += int(step == 7)
        vectors.append([(x - 8) / 4, (y - 8) / 4])
        metadata.append([label, hue])
        images[row, y - 2 : y + 3, x] = color
        images[row, y, x - 2 : x + 3] = color
    writer.add_embedding(
        np.array(vectors),
        metadata=metadata if labels else None,
        metadata_header=["observation", "color"] if labels else None,
        label_img=images.transpose(0, 3, 1, 2) / 255 if sprite else None,
        global_step=step,
        tag=tag,
    )
    if not sprite:
        return None
    (path,) = (output / f"{step:05d}").glob("*/sprite.png")
    with Image.open(path) as image:
        expected = np.concatenate(
            [np.concatenate(images[:2], axis=1), np.concatenate(images[2:], axis=1)]
        )
        np.testing.assert_array_equal(np.array(image), expected)
        return {
            "rgb": sha256(image.tobytes()).hexdigest(),
            "png": sha256(path.read_bytes()).hexdigest(),
        }


def main() -> None:
    """Create one shared configuration and all its sidecars in an empty directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")
    hashes = {}
    with SummaryWriter(logdir=str(output)) as writer:
        for step, tag, labels, sprite in (
            (0, "sky/positions", False, False),
            (2, "catalog", True, False),
            (4, "finder", False, True),
            (7, "sky/positions", True, True),
        ):
            digest = _embedding(writer, output, step, tag, labels, sprite)
            if digest is not None:
                hashes[step] = digest
    (source,) = output.glob("events.out.tfevents.*")
    source.rename(output / "events.out.tfevents.embedding-cases")
    (output / "decoded.json").write_text(json.dumps(hashes, indent=2) + "\n")


if __name__ == "__main__":
    main()
