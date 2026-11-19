"""Write and validate shared native summaries using the caller's writer."""

from io import BytesIO
from pathlib import Path

from google.protobuf import text_format
import numpy as np
from PIL import Image
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
from tensorboard.plugins.projector.projector_config_pb2 import ProjectorConfig
import torch
from torch.utils.tensorboard import SummaryWriter


def _images(writer: SummaryWriter, output: Path) -> None:
    tiles = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    tiles[0, 0, :, :] = 180
    tiles[1, 2, :, :] = 249
    tiles[0, 1, 1:3, 2] = 120
    writer.add_images("grid", tiles, 4, walltime=4)
    writer.add_image_with_boxes(
        "boxed",
        np.full((3, 16, 19), 40, dtype=np.uint8),
        np.array([[3, 4, 12, 11]]),
        4,
        walltime=4,
    )
    writer.flush()
    (event,) = output.glob("events.out.tfevents.*")
    payloads = {
        v.tag: v.image.encoded_image_string
        for e in LegacyEventFileLoader(str(event)).Load()
        for v in e.summary.value
        if v.HasField("image")
    }
    with Image.open(BytesIO(payloads["grid"])) as grid:
        np.testing.assert_array_equal(
            np.asarray(grid), np.concatenate(tiles.transpose(0, 2, 3, 1), axis=1)
        )
    with Image.open(BytesIO(payloads["boxed"])) as boxed:
        pixels = np.asarray(boxed)
        if boxed.size != (19, 16):
            raise ValueError("Unexpected dimensions of boxed output")
        np.testing.assert_array_equal(pixels[0, 0], [40, 40, 40])
        np.testing.assert_array_equal(pixels[8, 8], [40, 40, 40])
        for x, y in [(8, 4), (8, 11), (3, 8), (12, 8)]:
            np.testing.assert_array_equal(pixels[y, x], [255, 0, 0])


def _embeddings(writer: SummaryWriter, output: Path) -> None:
    vectors = np.array([[1, -2], [4.5, 8]])
    tiles = np.zeros((2, 3, 4, 4), dtype=np.float32)
    tiles[0, 1, 1:3, :] = 1
    tiles[1, 2, :, 1:3] = 1
    writer.add_embedding(vectors, global_step=4, tag="vectors")
    writer.add_embedding(
        vectors + 1, metadata=["oak", "ash"], global_step=4, tag="labels"
    )
    writer.add_embedding(
        vectors + 2, label_img=torch.from_numpy(tiles), global_step=4, tag="sprite"
    )
    config = text_format.Parse(
        (output / "projector_config.pbtxt").read_text(), ProjectorConfig()
    )
    names = [entry.tensor_name for entry in config.embeddings]
    if names != ["vectors:00004", "labels:00004", "sprite:00004"]:
        raise ValueError(f"Unexpected projector entries: {names}")
    for i, entry in enumerate(config.embeddings):
        np.testing.assert_array_equal(
            np.loadtxt(output / entry.tensor_path), vectors + i
        )
        if bool(entry.metadata_path) != (i == 1) or bool(entry.sprite.image_path) != (
            i == 2
        ):
            raise ValueError(f"Unexpected optional projector assets: {entry}")
    if (output / config.embeddings[1].metadata_path).read_bytes() != b"oak\nash\n":
        raise ValueError("Unexpected projector labels")
    with Image.open(output / config.embeddings[2].sprite.image_path) as sprite:
        expected = np.zeros((8, 8, 3), dtype=np.uint8)
        expected[:4, :8] = np.concatenate(tiles.transpose(0, 2, 3, 1), axis=1) * 255
        np.testing.assert_array_equal(np.asarray(sprite), expected)


def write_cases(writer: SummaryWriter, output: Path, *, colored_surface: bool) -> None:
    """Write shared summaries and check the native rasters and projector assets."""
    vertices = np.array(
        [[[0, 0, 0], [2, 0, 0], [0, 1, 0]], [[5, 5, 5], [7, 5, 5], [5, 6, 5]]]
    )
    faces = np.array([[[0, 1, 2]], [[2, 1, 0]]])
    colors = (
        np.array(
            [
                [[255, 0, 0], [0, 64, 0], [0, 0, 128]],
                [[0, 0, 128], [0, 64, 0], [255, 0, 0]],
            ]
        )
        if colored_surface
        else None
    )
    writer.add_scalars("score/split", {"left": 1.25, "right": 2.5}, 4, walltime=4)
    writer.add_pr_curve(
        "unweighted",
        np.array([1, 0, 1, 0]),
        np.array([1.0, 0.75, 0.5, 0.25]),
        4,
        num_thresholds=3,
        walltime=4,
    )
    writer.add_mesh("plain-cloud", vertices, global_step=4, walltime=4)
    writer.add_mesh(
        "surface", vertices, colors=colors, faces=faces, global_step=4, walltime=4
    )
    writer.add_hparams(
        {"optimizer": "sgd"},
        {"accuracy": 0.875},
        hparam_domain_discrete={"optimizer": ["adam", "sgd"]},
        run_name="trial",
        global_step=4,
    )
    _embeddings(writer, output)
    _images(writer, output)
