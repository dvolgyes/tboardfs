from io import BytesIO
from pathlib import Path
import shutil

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import soundfile as sf
from tensorboard.compat.proto import event_pb2, summary_pb2, tensor_pb2, types_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboardX import SummaryWriter
import torch


ROOT = Path("test-logs")
WALL_TIME = 1_700_000_000.0


def main() -> None:
    """Generate deterministic TensorBoard and TensorBoardX fixture logs."""
    if ROOT.exists():
        shutil.rmtree(ROOT)

    images = []
    waves = []
    histograms = []
    for index in range(5):
        grid = np.indices((8, 8)).sum(axis=0)
        red = ((grid + index) % 2) * 255
        green = np.full((8, 8), (index * 70) % 256, dtype=np.uint8)
        blue = np.flipud(red).astype(np.uint8)
        images.append(np.stack([red.astype(np.uint8), green, blue], axis=-1))
        times = np.linspace(0.0, 0.1, 800, endpoint=False, dtype=np.float32)
        waves.append(
            (0.2 * np.sin(2.0 * np.pi * (220 + index * 40) * times)).astype(np.float32)
        )
        histograms.append(np.linspace(-1.0, 1.0, 32, dtype=np.float32) + index * 0.25)

    tensorboardx_dir = ROOT / "tensorboardx"
    writer = SummaryWriter(log_dir=tensorboardx_dir)
    labels = np.array([0, 1, 1, 0, 1, 0], dtype=np.uint8)
    base_predictions = np.array([0.1, 0.8, 0.7, 0.3, 0.9, 0.4], dtype=np.float32)
    vertices = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    faces = torch.tensor([[[0, 1, 2]]])
    for index in range(3):
        step = index + 1
        wall_time = WALL_TIME + step
        writer.add_scalar("epoch", index, step, walltime=wall_time)
        writer.add_scalar("loss", 1.0 / step, step, walltime=wall_time)
        writer.add_scalar("train/loss", 2.0 / step, step, walltime=wall_time)
        writer.add_image(
            "sample", images[index], step, walltime=wall_time, dataformats="HWC"
        )
        writer.add_image(
            "eval/sample",
            images[index + 1],
            step,
            walltime=wall_time,
            dataformats="HWC",
        )
        writer.add_histogram("weights", histograms[index], step, walltime=wall_time)
        writer.add_audio(
            "tone", waves[index], step, sample_rate=8_000, walltime=wall_time
        )
        writer.add_text(
            "notes",
            f"### step {step}\n\nloss changed to {1.0 / step:.3f}",
            step,
            walltime=wall_time,
        )
        writer.add_pr_curve(
            "quality/pr",
            labels,
            np.clip(base_predictions - index * 0.05, 0.0, 1.0),
            step,
            walltime=wall_time,
        )
        writer.add_mesh(
            "shape/triangle",
            vertices=vertices,
            colors=torch.tensor([[[255, index * 60, 0], [0, 255, 0], [0, 0, 255]]]),
            faces=faces,
            global_step=step,
            walltime=wall_time,
        )
        frames = np.stack([images[index], images[index + 1], images[index + 2]])
        video = torch.from_numpy(frames.transpose(0, 3, 1, 2)[None] / 255.0)
        writer.add_video("clip", video, step, fps=2, walltime=wall_time)
    figure, axis = plt.subplots(figsize=(2, 2))
    axis.plot([0, 1, 2], [1, 0.5, 0.25])
    axis.set_title("loss")
    writer.add_figure("figure/curve", figure, 1, walltime=WALL_TIME + 1)
    writer.add_graph(torch.nn.Linear(2, 1), torch.zeros(1, 2))
    writer.add_embedding(
        torch.tensor([[0.0, 0.0], [1.0, 0.5], [0.5, 1.0]], dtype=torch.float32),
        metadata=["zero", "one", "two"],
        label_img=torch.from_numpy(np.stack(images[:3]).transpose(0, 3, 1, 2)),
        global_step=0,
        tag="embedding/sample",
    )
    writer.add_hparams(
        {"optimizer": "sgd", "lr": 0.1},
        {"hparam/loss": 0.25, "hparam/accuracy": 0.75},
    )
    writer.flush()
    writer.close()

    tensorboard_dir = ROOT / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    raw_writer = EventFileWriter(str(tensorboard_dir))
    for index in range(3):
        step = index + 1
        wall_time = WALL_TIME + step
        png_handle = BytesIO()
        Image.fromarray(images[index]).save(png_handle, format="PNG")
        eval_png_handle = BytesIO()
        Image.fromarray(images[index + 1]).save(eval_png_handle, format="PNG")
        wav_handle = BytesIO()
        sf.write(wav_handle, waves[index], 8_000, format="WAV")
        gif_handle = BytesIO()
        imageio.mimsave(
            gif_handle,
            [images[index], images[index + 1], images[index + 2]],
            format="GIF",
            duration=0.25,
        )
        counts, edges = np.histogram(histograms[index], bins=4)
        summary = summary_pb2.Summary(
            value=[
                summary_pb2.Summary.Value(tag="epoch", simple_value=float(index)),
                summary_pb2.Summary.Value(tag="loss", simple_value=1.5 / step),
                summary_pb2.Summary.Value(tag="train/loss", simple_value=2.5 / step),
                summary_pb2.Summary.Value(
                    tag="sample",
                    image=summary_pb2.Summary.Image(
                        height=8,
                        width=8,
                        colorspace=3,
                        encoded_image_string=png_handle.getvalue(),
                    ),
                ),
                summary_pb2.Summary.Value(
                    tag="eval/sample",
                    image=summary_pb2.Summary.Image(
                        height=8,
                        width=8,
                        colorspace=3,
                        encoded_image_string=eval_png_handle.getvalue(),
                    ),
                ),
                summary_pb2.Summary.Value(
                    tag="weights",
                    histo=summary_pb2.HistogramProto(
                        min=float(histograms[index].min()),
                        max=float(histograms[index].max()),
                        num=float(histograms[index].size),
                        sum=float(histograms[index].sum()),
                        sum_squares=float(np.square(histograms[index]).sum()),
                        bucket_limit=[float(edge) for edge in edges[1:]],
                        bucket=[float(count) for count in counts],
                    ),
                ),
                summary_pb2.Summary.Value(
                    tag="tone",
                    audio=summary_pb2.Summary.Audio(
                        sample_rate=8_000.0,
                        num_channels=1,
                        length_frames=len(waves[index]),
                        encoded_audio_string=wav_handle.getvalue(),
                        content_type="audio/wav",
                    ),
                ),
                summary_pb2.Summary.Value(
                    tag="notes",
                    tensor=tensor_pb2.TensorProto(
                        dtype=types_pb2.DT_STRING,
                        string_val=[
                            f"### step {step}\n\nraw TensorBoard writer".encode()
                        ],
                    ),
                    metadata=summary_pb2.SummaryMetadata(
                        plugin_data=summary_pb2.SummaryMetadata.PluginData(
                            plugin_name="text"
                        )
                    ),
                ),
            ]
        )
        plugin_payloads = (
            ("quality/pr", "pr_curve", b'{"name": "pr", "values": [1, 2, 3]}\n'),
            ("shape/triangle", "mesh", b'{"name": "mesh", "values": [1, 2, 3]}\n'),
            ("clip", "video", gif_handle.getvalue()),
            (
                "tensor/blob",
                "blob",
                b'{"name": "tensor", "values": [1, 2, 3]}\n' * 40,
            ),
            ("hparams/session", "hparams", b"session-start"),
        )
        for tag, plugin_name, payload in plugin_payloads:
            summary.value.append(
                summary_pb2.Summary.Value(
                    tag=tag,
                    tensor=tensor_pb2.TensorProto(
                        dtype=types_pb2.DT_STRING, string_val=[payload]
                    ),
                    metadata=summary_pb2.SummaryMetadata(
                        plugin_data=summary_pb2.SummaryMetadata.PluginData(
                            plugin_name=plugin_name,
                            content=payload if plugin_name == "hparams" else b"",
                        )
                    ),
                )
            )
        raw_writer.add_event(
            event_pb2.Event(wall_time=wall_time, step=step, summary=summary)
        )
    raw_writer.add_event(
        event_pb2.Event(wall_time=WALL_TIME, step=0, graph_def=b"graph-def-fixture")
    )
    raw_writer.flush()
    raw_writer.close()
    projector = tensorboard_dir / "projector"
    projector.mkdir()
    (projector / "projector_config.pbtxt").write_text(
        'embeddings { tensor_name: "embedding/sample" }\n'
    )
    (projector / "metadata.tsv").write_text("label\nzero\none\ntwo\n")


if __name__ == "__main__":
    main()
