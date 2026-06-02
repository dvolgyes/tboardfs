from io import BytesIO
import json
from pathlib import Path
import shutil

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import soundfile as sf
from tensorboard.compat.proto import (
    event_pb2,
    graph_pb2,
    node_def_pb2,
    summary_pb2,
    tensor_pb2,
    tensor_shape_pb2,
    types_pb2,
)
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboardX import SummaryWriter
import torch


ROOT = Path("test-logs")
WALL_TIME = 1_700_000_000.0


def main() -> None:
    """Generate deterministic TensorBoard and TensorBoardX fixture logs."""
    if ROOT.exists():
        shutil.rmtree(ROOT)

    images: list[np.ndarray] = []
    waves: list[np.ndarray] = []
    histograms: list[np.ndarray] = []
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
    predictions = np.array([0.1, 0.8, 0.7, 0.3, 0.9, 0.4], dtype=np.float32)
    for index in range(3):
        step = index + 1
        wall_time = WALL_TIME + step
        writer.add_scalar("epoch", index, step, walltime=wall_time)
        writer.add_scalar("loss", 1.0 / step, step, walltime=wall_time)
        writer.add_scalar("train/f1_score", 0.6 + index * 0.1, step, walltime=wall_time)
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
            np.clip(predictions - index * 0.05, 0.0, 1.0),
            step,
            walltime=wall_time,
        )
        frames = np.stack([images[index], images[index + 1], images[index + 2]])
        video = torch.from_numpy(frames.transpose(0, 3, 1, 2)[None] / 255.0)
        writer.add_video("clip", video, step, fps=2, walltime=wall_time)
    figure, axis = plt.subplots(figsize=(2, 2))
    axis.plot([0, 1, 2], [1, 0.5, 0.25])
    axis.set_title("loss")
    writer.add_figure("figure/curve", figure, 1, walltime=WALL_TIME + 1)
    writer.add_graph(
        torch.nn.Sequential(
            torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 1)
        ),
        torch.zeros(1, 4),
    )
    writer.add_custom_scalars(
        {"Metrics": {"loss_and_f1": ["Multiline", ["loss", "train/f1_score"]]}}
    )
    writer.add_scalar("hparam/loss", 0.25, 0, walltime=WALL_TIME)
    writer.add_scalar("hparam/f1_score", 0.82, 0, walltime=WALL_TIME)
    writer.flush()
    writer.close()

    raw_x = EventFileWriter(str(tensorboardx_dir))
    for index in range(3):
        step = index + 1
        shape = tensor_shape_pb2.TensorShapeProto(
            dim=[
                tensor_shape_pb2.TensorShapeProto.Dim(size=3),
                tensor_shape_pb2.TensorShapeProto.Dim(size=4),
            ]
        )
        tensor = (np.arange(12, dtype=np.float32).reshape(3, 4) + step).astype("<f4")
        gif_handle = BytesIO()
        imageio.mimsave(
            gif_handle,
            [images[step - 1], images[step], images[step + 1]],
            format="GIF",
            duration=0.25,
        )
        summary = summary_pb2.Summary(
            value=[
                summary_pb2.Summary.Value(
                    tag="box",
                    tensor=tensor_pb2.TensorProto(
                        dtype=types_pb2.DT_FLOAT,
                        tensor_shape=tensor_shape_pb2.TensorShapeProto(
                            dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=1)]
                        ),
                        tensor_content=np.asarray([step], dtype="<f4").tobytes(),
                    ),
                    metadata=summary_pb2.SummaryMetadata(
                        plugin_data=summary_pb2.SummaryMetadata.PluginData(
                            plugin_name="mesh"
                        )
                    ),
                ),
                summary_pb2.Summary.Value(
                    tag="activations",
                    tensor=tensor_pb2.TensorProto(
                        dtype=types_pb2.DT_FLOAT,
                        tensor_shape=shape,
                        tensor_content=tensor.tobytes(),
                    ),
                    metadata=summary_pb2.SummaryMetadata(
                        plugin_data=summary_pb2.SummaryMetadata.PluginData(
                            plugin_name="tensor"
                        )
                    ),
                ),
                summary_pb2.Summary.Value(
                    tag="clip/raw",
                    tensor=tensor_pb2.TensorProto(
                        dtype=types_pb2.DT_STRING, string_val=[gif_handle.getvalue()]
                    ),
                    metadata=summary_pb2.SummaryMetadata(
                        plugin_data=summary_pb2.SummaryMetadata.PluginData(
                            plugin_name="video"
                        )
                    ),
                ),
            ]
        )
        raw_x.add_event(
            event_pb2.Event(wall_time=WALL_TIME + step, step=step, summary=summary)
        )
    raw_x.add_event(
        event_pb2.Event(
            wall_time=WALL_TIME,
            step=0,
            summary=summary_pb2.Summary(
                value=[
                    summary_pb2.Summary.Value(
                        tag="session",
                        metadata=summary_pb2.SummaryMetadata(
                            plugin_data=summary_pb2.SummaryMetadata.PluginData(
                                plugin_name="hparams",
                                content=json.dumps(
                                    {"optimizer": "sgd", "lr": 0.1}
                                ).encode(),
                            )
                        ),
                    )
                ]
            ),
        )
    )
    raw_x.flush()
    raw_x.close()

    tensorboard_dir = ROOT / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    raw = EventFileWriter(str(tensorboard_dir))
    custom_scalars = json.dumps(
        {"Metrics": {"loss_and_f1": ["Multiline", ["loss", "train/f1_score"]]}}
    ).encode()
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
        pr = np.asarray(
            [[3, 0, 3, 0, 1.0, 1.0], [2, 1, 3, 0, 0.67, 1.0], [1, 0, 4, 1, 1.0, 0.5]],
            dtype="<f4",
        )
        tensor = (np.arange(12, dtype=np.float32).reshape(3, 4) + step).astype("<f4")
        pr_shape = tensor_shape_pb2.TensorShapeProto(
            dim=[
                tensor_shape_pb2.TensorShapeProto.Dim(size=3),
                tensor_shape_pb2.TensorShapeProto.Dim(size=6),
            ]
        )
        tensor_shape = tensor_shape_pb2.TensorShapeProto(
            dim=[
                tensor_shape_pb2.TensorShapeProto.Dim(size=3),
                tensor_shape_pb2.TensorShapeProto.Dim(size=4),
            ]
        )
        mesh_shape = tensor_shape_pb2.TensorShapeProto(
            dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=1)]
        )
        summary = summary_pb2.Summary(
            value=[
                summary_pb2.Summary.Value(tag="epoch", simple_value=float(index)),
                summary_pb2.Summary.Value(tag="loss", simple_value=1.5 / step),
                summary_pb2.Summary.Value(
                    tag="train/f1_score", simple_value=0.55 + index * 0.1
                ),
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
                summary_pb2.Summary.Value(
                    tag="quality/pr",
                    tensor=tensor_pb2.TensorProto(
                        dtype=types_pb2.DT_FLOAT,
                        tensor_shape=pr_shape,
                        tensor_content=pr.tobytes(),
                    ),
                    metadata=summary_pb2.SummaryMetadata(
                        plugin_data=summary_pb2.SummaryMetadata.PluginData(
                            plugin_name="pr_curve"
                        )
                    ),
                ),
                summary_pb2.Summary.Value(
                    tag="box",
                    tensor=tensor_pb2.TensorProto(
                        dtype=types_pb2.DT_FLOAT,
                        tensor_shape=mesh_shape,
                        tensor_content=np.asarray([step], dtype="<f4").tobytes(),
                    ),
                    metadata=summary_pb2.SummaryMetadata(
                        plugin_data=summary_pb2.SummaryMetadata.PluginData(
                            plugin_name="mesh"
                        )
                    ),
                ),
                summary_pb2.Summary.Value(
                    tag="clip",
                    tensor=tensor_pb2.TensorProto(
                        dtype=types_pb2.DT_STRING, string_val=[gif_handle.getvalue()]
                    ),
                    metadata=summary_pb2.SummaryMetadata(
                        plugin_data=summary_pb2.SummaryMetadata.PluginData(
                            plugin_name="video"
                        )
                    ),
                ),
                summary_pb2.Summary.Value(
                    tag="activations",
                    tensor=tensor_pb2.TensorProto(
                        dtype=types_pb2.DT_FLOAT,
                        tensor_shape=tensor_shape,
                        tensor_content=tensor.tobytes(),
                    ),
                    metadata=summary_pb2.SummaryMetadata(
                        plugin_data=summary_pb2.SummaryMetadata.PluginData(
                            plugin_name="tensor"
                        )
                    ),
                ),
            ]
        )
        raw.add_event(event_pb2.Event(wall_time=wall_time, step=step, summary=summary))

    raw.add_event(
        event_pb2.Event(
            wall_time=WALL_TIME,
            step=0,
            summary=summary_pb2.Summary(
                value=[
                    summary_pb2.Summary.Value(
                        tag="custom_scalars",
                        metadata=summary_pb2.SummaryMetadata(
                            plugin_data=summary_pb2.SummaryMetadata.PluginData(
                                plugin_name="custom_scalars", content=custom_scalars
                            )
                        ),
                    ),
                    summary_pb2.Summary.Value(
                        tag="session",
                        metadata=summary_pb2.SummaryMetadata(
                            plugin_data=summary_pb2.SummaryMetadata.PluginData(
                                plugin_name="hparams",
                                content=json.dumps(
                                    {"optimizer": "adam", "lr": 0.01}
                                ).encode(),
                            )
                        ),
                    ),
                ]
            ),
        )
    )

    graph = graph_pb2.GraphDef()
    graph.node.extend(
        [
            node_def_pb2.NodeDef(name="input", op="Placeholder"),
            node_def_pb2.NodeDef(name="weights", op="Const"),
            node_def_pb2.NodeDef(name="bias", op="Const"),
            node_def_pb2.NodeDef(
                name="matmul", op="MatMul", input=["input", "weights"]
            ),
            node_def_pb2.NodeDef(name="logits", op="BiasAdd", input=["matmul", "bias"]),
        ]
    )
    raw.add_event(
        event_pb2.Event(
            wall_time=WALL_TIME, step=0, graph_def=graph.SerializeToString()
        )
    )
    raw.flush()
    raw.close()

    for root in (tensorboardx_dir, tensorboard_dir):
        projector = root / "projector"
        projector.mkdir(parents=True, exist_ok=True)
        (projector / "projector_config.pbtxt").write_text(
            'embeddings { tensor_name: "embedding/sample" metadata_path: "metadata.tsv" }\n'
        )
        (projector / "metadata.tsv").write_text("label\nzero\none\ntwo\n")
        (projector / "tensors.tsv").write_text("0.0\t0.0\n1.0\t0.5\n0.5\t1.0\n")
        profile = root / "plugins" / "profile"
        profile.mkdir(parents=True, exist_ok=True)
        (profile / "trace.json").write_text(
            json.dumps(
                {"traceEvents": [{"name": "step", "ph": "X", "ts": 1, "dur": 3}]},
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
