"""Generate additional native summaries shared by the pinned historical writers."""

import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from tensorboard.plugins import projector
from tensorboard.plugins.base_plugin import TBContext
from tensorboard.plugins.custom_scalar import layout_pb2
from tensorboard.plugins.custom_scalar import summary as custom_scalar
from tensorboard.plugins.hparams import api as hp
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import summary as hp_summary
from tensorboard.plugins.mesh import summary as mesh
from tensorboard.plugins.pr_curve import summary as pr
from tensorboard.plugins.projector.projector_plugin import ProjectorPlugin
from tensorflow.python.client import timeline
from tensorflow.python.debug.lib import debug_data, debug_utils
from tensorflow.python.ops import summary_ops_v2


def _write_plugins(run: Path, session: tf.compat.v1.Session) -> None:
    """Write batched graph summaries and typed tensors with a native V1 writer."""
    v1 = tf.compat.v1
    vertices = tf.constant(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [3.0, 0.0, 1.0], [0.0, 2.0, 1.0]],
        ]
    )
    faces = tf.constant([[[0, 1, 2]], [[0, 2, 1]]])
    colors = tf.constant([[[255, 0, 0]] * 3, [[0, 0, 255]] * 3], tf.uint8)
    layout = layout_pb2.Layout()
    layout.category.add(title="validation").chart.add(title="range").margin.series.add(
        value="score", lower="low", upper="high"
    )
    values = [
        pr.op(
            "unweighted",
            tf.constant([True, False, True, False]),
            tf.constant([0.9, 0.7, 0.4, 0.1]),
            num_thresholds=3,
        ),
        pr.raw_data_op(
            "precomputed",
            [2, 2, 0],
            [2, 0, 0],
            [0, 2, 2],
            [0, 0, 2],
            [0.5, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            num_thresholds=3,
        ),
        mesh.op("surface_batch", vertices, faces=faces, colors=colors, collections=[]),
        mesh.op("cloud_batch", vertices, collections=[]),
        v1.summary.text("cube", tf.constant([[["one", "two"]], [["三", "x|y"]]])),
        v1.summary.tensor_summary(
            "wide", tf.constant([1.0 + 2.0**-40, -2.5], tf.float64)
        ),
        v1.summary.tensor_summary(
            "phase", tf.constant([1 + 2j, -3 - 4j], tf.complex64)
        ),
        v1.summary.tensor_summary("opaque", tf.constant([b"a\x00b", b"\xff\x80", b""])),
        custom_scalar.op(layout),
    ]
    coordinates = v1.get_variable("coordinates", initializer=[[1.0, 2.0], [3.0, 4.0]])
    saver = v1.train.Saver(save_relative_paths=True)
    session.run(v1.global_variables_initializer())
    saver.save(session, str(run / "coordinates"), write_meta_graph=False)
    np.testing.assert_array_equal(session.run(coordinates), [[1.0, 2.0], [3.0, 4.0]])
    pixels = np.array(
        [[[255, 0, 0], [0, 255, 0]], [[255, 0, 0], [0, 255, 0]]], np.uint8
    )
    png = session.run(tf.image.encode_png(pixels))
    np.testing.assert_array_equal(session.run(tf.image.decode_png(png)), pixels)
    (run / "sprite.png").write_bytes(png)
    (run / "labels.tsv").write_text("left\nright\n", encoding="utf-8")
    scene = {
        "label": "Left point",
        "isSelected": False,
        "dataSetDimensions": [2, 2],
        "selectedProjection": "pca",
        "pcaComponentDimensions": [0, 1],
        "projections": [
            {"pca-0": -(2**0.5), "pca-1": 0},
            {"pca-0": 2**0.5, "pca-1": 0},
        ],
        "selectedPoints": [0],
        "filteredPoints": None,
        "tSNEIteration": 0,
        "tSNEPerplexity": 30,
        "tSNELearningRate": 10,
        "tSNEis3d": False,
        "cameraDef": {
            "orthographic": True,
            "position": [0, 0, 3],
            "target": [0, 0, 0],
            "zoom": 1,
        },
    }
    (run / "bookmarks.json").write_text(
        json.dumps([scene], indent=2) + "\n", encoding="utf-8"
    )
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add(
        tensor_name="coordinates",
        metadata_path="labels.tsv",
        bookmarks_path="bookmarks.json",
    )
    embedding.sprite.image_path = "sprite.png"
    embedding.sprite.single_image_dim.extend([1, 2])
    with v1.summary.FileWriter(str(run), filename_suffix=".summaries") as writer:
        for value in session.run(values):
            writer.add_summary(value, global_step=7)
        projector.visualize_embeddings(
            writer if tf.__version__ == "1.15.0" else str(run), config
        )
    reopened = projector.ProjectorConfig()
    text_format.Parse(
        (run / "projector_config.pbtxt").read_text(encoding="utf-8"), reopened
    )
    if config != reopened:
        raise ValueError("The native projector config changed on reopening")
    plugin = ProjectorPlugin(TBContext(logdir=str(run)))
    configs, _ = plugin._read_latest_config_files([(".", str(run))])
    checkpoint = configs["."].model_checkpoint_path
    np.testing.assert_array_equal(
        tf.train.load_variable(checkpoint, embedding.tensor_name),
        [[1.0, 2.0], [3.0, 4.0]],
    )


def _write_cpu_capture(run: Path) -> None:
    """Capture one watched operation and a CPU timeline from a real Session run."""
    v1 = tf.compat.v1
    with tf.Graph().as_default(), v1.Session() as session:
        with tf.device("/CPU:0"):
            samples = v1.placeholder(tf.float32, shape=[2], name="samples")
            squared = tf.square(samples, name="watched_square")
        options = v1.RunOptions(
            trace_level=v1.RunOptions.FULL_TRACE, output_partition_graphs=True
        )
        dump = run / "plugins" / "debugger" / "capture"
        dump.parent.mkdir(parents=True)
        with TemporaryDirectory(prefix="tboardfs-tfdbg-", dir="/tmp") as temporary:
            native_dump = Path(temporary) / "capture"
            debug_utils.watch_graph(
                options,
                session.graph,
                debug_urls=[native_dump.as_uri()],
                node_name_regex_whitelist="^watched_square$",
            )
            metadata = v1.RunMetadata()
            result = session.run(
                squared, {samples: [-3, 2]}, options=options, run_metadata=metadata
            )
            shutil.move(str(native_dump), str(dump))
        np.testing.assert_array_equal(result, [9, 4])
        trace = timeline.Timeline(metadata.step_stats).generate_chrome_trace_format()
        if not any(
            "watched_square" in item.get("args", {}).get("name", "")
            for item in json.loads(trace)["traceEvents"]
        ):
            raise ValueError("The CPU trace did not record watched_square")
        trace_path = run / "plugins" / "profile" / "capture" / "cpu.trace.json"
        trace_path.parent.mkdir(parents=True)
        trace_path.write_text(trace.rstrip("\n") + "\n", encoding="utf-8")
        reader = debug_data.DebugDumpDir(
            str(dump), partition_graphs=metadata.partition_graphs
        )
        (watched,) = reader.get_tensors("watched_square", 0, "DebugIdentity")
        np.testing.assert_array_equal(watched, [9, 4])


def main() -> Path:
    """Write and reopen historical fixture artifacts in a new or empty directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    run: Path = parser.parse_args().output
    run.mkdir(parents=True, exist_ok=True)
    if any(run.iterdir()):
        parser.error(f"output directory is not empty: {run}")
    if tf.__version__ not in {"1.15.0", "2.0.0"}:
        parser.error(f"unsupported TensorFlow version: {tf.__version__}")
    with tf.Graph().as_default(), tf.compat.v1.Session() as session:
        _write_plugins(run, session)
        writer = tf.compat.v2.summary.create_file_writer(
            str(run), filename_suffix=".dashboard"
        )
        with writer.as_default():
            kind = hp.HParam("kind", hp.Discrete(["linear", "square"]))
            enabled = hp.HParam("enabled", hp.Discrete([False, True]))
            rate = hp.HParam("rate", hp.RealInterval(0.125, 0.5))
            bins = hp.HParam("bins", hp.IntInterval(2, 5))
            ops = [
                hp.hparams_config(
                    [kind, enabled, rate, bins],
                    [hp.Metric("score")],
                    time_created_secs=10,
                ),
                hp.hparams(
                    {kind: "square", enabled: True, rate: 0.25, bins: 3},
                    trial_id="sweep",
                    start_time_secs=11,
                ),
                tf.compat.v2.summary.experimental.write_raw_pb(
                    hp_summary.session_end_pb(
                        api_pb2.STATUS_SUCCESS, end_time_secs=12
                    ).SerializeToString(),
                    step=7,
                ),
            ]
            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        3, activation="relu", input_shape=(2,), name="hidden"
                    ),
                    tf.keras.layers.Dense(1, name="estimate"),
                ],
                name="historical_sequential",
            )
            ops.append(summary_ops_v2.keras_model("model", model, step=7))
        session.run(writer.init())
        np.testing.assert_array_equal(session.run(ops), [True] * len(ops))
        session.run(writer.close())
    _write_cpu_capture(run)
    for suffix in ("summaries", "dashboard"):
        (event,) = run.glob(f"events.out.tfevents.*.{suffix}")
        records = list(tf.compat.v1.train.summary_iterator(str(event)))
        if not any(record.HasField("summary") for record in records):
            raise ValueError(f"Native writer produced no summaries in {event}")
        event.rename(run / f"events.out.tfevents.{suffix}")
    return run
