#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "numpy>=2.1,<3.0",
#   "tensorboard>=2.20.0,<3.0",
#   "tensorflow>=2.21.0,<3.0",
# ]
# ///
"""Generate small TensorFlow 2 fixture logs under test-logs/tf2.

The tf.summary run holds tensor-format summaries: images, audio, histograms,
text, a tensor without a plugin name, hparams and a traced graph. The Keras
run is what the TensorBoard callback writes for a tiny model.
"""

from pathlib import Path
import shutil

from tensorboard.plugins.hparams import api as hp
import numpy as np
import tensorflow as tf

ROOT = Path("test-logs") / "tf2"


def main() -> None:
    """Write the tf.summary and Keras runs, replacing earlier ones."""
    if ROOT.exists():
        shutil.rmtree(ROOT)
    tf.keras.utils.set_random_seed(0)
    _write_summaries(ROOT / "summary")
    _fit_keras(ROOT / "keras")


def _write_summaries(run: Path) -> None:
    rng = np.random.default_rng(0)
    grid = np.indices((4, 6)).sum(axis=0) / 8.0
    writer = tf.summary.create_file_writer(str(run))
    with writer.as_default():
        hp.hparams_config(
            hparams=[hp.HParam("lr", hp.RealInterval(1e-4, 1e-1))],
            metrics=[hp.Metric("loss")],
        )
        hp.hparams({"lr": 0.01})
        for step in range(1, 4):
            tf.summary.scalar("loss", 1.0 / step, step=step)
            images = np.stack([grid, grid[::-1]])[..., None] * step / 3.0
            tf.summary.image("sample", images, step=step, max_outputs=2)
            wave = np.sin(np.linspace(0, 40 * step, 160)).reshape(1, 160, 1)
            tf.summary.audio("tone", wave.astype("float32"), 8000, step=step)
            tf.summary.histogram("weights", rng.normal(step, 1.0, 200), step=step)
            tf.summary.histogram("constant", np.full(10, float(step)), step=step)
            tf.summary.text("note", f"step **{step}**", step=step)
            tf.summary.text("table", [["a", "b"], [str(step), "x"]], step=step)
            tf.summary.write("raw", tf.reshape(tf.range(6.0) * step, (2, 3)), step)
        tf.summary.trace_on(graph=True)
        tf.function(lambda values: values * values)(tf.constant([1.0, 2.0]))
        tf.summary.trace_export("square", step=0)
    writer.close()


def _fit_keras(run: Path) -> None:
    features = np.linspace(-1.0, 1.0, 32, dtype="float32").reshape(16, 2)
    targets = features.sum(axis=1, keepdims=True)
    model = tf.keras.Sequential([tf.keras.Input((2,)), tf.keras.layers.Dense(1)])
    model.compile(optimizer="sgd", loss="mse")
    callback = tf.keras.callbacks.TensorBoard(str(run), histogram_freq=1)
    model.fit(
        features,
        targets,
        epochs=2,
        batch_size=8,
        validation_split=0.25,
        callbacks=[callback],
        verbose=0,
    )


if __name__ == "__main__":
    main()
