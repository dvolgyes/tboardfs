#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "lightning==2.6.6", "pytorch-lightning==2.6.6", "torch==2.12.0+cpu",
#   "tensorboard==2.21.0", "numpy==2.2.6", "pillow==11.3.0",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Train a small CPU segmentation model and log a dense TensorBoard preview."""

import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from PIL import Image, ImageDraw, ImageFont
from tensorboard.compat.proto.summary_pb2 import Summary
from torch import nn
from torch.nn.functional import max_pool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

_SIZE = 96
_LR = 0.005


def _histogram(
    writer: SummaryWriter, phase: str, step: int, scores: torch.Tensor
) -> None:
    """Retain all 100 probability bins, including empty bins."""
    values = scores.numpy().astype(np.float64)
    counts, edges = np.histogram(values, bins=100, range=(0, 1))
    writer.add_histogram_raw(
        f"{phase}/probabilities",
        min=values.min(),
        max=values.max(),
        num=values.size,
        sum=values.sum(),
        sum_squares=np.square(values).sum(),
        bucket_limits=edges[1:],
        bucket_counts=counts,
        global_step=step,
    )


def _metrics(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Measure pixel classification at a probability threshold of 0.5."""
    mask = scores >= 0.5
    tp = int((mask & labels).sum())
    predicted = int(mask.sum())
    actual = int(labels.sum())
    return {
        "accuracy": float((mask == labels).float().mean()),
        "precision": tp / max(1, predicted),
        "recall": tp / actual,
        "iou": tp / (predicted + actual - tp),
        "foreground_fraction": actual / labels.numel(),
        "probability_mean": float(scores.mean()),
        "probability_stddev": float(scores.std(correction=0)),
    }


class _Segmenter(LightningModule):  # type: ignore[misc]
    """Learn to recover the upright digit mask from its rotated image."""

    example_input_array: torch.Tensor
    stem: nn.Conv2d
    context: nn.Conv2d
    head: nn.Conv2d
    loss: nn.BCEWithLogitsLoss
    tb: TensorBoardLogger
    seed: int
    train_steps: int
    val_steps: int
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont

    def __init__(
        self,
        logger: TensorBoardLogger,
        epochs: int,
        train: int,
        val: int,
        seed: int,
        *,
        synthetic_loss: bool,
        stored_png: bool,
    ) -> None:
        super().__init__()
        self.tb = logger
        self.seed = seed
        self.train_steps, self.val_steps = train, val
        digits = len(str(epochs * (train + val) - 1))
        self.font = ImageFont.load_default(size=min(46, 114 // digits))
        self.stem = nn.Conv2d(1, 8, 3, padding=1)
        self.context = nn.Conv2d(8, 8, 3, dilation=2, padding=2)
        self.head = nn.Conv2d(8, 1, 1)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(4.0))
        self.example_input_array = torch.zeros(1, 1, _SIZE, _SIZE)
        self.save_hyperparameters(
            {
                "epochs": epochs,
                "train_steps": train,
                "val_steps": val,
                "seed": seed,
                "synthetic_loss": synthetic_loss,
                "stored_png": stored_png,
                "image_size": _SIZE,
                "learning_rate": _LR,
                "optimizer": "AdamW",
                "model": "residual_cnn",
            }
        )

    def _sample(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Render the step number and rotate it without changing the binary pixels."""
        image = Image.new("L", (_SIZE, _SIZE))
        draw = ImageDraw.Draw(image)
        draw.fontmode = "1"
        draw.text((_SIZE // 2, _SIZE // 2), str(step), 255, font=self.font, anchor="mm")
        rng = np.random.default_rng((self.seed, step))
        angle = rng.choice((-1, 1)) * rng.uniform(8, 28)
        rotated = image.rotate(float(angle), resample=Image.Resampling.NEAREST)
        pixels = torch.from_numpy(np.stack([np.asarray(image), np.asarray(rotated)]))
        raw, augmented = pixels[:, None, :, :].unbind()
        return raw, augmented

    def _images(
        self, phase: str, step: int, raw: torch.Tensor, augmented: torch.Tensor
    ) -> None:
        """Log the input, its augmentation and an explicit morphological baseline."""
        predicted = max_pool2d(raw.float(), 3, stride=1, padding=1).to(torch.uint8)
        writer = self.tb.experiment
        for name, image in {
            "input": raw,
            "augmented": augmented,
            "prediction": predicted,
            "label": raw,
            "difference": predicted ^ raw,
        }.items():
            tag = f"{phase}/{name}"
            if self.hparams.stored_png:
                writer.file_writer.add_summary(self._png(tag, image), step)
            else:
                writer.add_image(tag, image, step)

    @staticmethod
    def _png(tag: str, image: torch.Tensor) -> Summary:
        """Encode the same RGB pixels as SummaryWriter.add_image without compression."""
        pixels = Image.fromarray(image.squeeze(0).numpy()).convert("RGB")
        with BytesIO() as buffer:
            pixels.save(buffer, format="PNG", compress_level=0)
            encoded = buffer.getvalue()
        return Summary(
            value=[
                Summary.Value(
                    tag=tag,
                    image=Summary.Image(
                        height=_SIZE,
                        width=_SIZE,
                        colorspace=3,
                        encoded_image_string=encoded,
                    ),
                )
            ]
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Return foreground logits at the original image resolution."""
        features = self.stem(image).relu()
        return self.head(features + self.context(features).relu())

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Use AdamW with a fixed learning rate."""
        return torch.optim.AdamW(self.parameters(), lr=_LR)

    def on_fit_start(self) -> None:
        """Record run-wide hyperparameters and describe the baseline images."""
        count = sum(p.numel() for p in self.parameters())
        self.tb.log_hyperparams(dict(self.hparams), {"model/parameters": count})
        loss_note = (
            "The reported loss follows a seeded synthetic curve; model_loss records the CNN loss."
            if self.hparams.synthetic_loss
            else "The reported loss measures the CNN."
        )
        self.tb.experiment.add_text(
            "run/description",
            "Recover the upright iteration number from its rotated image. "
            "The prediction PNG is a three-by-three dilation baseline. "
            f"The histogram and classification metrics measure the trained CNN. {loss_note}",
            0,
        )

    def training_step(self, index: int, _: int) -> torch.Tensor:
        """Train on one image and log the training iteration."""
        return self._step(index, "train")

    def validation_step(self, index: int, _: int) -> None:
        """Evaluate a distinct image without updating the CNN."""
        self._step(self.train_steps + index, "val")

    def _step(self, index: int, phase: str) -> torch.Tensor:
        """Train or evaluate one sample at its shared train/validation step."""
        step = self.current_epoch * (self.train_steps + self.val_steps) + index
        raw, augmented = self._sample(step)
        labels = raw.bool().unsqueeze(0)
        logits = self(augmented.float().unsqueeze(0) / 255)
        loss = self.loss(logits, labels.float())
        self._images(phase, step, raw, augmented)
        scores = logits.detach().sigmoid()
        writer = self.tb.experiment
        _histogram(writer, phase, step, scores)
        metrics = _metrics(scores, labels)
        metrics.update(
            loss=float(loss.detach()), learning_rate=_LR, epoch=self.current_epoch
        )
        if self.hparams.synthetic_loss:
            rng = np.random.default_rng((self.seed, step, 1))
            metrics["model_loss"] = metrics["loss"]
            metrics["loss"] = float(rng.uniform(0.9, 1.1)) / (step + 1)
        if phase == "val":
            p, r = metrics["precision"], metrics["recall"]
            metrics["f1"] = 2 * p * r / (p + r) if p + r else 0
            writer.add_pr_curve(
                "val/pr_curve",
                labels.flatten(),
                scores.flatten(),
                step,
                num_thresholds=101,
            )
        self.tb.log_metrics(
            {f"{phase}/{name}": value for name, value in metrics.items()}, step
        )
        return loss


def main() -> None:
    """Generate a native event file in a new or empty output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("-e", "--epochs", type=int, default=2)
    parser.add_argument(
        "-t", "--train-steps", type=int, default=50, help="iterations per epoch"
    )
    parser.add_argument(
        "-v", "--val-steps", type=int, default=2, help="iterations per epoch"
    )
    parser.add_argument("-s", "--seed", type=int, default=1729)
    parser.add_argument(
        "-L",
        "--synthetic-loss",
        action="store_true",
        help="log a noisy inverse-step loss and keep the true loss as model_loss",
    )
    parser.add_argument(
        "-P",
        "--stored-png",
        action="store_true",
        help="write lossless PNGs without compression",
    )
    args = parser.parse_args()
    if min(args.epochs, args.train_steps, args.val_steps) < 1:
        parser.error("epoch and iteration counts must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in the range 0 to 4294967295")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error(f"output directory is not empty: {args.output}")

    seed_everything(args.seed)
    torch.set_num_threads(1)
    logger = TensorBoardLogger(
        str(args.output),
        name="segmentation",
        version="",
        log_graph=True,
        default_hp_metric=False,
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        deterministic=True,
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=logger,
    )
    trainer.fit(
        _Segmenter(
            logger,
            args.epochs,
            args.train_steps,
            args.val_steps,
            args.seed,
            synthetic_loss=args.synthetic_loss,
            stored_png=args.stored_png,
        ),
        train_dataloaders=DataLoader(range(args.train_steps), batch_size=None),
        val_dataloaders=DataLoader(range(args.val_steps), batch_size=None),
    )


if __name__ == "__main__":
    main()
