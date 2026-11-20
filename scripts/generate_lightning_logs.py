#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "lightning>=2.6.6,<3.0",
#   "numpy>=2.1,<3.0",
#   "tensorboard>=2.20.0,<3.0",
#   "torch>=2.12.0,<3.0",
# ]
# ///
"""Generate a small PyTorch Lightning fixture log under test-logs/lightning.

TensorBoardLogger writes the logged metrics, the epoch scalar, hparams with
hp_metric, hparams.yaml and the model graph. The model also writes an image,
a histogram, audio and text through the logger's SummaryWriter.
"""

import shutil
from pathlib import Path

import torch
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path("test-logs")
NAME = "lightning"


# lightning is not installed in the project environment, so mypy sees Any.
class _TinyModel(LightningModule):  # type: ignore[misc]
    """Fit y = x0 + x1 with one linear layer.

    Attributes:
        layer: the linear model.
        example_input_array: the input that traces the logged graph.
    """

    layer: torch.nn.Linear
    example_input_array: torch.Tensor

    def __init__(self, *, lr: float = 0.1) -> None:
        super().__init__()
        self.save_hyperparameters({"lr": lr})
        self.layer = torch.nn.Linear(2, 1)
        self.example_input_array = torch.zeros(1, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return the prediction."""
        return self.layer(features)

    def training_step(self, batch: list[torch.Tensor], _: int) -> torch.Tensor:
        """Log and return the training loss."""
        loss = torch.nn.functional.mse_loss(self(batch[0]), batch[1])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: list[torch.Tensor], _: int) -> None:
        """Log the validation loss."""
        loss = torch.nn.functional.mse_loss(self(batch[0]), batch[1])
        self.log("val_loss", loss)

    def on_validation_epoch_end(self) -> None:
        """Write media summaries once per epoch."""
        writer = self.logger.experiment
        step = self.current_epoch
        grid = torch.arange(24.0).reshape(1, 4, 6) / 23.0
        writer.add_image("pattern", grid * (step + 1) / 3, step)
        writer.add_histogram("layer/weight", self.layer.weight, step)
        tone = torch.sin(torch.linspace(0, 40.0 * (step + 1), 160))
        writer.add_audio("tone", tone, step, sample_rate=8000)
        writer.add_text("note", f"epoch **{step}**", step)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return plain SGD."""
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)


def main() -> None:
    """Train two epochs on CPU, replacing an earlier log."""
    if (ROOT / NAME).exists():
        shutil.rmtree(ROOT / NAME)
    seed_everything(0)
    features = torch.linspace(-1.0, 1.0, 32).reshape(16, 2)
    data = TensorDataset(features, features.sum(dim=1, keepdim=True))
    trainer = Trainer(
        max_epochs=2,
        accelerator="cpu",
        logger=TensorBoardLogger(ROOT, name=NAME, version=0, log_graph=True),
        log_every_n_steps=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(
        _TinyModel(),
        DataLoader(data, batch_size=8),
        DataLoader(data, batch_size=8),
    )


if __name__ == "__main__":
    main()
