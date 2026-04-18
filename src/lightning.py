"""
PyTorch Lightning training module for EEG spectrogram classification.

BrainLightning wraps any model that produces log-probabilities and trains it
with KL divergence loss against soft expert-vote targets, AdamW + optional
cosine LR schedule, and optional mixup augmentation.
"""

import numpy as np
import torch
from torch import nn
from pytorch_lightning import LightningModule


class BrainLightning(LightningModule):
    """Lightning training wrapper for EEG spectrogram classifiers.

    Supports:
      - KL divergence loss (model returns log-probs, targets are probability distributions)
      - Mixup augmentation (Beta(alpha, alpha) mixing in input and label space)
      - AdamW optimiser with optional cosine annealing LR schedule
      - Verbose per-epoch logging of loss and current learning rate
    """

    def __init__(
        self,
        model: nn.Module,
        n_classes: int = 6,
        lr: float = 1e-3,
        mixup: bool = False,
        mixup_alpha: float = 0.4,
        scheduler: bool = True,
        t_max: int = 50,
        weight_decay: float = 1e-4,
        verbose: bool = True,
    ):
        """
        Args:
            model:        Any nn.Module that accepts (batch, 4, 100, 25) and returns
                          log-softmax probabilities of shape (batch, n_classes).
            n_classes:    Number of output classes (saved to hparams, default 6).
            lr:           Initial learning rate for AdamW.
            mixup:        If True, apply mixup to each training batch.
            mixup_alpha:  Beta distribution parameter for mixup (default 0.4).
            scheduler:    If True, use CosineAnnealingLR; otherwise constant lr.
            t_max:        T_max for CosineAnnealingLR (in epochs).
            weight_decay: L2 regularisation for AdamW.
            verbose:      If True, print loss and lr at the end of each epoch.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model       = model
        self.mixup       = mixup
        self.mixup_alpha = mixup_alpha
        self.scheduler   = scheduler
        self.t_max       = t_max
        self.verbose     = verbose
        # model outputs log-probs; targets are probability distributions → KLDivLoss
        self.criterion   = nn.KLDivLoss(reduction="batchmean")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Compute training loss, optionally applying mixup."""
        x, y = batch

        if self.mixup:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            idx = torch.randperm(x.size(0))
            x   = lam * x + (1 - lam) * x[idx]
            y   = lam * y + (1 - lam) * y[idx]

        logits = self(x) #logits already returned after log_softmax so as log-space distribution
        loss = self.criterion(logits, y)
        self.log_dict(
            {"train_loss": loss,
             },
            on_step= False, on_epoch= True, prog_bar= True)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        """Compute validation loss."""
        x, y  = batch
        loss  = self.criterion(self(x), y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Return AdamW, optionally paired with CosineAnnealingLR."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        if self.scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=self.t_max, eta_min=1e-6
                    ),
                    "interval": "epoch",
                },
            }

        return {"optimizer": optimizer}

    def on_train_epoch_end(self) -> None:
        """Print train loss and current lr at epoch end (if verbose)."""
        if self.verbose:
            train_loss = self.trainer.callback_metrics.get("train_loss", float("nan"))
            lr         = self.trainer.optimizers[0].param_groups[0]["lr"]
            print(f"Epoch {self.current_epoch:03d} - train_loss: {train_loss:.4f} - lr: {lr:.2e}")

    def on_validation_epoch_end(self) -> None:
        """Print validation loss at epoch end (if verbose)."""
        if self.verbose:
            val_loss = self.trainer.callback_metrics.get("val_loss", float("nan"))
            print(f"Epoch {self.current_epoch:03d} | val_loss: {val_loss:.4f}")
