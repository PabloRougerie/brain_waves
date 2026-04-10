

import torch
from torch import nn
import numpy as np
from pytorch_lightning import LightningModule
from torchmetrics import KLDivergence



#=============
# FOR BASELINE
#=============

class BrainLightning(LightningModule):

    def __init__(self, model, n_classes= 6, lr=1e-3, mixup= False, mixup_alpha= 0.4):

        super().__init__()
        self.save_hyperparameters(ignore= ["model"])
        self.model = model
        self.mixup = mixup
        self.mixup_alpha = mixup_alpha
        #the model returns y_pred as log_proba, but y_true is proba. This is expected for the loss
        self.criterion = nn.KLDivLoss(reduction= "batchmean")


    def forward(self, x):
        return self.model(x)

    def training_step(self,batch, batch_idx):
        x, y = batch

        if self.mixup:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            idx = torch.randperm(x.size(0))
            x = lam * x + (1 - lam) * x[idx]
            y = lam * y + (1 - lam) * y[idx]


        logits = self(x) #logits already returned after log_softmax so as log-space distribution
        loss = self.criterion(logits, y)
        self.log_dict(
            {"train_loss": loss,
             },
            on_step= False, on_epoch= True, prog_bar= True)
        return loss

    def validation_step(self,batch, batch_idx):

        x, y = batch
        logits = self(x) #logits already returned after log_softmax so as log-space distribution
        loss = self.criterion(logits, y)
        self.log_dict(
            {"val_loss": loss},
            on_step= False, on_epoch= True, prog_bar= True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr = self.hparams.lr,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_train_epoch_end(self) -> None:

        train_loss = self.trainer.callback_metrics.get("train_loss", float("nan"))
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        print(f"Epoch {self.current_epoch:03d} - train_loss: {train_loss:.4f} - lr: {lr:.2e}")


    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get("val_loss", float("nan"))
        print(f"Epoch {self.current_epoch:03d} | val_loss:   {val_loss:.4f}")
