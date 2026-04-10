

import torch
from torch import nn
from pytorch_lightning import LightningModule
from torchmetrics import KLDivergence



#=============
# FOR BASELINE
#=============

class BrainLightning(LightningModule):

    def __init__(self, model, n_classes= 6, lr=1e-3):

        super().__init__()
        self.save_hyperparameters(ignore= ["model"])
        self.model = model
        #the model returns y_pred as log_proba, but y_true is proba. This is expected for the loss
        self.criterion = nn.KLDivLoss(reduction= "batchmean")
        self.train_kl = KLDivergence(reduction= "mean")
        self.val_kl = KLDivergence(reduction= "mean")


    def forward(self, x):
        return self.model(x)

    def training_step(self,batch, batch_idx):
        x, y = batch
        logits = self(x) #logits already returned after log_softmax so as log-space distribution
        loss = self.criterion(logits, y)
        self.train_kl.update(y, torch.exp(logits)) #check if correct for log
        self.log_dict(
            {"train_loss": loss,
             "train_kl": self.train_kl},
            on_step= False, on_epoch= True, prog_bar= True)
        return loss

    def validation_step(self,batch, batch_idx):

        x, y = batch
        logits = self(x) #logits already returned after log_softmax so as log-space distribution
        loss = self.criterion(logits, y)
        self.val_kl.update(y, torch.exp(logits)) #check if correct for log
        self.log_dict(
            {"val_loss": loss,
             "val_kl": self.val_kl},
            on_step= False, on_epoch= True, prog_bar= True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr = self.hparams.lr,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer= optimizer, mode= "min", factor= 0.5, patience= 3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_train_epoch_end(self) -> None:

        train_loss = self.trainer.callback_metrics.get("train_loss", float("nan"))
        train_kl = self.trainer.callback_metrics.get("train_kl", float("nan"))
        print(f"Epoch {self.current_epoch:03d} - train_loss: {train_loss:.4f} - train_kl: {train_kl:.4f}")


    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get("val_loss", float("nan"))
        val_kl = self.trainer.callback_metrics.get("val_kl", float("nan"))
        print(f"Epoch {self.current_epoch:03d} | val_loss:   {val_loss:.4f} | val_kl:   {val_kl:.4f}")
