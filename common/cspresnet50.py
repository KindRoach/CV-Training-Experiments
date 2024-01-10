from dataclasses import dataclass

import pytorch_lightning as pl
import timm
import torch
from torch.nn.functional import cross_entropy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional import accuracy


@dataclass
class CspResNet50Args:
    n_class: int
    num_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float


class CspResNet50(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        # save arguments as hyperparameters
        self.args = CspResNet50Args(**kwargs)
        self.save_hyperparameters()

        # load resnet-50 model from torchvision
        self.model = timm.create_model("cspresnet50", num_classes=self.args.n_class)

    def forward_loss(self, batch):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = cross_entropy(outputs, labels)
        acc = accuracy(outputs, labels, task="multiclass", num_classes=self.args.n_class)
        return acc, loss

    def training_step(self, batch, batch_idx):
        acc, loss = self.forward_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        acc, loss = self.forward_loss(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        acc, loss = self.forward_loss(batch)
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", acc, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )

        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=3)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_acc",
                "reduce_on_plateau": True
            }
        }
