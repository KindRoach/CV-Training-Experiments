import pandas
import pytorch_lightning as pl
import torch
from pandas import DataFrame
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchvision.transforms import Compose, transforms

from common.cspresnet50 import CspResNet50Args, CspResNet50
from common.image_dataset import ImageDataModule

torch.set_float32_matmul_precision('medium')


def load_datamodule(anno: DataFrame, transform: Compose, batch_size: int) -> ImageDataModule:
    train_anno = anno.loc[anno["data set"] == "train"]
    val_anno = anno.loc[anno["data set"] == "valid"]
    test_anno = anno.loc[anno["data set"] == "test"]

    train_imgs = train_anno["filepaths"].values.tolist()
    val_imgs = val_anno["filepaths"].values.tolist()
    test_imgs = test_anno["filepaths"].values.tolist()

    train_labels = train_anno["class id"].values.tolist()
    val_labels = val_anno["class id"].values.tolist()
    test_labels = test_anno["class id"].values.tolist()

    return ImageDataModule(
        (train_imgs, val_imgs, test_imgs),
        (train_labels, val_labels, test_labels),
        transform,
        batch_size
    )


def main():
    anno = pandas.read_csv("data/birds.csv")
    anno["filepaths"] = "data/" + anno["filepaths"]

    n_class = int(anno["class id"].max()) + 1
    args = CspResNet50Args(
        n_class,
        num_epochs=100,
        batch_size=256,
        learning_rate=1e-3,
        weight_decay=1e-5
    )

    transform = transforms.Compose([
        transforms.Resize((256, 192)),
        transforms.ToTensor(),
        transforms.Normalize([0.474, 0.469, 0.395], [0.236, 0.230, 0.252])
    ])

    data_module = load_datamodule(anno, transform, args.batch_size)
    model = CspResNet50(**args.__dict__)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=10, monitor="val_acc", mode="max",
        auto_insert_metric_name=False,
        filename="ep={epoch}-acc={val_acc:.3f}"
    )

    early_stop_callback = EarlyStopping(
        monitor="val_acc", mode="max",
        min_delta=0.00, patience=5,
        verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        default_root_dir="output",
        precision="16-mixed",
        callbacks=[checkpoint_callback, early_stop_callback])

    trainer.fit(model, data_module)
    trainer.test(model, data_module, verbose=True)
    print(f"best model saved as: {checkpoint_callback.best_model_path}")


if __name__ == '__main__':
    main()
