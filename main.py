import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchvision.transforms import transforms

from common import birds_525
from common.cspresnet50 import CspResNet50Args, CspResNet50
from common.image_dataset import ImageDataModule, RGB2BGR, ResizeKeepAspectRatio

torch.set_float32_matmul_precision('medium')


def main():
    imgs, labels = birds_525.load_img_labels()
    n_class = max(max(x) for x in labels) + 1

    args = CspResNet50Args(
        n_class,
        num_epochs=100,
        batch_size=256,
        learning_rate=1e-3,
        weight_decay=1e-5
    )

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

    input_size = (256, 192)

    rgb_normalize_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.474, 0.469, 0.395], [0.236, 0.230, 0.252])
    ])

    bgr_normalize_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.474, 0.469, 0.395], [0.236, 0.230, 0.252]),
        RGB2BGR()
    ])

    bgr_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.PILToTensor(),
        RGB2BGR(),
        transforms.ConvertImageDtype(torch.float)
    ])

    bgr_aa_transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        ResizeKeepAspectRatio(input_size),
        RGB2BGR()
    ])

    model = CspResNet50(**args.__dict__)
    data_module = ImageDataModule(imgs, labels, bgr_aa_transform, args.batch_size)

    trainer.fit(model, data_module)
    trainer.test(model, data_module, verbose=True)
    print(f"best model saved as: {checkpoint_callback.best_model_path}")


if __name__ == '__main__':
    main()
