import os
from typing import List, Tuple

import pytorch_lightning as pl
import torch
from PIL import Image
from kornia.color import rgb_to_bgr
from kornia.geometry import warp_affine
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, transforms
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, images_path: List[str], labels: List[int], transform):
        self.images_path = images_path
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        image = Image.open(self.images_path[idx])
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


class ImageDataModule(pl.LightningDataModule):
    def __init__(
            self,
            images_path: Tuple[List[str], List[str], List[str]],
            labels: Tuple[List[int], List[int], List[int]],
            transform: Compose, batch_size: int
    ):
        """
        :param images_path: train, validate, and test image paths
        :param labels: train, validate, and test labels
        """
        super().__init__()
        self.images_path = images_path
        self.labels = labels
        self.batch_size = batch_size
        self.transform = transform

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str):
        self.train_dataset = ImageDataset(self.images_path[0], self.labels[0], self.transform)
        self.val_dataset = ImageDataset(self.images_path[1], self.labels[1], self.transform)
        self.test_dataset = ImageDataset(self.images_path[2], self.labels[2], self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, self.batch_size, True,
            num_workers=os.cpu_count(),
            persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, self.batch_size,
            num_workers=os.cpu_count(),
            persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, self.batch_size,
            num_workers=os.cpu_count(),
            persistent_workers=True)


def calculate_dataset_statistics(imgs: List[str], input_shape: (int, int)) -> (list, list):
    """
    Reference: https://kozodoi.me/blog/20210308/compute-image-stats
    :return: Mean and std of images
    """
    transform = transforms.Compose([
        transforms.Resize(input_shape),
        transforms.ToTensor(),
    ])

    p_sum = torch.zeros(3, dtype=torch.float64)
    p_sum_sq = torch.zeros(3, dtype=torch.float64)
    for img in tqdm(imgs, desc="Calculating mean and std"):
        img = Image.open(img)
        img = transform(img)
        p_sum += img.sum(dim=(1, 2))
        p_sum_sq += (img ** 2).sum(dim=(1, 2))

    count = len(imgs) * input_shape[0] * input_shape[1]
    total_mean = p_sum / count
    total_var = (p_sum_sq / count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)

    return total_mean.tolist(), total_std.tolist()


class RGB2BGR(transforms.Lambda):
    def __init__(self):
        super().__init__(lambda img: rgb_to_bgr(img))


class ResizeKeepAspectRatio(transforms.Lambda):
    def __init__(self, target_size: (int, int)):
        def resize_keep_aspect_ratio(img):
            sh = img.shape[-2]
            sw = img.shape[-1]
            dh = target_size[0]
            dw = target_size[1]

            scale = min(dw / sw, dh / sh)

            scx = sw * 0.5
            scy = sh * 0.5

            dcx = dw * 0.5
            dcy = dh * 0.5

            x_offset = dcx - scx * scale
            y_offset = dcy - scy * scale

            mat = torch.tensor([[
                [scale, 0, x_offset],
                [0, scale, y_offset],
            ]])

            return warp_affine(img.unsqueeze(0), mat, (dh, dw)).squeeze(0)

        super().__init__(resize_keep_aspect_ratio)
