from pathlib import Path
import random
from typing import Tuple, List

import pandas
from pandas import DataFrame


def generate_csv():
    data_root = "data/objects256"

    train_imgs, val_imgs, test_imgs = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    for class_dir in Path(f"{data_root}/images").iterdir():
        class_id, class_name = class_dir.name.split(".")
        class_id = int(class_id)

        all_imgs = [str(x) for x in class_dir.iterdir()]

        # random split
        random.shuffle(all_imgs)
        train_size = int(len(all_imgs) * 0.8)
        val_size = int(len(all_imgs) * 0.1)
        test_size = len(all_imgs) - train_size - val_size

        train_imgs.extend(all_imgs[:train_size])
        train_labels.extend([class_id] * train_size)

        val_imgs.extend(all_imgs[train_size:train_size + val_size])
        val_labels.extend([class_id] * val_size)

        test_imgs.extend(all_imgs[train_size + val_size:])
        test_labels.extend([class_id] * test_size)

    splits = ["train"] * len(train_imgs) + ["valid"] * len(val_imgs) + ["test"] * len(test_imgs)
    anno = DataFrame({
        "label": train_labels + val_labels + test_labels,
        "img": train_imgs + val_imgs + test_imgs,
        "split": splits
    })
    anno.sort_values("label", inplace=True)
    anno.to_csv(f"{data_root}/objects256.csv", index=False)


def load_img_labels() -> Tuple[
    Tuple[List[str], List[str], List[str]],
    Tuple[List[int], List[int], List[int]]
]:
    data_root = "data/objects256"

    anno = pandas.read_csv(f"{data_root}/objects256.csv")

    train_anno = anno.loc[anno["split"] == "train"]
    val_anno = anno.loc[anno["split"] == "valid"]
    test_anno = anno.loc[anno["split"] == "test"]

    train_imgs = train_anno["img"].values.tolist()
    val_imgs = val_anno["img"].values.tolist()
    test_imgs = test_anno["img"].values.tolist()

    train_labels = train_anno["label"].values.tolist()
    val_labels = val_anno["label"].values.tolist()
    test_labels = test_anno["label"].values.tolist()

    imgs = train_imgs, val_imgs, test_imgs
    labels = train_labels, val_labels, test_labels
    return imgs, labels


if __name__ == '__main__':
    generate_csv()
    load_img_labels()
