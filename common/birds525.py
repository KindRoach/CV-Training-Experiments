from typing import Tuple, List

import pandas


def load_img_labels() -> Tuple[
    Tuple[List[str], List[str], List[str]],
    Tuple[List[int], List[int], List[int]]
]:
    data_root = "data/birds525"

    anno = pandas.read_csv(f"{data_root}/birds.csv")
    anno["filepaths"] = f"{data_root}/" + anno["filepaths"]

    train_anno = anno.loc[anno["data set"] == "train"]
    val_anno = anno.loc[anno["data set"] == "valid"]
    test_anno = anno.loc[anno["data set"] == "test"]

    train_imgs = train_anno["filepaths"].values.tolist()
    val_imgs = val_anno["filepaths"].values.tolist()
    test_imgs = test_anno["filepaths"].values.tolist()

    train_labels = train_anno["class id"].astype(int).values.tolist()
    val_labels = val_anno["class id"].astype(int).values.tolist()
    test_labels = test_anno["class id"].astype(int).values.tolist()

    imgs = train_imgs, val_imgs, test_imgs
    labels = train_labels, val_labels, test_labels
    return imgs, labels
