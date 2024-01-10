import pandas

from common.image_dataset import calculate_dataset_statistics

anno = pandas.read_csv("data/birds.csv")
anno["filepaths"] = "data/" + anno["filepaths"]
mean, std = calculate_dataset_statistics(anno["filepaths"], (256, 192))
print(f"mean: {mean}, std: {std}")
