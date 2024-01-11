import birds_525
from common.image_dataset import calculate_dataset_statistics

imgs, _ = birds_525.load_img_labels()
mean, std = calculate_dataset_statistics(imgs, (256, 192))
print(f"mean: {mean}, std: {std}")
