from common import birds525, objects256
from common.image_dataset import calculate_dataset_statistics

# imgs, _ = birds525.load_img_labels()
# mean, std = calculate_dataset_statistics(imgs[0] + imgs[1], (256, 192))
# print(f"birds525 --> mean: {mean}, std: {std}")

imgs, _ = objects256.load_img_labels()
mean, std = calculate_dataset_statistics(imgs[0] + imgs[1], (256, 192))
print(f"objects256 --> mean: {mean}, std: {std}")
