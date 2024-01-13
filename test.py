import torch
from PIL import Image
from torchvision.transforms import transforms

from common.image_dataset import ResizeKeepAspectRatio

img = Image.open("data/test/ABBOTTS BABBLER/1.jpg")

t = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    ResizeKeepAspectRatio((256, 192))
])
img = t(img)

print(img.shape)
