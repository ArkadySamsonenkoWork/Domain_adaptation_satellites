import torch, torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import rasterio
import numpy as np


classes = ["open water", "settlements", "bare soil", "forest", "grassland"]
PALLETE = [
        [0, 204, 242],
        [230, 0, 77],
        [204, 204, 204],
        [100, 180, 50],
        [180, 230, 77],
        [255, 230, 166],
        [150, 77, 255]
        ]

def de_transform(image, mean, std):
    """
    DeNormalizing images
    """
    image = image * std[:, None, None] + mean[:, None, None]
    return image

def normalize(band):
    band_min, band_max = (band.min(), band.max())
    return ((band - band_min) / ((band_max - band_min)))

def brighten(band):
    alpha = 0.13
    beta = 0
    return np.clip(alpha * band + beta, 0, 255)

def convert(im_path):
    with rasterio.open(im_path) as fin:
        red = fin.read(3)
        green = fin.read(2)
        blue = fin.read(1)

    red_b = brighten(red)
    blue_b = brighten(blue)
    green_b = brighten(green)

    red_bn = normalize(red_b)
    green_bn = normalize(green_b)
    blue_bn = normalize(blue_b)

    return np.dstack((blue_b, green_b, red_b)), np.dstack((red_bn, green_bn, blue_bn))