import argparse
import os
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import torchvision.transforms as T
import glob
from scipy.misc import imread, imresize, imsave
# from skimage.transform import resize as imresize
# from skimage.io import imread
import math
import sys
import pathlib
from tqdm import tqdm


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# IMAGENET_MEAN = [0.5, 0.5, 0.5]
# IMAGENET_STD = [0.5, 0.5, 0.5]

INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]


def imagenet_deprocess(rescale_image=True):
    transforms = [
        T.Normalize(mean=[0, 0, 0], std=INV_IMAGENET_STD),
        T.Normalize(mean=INV_IMAGENET_MEAN, std=[1.0, 1.0, 1.0]),
    ]
    if rescale_image:
        transforms.append(rescale)
    return T.Compose(transforms)

image_path = "./datasets/coco/val2017"
output_path = "./samples/tmp/coco/128/val"

if not os.path.exists(output_path):
    os.makedirs(output_path)

if not os.path.exists(image_path):
    print("image path does NOT exisit")

path = pathlib.Path(image_path)
files = list(path.glob('*.jpg')) + list(path.glob('*.png'))

for f in tqdm(files):
    img = imread(f)
    img = imresize(img, (128, 128))
    if img.shape[-1] != 3:
        print(f)
    img_name = str(f).split('/')[-1]
    output_file = os.path.join(output_path, img_name)
    imsave(output_file, img)
