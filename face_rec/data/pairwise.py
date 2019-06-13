import os
import time
import random

import torch
from torch.nn import DataParallel
import torchvision
import torchvision.transforms as TF

import numpy as np

from PIL import Image


def read_pairs(pair_list):
    with open(pair_list, 'r') as fd:
        lines = fd.readlines()
    data = set()
    for line in lines:
        p1, p2, same = line.split()
        data.add(p1)
        data.add(p2)
    return list(data)


def load_image(img_path, lfw_crop=False):
    image = Image.open(img_path)
    if image is None:
        return None
    if lfw_crop:
        off = 25
        image = image.crop((92 - off, 83 - off, 175 + off, 166 + off))
    image = image.convert('RGB')
    return image


class PairwiseDataset(torch.utils.data.Dataset):
    def __init__(self, root, img_list, transforms=None, lfw_crop=False):
        self.lfw_crop = lfw_crop
        self.root = root
        self.transforms = transforms
        self.samples = read_pairs(img_list)

    def from_path(self, path):
        return load_image(self.root + '/' + path, lfw_crop=self.lfw_crop)

    def __getitem__(self, i):
        path = self.samples[i]
        img = self.from_path(path)

        if self.transforms is not None:
            img = self.transforms(img)
        return img, path

    def __len__(self):
        return len(self.samples)

