import os
import sys
import random

import torch
import torchvision
from torchvision import transforms as T

from torchelie.datasets import HorizontalConcatDataset

from PIL import Image

from .imdbface import IMDBFace


def count_per_class(dataset):
    nclasses = len(dataset.classes)
    count = [0] * nclasses
    for item in dataset.imgs:
        count[item[-1]] += 1

    return torch.LongTensor(count)


class WithProb:
    def __init__(self, p, tfs):
        self.p = p
        self.tfs = tfs

    def __call__(self, x):
        if random.uniform(0, 1) < self.p:
            return self.tfs(x)
        return x

    def __repr__(self):
        return "WithProp(p={}, {})".format(self.p, repr(self.tfs))


class SubsampleResize:
    def __init__(self, out_size, p=0.05, max_ratio=4):
        self.max_ratio = max_ratio
        self.p = p
        self.out_size = out_size

    def __call__(self, x):
        down = WithProb(
            self.p,
            T.Resize(int(self.out_size // random.uniform(1, self.max_ratio))))
        up = T.Resize(self.out_size)
        return up(down(x))

    def __repr__(self):
        return "SubsampleResize(out_size={}, p={}, max_ratio={})".format(
            self.out_size, self.p, self.max_ratio)


def images(location, img_size):
    return torchvision.datasets.ImageFolder(location,
                                            transform=T.Compose([
                                                SubsampleResize(img_size + 5),
                                                T.RandomCrop(img_size),
                                                T.RandomHorizontalFlip(),
                                                WithProb(0.05, T.Grayscale(3)),
                                                T.ToTensor(),
                                                T.Normalize(mean=[0.5503, 0.4352, 0.3844], std=[0.2724, 0.2396, 0.2317])
                                            ]))


def imdb_face(root, img_size):
    return IMDBFace(root,
                    transforms=T.Compose([
                        T.RandomRotation(5, expand=False),
                        SubsampleResize(img_size + 5),
                        T.RandomCrop(img_size),
                        T.RandomHorizontalFlip(),
                        WithProb(0.05, T.Grayscale(3)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.5503, 0.4352, 0.3844], std=[0.2724, 0.2396, 0.2317])
                    ]))


def get_datasets(specs):
    if len(specs) == 1:
        return get_dataset(**specs[0])
    return HorizontalDatasetConcat([get_dataset(**spec) for spec in specs])


def get_dataset(name, location, size):
    if name == 'imdb':
        return imdb_face(location, size)
    if name == 'images':
        return images(location, size)
    assert False, name + " is not a valid dataset"
