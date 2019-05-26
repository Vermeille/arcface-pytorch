import os
import sys
import random

import torch
import torchvision
from torchvision import transforms as T

from PIL import Image

from .concat import DatasetConcat
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


def images(location, img_size):
    resize = T.Resize(img_size) if img_size is not None else (lambda x: x)
    return torchvision.datasets.ImageFolder(location,
                                            transform=T.Compose([
                                                resize,
                                                T.RandomHorizontalFlip(),
                                                WithProb(0.05, T.Grayscale(3)),
                                                T.ToTensor(),
                                                T.Normalize(
                                                    mean=[0.5, 0.5, 0.5],
                                                    std=[0.5, 0.5, 0.5])
                                            ]))


def imdb_face(root, img_size):
    return IMDBFace(root,
                    transforms=T.Compose([
                        T.RandomRotation(5, expand=False),
                        T.Resize(img_size + 5),
                        T.RandomCrop(img_size),
                        T.RandomHorizontalFlip(),
                        WithProb(0.05, T.Grayscale(3)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ]))


def get_datasets(specs):
    if len(specs) == 1:
        return get_dataset(**specs[0])
    return DatasetConcat([get_dataset(**spec) for spec in specs])


def get_dataset(name, location, size):
    if name == 'imdb':
        return imdb_face(location, size)
    if name == 'images':
        return images(location, size)
    assert False, name + " is not a valid dataset"


