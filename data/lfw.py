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


def load_image(img_path):
    image = Image.open(img_path)
    if image is None:
        return None
    off = 25
    image = image.crop((92 - off, 83 - off, 175 + off, 166 + off))
    image = image.convert('RGB')
    return image


class LFWUniqueTestSet(torch.utils.data.Dataset):
    def __init__(self, root, img_list, transforms=None):
        self.root = root
        self.transforms = transforms
        self.samples = read_pairs(img_list)

    def __getitem__(self, i):
        path = self.samples[i]
        img = load_image(self.root + '/' + path)

        if self.transforms is not None:
            img = self.transforms(img)
        return img, path

    def __len__(self):
        return len(self.samples)


class LFWTester:
    def __init__(self, root, pairs_file, device, sim_f, batch_size=64, viz=None):
        transforms = TF.Compose([
            TF.ToTensor(),
            TF.Normalize(mean=[0.5503, 0.4352, 0.3844], std=[0.2724, 0.2396, 0.2317])
            #TF.Normalize(mean=[0.4] * 3, std=[0.2] * 3)
        ])
        self.dataset = LFWUniqueTestSet(root, pairs_file, transforms)
        self.device = device
        self.batch_size = batch_size
        self.sim_f = sim_f
        self.pairs = LFWTester.read_pairs(pairs_file)
        self.viz = viz

        self.test_loader = torch.utils.data.DataLoader(
            self.dataset,
            pin_memory=True,
            batch_size=self.batch_size,
            num_workers=self.batch_size // 8)

    @staticmethod
    def read_pairs(pairs_file):
        with open(pairs_file, 'r') as fd:
            pairs = fd.readlines()
        return [pair.split() for pair in pairs]

    def __call__(self, model):
        model.eval()
        s = time.time()
        feats, cnt = self.get_features(model)
        t = time.time() - s
        print('total time is {}, average time is {}'.format(t, t / cnt))
        acc, th, loss = self.test_performance(feats)
        print('lfw face verification accuracy: ', acc, 'threshold: ', th)
        model.train()
        return acc, loss

    @staticmethod
    def cal_accuracy(y_score, y_true):
        y_score = np.asarray(y_score)
        y_true = np.asarray(y_true)
        best_acc = 0
        best_th = 0
        for i in range(len(y_score)):
            th = y_score[i]
            y_test = (y_score >= th)
            acc = np.mean((y_test == y_true).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_th = th

        return (best_acc, best_th)

    def test_performance(self, fe_dict):
        sims = []
        labels = []
        random.shuffle(self.pairs)
        for person1, person2, same in self.pairs:
            fe_1 = fe_dict[person1]
            fe_2 = fe_dict[person2]
            label = int(same)
            sim = self.sim_f(fe_1, fe_2)
            print(person1, person2, sim, same)

            sims.append(sim)
            labels.append(label)

        sims = torch.FloatTensor(sims)

        if self.viz is not None:
            self.viz.hist(sims, name='lfw sim distribution')

        loss = torch.nn.functional.binary_cross_entropy(
            torch.clamp(sims * 0.5 + 0.5, 0, 1),
            torch.FloatTensor(labels))
        acc, th = LFWTester.cal_accuracy(sims, labels)
        return acc, th, loss

    def get_features(self, model):
        features = {}
        cnt = 0
        with torch.no_grad():
            for imgs, paths in self.test_loader:
                data = imgs.to(self.device)
                output = model(data)
                output = output.data.cpu().numpy()

                for feat, path in zip(output, paths):
                    features[path] = feat
            cnt += 1

        return features, cnt
