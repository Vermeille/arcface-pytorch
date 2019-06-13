import random
import time
import copy
import numpy as np

import torch
import torchvision.transforms as TF

from face_rec.data.pairwise import PairwiseDataset
from face_rec.utils.inspector import ComparisonInspector


class PairwiseTester:
    def __init__(self,
                 root,
                 pairs_file,
                 device,
                 sim_f,
                 batch_size=64,
                 viz=None,
                 lfw_crop=False):
        transforms = TF.Compose([
            TF.Resize(64),
            TF.ToTensor(),
            TF.Normalize(mean=[0.5503, 0.4352, 0.3844],
                         std=[0.2724, 0.2396, 0.2317])
        ])
        self.dataset = PairwiseDataset(root,
                                       pairs_file,
                                       transforms=transforms,
                                       lfw_crop=lfw_crop)
        self.device = device
        self.batch_size = batch_size
        self.sim_f = sim_f
        self.pairs = PairwiseTester.read_pairs(pairs_file)
        self.inspector = ComparisonInspector(32, self.dataset, self.pairs)
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
        return {'acc': acc, 'loss': loss, 'thresh': th}

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

            sims.append(sim)
            labels.append(label)

        sims = torch.FloatTensor(sims)
        labels = torch.FloatTensor(labels)

        if self.viz is not None:
            fs = torch.nn.functional.normalize(torch.FloatTensor(
                list(fe_dict.values())).to(self.device),
                                               dim=1)
            self.viz.hist(torch.mm(fs, fs.t()).view(-1),
                          name='lfw sim distribution')
            self.viz.hist(torch.FloatTensor(list(fe_dict.values())).view(-1),
                          name='lfw features distribution')

        loss = torch.nn.functional.binary_cross_entropy(
            torch.clamp(sims * 0.5 + 0.5, 0, 1), labels)
        acc, th = PairwiseTester.cal_accuracy(sims, labels)
        self.inspector.center_value = th
        self.inspector.analyze(sims, labels)
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
