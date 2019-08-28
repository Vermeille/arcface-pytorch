import random
import time
import copy
import numpy as np

import torch
import torchvision.transforms as TF

from face_rec.data.pairwise import IdentificationDataset
from face_rec.testers.pairwise import PairwiseTester

def euclidean(x1, x2):
    return np.linalg.norm(x1 - x2)


def cosine(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


class IdentificationTester:
    def __init__(self, root, id_files, device, sim_f, batch_size, viz=None):
        transforms = TF.Compose([
            TF.Resize(64),
            TF.ToTensor(),
            TF.Normalize(mean=[0.5503, 0.4352, 0.3844],
                         std=[0.2724, 0.2396, 0.2317])
        ])
        seld.dataset = IdentificationDataset(root, id_files,
                transforms=transforms)
        self.sets = IndentificationTester.read_set(id_files)

        self.test_loader = torch.utils.data.DataLoader(
            self.dataset,
            pin_memory=True,
            batch_size=self.batch_size,
            num_workers=self.batch_size // 8)

    @staticmethod
    def read_set(fname):
        refs = []
        test = []
        with open(fname, 'r') as f:
            lines = iter(f)
            next(lines) # skip 'REF'
            l = next(lines)
            while l != 'TEST':
                refs.append(l)
                l = next(lines)
            l = next(l) # skip 'TEST'
            while l is not None:
                test.append(l)
        return refs, test


    def __call__(self, model):
        fe_dict = {}
        with torch.no_grad():
            for imgs, paths in self.test_loader:
                output = model(imgs)
                for fe, path in zip(output, paths):
                    fe_dict[path] = fe
        return fe_dict


class TestRunner:
    def __init__(self, testers):
        self.testers = testers

    def __call__(self, model):
        full_out = {}
        for tag, tester in self.testers.items():
            out = tester(model)
            for k, v in out.items():
                full_out["{}:{}".format(tag, k)] = v
        return full_out

    def show(self, viz):
        for tag, tester in self.testers.items():
            tester.inspector.show(viz, tag)


def get_tester(device, args):
    args = copy.deepcopy(args)

    testers = {}
    for test in args:
        ty = test.pop('type')
        tag = test.pop('name')
        if ty == 'pairwise':
            testers[tag] = PairwiseTester(test['root'], test['pairs_file'],
                                          device,
                                          globals()[test['similarity']],
                                          test.get('batch_size', 64),
                                          lfw_crop=tag=='lfw')
        else:
            raise Exception(name + ' is not a testset')
    return TestRunner(testers)
