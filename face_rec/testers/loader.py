import random
import time
import copy
import numpy as np

import torch
import torchvision.transforms as TF


def euclidean(x1, x2):
    return np.linalg.norm(x1 - x2)


def cosine(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


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
