import copy
import numpy as np

from data.pairwise import PairwiseTester


def euclidean(x1, x2):
    return np.linalg.norm(x1 - x2)


def cosine(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


class TestRunner:
    def __init__(self, testers):
        self.testers = testers

    def __call__(self, model):
        full_out = {}
        for tag, tester in testers.items():
            out = tester(model)
            for k, v in out.items():
                full_out["{}:{}".format(tag, k)] = v
        return full_out


def get_tester(device, args):
    args = copy.deepcopy(args)

    testers = {}
    for test in args:
        ty = args.pop('name')
        tag = args.pop('tag')
        if ty == 'pairwise':
            testers[tag] = PairwiseTester(args['root'], args['pairs_file'],
                             device,
                             globals()[args['similarity']],
                             args.get('batch_size', 64))
        else:
            raise Exception(name + ' is not a testset')
    return TestRunner(testers)
