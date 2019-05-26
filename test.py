import numpy as np

from data.lfw import LFWTester


def euclidean(x1, x2):
    return np.linalg.norm(x1 - x2)


def cosine(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def get_tester(name, device, args):
    if name == 'lfw':
        return LFWTester(args['root'], args['pairs_file'], device,
                         globals()[args['similarity']],
                         args.get('batch_size', 64))
    raise Exception(name + ' is not a testset')
