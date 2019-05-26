import os

from torch.utils import data
import numpy as np
from PIL import Image


class IMDBFace(data.Dataset):
    def __init__(self, root, transforms=None):
        self.transforms = transforms
        imgs = []
        self.class_to_idx = {}
        self.classes = []
        self.bbs = []
        with open(root + '/bbs2.txt', 'r') as fd:
            for l in fd.readlines():
                p, x, y, x2, y2 = l.split(' ')
                imgs.append([os.path.join(root, p), self.get_label(p)])
                self.bbs.append([x, y, x2, y2])

        self.imgs = imgs

    def get_label(self, p):
        p = p.split('/')[1]
        try:
            idx = self.class_to_idx[p]
        except:
            self.classes.append(p)
            idx = len(self.class_to_idx)
            self.class_to_idx[p] = idx
        return idx

    def refine_bb(self, bb, sz):
        bb = list(map(int, bb))
        w = bb[2] - bb[0]
        h = bb[3] - bb[1]
        bb[0] = max(0, bb[0] - 0.2 * w)
        bb[2] = min(sz[0], bb[2] + 0.2 * w)
        bb[1] = max(0, bb[1] - 0.1 * h)
        bb[3] = min(sz[1], bb[3] + 0.2 * h)
        return tuple(map(int, bb))

    def read_throw(self, index):
        splits = self.imgs[index]
        bb = self.bbs[index]

        img_path = splits[0]
        data = Image.open(img_path)
        data.load()
        data = data.convert('RGB')
        data = data.crop(self.refine_bb(bb, data.size))

        if self.transforms is not None:
            data = self.transforms(data)

        label = np.int32(splits[-1])
        return data, label

    def __getitem__(self, index):
        try:
            data, label = self.read_throw(index)
            return data, label
        except Exception as e:
            print(str(e))
            return self.__getitem__(index + 1)

    def __len__(self):
        return len(self.imgs)
