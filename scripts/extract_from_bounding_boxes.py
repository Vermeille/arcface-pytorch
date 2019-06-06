from collections import defaultdict
import os
from multiprocessing import Pool

from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as TF

from data.boundingboxed import BoundingBoxedDataset

root = '/hdd/data/resources/ph/'
save_dir = 'prepro64'
original = BoundingBoxedDataset(root, TF.Resize(64))

def save(dat):
    i, x = dat
    img, psnum = x
    try:
        os.mkdir(save_dir + '/' + names[psnum])
    except:
        pass
    img.save(save_dir + '/' + names[psnum] + '/' + str(i) + '.png')
    return img


try:
    os.mkdir(save_dir)
except:
    pass


if __name__ == '__main__':
    names = {i: nm for nm, i in original.class_to_idx.items()}

    p = Pool(64)
    print('Go')
    for _ in p.imap(save, enumerate(tqdm(original)), 256):
        pass
