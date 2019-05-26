import torch


class CatedSamples:
    def __init__(self, samples):
        self.samples = samples
        self.n_classes = [
            len(set(samp[1] for samp in sample)) for sample in samples
        ]

    def __len__(self):
        return sum(len(ds) for ds in self.samples)

    def __getitem__(self, i):
        class_offset = 0
        for samp, n_class in zip(self.samples, self.n_classes):
            if i < len(samp):
                return samp[i][0], samp[i][1] + class_offset
            i -= len(samp)
            class_offset += n_class
        raise IndexError


class CatedLists:
    def __init__(self, ls):
        self.ls = ls

    def __len__(self):
        return sum(len(ds) for ds in self.ls)

    def __getitem__(self, i):
        for l in self.ls:
            if i < len(l):
                return l[i]
            i -= len(l)
        raise IndexError


class DatasetConcat(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

        self.classes = CatedLists([ds.classes for ds in datasets])
        self.imgs = CatedSamples([ds.imgs for ds in datasets])
        self.class_to_idx = {i: nm for i, nm in enumerate(self.classes)}

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        class_offset = 0
        for ds in self.datasets:
            if i < len(ds):
                x, t = ds[i]
                return x, t + class_offset
            i -= len(ds)
            class_offset += len(ds.classes)
        raise IndexError
