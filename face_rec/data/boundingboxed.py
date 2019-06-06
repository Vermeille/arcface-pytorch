from torch.utils.data.dataset import Dataset
from PIL import Image
import glob
import json

class BoundingBoxedDataset(Dataset):
    def __init__(self, dir, tfms=None):
        self.faces = []
        self.classes = set()
        self.tfms = tfms
        for f in glob.glob(dir + '/*/faces.js'):
            ps = f.split('/')[-2]
            self.classes.add(ps)
            try:
                faces = json.load(open(f))
            except Exception as e:
                print('ERROR reading', f, ':', str(e))
                continue
            for photo in faces:
                fn = photo['file']
                for box in photo['faces']:
                    self.faces.append((ps, dir + '/' + ps + '/' + fn, box))
        self.class_to_idx = {p: i for i, p in enumerate(self.classes)}


    def __len__(self):
        return len(self.faces)

    def __getitem__(self, i):
        ps, fn, box = self.faces[i]
        img = Image.open(fn)
        img = img.convert('RGB')
        bh, bw = box[2] - box[0], box[1] - box[3]
        cx = (box[1] + box[3]) // 2
        cy = (box[0] + box[2]) // 2
        sz = max(bw, bh)
        crop = (cx - sz // 2, cy - sz // 2, cx + sz // 2, cy + sz // 2)
        img = img.crop(crop)
        if self.tfms is not None:
            img = self.tfms(img)
        return img, self.class_to_idx[ps]

