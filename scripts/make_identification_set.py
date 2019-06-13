import copy
import random
import sys
from collections import defaultdict

from torchvision.datasets import ImageFolder

root = sys.argv[1]
if not root.endswith('/'):
    root += "/"
folder = ImageFolder(root)
samples = copy.deepcopy(folder.samples)

per_class = defaultdict(list)
for path, klass in samples:
    path = path[len(root):]
    per_class[klass].append(path)

ref = []
for c in per_class:
    if len(per_class[c]) < 10:
        continue
    c_ref = []
    for i in range(5):
        idx = random.randrange(len(per_class[c]))
        c_samples = per_class[c]
        c_ref.append(c_samples[idx])
        del c_samples[idx]
    ref += c_ref

test = []
for kl, paths in per_class.items():
    for path in paths:
        test.append(path)

with open(sys.argv[2], 'w') as f:
    f.write('RES\n')
    for r in ref:
        f.write(r + '\n')
    f.write('TEST\n')
    for t in test:
        f.write(t + '\n')
