import os
import math
import random
import sys
import shutil

from collections import defaultdict

from torchvision.datasets import ImageFolder

to_gen = 3000
root = sys.argv[1]
if not root.endswith('/'):
    root += "/"
folder = ImageFolder(root)
samples = folder.samples

per_class = defaultdict(list)
for path, klass in samples:
    path = path[len(root):]
    per_class[klass].append(path)

numel = sorted([(k, len(v)) for k, v in per_class.items()], key=lambda x: x[1])

small = []
s = 0
for k, tot in numel:
    s += tot
    small.append(k)
    if s > 100 * math.sqrt(to_gen):
        break

print('choosing from:', [folder.classes[s] for s in small])
classes = small

pairs = set()
used = set()
for i in range(to_gen):
    klass = random.choice(classes)
    line = (random.choice(per_class[klass]), random.choice(per_class[klass]), '1')
    line_rev = (line[1], line[0], line[2])
    while line in pairs or line_rev in pairs or line[1] == line[0]:
        klass = random.choice(classes)
        line = (random.choice(per_class[klass]),
                random.choice(per_class[klass]), '1')
        line_rev = (line[1], line[0], line[2])
    used.add(klass)
    pairs.add(line)

diff = set()
for i in range(to_gen):
    klass1 = random.choice(classes)
    klass2 = random.choice(classes)
    line = (random.choice(per_class[klass1]), random.choice(per_class[klass2]), '0')
    line_rev = (line[1], line[0], line[2])
    while line in diff or line_rev in diff or klass1 == klass2:
        klass1 = random.choice(classes)
        klass2 = random.choice(classes)
        line = (random.choice(per_class[klass1]), random.choice(per_class[klass2]), '0')
        line_rev = (line[1], line[0], line[2])
    used.add(klass1)
    used.add(klass2)
    diff.add(line)


with open(sys.argv[2], 'w') as f:
    for p in pairs:
        f.write(" ".join(p) + "\n")
    for d in diff:
        f.write(" ".join(d) + "\n")

os.makedirs(root + '/../holdout')
for kid in used:
    dst = root + '/../holdout/' + folder.classes[kid]
    shutil.move(root + '/' + folder.classes[kid], dst)
