import sys

import torch

from face_rec.models.loader import get_model

def load_model(filename):
    return get_model(torch.load(filename, map_location='cpu')['model'])


def to_onnx(model, filename):
    torch.onnx.export(model, torch.randn(1, 3, 64, 64), filename, verbose=True)

model = load_model(sys.argv[1])
to_onnx(model, sys.argv[2])
