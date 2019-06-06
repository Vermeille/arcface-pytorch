import copy

import torch

from .resnet import *
from .focal_loss import FocalLoss
import face_rec.models.metrics as metrics


def get_model(args):
    args = copy.deepcopy(args)
    name = args.pop('name')
    state_dict = args.pop('state_dict', None)
    model = globals()[name](**args)
    if state_dict:
        model.load_state_dict(state_dict)
        print('Loaded model from checkpoint')
    else:
        print('Created untrained model')
    print(model)
    return model


def get_loss(name, inv_count, args):
    if name == 'focal':
        return FocalLoss(
            gamma=args.get('gamma', 4),
            weight=inv_count if args.get('balance', False) else None)
    if name == 'cross_entropy':
        return torch.nn.CrossEntropyLoss(
            weight=inv_count if args.get('balance', False) else None)
    assert False, name + ' is not a loss'


def get_optimizer(params, args):
    args = copy.deepcopy(args)
    name = args.pop('name')
    state_dict = args.pop('state_dict', None)
    optim_klass = torch.optim.__dict__[name]
    optim = optim_klass(params, **args)
    if state_dict:
        optim.load_state_dict(state_dict)
        print('Loaded optimizer from checkpoint')
    else:
        print('Created new optimizer')
    print(optim)
    return optim


def get_metric(num_classes, args):
    args = copy.deepcopy(args)
    dim = args.pop('dim', 512)
    name = args.pop('name')
    state_dict = args.pop('state_dict', None)
    m = metrics.__dict__[name](dim, num_classes, **args)
    if state_dict:
        m.load_state_dict(state_dict)
        print('Loaded metric from checkpoint')
    else:
        print('Created new metric')
    print(m)
    return m
