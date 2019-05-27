import copy
import argparse
import sys
import ast
import types
from pathlib import Path

import torch
import yaml


def parse(filename):
    with open(filename, 'r') as f:
        dat = yaml.load(f)
    return dat


def override_with_path(d, k, v):
    def _override_with_path(d, k, v):
        if len(k) == 1:
            d[k[0]] = v
        else:
            _override_with_path(d[k[0]], k[1:], v)

    _override_with_path(d, k.split('.'), v)


def override_dict(d1, d2):
    for k, v in d2.items():
        if k not in d1:
            d1[k] = v
            continue

        if isinstance(d1[k], dict):
            override_dict(d1[k], v)
        else:
            d1[k] = v


def load_default():
    basedir = Path(__file__).resolve().parent.parent
    default = basedir / 'configs/default.yml'
    return parse(str(default))


def from_cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('-X', action='append', default=[])
    cmdargs = parser.parse_args()

    opts = load_default()
    if cmdargs.config.endswith('.yaml') or cmdargs.config.endswith('.yml'):
        yaml_opt = parse(cmdargs.config)
        override_dict(opts, yaml_opt)
        print('Loaded config from yaml')
    else:
        opts = torch.load(cmdargs.config)
    for arg in cmdargs.X:
        k, v = arg.split('=')
        try:
            ast.literal_eval(v)
        except:
            pass
        override_with_path(opts, k, v)
    return opts


class Checkpointer:
    def __init__(self, model, metric, optimizer, session_name, options):
        self.model = model
        self.metric = metric
        self.optimizer = optimizer
        self.name = session_name
        self.options = options

    def save(self, iter_n, **more_stuff_to_save):
        new_dict = copy.deepcopy(self.options)
        new_dict['model']['state_dict'] = self.model.state_dict()
        new_dict['optimizer']['state_dict'] = self.optimizer.state_dict()
        new_dict['metric']['state_dict'] = self.metric.state_dict()

        for k, v in more_stuff_to_save:
            override_with_path(new_dict, k, v)
        new_dict['trainer']['iter_n'] = iter_n
        torch.save(new_dict, "ckpt/{}_{}.pth".format(self.name, self.name,
                                                   iter_n))
