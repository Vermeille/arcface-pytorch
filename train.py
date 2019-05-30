import os
import itertools
from datetime import datetime
import random
import time
import sys

import torch
import torch.nn.functional as F
import torchvision
from torch.utils import data
from torch.nn import DataParallel

import numpy as np

import data.dataset as dataset
import configurator
from test import *
from utils import Visualizer, view_model
from models import *
from test import get_tester
import models.loader as loader
import training.scheduling as scheduling

def ortho(w, strength=1e-4):
    return strength * (torch.mm(w, w.t()) * (1 - torch.eye(w.shape[0],
        device=w.device))).pow(2).mean()

class Inspector:
    def __init__(self, topk, labels, center_value=0):
        self.labels = labels
        self.center_value = center_value
        self.topk = topk
        self.reset()

    def reset(self):
        self.best = []
        self.worst = []
        self.confused = []

    def analyze(self, batch, pred, true, pred_label=None):
        for_label = pred[range(batch.shape[0]), true]
        if pred_label is None:
            pred_label = pred.argmax(dim=1)
        this_data = list(zip(batch, for_label, true, pred_label == true))

        self.best += this_data
        self.best.sort(key=lambda x: -x[1])
        self.best = self.best[:self.topk]

        self.worst += this_data
        self.worst.sort(key=lambda x: x[1])
        self.worst = self.worst[:self.topk]

        self.confused += this_data
        self.confused.sort(key=lambda x: abs(self.center_value - x[1]))
        self.confused = self.confused[:self.topk]

    def _report(self, dat):
        def cos_as_bar(cos):
            return '<div style="width:{}%;background-color:{};height:5px"></div>'.format(
                abs(cos) * 100, "green" if cos >= 0 else "red")

        html = ['<div style="display:flex;flex-wrap:wrap">']
        for img, p, cls, correct in dat:
            img -= img.min()
            img /= img.max()
            html.append(
                    '<div style="padding:3px;width:{}px">{}{}{}{}</div>'.format(
                    dat[0][0].shape[2], visualizer.img2html(img),
                    cos_as_bar(p.item()),
                    '✓' if correct.item() else '✗',
                    self.labels[cls.item()].replace('_', ' ').replace('-', ' ')))
        html.append('</div>')
        return ''.join(html)

    def show(self, visualizer):
        html = [
            '<h1>Best predictions</h1>',
            self._report(self.best), '<h1>Worst predictions</h1>',
            self._report(self.worst), '<h1>Confusions</h1>',
            self._report(self.confused)
        ]
        visualizer.html(''.join(html), win='report', opts=dict(title='report'))


if __name__ == '__main__':
    opt = configurator.from_cmdline()
    trainopts = opt['trainer']
    if opt.get('session_name', False):
        visualizer = Visualizer(env=opt['session_name'],
                                port=opt['visdom_port'])
        if 'visualizer_data' in opt:
            visualizer.load_state_dict(opt['visualizer_data'])
    device = torch.device(opt['device'])

    train_dataset = dataset.get_datasets(opt['datasets'])
    print(train_dataset)
    count_per_class = dataset.count_per_class(train_dataset).to(device)
    tester = get_tester(opt['tester']['name'], opt['device'], opt['tester'])
    tester.viz = visualizer
    state = {
        'train_dataset': train_dataset,
        'device': device,
        'count_per_class': count_per_class,
        'inv_count_per_class': 1.0 / count_per_class.float()
    }
    num_classes = len(train_dataset.classes)
    print(num_classes, 'classes')

    criterion = loader.get_loss(opt['loss']['name'],
                                state['inv_count_per_class'], opt['loss'])

    model = loader.get_model(opt['model'])

    metric_fc = loader.get_metric(num_classes, opt['metric'])

    optimizer = loader.get_optimizer(
        itertools.chain(model.parameters(), metric_fc.parameters()),
        opt['optimizer'])

    sched = scheduling.get_scheduler(optimizer, opt['scheduler'],
                                     trainopts.get('iter_n', 0))

    ckpt = configurator.Checkpointer(model, metric_fc, optimizer,
                                     opt['session_name'], opt)
    model.to(device)
    metric_fc.to(device)

    trainloader = data.DataLoader(train_dataset,
                                  pin_memory=True,
                                  batch_size=trainopts['batch_size'],
                                  shuffle=True,
                                  num_workers=trainopts['num_workers'])

    inspector = Inspector(16, train_dataset.classes)
    iters = trainopts.get('iter_n', 0)
    start = time.time()
    for i in range(trainopts.get('start_epoch', 0), trainopts['max_epoch']):
        tot_loss = 0
        inspector.reset()
        for ii, batch in enumerate(trainloader):
            model.train()
            data_input_cpu, label_cpu = batch
            data_input = data_input_cpu.to(device, non_blocking=True)  #.half()
            label = label_cpu.to(device, non_blocking=True).long()

            feature = model(data_input)
            output, cosine = metric_fc(feature, label)
            cosine = cosine.detach().to('cpu', non_blocking=True)
            clf_loss = criterion(output, label)
            output_label = torch.argmax(cosine, dim=1)
            inspector.analyze(data_input_cpu, cosine, label_cpu, output_label)

            #ortho_loss = ortho(metric_fc.weight[label, :], 0.001)
            ortho_loss = ortho(feature, 1e-5)
            loss = clf_loss + ortho_loss
            loss.backward()

            tot_loss += loss.item()
            sched.step(loss.item())
            if iters % trainopts.get('n_accumulations', 1) == 0:
                if trainopts.get('clip_grad', False) != False:
                    norm = torch.nn.utils.clip_grad.clip_grad_norm_(
                        itertools.chain(model.parameters(),
                                        metric_fc.parameters()),
                        trainopts['clip_grad'])
                    print('param norm:', norm)
                optimizer.step()
                optimizer.zero_grad()

            if iters % trainopts['print_freq'] == 0:
                visualizer.hist(feature.std(dim=0).detach().cpu(), 'feature distribution during train')
                acc = torch.sum((output_label == label_cpu).int()).item()
                speed = trainopts['print_freq'] / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print( '{} train epoch {} iter {} / {}, {} iters/s loss {} acc {}' .format(time_str, i, ii, len(trainloader), speed, loss.item(), acc))
                if trainopts['display']:
                    visualizer.display_current_results(iters,
                                                       ortho_loss.item(),
                                                       name='ortho_loss')
                    visualizer.display_current_results(iters,
                                                       clf_loss.item(),
                                                       name='train_loss')
                    visualizer.display_current_results(iters,
                                                       acc,
                                                       name='train_acc')
                    visualizer.hist(cosine[range(cosine.shape[0]), label_cpu],
                                    name='confidence out for good')
                    visualizer.hist(cosine, name='cosines')
                    visualizer.display_current_results(
                        iters,
                        optimizer.param_groups[0]['lr'],
                        name='learning_rate',
                        smooth=0)
                    visualizer.images(data_input[:16] * 0.5 + 0.5)
                    inspector.show(visualizer)

                start = time.time()

            if iters % trainopts['save_interval'] == 0:
                print('SAVING MODEL')
                ckpt.save(iters, {'visualizer_data': visualizer.state_dict()})

            if iters % trainopts['test_interval'] == 0:
                acc, test_loss = tester(model)
                if trainopts['display']:
                    visualizer.display_current_results(iters,
                                                       acc,
                                                       name='test_acc')
                    visualizer.display_current_results(iters,
                                                       test_loss,
                                                       name='test_loss')
            iters += 1
