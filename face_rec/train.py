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

import face_rec.data.dataset as dataset
import face_rec.configurator as configurator
from face_rec.utils import Visualizer, view_model
from face_rec.models import *
from face_rec.testers.loader import get_tester
import face_rec.models.loader as loader
import face_rec.training.scheduling as scheduling
from face_rec.utils.inspector import ClassificationInspector


def ortho_l2(w, strength=1e-4):
    return strength * (
        torch.mm(w, w.t()) *
        (1 - torch.eye(w.shape[0], device=w.device))).pow(2).sum(dim=1).mean()


def ortho_abs(w, strength=1e-3):
    return strength * (torch.mm(w, w.t()) *
                       (1 - torch.eye(w.shape[0], device=w.device))).abs().sum(
                           dim=1).mean()


def ortho(w, strength=1e-3):
    cosine = torch.mm(w, w.t())
    no_diag = (1 - torch.eye(w.shape[0], device=w.device))
    return strength * (cosine * no_diag -
                       0.5).clamp(min=0).pow(2).sum(dim=1).mean()




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
    tester = get_tester(opt['device'], opt['testers'])
    tester.viz = visualizer
    state = {
        'train_dataset': train_dataset,
        'device': device,
        'count_per_class': count_per_class,
        'inv_count_per_class': len(train_dataset) / count_per_class.float()
    }
    num_classes = len(train_dataset.classes)
    print(num_classes, 'classes')

    criterion = loader.get_loss(opt['loss']['name'],
                                state['inv_count_per_class'], opt['loss'])

    model = loader.get_model(opt['model'])

    metric_fc = loader.get_metric(num_classes, opt['metric'])

    model.to(device)
    metric_fc.to(device)

    optimizer = loader.get_optimizer(
        itertools.chain(model.parameters(), metric_fc.parameters()),
        opt['optimizer'])

    sched = scheduling.get_scheduler(optimizer, opt['scheduler'],
                                     trainopts.get('iter_n', 0))

    ckpt = configurator.Checkpointer(model, metric_fc, optimizer,
                                     opt['session_name'], opt)
    trainloader = data.DataLoader(train_dataset,
                                  pin_memory=True,
                                  batch_size=trainopts['batch_size'],
                                  shuffle=True,
                                  num_workers=trainopts['num_workers'])

    inspector = ClassificationInspector(16, train_dataset.classes)
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
            feature_norm = torch.nn.functional.normalize(feature, dim=1)
            output, cosine = metric_fc(feature, label)
            cosine = cosine.detach().to('cpu', non_blocking=True)
            clf_loss = criterion(output, label)
            output_label = torch.argmax(cosine, dim=1)
            inspector.analyze(data_input_cpu, cosine, label_cpu, output_label)

            if trainopts.get('ortho_reg', False):
                #ortho_loss = ortho(feature_norm, trainopts['ortho_reg'])
                ortho_loss = ortho(
                    torch.nn.functional.normalize(metric_fc.weight[label, :]),
                    trainopts['ortho_reg'])
                loss = clf_loss + ortho_loss
            else:
                loss = clf_loss

            loss.backward()

            tot_loss += loss.item()
            sched.step(loss.item())
            if iters % trainopts.get('n_accumulations', 1) == 0:
                if trainopts.get('clip_grad', False) != False:
                    norm = torch.nn.utils.clip_grad.clip_grad_norm_(
                        itertools.chain(model.parameters(),
                                        metric_fc.parameters()),
                        trainopts['clip_grad'])
                    print('grad norm:', norm)
                optimizer.step()
                optimizer.zero_grad()

            if iters % trainopts['print_freq'] == 0:
                visualizer.hist(
                    feature.view(-1).detach().cpu(),
                    'feature distribution during train')
                visualizer.hist(
                    torch.mm(feature_norm,
                             feature_norm.t()).detach().cpu().view(-1),
                    'training batch cosine distribution')
                acc = torch.sum((output_label == label_cpu).int()).item()
                speed = trainopts['print_freq'] / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print(
                    '{} train epoch {} iter {} / {}, {} iters/s loss {} acc {}'
                    .format(time_str, i, ii, len(trainloader), speed,
                            loss.item(), acc))
                if trainopts['display']:
                    if trainopts.get('ortho_reg', False):
                        visualizer.display_current_results(iters,
                                                           ortho_loss.item(),
                                                           name='ortho_loss',
                                                           smooth=0)
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
                test_res = tester(model)
                tester.show(visualizer)
                print(test_res)
                if trainopts['display']:
                    for k, v in test_res.items():
                        visualizer.display_current_results(iters, v, name=k)
            iters += 1
