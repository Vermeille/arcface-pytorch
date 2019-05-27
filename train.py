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


def report(imgs, pred):
    html = []
    for img, p in zip(imgs, pred):
        print(pred.shape, p.shape)
        html.append(
            visualizer.img2html(img) + '<br/>' + str(p.item()) + '<br/>')
        visualizer.html(''.join(html), win='report', opts=dict(title='report'))


if __name__ == '__main__':
    opt = configurator.from_cmdline()
    trainopts = opt['trainer']
    if opt.get('session_name', False):
        visualizer = Visualizer(env=opt['session_name'])
    device = torch.device(opt['device'])

    train_dataset = dataset.get_datasets(opt['datasets'])
    count_per_class = dataset.count_per_class(train_dataset).to(device)
    tester = get_tester(opt['tester']['name'], opt['device'], opt['tester'])
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

    #view_model(model, train_dataset[0])
    ckpt = configurator.Checkpointer(model, metric_fc, optimizer,
                                     opt['session_name'], opt)
    model.to(device)
    metric_fc.to(device)

    trainloader = data.DataLoader(train_dataset,
                                  pin_memory=True,
                                  batch_size=trainopts['batch_size'],
                                  shuffle=True,
                                  num_workers=trainopts['num_workers'])

    iters = trainopts.get('iter_n', 0)
    start = time.time()
    for i in range(trainopts.get('start_epoch', 0), trainopts['max_epoch']):
        tot_loss = 0
        for ii, batch in enumerate(trainloader):
            data_input, label = batch
            data_input = data_input.to(device, non_blocking=True)  #.half()
            label = label.to(device, non_blocking=True).long()

            feature = model(data_input)
            output, cosine = metric_fc(feature, label)
            loss = criterion(output, label)
            loss.backward()
            tot_loss += loss.item()
            if ii % trainopts.get('n_accumulations', 1) == 0:
                sched.step(loss.item())
                optimizer.zero_grad()

            if iters % trainopts['print_freq'] == 0:
                cosine = cosine.data.cpu().numpy()
                output_label = np.argmax(cosine, axis=1)
                report(batch[0] * 0.5 + 0.5, cosine[range(cosine.shape[0]), batch[1]])
                label = label.data.cpu().numpy()
                acc = np.sum((output_label == label).astype(int))
                speed = trainopts['print_freq'] / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print(
                    '{} train epoch {} iter {} / {}, {} iters/s loss {} acc {}'
                    .format(time_str, i, ii, len(trainloader), speed,
                            loss.item(), acc))
                if trainopts['display']:
                    visualizer.display_current_results(iters,
                                                       loss.item(),
                                                       name='train_loss')
                    visualizer.display_current_results(iters,
                                                       acc,
                                                       name='train_acc')
                    visualizer.hist(cosine[range(cosine.shape[0]), label],
                                    name='confidence out for good')
                    visualizer.display_current_results(
                        iters,
                        optimizer.param_groups[0]['lr'],
                        name='learning_rate',
                        smooth=0)
                    visualizer.images(data_input[:16] * 0.5 + 0.5)

                start = time.time()

            if iters % trainopts['save_interval'] == 0:
                print('SAVING MODEL')
                ckpt.save(ii)

            if iters % trainopts['test_interval'] == 0:
                model.eval()
                acc, test_loss = tester(model)
                if trainopts['display']:
                    visualizer.display_current_results(iters,
                                                       acc,
                                                       name='test_acc')
                    visualizer.display_current_results(iters,
                                                       test_loss,
                                                       name='test_loss')
                model.train()
            iters += 1
