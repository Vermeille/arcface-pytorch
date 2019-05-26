import visdom
import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
import math


class Visualizer(object):

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(port='8097', env=env, **kwargs)
        self.vis.close()

        self.iters = {}
        self.lines = {}

    def display_current_results(self, iters, x, name='train_loss', smooth=0.9):
        if name not in self.iters:
            self.iters[name] = []

        if name not in self.lines:
            self.lines[name] = []

        self.iters[name].append(iters)
        if len(self.lines[name]) > 0:
            self.lines[name].append(smooth * self.lines[name][-1] + (1 - smooth) * x)
        else:
            self.lines[name].append(x)

        self.vis.line(X=np.array(self.iters[name]),
                      Y=np.array(self.lines[name]),
                      win=name,
                      opts=dict(legend=[name], title=name))

    def image(self, img):
        self.vis.image(img, win='example')

    def images(self, imgs, nrow=None, name='example'):
        if nrow is None:
            nrow = max(1, int(math.sqrt(len(imgs))))
        self.vis.images(imgs, win=name, nrow=nrow)

    def test_images(self, imgs):
        self.vis.images(imgs, win='test example',
                nrow=max(1, int(math.sqrt(len(imgs)))),
                opts={'title': 'test images'})

    def hist(self, dat, name):
        self.vis.histogram(dat.reshape(1, -1), win=name, opts=dict(numbins=50))

    def display_roc(self, y_true, y_pred):
        fpr, tpr, ths = roc_curve(y_true, y_pred)
        self.vis.line(X=fpr,
                      Y=tpr,
                      # win='roc',
                      opts=dict(legend=['roc'],
                                title='roc'))
