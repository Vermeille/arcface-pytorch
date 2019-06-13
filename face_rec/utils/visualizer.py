import math
import time
import base64 as b64
from io import BytesIO

import visdom
import torch

from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
import numpy as np
from PIL import Image


class Visualizer(object):
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.vis.close()

        self.iters = {}
        self.lines = {}

    def state_dict(self):
        return {'iters': self.iters, 'lines': self.lines}

    def load_state_dict(self, loaded):
        self.iters = loaded['iters']
        self.lines = loaded['lines']

    def display_current_results(self, iters, x, name='train_loss', smooth=0.9):
        if name not in self.iters:
            self.iters[name] = []

        if name not in self.lines:
            self.lines[name] = []

        self.iters[name].append(iters)
        if len(self.lines[name]) > 0:
            self.lines[name].append(smooth * self.lines[name][-1] +
                                    (1 - smooth) * x)
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
        self.vis.images(imgs,
                        win='test example',
                        nrow=max(1, int(math.sqrt(len(imgs)))),
                        opts={'title': 'test images'})

    def hist(self, dat, name):
        self.vis.histogram(dat.reshape(1, -1), win=name, opts=dict(numbins=50,
            title=name))

    def display_roc(self, y_true, y_pred):
        fpr, tpr, ths = roc_curve(y_true, y_pred)
        self.vis.line(
            X=fpr,
            Y=tpr,
            # win='roc',
            opts=dict(legend=['roc'], title='roc'))

    @staticmethod
    def img2html(img, opts=None):
        if isinstance(img, torch.Tensor):
            img = img.cpu().detach().numpy()

        opts = {} if opts is None else opts
        if isinstance(img, np.ndarray):
            nchannels = img.shape[0] if img.ndim == 3 else 1
            if nchannels == 1:
                img = np.squeeze(img)
                img = img[np.newaxis, :, :].repeat(3, axis=0)

            if 'float' in str(img.dtype):
                if img.max() <= 1:
                    img = img * 255.
                img = np.uint8(img)

            img = np.transpose(img, (1, 2, 0))
            img = Image.fromarray(img)

        opts['width'] = opts.get('width', img.width)
        opts['height'] = opts.get('height', img.height)

        buf = BytesIO()
        image_type = 'png'
        imsave_args = {}
        if 'jpgquality' in opts:
            image_type = 'jpeg'
            imsave_args['quality'] = opts['jpgquality']

        img.save(buf, format=image_type.upper(), **imsave_args)

        b64encoded = b64.b64encode(buf.getvalue()).decode('utf-8')

        return '<img src="data:image/{};base64,{}"/>'.format(
            image_type, b64encoded)

    def html(self, html, **kwargs):
        self.vis.text(html, **kwargs)

