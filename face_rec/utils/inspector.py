import torch

from face_rec.utils.visualizer import Visualizer

class ClassificationInspector:
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
                    dat[0][0].shape[2], Visualizer.img2html(img),
                    cos_as_bar(p.item()), '✓' if correct.item() else '✗',
                    self.labels[cls.item()].replace('_',
                                                    ' ').replace('-', ' ')))
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

class ComparisonInspector:
    def __init__(self, nb_to_vis, dataset, pairs, center_value=0):
        self.center_value = center_value
        self.topk = nb_to_vis
        self.dataset = dataset
        self.pairs = pairs
        self.reset()

    def reset(self):
        self.best = []
        self.worst = []
        self.confused = []

    def analyze(self, pred, labels):
        with torch.no_grad():
            sorted_pred = torch.argsort(pred)
            sorted_labels = (labels > 0.5)[sorted_pred]
            matching_idx = sorted_pred[sorted_labels.nonzero()]
            diff_idx = sorted_pred[(sorted_labels == 0).nonzero()]

        self.best_same = [(pred[i], self.pairs[i]) for i in matching_idx[-self.topk:]]
        self.worst_same = [(pred[i], self.pairs[i]) for i in matching_idx[:self.topk]]

        self.best_diff = [(pred[i], self.pairs[i]) for i in diff_idx[:self.topk]]
        self.worst_diff = [(pred[i], self.pairs[i]) for i in diff_idx[-self.topk:]]

        idxs = list(range(len(pred)))
        idxs.sort(key=lambda i: torch.abs(pred[i] - self.center_value))
        self.confused = [(pred[i], self.pairs[i])
                for i in idxs[:self.topk]]

    def _report(self, dat):
        def cos_as_bar(cos):
            return '<div style="width:{}%;background-color:{};height:5px"></div>'.format(
                abs(cos) * 100, "green" if cos >= 0 else "red")

        html = ['<div style="display:flex;flex-wrap:wrap">']
        for p, pair in dat:
            img1 = self.dataset.from_path(pair[0])
            html.append(
                ('<div style="padding:3px;width:{}px">'
                    '<div style="display:inline">'
                        '<div onclick="javascript:prompt(\'path\', \'{}\')">'
                            '{}'
                        '</div>'
                        '<div onclick="javascript:prompt(\'path\', \'{}\')">'
                            '{}'
                        '</div>'
                    '</div>'
                    '{}{}'
                '</div>').format(
                    img1.width,
                    pair[0], Visualizer.img2html(img1),
                    pair[1], Visualizer.img2html(self.dataset.from_path(pair[1])),
                    cos_as_bar(p.item() - self.center_value),
                    'SAME' if pair[2] == '1' else 'DIFFERENT'))

        html.append('</div>')
        return ''.join(html)

    def show(self, visualizer, title):
        html = [
            '<h1>Best same</h1>',
            self._report(self.best_same),
            '<h1>Worst same</h1>',
            self._report(self.worst_same),
            '<h1>Best different</h1>',
            self._report(self.best_diff),
            '<h1>Worst different</h1>',
            self._report(self.worst_diff),
            '<h1>Confusions</h1>',
            self._report(self.confused)
        ]
        visualizer.html(''.join(html), win='report'+title, opts=dict(title='report '
            + title))
