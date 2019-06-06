from face_rec.utils.visualizer import Visualizer

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
