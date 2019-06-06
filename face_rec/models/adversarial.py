import torch
import torch.nn as nn
import torch.nn.functional as F


class Adversarial(nn.Module):
    def __init__(self, vis):
        super(Adversarial, self).__init__()
        self.vis = vis
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(3, 2, padding=1),

            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(3, 2, padding=1),

            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(3, 2, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),

            nn.AdaptiveAvgPool2d(1)
        )
        self.lin1 = nn.Linear(1024, 1024)
        self.lin2 = nn.Linear(1024, 1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


    def correct(self, x, f, y):
        for p in self.parameters():
            p.requires_grad=True
        xd = x.detach()
        fd = f.detach()

        ypred = self(xd, fd)
        return F.cross_entropy(ypred, y)


    def adversarial(self, x, f):
        for p in self.parameters():
            p.requires_grad=False

        self.eval()
        ypred = self(x, f)
        self.train()

        logprob = F.log_softmax(ypred, dim=1)
        prob = torch.exp(logprob)
        H = -torch.sum(prob * logprob, dim=1)
        if False:
            with torch.no_grad():
                showable = x * 0.5 + 0.5
                prob = prob.view(-1)
                showable *= torch.where(prob.view(-1) > 0.5,
                        torch.full_like(prob, 1),
                        torch.full_like(prob, 0.5)).view(-1, 1, 1, 1)
                self.vis.images(showable[:16], nrow=2, name='adv')
        return H.mean()


    def losses(self, x, f, y, adv_batch):
        x, f, y = self.prepare_batch(x, f, y, adv_batch)

        correct_loss = self.correct(x, f, y)
        entropy = self.adversarial(x, f)
        return correct_loss, entropy


    def forward(self, x, f):
        res = self.model(x)
        res = res.view(res.shape[0], -1)
        y = self.lin2(F.relu(self.lin1(torch.cat([res, f], dim=1)), inplace=True))
        y = y.view(x.shape[0] // 2, 2)
        return y


    def prepare_batch(self, x, f, y, adv_batch):
        xs = []
        fs = []

        for i in range(len(x) // 2):
            xs.append(x[i])
            xs.append(adv_batch[i])

            fs.append(f[i])
            fs.append(f[i])

        return (torch.stack(xs, dim=0),
                torch.stack(fs, dim=0),
                torch.LongTensor([0] * (len(xs) // 2)).to(x.device))

