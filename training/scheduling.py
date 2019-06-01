import copy

from torch.optim.lr_scheduler import *

import utils.maths


class CurriculumScheduler:
    def __init__(self, optimizer, schedule, last_iter=-1):
        print(type(schedule), schedule)
        self.optimizer = optimizer
        self.schedule = schedule
        self.last_iter = last_iter

    def step(self, *unused):
        self.last_iter += 1
        the_lr = self.schedule[-1][1]
        the_mom = self.schedule[-1][2]
        for lo, hi in zip(self.schedule[:-1], self.schedule[1:]):
            limit_lo, lr_lo, mom_lo = lo
            lim_hi, lr_hi, mom_hi = hi

            if limit_lo <= self.last_iter < lim_hi:
                t = utils.maths.get_t(limit_lo, lim_hi, self.last_iter)
                the_lr = utils.maths.lerp(lr_lo, lr_hi, t)
                the_mom = utils.maths.lerp(mom_lo, mom_hi, t)

        for group in self.optimizer.param_groups:
            group['lr'] = the_lr
            group['momentum'] = the_mom


    def __repr__(self):
        return "CurriculumScheduler({})".format(self.schedule)

class NoSched:
    def __init__(self, optimizer, last_iter):
        pass

    def step(self, *unused):
        pass

    def __repr__(self):
        return "NoScheduler()"

def get_scheduler(optimizer, args, last_epoch=-1):
    args = copy.deepcopy(args)
    name = args.pop('name')
    sched = globals()[name](optimizer, last_iter=last_epoch, **args)
    print('Created new scheduler')
    print(sched)
    return sched
