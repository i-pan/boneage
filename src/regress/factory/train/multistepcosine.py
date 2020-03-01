import math
import numpy as np

from torch.optim.lr_scheduler import _LRScheduler

class MultiStepCosineAnneal(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma once the
    number of epoch reaches one of the milestones. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside
    this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, 
                 optimizer, 
                 initial_lr,
                 milestones, 
                 epochs, 
                 steps_per_epoch,
                 gamma=0.1, 
                 final_lr=1e-12,
                 last_epoch=-1):
        self.optimizer = optimizer
        self.final_lr = final_lr
        assert type(milestones) is list
        self.milestones = [m*steps_per_epoch for m in milestones]
        self.gamma = gamma
        self.previous_epoch = 0 
        self.steps_per_milestone = [0]
        self.steps_per_milestone.append(self.milestones[0])
        for i in range(len(self.milestones)-1):
            self.steps_per_milestone.append(self.milestones[i+1]-self.milestones[i])
        self.steps_per_milestone.append(epochs*steps_per_epoch - self.milestones[-1])

        self.lr_per_milestone = [initial_lr * gamma ** (ind+1) for ind, _ in enumerate(milestones)]

        # Initialize learning rate variables
        if last_epoch == -1:
            for idx, group in enumerate(self.optimizer.param_groups):
                group['initial_lr'] = initial_lr
                group['next_lr'] = group['initial_lr'] * gamma

        super(MultiStepCosineAnneal, self).__init__(optimizer, last_epoch)
        self.milestone_steps = 0 # Ensure that it is 0

        # [30, 80]
        # [0.001, 0.0001]

    def get_milestone(self):
        if np.sum(self.last_epoch < np.asarray(self.milestones)) == 0:
            return len(self.milestones)
        return list(self.last_epoch < np.asarray(self.milestones)).index(True)

    def get_lr(self):
        # Using self.previous_epoch protects against when 
        # you call get_lr() independently of step()

        if self.previous_epoch != self.last_epoch:

            # Check to see if milestone has been reached:
            if self.last_epoch in self.milestones:
                for group in self.optimizer.param_groups:
                    group['initial_lr'] *= self.gamma
                    if self.last_epoch == self.milestones[-1]:
                        group['next_lr'] = self.final_lr
                    else:
                        group['next_lr'] = group['initial_lr'] * self.gamma

        self.previous_epoch = self.last_epoch
        
        pct = (self.last_epoch - np.sum(self.steps_per_milestone[:self.get_milestone()+1])) / self.steps_per_milestone[self.get_milestone()+1]
        return [self._annealing_cos(group['initial_lr'], group['next_lr'], pct) for group in self.optimizer.param_groups]


    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs]

    def _annealing_cos(self, start, end, pct):
        "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

