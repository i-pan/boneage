import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridLoss(nn.Module):

    def __init__(self,
                 w1=1.,w2=1.):
        super(HybridLoss, self).__init__()
        self.w1 = w1
        self.w2 = w2

    def forward(self, y_pred, y_true):
        return self.w1 * F.l1_loss(y_pred, y_true) \
               + self.w2 * F.mse_loss(y_pred, y_true)

class BalancedHybridLoss(nn.Module):

    def __init__(self, 
                 strata_weights,
                 w1=1.,w2=1.):
        super(BalancedHybridLoss, self).__init__()
        self.w1 = w1
        self.w2 = w2
        strata = torch.from_numpy(np.asarray([0,24]+list(np.arange(12*3, 12*17, 12))+[230]) / 228.)
        self.strata = torch.cat((strata[:-1].unsqueeze(1), strata[1:].unsqueeze(1)), dim=1).cuda()
        self.strata_weights = torch.from_numpy(strata_weights).cuda()

    def calculate_loss(self, y_pred, y_true):
        return self.w1 * F.l1_loss(y_pred, y_true, reduction='none') \
               + self.w2 * F.mse_loss(y_pred, y_true, reduction='none')

    def determine_strata(self, value):
        upper = (value <  self.strata[:,1]).float()
        lower = (value >= self.strata[:,0]).float()
        return (upper+lower == 2).nonzero()

    def forward(self, y_pred, y_true):
        losses = self.calculate_loss(y_pred, y_true)
        strata = torch.cat([self.determine_strata(i) for i in y_true])[:,0]  
        return (self.strata_weights[strata] * losses).mean()
