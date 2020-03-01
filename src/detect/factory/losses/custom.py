import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class QuatreCrossEntropy(nn.Module):

    def __init__(self, weights=[2.,1.,1.,0.4]):
        super(QuatreCrossEntropy, self).__init__()
        weights = [float(_) for _ in weights]
        weights = torch.tensor(weights).cuda()
        self.weights = weights / torch.sum(weights)

    def forward(self, y_pred, y_true):
        p1, p2, p3, p4 = y_pred
        t1 = y_true['grapheme_root']
        t2 = y_true['vowel_diacritic']
        t3 = y_true['consonant_diacritic']
        t4 = y_true['grapheme']
        # grapheme_root, vowel_diacritic, consonant_diacritic
        loss1 = F.cross_entropy(p1, t1)
        loss2 = F.cross_entropy(p2, t2)
        loss3 = F.cross_entropy(p3, t3)
        loss4 = F.cross_entropy(p4, t4)
        return self.weights[0] * loss1 \
               + self.weights[1] * loss2 \
               + self.weights[2] * loss3 \
               + self.weights[3] * loss4

class TripleCrossEntropy(nn.Module):

    def __init__(self, weights=[2.,1.,1.]):
        super(TripleCrossEntropy, self).__init__()
        weights = [float(_) for _ in weights]
        weights = torch.tensor(weights).cuda()
        self.weights = weights / torch.sum(weights)

    def forward_single(self, y_pred, y_true):
        p1, p2, p3 = y_pred
        t1 = y_true['grapheme_root']
        t2 = y_true['vowel_diacritic']
        t3 = y_true['consonant_diacritic']
        # grapheme_root, vowel_diacritic, consonant_diacritic
        loss1 = F.cross_entropy(p1, t1)
        loss2 = F.cross_entropy(p2, t2)
        loss3 = F.cross_entropy(p3, t3)
        return self.weights[0] * loss1 \
               + self.weights[1] * loss2 \
               + self.weights[2] * loss3

    def forward(self, y_pred, y_true):
        return self.forward_single(y_pred, y_true)


class GraphemeCE(nn.Module):

    def __init__(self):
        super(GraphemeCE, self).__init__()

    def forward_single(self, y_pred, y_true):
        return F.cross_entropy(y_pred, y_true['grapheme'])

    def forward(self, y_pred, y_true):
        return self.forward_single(y_pred, y_true)

class WeightedGraphemeCE(nn.Module):

    def __init__(self, w):
        super(WeightedGraphemeCE, self).__init__()
        self.w = torch.tensor(w).float().cuda()

    def forward_single(self, y_pred, y_true):
        return F.cross_entropy(y_pred, y_true['grapheme'], weight=self.w)

    def forward(self, y_pred, y_true):
        return self.forward_single(y_pred, y_true)

class MixupGraphemeCE(GraphemeCE):

    def forward(self, y_pred, y_true):
        if 'lam' in y_true.keys():
            # Training
            y_true1 = y_true['y_true1']
            y_true2 = y_true['y_true2']
            lam = y_true['lam']
            mix_loss1 = self.forward_single(y_pred, y_true1)
            mix_loss2 = self.forward_single(y_pred, y_true2)
            return lam * mix_loss1 + (1. - lam) * mix_loss2
        else:
            # Validation
            return self.forward_single(y_pred, y_true)

class WeightedTripleCE(nn.Module):

    def __init__(self, w1,w2,w3, weights=[2.,1.,1.]):
        super(WeightedTripleCE, self).__init__()
        w1 = torch.tensor(w1).float().cuda() if type(w1) != type(None) else None
        w2 = torch.tensor(w2).float().cuda() if type(w2) != type(None) else None
        w3 = torch.tensor(w3).float().cuda() if type(w3) != type(None) else None
        weights = [float(_) for _ in weights]
        weights = torch.tensor(weights).cuda()
        self.w1 = w1 / torch.sum(w1) if type(w1) != type(None) else None
        self.w2 = w2 / torch.sum(w2) if type(w2) != type(None) else None
        self.w3 = w3 / torch.sum(w3) if type(w3) != type(None) else None
        self.weights = weights / torch.sum(weights)

    def forward_single(self, y_pred, y_true):
        p1, p2, p3 = y_pred
        t1 = y_true['grapheme_root']
        t2 = y_true['vowel_diacritic']
        t3 = y_true['consonant_diacritic']
        # grapheme_root, vowel_diacritic, consonant_diacritic
        loss1 = F.cross_entropy(p1, t1, weight=self.w1)
        loss2 = F.cross_entropy(p2, t2, weight=self.w2)
        loss3 = F.cross_entropy(p3, t3, weight=self.w3)
        return self.weights[0] * loss1 \
               + self.weights[1] * loss2 \
               + self.weights[2] * loss3

    def forward(self, y_pred, y_true):
        return self.forward_single(y_pred, y_true)

class MixupTripleCE(TripleCrossEntropy):

    def forward(self, y_pred, y_true):
        if 'lam' in y_true.keys():
            # Training
            y_true1 = y_true['y_true1']
            y_true2 = y_true['y_true2']
            lam = y_true['lam']
            p1, p2, p3 = y_pred 
            mix_loss1 = self.forward_single(y_pred, y_true1)
            mix_loss2 = self.forward_single(y_pred, y_true2)
            return lam * mix_loss1 + (1. - lam) * mix_loss2
        else:
            # Validation
            return self.forward_single(y_pred, y_true)

class WeightedMixupTripleCE(WeightedTripleCE):

    def forward(self, y_pred, y_true):
        if 'lam' in y_true.keys():
            # Training
            y_true1 = y_true['y_true1']
            y_true2 = y_true['y_true2']
            lam = y_true['lam']
            p1, p2, p3 = y_pred 
            mix_loss1 = self.forward_single(y_pred, y_true1)
            mix_loss2 = self.forward_single(y_pred, y_true2)
            return lam * mix_loss1 + (1. - lam) * mix_loss2
        else:
            # Validation
            return self.forward_single(y_pred, y_true)

