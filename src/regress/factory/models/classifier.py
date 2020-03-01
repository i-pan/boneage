import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import *


class Regressor(nn.Module):

    def __init__(self,
                 backbone,
                 dropout,
                 pretrained,
                 embed_dim=16,
                 final_bn=True,
                 final_act='sigmoid'):

        super(Regressor, self).__init__()
        
        # TODO:
        # I don't like using wild imports and eval ...
        # but I can't figure out how to import the backbones
        self.backbone, dim_feats = eval(backbone)(pretrained=pretrained)
        self.dropout = nn.Dropout(dropout)
        self.embed = nn.Embedding(2, embed_dim)
        self.bn = nn.BatchNorm1d(dim_feats + embed_dim) if final_bn else None
        self.fc1 = nn.Linear(dim_feats + embed_dim, int(dim_feats / 4))
        self.fc2 = nn.Linear(int(dim_feats / 4), 1)
        if final_act == 'sigmoid':
            self.act = torch.sigmoid
        elif final_act == 'relu':
            self.act = F.relu

    def forward(self, x, male):
        x = self.backbone(x)
        x = self.dropout(x)
        e = self.embed(male)
        x = torch.cat([x, e], dim=1)
        if self.bn: x = self.bn(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.act(x)[:,0]

class SimpleRegressor(nn.Module):

    def __init__(self,
                 backbone,
                 dropout,
                 pretrained):

        super(SimpleRegressor, self).__init__()
        
        # TODO:
        # I don't like using wild imports and eval ...
        # but I can't figure out how to import the backbones
        self.backbone, dim_feats = eval(backbone)(pretrained=pretrained)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dim_feats, 1)

    def forward(self, x, male=None):
        x = self.backbone(x)
        x = self.dropout(x)
        return torch.sigmoid(self.fc(x))[:,0]

