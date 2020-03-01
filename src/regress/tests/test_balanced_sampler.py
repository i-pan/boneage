import sys ; sys.path.insert(0, '..')

import argparse
import pandas as pd
import numpy as np
import torch
import yaml
import os, os.path as osp

from factory import set_reproducibility

import factory.train
import factory.evaluate as evaluate
import factory.builder as builder
import factory.models as models
import factory.losses as losses
import factory.optim as optim
import factory.train.scheduler as scheduler

from run import get_train_valid_test, set_inference_batch_size

with open('../configs/sampling/i0o0.yaml') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

cfg = set_inference_batch_size(cfg)

cfg['dataset']['data_dir'] = osp.join('..', cfg['dataset']['data_dir'])
cfg['dataset']['csv_filename'] = osp.join('..', cfg['dataset']['csv_filename'])
df = pd.read_csv(cfg['dataset']['csv_filename'])

ofold = cfg['dataset']['outer_fold']
ifold = cfg['dataset']['inner_fold']

train_df, valid_df, test_df = get_train_valid_test(cfg, df, ofold, ifold)

train_images = [osp.join(cfg['dataset']['data_dir'], _) for _ in train_df['imgfile']]
valid_images = [osp.join(cfg['dataset']['data_dir'], _) for _ in valid_df['imgfile']]

train_labels = list(train_df['boneage'])
valid_labels = list(valid_df['boneage'])

train_male = list(train_df['male'].astype('float32'))
valid_male = list(valid_df['male'].astype('float32'))

train_coords = {k : np.asarray(train_df[k]) for k in ['x1','y1','x2','y2']}
valid_coords = {k : np.asarray(valid_df[k]) for k in ['x1','y1','x2','y2']}

train_loader = builder.build_dataloader(cfg, data_info={'imgfiles': train_images, 'labels': train_labels, 'male': train_male, 'coords': train_coords}, mode='train')
train_loader.dataset.scale = 1.
train_iter = iter(train_loader)

labels = []
while 1:
    data = next(train_iter)
    labels.append(data[1])

labels = [_.cpu().numpy() for _ in labels]
labels = np.concatenate(labels)
strata = pd.cut(labels, bins=[0,24]+list(np.arange(12*3, 12*17, 12))+[230], labels=range(16))



