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

with open('../configs/baseline.yaml') as f:
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
valid_loader = builder.build_dataloader(cfg, data_info={'imgfiles': valid_images, 'labels': valid_labels, 'male': valid_male, 'coords': valid_coords}, mode='valid')

train_iter = iter(train_loader)
valid_iter = iter(valid_loader)

data = next(valid_iter)


import matplotlib.pyplot as plt

plt.imshow(data[0][1][0], cmap='gray'); plt.show()

from factory.evaluate import grid_patches



# Abbreviate valid_df
valid_df = valid_df.iloc[:1000]

print('TRAIN: n={}'.format(len(train_df)))
print('VALID: n={}'.format(len(valid_df)))

train_images = [osp.join(cfg['dataset']['data_dir'], '{}.png'.format(_)) for _ in train_df['image_id']]
valid_images = [osp.join(cfg['dataset']['data_dir'], '{}.png'.format(_)) for _ in valid_df['image_id']]

train_labels = []
for rownum in range(len(train_df)):
    rowlabel = {
        'grapheme_root': train_df['grapheme_root'].iloc[rownum],
        'vowel_diacritic': train_df['vowel_diacritic'].iloc[rownum],
        'consonant_diacritic': train_df['consonant_diacritic'].iloc[rownum]            
    }
    train_labels.append(rowlabel)

valid_labels = []
for rownum in range(len(valid_df)):
    rowlabel = {
        'grapheme_root': valid_df['grapheme_root'].iloc[rownum],
        'vowel_diacritic': valid_df['vowel_diacritic'].iloc[rownum],
        'consonant_diacritic': valid_df['consonant_diacritic'].iloc[rownum]            
    }
    valid_labels.append(rowlabel)

train_loader = builder.build_dataloader(cfg, data_info={'imgfiles': train_images, 'labels': train_labels}, mode='train')
valid_loader = builder.build_dataloader(cfg, data_info={'imgfiles': valid_images, 'labels': valid_labels}, mode='valid')

model = builder.build(models, cfg['model']['name'], cfg['model']['params'])
model = model.train().cuda()
criterion = builder.build(losses, cfg['loss']['name'], cfg['loss']['params'])

predictor = evaluate.Predictor(loader=valid_loader)
y_true, y_pred = predictor.predict(model, criterion, 0)





out = model(data[0])
criterion(out, data[1])

