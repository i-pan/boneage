import argparse
import logging
import pandas as pd
import numpy as np
import pickle
import copy
import torch
import yaml
import os, os.path as osp

from factory import set_reproducibility

import factory.train
import factory.evaluate
import factory.builder as builder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('mode', type=str) 
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=-1)
    return parser.parse_args()

def create_logger(cfg, mode):
    logfile = osp.join(cfg['evaluation']['params']['save_checkpoint_dir'], 'log_{}.txt'.format(mode))
    if osp.exists(logfile): os.system('rm {}'.format(logfile))
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger('root')
    logger.addHandler(logging.FileHandler(logfile, 'a'))
    return logger

def set_inference_batch_size(cfg):
    if 'evaluation' in cfg.keys() and 'train' in cfg.keys():
        if 'batch_size' not in cfg['evaluation'].keys(): 
            cfg['evaluation']['batch_size'] = 2*cfg['train']['batch_size']

    if 'test' in cfg.keys() and 'train' in cfg.keys():
        if 'batch_size' not in cfg['test'].keys(): 
            cfg['test']['batch_size'] = 2*cfg['train']['batch_size']

    if 'predict' in cfg.keys() and 'train' in cfg.keys():
        if 'batch_size' not in cfg['predict'].keys(): 
            cfg['predict']['batch_size'] = 2*cfg['train']['batch_size']

    return cfg 

def check_param(cfg, param):
    return param in cfg.keys() and type(param) != type(None)

def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    # We will set all the seeds we can, in vain ...
    set_reproducibility(cfg['seed'])
    # Set GPU
    torch.cuda.set_device(args.gpu)

    if args.num_workers > 0:
        if 'transform' not in cfg.keys():
            cfg['transform'] = {}
        cfg['transform']['num_workers'] = args.num_workers

    cfg = set_inference_batch_size(cfg)

    if args.mode == 'predict':
        predict(args, cfg)
        return
    elif args.mode == 'predict_ensemble':
        predict_ensemble(args, cfg)
        return

    if 'mixup' not in cfg['train']['params'].keys():
        cfg['train']['params']['mixup'] = None

    # Make directory to save checkpoints
    if not osp.exists(cfg['evaluation']['params']['save_checkpoint_dir']): 
        os.makedirs(cfg['evaluation']['params']['save_checkpoint_dir'])

    # Load in labels with CV splits
    df = pd.read_csv(cfg['dataset']['csv_filename'])
    if 'chromosome' in cfg['dataset'].keys():
        if cfg['dataset']['chromosome'].lower() == 'xy': 
            df = df[df['male']]
        elif cfg['dataset']['chromosome'].lower() == 'xx':
            df = df[~df['male']]
        else:
            raise Exception('`chromosome` must be one of [`XY`,`XX`]')

    ofold = cfg['dataset']['outer_fold']
    ifold = cfg['dataset']['inner_fold']

    train_df, valid_df, test_df = get_train_valid_test(cfg, df, ofold, ifold)

    logger = create_logger(cfg, args.mode)
    logger.info('Saving to {} ...'.format(cfg['evaluation']['params']['save_checkpoint_dir']))

    if args.mode == 'find_lr':
        cfg['optimizer']['params']['lr'] = cfg['find_lr']['params']['start_lr']
        find_lr(args, cfg, train_df, valid_df)
    elif args.mode == 'train':
        train(args, cfg, train_df, valid_df)
    elif args.mode == 'test':
        test(args, cfg, test_df)

def get_train_valid_test(cfg, df, ofold, ifold):
    # Get train/validation set
    if cfg['train']['outer_only']: 
        # valid and test are essentially the same here
        train_df = df[df['outer'] != ofold]
        valid_df = df[df['outer'] == ofold]
        test_df  = df[df['outer'] == ofold]
    else:
        test_df = df[df['outer'] == ofold]
        df = df[df['outer'] != ofold]
        train_df = df[df['inner{}'.format(ofold)] != ifold]
        valid_df = df[df['inner{}'.format(ofold)] == ifold]
    return train_df, valid_df, test_df

def get_invfreq_weights(values, scale=None):
    logger = logging.getLogger('root')
    values, counts = np.unique(values, return_counts=True)
    num_samples = np.sum(counts)
    freqs = counts / float(num_samples)
    max_freq = np.max(freqs)
    invfreqs = max_freq / freqs
    if scale == 'log':
        logger.info('  Log scaling ...') 
        invfreqs = np.log(invfreqs+1)
    elif scale == 'sqrt':
        logger.info('  Square-root scaling ...')
        invfreqs = np.sqrt(invfreqs)
    invfreqs = invfreqs / np.sum(invfreqs)
    return invfreqs

def setup(args, cfg, train_df, valid_df):

    logger = logging.getLogger('root')

    train_images = [osp.join(cfg['dataset']['data_dir'], _) for _ in train_df['imgfile']]
    valid_images = [osp.join(cfg['dataset']['data_dir'], _) for _ in valid_df['imgfile']]

    train_labels = list(train_df['boneage'])
    valid_labels = list(valid_df['boneage'])

    train_male = list(train_df['male'].astype('float32'))
    valid_male = list(valid_df['male'].astype('float32'))

    if cfg['dataset']['coords']:
        train_coords = {k : np.asarray(train_df[k]) for k in ['x1','y1','x2','y2']}
        valid_coords = {k : np.asarray(valid_df[k]) for k in ['x1','y1','x2','y2']}
    else:
        train_coords = None
        valid_coords = None

    train_loader = builder.build_dataloader(cfg, data_info={'imgfiles': train_images, 'labels': train_labels, 'male': train_male, 'coords': train_coords}, mode='train')
    valid_loader = builder.build_dataloader(cfg, data_info={'imgfiles': valid_images, 'labels': valid_labels, 'male': valid_male, 'coords': valid_coords}, mode='valid')
    
    # Adjust steps per epoch if necessary (i.e., equal to 0)
    # We assume if gradient accumulation is specified, then the user
    # has already adjusted the steps_per_epoch accordingly in the 
    # config file
    steps_per_epoch = cfg['train']['params']['steps_per_epoch']
    gradient_accmul = cfg['train']['params']['gradient_accumulation']
    if steps_per_epoch == 0:
        cfg['train']['params']['steps_per_epoch'] = len(train_loader)


    # Generic build function will work for model/loss
    logger.info('Building [{}] architecture ...'.format(cfg['model']['name']))
    logger.info('  Using [{}] backbone ...'.format(cfg['model']['params']['backbone']))
    logger.info('  Pretrained weights : {}'.format(cfg['model']['params']['pretrained']))
    model = builder.build_model(cfg['model']['name'], cfg['model']['params'])
    model = model.train().cuda()

    if cfg['loss']['name'] == 'BalancedHybridLoss':
        strata_weights = pd.cut(train_df['boneage'], bins=[0,24]+list(np.arange(12*3, 12*17, 12))+[228], labels=range(16))
        strata_weights = pd.DataFrame(strata_weights.value_counts()).reset_index().sort_values('index', ascending=True)
        strata_weights = strata_weights['boneage'].max() / strata_weights['boneage']
        strata_weights = np.asarray(strata_weights)
        cfg['loss']['params']['strata_weights'] = strata_weights
    criterion = builder.build_loss(cfg['loss']['name'], cfg['loss']['params'])
    optimizer = builder.build_optimizer(
        cfg['optimizer']['name'], 
        model.parameters(), 
        cfg['optimizer']['params'])
    scheduler = builder.build_scheduler(
        cfg['scheduler']['name'], 
        optimizer, 
        cfg=cfg)

    return cfg, \
           train_loader, \
           valid_loader, \
           model, \
           optimizer, \
           criterion, \
           scheduler 

def find_lr(args, cfg, train_df, valid_df):

    logger = logging.getLogger('root')

    logger.info('FINDING LR ...')

    cfg, \
    train_loader, \
    valid_loader, \
    model, \
    optimizer, \
    criterion, \
    scheduler = setup(args, cfg, train_df, valid_df)

    finder = factory.train.LRFinder(
        loader=train_loader,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        save_checkpoint_dir=cfg['evaluation']['params']['save_checkpoint_dir'],
        logger=logger,
        gradient_accumulation=cfg['train']['params']['gradient_accumulation'],
        mixup=cfg['train']['params']['mixup'])

    finder.find_lr(**cfg['find_lr']['params'])

    logger.info('Results are saved in : {}'.format(osp.join(finder.save_checkpoint_dir, 'lrfind.csv')))

def train(args, cfg, train_df, valid_df):
    
    logger = logging.getLogger('root')

    logger.info('TRAINING : START')

    logger.info('TRAIN: n={}'.format(len(train_df)))
    logger.info('VALID: n={}'.format(len(valid_df)))

    cfg, \
    train_loader, \
    valid_loader, \
    model, \
    optimizer, \
    criterion, \
    scheduler = setup(args, cfg, train_df, valid_df)

    evaluator = getattr(factory.evaluate, cfg['evaluation']['evaluator'])
    evaluator = evaluator(loader=valid_loader,
        **cfg['evaluation']['params'])

    trainer = getattr(factory.train, cfg['train']['trainer'])
    trainer = trainer(loader=train_loader,
        model=model,
        optimizer=optimizer,
        schedule=scheduler,
        criterion=criterion,
        evaluator=evaluator,
        logger=logger)
    trainer.train(**cfg['train']['params'])


def test(args, cfg, test_df):

    logger = logging.getLogger('root')
    logger.info('TESTING : START')
    logger.info('TEST: n={}'.format(len(test_df)))

    test_images = [osp.join(cfg['dataset']['data_dir'], _) for _ in test_df['imgfile']]
    test_labels = list(test_df['boneage'])
    test_male = list(test_df['male'].astype('float32'))
    if cfg['dataset']['coords']:
        test_coords = {k : np.asarray(test_df[k]) for k in ['x1','y1','x2','y2']}
    else:
        test_coords = None

    test_loader = builder.build_dataloader(cfg, data_info={'imgfiles': test_images, 'labels': test_labels, 'male': test_male, 'coords': test_coords}, mode='test')

    model = builder.build_model(cfg['model']['name'], cfg['model']['params'])
    model.load_state_dict(torch.load(cfg['test']['checkpoint'], map_location=lambda storage, loc: storage))
    model = model.eval().cuda()

    if cfg['test']['params'] is None:
        cfg['test']['params'] = {}
        if 'patch' in cfg['evaluation']['params'].keys(): 
            cfg['test']['params']['patch'] = cfg['evaluation']['params']['patch']

    predictor = getattr(factory.evaluate, cfg['test']['predictor'])
    predictor = predictor(loader=test_loader,
        **cfg['test']['params'])

    y_true, y_pred, _ = predictor.predict(model, criterion=None, epoch=None)

    if 'percentile' in cfg['test'].keys():
        y_pred = np.percentile(y_pred, cfg['test']['percentile'], axis=1)

    if not osp.exists(cfg['test']['save_preds_dir']):
        os.makedirs(cfg['test']['save_preds_dir'])

    with open(osp.join(cfg['test']['save_preds_dir'], 'predictions.pkl'), 'wb') as f:
        pickle.dump({
            'y_true': y_true,
            'y_pred': y_pred,
            'imgfiles': [im.split('/')[-1] for im in test_images]
        }, f)


def predict(args, cfg):

    df = pd.read_csv(cfg['predict']['csv_filename'])

    logger = logging.getLogger('root')
    logger.info('PREDICT : START')
    logger.info('PREDICT: n={}'.format(len(df)))

    images = [osp.join(cfg['predict']['data_dir'], _) for _ in df['imgfile']]
    male = list(df['male'].astype('float32'))
    if cfg['predict']['coords']:
        coords = {k : np.asarray(df[k]) for k in ['x1','y1','x2','y2']}
    else:
        coords = None

    loader = builder.build_dataloader(cfg, data_info={'imgfiles': images, 'labels': [0]*len(images), 'male': male, 'coords': coords}, mode='predict')

    model = builder.build_model(cfg['model']['name'], cfg['model']['params'])
    model.load_state_dict(torch.load(cfg['predict']['checkpoint'], map_location=lambda storage, loc: storage))
    model = model.eval().cuda()

    if cfg['predict']['params'] is None:
        cfg['predict']['params'] = {}
        if 'patch' in cfg['evaluation']['params'].keys(): 
            cfg['predict']['params']['patch'] = cfg['evaluation']['params']['patch']

    predictor = getattr(factory.evaluate, cfg['predict']['predictor'])
    predictor = predictor(loader=loader,
        **cfg['predict']['params'])

    _, y_pred, _ = predictor.predict(model, criterion=None, epoch=None)

    if 'percentile' in cfg['predict'].keys() and cfg['predict']['params']['patch']:
        y_pred = np.percentile(y_pred, cfg['predict']['percentile'], axis=1)

    if not osp.exists(cfg['predict']['save_preds_dir']):
        os.makedirs(cfg['predict']['save_preds_dir'])

    with open(osp.join(cfg['predict']['save_preds_dir'], 'predictions.pkl'), 'wb') as f:
        pickle.dump({
            'y_pred': y_pred,
            'imgfiles': [im.split('/')[-1] for im in images]
        }, f)


def predict_ensemble(args, cfg):

    df = pd.read_csv(cfg['predict']['csv_filename'])

    BATCH_SIZE = None
    if 'batch_size' in cfg['predict'].keys():
        BATCH_SIZE = cfg['predict']['batch_size']

    model_cfgs = []
    for cfgfile in cfg['model_configs']:
        with open(cfgfile) as f:
            model_cfgs.append(yaml.load(f, Loader=yaml.FullLoader))


    logger = logging.getLogger('root')
    logger.info('PREDICT : START')
    logger.info('PREDICT: n={}'.format(len(df)))

    images = [osp.join(cfg['predict']['data_dir'], _) for _ in df['imgfile']]
    male = list(df['male'].astype('float32'))
    if cfg['predict']['coords']:
        coords = {k : np.asarray(df[k]) for k in ['x1','y1','x2','y2']}
    else:
        coords = None

    loaders = []
    models  = []
    for model_cfg in model_cfgs:
        model_cfg = set_inference_batch_size(model_cfg)
        if 'predict' not in model_cfg.keys():
            model_cfg['predict'] = copy.deepcopy(model_cfg['test'])
        if BATCH_SIZE:
            model_cfg['predict']['batch_size'] = BATCH_SIZE
        loaders.append(builder.build_dataloader(model_cfg, data_info={'imgfiles': images, 'labels': [0]*len(images), 'male': male, 'coords': coords}, mode='predict'))
        model = builder.build_model(model_cfg['model']['name'], model_cfg['model']['params'])
        model.load_state_dict(torch.load(model_cfg['predict']['checkpoint'], map_location=lambda storage, loc: storage))
        model = model.eval().cuda()
        models.append(model)

    for model_cfg in model_cfgs:
        if model_cfg['predict']['params'] is None:
            model_cfg['predict']['params'] = {}
            if 'patch' in model_cfg['evaluation']['params'].keys(): 
                model_cfg['predict']['params']['patch'] = model_cfg['evaluation']['params']['patch']

    predictors = []
    for ind, model_cfg in enumerate(model_cfgs):
        predictor = getattr(factory.evaluate, model_cfg['predict']['predictor'])
        predictor = predictor(loader=loaders[ind],
            **model_cfg['predict']['params'])
        predictors.append(predictor)

    y_pred_list = []
    for ind, model_cfg in enumerate(model_cfgs):
        _, y_pred, _ = predictors[ind].predict(models[ind], criterion=None, epoch=None)
        if 'percentile' in model_cfg['predict'].keys() and model_cfg['predict']['params']['patch']:
            y_pred = np.percentile(y_pred, model_cfg['predict']['percentile'], axis=1)
        y_pred_list.append(y_pred)

    y_pred = np.mean(np.asarray(y_pred_list), axis=0)

    if not osp.exists(cfg['predict']['save_preds_dir']):
        os.makedirs(cfg['predict']['save_preds_dir'])

    with open(osp.join(cfg['predict']['save_preds_dir'], 'predictions.pkl'), 'wb') as f:
        pickle.dump({
            'y_pred': y_pred,
            'imgfiles': [im.split('/')[-1] for im in images]
        }, f)

if __name__ == '__main__':
    main()












