from torch.utils.data import DataLoader

import re
import os.path as osp

from pathlib import Path
from functools import partial

import factory.data.datasets as datasets
import factory.data.transforms as transforms 
import factory.optim as optim
import factory.train.scheduler as scheduler

from mmdet.models.builder import build_detector
from mmdet.datasets.loader import build_dataloader as _build_dataloader
from mmcv.utils.config import Config, ConfigDict
from mmcv.parallel import MMDataParallel

_PATH = Path(__file__).parent

def build(lib, name, params):
    '''
    Generic build function.
    '''
    return getattr(lib, name)(**params) if params is not None else getattr(lib, name)()

def build_model(cfg, gpu):
    '''
    Return model wrapped in MMDataParallel.
    TODO: support multi-GPU
    '''
    if type(gpu) != list: gpu = [gpu]

    model_type = cfg['model'].pop('config')
    train_cfg = cfg['model'].pop('train_cfg')
    test_cfg = cfg['model'].pop('test_cfg')

    mmdet_cfg = Config.fromfile(osp.join(_PATH, 'models', '{}.py'.format(model_type)))

    # Change model parameters
    if cfg['model'] is not None:
        if type(cfg['model']['bbox_head']) == list:
            assert len(cfg['model']['bbox_head']) == len(mmdet_cfg.model.bbox_head)
            bbox_head_cfg = cfg['model'].pop('bbox_head')
            for ind, i in enumerate(bbox_head_cfg):
                if i is None: continue
                assert type(i) is dict
                assert type(mmdet_cfg.model.bbox_head[ind]) is ConfigDict
                mmdet_cfg.model.bbox_head[ind].update(i)
        mmdet_cfg.model.update(cfg['model'])
    if train_cfg is not None:
        if type(train_cfg['rcnn']) == list:
            assert len(train_cfg['rcnn']) == len(mmdet_cfg.train_cfg.rcnn)
            rcnn_cfg = train_cfg.pop('rcnn')
            for ind, i in enumerate(rcnn_cfg):
                if i is None: continue
                assert type(i) is dict
                assert type(mmdet_cfg.train_cfg.rcnn[ind]) is ConfigDict
                mmdet_cfg.train_cfg.rcnn[ind].update(i)
        mmdet_cfg.train_cfg.update(train_cfg) 
    if test_cfg is not None:
        mmdet_cfg.test_cfg.update(test_cfg) 

    model = build_detector(mmdet_cfg.model, 
        train_cfg=mmdet_cfg.train_cfg, 
        test_cfg=mmdet_cfg.test_cfg)
    model = MMDataParallel(model, device_ids=gpu)
    return model

def build_optimizer(name, model, params):
    if hasattr(model, 'module'):
        model = model.module
    return getattr(optim, name)(params=model.parameters(), **params)

def build_scheduler(name, optimizer, cfg):
    # loader is only needed for OnecycleLR
    # So that I can specify steps_per_epoch = 0 for convenience
    params = cfg['scheduler']['params']

    # Some schedulers will require manipulation of config params
    # My specifications were to make it more intuitive for me
    if name == 'CosineAnnealingWarmRestarts':
        params = {
            # Use num_epochs from training parameters
            'T_0': int(cfg['train']['params']['num_epochs'] / params['num_snapshots']),
            'eta_min': params['final_lr']
        }

    if name in ('OneCycleLR', 'CustomOneCycleLR'):
        # Use steps_per_epoch from training parameters
        steps_per_epoch = cfg['train']['params']['steps_per_epoch']
        params['steps_per_epoch'] = steps_per_epoch
        # And num_epochs
        params['epochs'] = cfg['train']['params']['num_epochs']
        # Use learning rate from optimizer parameters as initial learning rate
        init_lr  = cfg['optimizer']['params']['lr']
        final_lr = params.pop('final_lr')
        params['div_factor'] = params['max_lr'] / init_lr
        params['final_div_factor'] = init_lr / final_lr

    schedule = getattr(scheduler, name)(optimizer=optimizer, **params)
    
    # Some schedulers might need more manipulation after instantiation
    if name == 'CosineAnnealingWarmRestarts':
        schedule.T_cur = 0

    # Set update frequency
    if name in ('CosineAnnealingWarmRestarts', 'OneCycleLR', 'CustomOneCycleLR'):
        schedule.update = 'on_batch'
    elif name in ('ReduceLROnPlateau'):
        schedule.update = 'on_valid'
    else:
        schedule.update = 'on_epoch'

    return schedule

def build_dataset(cfg, ann, mode):

    if mode in ('train','valid'):
        if mode == 'train':
            if 'concat' not in cfg['transform'].keys(): 
                print ('`concat` not specified. Setting `concat=True` as default ...')
                cfg_concat = True
            else:
                cfg_concat = cfg['transform']['concat']
        elif mode == 'valid':
            cfg_concat = False
        # Data pipelines for training should have a `valid` mode
        pipeline = getattr(transforms, cfg['transform']['augment']['train'])
        pipeline = pipeline(resize_to=cfg['transform']['resize_to'],
                            mode=mode,
                            concat=cfg_concat,
                            **cfg['transform']['preprocess'])
    else:
        pipeline = getattr(transforms, cfg['transform']['augment']['infer'])
        pipeline = pipeline(resize_to=cfg['transform']['resize_to'],
                            **cfg['transform']['preprocess'])

    #TODO: need some way of handling TTA
    # if mode == 'test' and cfg['test']['tta']:
    #   ...

    dset_params = cfg['dataset']['params'] if cfg['dataset']['params'] is not None else {}

    dset = getattr(datasets, cfg['dataset']['name'])(
        annotations=ann,
        pipeline=pipeline,
        data_root=cfg['dataset']['data_dir'],
        test_mode=mode in ('test', 'predict'), 
        filter_empty=mode == 'train',
        **dset_params)

    return dset

def build_dataloader(cfg, ann, mode):

    dset = build_dataset(cfg, ann, mode)
    
    if mode == 'train': 
        loader = _build_dataloader(dset, 
                                   imgs_per_gpu=cfg['train']['batch_size'], 
                                   workers_per_gpu=cfg['transform']['num_workers'],
                                   dist=False)
    elif mode == 'valid':
        # We don't really use DataLoader during validation
        loader = _build_dataloader(dset, 
                                   imgs_per_gpu=1,
                                   workers_per_gpu=cfg['transform']['num_workers'],
                                   shuffle=False, 
                                   dist=False)
    elif mode == 'test':
        loader = _build_dataloader(dset, 
                                   imgs_per_gpu=1, 
                                   workers_per_gpu=cfg['transform']['num_workers'],
                                   shuffle=False,
                                   dist=False)

    return loader