import torch
import pandas as pd
import numpy as np
import os, os.path as osp

from tqdm import tqdm

from mmcv.parallel import scatter, collate 

from .metrics import *

from ..data import cudaify


class Predictor(object):

    def __init__(self, 
                 dataset, 
                 predict_mode=False):

        self.dataset = dataset
        self.predict_mode = predict_mode

    def predict(self, model, epoch):
        self.epoch = epoch

        y_true = []
        y_pred = []
        names  = []

        with torch.no_grad():
            for ind in tqdm(range(len(self.dataset)), total=len(self.dataset)):
                # Get data
                data = self.dataset[ind]
                # Wrap img, img_meta in list 
                # Not sure why I have to do this ...
                if type(data['img']) != list and type(data['img_meta']) != list:
                    data['img'] = [data['img']]
                    data['img_meta'] = [data['img_meta']]
                data_gpu = collate([data], samples_per_gpu=1)
                if not self.predict_mode:
                    # Get annotations
                    ann = self.dataset.get_ann_info(ind)
                    bboxes = ann['bboxes']
                    labels = ann['labels']
                    y_true.append({'bboxes': bboxes, 'labels': labels})
                names.append(self.dataset.img_infos[ind]['filename'])
                # We can alter NMS params using model.module.test_cfg
                # If we want to tune thresholds/NMS thresholds
                ##
                # Get model output
                output = model(**data_gpu, return_loss=False, rescale=True)
                # output is a list with length = num_classes - 1 
                # Each element in output corresponds to a list of predicted
                # boxes for that class
                y_pred.append(output)

        return y_true, y_pred, names

class Evaluator(Predictor):

    def __init__(self,
                 dataset,
                 metrics,
                 valid_metric,
                 mode,
                 improve_thresh,
                 prefix,
                 save_checkpoint_dir,
                 save_best,
                 early_stopping=np.inf,
                 class_thresholds=np.arange(0.05, 0.95, 0.05),
                 iou_thresholds=np.arange(0.5, 0.95, 0.05)):
        
        super(Evaluator, self).__init__(dataset=dataset, predict_mode=False)

        if type(metrics) is not list: metrics = list(metrics)
        assert valid_metric in metrics

        self.dataset = dataset
        # List of strings corresponding to desired metrics
        # These strings should correspond to function names defined
        # in metrics.py
        self.metrics = metrics
        # valid_metric should be included within metrics
        # This specifies which metric we should track for validation improvement
        self.valid_metric = valid_metric
        # Mode should be one of ['min', 'max']
        # This determines whether a lower (min) or higher (max) 
        # valid_metric is considered to be better
        self.mode = mode
        # This determines by how much the valid_metric needs to improve
        # to be considered an improvement
        self.improve_thresh = improve_thresh
        # Specifies part of the model name
        self.prefix = prefix
        self.save_checkpoint_dir = save_checkpoint_dir
        # save_best = True, overwrite checkpoints if score improves
        # If False, save all checkpoints
        self.save_best = save_best
        self.metrics_file = os.path.join(save_checkpoint_dir, 'metrics.csv')
        if os.path.exists(self.metrics_file): os.system('rm {}'.format(self.metrics_file))
        # How many epochs of no improvement do we wait before stopping training?
        self.early_stopping = early_stopping
        self.stopping = 0
        self.class_thresholds = class_thresholds
        self.iou_thresholds = iou_thresholds

        self.history = []
        self.epoch = None

        self.reset_best()

    def reset_best(self):
        self.best_model = None
        self.best_score = -np.inf

    def set_logger(self, logger):
        self.logger = logger
        self.print  = self.logger.info

    def validate(self, model, epoch):
        y_true, y_pred, _ = self.predict(model, epoch)
        valid_metric = self.calculate_metrics(y_true, y_pred)
        self.save_checkpoint(model, valid_metric)
        return valid_metric

    def generate_metrics_df(self):
        df = pd.concat([pd.DataFrame(d, index=[0]) for d in self.history])
        df.to_csv(self.metrics_file, index=False)

    # Used by Trainer class
    def check_stopping(self):
        return self.stopping >= self.early_stopping

    def check_improvement(self, score):
        # If mode is 'min', make score negative
        # Then, higher score is better (i.e., -0.01 > -0.02)
        score = -score if self.mode == 'min' else score
        improved = score >= (self.best_score + self.improve_thresh)
        if improved:
            self.stopping = 0
        else:
            self.stopping += 1
        return improved

    def save_checkpoint(self, model, valid_metric):
        save_file = '{}_{}_VM-{:.4f}.pth'.format(self.prefix, str(self.epoch).zfill(3), valid_metric).upper()
        save_file = os.path.join(self.save_checkpoint_dir, save_file)
        if self.save_best:
            if self.check_improvement(valid_metric):
                if self.best_model is not None: 
                    os.system('rm {}'.format(self.best_model))
                self.best_model = save_file
                self.best_score = valid_metric
                torch.save(model.state_dict(), save_file)
        else:
            torch.save(model.state_dict(), save_file)

    def calculate_metrics(self, y_true, y_pred):
        metrics_dict = {}
        for metric in self.metrics:
            metric = eval(metric)
            metrics_dict.update(metric(y_true, y_pred, 
                class_thresholds=self.class_thresholds, 
                iou_thresholds=self.iou_thresholds))        
        print_results = 'epoch {epoch} // VALIDATION'.format(epoch=self.epoch)
        max_str_len = np.max([len(k) for k in metrics_dict.keys()])
        for key in metrics_dict.keys():
            self.print('{key} | {value:.5g}'.format(key=key.ljust(max_str_len), value=metrics_dict[key]))
        valid_metric = metrics_dict[self.valid_metric]
        metrics_dict.update({'vm': valid_metric, 'epoch': int(self.epoch)})
        self.history.append(metrics_dict)
        self.generate_metrics_df()
        return valid_metric


