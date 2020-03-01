import torch
import pandas as pd
import numpy as np
import os, os.path as osp

from tqdm import tqdm

from .metrics import *

from ..data import cudaify


def grid_patches(img, 
                 patch_size=224, 
                 num_rows=7, 
                 num_cols=7, 
                 return_coords=False):
    """
    Generates <num_rows> * <num_cols> patches from an image. 
    Centers of patches gridded evenly length-/width-wise. 
    """
    # This typically doesn't happen, but if one of your original image 
    # dimensions is smaller than the patch size, the image will be resized
    # (aspect ratio maintained) such that the smaller dimension is equal
    # to the patch size. (Maybe it should be padded instead?)
    if np.min(img.size()[:2]) < patch_size:
        resize_factor = patch_size / float(np.min(img.shape[:2]))
        new_h = int(np.round(resize_factor*img.shape[0]))
        new_w = int(np.round(resize_factor*img.shape[1]))
        img = scipy.misc.imresize(img, (new_h, new_w))
    row_start = int(patch_size / 2)
    row_end = int(img.shape[0] - patch_size / 2)
    col_start = int(patch_size / 2)
    col_end = int(img.shape[1] - patch_size / 2)
    row_inc = int((row_end - row_start) / (num_rows - 1))
    col_inc = int((col_end - col_start) / (num_cols - 1))
    if row_inc == 0: row_inc = 1
    if col_inc == 0: col_inc = 1  
    patch_list = [] 
    coord_list = [] 
    for i in range(row_start, row_end+1, row_inc):
        for j in range(col_start, col_end+1, col_inc):
            x0 = int(i-patch_size/2) ; x1 = int(i+patch_size/2)
            y0 = int(j-patch_size/2) ; y1 = int(j+patch_size/2)
            patch = img[x0:x1, y0:y1]
            assert patch.shape[:2] == (patch_size, patch_size)
            patch_list.append(patch)
            coord_list.append([x0,x1,y0,y1])
    if return_coords:
        return patch_list, coord_list
    else:
        return patch_list 

class Predictor(object):

    def __init__(self,
                 loader,
                 patch=False,
                 labels_available=True,
                 cuda=True):

        self.loader = loader
        self.cuda = cuda
        if patch:
            self.predict = self.predict_patch
            # Patches should be square ...
            self.patch_size = self.loader.dataset.crop[0].height
            self.loader.dataset.crop = None
        else:
            self.predict = self.predict_whole
        self.labels_available = labels_available


    def predict_patch(self, model, criterion, epoch):
        self.epoch = epoch
        y_pred = []
        y_true = []   
        with torch.no_grad():
            losses = []
            for data in tqdm(self.loader, total=len(self.loader)):
                batch, labels, male = data
                if self.cuda: 
                    batch, labels, male = cudaify(batch, labels, male)
                assert batch.shape[0] == 1
                # batch.shape = (1, C, H, W)
                batch = batch[0]
                batch = batch.transpose(0,2).transpose(0,1)
                patches = torch.stack(grid_patches(batch, patch_size=self.patch_size))
                patches = patches.transpose(1,3).transpose(2,3)
                output = model(patches, male.repeat(len(patches)))
                if criterion:
                    losses.append(criterion(output, labels.repeat(len(output))).item())
                y_pred.append(output.unsqueeze(0).cpu().numpy())
                if self.labels_available:
                    y_true.extend(labels.cpu().numpy())
        y_pred = np.vstack(y_pred) * self.loader.dataset.scale
        if self.labels_available:
            y_true = np.asarray(y_true) * self.loader.dataset.scale   
        return y_true, y_pred, losses


    def predict_whole(self, model, criterion, epoch):
        self.epoch = epoch
        y_pred = []
        y_true = []   
        with torch.no_grad():
            losses = []
            for data in tqdm(self.loader, total=len(self.loader)):
                batch, labels, male = data
                if self.cuda: 
                    batch, labels, male = cudaify(batch, labels, male)
                output = model(batch, male)
                if criterion:
                    losses.append(criterion(output, labels).item())
                y_pred.extend(output.cpu().numpy())
                if self.labels_available:
                    y_true.extend(labels.cpu().numpy())
        y_pred = np.asarray(y_pred) * self.loader.dataset.scale
        if self.labels_available:
            y_true = np.asarray(y_true) * self.loader.dataset.scale   
        return y_true, y_pred, losses

class Evaluator(Predictor):

    def __init__(self,
                 loader,
                 metrics,
                 valid_metric,
                 mode,
                 improve_thresh,
                 prefix,
                 save_checkpoint_dir,
                 save_best,
                 early_stopping=np.inf,
                 patch=False,
                 cuda=True):
        
        super(Evaluator, self).__init__(loader=loader, cuda=cuda, patch=patch)

        if type(metrics) is not list: metrics = list(metrics)
        assert valid_metric in metrics or valid_metric == 'loss'

        self.loader = loader
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

        self.history = []
        self.epoch = None

        self.reset_best()

    def reset_best(self):
        self.best_model = None
        self.best_score = -np.inf

    def set_logger(self, logger):
        self.logger = logger
        self.print  = self.logger.info

    def validate(self, model, criterion, epoch):
        y_true, y_pred, losses = self.predict(model, criterion, epoch)
        valid_metric = self.calculate_metrics(y_true, y_pred, losses)
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
                self.best_score = -valid_metric if self.mode == 'min' else valid_metric
                torch.save(model.state_dict(), save_file)
        else:
            torch.save(model.state_dict(), save_file)

    def calculate_metrics(self, y_true, y_pred, losses):
        metrics_dict = {}
        metrics_dict['loss'] = np.mean(losses)
        for metric in self.metrics:
            metric = eval(metric)
            metrics_dict.update(metric(y_true, y_pred))
        print_results = 'epoch {epoch} // VALIDATION'.format(epoch=self.epoch)
        max_str_len = np.max([len(k) for k in metrics_dict.keys()])
        for key in metrics_dict.keys():
            self.print('{key} | {value:.5g}'.format(key=key.ljust(max_str_len), value=metrics_dict[key]))
        valid_metric = metrics_dict[self.valid_metric]
        metrics_dict.update({'vm': valid_metric, 'epoch': int(self.epoch)})
        self.history.append(metrics_dict)
        self.generate_metrics_df()
        return valid_metric


