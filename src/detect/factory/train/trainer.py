import torch
import torch.nn as nn
import torch.nn.functional as F

import datetime
import time
import logging
import pandas as pd
import numpy as np
import os

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from collections import OrderedDict
from tqdm import tqdm
from torch.nn.utils import clip_grad

from ..data import cudaify

class TimeTracker(object):

    def __init__(self, length=100):
        self.length = length
        self.load_time = []
        self.step_time = []

    def set_time(self, t):
        self.load_time.append(t[0])
        self.step_time.append(t[1])

    def get_time(self):
        return (np.mean(self.load_time[-int(self.length):]),
                np.mean(self.step_time[-int(self.length):]))

class LossTracker(object): 

    def __init__(self, num_moving_average=1000): 
        self.losses = []
        self.history = []
        self.avg = num_moving_average

    def set_loss(self, minibatch_loss): 
        self.losses.append(minibatch_loss) 

    def get_loss(self): 
        self.history.append(np.mean(self.losses[-self.avg:]))
        return self.history[-1]

    def reset(self): 
        self.losses = [] 

    def get_history(self): 
        return self.history

class MultiLossTracker(object): 

    def __init__(self, num_moving_average=1000): 
        self.losses = []
        self.history = []
        self.avg = num_moving_average

    def set_loss(self, minibatch_loss): 
        self.losses.append(minibatch_loss) 

    def get_loss(self): 
        # Each element in self.losses is an OrderedDict
        # Take the most recent N losses
        recent_losses = self.losses[-self.avg:]
        # Set up an initial dict where all loss values are 0
        loss_dict = {loss_name : 0 for loss_name in recent_losses[-1].keys()}
        for losses in recent_losses:
            for loss_name, loss_item in losses.items():
                # Start accumulating the loss values
                loss_dict[loss_name] += loss_item / len(recent_losses)
        self.history.append(loss_dict)
        return self.history[-1]

    def reset(self): 
        self.losses = [] 

    def get_history(self): 
        return self.history

class Trainer(object):

    def __init__(self, 
        loader,
        model, 
        optimizer,
        schedule, 
        evaluator,
        logger):

        self.model = model 
        self.loader = loader
        self.generator = self._data_generator()
        self.optimizer = optimizer
        self.scheduler = schedule
        self.evaluator = evaluator

        self.logger = logger
        self.print = self.logger.info
        self.evaluator.set_logger(self.logger)

        self.loss_tracker = LossTracker(num_moving_average=1000)
        self.multiloss_tracker = MultiLossTracker(num_moving_average=1000)
        self.time_tracker = TimeTracker(length=100)

        #self.logfile = os.path.join(self.evaluator.save_checkpoint_dir, 'log.txt')
        #self.init_logger()

    # def init_logger(self):
    #     if os.path.exists(self.logfile): os.system('rm {}'.format(self.logfile))
    #     logging.basicConfig(level=logging.INFO, format='%(message)s')
    #     self.logger = logging.getLogger()
    #     self.logger.addHandler(logging.FileHandler(self.logfile, 'a'))
    #     self.print = self.logger.info
    #     self.evaluator.set_logger(self.logger)

    def check_end_train(self): 
        return self.current_epoch >= self.num_epochs

    def check_end_epoch(self):
        return (self.steps % self.steps_per_epoch) == 0 and (self.steps > 0)

    def check_validation(self):
        # We add 1 to current_epoch when checking whether to validate
        # because epochs are 0-indexed. E.g., if validate_interval is 2,
        # we should validate after epoch 1. We need to add 1 so the modulo
        # returns 0
        return self.check_end_epoch() and self.steps > 0 and ((self.current_epoch + 1) % self.validate_interval) == 0

    def scheduler_step(self):
        if isinstance(self.scheduler, CosineAnnealingWarmRestarts):
            self.scheduler.step(self.current_epoch + self.steps * 1./self.steps_per_epoch)
        else:
            self.scheduler.step()

    # Wrap data loader in a generator ...
    def _data_generator(self):
        while 1:
            for data in self.loader:
                yield data

    # Move the model forward ...
    def _fetch_data(self): 
        batch = next(self.generator)
        return batch

    def clip_grads(self, params):
        clip_grad.clip_grad_norm_(
            filter(lambda p: p.requires_grad, params), **self.grad_clip)

    @staticmethod
    def parse_losses(losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    '{} is not a tensor or list of tensors'.format(loss_name))

        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

        log_vars['loss'] = loss
        for name in log_vars:
            log_vars[name] = log_vars[name].item()

        return loss, log_vars

    # With closure
    def _step(self):
        data_start = time.time()
        batch = self._fetch_data()
        data_time = time.time() - data_start

        def closure():
            self.optimizer.zero_grad()
            losses = self.model(**batch, return_loss=True)
            loss, log_vars = self.parse_losses(losses)
            loss.backward() 
            if self.grad_clip: self.clip_grads(self.model.parameters())
            self.loss_tracker.set_loss(loss.item())
            self.multiloss_tracker.set_loss(log_vars)
            return loss 

        step_start = time.time()
        loss = self.optimizer.step(closure=closure)
        step_time = time.time() - step_start

        self.time_tracker.set_time((data_time, step_time))

    # This version loads in the effective batch size
    # and processes the batch in chunks. 
    # Might be better ...

    def _accumulate_step(self):

        data_start = time.time()
        batch, labels = self._fetch_data()
        data_time = time.time() - data_start 
        batch_size = batch.size()[0]
        splits = torch.split(torch.arange(batch_size), int(batch_size/self.gradient_accumulation))

        def closure(): 
            self.optimizer.zero_grad()
            tracker_loss = 0.
            for i in range(int(self.gradient_accumulation)):
                step_start = time.time() 
                output = self.model(batch[splits[i]])
                loss = self.criterion(output, 
                    {k : v[splits[i]] for k,v in labels.items()})
                tracker_loss += loss.item()
                if i < (self.gradient_accumulation - 1):
                    retain = True
                else:
                    retain = False
                (loss / self.gradient_accumulation).backward()#retain_graph=retain) 
                if self.grad_clip: self.clip_grads(self.model.parameters())
            self.loss_tracker.set_loss(tracker_loss / self.gradient_accumulation)
            return tracker_loss

        step_start = time.time()
        loss = self.optimizer.step(closure=closure)
        step_time = time.time() - step_start

        self.time_tracker.set_time((data_time, step_time))        
       

    def train_step(self):
        self._accumulate_step() if self.gradient_accumulation > 1 else self._step()
    #

    def print_progress(self):
        statement = 'epoch {epoch}, batch {batch}/{steps_per_epoch}: loss={train_loss:.4f}'.format(
                     epoch=str(self.current_epoch).zfill(len(str(self.num_epochs))), \
                     batch=str(self.steps).zfill(len(str(self.steps_per_epoch))), \
                     steps_per_epoch=self.steps_per_epoch, \
                     train_loss=self.loss_tracker.get_loss())
        multiloss = self.multiloss_tracker.get_loss()
        _loss = multiloss.pop('loss')
        multiloss_string = ''
        for loss_name, loss_val in multiloss.items():
            statement += ' {}={:.4f}'.format(loss_name, loss_val)
        statement += ' (data: {load_time:.3f}s/batch, step: {step_time:.3f}s/batch, lr: {learning_rate:.1e})'.format(load_time=self.time_tracker.get_time()[0],
                     step_time=self.time_tracker.get_time()[1],
                     learning_rate=self.optimizer.param_groups[0]['lr'])
        self.print(statement)

    def init_training(self, 
                      gradient_accumulation, 
                      num_epochs,
                      steps_per_epoch,
                      validate_interval,
                      grad_clip,
                      cuda):

        self.gradient_accumulation = float(gradient_accumulation)
        self.num_epochs = num_epochs
        self.steps_per_epoch = len(self.loader) if steps_per_epoch == 0 else steps_per_epoch
        self.validate_interval = validate_interval
        self.grad_clip = grad_clip
        self.cuda = True

        self.steps = 0 
        self.current_epoch = 0

        self.optimizer.zero_grad()

    def train(self, 
              gradient_accumulation,
              num_epochs, 
              steps_per_epoch, 
              validate_interval,
              grad_clip=None,
              verbosity=100,
              cuda=True): 
        # Epochs are 0-indexed
        self.init_training(gradient_accumulation, num_epochs, steps_per_epoch, validate_interval, grad_clip, cuda)
        start_time = datetime.datetime.now()
        while 1: 
            self.train_step()
            self.steps += 1
            if self.scheduler.update == 'on_batch':
                 self.scheduler_step()
            # Check- print training progress
            if self.steps % verbosity == 0 and self.steps > 0:
                self.print_progress()
            # Check- run validation
            if self.check_validation():
                self.print('VALIDATING ...')
                validation_start_time = datetime.datetime.now()
                # Start validation
                self.model.eval()
                valid_metric = self.evaluator.validate(self.model, 
                    str(self.current_epoch).zfill(len(str(self.num_epochs))))
                if self.scheduler.update == 'on_valid':
                    self.scheduler.step(valid_metric)
                # End validation
                self.model.train()
                self.print('Validation took {} !'.format(datetime.datetime.now() - validation_start_time))
            # Check- end of epoch
            if self.check_end_epoch():
                if self.scheduler.update == 'on_epoch':
                    self.scheduler.step()
                self.current_epoch += 1
                self.steps = 0
                # RESET BEST MODEL IF USING COSINEANNEALINGWARMRESTARTS
                if isinstance(self.scheduler, CosineAnnealingWarmRestarts):
                    if self.current_epoch % self.scheduler.T_0 == 0:
                        self.evaluator.reset_best()
            #
            if self.evaluator.check_stopping(): 
                # Make sure to set number of epochs to max epochs
                # Remember, epochs are 0-indexed and we added 1 already
                # So, this should work (e.g., epoch 99 would now be epoch 100,
                # thus training would stop after epoch 99 if num_epochs = 100)
                self.current_epoch = num_epochs
            if self.check_end_train():
                # Break the while loop
                break
        self.print('TRAINING : END') 
        self.print('Training took {}\n'.format(datetime.datetime.now() - start_time))








