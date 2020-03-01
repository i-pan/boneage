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

from tqdm import tqdm

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

class Step(object):

    def __init__(self, 
        loader,
        model, 
        optimizer,
        criterion):

        self.loader = loader
        self.generator = self._data_generator()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.gradient_accumulation = 1

        self.loss_tracker = LossTracker(num_moving_average=1000)
        self.time_tracker = TimeTracker(length=100)

    # Wrap data loader in a generator ...
    def _data_generator(self):
        while 1:
            for data in self.loader:
                yield data

    # Move the model forward ...
    def _fetch_data(self): 
        batch, labels, male = next(self.generator)
        if self.cuda:
            batch, labels, male = cudaify(batch, labels, male)
        if self.mixup:
            mixup_start = time.time()
            lam = np.random.beta(self.mixup, self.mixup)
            index = torch.randperm(batch.size()[0])
            batch = lam * batch + (1 - lam) * batch[index]
            if type(labels) == dict: 
                labels_dict = {
                    'y_true1': labels,
                    'y_true2': {k : v[index] for k,v in labels.items()},
                    'lam': lam
                }
            else:
                labels_dict = {'y_true1': labels, 'y_true2': labels[index], 'lam': lam}
            return (batch, labels_dict)
        else:
            return (batch, labels, male)

    # With closure
    def _step(self):
        data_start = time.time()
        batch, labels, male = self._fetch_data()
        data_time = time.time() - data_start

        def closure():
            self.optimizer.zero_grad()
            output = self.model(batch, male)
            loss = self.criterion(output, labels)
            loss.backward() 
            self.loss_tracker.set_loss(loss.item())

        step_start = time.time()
        self.optimizer.step(closure=closure)
        step_time = time.time() - step_start

        self.time_tracker.set_time((data_time, step_time))

    @staticmethod
    def _separate_batch(labels, indices):
        # For gradient accumulation with mixup
        l = {}
        for ko,vo in labels.items():
            l[ko] = {}
            if ko != 'lam':
                for ki,vi in vo.items():
                    l[ko][ki] = vi[indices]
        l['lam'] = labels['lam']
        return l

    def _accumulate_step(self):

        data_start = time.time()
        batch, labels, male = self._fetch_data()
        data_time = time.time() - data_start 
        batch_size = batch.size()[0]
        splits = torch.split(torch.arange(batch_size), int(batch_size/self.gradient_accumulation))

        def closure(): 
            self.optimizer.zero_grad()
            tracker_loss = 0.
            for i in range(int(self.gradient_accumulation)):
                step_start = time.time() 
                output = self.model(batch[splits[i]], male[splits[i]])
                if self.mixup:
                    loss = self.criterion(output, 
                        self._separate_batch(labels, splits[i]))                    
                else:
                    loss = self.criterion(output, labels[splits[i]])
                tracker_loss += loss.item()
                if i < (self.gradient_accumulation - 1):
                    retain = True
                else:
                    retain = False
                (loss / self.gradient_accumulation).backward()#retain_graph=retain) 
            self.loss_tracker.set_loss(tracker_loss / self.gradient_accumulation)

        step_start = time.time()
        self.optimizer.step(closure=closure)
        step_time = time.time() - step_start

        self.time_tracker.set_time((data_time, step_time))          

    def train_step(self):
        self._accumulate_step() if self.gradient_accumulation > 1 else self._step()

class Trainer(Step):

    def __init__(self, 
        loader,
        model, 
        optimizer,
        schedule, 
        criterion, 
        evaluator,
        logger):

        super(Trainer, self).__init__(
            loader=loader, 
            model=model, 
            optimizer=optimizer, 
            criterion=criterion
        )

        self.scheduler = schedule
        self.evaluator = evaluator

        self.logger = logger
        self.print = self.logger.info
        self.evaluator.set_logger(self.logger)
        
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

    def print_progress(self):
        self.print('epoch {epoch}, batch {batch}/{steps_per_epoch}: loss={train_loss:.4f} (data: {load_time:.3f}s/batch, step: {step_time:.3f}s/batch, lr: {learning_rate:.1e})'
                .format(epoch=str(self.current_epoch).zfill(len(str(self.num_epochs))), \
                        batch=str(self.steps).zfill(len(str(self.steps_per_epoch))), \
                        steps_per_epoch=self.steps_per_epoch, \
                        train_loss=self.loss_tracker.get_loss(), \
                        load_time=self.time_tracker.get_time()[0],
                        step_time=self.time_tracker.get_time()[1],
                        learning_rate=self.optimizer.param_groups[0]['lr']))

    def init_training(self, 
                      gradient_accumulation, 
                      num_epochs,
                      steps_per_epoch,
                      validate_interval,
                      mixup,
                      cuda):

        self.gradient_accumulation = float(gradient_accumulation)
        self.num_epochs = num_epochs
        self.steps_per_epoch = len(self.loader) if steps_per_epoch == 0 else steps_per_epoch
        self.validate_interval = validate_interval
        self.mixup = mixup
        self.cuda = True

        self.steps = 0 
        self.current_epoch = 0

        self.optimizer.zero_grad()

    def train(self, 
              gradient_accumulation,
              num_epochs, 
              steps_per_epoch, 
              validate_interval,
              verbosity=100,
              mixup=None,
              cuda=True): 
        # Epochs are 0-indexed
        self.init_training(gradient_accumulation, num_epochs, steps_per_epoch, validate_interval, mixup, cuda)
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
                    self.criterion, 
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








