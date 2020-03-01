# Taken and modified from: 
# https://github.com/davidtvs/pytorch-lr-finder/torch_lr_finder/lr_finder.py

from __future__ import print_function, with_statement, division
import copy
import os
import torch
import pandas as pd
import numpy as np

from collections import OrderedDict
from tqdm import tqdm
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.utils import clip_grad
from ..data import cudaify

global MPL_AVAIL

try:
    import matplotlib.pyplot as plt
    MPL_AVAIL = True
except:
    print('matplotlib is unavailable !')
    MPL_AVAIL = False


class LRFinder(object):
    """Learning rate range test.

    The learning rate range test increases the learning rate in a pre-training run
    between two boundaries in a linear or exponential manner. It provides valuable
    information on how well the network can be trained over a range of learning rates
    and what is the optimal learning rate.


    Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    fastai/lr_find: https://github.com/fastai/fastai

    """

    def __init__(self, 
                 loader,
                 model, 
                 optimizer, 
                 save_checkpoint_dir,
                 logger,
                 gradient_accumulation=1,
                 cuda=True):

        self.loader = loader
        self.generator = self._data_generator()
        self.model = model
        self.optimizer = optimizer
        self.history = {"lr": [], "loss": []}
        self.save_checkpoint_dir = save_checkpoint_dir
        self.logger = logger
        self.print = self.logger.info
        self.gradient_accumulation = gradient_accumulation
        self.cuda = True

    # Wrap data loader in a generator ...
    def _data_generator(self):
        while 1:
            for data in self.loader:
                yield data

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
        batch = self._fetch_data()

        def closure():
            self.optimizer.zero_grad()
            losses = self.model(**batch, return_loss=True)
            loss, log_vars = self.parse_losses(losses)
            loss.backward() 
            if self.grad_clip: self.clip_grads(self.model.parameters())
            return loss 

        loss = self.optimizer.step(closure=closure)

        return loss.item()

    # This version loads in the effective batch size
    # and processes the batch in chunks. 
    # Might be better ...

    def _accumulate_step(self):

        batch, labels = self._fetch_data()
        batch_size = batch.size()[0]
        splits = torch.split(torch.arange(batch_size), int(batch_size/self.gradient_accumulation))

        def closure(): 
            self.optimizer.zero_grad()
            tracker_loss = 0.
            for i in range(int(self.gradient_accumulation)):
                output = self.model(batch[splits[i]])
                loss = self.criterion(output, 
                    {k : v[splits[i]] for k,v in labels.items()})
                tracker_loss += loss.item()
                if i < (self.gradient_accumulation - 1):
                    retain = True
                else:
                    retain = False
                (loss / self.gradient_accumulation).backward()#retain_graph=retain) 

            return tracker_loss

        loss = self.optimizer.step(closure=closure)

       
        return loss.item()

    def train_step(self):
        return self._accumulate_step() if self.gradient_accumulation > 1 else self._step()
    #

    def find_lr(
        self,
        start_lr,
        end_lr,
        num_iter,
        save_fig,
        grad_clip=None,
        step_mode="exp",
        smooth_f=0.05,
        diverge_th=5,
    ):
        """Performs the learning rate range test.

        Arguments:
            train_loader (torch.utils.data.DataLoader): the training set data laoder.
            val_loader (torch.utils.data.DataLoader, optional): if `None` the range test
                will only use the training loss. When given a data loader, the model is
                evaluated after each iteration on that dataset and the evaluation loss
                is used. Note that in this mode the test takes significantly longer but
                generally produces more precise results. Default: None.
            end_lr (float, optional): the maximum learning rate to test. Default: 10.
            num_iter (int, optional): the number of iterations over which the test
                occurs. Default: 100.
            step_mode (str, optional): one of the available learning rate policies,
                linear or exponential ("linear", "exp"). Default: "exp".
            smooth_f (float, optional): the loss smoothing factor within the [0, 1[
                interval. Disabled if set to 0, otherwise the loss is smoothed using
                exponential smoothing. Default: 0.05.
            diverge_th (int, optional): the test is stopped when the loss surpasses the
                threshold:  diverge_th * best_loss. Default: 5.

        """
        self.grad_clip = grad_clip
        # Reset test results
        self.history = {"lr": [], "loss": []}
        self.best_loss = None

        for idx, group in enumerate(self.optimizer.param_groups):
            self.optimizer.param_groups[idx]['lr'] = start_lr

        # Initialize the proper learning rate policy
        if step_mode.lower() == "exp":
            lr_schedule = ExponentialLR(self.optimizer, end_lr, num_iter)
        elif step_mode.lower() == "linear":
            lr_schedule = LinearLR(self.optimizer, end_lr, num_iter)
        else:
            raise ValueError("expected one of (exp, linear), got {}".format(step_mode))

        if smooth_f < 0 or smooth_f >= 1:
            raise ValueError("smooth_f is outside the range [0, 1[")

        # Create an iterator to get data batch by batch
        for iteration in tqdm(range(num_iter)):
            # Get a new set of inputs and labels
            loss = self.train_step()

            # Update the learning rate
            lr_schedule.step()
            self.history["lr"].append(lr_schedule.get_lr()[0])

            # Track the best loss and smooth it if smooth_f is specified
            if iteration == 0:
                self.best_loss = loss
            else:
                if smooth_f > 0:
                    loss = smooth_f * loss + (1 - smooth_f) * self.history["loss"][-1]
                if loss < self.best_loss:
                    self.best_loss = loss

            # Check if the loss has diverged; if it has, stop the test
            self.history["loss"].append(loss)
            if loss > diverge_th * self.best_loss:
                self.print("Stopping early, the loss has diverged ...")
                break

        pd.DataFrame(self.history).to_csv(os.path.join(self.save_checkpoint_dir, 'lrfind.csv'))
        if MPL_AVAIL and save_fig: self.plot(os.path.join(self.save_checkpoint_dir, 'lrfind.png'))

        self.print("Learning rate search finished !")

    def plot(self, save_file, skip_start=10, skip_end=5, log_lr=True, show_lr=None):
        """Plots the learning rate range test.

        Arguments:
            skip_start (int, optional): number of batches to trim from the start.
                Default: 10.
            skip_end (int, optional): number of batches to trim from the start.
                Default: 5.
            log_lr (bool, optional): True to plot the learning rate in a logarithmic
                scale; otherwise, plotted in a linear scale. Default: True.
            show_lr (float, optional): is set, will add vertical line to visualize
                specified learning rate; Default: None
        """

        if skip_start < 0:
            raise ValueError("skip_start cannot be negative")
        if skip_end < 0:
            raise ValueError("skip_end cannot be negative")
        if show_lr is not None and not isinstance(show_lr, float):
            raise ValueError("show_lr must be float")

        # Get the data to plot from the history dictionary. Also, handle skip_end=0
        # properly so the behaviour is the expected
        lrs = self.history["lr"]
        losses = self.history["loss"]
        if skip_end == 0:
            lrs = lrs[skip_start:]
            losses = losses[skip_start:]
        else:
            lrs = lrs[skip_start:-skip_end]
            losses = losses[skip_start:-skip_end]

        # Plot loss as a function of the learning rate
        plt.plot(lrs, losses)
        if log_lr:
            plt.xscale("log")
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")

        if show_lr is not None:
            plt.axvline(x=show_lr, color="red")
        plt.savefig(save_file)


class LinearLR(_LRScheduler):
    """Linearly increases the learning rate between two boundaries over a number of
    iterations.

    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float, optional): the initial learning rate which is the lower
            boundary of the test. Default: 10.
        num_iter (int, optional): the number of iterations over which the test
            occurs. Default: 100.
        last_epoch (int): the index of last epoch. Default: -1.

    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]


class ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations.

    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float, optional): the initial learning rate which is the lower
            boundary of the test. Default: 10.
        num_iter (int, optional): the number of iterations over which the test
            occurs. Default: 100.
        last_epoch (int): the index of last epoch. Default: -1.

    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


class StateCacher(object):
    def __init__(self, in_memory, cache_dir=None):
        self.in_memory = in_memory
        self.cache_dir = cache_dir

        if self.cache_dir is None:
            import tempfile
            self.cache_dir = tempfile.gettempdir()
        else:
            if not os.path.isdir(self.cache_dir):
                raise ValueError('Given `cache_dir` is not a valid directory.')

        self.cached = {}

    def store(self, key, state_dict):
        if self.in_memory:
            self.cached.update({key: copy.deepcopy(state_dict)})
        else:
            fn = os.path.join(self.cache_dir, 'state_{}_{}.pt'.format(key, id(self)))
            self.cached.update({key: fn})
            torch.save(state_dict, fn)

    def retrieve(self, key):
        if key not in self.cached:
            raise KeyError('Target {} was not cached.'.format(key))

        if self.in_memory:
            return self.cached.get(key)
        else:
            fn = self.cached.get(key)
            if not os.path.exists(fn):
                raise RuntimeError('Failed to load state in {}. File does not exist anymore.'.format(fn))
            state_dict = torch.load(fn, map_location=lambda storage, location: storage)
            return state_dict

    def __del__(self):
        """Check whether there are unused cached files existing in `cache_dir` before
        this instance being destroyed."""
        if self.in_memory:
            return

        for k in self.cached:
            if os.path.exists(self.cached[k]):
                os.remove(self.cached[k])
