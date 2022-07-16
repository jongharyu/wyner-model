import argparse
import datetime
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from tempfile import mkdtemp

import torch
from numpy import inf


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary/41274937#41274937
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def get_running_path(config, model_name):
    assert config.experiment, 'experiment identifier must be specified'
    run_id = datetime.datetime.now().isoformat()
    experiment_dir = Path('{}/experiments/{}/{}'.format(
        config.main_path,
        model_name,
        config.experiment
    ))
    experiment_dir.mkdir(parents=True, exist_ok=True if config.temp or config.overwrite else False)
    run_path = str(experiment_dir)
    if config.temp:
        run_path = mkdtemp(prefix=run_id, dir=run_path)

    return run_id, run_path


def save_config(config, run_path):
    with open('{}/config.json'.format(run_path), 'w') as fp:
        json.dump(config.__dict__, fp)
    # -- also save object because we want to recover these for other things
    torch.save(config, '{}/config.rar'.format(run_path))
    print(config)


def save_vars(vs, filepath, safe=True):
    """
    Saves variables to the given filepath in a safe manner.
    """
    if os.path.exists(filepath) and safe:
        shutil.copyfile(filepath, '{}.old'.format(filepath))
    torch.save(vs, filepath)


def save_model(model, filepath, safe=True):
    """
    To load a saved model, simply use
        model = TheModelClass(*args, **kwrags
        model.load_state_dict(torch.load('path-to-saved-model'))
    Note: This function can be also used to save a optimizer.
    """
    save_vars(model.state_dict(), filepath, safe=safe)


# Classes
class Constants(object):
    eta = 1e-6
    log2 = math.log(2)
    log2pi = math.log(2 * math.pi)
    logceilc = 88  # largest cuda v s.t. exp(v) < inf
    logfloorc = -104  # smallest cuda v s.t. exp(v) > 0


# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
class Logger(object):
    def __init__(self, filename, mode="a"):
        self.terminal = sys.stdout
        self.log = open(filename, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        self.log.flush()


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.begin = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.begin
        self.elapsedH = time.gmtime(self.elapsed)
        print('====> [{}] Time: {:7.3f}s or {}\n'
              .format(self.name,
                      self.elapsed,
                      time.strftime("%H:%M:%S", self.elapsedH)))


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.

    - Reference: https://github.com/Bjarten/early-stopping-pytorch/blob/7ec86aa946468877bd74427f183d7d68a3eb2dc9/pytorchtools.py
    """

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', save=True):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = inf
        self.delta = delta
        self.path = path
        self.save = save

    def __call__(self, val_loss, model, agg):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, agg)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, agg)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, agg):
        """
        Saves model when validation loss decrease.
        """
        if self.save:
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            save_model(model, self.path + '/model.rar')
            save_vars(agg, self.path + '/losses.rar')
            # torch.save(model.state_dict(), self.path)  # original code
        self.val_loss_min = val_loss


# an alternative for torch.repeat_interleave
# it is also recommended not to use repeat_interleave for reproducibility (?)
# Reference: https://yinwenpeng.wordpress.com/2020/11/12/inconsistent-performance-in-pytorch/
def repeat_interleave(a, dim, n_tile):
    # example: (1, 2, 3) with n_tile=2 -> (1, 1, 2, 2, 3, 3)
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.cat([init_dim * torch.arange(n_tile, device=a.device) + i for i in range(init_dim)])
    if a.device.type == 'cuda':
        order_index = torch.cuda.LongTensor(order_index)
    else:
        order_index = torch.LongTensor(order_index)
    return torch.index_select(a, dim, order_index)