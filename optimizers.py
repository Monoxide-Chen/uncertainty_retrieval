import math

import torch
from torch.optim import Adam, SGD, RAdam
from torch.optim.lr_scheduler import _LRScheduler, StepLR, MultiStepLR, LinearLR, SequentialLR

class CAWRwithWarmup(_LRScheduler):
    """Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart. Default: 1.
        T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 2.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        start_factor (float): The number we multiply learning rate in the first epoch.
            The multiplication factor changes towards end_factor in the following epochs.
            Default: 0.1.
        end_factor (float): The number we multiply learning rate at the end of linear changing
            process. Default: 1.0.
        warmup_iters (int): The number of iterations that multiplicative factor reaches to 1.
            Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """
    def __init__(self, optimizer, T_0=1, T_mult=2, eta_min=0, warmup_iters=5,
                start_factor=0.1, end_factor=1.0, last_epoch=-1, verbose=False):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        self.warmup_iters = warmup_iters
        self.start_factor = start_factor
        self.end_factor = end_factor
        super(CAWRwithWarmup, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0:
            return [base_lr * self.start_factor for base_lr in self.base_lrs]

        if self.last_epoch >= self.warmup_iters:
            return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]
        else:
            return [group['lr'] * (1. + (self.end_factor - self.start_factor) /
                    (self.warmup_iters * self.start_factor + (self.last_epoch - 1) * (self.end_factor - self.start_factor)))
                    for group in self.optimizer.param_groups]
    def step(self, epoch=None):
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            if epoch >= self.warmup_iters:
                self.T_cur = self.T_cur + 1
                if self.T_cur >= self.T_i:
                    self.T_cur = self.T_cur - self.T_i
                    self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch > self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group['lr'] = lr
                self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


def create_optimizers(models, config):
    optimizer = config['optimizer']
    lr = config['lr']
    weight_decay = config['weight_decay']
    momentum = config['momentum']

    parameterized_models = {key: model for key, model in models.items() if len(list(model.parameters())) > 0}

    lr_dict = {key: lr for key in parameterized_models}
    assert 'text_encoder' and 'lower_image_encoder' and 'upper_image_encoder' in parameterized_models
    if config['text_encoder'] == 'roberta':
        lr_dict['text_encoder'] = lr * 0.01

    if optimizer == 'Adam':
        optimizers = {key: Adam(model.parameters(), lr=lr_dict[key], weight_decay=weight_decay)
                      for key, model in parameterized_models.items()}
    elif optimizer == 'SGD':
        optimizers = {key: SGD(model.parameters(), lr=lr_dict[key], weight_decay=weight_decay, momentum=momentum, nesterov=False)
                      for key, model in parameterized_models.items()}
    elif optimizer == 'RAdam':
        optimizers = {key: RAdam(model.parameters(), lr=lr_dict[key], weight_decay=weight_decay)
                      for key, model in parameterized_models.items()}
    return optimizers


def create_lr_schedulers(optimizers, config):
    decay_step = config['decay_step']
    decay_step_second = config['decay_step_second']
    gamma = config['gamma']
    scheduler_mode = config['lr_scheduler']
    warmup_iters = config['warmup_iters']

    if scheduler_mode == 'MultiStep':
        if decay_step_second <= decay_step:
            raise ValueError("second decay step should be greater than first step")
        return {key: MultiStepLR(optimizer, milestones=[decay_step-warmup_iters, decay_step_second-warmup_iters], gamma=gamma) for key, optimizer in optimizers.items()}
    elif scheduler_mode == 'Step':
        return {key: StepLR(optimizer, step_size=decay_step, gamma=gamma) for key, optimizer in optimizers.items()}
    elif scheduler_mode == 'CAWRwithWarmup':
        return {key: CAWRwithWarmup(optimizer, T_0=1, T_mult=2, warmup_iters=warmup_iters, start_factor=0.01) 
            for key, optimizer in optimizers.items()}
    elif scheduler_mode == 'MultiStepWithWarmup':
        if decay_step_second <= decay_step:
            raise ValueError("second decay step should be greater than first step")
        scheduler_dict = {}
        for key, optimizer in optimizers.items():
            scheduler1 = LinearLR(optimizer, start_factor=gamma**2, total_iters=warmup_iters)
            scheduler2 = MultiStepLR(optimizer, milestones=[decay_step-warmup_iters, decay_step_second-warmup_iters], gamma=gamma)
            scheduler_dict[key] = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_iters])
        return scheduler_dict
    elif scheduler_mode == 'StepWithWarmup':
        for key, optimizer in optimizers.items():
            scheduler1 = LinearLR(optimizer, start_factor=gamma**2, total_iters=warmup_iters)
            scheduler2 = StepLR(optimizer, step_size=decay_step, gamma=gamma)
            scheduler_dict[key] = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_iters])
        
        return scheduler_dict
