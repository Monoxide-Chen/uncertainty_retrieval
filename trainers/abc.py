import abc
from abc import ABC
from typing import Sequence, Tuple, Any

import torch
from torch import nn as nn


class AbstractBaseLogger(ABC):
    @abc.abstractmethod
    def log(self, log_data: dict, step: int, commit: bool) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def complete(self, log_data: dict, step: int) -> None:
        raise NotImplementedError


class LoggingService(object):
    def __init__(self, loggers: Sequence[AbstractBaseLogger]):
        self.loggers = loggers

    def log(self, log_data: dict, step: int, commit=False):
        for logger in self.loggers:
            logger.log(log_data, step, commit=commit)

    def complete(self, log_data: dict, step: int):
        for logger in self.loggers:
            logger.complete(log_data, step)


class AbstractBaseMetricLoss(nn.Module, ABC):
    @abc.abstractmethod
    def forward(self, ref_features: torch.Tensor, tar_features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def code(cls) -> str:
        raise NotImplementedError


class AbstractBaseImageLowerEncoder(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def layer_shapes(self):
        raise NotImplementedError


class AbstractBaseImageUpperEncoder(nn.Module, abc.ABC):
    def __init__(self, lower_feature_shape, feature_size, pretrained=True, *args, **kwargs):
        super().__init__()
        self.lower_feature_shape = lower_feature_shape
        self.feature_size = feature_size

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class AbstractBaseTextEncoder(nn.Module, abc.ABC):
    @abc.abstractmethod
    def __init__(self, vocabulary_len, padding_idx, feature_size, *args, **kwargs):
        super().__init__()
        self.feature_size = feature_size

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, lengths: torch.LongTensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def code(cls) -> str:
        raise NotImplementedError


class AbstractBaseCompositor(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, mid_image_features, text_features, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def code(cls) -> str:
        raise NotImplementedError


class AbstractGlobalStyleTransformer(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, normed_x, t, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def code(cls) -> str:
        raise NotImplementedError


class AbstractBaseTrainer(ABC):
    def __init__(self, models, train_dataloader, criterions, optimizers, lr_schedulers, num_epochs,
                 train_loggers, val_loggers, evaluators, train_evaluator, *args, **kwargs):
        self.models = models
        self.train_dataloader = train_dataloader
        self.criterions = criterions
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.num_epochs = num_epochs
        self.train_logging_service = LoggingService(train_loggers)
        self.val_logging_service = LoggingService(val_loggers)
        self.evaluators = evaluators
        self.train_evaluator = train_evaluator
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.start_epoch = kwargs['start_epoch'] if 'start_epoch' in kwargs else 0

    def train_one_epoch(self, epoch) -> dict:
        raise NotImplementedError

    def run(self) -> dict:
        self._load_models_to_device()
        for epoch in range(self.start_epoch, self.num_epochs):
            for phase in ['train', 'val']:
                if phase == 'train':
                    self._to_train_mode()
                    train_results = self.train_one_epoch(epoch)
                    self.train_logging_service.log(train_results, step=epoch)
                    print(train_results)
                else:
                    self._to_eval_mode()
                    all_val_results = {}
                    for key, evaluator in self.evaluators.items():
                        print('results on ' + key)
                        val_results, _ = evaluator.evaluate(epoch)
                        for i in val_results.items():
                            all_val_results[key + '_' + i[0]] = i[1]                       

                    # train_val_results = self.train_evaluator.evaluate(epoch)
                    # self.train_logging_service.log(train_val_results, step=epoch)
                    model_state_dicts = self._get_state_dicts(self.models)
                    optimizer_state_dicts = self._get_state_dicts(self.optimizers)
                    all_val_results['model_state_dict'] = model_state_dicts
                    all_val_results['optimizer_state_dict'] = optimizer_state_dicts
                    self.val_logging_service.log(all_val_results, step=epoch, commit=True)

        return self.models

    def _load_models_to_device(self):
        for model in self.models.values():
            model.to(self.device)

    def _to_train_mode(self, keys=None):
        keys = keys if keys else self.models.keys()
        for key in keys:
            self.models[key].train()

    def _to_eval_mode(self, keys=None):
        keys = keys if keys else self.models.keys()
        for key in keys:
            self.models[key].eval()

    def _reset_grad(self, keys=None):
        keys = keys if keys else self.optimizers.keys()
        for key in keys:
            self.optimizers[key].zero_grad()

    def _update_grad(self, keys=None, exclude_keys=None):
        keys = keys if keys else list(self.optimizers.keys())
        if exclude_keys:
            keys = [key for key in keys if key not in exclude_keys]
        for key in keys:
            self.optimizers[key].step()

    def _step_schedulers(self):
        for scheduler in self.lr_schedulers.values():
            scheduler.step()

    @staticmethod
    def _get_state_dicts(dict_of_models):
        state_dicts = {}
        for model_name, model in dict_of_models.items():
            if isinstance(model, nn.DataParallel):
                state_dicts[model_name] = model.module.state_dict()
            else:
                state_dicts[model_name] = model.state_dict()
        return state_dicts

    @classmethod
    def code(cls) -> str:
        raise NotImplementedError


METRIC_LOGGING_KEYS = {
    'train_loss': 'train/loss',
    'val_loss': 'val/loss',
    'val_correct': 'val/correct'
}
STATE_DICT_KEY = 'state_dict'
