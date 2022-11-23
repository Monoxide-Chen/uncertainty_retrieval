import os

import torch

from trainers.abc import AbstractBaseLogger


def _checkpoint_file_path(export_path, filename):
    return os.path.join(export_path, filename)


def _set_up_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def _save_state_dict_with_step(log_data, step, path, filename):
    log_data = {k: v for k, v in log_data.items() if isinstance(v, dict)}
    log_data['step'] = step
    torch.save(log_data, _checkpoint_file_path(path, filename))


class RecentModelTracker(AbstractBaseLogger):
    def __init__(self, export_path, ckpt_filename='recent.pth'):
        self.export_path = export_path
        _set_up_path(self.export_path)
        self.ckpt_filename = ckpt_filename

    def log(self, log_data, step, commit=False):
        _save_state_dict_with_step(log_data, step, self.export_path, self.ckpt_filename)

    def complete(self, log_data, step):
        pass


class BestModelTracker(AbstractBaseLogger):
    def __init__(self, export_path, ckpt_filename='best.pth', metric_key='recall_@10'):
        self.export_path = export_path
        _set_up_path(self.export_path)

        self.metric_key = metric_key
        self.ckpt_filename = ckpt_filename

        self.best_value = -9e9

    def log(self, log_data, step, commit=False):
        recent_values = 0
        num_values = 0
        for i in log_data:
            if self.metric_key not in i:
                continue
            recent_values += log_data[i]
            num_values += 1

        if num_values < 1:
            print("WARNING: The key: {} is not in logged data. Not saving best model".format(self.metric_key))
            return
        recent_value = recent_values / num_values
        if self.best_value < recent_value:
            self.best_value = recent_value
            _save_state_dict_with_step(log_data, step, self.export_path, self.ckpt_filename)
            print("Update Best {} Model at Step {} with value {}".format(self.metric_key, step, self.best_value))

    def complete(self, *args, **kwargs):
        pass
