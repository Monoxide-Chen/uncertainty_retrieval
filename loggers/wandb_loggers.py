import wandb

from trainers.abc import AbstractBaseLogger


class WandbSimplePrinter(AbstractBaseLogger):
    def __init__(self, prefix):
        self.prefix = prefix

    def log(self, log_data, step, commit=False):
        log_metrics = {self.prefix + k: v for k, v in log_data.items() if not isinstance(v, dict)}
        wandb.log(log_metrics, step=step, commit=commit)

    def complete(self, log_data, step):
        self.log(log_data, step)


class WandbSummaryPrinter(AbstractBaseLogger):
    def __init__(self, prefix, summary_keys: list):
        self.prefix = prefix
        self.summary_keys = summary_keys
        self.previous_best_vals = {key: 0 for key in self.summary_keys}

    def log(self, log_data, step, commit=False):
        for key in self.summary_keys:
            if key in log_data:
                log_value = log_data[key]
                if log_value > self.previous_best_vals[key]:
                    wandb.run.summary[self.prefix+key] = log_value
                    self.previous_best_vals[key] = log_value

    def complete(self, log_data, step):
        self.log(log_data, step)
