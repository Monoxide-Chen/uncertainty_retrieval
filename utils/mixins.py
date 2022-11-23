import abc

import torch
import torch.nn as nn


class AbstractGradientControl(abc.ABC):
    @abc.abstractmethod
    def stash_grad(self, grad_dict):
        raise NotImplementedError

    @abc.abstractmethod
    def restore_grad(self, grad_dict):
        raise NotImplementedError


class GradientControlMixin(AbstractGradientControl):
    def stash_grad(self, grad_dict):
        for k, v in self.named_parameters():
            if k in grad_dict:
                grad_dict[k] += v.grad.clone()
            else:
                grad_dict[k] = v.grad.clone()
        self.zero_grad()
        return grad_dict

    def restore_grad(self, grad_dict):
        for k, v in self.named_parameters():
            grad = grad_dict[k] if k in grad_dict else torch.zeros_like(v.grad)

            if v.grad is None:
                v.grad = grad
            else:
                v.grad += grad


class GradientControlDataParallel(nn.DataParallel, AbstractGradientControl):
    def stash_grad(self, grad_dict):
        if isinstance(self.module, GradientControlMixin):
            return self.module.stash_grad(grad_dict)
        else:
            raise RuntimeError("A module should be an instance of GradientControlMixin")

    def restore_grad(self, grad_dict):
        if isinstance(self.module, GradientControlMixin):
            self.module.restore_grad(grad_dict)
        else:
            raise RuntimeError("A module should be an instance of GradientControlMixin")
