import torch


def is_almost_equal(tens_a, tens_b, delta=1e-5):
    return torch.all(torch.lt(torch.abs(torch.add(tens_a, -tens_b)), delta))
