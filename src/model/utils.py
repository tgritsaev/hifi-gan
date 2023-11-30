from torch import nn
from torch.nn.utils import weight_norm, spectral_norm


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def SNConv1d(*args, **kwargs):
    return spectral_norm(nn.Conv1d(*args, **kwargs))


def WNConv2d(*args, **kwargs):
    return weight_norm(nn.Conv2d(*args, **kwargs))
