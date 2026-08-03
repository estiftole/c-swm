import torch
from torch import nn
import torch.nn.functional as F

ACT_FNS = {
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(),
    'elu': nn.ELU(),
    'sigmoid': nn.Sigmoid(),
    'softplus': nn.Softplus(),
}

def get_act_fn(act_fn):
    if act_fn in ACT_FNS:
        return ACT_FNS[act_fn]
    else:
        raise ValueError(f'Invalid argument for activation function "{act_fn}"')

def to_one_hot(indices, max_index):
    """Modern one-hot encoding using PyTorch built-ins."""
    return F.one_hot(indices, num_classes=max_index).to(torch.float32)
