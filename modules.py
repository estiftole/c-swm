import numpy as np
import torch
from torch import nn
import utils

class EncoderCNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_slots: int, act_fn: str ="sigmoid", act_fn_hid: str ="relu") -> None:
        super(Encoder, self).__init__()

        self.cnn1 = nn.Conv2d(input_dim, hidden_dim, (3,3), padding=1)
        self.act1 = utils.get_act_fn(act_fn_hid)
        self.ln1 = nn.BatchNorm2d(hidden_dim)

        self.cnn2 = nn.Conv2d(hidden_dim, hidden_dim, (3,3), padding=1)
        self.act2 = utils.get_act_fn(act_fn_hid)
        self.ln2 = nn.BatchNorm2d(hidden_dim)

        self.cnn3 = nn.Conv2d(hidden_dim, hidden_dim, (3,3), padding=1)
        self.act3 = utils.get_act_fn(act_fn_hid)
        self.ln3 = nn.BatchNorm2d(hidden_dim)

        self.cnn4 = nn.Conv2d(hidden_dim, num_slots, (3, 3), padding=1)
        self.act4 = utils.get_act_fn(act_fn)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.ln1(self.cnn1(obs)))
        h = self.act2(self.ln2(self.cnn2(h)))
        h = self.act3(self.ln3(self.cnn3(h)))
        h = self.act4(self.cnn4(h))

        return h

class EncoderMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_slots: int, act_fn: str = 'relu', act_fn_hid: str = 'relu') -> None:
        super(EncoderMLP, self).__init__()
        self.num_slots = num_slots
        self.input_dim = input_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = utils.get_act_fn(act_fn_hid)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = utils.get_act_fn(act_fn_hid)
        self.ln = nn.LayerNorm(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, slot: torch.Tensor) -> torch.Tensor:
        h_flat = slot.view(-1, self.num_slots, self.input_dim)
        h = self.fc1(h_flat)
        h = self.act1(h)

        h = self.fc2(h)
        h = self.ln(h)
        h = self.act2(h)

        h = self.fc3(h)

        return h
