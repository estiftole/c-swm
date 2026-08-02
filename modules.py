from typing import Optional, Tuple
import utils
import numpy as np
import torch
from torch import nn

class EncoderCNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_slots: int, act_fn: str ="sigmoid", act_fn_hid: str ="relu") -> None:
        super().__init__()

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
        super().__init__()
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

        h = self.act1(self.fc1(h_flat))
        h = self.act2(self.ln(self.fc2(h)))
        h = self.fc3(h)

        return h

class EdgeModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, act_fn: str = 'relu') -> None:
        super().__init__()

        self.fc1 = nn.Linear(input_dim*2, hidden_dim)
        self.act1 = utils.get_act_fn(act_fn)
        self.fc2 =  nn.Linear(hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.act2 = utils.get_act_fn(act_fn)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        h = torch.cat([source, target], dim=1)

        h = self.act1(self.fc1(h))
        h = self.act2(self.ln(self.fc2(h)))
        h = self.fc3(h)

        return h

class NodeModel(nn.Module):
    def __init__(self, node_input_dim: int, hidden_dim: int, out_dim: int, act_fn: str = 'relu'):
        super().__init__()

        self.fc1 = nn.Linear(node_input_dim, hidden_dim)
        self.act1 = utils.get_act_fn(act_fn)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.act2 = utils.get_act_fn(act_fn)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, node_feat: torch.Tensor, edge_index: torch.Tensor|None, edge_feat: torch.Tensor|None) -> torch.Tensor:
        if edge_feat is not None and edge_index is not None:
            row, _ = edge_index
            # Aggregate messages passed to each node
            agg = edge_feat.new_zeros(node_feat.size(0), edge_feat.size(1)).index_add_(0, row, edge_feat)
            h = torch.cat([node_feat, agg], dim=1)
        else:
            h = node_feat

        h = self.act1(self.fc1(h))
        h = self.act2(self.ln(self.fc2(h)))
        h = self.fc3(h)

        return h
