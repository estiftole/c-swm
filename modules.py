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
    def __init__(self, node_input_dim: int, input_dim: int, hidden_dim: int, act_fn: str = 'relu'):
        super().__init__()

        self.fc1 = nn.Linear(node_input_dim, hidden_dim)
        self.act1 = utils.get_act_fn(act_fn)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.act2 = utils.get_act_fn(act_fn)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def forward(self, node_feat: torch.Tensor, edge_index: torch.Tensor|None, edge_feat: torch.Tensor|None) -> torch.Tensor:
        if edge_feat is not None and edge_index is not None:
            row, _ = edge_index
            # Aggregate messages passed to each node
            agg = edge_feat.new_zeros(node_feat.size(0), edge_feat.size(1)).index_add_(0, row, edge_feat)
        else:
            edge_dim = self.fc1.in_features - node_feat.size(1)
            agg = node_feat.new_zeros(node_feat.size(0), edge_dim)

        h = torch.cat([node_feat, agg], dim=1)
        h = self.act1(self.fc1(h))
        h = self.act2(self.ln(self.fc2(h)))
        h = self.fc3(h)

        return h

class TransitionModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int, num_slots: int, act_fn: str = 'relu') -> None:
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        node_input_dim = hidden_dim + input_dim + self.action_dim
        self._node_fn = NodeModel(node_input_dim, input_dim, hidden_dim, act_fn)
        self._edge_fn = EdgeModel(input_dim, hidden_dim, act_fn)

        self.edge_list = None
        self.batch_size = 0

    def _get_edge_list_fully_connected(self, batch_size: int, num_slots: int, device: torch.device) -> torch.Tensor:
        if self.edge_list is None or self.batch_size != batch_size:
            self.batch_size = batch_size

            nodes = torch.arange(num_slots, device=device)
            senders, receivers = torch.meshgrid(nodes, nodes, indexing="ij")
            mask = senders != receivers
            base_senders = senders[mask]
            base_receivers = receivers[mask]

            batch_offsets = torch.arange(0, batch_size * num_slots, num_slots, device=device)
            senders_batch = base_senders.unsqueeze(0) + batch_offsets.unsqueeze(1)
            receivers_batch = base_receivers.unsqueeze(0) + batch_offsets.unsqueeze(1)

            self.edge_list = torch.stack([
                senders_batch.reshape(-1),
                receivers_batch.reshape(-1)
            ], dim=0)

        return self.edge_list

    def forward(self, states, action) -> torch.Tensor:
        batch_size, num_nodes, _ = states.shape

        node_feat = states.view(-1, self.input_dim)

        edge_feat = None
        edge_index = None

        if num_nodes > 1:
            edge_index = self._get_edge_list_fully_connected(
                batch_size, num_nodes, states.device)

            row, col = edge_index
            edge_feat = self._edge_fn(node_feat[row], node_feat[col])

        action_vec = utils.to_one_hot(action, self.action_dim * num_nodes)
        action_vec = action_vec.view(-1, self.action_dim)

        # Attach action to each state
        node_feat = torch.cat([node_feat, action_vec], dim=-1)

        node_feat = self._node_fn(node_feat, edge_index, edge_feat)

        # [batch_size, num_nodes, hidden_dim]
        return node_feat.view(batch_size, num_nodes, -1)
