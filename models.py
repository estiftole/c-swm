from modules import *
import numpy as np

class ContrastiveSWM(nn.Module):
    def __init__(self, embedding_dim: int, input_dims: tuple, hidden_dim: int,
        num_slots: int, action_dim: int,
        hinge=1., sigma=0.5, global_action: bool = False
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.num_objects = num_slots
        self.hinge = hinge
        self.sigma = sigma

        self.pos_loss = 0
        self.neg_loss = 0

        num_channels = input_dims[0]
        width_height = input_dims[1:]

        self.obj_extractor = EncoderCNN(num_channels, hidden_dim // 16 , num_slots)
        self.obj_encoder = EncoderMLP(int(np.prod(width_height)), hidden_dim, embedding_dim, num_slots)
        self.transition_model = TransitionModel(embedding_dim, hidden_dim, action_dim, num_slots, global_action=global_action)

        self.width = width_height[0]
        self.height = width_height[1]

    def energy(self, state, action, next_state, no_trans=False):
        """Energy function based on normalized squared L2 norm."""

        norm = 0.5 / (self.sigma**2)

        if no_trans:
            diff = state - next_state
        else:
            pred_trans = self.transition_model(state, action)
            diff = state + pred_trans - next_state

        return norm * diff.pow(2).sum(2).mean(1)

    def transition_loss(self, state, action, next_state):
        return self.energy(state, action, next_state).mean()

    def contrastive_loss(self, obs, action, next_obs):

        objs = self.obj_extractor(obs)
        next_objs = self.obj_extractor(next_obs)

        state = self.obj_encoder(objs)
        next_state = self.obj_encoder(next_objs)

        # Sample negative state across episodes at random
        batch_size = state.size(0)
        perm = np.random.permutation(batch_size)
        neg_state = state[perm]

        self.pos_loss = self.energy(state, action, next_state)
        zeros = torch.zeros_like(self.pos_loss)

        self.pos_loss = self.pos_loss.mean()
        self.neg_loss = torch.max(
            zeros, self.hinge - self.energy(
                state, action, neg_state, no_trans=True)).mean()

        loss = self.pos_loss + self.neg_loss

        return loss

    def forward(self, obs):
        return self.obj_encoder(self.obj_extractor(obs))
