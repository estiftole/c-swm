from modules import *

class ContrastiveSWM(nn.Module):
    def __init__(self, embedding_dim: int, input_dim: int, hidden_dim: int, num_slots: int, action_dim: int, act_fn: str = 'relu') -> None:
        super().__init__()
        self.obj_extractor = EncoderCNN(input_dim, hidden_dim, num_slots, act_fn)
        self.obj_encoder = EncoderMLP(input_dim, hidden_dim, embedding_dim, num_slots, act_fn)
        self.transition_model = TransitionModel(input_dim, hidden_dim, out_dim, action_dim, act_fn)
