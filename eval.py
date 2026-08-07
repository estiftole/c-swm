import argparse
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data

import models
import utils

torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--save-folder', type=str, default='checkpoints', help='Path to checkpoints.')
parser.add_argument('--num-steps', type=int, default=1, help='Number of prediction steps to evaluate.')
parser.add_argument('--dataset', type=str, default='data/shapes_eval.h5', help='Dataset string.')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable CUDA training.')

args_eval = parser.parse_args()

meta_file = os.path.join(args_eval.save_folder, 'metadata.pkl')
model_file = os.path.join(args_eval.save_folder, 'model.pt')
decoder_file = os.path.join(args_eval.save_folder, 'decoder.pt')

args = pickle.load(open(meta_file, 'rb'))['args']

args.cuda = not args_eval.no_cuda and torch.cuda.is_available()
args.batch_size = 100
args.dataset = args_eval.dataset
args.seed = 0

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

device = torch.device('cuda' if args.cuda else 'cpu')

dataset = utils.PathDataset(hdf5_file=args.dataset, path_length=args_eval.num_steps)
eval_loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

obs_batch, action_batch = next(iter(eval_loader))
input_shape = obs_batch[0][0].size()
max_action_in_batch = int(action_batch[0].max().item())

if max_action_in_batch >= args.action_dim:
    raise ValueError(
        f"CRITICAL: Eval dataset contains action index {max_action_in_batch}, "
        f"but loaded model --action-dim is only {args.action_dim}."
    )

model = models.ContrastiveSWM(
    embedding_dim=args.embedding_dim,
    hidden_dim=args.hidden_dim,
    action_dim=args.action_dim,
    input_dims=input_shape,
    num_slots=args.num_objects,
    sigma=args.sigma,
    hinge=args.hinge,
    global_action=args.global_action,
).to(device)

model.load_state_dict(torch.load(model_file))
model.eval()

decoder = None
if getattr(args, 'decoder', False):
    encoder_type = getattr(args, 'encoder', 'medium')
    if encoder_type == 'large':
        decoder = models.DecoderCNNLarge(input_dim=args.embedding_dim, num_objects=args.num_objects, hidden_dim=args.hidden_dim // 16, output_size=input_shape).to(device)
    elif encoder_type == 'medium':
        decoder = models.DecoderCNNMedium(input_dim=args.embedding_dim, num_objects=args.num_objects, hidden_dim=args.hidden_dim // 16, output_size=input_shape).to(device)
    elif encoder_type == 'small':
        decoder = models.DecoderCNNSmall(input_dim=args.embedding_dim, num_objects=args.num_objects, hidden_dim=args.hidden_dim // 16, output_size=input_shape).to(device)

    if os.path.exists(decoder_file):
        decoder.load_state_dict(torch.load(decoder_file))
        decoder.eval()
    else:
        print(f"Warning: {decoder_file} not found. Reconstruction loss will be skipped.")
        decoder = None

topk = [1]
hits_at = defaultdict(int)
num_samples = 0
rr_sum = 0
bce_loss_sum = 0.0

print(f'Running eval on {args_eval.num_steps} steps...')
with torch.no_grad():
    for batch_idx, data_batch in enumerate(eval_loader):
        data_batch = [[t.to(device) for t in tensor] for tensor in data_batch]
        observations, actions = data_batch

        if observations[0].size(0) != args.batch_size:
            continue

        obs = observations[0]
        next_obs = observations[-1]

        state = model.obj_encoder(model.obj_extractor(obs))
        next_state = model.obj_encoder(model.obj_extractor(next_obs))

        pred_state = state
        for i in range(args_eval.num_steps):
            pred_trans = model.transition_model(pred_state, actions[i])
            pred_state = pred_state + pred_trans

        if decoder is not None:
            pred_rec = torch.sigmoid(decoder(pred_state))
            bce_loss_sum += F.binary_cross_entropy(pred_rec, next_obs, reduction='sum').item()

            if batch_idx == 0:
                rollout_data = {
                    'initial_obs': obs.cpu(),
                    'true_final_obs': next_obs.cpu(),
                    'pred_final_obs': pred_rec.cpu(),
                    'num_steps': args_eval.num_steps
                }
                torch.save(rollout_data, os.path.join(args_eval.save_folder, f'rollout_{args_eval.num_steps}steps.pt'))

        # --- PER-BATCH METRICS COMPUTATION (100 candidates per batch) ---
        b_size = pred_state.size(0)
        pred_flat = pred_state.view(b_size, -1)
        next_flat = next_state.view(b_size, -1)

        dist_matrix = utils.pairwise_distance_matrix(pred_flat.cpu(), next_flat.cpu())
        dist_np = dist_matrix.numpy()

        indices = np.stack([np.lexsort((np.arange(len(row)), row)) for row in dist_np], axis=0)
        indices = torch.from_numpy(indices).long()

        labels = torch.arange(b_size).unsqueeze(-1)

        num_samples += b_size

        for k in topk:
            hits_at[k] += (indices[:, :k] == labels).sum().item()

        _, ranks = (indices == labels).max(1)
        rr_sum += torch.reciprocal(ranks.double() + 1).sum().item()

print(f'Processed {num_samples} evaluation samples across batches.')
for k in topk:
    print(f'Hits @ {k}: {hits_at[k] / float(num_samples):.4f}')

print(f'MRR: {rr_sum / float(num_samples):.4f}')

if decoder is not None and num_samples > 0:
    avg_pixel_bce = bce_loss_sum / (num_samples * input_shape.numel())
    print(f'Pixel BCE Loss: {avg_pixel_bce:.6f}')
