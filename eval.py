import argparse
import torch
import utils
import os
import pickle


from torch.utils import data
import numpy as np
from collections import defaultdict

import models

torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--save-folder', type=str,
                    default='checkpoints',
                    help='Path to checkpoints.')
parser.add_argument('--num-steps', type=int, default=1,
                    help='Number of prediction steps to evaluate.')
parser.add_argument('--dataset', type=str,
                    default='data/shapes_eval.h5',
                    help='Dataset string.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')

args_eval = parser.parse_args()


meta_file = os.path.join(args_eval.save_folder, 'metadata.pkl')
model_file = os.path.join(args_eval.save_folder, 'model.pt')

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

dataset = utils.PathDataset(
    hdf5_file=args.dataset, path_length=args_eval.num_steps)
eval_loader = data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

# Get data sample
obs = next(iter(eval_loader))[0]
input_shape = obs[0][0].size()

model = models.ContrastiveSWM(
    embedding_dim=args.embedding_dim,
    hidden_dim=args.hidden_dim,
    action_dim=args.action_dim,
    input_dims=input_shape,
    num_slots=args.num_objects,
    sigma=args.sigma,
    hinge=args.hinge,
    # ignore_action=args.ignore_action,
    # copy_action=args.copy_action,
    # encoder=args.encoder
).to(device)

model.load_state_dict(torch.load(model_file))
model.eval()

# topk = [1, 5, 10]
topk = [1]
hits_at = defaultdict(int)
num_samples = 0
rr_sum = 0

pred_states = []
next_states = []

with torch.no_grad():

    for batch_idx, data_batch in enumerate(eval_loader):
        data_batch = [[t.to(
            device) for t in tensor] for tensor in data_batch]
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

        pred_states.append(pred_state.cpu())
        next_states.append(next_state.cpu())

        print("Action shape:", actions[0].shape, "Type:", actions[0].dtype)
    pred_state_cat = torch.cat(pred_states, dim=0)
    next_state_cat = torch.cat(next_states, dim=0)

    full_size = pred_state_cat.size(0)

    # Flatten object/feature dimensions -> [B, N * D]
    next_state_flat = next_state_cat.view(full_size, -1)
    pred_state_flat = pred_state_cat.view(full_size, -1)

    # dist_matrix[i, j] = distance from predicted state i to true next_state j
    dist_matrix = utils.pairwise_distance_matrix(pred_state_flat, next_state_flat)

    # Diagnostic for distance matrix and duplicates
    true_dist_0 = dist_matrix[0, 0].item()
    min_dist_0, min_idx_0 = dist_matrix[0].min(dim=0)

    print(f"Sample 0 True Target Distance:     {true_dist_0:.4f}")
    print(f"Sample 0 Closest Target Distance:  {min_dist_0.item():.4f} (at Index {min_idx_0.item()})")
    print(f"Number of targets closer than true target: {(dist_matrix[0] < dist_matrix[0, 0]).sum().item()}")

    # Check ground-truth distance between Target 0 and Target 36
    target_0_vs_36_dist = torch.norm(next_state_flat[0] - next_state_flat[36]).item()
    print(f"Ground Truth Distance between Target 0 & Target 36: {target_0_vs_36_dist:.4f}")

    # Sort distances in ascending order (smallest distance first)
    dist_np = dist_matrix.numpy()
    indices = np.stack([np.lexsort((np.arange(len(row)), row)) for row in dist_np], axis=0)
    indices = torch.from_numpy(indices).long()

    # True target for sample i is index i
    labels = torch.arange(full_size).unsqueeze(-1)

    print('Processed {} batches of size {}'.format(batch_idx + 1, args.batch_size))
    print('Size of current topk evaluation batch: {}'.format(full_size))

    num_samples += full_size

    for k in topk:
        hits_at[k] += (indices[:, :k] == labels).sum().item()

    _, ranks = (indices == labels).max(1)
    rr_sum += torch.reciprocal(ranks.double() + 1).sum().item()

    pred_states = []
    next_states = []

    # Quick C-SWM Health Diagnostic
    pos_energy = (pred_state_flat - next_state_flat).pow(2).sum(dim=-1).mean().item()
    latent_std = pred_state_flat.std(dim=0).mean().item()

    print(f"--> Pos Energy (should be near 0): {pos_energy:.4f}")
    print(f"--> Latent Std (should be > 0.1):   {latent_std:.4f}")

    neg_energy = (dist_matrix.sum() - torch.diag(dist_matrix).sum()).item() / (full_size * (full_size - 1))

    print(f"--> Pos Energy: {pos_energy:.4f}")
    print(f"--> Neg Energy: {neg_energy:.4f}")
    print(f"--> Separation Ratio (Neg / Pos): {neg_energy / pos_energy:.2f}x")

for k in topk:
    print('Hits @ {}: {}'.format(k, hits_at[k] / float(num_samples)))

print('MRR: {}'.format(rr_sum / float(num_samples)))

# uv run eval.py --dataset data/pong_eval.h5 --save-folder checkpoints/pong --num-steps 1
