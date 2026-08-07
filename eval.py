import argparse
import torch
import torch.nn.functional as F
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

dataset = utils.PathDataset(
    hdf5_file=args.dataset, path_length=args_eval.num_steps)
eval_loader = data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

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
    global_action=args.global_action,
).to(device)

model.load_state_dict(torch.load(model_file))
model.eval()

decoder = None
if getattr(args, 'decoder', False):
    if args.encoder == 'large':
        decoder = models.DecoderCNNLarge(input_dim=args.embedding_dim, num_objects=args.num_objects, hidden_dim=args.hidden_dim // 16, output_size=input_shape).to(device)
    elif args.encoder == 'medium':
        decoder = models.DecoderCNNMedium(input_dim=args.embedding_dim, num_objects=args.num_objects, hidden_dim=args.hidden_dim // 16, output_size=input_shape).to(device)
    elif args.encoder == 'small':
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

pred_states = []
next_states = []

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

        pred_states.append(pred_state.cpu())
        next_states.append(next_state.cpu())

    pred_state_cat = torch.cat(pred_states, dim=0)
    next_state_cat = torch.cat(next_states, dim=0)
    full_size = pred_state_cat.size(0)

    next_state_flat = next_state_cat.view(full_size, -1)
    pred_state_flat = pred_state_cat.view(full_size, -1)

    dist_matrix = utils.pairwise_distance_matrix(pred_state_flat, next_state_flat)
    dist_np = dist_matrix.numpy()
    indices = np.stack([np.lexsort((np.arange(len(row)), row)) for row in dist_np], axis=0)
    indices = torch.from_numpy(indices).long()

    labels = torch.arange(full_size).unsqueeze(-1)

    print('Processed {} batches of size {}'.format(batch_idx + 1, args.batch_size))
    num_samples += full_size

    for k in topk:
        hits_at[k] += (indices[:, :k] == labels).sum().item()

    _, ranks = (indices == labels).max(1)
    rr_sum += torch.reciprocal(ranks.double() + 1).sum().item()

for k in topk:
    print('Hits @ {}: {}'.format(k, hits_at[k] / float(num_samples)))

print('MRR: {}'.format(rr_sum / float(num_samples)))

if decoder is not None:
    avg_pixel_bce = bce_loss_sum / (num_samples * input_shape.numel())
    print('Pixel BCE Loss: {:.6f}'.format(avg_pixel_bce))
